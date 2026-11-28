"""Bound audit entrypoint around unmodified, pinned MS-SWIFT SFT.

The hook observes the real collated training batch before SWIFT mutates it.
It never changes input IDs, masks, labels, losses, gradients, or sampler order.
No Tensor/Torch imports occur during local planning or unit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def token_hash(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("ascii")).hexdigest()


def describe_batch(input_ids, labels, attention_mask=None):
    """Pure-Python reference: excludes the first causal label and masked padding."""
    if len(input_ids) != len(labels) or not input_ids:
        raise ValueError("audit requires a nonempty batch with aligned labels")
    masks = attention_mask if attention_mask is not None else [[1] * len(row) for row in input_ids]
    if len(masks) != len(input_ids):
        raise ValueError("audit mask batch mismatch")
    active_rows, target_rows, lengths = [], [], []
    for ids, targets, mask in zip(input_ids, labels, masks):
        if len(ids) != len(targets) or len(mask) != len(ids):
            raise ValueError("audit sequence dimensions disagree")
        if any(x not in (0, 1) for x in mask):
            raise ValueError("audit expects a 2D binary attention mask")
        if any(target != -100 and not visible for target, visible in zip(targets, mask)):
            raise ValueError("padding unexpectedly contributes supervised labels")
        active_rows.append([x for x, visible in zip(ids, mask) if visible])
        target_rows.append([x for i, (x, visible) in enumerate(zip(targets, mask)) if i > 0 and visible and x != -100])
        lengths.append(sum(mask))
    return {"sequences": len(input_ids), "input_tokens": sum(lengths),
            "supervised_tokens": sum(map(len, target_rows)), "input_lengths": lengths,
            "input_token_ids_sha256": token_hash(active_rows),
            "causal_supervised_token_ids_sha256": token_hash(target_rows)}


def install_audit(directory, *, plan_path, drive_run):
    import torch
    from transformers import TrainerCallback
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    from cpt.swift_verify import _runtime_provenance

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # Check installed wheel/VCS metadata before applying our in-memory hook.
    provenance = _runtime_provenance()
    original = Seq2SeqTrainer.training_step
    original_file = Path(sys.modules[Seq2SeqTrainer.__module__].__file__)
    save(directory / "hook_provenance.json", {
        "schema_version": "paired-training-audit-v1", "runtime": provenance,
        "audit_driver_sha256": file_hash(__file__), "upstream_file": str(original_file),
        "upstream_file_sha256": file_hash(original_file),
        "hook": "swift.trainers.seq2seq_trainer.Seq2SeqTrainer.training_step",
        "changes_training_tensors": False,
        "timing_note": "CUDA synchronizations and metadata copies add measurement overhead; both arms use the same hook."})
    state = {"microsteps": 0, "optimizer_steps": 0, "pending": [], "started": None}
    from best_checkpoint import callback_for
    best_callback = callback_for(plan_path, directory, drive_run, TrainerCallback)

    class AuditCallback(TrainerCallback):
        def on_step_end(self, args, trainer_state, control, **kwargs):
            torch.cuda.synchronize()
            rows = state["pending"]
            if not rows:
                raise ValueError("optimizer step completed without observed microbatches")
            item = {"optimizer_step": trainer_state.global_step, "microsteps": len(rows),
                    "sequences": sum(row["sequences"] for row in rows),
                    "input_tokens": sum(row["input_tokens"] for row in rows),
                    "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
                    "ordered_microbatch_label_hashes": [row["causal_supervised_token_ids_sha256"] for row in rows],
                    "wall_seconds": time.monotonic() - state["started"],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "optimizer_step_skipped": bool(getattr(state["trainer"].accelerator, "optimizer_step_was_skipped", False))}
            append(directory / "optimizer_steps.jsonl", item)
            state["optimizer_steps"] += 1
            state["pending"] = []
            state["started"] = None
            return control

    def training_step(trainer, model, inputs, *args, **kwargs):
        if "trainer" not in state:
            state["trainer"] = trainer
            trainer.add_callback(AuditCallback())
            trainer.add_callback(best_callback)
        if state["started"] is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            state["started"] = time.monotonic()
        if "input_ids" not in inputs or "labels" not in inputs:
            raise ValueError("audited pilot requires explicit input_ids and labels")
        to_list = lambda value: value.detach().cpu().tolist()
        item = describe_batch(to_list(inputs["input_ids"]), to_list(inputs["labels"]),
                              to_list(inputs["attention_mask"]) if "attention_mask" in inputs else None)
        item.update(microstep=state["microsteps"] + 1, optimizer_step_before=trainer.state.global_step,
                    status="started")
        append(directory / "microbatch_events.jsonl", item)
        began = time.monotonic()
        try:
            loss = original(trainer, model, inputs, *args, **kwargs)
            torch.cuda.synchronize()
            item.update(status="backward_completed", wall_seconds=time.monotonic() - began,
                        returned_loss=float(loss.detach().float().cpu().item()))
        except BaseException as exc:
            item.update(status="failed", error_type=type(exc).__name__)
            append(directory / "microbatch_events.jsonl", item)
            raise
        state["microsteps"] += 1
        state["pending"].append(item)
        append(directory / "microbatch_events.jsonl", item)
        return loss

    Seq2SeqTrainer.training_step = training_step
    return state


def summarize(directory, expected_steps=32, expected_sequences=256):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "optimizer_steps.jsonl").read_text().splitlines() if line.strip()]
    if [r["optimizer_step"] for r in rows] != list(range(1, expected_steps + 1)):
        raise ValueError("observed optimizer steps differ from the declared complete pilot")
    sequences = sum(r["sequences"] for r in rows)
    if sequences != expected_sequences or any(r["optimizer_step_skipped"] for r in rows):
        raise ValueError("pilot consumed an unexpected number of sequences or skipped an optimizer update")
    result = {"schema_version": "paired-training-exposure-v1", "optimizer_steps": len(rows),
              "consumed_sequences": sequences, "input_tokens": sum(r["input_tokens"] for r in rows),
              "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
              "ordered_consumed_label_hashes_sha256": token_hash([h for row in rows for h in row["ordered_microbatch_label_hashes"]]),
              "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
              "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
              "observed_optimizer_wall_seconds": sum(r["wall_seconds"] for r in rows),
              "optimizer_steps_sha256": file_hash(directory / "optimizer_steps.jsonl"),
              "quality_improvement_measured": False}
    save(directory / "summary.json", result)
    return result


def child(plan_path):
    from cpt.launch import launch
    result = launch(plan_path, execute=False)
    if not result["training_ready"]:
        raise ValueError("unapproved data cannot enter the audited trainer")
    plan = json.loads(Path(plan_path).read_text())
    if plan["recipe"]["method"] != "sft_lora" or plan.get("init_adapter"):
        raise ValueError("paired arms require independent fresh Base plus LoRA")
    drive_run = os.environ.get("ARIADNE_DRIVE_RUN")
    if not drive_run or not Path(drive_run).is_dir():
        raise ValueError("paired SFT requires the owned mounted Drive run directory")
    directory = Path(plan["output_dir"]) / "training_audit"
    observed = install_audit(directory, plan_path=plan_path, drive_run=drive_run)
    sys.argv = ["swift.cli.sft", *result["argv"][3:]]
    runpy.run_module("swift.cli.sft", run_name="__main__")
    train_rows = sum(1 for line in Path(plan["datasets"]["train"]).read_text().splitlines() if line.strip())
    result = summarize(directory, plan["recipe"]["parameters"]["max_steps"], train_rows * 2)
    from best_checkpoint import selected_exposure, inspect_checkpoint
    best = json.loads((directory / "best.json").read_text())
    trainer = observed["trainer"]
    if (trainer.state.global_step != result["optimizer_steps"]
            or trainer.state.best_global_step != best["step"]
            or trainer.state.best_metric != best["eval_loss"]):
        raise ValueError("final native training state differs from published best")
    checkpoint = Path(plan["checkpoint_dir"]) / f"checkpoint-{best['step']}"
    if (inspect_checkpoint(checkpoint, best["step"], best["eval_loss"]) != best["files"]
            or len(list(Path(plan["checkpoint_dir"]).glob("checkpoint-*"))) != 1):
        raise ValueError("native best-only retention failed")
    rows = [json.loads(line) for line in (directory / "optimizer_steps.jsonl").read_text().splitlines()]
    result.update(selected_exposure(rows, best["step"]), best_eval_loss=best["eval_loss"])
    save(directory / "summary.json", result)
    from shutil import copyfile
    target = Path(drive_run) / "progress" / Path(plan_path).parent.parent.name
    copyfile(directory / "summary.json", target / "summary.json")
    return result


def launch_audited(plan_path):
    from cpt.launch import launch, _environment
    from cpt.swift_verify import verify_template
    path = Path(plan_path).resolve(strict=True)
    preflight = launch(path, execute=False)  # reconstructs and verifies all bound artifacts
    if not preflight["training_ready"]:
        raise ValueError("training denied: " + ", ".join(preflight["blockers"]))
    plan = json.loads(path.read_text())
    out = Path(plan["output_dir"])
    if path.parent != out or (out / "launch_receipt.json").exists() or Path(plan["checkpoint_dir"]).exists():
        raise ValueError("training must use its fresh bound plan directory")
    parameters = plan["recipe"]["parameters"]
    train_rows = sum(1 for line in Path(plan["datasets"]["train"]).read_text().splitlines() if line.strip())
    expected_steps = ((train_rows + 7) // 8) * 2
    if (not 1 <= train_rows <= 128 or parameters.get("max_steps") != expected_steps
            or parameters.get("gradient_accumulation_steps") != 8
            or parameters.get("per_device_train_batch_size") != 1
            or parameters.get("num_train_epochs") != 2):
        raise ValueError("pilot must use actual admitted rows for exactly two epochs, batch1/accum8, at most128train")
    environment = _environment()
    verification = verify_template(plan)
    if not verification.get("verified"):
        raise ValueError("real pinned SWIFT template verification failed")
    save(out / "template_verification.json", verification)
    argv = [sys.executable, str(Path(__file__).resolve()), "child", "--plan", str(path)]
    contract = {"schema_version": "paired-audited-entry-v1", "plan_sha256": file_hash(path),
                "driver_sha256": file_hash(__file__), "original_swift_argv": preflight["argv"],
                "audited_argv": argv, "instrumentation_only": False, "training_tensors_unchanged": True,
                "best_policy": "native best; finite dev loss; verify local and Drive before pruning"}
    save(out / "audit_execution_contract.json", contract)
    receipt = {"schema_version": "cpt-launch-receipt-v1", "status": "starting",
               "plan_sha256": file_hash(path), "environment": environment,
               "template_verification_sha256": file_hash(out / "template_verification.json"),
               "argv": preflight["argv"], "audited_entry_sha256": file_hash(out / "audit_execution_contract.json"),
               "training_completed": False}
    save(out / "launch_receipt.json", receipt)
    try:
        result = subprocess.run(argv, shell=False, check=False, env=os.environ.copy())
    except BaseException as exc:
        receipt.update(status="interrupted_or_failed", error_type=type(exc).__name__)
        save(out / "launch_receipt.json", receipt)
        raise
    receipt.update(status="process_exited", returncode=result.returncode,
                   note="Checkpoint reload and held-out comparison remain separate checks.")
    save(out / "launch_receipt.json", receipt)
    if result.returncode:
        raise RuntimeError(f"audited SWIFT process exited {result.returncode}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", choices=["launch", "child", "summarize"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "launch" and not args.execute:
        parser.error("real training requires --execute")
    if args.action == "summarize":
        plan = json.loads(args.plan.read_text())
        train_rows = sum(1 for line in Path(plan["datasets"]["train"]).read_text().splitlines() if line.strip())
        result = summarize(Path(plan["output_dir"]) / "training_audit",
                           plan["recipe"]["parameters"]["max_steps"], train_rows * 2)
    else:
        result = (launch_audited if args.action == "launch" else child)(args.plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
