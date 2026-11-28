"""Phased, bounded Colab A100 engineering probes and paired SFT pilot.

This driver never allocates or stops an account runtime. The operator owns that
lifecycle and must download artifacts.zip before stopping its session.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile

import remote_job as legacy
from training_audit import append, file_hash, save


def run_root(run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id):
        raise ValueError("invalid run ID")
    root = Path("/content") / ("ariadne_paired_" + run_id)
    if root.is_symlink():
        raise ValueError("run root cannot be a symlink")
    return root


def verify_bundle(root):
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for name, spec in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.is_symlink() or file_hash(path) != spec["sha256"]:
            raise ValueError(f"transport input changed: {name}")
    return manifest


def _board_sample(path):
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        append(path, {"monotonic": time.monotonic(), "returncode": result.returncode,
                      "board_sample": result.stdout.strip()})
    except (OSError, subprocess.TimeoutExpired) as exc:
        append(path, {"monotonic": time.monotonic(), "error_type": type(exc).__name__})


def terminate_group(proc):
    """leader 退出不代表同组后代退出；TERM 宽限后仍对自有组补 KILL。"""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def command(root, name, argv, *, timeout=1800, cwd=None):
    """Only kill our process group; retain failure receipts and board samples."""
    if not re.fullmatch(r"[a-z0-9_-]+", name) or not 0 < timeout <= 7200:
        raise ValueError("invalid bounded stage")
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    budget_path = evidence / "runtime_budget.json"
    if not budget_path.exists():
        save(budget_path, {"started_unix": time.time(), "maximum_wall_seconds": 21600,
                          "note": "Six-hour run ceiling includes setup and idle phase gaps; operator still must stop the VM."})
    budget = json.loads(budget_path.read_text())
    remaining_budget = budget["maximum_wall_seconds"] - (time.time() - budget["started_unix"])
    if remaining_budget <= 15:
        raise RuntimeError("six-hour runtime budget exhausted; archive artifacts and stop the owned VM")
    timeout = min(timeout, remaining_budget - 10)
    argv = [str(value) for value in argv]
    base = evidence / (name + ".json")
    if base.exists():
        old = json.loads(base.read_text())
        if old.get("status") == "succeeded" and old.get("argv") == argv:
            print(f"[paired] {name}: already succeeded; preserving evidence", flush=True)
            return old
        raise ValueError("stage already attempted; preserve it and use a fresh run ID")
    logfile, samples = evidence / (name + ".log"), evidence / (name + "_gpu.jsonl")
    started = time.monotonic()
    receipt = {"argv": argv, "timeout_seconds": timeout, "status": "starting", "pid": None,
               "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save(base, receipt)
    print(f"[paired] {name}: starting, limit {timeout}s", flush=True)
    proc = None
    try:
        with logfile.open("xb") as stream:
            proc = subprocess.Popen(argv, cwd=cwd or root, env=legacy.environment(root),
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(status="running", pid=proc.pid)
            save(base, receipt)
            while proc.poll() is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                try:
                    proc.wait(timeout=min(2, remaining))
                except subprocess.TimeoutExpired:
                    _board_sample(samples)
            receipt.update(returncode=proc.returncode, status="succeeded" if proc.returncode == 0 else "failed")
            if proc.returncode:
                raise RuntimeError(f"{name} exited {proc.returncode}")
    except BaseException as exc:
        receipt.update(status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
                       error_type=type(exc).__name__)
        if proc is not None and proc.poll() is None:
            terminate_group(proc)
            receipt["returncode"] = proc.poll()
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic() - started,
                       log_sha256=file_hash(logfile) if logfile.exists() else None)
        save(base, receipt)
        print(f"[paired] {name}: {receipt['status']}", flush=True)
    return receipt


def archive_results(root):
    """Lightweight evidence/inputs/plans; complete best is published separately."""
    destination = root / "artifacts.zip"
    entries = [root / "bundle_manifest.json"]
    for directory in (root / "evidence", root / "inputs", root / "experiments/ms_swift/.cache/paired"):
        if directory.is_dir():
            entries.extend(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink()
                           and not any(re.fullmatch(r"checkpoint-[0-9]+", part) for part in path.relative_to(root).parts))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(entries)):
            if path.exists():
                archive.write(path, path.relative_to(root).as_posix())
    print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size,
                      "sha256": file_hash(destination)}), flush=True)
    return destination


def setup(root):
    # Reuse successful Python3.12 / exact package pins without editing the prior driver.
    legacy.command = command
    legacy.setup(root)


def probe_child(root, kind, length):
    """Disposable synthetic workload. No corpus admission and no quality claims."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration, set_seed
    from cpt.launch import _environment
    from cpt.swift_verify import _runtime_provenance

    output = root / "evidence" / f"probe_{kind}_{length}_result.json"
    if output.exists():
        raise ValueError("probe result already exists")
    environment = _environment()
    provenance = _runtime_provenance()
    assets = root / "experiments/ms_swift/.cache/model_assets/gemma4-e2b"
    lock = json.loads((assets / "model.lock.json").read_text())
    report = {"schema_version": "paired-resource-probe-v1", "kind": kind, "input_length": length,
              "environment": environment, "runtime_provenance": provenance,
              "status": "starting", "steps": [], "model_revision": lock["revision"],
              "data_kind": "deterministic synthetic token workload; no QA or evaluation data",
              "quality_improvement_measured": False, "checkpoint_saved": False,
              "scope": "Resource engineering only. Direct HF/PEFT workload, not a completed MS-SWIFT dataset pilot."}
    began = time.monotonic()
    save(output, report)
    try:
        set_seed(17)
        tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
        body = tokenizer.encode("Deterministic engineering resource workload. This text contains no training or benchmark answer.\n", add_special_tokens=False)
        ids = ([tokenizer.bos_token_id] + (body * math.ceil(length / len(body))))[:length]
        tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
        mask = torch.ones_like(tokens)
        report["input_ids_sha256"] = __import__("hashlib").sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        config = AutoConfig.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
        model = Gemma4ForConditionalGeneration.from_pretrained(lock["repository"], revision=lock["revision"],
                config=config, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        if kind == "train":
            from peft import LoraConfig, get_peft_model
            targets = [name for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear) and name.startswith("model.language_model.")]
            if not targets:
                raise ValueError("no language-model linear modules found")
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                                  target_modules=targets, task_type="CAUSAL_LM"))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model.train()
            model.config.use_cache = False
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            report.update(lora_target_modules=targets, trainable_parameters=sum(p.numel() for p in trainable),
                          gradient_accumulation_steps=1, gradient_checkpointing=True,
                          supervised_tokens_per_step=32, objective="next-token cross entropy on final32token targets; full prompt attention")
            for step in range(1, 4):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                # Predict the last32 tokens from preceding32 positions, avoiding a full L*V logit tensor.
                positions = torch.arange(length - 33, length - 1, device="cuda")
                result = model(input_ids=tokens, attention_mask=mask, use_cache=False, logits_to_keep=positions)
                logits = result.logits.float()
                if logits.shape[:2] != (1, 32):
                    raise ValueError("unexpected reduced-logit shape")
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, -32:].reshape(-1))
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("nonfinite resource loss")
                loss.backward()
                gradients = [p.grad for p in trainable if p.grad is not None]
                if not gradients or any(not bool(torch.isfinite(grad).all().item()) for grad in gradients):
                    raise ValueError("missing or nonfinite LoRA gradients")
                optimizer.step()
                torch.cuda.synchronize()
                report["steps"].append({"optimizer_step": step, "input_tokens": tokens.numel(),
                    "supervised_tokens": 32, "loss": float(loss.detach().cpu().item()),
                    "wall_seconds": time.monotonic() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gradient_tensors": len(gradients)})
                save(output, report)
                del loss, logits, result, gradients
        else:
            model.eval()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(input_ids=tokens, attention_mask=mask, do_sample=False,
                                           max_new_tokens=32, min_new_tokens=32, use_cache=True,
                                           pad_token_id=0, eos_token_id=None)
            torch.cuda.synchronize()
            new_tokens = generated.shape[-1] - length
            if new_tokens != 32:
                raise ValueError("inference workload did not generate the full32tokens")
            report["steps"].append({"input_tokens": tokens.numel(), "generated_tokens": new_tokens,
                    "wall_seconds": time.monotonic() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        report["status"] = "succeeded"
    except BaseException as exc:
        report.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed",
                      error_type=type(exc).__name__, error=str(exc)[:1500])
        raise
    finally:
        report["total_wall_seconds"] = time.monotonic() - began
        save(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def probe(root):
    legacy.check_gpu(root)
    results = []
    for kind, lengths in (("train", (4096, 8192, 16384)), ("infer", (4096, 8192, 16384, 32768))):
        for length in lengths:
            stage = f"probe_{kind}_{length}"
            try:
                command(root, stage, [root / ".venv/bin/python", Path(__file__).resolve(),
                        "--run-id", root.name.removeprefix("ariadne_paired_"), "--action", "probe-child",
                        "--probe-kind", kind, "--probe-length", str(length), "--execute"],
                        timeout=900 if kind == "train" else 600)
            except (RuntimeError, subprocess.TimeoutExpired):
                # OOM/timeout is an observed unsupported point, not a successful length.
                pass
            result_path = root / "evidence" / (stage + "_result.json")
            stage_result = json.loads((root / "evidence" / (stage + ".json")).read_text())
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            results.append({"kind": kind, "length": length, "process_status": stage_result["status"],
                            "result_status": result.get("status", "no_result"),
                            "completed_steps": len(result.get("steps", [])),
                            "result_path": str(result_path) if result_path.exists() else None})
            save(root / "evidence/resource_summary.json", {"points": results, "quality_improvement_measured": False})
    return results


def paired_recipe(train_count):
    from cpt.schema import default_recipe, validate_recipe
    if type(train_count) is not int or not 1 <= train_count <= 128:
        raise ValueError("paired pilot requires 1..128 genuinely admitted training pairs")
    steps = math.ceil(train_count / 8) * 2
    recipe = default_recipe("sft_lora", sequence_length=8192, smoke=True)
    recipe["parameters"].update(max_steps=steps, num_train_epochs=2, save_steps=2,
            eval_steps=2, save_strategy="best", metric_for_best_model="eval_loss",
            greater_is_better=False, load_best_model_at_end=True, save_total_limit=2,
            add_version=False, save_only_model=False, use_flash_ckpt=False,
            seed=17, data_seed=17, gradient_accumulation_steps=8,
            per_device_train_batch_size=1, per_device_eval_batch_size=1)
    validate_recipe(recipe)
    return recipe


def paired_record_contract(records):
    """Validate paired identities and targets before any export or GPU work.

    The generated prompt ends in ``</source_excerpts>\n\nQuestion: ...``.
    Source text may differ in length; question and answer comparisons
    preserve the original strings, including whitespace. Matching answers alone
    do not demonstrate that two arms contain the same training questions.
    """
    if not isinstance(records, dict) or set(records) != {"short", "long"}:
        raise ValueError("paired records require exactly short and long arms")
    questions = {}
    for arm in ("short", "long"):
        rows = records[arm]
        if not isinstance(rows, list) or not rows:
            raise ValueError("paired arm records must be nonempty lists")
        seen, questions[arm] = set(), []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("paired records must be objects")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError("paired IDs must be nonempty and unique within each arm")
            seen.add(identifier)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError("paired arms require one no-system user/assistant conversation")
            for message, role in zip(messages, ("user", "assistant")):
                if (not isinstance(message, dict) or set(message) != {"role", "content"}
                        or message.get("role") != role or not isinstance(message.get("content"), str)
                        or not message["content"].strip()):
                    raise ValueError("paired arms require exact no-system user/assistant roles and nonempty text")
            sources = row.get("source_document_ids")
            if (not isinstance(sources, list) or not sources
                    or any(not isinstance(source, str) or not source.strip() for source in sources)
                    or len(set(sources)) != len(sources)):
                raise ValueError("paired source IDs must be a nonempty unique ordered list")
            if row.get("split") not in {"train", "dev"}:
                raise ValueError("paired records must use train/dev splits")
            context, separator, question = messages[0]["content"].rpartition("\n\nQuestion: ")
            if not separator or not context.endswith("</source_excerpts>") or not question.strip():
                raise ValueError("paired prompts require the strict generated Question suffix")
            questions[arm].append(question)
    if len(records["short"]) != len(records["long"]):
        raise ValueError("paired arm record counts differ")
    for index, (short, long) in enumerate(zip(records["short"], records["long"])):
        if short["id"] != long["id"]:
            raise ValueError("paired IDs and record order must be identical across arms")
        if questions["short"][index] != questions["long"][index]:
            raise ValueError("paired arms must contain the exact same question")
        if (short["split"] != long["split"] or short["source_document_ids"] != long["source_document_ids"]
                or short["messages"][-1] != long["messages"][-1]):
            raise ValueError("paired arms must align split, ordered sources and exact answers")


def prepare_child(root, licenses):
    from cpt.sft import prepare_sft
    from cpt.launch import plan
    from cpt.swift_verify import verify_template
    work = root / "experiments/ms_swift"
    destination = work / ".cache/paired"
    inputs = root / "inputs"
    records = {arm: [json.loads(line) for line in (inputs / f"{arm}_records.jsonl").read_text().splitlines() if line.strip()]
               for arm in ("short", "long")}
    paired_record_contract(records)
    counts = {split: sum(row["split"] == split for row in records["short"]) for split in ("train", "dev")}
    if sum(counts.values()) != len(records["short"]) or not 1 <= counts["dev"] <= 32:
        raise ValueError("pilot needs nonempty train/dev only and at most32devpairs")
    recipe = paired_recipe(counts["train"])
    save(destination / "recipe.json", recipe)
    results = {}
    for arm in ("short", "long"):
        manifest = prepare_sft(inputs / f"{arm}_records.jsonl", inputs / "sources.jsonl",
                work / ".cache/model_assets/gemma4-e2b", destination / arm / "prepared",
                reviews_path=inputs / f"{arm}_reviews.jsonl", source_reviews_path=inputs / "source_reviews.jsonl",
                allowed_licenses=licenses, sequence_length=8192, mode="training")
        if manifest["counts"] != counts or manifest["approved_records"] != sum(counts.values()):
            raise ValueError("all paired rows must be admitted; silent row filtering changes the experiment")
        launch_plan = plan(destination / arm / "prepared", destination / "recipe.json", destination / arm / "plan")
        if not launch_plan["training_ready"]:
            raise ValueError("paired launch plan remains blocked")
        actual = verify_template(launch_plan)
        save(destination / arm / "template_preflight.json", actual)
        for split, observed in actual["datasets"].items():
            if observed["records"] != counts[split] or observed["dropped_records"]:
                raise ValueError("real SWIFT preprocessing changed the admitted pair count")
            if arm == "short" and observed["max_input_length"] > 2048:
                raise ValueError("short arm exceeds2048 native chat tokens")
            if arm == "long" and (observed["min_input_length"] < 4096 or observed["max_input_length"] > 8192):
                raise ValueError("long arm must be4096..8192 actual native chat tokens")
        results[arm] = {"plan": str(destination / arm / "plan/launch_plan.json"),
                        "plan_sha256": file_hash(destination / arm / "plan/launch_plan.json"),
                        "template_preflight_sha256": file_hash(destination / arm / "template_preflight.json"),
                        "datasets": actual["datasets"]}
    for split in ("train", "dev"):
        short, long = results["short"]["datasets"][split], results["long"]["datasets"][split]
        if any(short[key] != long[key] for key in ("records", "loss_tokens", "ordered_loss_token_ids_sha256", "loss_token_histogram_sha256")):
            raise ValueError("paired arms have different actual supervised label sequences")
    report = {"schema_version": "paired-pilot-preparation-v1", "verified": True, "counts": counts,
              "arms": results, "optimizer_steps_per_arm": recipe["parameters"]["max_steps"],
              "epochs": 2, "microbatch": 1, "gradient_accumulation": 8, "seed": 17,
              "proposal_counts": {"train": 128, "dev": 32},
              "sample_size_amendment": counts != {"train": 128, "dev": 32},
              "amendment_reason": "Use actual reviewed independent pairs; do not duplicate records to fill the proposal.",
              "budget_comparison": "Equal examples, targets, optimizer steps and order; unequal input-token exposure is measured, not hidden.",
              "training_completed": False, "quality_improvement_measured": False}
    save(destination / "pair_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def verified_pair(root):
    path = root / "experiments/ms_swift/.cache/paired/pair_verification.json"
    report = json.loads(path.read_text())
    if report.get("schema_version") != "paired-pilot-preparation-v1" or not report.get("verified"):
        raise ValueError("paired preparation did not pass")
    for arm, item in report["arms"].items():
        plan_path = Path(item["plan"])
        expected = root / f"experiments/ms_swift/.cache/paired/{arm}/plan/launch_plan.json"
        if plan_path != expected or file_hash(plan_path) != item["plan_sha256"]:
            raise ValueError("paired plan changed since comparison")
    return report


def checkpoint_for(root, arm):
    plan = root / f"experiments/ms_swift/.cache/paired/{arm}/plan/launch_plan.json"
    data = json.loads(plan.read_text())
    from best_checkpoint import inspect_checkpoint
    selection = json.loads((plan.parent / "training_audit/best.json").read_text())
    files = sorted(Path(data["checkpoint_dir"]).rglob("adapter_model.safetensors"))
    if len(files) != 1:
        raise ValueError("expected exactly one retained best adapter")
    checkpoint = files[0].parent
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if (state["global_step"] != selection["step"]
            or inspect_checkpoint(checkpoint, selection["step"], selection["eval_loss"]) != selection["files"]):
        raise ValueError("checkpoint differs from the verified best selection")
    return checkpoint


def reload_child(root, arm):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration
    from gemma_eval.core import adapter_files
    checkpoint = checkpoint_for(root, arm)
    provenance = adapter_files(checkpoint)  # successful unchanged plan/data/receipt ancestry
    plan = json.loads((root / f"experiments/ms_swift/.cache/paired/{arm}/plan/launch_plan.json").read_text())
    assets = Path(plan["model_assets"])
    model = Gemma4ForConditionalGeneration.from_pretrained(plan["recipe"]["model"]["repo"],
            revision=plan["recipe"]["model"]["revision"],
            config=AutoConfig.from_pretrained(assets, local_files_only=True, trust_remote_code=False),
            local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
    tokens = tokenizer("A short engineering adapter reload test.", return_tensors="pt")["input_ids"].to("cuda")
    with torch.inference_mode():
        logits = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits
    torch.cuda.synchronize()
    if not bool(torch.isfinite(logits).all().item()) or any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("adapter reload returned invalid logits or non-CUDA parameters")
    report = {"schema_version": "paired-checkpoint-reload-v1", "arm": arm, "checkpoint": str(checkpoint),
              "adapter_provenance": provenance, "checkpoint_reload_verified": True, "finite_logits": True,
              "quality_improvement_measured": False}
    save(root / f"evidence/{arm}_checkpoint_reload.json", report)
    print(json.dumps(report, indent=2), flush=True)


def train(root, arm):
    pair = verified_pair(root)
    python = root / ".venv/bin/python"
    command(root, "train_" + arm, [python, root / "experiments/colab/training_audit.py", "launch",
            "--plan", pair["arms"][arm]["plan"], "--execute"], timeout=1800)
    command(root, "reload_" + arm, [python, Path(__file__).resolve(), "--run-id",
            root.name.removeprefix("ariadne_paired_"), "--action", "reload-child", "--arm", arm, "--execute"], timeout=600)
    audit_paths = {a: root / f"experiments/ms_swift/.cache/paired/{a}/plan/training_audit/summary.json"
                   for a in ("short", "long")}
    if all(path.is_file() for path in audit_paths.values()):
        observations = {a: json.loads(path.read_text()) for a, path in audit_paths.items()}
        keys = ("optimizer_steps", "consumed_sequences", "supervised_tokens", "ordered_consumed_label_hashes_sha256")
        equal = all(observations["short"][key] == observations["long"][key] for key in keys)
        save(root / "evidence/consumed_pair_comparison.json", {"equal_supervised_exposure": equal,
             "arms": observations, "compared_fields": keys, "quality_improvement_measured": False})
        if not equal:
            raise ValueError("observed training supervision/order differs between paired arms")


def evaluate(root, arm):
    work, python = root / "experiments/ms_swift", root / ".venv/bin/python"
    for chosen in (("base", "short", "long") if arm == "all" else (arm,)):
        output = work / ".cache/paired/evaluation" / chosen
        argv = [python, "-m", "gemma_eval", "prepare", "--kind", "chat",
                "--input", root / "inputs/eval_records.jsonl", "--model-assets", work / ".cache/model_assets/gemma4-e2b",
                "--output", output / "plan", "--context-length", "16384", "--quantization", "bf16",
                "--cache-dir", root / "hf_cache/hub"]
        if chosen != "base":
            verified_pair(root)
            reload_path = root / f"evidence/{chosen}_checkpoint_reload.json"
            if not json.loads(reload_path.read_text()).get("checkpoint_reload_verified"):
                raise ValueError("evaluate only successfully reloaded adapters")
            argv.extend(["--adapter", checkpoint_for(root, chosen)])
        command(root, "eval_prepare_" + chosen, argv, timeout=300, cwd=work)
        try:
            command(root, "eval_" + chosen, [python, "-m", "gemma_eval.bounded", "--plan", output / "plan/plan.json",
                    "--output", output / "run", "--max-seconds", "5400", "--execute"], timeout=5460, cwd=work)
        except RuntimeError:
            # CLI 对样本失败也返回1；仅完整、有模型、哈希一致的结果可继续比较。
            from gemma_eval.core import load_plan
            from evaluation_outcome import completed_evaluation
            _, requests = load_plan(output / "plan/plan.json")
            outcome = completed_evaluation(output / "run", [row["id"] for row in requests])
            save(root / ("evidence/eval_outcome_" + chosen + ".json"), outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--action", required=True, choices=["setup", "probe", "probe-child", "prepare", "prepare-child", "train-short", "train-long", "reload-child", "eval", "archive"])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-license", action="append", default=[])
    parser.add_argument("--probe-kind", choices=["train", "infer"])
    parser.add_argument("--probe-length", type=int, choices=[4096, 8192, 16384, 32768])
    parser.add_argument("--arm", choices=["short", "long"])
    parser.add_argument("--eval-arm", choices=["base", "short", "long", "all"], default="all")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("cloud execution requires Linux; local bundle/tests remain portable")
    root = run_root(args.run_id)
    if args.action == "setup" and not root.exists():
        if not args.bundle or not args.bundle_sha256:
            parser.error("setup requires an explicit bundle and SHA256")
        legacy.unpack(args.bundle, args.bundle_sha256, root)
    verify_bundle(root)
    if args.action in {"probe", "probe-child", "train-short", "train-long", "reload-child", "eval"} and not args.execute:
        parser.error("GPU execution requires --execute")
    if args.action == "probe-child":
        if args.probe_kind is None or args.probe_length is None:
            parser.error("probe child requires kind and length")
        probe_child(root, args.probe_kind, args.probe_length)
        return
    if args.action == "prepare-child":
        prepare_child(root, args.allow_license)
        return
    if args.action == "reload-child":
        if not args.arm:
            parser.error("reload requires --arm")
        reload_child(root, args.arm)
        return
    event = {"action": args.action, "status": "starting", "driver_sha256": file_hash(__file__),
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    append(root / "evidence/progress.jsonl", event) if (root / "evidence").exists() else None
    try:
        if args.action == "setup":
            setup(root)
        elif args.action == "probe":
            probe(root)
        elif args.action == "prepare":
            argv = [root / ".venv/bin/python", Path(__file__).resolve(), "--run-id", args.run_id, "--action", "prepare-child"]
            for license_name in args.allow_license:
                argv.extend(["--allow-license", license_name])
            command(root, "paired_prepare", argv, timeout=600)
        elif args.action in {"train-short", "train-long"}:
            train(root, args.action.removeprefix("train-"))
        elif args.action == "eval":
            evaluate(root, args.eval_arm)
        elif args.action == "archive":
            pass
        event["status"] = "completed"
    except BaseException as exc:
        event.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        append(root / "evidence/progress.jsonl", event)
        archive_results(root)


if __name__ == "__main__":
    main()
