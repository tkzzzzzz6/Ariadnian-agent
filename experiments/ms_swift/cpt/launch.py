"""Reviewable plans and opt-in training in an isolated Linux CUDA environment."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .core import digest, file_hash, new_output, read_jsonl, write_json
from .schema import load_recipe, recipe_sha256, validate_model_lock, validate_recipe, BITSANDBYTES_VERSION


def _locked_assets(path):
    root = Path(path).resolve()
    lock_path = root / "model.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_model_lock(lock, root)
    return lock, file_hash(lock_path)


def _artifact(root, manifest, key):
    spec = manifest["artifacts"][key]
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != spec["sha256"]:
        raise ValueError(f"prepared artifact hash/path mismatch: {key}")
    return path


def _argv(recipe, datasets, checkpoint_dir):
    sft = recipe["method"].startswith("sft_")
    args = [sys.executable, "-m", "swift.cli.sft" if sft else "swift.cli.pt", "--model", recipe["model"]["repo"],
            "--model_revision", recipe["model"]["revision"], "--use_hf", "true",
            "--dataset", datasets["train"], "--val_dataset", datasets["dev"],
            *([] if sft else ["--columns", '{"text":"response"}']), "--output_dir", checkpoint_dir,
            "--push_to_hub", "false"]
    for key, value in recipe["parameters"].items():
        args.append("--" + key)
        if isinstance(value, list):
            args.extend(str(v) for v in value)
        elif isinstance(value, bool):
            args.append(str(value).lower())
        else:
            args.append(str(value))
    return args


def _build(prepared_dir, recipe, output_dir, strategy, init_adapter=None):
    validate_recipe(recipe)
    if recipe["method"].startswith("sft_"):
        from .sft import build_sft
        return build_sft(prepared_dir, recipe, output_dir, init_adapter=init_adapter)
    if init_adapter:
        raise ValueError("initial adapters are supported only for the CPT-to-SFT continuation")
    if strategy not in {"ordered", "shuffled"}:
        raise ValueError("strategy must be ordered or shuffled")
    root = Path(prepared_dir).resolve()
    path = root / "prepare_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "cpt-prepared-v1":
        raise ValueError("unsupported prepared manifest")
    model_assets = Path(manifest["model_assets"]).resolve()
    lock, lock_sha = _locked_assets(model_assets)
    if recipe["model"] != {"repo": lock["repository"], "revision": lock["revision"]}:
        raise ValueError("recipe and native tokenizer refer to different model revisions")
    if manifest["tokenizer"].get("model_lock_sha256") != lock_sha:
        raise ValueError("prepared tokenizer lock differs from current model assets")
    if manifest["sequence_length"] != recipe["parameters"]["max_length"]:
        raise ValueError("recipe max_length differs from native-tokenizer preparation")
    # Bind every artifact, including receipts and excluded documents, so an
    # edited report or one arm's changed content cannot silently change a run.
    for key in manifest["artifacts"]:
        _artifact(root, manifest, key)
    datasets = {"train": str(_artifact(root, manifest, "train_" + strategy)),
                "dev": str(_artifact(root, manifest, "dev_ordered"))}
    blockers = []
    if manifest.get("mode") != "training" or manifest.get("training_ready") is not True:
        blockers.append("preview_or_unapproved_data")
    admission = read_jsonl(_artifact(root, manifest, "admission_report.jsonl"))
    eligibility = {r["document_id"]: r for r in admission}
    assignments = read_jsonl(_artifact(root, manifest, "split_assignments.jsonl"))
    splits = {r["document_id"]: r["split"] for r in assignments}
    if len(eligibility) != len(admission) or len(splits) != len(assignments):
        raise ValueError("duplicate admission or split assignment record")
    lineage = read_jsonl(_artifact(root, manifest, "sequence_lineage.jsonl"))
    if any(not eligibility.get(r["document_id"], {}).get("eligible") for r in lineage):
        blockers.append("exported_document_not_admitted")
    if any(r["split"] not in {"train", "dev"} for r in lineage) or manifest.get("test_documents_exported") != 0:
        raise ValueError("test/holdout data cannot be supplied to CPT")
    for item in lineage:
        admitted = eligibility.get(item["document_id"], {})
        if (splits.get(item["document_id"]) != item["split"] or admitted.get("split") != item["split"]
                or admitted.get("document_sha256") != item["document_sha256"]
                or any(span.get("document_id") != item["document_id"] for span in item["spans"])):
            raise ValueError("sequence lineage conflicts with source admission or split assignment")
    # A valid hash of another dataset is insufficient: bind every exported row
    # to its particular split/strategy/source lineage, including unselected arms.
    text_hashes = {}
    for split in ("train", "dev"):
        for arm in ("ordered", "shuffled"):
            key = f"{split}_{arm}"
            rows = read_jsonl(_artifact(root, manifest, key))
            expected = [r for r in lineage if r["split"] == split and r["strategy"] == arm]
            expected = sorted(expected, key=lambda r: r["row"])
            if ([r["row"] for r in expected] != list(range(len(rows)))
                    or len(expected) != len(rows)
                    or any(set(r) != {"text"} or not isinstance(r["text"], str) or not r["text"].strip() for r in rows)
                    or any(digest(row["text"]) != source["text_sha256"] for row, source in zip(rows, expected))):
                raise ValueError(f"dataset rows conflict with bound split/strategy lineage: {key}")
            text_hashes[key] = {digest(row["text"]) for row in rows}
    if (datasets["train"] == datasets["dev"]
            or text_hashes["train_" + strategy] & text_hashes["dev_ordered"]):
        raise ValueError("train/dev datasets share a path or exact sequence text")
    for split, data_path in datasets.items():
        rows = read_jsonl(data_path)
        if not rows:
            blockers.append(f"empty_{split}_dataset")
        if any(set(r) != {"text"} or not isinstance(r["text"], str) or not r["text"].strip() for r in rows):
            raise ValueError("Swift exports must contain text-only nonempty rows")
    checkpoint = str(Path(output_dir).resolve() / "checkpoints")
    return {"schema_version": "cpt-launch-plan-v1", "recipe": recipe,
            "recipe_sha256": recipe_sha256(recipe), "prepared_dir": str(root),
            "prepare_manifest_sha256": file_hash(path), "model_assets": str(model_assets),
            "model_lock_sha256": lock_sha, "strategy": strategy, "datasets": datasets,
            "dataset_sha256": {k: file_hash(v) for k, v in datasets.items()},
            "output_dir": str(Path(output_dir).resolve()), "checkpoint_dir": checkpoint,
            "argv": _argv(recipe, datasets, checkpoint), "blockers": sorted(set(blockers)),
            "training_ready": not blockers,
            "note": "Planning does not execute training. Runtime CUDA/version/template checks remain mandatory. Fixed steps are a pilot, not an equal-token comparison."}


def plan(prepared_dir, recipe_path, output_dir, *, strategy="ordered", init_adapter=None) -> dict:
    result = _build(prepared_dir, load_recipe(recipe_path), output_dir, strategy, init_adapter)
    out = new_output(output_dir)
    write_json(out / "launch_plan.json", result)
    return result


def _adapter_provenance(directory, recipe):
    root = Path(directory).resolve(strict=True)
    config_path, weights = root / "adapter_config.json", root / "adapter_model.safetensors"
    if not config_path.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("CPT continuation requires adapter_config.json and nonempty safetensors weights")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != recipe["model"]["repo"]:
        raise ValueError("initial adapter base model/type mismatch")
    if config.get("r") != recipe["parameters"]["lora_rank"] or config.get("lora_alpha") != recipe["parameters"]["lora_alpha"]:
        raise ValueError("continuation LoRA rank/alpha differs from the bound recipe")
    owner = next((p for p in list(root.parents)[:3] if (p / "launch_plan.json").is_file()), None)
    if owner is None: raise ValueError("initial adapter lacks its source launch plan")
    parent_path = owner / "launch_plan.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    receipt_path = owner / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (parent.get("schema_version") != "cpt-launch-plan-v1" or parent["recipe"]["model"] != recipe["model"]
            or not parent["recipe"]["method"].startswith("pt_")
            or not root.is_relative_to(Path(parent["checkpoint_dir"]).resolve())
            or receipt.get("plan_sha256") != file_hash(parent_path)
            or receipt.get("returncode") != 0 or receipt.get("status") != "process_exited"):
        raise ValueError("initial adapter parent CPT provenance or process receipt is invalid")
    if parent["recipe"]["method"].endswith("qlora") != recipe["method"].endswith("qlora"):
        raise ValueError("CPT-to-SFT continuation must preserve the quantization method")
    parent_current = _build(parent["prepared_dir"], parent["recipe"], parent["output_dir"], parent["strategy"])
    if parent_current != {**parent, "argv": [sys.executable, *parent["argv"][1:]]} or not parent_current["training_ready"]:
        raise ValueError("initial adapter parent plan/data changed")
    return {"path": str(root), "parent_plan_sha256": file_hash(parent_path),
            "parent_receipt_sha256": file_hash(receipt_path),
            "files": {p.name: file_hash(p) for p in (config_path, weights)},
            "note": "Hashes bind the adapter; quality and numerical equivalence require a separate evaluation."}


def _quant_environment(recipe):
    if not recipe["method"].endswith("qlora"):
        return None
    try:
        actual = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("QLoRA requires the separate requirements-train-qlora.txt dependencies") from exc
    if actual != BITSANDBYTES_VERSION:
        raise ValueError(f"bitsandbytes must be {BITSANDBYTES_VERSION}; found {actual}")
    # Import failure (CUDA extension/ABI mismatch) must happen before launch.
    import bitsandbytes as bnb
    if not hasattr(bnb.nn, "Linear4bit"):
        raise ValueError("bitsandbytes Linear4bit runtime is unavailable")
    import torch
    # Small real CUDA extension probe, not a model/long-sequence memory test.
    # Merely importing bitsandbytes can succeed even when its CUDA backend cannot.
    with torch.no_grad():
        probe = torch.linspace(-1, 1, 4096, device="cuda", dtype=torch.bfloat16)
        packed, state = bnb.functional.quantize_4bit(probe, quant_type="nf4", compress_statistics=True)
        restored = bnb.functional.dequantize_4bit(packed, quant_state=state)
        torch.cuda.synchronize()
        if restored.shape != probe.shape or not bool(torch.isfinite(restored).all().item()):
            raise ValueError("bitsandbytes NF4 CUDA kernel smoke check failed")
    return {"bitsandbytes": actual, "quantization": "NF4 double quant; BF16 compute",
            "quantization_kernel_smoke_verified": True,
            "note": "Embeddings/PLE/lm_head are not assumed to be 4-bit; inspect actual loaded modules during profiling."}


def _environment():
    if platform.system() != "Linux":
        raise ValueError("training requires the separate Linux CUDA training environment; planning works on Windows")
    versions = {}
    for name, expected in {"ms-swift": "4.6.0", "torch": "2.8.0", "transformers": "5.12.1",
                           "peft": "0.20.0", "accelerate": "1.12.0", "datasets": "4.8.4", "trl": "0.29.1"}.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"training dependency missing: {name}") from exc
        if actual.split("+")[0] != expected:
            raise ValueError(f"training dependency mismatch: {name} expected {expected}, found {actual}")
        versions[name] = actual
    import torch
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("this bf16 recipe requires a CUDA GPU with native BF16 support")
    return {"versions": versions, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "visible_gpu_count": torch.cuda.device_count()}


def launch(plan_path, *, execute=False) -> dict:
    path = Path(plan_path).resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != "cpt-launch-plan-v1":
        raise ValueError("unsupported launch plan")
    current = _build(saved["prepared_dir"], saved["recipe"], saved["output_dir"], saved["strategy"],
                     saved.get("init_adapter", {}).get("path"))
    # argv is reconstructed, never executed from arbitrary supplied JSON.
    # The interpreter legitimately changes when moving to the training venv.
    saved_compare = {**saved, "argv": [sys.executable, *saved["argv"][1:]]}
    if current != saved_compare:
        raise ValueError("launch plan or its bound data/model/recipe has changed; create a fresh plan")
    if not execute:
        return {"executed": False, "training_ready": current["training_ready"],
                "blockers": current["blockers"], "argv": current["argv"]}
    if current["blockers"]:
        raise ValueError("training denied: " + ", ".join(current["blockers"]))
    out = Path(current["output_dir"])
    if path.parent != out:
        raise ValueError("launch plan must remain in its bound output directory; re-plan after server transfer")
    if (out / "launch_receipt.json").exists() or Path(current["checkpoint_dir"]).exists():
        raise ValueError("this run already started or has checkpoints; choose a new run directory")
    environment = _environment()
    quantization = _quant_environment(current["recipe"])
    if quantization:
        environment["quantization"] = quantization
    if current.get("init_adapter"):
        from safetensors import safe_open
        with safe_open(str(Path(current["init_adapter"]["path"]) / "adapter_model.safetensors"), framework="pt", device="cpu") as tensors:
            if not list(tensors.keys()):
                raise ValueError("initial adapter contains no tensors")
    from .swift_verify import verify_template
    verification = verify_template(current)
    if verification.get("verified") is not True:
        raise ValueError("SWIFT template verification did not pass")
    write_json(out / "template_verification.json", verification)
    receipt = {"schema_version": "cpt-launch-receipt-v1", "status": "starting",
               "plan_sha256": file_hash(path), "environment": environment,
               "template_verification_sha256": file_hash(out / "template_verification.json"),
               "argv": current["argv"], "training_completed": False}
    write_json(out / "launch_receipt.json", receipt)
    env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"}
    try:
        result = subprocess.run(current["argv"], shell=False, env=env, check=False)
    except BaseException as exc:
        receipt.update(status="interrupted_or_failed", error_type=type(exc).__name__)
        write_json(out / "launch_receipt.json", receipt)
        raise
    receipt.update(status="process_exited", returncode=result.returncode,
                   training_completed=False,
                   note="Exit code alone is not a validated checkpoint or measured CPT gain. Inspect checkpoints/loss and run pinned before/after evaluations.")
    write_json(out / "launch_receipt.json", receipt)
    if result.returncode:
        raise ValueError(f"SWIFT exited with code {result.returncode}; retained launch receipt")
    return receipt
