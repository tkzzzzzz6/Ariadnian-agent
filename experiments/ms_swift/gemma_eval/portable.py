"""Read-only provenance inspection of a downloaded Linux training archive.

This is deliberately NOT a portable launch or adapter-admission mechanism. It
records an explicit original-root/local-root map while leaving historical plan,
receipt, and prepared bytes untouched. Evaluation still requires a fresh plan
at the original cloud paths and core.adapter_files' complete admission checks.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from cpt.core import file_hash
from cpt.launch import _artifact
from cpt.schema import MODEL_REPO, MODEL_REVISION
from .core import adapter_base_matches


def inspect_archive(artifact_root, original_root, adapter_relative):
    root = Path(artifact_root).resolve(strict=True)
    original = PurePosixPath(original_root)
    if not original.is_absolute() or ".." in original.parts or "\\" in str(original_root):
        raise ValueError("original_root must be a canonical absolute Linux path")

    def mapped(remote):
        path = PurePosixPath(remote)
        if not path.is_absolute() or ".." in path.parts or "\\" in str(remote):
            raise ValueError("archive metadata contains a noncanonical remote path")
        try:
            relative = path.relative_to(original)
        except ValueError as exc:
            raise ValueError("archive metadata path escapes the explicit original root") from exc
        local = (root / Path(*relative.parts)).resolve(strict=True)
        if not local.is_relative_to(root):
            raise ValueError("mapped archive path escapes the local root")
        return local

    relative = PurePosixPath(adapter_relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(adapter_relative):
        raise ValueError("adapter_relative must remain inside the archive root")
    adapter = mapped(str(original / relative))
    owner = next((p for p in [adapter, *list(adapter.parents)[:3]]
                  if p.is_relative_to(root) and (p / "launch_plan.json").is_file()), None)
    if owner is None:
        raise ValueError("archive adapter has no bound original launch plan")
    plan_path, receipt_path = owner / "launch_plan.json", owner / "launch_receipt.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (plan.get("schema_version") != "cpt-launch-plan-v1"
            or plan.get("recipe", {}).get("model") != {"repo": MODEL_REPO, "revision": MODEL_REVISION}
            or receipt.get("plan_sha256") != file_hash(plan_path)
            or receipt.get("status") != "process_exited" or receipt.get("returncode") != 0
            or plan.get("training_ready") is not True or plan.get("blockers") != []):
        raise ValueError("archive plan/model/success receipt mismatch")
    if mapped(plan["output_dir"]) != owner or not adapter.is_relative_to(mapped(plan["checkpoint_dir"])):
        raise ValueError("adapter or plan is outside its original bound directory")
    prepared = mapped(plan["prepared_dir"])
    manifest_path = prepared / "prepare_manifest.json"
    if file_hash(manifest_path) != plan["prepare_manifest_sha256"]:
        raise ValueError("archived prepared manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "training" or manifest.get("training_ready") is not True:
        raise ValueError("archived export was preview or not admitted")
    artifacts = {key: _artifact(prepared, manifest, key) for key in manifest["artifacts"]}
    for split in ("train", "dev"):
        path = mapped(plan["datasets"][split])
        if not path.is_relative_to(prepared) or file_hash(path) != plan["dataset_sha256"][split]:
            raise ValueError("archived dataset hash/path mismatch")
    verification_path = owner / "template_verification.json"
    if file_hash(verification_path) != receipt.get("template_verification_sha256"):
        raise ValueError("archived template verification hash mismatch")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("verified") is not True or verification.get("model") != plan["recipe"]["model"]:
        raise ValueError("archived template verification did not pass")
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (not adapter_base_matches(config.get("base_model_name_or_path")) or config.get("peft_type") != "LORA"
            or config.get("revision") not in (None, MODEL_REVISION)):
        raise ValueError("archive adapter model/revision/method mismatch")
    weights = sorted(adapter.glob("adapter_model*.safetensors"))
    if not weights:
        raise ValueError("archived adapter has no safetensors weights")
    def bound(path):
        return {"relative_path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
    return {"schema_version": "gemma-portable-archive-inspection-v1",
            "root_mapping": {"original": str(original), "local": str(root)},
            "adapter_relative": relative.as_posix(), "source_plan": bound(plan_path),
            "source_receipt": bound(receipt_path), "source_template_verification": bound(verification_path),
            "source_prepared_manifest": bound(manifest_path),
            "prepared_artifacts": {key: bound(path) for key, path in artifacts.items()},
            "adapter_files": {path.name: file_hash(path) for path in [config_path, *weights]},
            "original_model_lock_sha256": plan["model_lock_sha256"],
            "historical_file_hashes_verified": True, "historical_files_rewritten": False,
            "admission_revalidated": False, "execution_ready": False,
            "note": "Read-only archive provenance. Hash agreement is not renewed admission, portable execution, adapter reload, or quality evidence."}
