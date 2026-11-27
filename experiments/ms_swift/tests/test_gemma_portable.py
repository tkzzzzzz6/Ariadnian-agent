"""Archive integrity inspection never grants relocated execution or admission."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpt.core import file_hash, write_json, write_jsonl
from gemma_eval.core import MODEL_REPO, MODEL_REVISION, adapter_base_matches, adapter_files
from gemma_eval.portable import inspect_archive


@pytest.mark.parametrize("value,expected", [
    (MODEL_REPO, True),
    (f"/content/run/hf_cache/hub/models--google--gemma-4-E2B/snapshots/{MODEL_REVISION}", True),
    (f"D:/cache/models--google--gemma-4-E2B/snapshots/{MODEL_REVISION}", True),
    (f"models--google--gemma-4-E2B/snapshots/{MODEL_REVISION}", False),
    (f"/cache/../models--google--gemma-4-E2B/snapshots/{MODEL_REVISION}", False),
    (f"/cache/models--google--gemma-4-E2B-IT/snapshots/{MODEL_REVISION}", False),
    ("/cache/models--google--gemma-4-E2B/snapshots/main", False), (None, False),
])
def test_adapter_source_declaration_accepts_only_pinned_repo_snapshot(value, expected):
    assert adapter_base_matches(value) is expected


def archive(tmp_path):
    original = "/content/toy-run"
    prepared = tmp_path / "prepared"
    owner = tmp_path / "plan"
    adapter = owner / "checkpoints/checkpoint-1"
    prepared.mkdir()
    adapter.mkdir(parents=True)
    model = {"repo": MODEL_REPO, "revision": MODEL_REVISION}
    artifacts = {}
    for name in ("train", "dev", "admission", "lineage", "source_admission", "source_splits"):
        path = prepared / f"{name}.jsonl"
        write_jsonl(path, [{"fixture": name}])
        artifacts[name] = {"path": path.name, "sha256": file_hash(path)}
    write_json(prepared / "prepare_manifest.json", {"schema_version": "sft-prepared-v1", "mode": "training",
        "training_ready": True, "artifacts": artifacts})
    write_json(owner / "launch_plan.json", {"schema_version": "cpt-launch-plan-v1", "recipe": {"model": model},
        "prepared_dir": original + "/prepared", "prepare_manifest_sha256": file_hash(prepared / "prepare_manifest.json"),
        "output_dir": original + "/plan", "checkpoint_dir": original + "/plan/checkpoints",
        "training_ready": True, "blockers": [], "model_lock_sha256": "0" * 64,
        "datasets": {split: original + "/prepared/" + split + ".jsonl" for split in ("train", "dev")},
        "dataset_sha256": {split: file_hash(prepared / (split + ".jsonl")) for split in ("train", "dev")}})
    write_json(owner / "template_verification.json", {"verified": True, "model": model})
    write_json(owner / "launch_receipt.json", {"status": "process_exited", "returncode": 0,
        "plan_sha256": file_hash(owner / "launch_plan.json"),
        "template_verification_sha256": file_hash(owner / "template_verification.json")})
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": MODEL_REPO, "peft_type": "LORA"})
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture only, no model tensors")
    return original, adapter.relative_to(tmp_path).as_posix()


def test_inspection_preserves_historical_bytes_and_does_not_grant_execution(tmp_path):
    original, relative = archive(tmp_path)
    before = {p.relative_to(tmp_path).as_posix(): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = inspect_archive(tmp_path, original, relative)
    assert result["historical_file_hashes_verified"] is True
    assert result["admission_revalidated"] is False and result["execution_ready"] is False
    assert before == {p.relative_to(tmp_path).as_posix(): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, OSError)):
        adapter_files(tmp_path / relative)


def test_archive_tamper_and_root_escape_fail(tmp_path):
    original, relative = archive(tmp_path)
    with pytest.raises(ValueError, match="inside the archive"):
        inspect_archive(tmp_path, original, "../adapter")
    (tmp_path / "prepared/admission.jsonl").write_text("tampered evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        inspect_archive(tmp_path, original, relative)


def test_failed_training_receipt_cannot_be_reported_as_successful_archive(tmp_path):
    original, relative = archive(tmp_path)
    path = tmp_path / "plan/launch_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["returncode"] = 1
    write_json(path, receipt)
    with pytest.raises(ValueError, match="success receipt"):
        inspect_archive(tmp_path, original, relative)
