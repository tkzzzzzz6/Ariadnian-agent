"""CPU contracts only. Mock generations are not model evaluation results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cpt.core import file_hash, write_json, write_jsonl
from gemma_eval import core, runtime


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(c) + 10 for c in text])


@pytest.fixture
def native(monkeypatch, tmp_path):
    metadata = {"asset_directory": str(tmp_path / "assets"), "revision": core.MODEL_REVISION}
    monkeypatch.setattr(core, "tokenizer_assets", lambda _: (TinyTokenizer(), metadata))
    return tmp_path / "assets"


def example(eid="one", prompt="Question", cap=4):
    return {"id": eid, "benchmark": "longbench-v1", "task": "qasper", "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "official_max_new_tokens": cap}


def benchmark_file(tmp_path, examples):
    path = tmp_path / "selected_records.jsonl"
    write_jsonl(path, examples)
    write_json(tmp_path / "fetch_manifest.json", {"records": len(examples),
        "selected_records_sha256": file_hash(path), "source_manifests": [{"selected_ids": [x["id"] for x in examples]}]})
    return path


def test_complete_prompt_and_generation_reserve_determine_overflow():
    rows = core.benchmark_rows([example(prompt="abcd", cap=4)], TinyTokenizer(), 8)
    assert rows[0]["input_tokens"] == 5  # four prompt tokens plus BOS
    assert rows[0]["status"] == "skipped_context_limit"
    assert rows[0]["prompt"] == "abcd"
    assert core.benchmark_rows([example(prompt="abcd", cap=4)], TinyTokenizer(), 9)[0]["status"] == "ready"


def test_frozen_prompt_hash_rejects_accidental_question_edits():
    record = example()
    record["prompt"] += " reference answer"
    with pytest.raises(ValueError, match="prompt hash"):
        core.benchmark_rows([record], TinyTokenizer(), 100)


def test_profile_exact_lengths_are_resource_probes(tmp_path, native):
    seed = tmp_path / "seed.txt"
    seed.write_text("context", encoding="utf-8")
    plan = core.prepare(seed, native, tmp_path / "profile", kind="profile", lengths=[16, 32], max_new_tokens=3)
    _, requests = core.load_plan(tmp_path / "profile/plan.json")
    assert [r["input_tokens"] for r in requests] == [16, 32]
    assert all(r["input_ids"][0] == 2 for r in requests)
    assert plan["executed"] is False
    assert "no quality metric" in plan["protocol"]


def test_updated_request_hash_cannot_hide_token_tampering(tmp_path, native):
    source = benchmark_file(tmp_path, [example()])
    plan = core.prepare(source, native, tmp_path / "plan", context_length=100)
    requests_path = Path(plan["requests"]["path"])
    records = core.read_jsonl(requests_path)
    records[0]["input_ids"][2] += 1
    write_jsonl(requests_path, records)
    plan["requests"]["sha256"] = file_hash(requests_path)
    write_json(tmp_path / "plan/plan.json", plan)
    with pytest.raises(ValueError, match="native encoding"):
        core.load_plan(tmp_path / "plan/plan.json")


def test_source_manifest_order_is_required(tmp_path, native):
    path = benchmark_file(tmp_path, [example("one"), example("two")])
    manifest = json.loads((tmp_path / "fetch_manifest.json").read_text())
    manifest["source_manifests"][0]["selected_ids"].reverse()
    write_json(tmp_path / "fetch_manifest.json", manifest)
    with pytest.raises(ValueError, match="source manifest ID"):
        core.prepare(path, native, tmp_path / "out")


def test_modified_source_rejected_even_when_requests_unchanged(tmp_path, native):
    path = benchmark_file(tmp_path, [example()])
    core.prepare(path, native, tmp_path / "plan")
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source/request file changed"):
        core.load_plan(tmp_path / "plan/plan.json")


def test_nll_windows_score_every_target_exactly_once():
    ids = list(range(2, 19))
    windows = list(core.window_tokens(ids, 5))
    assert [target for w in windows for target in w["labels"][1:]] == ids[1:]
    assert sum(w["loss_tokens"] for w in windows) == len(ids) - 1
    assert all(w["labels"][0] == -100 for w in windows)
    assert all(len(w["input_ids"]) <= 5 for w in windows)


def test_nll_rejects_training_split(tmp_path, native):
    path = tmp_path / "corpus.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dev/test"):
        core.prepare(path, native, tmp_path / "out", kind="nll", split="train")


def test_adapter_without_training_provenance_is_not_a_verified_checkpoint(tmp_path):
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": core.MODEL_REPO,
                                                   "revision": core.MODEL_REVISION, "peft_type": "LORA"})
    (tmp_path / "adapter_model.safetensors").write_bytes(b"test fixture")
    with pytest.raises(ValueError, match="bound training launch plan"):
        core.adapter_files(tmp_path)


def run_plan(kind="benchmark"):
    return {"kind": kind, "protocol": "unit-test-only", "context_length": 8,
            "model": {"repo": core.MODEL_REPO, "revision": core.MODEL_REVISION}, "quantization": "nf4"}


def request(eid="one", status="ready"):
    return {"id": eid, "task": "qasper", "status": status, "input_ids": [2, 5, 6],
            "input_tokens": 3, "max_new_tokens": 3}


def test_load_failure_keeps_every_id_and_missing_module_diagnostic(tmp_path):
    def load(_):
        raise ModuleNotFoundError("missing", name="torch")
    result = runtime.execute_rows(run_plan(), [request(), request("two", "skipped_context_limit")], tmp_path / "run", load)
    assert result["total"] == 2
    assert result["successful"] == 0
    assert result["statuses"] == {"model_load_failed": 1, "skipped_context_limit": 1}
    assert result["load_error"]["missing_module"] == "torch"
    assert result["model_loaded"] is False
    assert len(core.read_jsonl(tmp_path / "run/predictions.jsonl")) == 2


def test_all_overflow_does_not_load_model(tmp_path):
    factory = Mock(side_effect=AssertionError("must not load"))
    result = runtime.execute_rows(run_plan(), [request(status="skipped_context_limit")], tmp_path / "run", factory)
    assert not factory.called
    assert result["failed_or_skipped"] == 1


def test_oom_and_recovery_failure_remain_in_denominator(tmp_path):
    fake = SimpleNamespace(metadata={"unit_test": True},
        generate=Mock(side_effect=[RuntimeError("CUDA out of memory"), {"status": "ok", "prediction": "fixture answer"}]),
        recover=Mock(side_effect=RuntimeError("recovery failure")))
    result = runtime.execute_rows(run_plan(), [request(), request("two")], tmp_path / "run", lambda _: fake)
    rows = core.read_jsonl(tmp_path / "run/predictions.jsonl")
    assert result["successful"] == 1 and result["total"] == 2
    assert rows[0]["error_category"] == "cuda_out_of_memory"
    assert rows[0]["error_stage"] == "benchmark"
    assert "recovery_error" in rows[0]


def test_incomplete_nll_does_not_claim_overall_loss(tmp_path):
    fake = SimpleNamespace(metadata={"unit_test": True},
        nll=Mock(side_effect=[{"status": "ok", "nll_sum": 4.0, "loss_tokens": 2}, RuntimeError("failed window")]),
        recover=lambda: None)
    rows = [{**request(str(i)), "loss_tokens": 2} for i in range(2)]
    result = runtime.execute_rows(run_plan("nll"), rows, tmp_path / "run", lambda _: fake)
    assert result["nll"]["complete"] is False
    assert result["nll"]["mean_nll"] is None
    assert result["nll"]["conditional_mean_nll_successful_documents"] == 2.0
    assert result["nll"]["scored_loss_tokens"] == 2
    assert result["nll"]["planned_loss_tokens"] == 4


def test_existing_results_never_overwritten(tmp_path):
    (tmp_path / "summary.json").write_text("previous evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        runtime.execute_rows(run_plan(), [request()], tmp_path, Mock())
    assert (tmp_path / "summary.json").read_text() == "previous evidence"


def cached_metadata_fixture(tmp_path):
    assets, cached = tmp_path / "assets", tmp_path / "cache"
    assets.mkdir()
    cached.mkdir()
    files = {}
    for name, value in {"config.json": {"model_type": "gemma4"},
                        "generation_config.json": {"do_sample": True, "eos_token_id": 1}}.items():
        write_json(assets / name, value)
        (cached / name).write_bytes((assets / name).read_bytes())
        files[name] = {"bytes": (assets / name).stat().st_size, "sha256": file_hash(assets / name)}
    write_json(assets / "model.lock.json", {"files": files})
    plan = {"model": {"repo": core.MODEL_REPO, "revision": core.MODEL_REVISION},
            "tokenizer": {"asset_directory": str(assets), "model_lock_sha256": file_hash(assets / "model.lock.json")}}
    resolver = lambda repo, name, **kwargs: str(cached / name)
    return plan, cached, resolver


@pytest.mark.parametrize("name", ["config.json", "generation_config.json"])
def test_cache_metadata_cannot_override_locked_assets(tmp_path, name):
    plan, cached, resolver = cached_metadata_fixture(tmp_path)
    assert set(runtime.verify_cached_metadata(plan, resolver)) == {"config.json", "generation_config.json"}
    write_json(cached / name, {"eos_token_id": 999, "stop_strings": ["modified"]})
    with pytest.raises(ValueError, match="cached runtime metadata differs"):
        runtime.verify_cached_metadata(plan, resolver)


def test_absent_cache_metadata_never_triggers_network_download(tmp_path):
    plan, _, _ = cached_metadata_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="cached runtime metadata is absent"):
        runtime.verify_cached_metadata(plan, lambda *args, **kwargs: None)


def test_greedy_config_is_new_and_has_no_hidden_stop_or_sampling_policy():
    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs
    first = runtime.greedy_generation_config(Config)
    first.values.update(do_sample=True, stop_strings=["injected"], eos_token_id=999)
    second = runtime.greedy_generation_config(Config)
    assert second.values["do_sample"] is False
    assert second.values["stop_strings"] is None
    assert second.values["eos_token_id"] == 1
    assert second.values["num_beams"] == 1
    assert second.values["min_length"] == 0
    assert second.values["forced_eos_token_id"] is None


def test_weight_hash_checks_content_not_multilevel_cache_symlink_name(tmp_path):
    content = b"small model-weight fixture"
    # Hub/Xet identifiers can be 64 hex characters without being file SHA-256.
    blob = tmp_path / ("b" * 64)
    blob.write_bytes(content)
    intermediate = tmp_path / "xet-reference"
    snapshot = tmp_path / "model.safetensors"
    try:
        os.symlink(blob, intermediate)
        os.symlink(intermediate, snapshot)
    except OSError:
        pytest.skip("host does not allow creating symbolic links")
    expected = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    assert snapshot.resolve().name != expected["sha256"]
    assert runtime._verify_weight_content(snapshot, expected) == expected["sha256"]


def test_weight_content_change_is_rejected_even_when_size_matches(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"bad")
    expected = {"bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}
    with pytest.raises(ValueError, match="pinned LFS SHA-256"):
        runtime._verify_weight_content(path, expected)


def test_weight_size_checked_before_reading_contents(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"partial")
    with pytest.raises(ValueError, match="pinned LFS size"):
        runtime._verify_weight_content(path, {"bytes": 1000, "sha256": "f" * 64})


def test_generation_limits_are_copied_without_mutating_other_requests():
    base = SimpleNamespace(max_new_tokens=None, use_cache=False, do_sample=False, eos_token_id=1)
    first = runtime.request_generation_config(base, 32)
    second = runtime.request_generation_config(base, 128)
    assert base.max_new_tokens is None and base.use_cache is False
    assert first.max_new_tokens == 32 and second.max_new_tokens == 128
    assert first.use_cache is True and first.eos_token_id == 1 and first.do_sample is False


def test_runtime_implementation_hashes_persist_even_on_load_failure(tmp_path):
    result = runtime.execute_rows(run_plan(), [request()], tmp_path / "run",
                                  Mock(side_effect=ModuleNotFoundError("missing", name="torch")))
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert result["implementation_sha256"] == manifest["implementation_sha256"]
    assert "gemma_eval/runtime.py" in result["implementation_sha256"]
    assert "gemma_eval/model_weights.lock.json" in result["implementation_sha256"]
    assert all(len(value) == 64 for value in result["implementation_sha256"].values())
