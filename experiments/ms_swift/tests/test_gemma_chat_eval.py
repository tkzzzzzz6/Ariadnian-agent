"""Protocol/integrity tests only; fake generations never stand in for model scores."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from cpt.core import file_hash, write_json, write_jsonl
from gemma_eval import chat, core, runtime


class ToyGemmaTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        pieces = re.split("(" + "|".join(re.escape(t) for t in chat.SPECIAL_TOKENS) + ")", text)
        ids = []
        for part in pieces:
            ids.extend([chat.SPECIAL_TOKENS[part]] if part in chat.SPECIAL_TOKENS else [ord(x) + 1000 for x in part])
        return SimpleNamespace(ids=ids)


@pytest.fixture
def native(monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    metadata = {"asset_directory": str(assets), "revision": core.MODEL_REVISION}
    monkeypatch.setattr(core, "tokenizer_assets", lambda _: (ToyGemmaTokenizer(), metadata))
    return assets


def example(eid="one", prompt="question", cap=4):
    return {"id": eid, "benchmark": "frozen-test-only", "task": "retrieval",
            "prompt": prompt, "max_new_tokens": cap, "evaluation_only": True,
            "answers": ["hidden gold answer"], "cluster_id": "source-1"}


def input_file(tmp_path, rows):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, rows)
    write_json(tmp_path / "eval_manifest.json", {"schema_version": "gemma-eval-input-v1",
        "records_sha256": file_hash(path), "records": len(rows), "selected_ids": [r["id"] for r in rows],
        "source_manifests": [{"source": "offline contract fixture, not a benchmark result"}]})
    return path


def test_exact_chat_wrapper_and_single_format_bos():
    tokenizer = ToyGemmaTokenizer()
    prompt = " 中文\nquestion? "
    encoded = chat.encode_chat(tokenizer, prompt)
    assert encoded == tokenizer.encode("<bos><|turn>user\n" + prompt + "<turn|>\n<|turn>model\n").ids
    assert encoded[0] == 2 and encoded.count(2) == 1
    assert encoded.count(105) == 2 and encoded.count(106) == 1
    assert encoded[-1] != 1
    assert tokenizer.encode(prompt).ids == encoded[7:7 + len(prompt)]


def test_hf_and_native_encoders_use_same_protocol():
    native = ToyGemmaTokenizer()
    hf = SimpleNamespace(encode=lambda text, **kwargs: native.encode(text, **kwargs).ids)
    assert chat.encode_chat(native, "test") == chat.encode_chat(hf, "test")
    with pytest.raises(ValueError, match="special token mismatch"):
        chat.encode_chat(SimpleNamespace(encode=lambda *a, **k: [7]), "test")


def test_generation_budget_includes_chat_format_and_reserve():
    tokenizer = ToyGemmaTokenizer()
    source = example()
    tokens = len(chat.encode_chat(tokenizer, source["prompt"]))
    row = chat.chat_rows([source], tokenizer, tokens + source["max_new_tokens"] - 1)[0]
    assert row["status"] == "skipped_context_limit"
    assert row["input_ids"] == chat.encode_chat(tokenizer, source["prompt"])
    assert row["prompt"] == source["prompt"]
    assert "answers" not in row and "hidden gold answer" not in row["prompt"]
    assert chat.chat_rows([source], tokenizer, tokens + source["max_new_tokens"])[0]["status"] == "ready"


@pytest.mark.parametrize("edit", [
    {"evaluation_only": False}, {"evaluation_only": None}, {"task": ""}, {"benchmark": None},
    {"max_new_tokens": True}, {"max_new_tokens": 0}, {"official_max_new_tokens": 5},
    {"prompt_sha256": "0" * 64},
])
def test_invalid_source_contract_is_rejected(edit):
    with pytest.raises(ValueError):
        chat.chat_rows([{**example(), **edit}], ToyGemmaTokenizer(), 100)


def test_plan_reconstruction_binds_protocol_sources_and_full_ids(tmp_path, native):
    source = input_file(tmp_path, [example(), example("two", "another")])
    plan = core.prepare(source, native, tmp_path / "plan", kind="chat", context_length=100)
    reloaded, rows = core.load_plan(tmp_path / "plan/plan.json")
    assert reloaded == plan and [r["id"] for r in rows] == ["one", "two"]
    assert plan["chat_protocol"] == chat.PROTOCOL and plan["protocol"] == chat.PROTOCOL_VERSION
    # Hash rewrites cannot hide a changed token sequence: reconstruct from source.
    rows[0]["input_ids"][3] += 1
    write_jsonl(plan["requests"]["path"], rows)
    plan["requests"]["sha256"] = file_hash(plan["requests"]["path"])
    write_json(tmp_path / "plan/plan.json", plan)
    with pytest.raises(ValueError, match="native encoding"):
        core.load_plan(tmp_path / "plan/plan.json")


def test_fixed_protocol_cannot_be_changed_in_a_plan(tmp_path, native):
    source = input_file(tmp_path, [example()])
    plan = core.prepare(source, native, tmp_path / "plan", kind="chat")
    plan["chat_protocol"]["stop_token_ids"] = [1]
    write_json(tmp_path / "plan/plan.json", plan)
    with pytest.raises(ValueError, match="protocol changed"):
        core.load_plan(tmp_path / "plan/plan.json")
    assert chat.PROTOCOL["stop_token_ids"] == [1, 106, 50]


def test_manifest_order_and_evidence_tamper_rejected(tmp_path, native):
    source = input_file(tmp_path, [example()])
    evidence = tmp_path / "source.lock.json"
    evidence.write_text("source snapshot", encoding="utf-8")
    manifest_path = tmp_path / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_manifests"][0]["file"] = {"path": evidence.name, "sha256": file_hash(evidence)}
    write_json(manifest_path, manifest)
    core.prepare(source, native, tmp_path / "plan", kind="chat")
    evidence.write_text("modified snapshot", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence file changed"):
        core.load_plan(tmp_path / "plan/plan.json")
    manifest["selected_ids"] = ["different"]
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="adjacent eval manifest"):
        core.prepare(source, native, tmp_path / "other", kind="chat")


def test_answers_and_source_file_change_are_detected(tmp_path, native):
    source = input_file(tmp_path, [example()])
    core.prepare(source, native, tmp_path / "plan", kind="chat")
    write_jsonl(source, [{**example(), "answers": ["changed gold"]}])
    with pytest.raises(ValueError, match="source/request file changed"):
        core.load_plan(tmp_path / "plan/plan.json")


def test_greedy_chat_stopping_is_explicit_and_legacy_unchanged():
    legacy = runtime.greedy_generation_config(SimpleNamespace)
    current = runtime.greedy_generation_config(SimpleNamespace, chat=True)
    assert legacy.eos_token_id == 1
    assert current.eos_token_id == [1, 106, 50]
    assert current.bos_token_id == 2 and current.do_sample is False and current.num_beams == 1
    per_case = runtime.request_generation_config(current, 64)
    assert per_case.max_new_tokens == 64 and not hasattr(current, "max_new_tokens")


def test_failure_and_overflow_keep_all_ids_without_quality_claim(tmp_path, native):
    source = input_file(tmp_path, [example("short"), example("long", "x" * 200)])
    plan = core.prepare(source, native, tmp_path / "plan", kind="chat", context_length=100)
    _, rows = core.load_plan(tmp_path / "plan/plan.json")
    def failed_loader(_):
        raise RuntimeError("template verification failed")
    summary = runtime.execute_rows(plan, rows, tmp_path / "results", failed_loader)
    assert summary["total"] == 2 and summary["failed_or_skipped"] == 2
    assert summary["statuses"] == {"model_load_failed": 1, "skipped_context_limit": 1}
    assert summary["quality_scores"] is None
    assert [r["id"] for r in core.read_jsonl(tmp_path / "results/predictions.jsonl")] == ["short", "long"]


def test_real_template_verifier_compares_all_rows_without_weights(tmp_path, monkeypatch):
    import cpt.swift_verify as swift_verify
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_text("inert unit test fixture", encoding="utf-8")
    lock = {"repository": core.MODEL_REPO, "revision": core.MODEL_REVISION,
            "files": {"tokenizer.json": {"sha256": file_hash(assets / "tokenizer.json")}}}
    write_json(assets / "model.lock.json", lock)
    plan = {"model": {"repo": core.MODEL_REPO, "revision": core.MODEL_REVISION},
            "tokenizer": {"asset_directory": str(assets), "model_lock_sha256": file_hash(assets / "model.lock.json")}}
    rows = chat.chat_rows([example(), example("overflow", "x" * 300)], ToyGemmaTokenizer(), 100)
    calls = []
    class Gemma4Processor:
        pass
    class Template:
        def set_mode(self, mode):
            assert mode == "transformers"
        def encode(self, record):
            calls.append(record)
            assert len(record["messages"]) == 1 and record["messages"][0]["role"] == "user"
            return {"input_ids": chat.encode_chat(ToyGemmaTokenizer(), record["messages"][0]["content"])}
    def processor(path, **kwargs):
        assert kwargs["download_model"] is False
        assert sorted(p.name for p in Path(path).iterdir()) == ["tokenizer.json"]
        return Gemma4Processor()
    def template(_, **kwargs):
        assert kwargs["max_length"] is None and kwargs["truncation_strategy"] == "raise"
        assert kwargs["template_type"] == "gemma4_nothinking"
        return Template()
    monkeypatch.setattr(chat, "validate_model_lock", lambda *a: None)
    monkeypatch.setattr(swift_verify, "_runtime_provenance", lambda: {"mock": True})
    monkeypatch.setattr(swift_verify, "_load_backend", lambda: (processor, template, None))
    report = chat.verify_swift_rows(plan, rows)
    assert len(calls) == 2 and report["records"] == 2
    assert report["weights_loaded"] is False and report["verified"] is True
    rows[0]["input_ids"][-1] += 1
    with pytest.raises(ValueError, match="real SWIFT chat template"):
        chat.verify_swift_rows(plan, rows)
