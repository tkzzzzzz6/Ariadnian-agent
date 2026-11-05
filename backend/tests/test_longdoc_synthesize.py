"""Natural-QA synthesis safety/lineage tests; all model transport is mocked."""
from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from longdoc_sources import synthesize as synth


def document(did="doc-a", language="en", domain="science", kind="paper", split="test"):
    return {"id": did, "family_id": "family-" + did, "title": "Calibration study " + did,
            "language": language, "domain": domain, "kind": kind, "split": split,
            "source_url": "https://user:password@example.test/article?token=private", "license": "CC-BY-4.0", "revision": "v1",
            "blocks": [{"id": "methods", "type": "paragraph", "section_path": ["Methods"], "page": None,
                        "text": f"Under protocol {did}, samples are retained only when the instrument stays inside its calibrated operating range. Repeated measurements are collected before evaluation."},
                       {"id": "results", "type": "paragraph", "section_path": ["Results"], "page": None,
                        "text": f"The evaluation for {did} reports stable measurements in the retained samples. Reliability outside the calibrated range was not measured, so the conclusion does not extend there."}]}


def candidate(blocks, anchor="doc-a"):
    chosen = [b for b in blocks if b["document_id"] == anchor][:2]
    return {"question": f"For {anchor}, how do the retention rule and the evaluation jointly constrain when the reliability conclusion can be used?",
            "reference_answer": "The rule limits the conclusion to retained observations within the calibration conditions; it cannot establish reliability for excluded observations outside those conditions.",
            "reasoning_type": "条件整合",
            "evidence": [{"document_id": b["document_id"], "block_id": b["block_id"], "quote": b["text"][:110]} for b in chosen]}


@pytest.fixture
def fake_transport(monkeypatch):
    from app import llm, tracing
    from app.config import ModelTier

    @contextmanager
    def context(model, tier):
        yield SimpleNamespace(tiers={"write": ModelTier("write", "openai/mock", "mock-secret", None, 200, 7, ("openai/fallback",))})
    monkeypatch.setattr(synth, "app_context", context)
    monkeypatch.setattr(llm, "_check_ready", lambda selected: None)
    monkeypatch.setattr(llm, "resolve_model", lambda model: model)
    monkeypatch.setattr(llm, "_cost", lambda response: 0.004)
    monkeypatch.setattr(tracing, "scrub", lambda value, limit=2000: str(value).replace("mock-secret", "[redacted]")[:limit])
    calls = []
    state = {"finish_reason": "stop", "error": None, "extra": None}
    def completion(kwargs, tier, purpose, operation, attempt):
        calls.append(kwargs)
        assert kwargs["num_retries"] == 0 and kwargs["max_tokens"] == 4096 and kwargs["timeout"] == 60
        assert "fallbacks" not in kwargs and attempt == 1
        if state["error"]:
            raise RuntimeError(state["error"])
        payload = json.loads(kwargs["messages"][-1]["content"])
        task = candidate(payload["source_blocks"], payload["anchor_document_id"])
        tasks = [task]
        if state["extra"]:
            tasks.append(state["extra"](task))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"tasks": tasks})), finish_reason=state["finish_reason"])],
                               usage={"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "prompt_tokens_details": {"cached_tokens": 50}}, model="mock-response")
    monkeypatch.setattr(llm, "_completion", completion)
    return calls, state


def test_default_dry_run_never_reads_model_configuration(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model opt-in was supplied")
    monkeypatch.setattr(synth, "app_context", forbidden)
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(document()) + "\n", encoding="utf-8")
    output = tmp_path / "dry"
    assert synth.main(["--documents", str(path), "--output", str(output), "--limit", "6", "--timeout60"]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "dry_run" and manifest["transport_calls_attempted"] == 0
    assert manifest["usage"]["cost_usd"] is None
    assert (output / "qa_candidates.jsonl").read_text(encoding="utf-8") == ""
    assert "private" not in (output / "generation_plan.jsonl").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        synth.main(["--documents", str(path), "--output", str(output)])


def test_six_anchors_cover_languages_and_multiple_domains_before_large_english_groups():
    docs = [document(f"en-{domain}-{i}", domain=domain) for domain in ("agriculture", "learning", "software") for i in range(15)]
    docs += [document(f"zh-{i}", language="zh", domain="textbook", kind="book") for i in range(10)]
    chosen = synth.balanced(docs, 6, 17)
    assert [d["language"] for d in chosen] == ["en", "zh", "en", "zh", "en", "zh"]
    assert len({d["domain"] for d in chosen}) == 4
    assert [d["id"] for d in chosen] == [d["id"] for d in synth.balanced(list(reversed(docs)), 6, 17)]


def test_context_budget_and_partner_never_cross_splits():
    docs = [document(), document("same-split"), document("different-split", split="train")]
    for doc in docs:
        for block in doc["blocks"]:
            block["text"] *= 40
    blocks = synth.select_blocks(docs[0], docs, max_blocks=12, max_chars=9000, max_tokens=1800, length_counter=lambda text: len(text) // 4)
    assert len(blocks) <= 12 and "different-split" not in {b["document_id"] for b in blocks}
    assert len({b["document_id"] for b in blocks}) == 2
    messages = synth.messages_for(docs[0], blocks, 2)
    assert sum(len(m["content"]) for m in messages) <= 9000
    assert sum(len(m["content"]) // 4 for m in messages) <= 1800
    assert all(b["text"] in next(d for d in docs if d["id"] == b["document_id"])["blocks"][0 if b["block_id"] == "methods" else 1]["text"] for b in blocks)


def test_strict_evidence_checks_reject_fabrication_and_lookup_shortcuts():
    doc = document()
    blocks = synth.select_blocks(doc, [doc])
    good = candidate(blocks)
    assert synth.validate_candidate(good, blocks)["reasoning_type"] == "条件整合"
    bad = json.loads(json.dumps(good))
    bad["evidence"][1]["quote"] = "Fabricated result absent from the sources"
    with pytest.raises(ValueError, match="exact visible"):
        synth.validate_candidate(bad, blocks)
    bad = {**good, "question": "Please explain how these conditions interact: " + good["evidence"][0]["quote"][:70]}
    with pytest.raises(ValueError, match="lookup shortcut"):
        synth.validate_candidate(bad, blocks)
    bad = {**good, "annotation_status": "human_verified"}
    with pytest.raises(ValueError, match="exactly"):
        synth.validate_candidate(bad, blocks)
    same_section = [{**b, "section_path": ["Same section"]} for b in blocks]
    with pytest.raises(ValueError, match="different named sections"):
        synth.validate_candidate(good, same_section)


def test_one_bounded_transport_keeps_usage_lineage_and_unreviewed_labels(fake_transport):
    calls, state = fake_transport
    tasks, records, plans, metadata = synth.generate([document()], use_configured_model=True, limit_documents=1)
    assert len(calls) == len(records) == len(tasks) == 1
    task = tasks[0]
    assert task["annotation_status"] == "ai_generated" and task["review_status"] == "unreviewed"
    assert task["semantic_reasoning_validated"] is False and task["training_eligible"] is False
    assert task["source_lineage"][0]["revision"] == "v1" and task["split"] == "test"
    assert len(task["generation"]["messages_sha256"]) == 64
    assert task["generation"]["usage_scope"] == "shared_generation_call_not_per_task"
    assert metadata["usage"]["total_tokens"] == 1200 and metadata["usage"]["cost_usd"] == 0.004
    assert metadata["coverage"]["language"] == {"en": 1} and "accuracy" not in metadata
    assert records[0]["status"] == "partial"  # only one of two requested candidates was returned
    assert "mock-secret" not in json.dumps([tasks, records, plans, metadata])
    assert synth.validate([document()], tasks)["valid"]


@pytest.mark.parametrize("thinking", ["default", "disabled", "enabled"])
def test_thinking_override_is_explicit_per_call_and_recorded(fake_transport, thinking):
    calls, state = fake_transport
    tasks, records, plans, metadata = synth.generate([document()], use_configured_model=True,
                                                    limit_documents=1, thinking=thinking)
    assert len(calls) == 1
    if thinking == "default":
        assert "extra_body" not in calls[0]
    else:
        assert calls[0]["extra_body"] == {"thinking": {"type": thinking}}
    assert metadata["requested_thinking_mode"] == thinking
    assert records[0]["parameters"]["requested_thinking_mode"] == thinking
    assert calls[0]["num_retries"] == 0 and calls[0]["max_tokens"] == 4096


@pytest.mark.parametrize("failure", ["length", "transport", "invalid_candidate"])
def test_failed_or_limited_outputs_are_retained_without_extra_calls(fake_transport, failure):
    calls, state = fake_transport
    if failure == "length":
        state["finish_reason"] = "length"
    elif failure == "transport":
        state["error"] = "Authorization=Bearer mock-secret"
    else:
        state["extra"] = lambda task: {**task, "question": task["question"] + " Additional perspective?", "evidence": [{"document_id": "nonexistent", "block_id": "none", "quote": "Unsupported model-created evidence text."}, task["evidence"][1]]}
    tasks, records, plans, metadata = synth.generate([document()], use_configured_model=True, limit_documents=1)
    assert len(calls) == 1 and len(records) == 1 and metadata["transport_calls_attempted"] == 1
    assert "mock-secret" not in json.dumps(records)
    if failure == "invalid_candidate":
        assert len(tasks) == 1 and len(records[0]["candidate_errors"]) == 1
        assert metadata["usage"]["total_tokens"] == 1200
    else:
        assert tasks == [] and records[0]["status"] == "failed"
        assert records[0]["error_type"]
        assert records[0]["usage"]["total_tokens"] == (None if failure == "transport" else 1200)
