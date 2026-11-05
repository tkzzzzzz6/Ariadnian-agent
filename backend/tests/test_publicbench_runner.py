"""Public-benchmark protocol tests; all model calls and scores are authored mocks."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publicbench import runner


def example(eid="source-1", benchmark="longbench-v1", task="qasper", context="The instrument measurement is described in this paragraph."):
    return {"id": eid, "official_id": eid, "benchmark": benchmark, "task": task, "domain": "single_document_qa", "language": "en", "context": context, "question": "Explain the measurement.", "reference_answers": ["HIDDEN_GOLD_NOT_SENT"], "answers": ["HIDDEN_GOLD_NOT_SENT"], "prompt_template": "Read the document:\n{context}\nQuestion: {input}\nAnswer:", "official_max_new_tokens": 17, "metric_name": "qa_f1", "source_revision": "fixed-test-revision", "source_file_sha256": "fixture-hash", "raw_record": {"title": "Instrument study"}, "evaluation_only": True}


def fake_render(record, context_override=None):
    context = record["context"] if context_override is None else context_override
    return record["prompt_template"].replace("{context}", context).replace("{input}", record["question"])


def fake_score(record, prediction, status="ok"):
    score = float(status == "ok" and prediction == record["reference_answers"][0])
    return {"score": score, "score_percent": score * 100, "metric": "fixture_exact_match"}


def fake_aggregate(rows):
    def aggregate(items):
        return {"n": len(items), "coverage": sum(r["status"] == "ok" for r in items) / max(len(items), 1), "mean_score": sum(r["score"] for r in items) / max(len(items), 1), "score_percent": 100 * sum(r["score"] for r in items) / max(len(items), 1)}
    result = aggregate(rows)
    for field in ("benchmark", "task", "domain"):
        result["by_" + field] = {value: aggregate([r for r in rows if r[field] == value]) for value in {r[field] for r in rows}}
    return result


def write_sample(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    manifest = {"records": len(records), "selected_ids": [record["id"] for record in records], "selected_records_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (path.parent / "fetch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def transport(monkeypatch):
    from app import llm, tracing
    from app.config import ModelTier
    @contextmanager
    def context(model, tier):
        yield SimpleNamespace(tiers={"write": ModelTier("write", "openai/mock", "fixture-api-secret", None, 300, 9, ("openai/fallback",))})
    monkeypatch.setattr(runner, "app_context", context)
    monkeypatch.setattr(runner, "render_prompt", fake_render)
    monkeypatch.setattr(runner, "score_prediction", fake_score)
    monkeypatch.setattr(runner, "aggregate_scores", fake_aggregate)
    monkeypatch.setattr(llm, "_check_ready", lambda tier: None)
    monkeypatch.setattr(llm, "resolve_model", lambda model: model)
    monkeypatch.setattr(llm, "_kwargs", lambda tier, model: {"model": model, "num_retries": tier.retries, "timeout": tier.timeout})
    monkeypatch.setattr(llm, "_cost", lambda response: .002)
    monkeypatch.setattr(tracing, "_secrets", lambda: {"fixture-api-secret"})
    monkeypatch.setattr(tracing, "scrub", lambda text, limit=2000: str(text).replace("fixture-api-secret", "[redacted]")[:limit])
    calls, state = [], {"prediction": "HIDDEN_GOLD_NOT_SENT", "finish_reason": "stop", "error": None, "interrupt_on_call": None}
    def completion(kwargs, *args, **kw):
        calls.append(kwargs)
        if len(calls) == state["interrupt_on_call"]:
            raise KeyboardInterrupt()
        if state["error"]:
            raise RuntimeError(state["error"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=state["prediction"]), finish_reason=state["finish_reason"])], usage={"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105, "prompt_tokens_details": {"cached_tokens": 9}, "completion_tokens_details": {"reasoning_tokens": 2}}, model="fixture-response-model")
    monkeypatch.setattr(llm, "_completion", completion)
    return calls, state


def run(records, **kwargs):
    return runner.run_records(records, length_counter=len, tokenizer_metadata={"name": "fixture-character-counter-not-model-tokenizer"}, thinking="disabled", **kwargs)


def test_full_prompt_is_official_template_plain_text_without_hidden_answers(transport):
    calls, _ = transport
    record = example()
    rows, summary = run([record], context_tokens=1000)
    assert len(calls) == 1
    assert calls[0]["messages"] == [{"role": "user", "content": fake_render(record)}]
    assert "HIDDEN_GOLD_NOT_SENT" not in json.dumps(calls[0]["messages"])
    assert "response_format" not in calls[0]
    assert calls[0]["max_tokens"] == 17
    assert calls[0]["num_retries"] == 0
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert rows[0]["score"] == 1
    assert rows[0]["source_sha256"] == record["source_file_sha256"]
    assert rows[0]["protocol"] == "official_prompt_full_context_subset"
    assert not rows[0]["context_selection"]["truncated"]
    assert summary["usage"]["prompt_tokens"] == 100
    assert summary["usage"]["cached_tokens"] == 9
    assert summary["usage"]["reasoning_tokens"] == 2


def test_full_context_guard_keeps_skipped_example_in_denominator_without_calling(transport):
    calls, _ = transport
    rows, summary = run([example(context="long source " * 100)], context_tokens=10)
    assert not calls
    assert rows[0]["status"] == "skipped_context_limit"
    assert rows[0]["score"] == 0
    assert not rows[0]["context_selection"]["truncated"]
    assert summary["metrics"]["n"] == 1
    assert summary["metrics"]["coverage"] == 0
    assert summary["usage"]["prompt_tokens"] is None


@pytest.mark.parametrize("strategy", ["bm25", "structure", "truncate"])
def test_budget_variants_are_explicit_modified_protocols_and_send_no_citation_json(transport, strategy):
    calls, _ = transport
    record = example(context="# Measurement\n\nThe measurement calibration procedure is important. " * 20)
    rows, summary = run([record], context_tokens=35, strategy=strategy, chunk_chars=60)
    assert len(calls) == 1
    assert rows[0]["protocol"] == "modified_context_or_generation_budget_subset"
    assert rows[0]["context_selection"]["selected_context_reference_tokens"] <= 35
    assert rows[0]["context_selection"]["modified"]
    assert calls[0]["messages"][0]["content"].startswith("Read the document:")
    assert "response_format" not in calls[0]
    assert "HIDDEN_GOLD_NOT_SENT" not in calls[0]["messages"][0]["content"]
    assert summary["strategy"] == strategy


def test_non_json_v2_answer_uses_source_prompt_without_forcing_bare_letter(transport):
    calls, state = transport
    record = example(benchmark="longbench-v2", task="single_document_qa")
    record["prompt_template"] = "{context}\n{input}\nReply in the form The correct answer is (insert answer here)."
    record["reference_answers"] = ["The correct answer is (A)"]
    state["prediction"] = "The correct answer is (A)"
    rows, _ = run([record], context_tokens=1000)
    assert rows[0]["status"] == "ok" and rows[0]["score"] == 1
    assert "The correct answer is (insert answer here)" in calls[0]["messages"][0]["content"]
    assert len(calls[0]["messages"]) == 1


def test_output_cap_override_is_labeled_modified_and_never_increased(transport):
    calls, _ = transport
    rows, _ = run([example()], context_tokens=1000, max_output_tokens=4)
    assert calls[0]["max_tokens"] == 4
    assert rows[0]["protocol"] == "modified_context_or_generation_budget_subset"
    rows, _ = run([example()], context_tokens=1000, max_output_tokens=100)
    assert calls[-1]["max_tokens"] == 17
    assert rows[0]["protocol"] == "official_prompt_full_context_subset"


def test_empty_length_limited_answer_and_transport_error_do_not_retry_or_disappear(transport):
    calls, state = transport
    state.update(prediction="", finish_reason="length")
    rows, summary = run([example()], context_tokens=1000)
    assert len(calls) == 1
    assert rows[0]["status"] == "failed" and rows[0]["score"] == 0
    assert summary["metrics"]["n"] == 1
    state["error"] = "fixture-api-secret must never leak"
    rows, summary = run([example()], context_tokens=1000)
    assert len(calls) == 2
    assert rows[0]["status"] == "failed"
    storage = summary.redact_for_storage({"rows": rows, "summary": summary})
    assert "fixture-api-secret" not in json.dumps(storage)


def test_rescore_no_api_recomputes_scores_and_counts_missing_predictions(transport, monkeypatch):
    calls, _ = transport
    def forbidden(*args, **kwargs):
        raise AssertionError("Rescore must not enter model context")
    monkeypatch.setattr(runner, "app_context", forbidden)
    records = [example(), example("source-2")]
    predictions = [{"id": "source-1", "status": "ok", "prediction": "HIDDEN_GOLD_NOT_SENT", "score": 999}]
    rows, summary = runner.rescore_records(records, predictions)
    assert not calls
    assert rows[0]["score"] == 1
    assert rows[1]["status"] == "missing_prediction" and rows[1]["score"] == 0
    assert summary["metrics"]["mean_score"] == .5
    assert summary["transport_calls_attempted"] == 0
    with pytest.raises(ValueError, match="different benchmark example"):
        runner.rescore_records(records[:1], [{**predictions[0], "example_sha256": "incorrect"}])


def test_rescore_propagates_official_empty_answer_failure_to_coverage(transport, monkeypatch):
    from publicbench.metrics import score_prediction as official_score
    monkeypatch.setattr(runner, "score_prediction", official_score)
    rows, summary = runner.rescore_records([example()], [{"id": "source-1", "status": "ok", "prediction": ""}])
    assert rows[0]["status"] == "empty_or_missing_prediction"
    assert rows[0]["score"] == 0
    assert summary["metrics"]["coverage"] == 0
    assert not transport[0]


def test_validation_before_model_context_rejects_duplicate_or_missing_source_contract(transport, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid inputs must not access model credentials")
    monkeypatch.setattr(runner, "app_context", forbidden)
    with pytest.raises(ValueError, match="duplicate"):
        run([example(), example()], context_tokens=100)
    record = example()
    record["official_max_new_tokens"] = 0
    with pytest.raises(ValueError, match="official_max"):
        run([record], context_tokens=100)


def test_cli_sampling_manifest_mismatch_is_rejected_before_any_model_request(transport, tmp_path, monkeypatch):
    from publicbench import __main__ as cli
    path = tmp_path / "selected_records.jsonl"
    path.write_text(json.dumps(example()) + "\n", encoding="utf-8")
    (tmp_path / "fetch_manifest.json").write_text(json.dumps({"selected_records_sha256": "wrong-file-hash"}), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched inputs must be rejected before model setup")
    monkeypatch.setattr(runner, "run_records", forbidden)
    output = tmp_path / "output"
    code = cli.main(["run", "--input", str(path), "--output", str(output), "--use-configured-model", "--limit", "1", "--tokenizer", "unused-tokenizer"])
    assert code == 2
    assert not transport[0]
    assert not output.exists()


def test_checkpoint_redacts_after_scoring_and_preserves_consumed_cases_on_interrupt(transport, tmp_path, monkeypatch):
    from publicbench import __main__ as cli
    calls, state = transport
    state.update(prediction="fixture-api-secret", interrupt_on_call=2)
    records = [example(), example("source-2")]
    for record in records:
        record["reference_answers"] = ["fixture-api-secret"]
    path = tmp_path / "selected_records.jsonl"
    write_sample(path, records)
    monkeypatch.setattr(cli, "load_tokenizer", lambda path: (len, {"name": "fixture-counter"}))
    output = tmp_path / "output"
    code = cli.main(["run", "--input", str(path), "--output", str(output), "--use-configured-model", "--limit", "2", "--tokenizer", "fixture", "--thinking", "disabled"])
    assert code == 130 and len(calls) == 2
    partial = (output / "predictions.partial.jsonl").read_text(encoding="utf-8")
    assert len(partial.splitlines()) == 1
    assert "fixture-api-secret" not in partial
    record = json.loads(partial)
    assert record["score"] == 1 and record["prediction"] == "[已脱敏]"
    manifest = json.loads((output / "inprogress_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["completed"] == 1
    assert manifest["usage"]["prompt_tokens"] == 100
    assert manifest["transport_calls_attempted"] == 2
    assert manifest["active_id"] == "source-2"
    assert manifest["active_request_usage_unknown"]
    assert not (output / "summary.json").exists()


def test_cli_refuses_to_change_an_existing_run_even_when_it_has_checkpoint_state(transport, tmp_path, monkeypatch):
    from publicbench import __main__ as cli
    path = tmp_path / "selected_records.jsonl"
    write_sample(path, [example()])
    monkeypatch.setattr(cli, "load_tokenizer", lambda path: (len, {"name": "fixture-counter"}))
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "inprogress_manifest.json"
    original = '{"status":"already complete","completed":42}\n'
    marker.write_text(original, encoding="utf-8")
    code = cli.main(["run", "--input", str(path), "--output", str(output), "--use-configured-model", "--limit", "1", "--tokenizer", "fixture"])
    assert code == 2
    assert marker.read_text(encoding="utf-8") == original
    assert not transport[0]


@pytest.mark.parametrize("fault", ["missing", "wrong_count", "wrong_ids"])
def test_live_cli_requires_complete_sampling_provenance_before_model_setup(transport, tmp_path, monkeypatch, fault):
    from publicbench import __main__ as cli
    path = tmp_path / "selected_records.jsonl"
    write_sample(path, [example()])
    manifest_path = tmp_path / "fetch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if fault == "missing":
        manifest_path.unlink()
    else:
        manifest["records" if fault == "wrong_count" else "selected_ids"] = 99 if fault == "wrong_count" else ["not-the-source-ID"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified benchmark input must not access model setup")
    monkeypatch.setattr(runner, "run_records", forbidden)
    code = cli.main(["run", "--input", str(path), "--output", str(tmp_path / "out"), "--use-configured-model", "--limit", "1", "--tokenizer", "unused"])
    assert code == 2
    assert not transport[0]


@pytest.mark.parametrize("fault", ["raw_hash", "reference", "prompt_hash", "rendering"])
def test_source_backed_records_validate_raw_values_and_current_prompt_before_calls(transport, monkeypatch, fault):
    record = example()
    raw = {"_id": record["official_id"], "context": record["context"], "input": record["question"], "answers": record["reference_answers"]}
    record["raw_record"] = raw
    record["raw_record_sha256"] = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record["prompt"] = fake_render(record)
    record["prompt_sha256"] = hashlib.sha256(record["prompt"].encode()).hexdigest()
    if fault == "raw_hash":
        raw["context"] += " changed"
    elif fault == "reference":
        record["reference_answers"] = ["changed hidden gold"]
    elif fault == "prompt_hash":
        record["prompt_sha256"] = "incorrect"
    else:
        record["prompt"] = "outdated prompt from an incompatible renderer"
        record["prompt_sha256"] = hashlib.sha256(record["prompt"].encode()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("Source corruption must be found before model setup")
    monkeypatch.setattr(runner, "app_context", forbidden)
    with pytest.raises(ValueError):
        run([record])
    assert not transport[0]


def test_rescore_without_fetch_manifest_is_explicitly_unverified(transport, tmp_path):
    from publicbench import __main__ as cli
    path, predictions = tmp_path / "selected_records.jsonl", tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(example()) + "\n", encoding="utf-8")
    predictions.write_text(json.dumps({"id": "source-1", "status": "ok", "prediction": "HIDDEN_GOLD_NOT_SENT", "protocol": "official_prompt_full_context_subset"}) + "\n", encoding="utf-8")
    out = tmp_path / "rescored"
    assert cli.main(["rescore", "--input", str(path), "--predictions", str(predictions), "--output", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert not summary["official_subset_provenance_verified"]
    assert summary["provenance_status"] == "unverified_input_without_fetch_manifest"
    row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["protocol"] == "unverified_provenance_rescore"
    assert not transport[0]
