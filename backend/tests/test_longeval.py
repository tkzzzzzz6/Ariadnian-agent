"""Contract and metric tests use tiny authored fixtures, never claimed real runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from longeval.adapters import import_benchmark
from longeval.core import answer_scores, build_diagnostics, evaluate, resolve_splits, score_prediction, validate


def document(did="d1", family="f1", split="test"):
    return {"id": did, "family_id": family, "title": "Temperature study", "source_url": "https://example.org/source", "license": "CC-BY-4.0", "revision": "test-fixture", "language": "en", "domain": "science", "kind": "paper", "split": split, "blocks": [{"id": "intro", "type": "paragraph", "text": "The earlier baseline measurement used cold water.", "section_path": ["Introduction"], "page": 1}, {"id": "result", "type": "paragraph", "text": "Measurements ended. The measured temperature was 20 degrees.", "section_path": ["Results"], "page": 2}]}


def task(tid="q1"):
    return {"id": tid, "task_type": "extractive_qa", "document_ids": ["d1"], "question": "What temperature was measured?", "reference_answer": "20 degrees", "evidence": [{"document_id": "d1", "block_id": "result", "quote": "temperature was 20 degrees"}], "answerable": True, "language": "en", "annotation_status": "unreviewed", "split": "test"}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_validation_rejects_nonexistent_quote_and_task_split_leakage():
    bad = task()
    bad["evidence"][0]["quote"] = "invented gold"
    bad["split"] = "train"
    result = validate([document()], [bad])
    assert not result["valid"]
    assert any("quote is absent" in e for e in result["errors"])
    assert any("split mismatch" in e for e in result["errors"])


def test_family_and_exact_copy_and_declared_cluster_leakage():
    d1, d2 = document(), document("d2", "f1", "train")
    with pytest.raises(ValueError, match="split leakage"):
        resolve_splits([d1, d2])
    d2["family_id"] = "unrelated-family"
    with pytest.raises(ValueError, match="split leakage"):
        resolve_splits([d1, d2])  # Identical source text still cannot cross splits.
    d2["blocks"][0]["text"] = "An independently worded second version."
    d1["duplicate_cluster_id"] = d2["duplicate_cluster_id"] = "same-source-cluster"
    with pytest.raises(ValueError, match="split leakage"):
        resolve_splits([d1, d2])
    del d1["split"]
    del d2["split"]
    splits = resolve_splits([d1, d2])
    assert splits["d1"] == splits["d2"]
    assert splits == resolve_splits([d2, d1])


def test_derivative_lineage_keeps_source_documents_and_parent_families_together():
    d1, d2 = document(), document("d2", "family2", "train")
    d2["blocks"][0]["text"] = "A distinct compilation with only a subset of source text."
    d2["source_document_ids"] = ["d1"]
    with pytest.raises(ValueError, match="split leakage"):
        resolve_splits([d1, d2])
    del d2["source_document_ids"]
    d2["parent_family_ids"] = ["f1"]
    with pytest.raises(ValueError, match="split leakage"):
        resolve_splits([d1, d2])
    del d1["split"]
    del d2["split"]
    assert resolve_splits([d1, d2])["d1"] == resolve_splits([d2, d1])["d2"]


def test_nullable_assessment_source_metadata_is_valid_but_warns():
    d = document()
    d.update(source_url=None, license=None, revision=None)
    result = validate([d], [task()])
    assert result["valid"]
    assert len(result["warnings"]) == 3
    assert all("no source/training rights inferred" in warning for warning in result["warnings"])


@pytest.mark.parametrize("value", [None, 42, [], {}])
def test_malformed_contract_returns_errors_instead_of_attribute_error(value):
    d = document()
    d["license"] = value
    if value is None:
        assert validate([d], [task()])["valid"]
    else:
        assert not validate([d], [task()])["valid"]
    bad_task = task()
    bad_task.update(annotation_status=value, split=value)
    bad_task["evidence"][0]["document_id"] = value
    assert not validate([document()], [bad_task])["valid"]
    assert not validate([value], [value])["valid"]


def test_truncated_right_block_is_not_full_evidence_hit():
    d, t = document(), task()
    pred = {"id": "q1", "evidence": [{"document_id": "d1", "block_id": "result", "quote": "Measurements ended."}], "answer": "20 degrees"}
    scored = score_prediction(t, pred, {"d1": d})
    assert scored["evidence_recall"] == 0
    assert scored["evidence_complete_hit"] == 0
    assert scored["evidence_span_recall"] == 0
    assert scored["answer_em"] == 1


def test_span_precision_and_multi_evidence_complete_hit():
    d, t = document(), task()
    t["evidence"].append({"document_id": "d1", "block_id": "intro", "quote": "cold water"})
    predicted = {"id": "q1", "evidence": [{"document_id": "d1", "block_id": "result"}]}
    row = score_prediction(t, predicted, {"d1": d})
    assert row["evidence_recall"] == .5
    assert row["evidence_precision"] == 1
    assert row["evidence_complete_hit"] == 0
    assert 0 < row["evidence_span_precision"] < 1
    predicted["evidence"].append({"document_id": "d1", "block_id": "intro", "quote": "cold water"})
    predicted["evidence"].append(predicted["evidence"][0].copy())
    row = score_prediction(t, predicted, {"d1": d})
    assert row["evidence_recall"] == row["evidence_precision"] == row["evidence_complete_hit"] == 1


def test_missing_model_predictions_count_as_failure_and_no_abstention_pass():
    d = document()
    t1, t2 = task(), task("q2")
    t2.update(task_type="unanswerable", question="What unreported humidity was measured?", reference_answer="", answerable=False, evidence=[])
    preds = [{"id": "q1", "answer": "20 degrees", "predicted_answerable": True, "evidence": t1["evidence"]}]
    rows, summary = evaluate([d], [t1, t2], predictions=preds)
    overall = summary["overall"]
    assert overall["answer_em"] == .5
    assert overall["answer_coverage"] == .5
    assert overall["prediction_coverage"] == .5
    assert overall["unanswerable_rejection_accuracy"] == 0
    assert rows[1]["status"] == "missing_prediction"
    preds.append({"id": "q2", "answer": "", "abstain": True, "evidence": []})
    _, complete = evaluate([d], [t1, t2], predictions=preds)
    assert complete["overall"]["answer_em"] == 1
    assert complete["overall"]["unanswerable_rejection_accuracy"] == 1


@pytest.mark.parametrize("prediction", [
    {"answer": "invented answer", "predicted_answerable": False, "evidence": []},
    {"answer": "", "predicted_answerable": False, "abstain": False, "evidence": []},
    {"answer": "", "predicted_answerable": "unknown", "evidence": []},
    {"answer": "", "predicted_answerable": False, "evidence": [{"document_id": "d1", "block_id": "intro"}]},
])
def test_contradictory_imported_abstentions_are_failures_not_correct_rejections(prediction):
    t = task()
    t.update(task_type="unanswerable", question="What unreported humidity was measured?", reference_answer="", answerable=False, evidence=[])
    rows, summary = evaluate([document()], [t], predictions=[{"id": "q1", **prediction}])
    assert rows[0]["status"] == "invalid_prediction"
    assert summary["overall"]["unanswerable_rejection_accuracy"] == 0
    assert summary["overall"]["prediction_coverage"] == 0
    assert summary["overall"]["answer_em"] == 0


def test_repeated_gold_quote_can_match_a_later_prediction_and_offsets_disambiguate():
    d, t = document(), task()
    text = "20 degrees early. Later the measured temperature was 20 degrees."
    d["blocks"][1]["text"] = text
    t["evidence"][0]["quote"] = "20 degrees"
    quote = "Later the measured temperature was 20 degrees."
    pred = {"id": "q1", "answer": "20 degrees", "evidence": [{"document_id": "d1", "block_id": "result", "quote": quote}]}
    row = score_prediction(t, pred, {"d1": d})
    assert row["evidence_complete_hit"] == row["evidence_span_recall"] == 1
    assert row["evidence_span_precision"] == pytest.approx(len("20 degrees") / len(quote))
    assert row["ambiguous_quote_entries"] == 1
    t["evidence"][0].update(start_char=0, end_char=len("20 degrees"))
    assert score_prediction(t, pred, {"d1": d})["evidence_complete_hit"] == 0
    t["evidence"][0].update(start_char=text.rfind("20 degrees"), end_char=text.rfind("20 degrees") + len("20 degrees"))
    assert validate([d], [t])["valid"]
    assert score_prediction(t, pred, {"d1": d})["evidence_complete_hit"] == 1
    t["evidence"][0]["start_char"] += 1
    assert not validate([d], [t])["valid"]


def test_one_ambiguous_prediction_is_not_counted_as_two_distinct_occurrences():
    d, t = document(), task()
    text = "20 degrees early. Later the measured temperature was 20 degrees."
    d["blocks"][1]["text"] = text
    t["evidence"] = [{"document_id": "d1", "block_id": "result", "quote": "20 degrees", "start_char": start, "end_char": start + len("20 degrees")} for start in (0, text.rfind("20 degrees"))]
    repeated = {"document_id": "d1", "block_id": "result", "quote": "20 degrees"}
    pred = {"id": "q1", "answer": "20 degrees", "evidence": [repeated, repeated.copy()]}
    row = score_prediction(t, pred, {"d1": d})
    assert row["evidence_complete_hit"] == 0
    assert row["evidence_span_recall"] == .5


def test_failed_or_invalid_prediction_cannot_score_gold_answer():
    t, d = task(), document()
    prediction = {"id": "q1", "answer": "20 degrees", "evidence": [{"document_id": "d1", "block_id": "result", "quote": "not in source"}]}
    row = score_prediction(t, prediction, {"d1": d})
    assert row["status"] == "invalid_prediction"
    assert row["answer_em"] == row["evidence_recall"] == 0
    assert not row["answer_scored"]


@pytest.mark.parametrize("strategy", ["bm25", "structure", "truncate"])
def test_offline_strategies_never_claim_model_answer(strategy):
    rows, summary = evaluate([document()], [task()], strategy=strategy, budget_bytes=35)
    assert summary["mode"] == "offline_context_selection_only"
    assert summary["length_unit"] == "utf8_bytes"
    assert summary["overall"]["answer_em"] is None
    assert summary["overall"]["answer_coverage"] == 0
    assert rows[0]["prediction"]["answer"] is None
    assert rows[0]["prediction"]["selected_bytes"] <= 35


def test_labels_and_diagnostics_remain_distinct():
    docs = [document()]
    diagnostics = build_diagnostics(docs)
    assert {t["task_type"] for t in diagnostics} == {"evidence_lookup", "cross_section_evidence", "absent_marker"}
    assert all(t["annotation_status"] == "controlled_synthetic" for t in diagnostics)
    assert validate(docs, diagnostics)["valid"]
    actual = task()
    actual["annotation_status"] = "human_verified"
    _, summary = evaluate(docs, diagnostics + [actual])
    assert set(summary["groups"]["annotation_status"]) == {"controlled_synthetic", "human_verified"}
    negative = [r for r in build_diagnostics(docs) if not r["answerable"]][0]
    rows, _ = evaluate(docs, [negative])
    assert rows[0]["unanswerable_rejection_accuracy"] == 1


def test_diagnostics_limit_covers_multiple_domains():
    docs = []
    for index in range(12):
        doc = document(f"d{index}", f"family{index}")
        doc["domain"] = "science" if index < 10 else "history"
        doc["blocks"][0]["text"] += f" Source variation {index}."
        docs.append(doc)
    tasks = build_diagnostics(docs, max_tasks=2, seed=13)
    by_id = {d["id"]: d for d in docs}
    assert {by_id[t["document_ids"][0]]["domain"] for t in tasks} == {"science", "history"}
    assert tasks == build_diagnostics(docs, max_tasks=2, seed=13)


def test_reject_duplicate_unknown_predictions_and_invalid_reference_alternatives():
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate([document()], [task()], predictions=[{"id": "q1"}, {"id": "q1"}])
    with pytest.raises(ValueError, match="unknown prediction"):
        evaluate([document()], [task()], predictions=[{"id": "unknown"}])
    t = task()
    t["reference_answers"] = "not a list"
    assert not validate([document()], [t])["valid"]


def test_chinese_answers_and_actual_length_counter():
    assert answer_scores("测量温度为二十度。", ["测量温度为二十度"]) == {"answer_em": 1.0, "answer_f1": 1.0}
    rows, summary = evaluate([document()], [task()], length_counter=lambda _: 9000, length_unit="tokens:test-local")
    assert rows[0]["length_bucket"] == "8k-32k"
    assert summary["length_unit"] == "tokens:test-local"


def test_reference_token_budget_counts_separator_and_preserves_unicode():
    d = document()
    d["blocks"][0]["text"] = "中文文本不应出现损坏的字符。"
    rows, summary = evaluate([d], [task()], strategy="truncate", budget_tokens=5, length_counter=len, length_unit="tokens:fixture-character-tokenizer")
    assert rows[0]["prediction"]["selected_tokens"] <= 5
    assert rows[0]["prediction"]["evidence"][0]["quote"] == "中文文本不"
    assert summary["budget_tokens"] == 5
    assert summary["budget_bytes"] is None
    with pytest.raises(ValueError, match="tokenizer"):
        evaluate([d], [task()], budget_tokens=5)


def test_token_budget_stops_after_last_truncated_block_without_scanning_a_thousand_blocks():
    from longeval.core import select_context
    d = document()
    d["blocks"] = [{"id": f"block-{index}", "type": "paragraph", "text": "measurement " * 100, "section_path": [], "page": None} for index in range(1000)]
    calls = []
    def counter(text):
        calls.append(len(text))
        return len(text)
    selected = select_context(task(), {"d1": d}, "truncate", 16384, budget_tokens=800, length_counter=counter)
    assert selected["selected_tokens"] == 800
    assert len(selected["evidence"]) == 1
    assert len(calls) <= 15
    assert counter("\n\n".join(ev["quote"] for ev in selected["evidence"])) <= 800


def test_exact_token_budget_and_empty_last_prefix_stop_immediately():
    from longeval.core import select_context
    d = document()
    d["blocks"] = [{"id": f"block-{index}", "type": "paragraph", "text": "ab", "section_path": [], "page": None} for index in range(1000)]
    calls = []
    def counter(text):
        calls.append(len(text))
        return len(text)
    exact = select_context(task(), {"d1": d}, "truncate", 16384, budget_tokens=2, length_counter=counter)
    assert exact["selected_tokens"] == 2
    assert len(calls) == 1
    calls.clear()
    partial = select_context(task(), {"d1": d}, "truncate", 16384, budget_tokens=3, length_counter=counter)
    assert partial["selected_tokens"] == 2
    assert len(partial["evidence"]) == 1
    assert len(calls) <= 5


def test_live_prompt_excludes_gold_and_output_schema_rejects_hallucinated_sources():
    from longeval.core import select_context
    from longeval.live import messages_for, parse_output
    d, t = document(), task()
    t["reference_answer"] = "DO_NOT_INCLUDE_THIS_HIDDEN_GOLD"
    selected = select_context(t, {"d1": d}, "bm25", 16384)
    messages = messages_for(t, selected, {"d1": d})
    assert "DO_NOT_INCLUDE_THIS_HIDDEN_GOLD" not in json.dumps(messages)
    assert "reference_answer" not in messages[1]["content"]
    invalid = {"answer": "20 degrees", "predicted_answerable": True, "evidence": [{"document_id": "d1", "block_id": "fabricated", "quote": "temperature"}]}
    with pytest.raises(ValueError, match="unknown evidence"):
        parse_output(json.dumps(invalid), selected)
    with pytest.raises(ValueError, match="invalid"):
        parse_output(json.dumps({"answer": "", "predicted_answerable": "unknown", "evidence": []}), selected)


@pytest.mark.parametrize("thinking", ["default", "disabled", "enabled"])
def test_live_reuses_app_transport_with_strict_cap_and_usage(monkeypatch, tmp_path, thinking):
    from types import SimpleNamespace
    from app import llm
    from longeval.live import run_live
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "settings.json").write_text(json.dumps({"LLM_API_KEY": "test-secret-do-not-leak", "MODEL_WRITE": "openai/mock-long-eval"}), encoding="utf-8")
    monkeypatch.setenv("ARIA_DATA_DIR", str(configured))
    requests = []
    def completion(kwargs, *args, **kw):
        requests.append(kwargs)
        output = {"answer": "20 degrees", "predicted_answerable": True, "evidence": task()["evidence"]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(output)), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25, total_tokens=125), model="mock-provider-model")
    monkeypatch.setattr(llm, "_completion", completion)
    predictions, metadata = run_live([document()], [task()], max_output_tokens=7, timeout=12, thinking=thinking)
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == 7
    assert requests[0]["num_retries"] == 0
    if thinking == "default":
        assert "extra_body" not in requests[0]
        assert metadata["thinking_extra_body"] is None
    else:
        assert requests[0]["extra_body"] == {"thinking": {"type": thinking}}
        assert metadata["thinking_extra_body"] == requests[0]["extra_body"]
    assert metadata["thinking"] == thinking
    assert predictions[0]["parameters"]["thinking"] == thinking
    assert predictions[0]["status"] == "ok"
    assert metadata["usage"]["prompt_tokens"] == 100
    assert metadata["usage"]["completion_tokens"] == 25
    assert "test-secret-do-not-leak" not in json.dumps([predictions, metadata])
    assert not (configured / "aria.sqlite").exists()
    assert "thinking" not in (configured / "settings.json").read_text(encoding="utf-8")


def test_live_cli_redacts_all_saved_fields_after_scoring_raw_quotes(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app import llm
    from longeval.__main__ import main
    secret = "mock-secret-in-valid-model-json"
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "settings.json").write_text(json.dumps({"LLM_API_KEY": secret, "MODEL_WRITE": "openai/mock-long-eval"}), encoding="utf-8")
    monkeypatch.setenv("ARIA_DATA_DIR", str(configured))
    d, t = document(), task()
    d["blocks"][1]["text"] = f"The measured label was {secret}."
    t["reference_answer"] = secret
    t["evidence"][0]["quote"] = secret
    docs_path, tasks_path, out = tmp_path / "documents.jsonl", tmp_path / "tasks.jsonl", tmp_path / "out"
    write_jsonl(docs_path, [d])
    write_jsonl(tasks_path, [t])
    def completion(*args, **kwargs):
        output = {"answer": secret, "predicted_answerable": True, "evidence": t["evidence"]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(output)), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25, total_tokens=125), model="provider-" + secret)
    monkeypatch.setattr(llm, "_completion", completion)
    code = main(["run", "--documents", str(docs_path), "--tasks", str(tasks_path), "--output-dir", str(out), "--use-configured-model", "--limit", "1"])
    assert code == 0
    for artifact in out.iterdir():
        assert secret not in artifact.read_text(encoding="utf-8"), artifact.name
    scored = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8"))
    assert scored["answer_em"] == scored["evidence_complete_hit"] == 1
    assert "[已脱敏]" in scored["prediction"]["answer"]
    assert "[已脱敏]" in scored["prediction"]["evidence"][0]["quote"]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["storage_redaction"]["scored_before_redaction"]


def test_storage_redactor_is_pure_and_does_not_read_app_configuration(monkeypatch):
    from app import tracing
    from longeval.live import LiveMetadata
    def forbidden(*args, **kwargs):
        raise AssertionError("Storage redaction must not read or cache app settings")
    monkeypatch.setattr(tracing, "scrub", forbidden)
    monkeypatch.setattr(tracing, "get_settings", forbidden)
    metadata = LiveMetadata({"mode": "test"}, secrets={"snapshot-secret"})
    assert metadata.redact_for_storage({"nested": ["snapshot-secret", "Bearer opaque-token"]}) == {"nested": ["[已脱敏]", "Bearer [已脱敏]"]}
    assert "snapshot-secret" not in json.dumps(metadata)


def test_live_context_restores_cached_settings_overrides_and_initialized_database(tmp_path):
    original = tmp_path / "original-app"
    configured = tmp_path / "other-configuration"
    configured.mkdir()
    (configured / "settings.json").write_text(json.dumps({"LLM_API_KEY": "mock-context-secret", "MODEL_WRITE": "openai/mock"}), encoding="utf-8")
    code = r'''
import os, sys
from app import config, db, llm
from longeval.live import app_context, LiveMetadata
before = config.get_settings()
before_getter = config.get_settings
before_overrides = config._overrides
db.init_db()
before_db_flag = db._initialized
before_db_path = db.db_path()
before_misses = config.get_settings.cache_info().misses
with db.get_conn() as conn:
    conn.execute("CREATE TABLE context_regression_guard (value TEXT)")
    conn.execute("INSERT INTO context_regression_guard VALUES ('still present')")
os.environ['ARIA_DATA_DIR'] = sys.argv[1]
class DeliberateFailure(Exception): pass
for fail in (False, True):
    try:
        with app_context(None, 'write') as settings:
            assert settings is config.get_settings()
            assert settings.data_dir != before.data_dir
            assert not settings.llm_fake
            assert db.db_path() != before_db_path
            with db.get_conn() as conn:
                assert conn.execute("SELECT name FROM sqlite_master WHERE name='papers'").fetchone()
                assert not conn.execute("SELECT name FROM sqlite_master WHERE name='context_regression_guard'").fetchone()
            metadata = LiveMetadata({'mode':'test'}, secrets={'mock-context-secret'})
            if fail: raise DeliberateFailure()
    except DeliberateFailure:
        pass
    assert config.get_settings is before_getter
    assert llm.get_settings is before_getter
    assert config.get_settings() is before
    assert config.get_settings.cache_info().misses == before_misses
    assert config._overrides is before_overrides
    assert db._initialized is before_db_flag
    assert db.db_path() == before_db_path
    with db.get_conn() as conn:
        assert conn.execute('SELECT value FROM context_regression_guard').fetchone()[0] == 'still present'
    assert metadata.redact_for_storage({'answer':'mock-context-secret'}) == {'answer':'[已脱敏]'}
    assert config.get_settings() is before
print('cached settings, overrides and database restored after success and failure')
'''
    env = {**os.environ, "ARIA_DATA_DIR": str(original), "LLM_FAKE": "1", "PYTHON_DOTENV_DISABLED": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-c", code, str(configured)], cwd=BACKEND, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (configured / "aria.sqlite").exists()


def test_cli_checks_full_task_input_before_split_and_limit(tmp_path, capsys):
    from longeval.__main__ import main
    d1, d2 = document(), document("d2", "family2", "train")
    d2["blocks"][0]["text"] += " Independent second source."
    t1, t2 = task(), task("q2")
    t2.update(document_ids=["d2"], split="train")
    t2["evidence"][0]["document_id"] = "d2"
    docs_path, tasks_path, out = tmp_path / "documents.jsonl", tmp_path / "tasks.jsonl", tmp_path / "out"
    write_jsonl(docs_path, [d1, d2])
    write_jsonl(tasks_path, [t1, t2])
    code = main(["run", "--documents", str(docs_path), "--tasks", str(tasks_path), "--output-dir", str(out), "--split", "test", "--limit", "1"])
    assert code == 2
    assert "identical question appears across splits" in capsys.readouterr().err
    assert not out.exists()


def test_benchmark_adapters_preserve_units_and_do_not_invent_evidence():
    records = [{"_id": "source-7", "context": "The source context.", "question": "Which item?", "choice_A": "one", "choice_B": "two", "choice_C": "three", "choice_D": "four", "answer": "B", "length": 10000}]
    docs, tasks = import_benchmark(records, "longbench-v2", license_name="unknown", source_url="https://huggingface.co/datasets/THUDM/LongBench-v2")
    assert validate(docs, tasks)["valid"]
    assert tasks[0]["annotation_status"] == "unreviewed"
    assert tasks[0]["evidence"] == []
    assert docs[0]["metadata"]["benchmark_reported_length_unit"] == "words"
    assert docs[0]["metadata"]["official_full_run"] is False
    docs, tasks = import_benchmark([{"index": 2, "input": "Return code 42", "outputs": ["42"], "length": 8192}], "ruler", license_name="unknown", source_url="https://github.com/NVIDIA/RULER")
    assert validate(docs, tasks)["valid"]
    assert tasks[0]["annotation_status"] == "controlled_synthetic"
    assert docs[0]["metadata"]["benchmark_reported_length_unit"] == "generator_tokenizer_tokens"


def test_cli_roundtrip_no_network_no_model(tmp_path):
    docs_path, tasks_path, out = tmp_path / "documents.jsonl", tmp_path / "tasks.jsonl", tmp_path / "results"
    write_jsonl(docs_path, [document()])
    def run(*args):
        result = subprocess.run([sys.executable, "-m", "longeval", *map(str, args)], cwd=BACKEND, capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    run("build-diagnostics", "--documents", docs_path, "--output", tasks_path)
    run("validate", "--documents", docs_path, "--tasks", tasks_path, "--output-dir", out)
    run("run", "--documents", docs_path, "--tasks", tasks_path, "--output-dir", out, "--strategy", "bm25")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["answer_em"] is None
    assert len((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "controlled_synthetic" in (out / "report.md").read_text(encoding="utf-8")
