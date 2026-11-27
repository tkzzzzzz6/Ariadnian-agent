"""Scorer contracts use toy predictions, never reported as measured model quality."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpt.core import digest, file_hash, write_json, write_jsonl
from gemma_eval.chat import PROTOCOL_VERSION
from pilot_eval import scoring


def example(eid="one", cluster="source-1", answer="yes", **kwargs):
    return {"id": eid, "benchmark": "boolq", "task": "boolq", "cluster_id": cluster,
            "evaluation_only": True, "score_kind": "boolq_yes_no", "answers": [answer],
            "prompt": "Answer yes or no. Toy test only.", "native_input_tokens": 500, **kwargs}


def prediction(row, text="yes", status="ok", **kwargs):
    return {"id": row["id"], "prediction": text, "status": status, "example_sha256": digest(row),
            "protocol": PROTOCOL_VERSION, **kwargs}


@pytest.mark.parametrize("text,expected", [("yes", "yes"), (" NO!\n", "no"), ("Yes。", "yes"),
    ("yes, because it is true", None), ("answer: yes", None), ("yesterday", None), ("true", None),
    ("yes\nno", None), ("no. Additional explanation", None), ("", None)])
def test_boolq_whole_response_control(text, expected):
    assert scoring.yes_no(text) == expected


def test_failures_empty_and_missing_are_zero_in_every_denominator():
    rows = [example(), example("two", "source-2"), example("three", "source-3")]
    metrics = scoring.Metrics(rows)
    predictions = [prediction(rows[0]), prediction(rows[1], "yes", "failed")]
    report, details = scoring.summarize(rows, {"base": predictions}, metrics, repetitions=100)
    overall = next(group for group in report["groups"]["base"] if group["length_bucket"] == "all_lengths")
    assert overall["records"] == 3 and overall["score_percent"] == 33.33
    assert [item["score"] for item in details] == [1.0, 0.0, 0.0]
    assert overall["statuses"] == {"ok": 1, "failed": 1, "missing_prediction": 1}
    assert report["overall_score"] is None and report["benchmarks"]["base"]["boolq"]["score"] is None


def test_duplicate_unknown_and_unbound_predictions_rejected():
    row = example()
    good = prediction(row)
    for predictions in ([good, good], [prediction(example("unknown"))], [{**good, "example_sha256": "0" * 64}],
                        [{**good, "protocol": "legacy_base_completion"}], [{**good, "status": "invented_status"}]):
        with pytest.raises(ValueError):
            scoring.align_predictions([row], predictions)
    with pytest.raises(ValueError, match="duplicate"):
        scoring.validate_records([row, row])


def test_id_join_is_order_independent_and_reference_edits_rejected():
    rows = [example(), example("two", "source-2", answer="no")]
    reversed_predictions = [prediction(rows[1], "no"), prediction(rows[0], "yes")]
    assert [p["id"] for p in scoring.align_predictions(rows, reversed_predictions)] == ["one", "two"]
    modified = deepcopy(rows)
    modified[0]["answers"] = ["no"]
    with pytest.raises(ValueError, match="source/example hash"):
        scoring.align_predictions(modified, reversed_predictions)


def test_bounded_failure_rows_are_supported_and_not_counted_as_hits():
    row = example()
    for status in ("worker_start_failed", "not_attempted_after_start_failure", "interrupted_by_user",
                   "not_attempted_after_interruption", "missing_worker_prediction", "not_attempted_after_worker_exit",
                   "interrupted_timeout", "not_attempted_after_timeout", "skipped_context_limit"):
        aligned = scoring.align_predictions([row], [prediction(row, "yes", status)])
        assert scoring.Metrics([row]).one(row, aligned[0]) == 0


def test_longbench_first_line_and_best_reference_use_official_function():
    calls = []
    def metric(text, answer, **kwargs):
        calls.append((text, answer, kwargs))
        return float(text == answer)
    scorer = scoring.Metrics([])
    scorer.longbench = SimpleNamespace(classification_score=metric)
    row = example(task="trec", benchmark="longbench-v1", score_kind="longbench", answers=["first", "yes"],
                  metric_name="classification_score", all_classes=["first", "yes"])
    assert scorer.one(row, prediction(row, "\nyes\nmisleading explanation")) == 1
    assert [x[:2] for x in calls] == [("yes", "first"), ("yes", "yes")]
    assert calls[0][2] == {"all_classes": ["first", "yes"]}


def test_longbench_v2_requires_official_phrase():
    assert scoring.v2_answer("B") is None
    assert scoring.v2_answer("The correct answer is (B).") == "B"
    assert scoring.v2_answer("The correct answer is **C**") == "C"
    assert scoring.v2_answer("the correct answer is (B)") is None


def test_cluster_bootstrap_is_paired_and_preserves_cluster_multiplicity():
    rows = [example("one", "shared"), example("two", "shared"), example("three", "other")]
    before, after = [0., 0., 1.], [1., 1., 0.]
    result = scoring.paired_cluster_bootstrap(rows, before, after, repetitions=500, seed=19)
    assert result["clusters"] == 2 and result["records"] == 3
    assert result["delta_percentage_points"] == pytest.approx(100 / 3)
    assert result["ci95_percentage_points"] == [-100., 100.]
    assert result == scoring.paired_cluster_bootstrap(rows, before, after, repetitions=500, seed=19)
    single = scoring.paired_cluster_bootstrap(rows[:2], before[:2], after[:2], repetitions=100)
    assert single["ci95_percentage_points"] is None
    assert single["ci_unavailable_reason"] == "fewer_than_two_source_clusters"


def test_comparisons_remain_task_and_length_specific():
    rows = [example(), example("two", "source-2", native_input_tokens=6000),
            example("three", "source-3", task="other_control")]
    base = [prediction(r, "no") for r in rows]
    tuned = [prediction(r, "yes") for r in rows]
    report, _ = scoring.summarize(rows, {"base": base, "short": tuned, "long": tuned}, scoring.Metrics(rows), repetitions=100)
    assert {(c["before"], c["after"]) for c in report["comparisons"]} == {("base", "short"), ("base", "long"), ("short", "long")}
    assert {c["task"] for c in report["comparisons"]} == {"boolq", "other_control"}
    assert {c["length_bucket"] for c in report["comparisons"]} == {"all_lengths", "<=4k", "4k-8k"}
    assert all("ci95_percentage_points" in c for c in report["comparisons"])


def test_task_macro_equal_weights_and_task_stratified_cluster_units():
    rows = [example("a1", "a1", task="a"), example("a2", "a2", task="a"),
            *[example(f"b{i}", f"b{i}", task="b") for i in range(6)]]
    result = scoring.task_stratified_macro(rows, [0.] * 8, [1., 1., *([0.] * 6)], repetitions=100)
    assert result["after_score_percent"] == 50.  # equal task weight; pooled micro would be 25
    assert result["ci95_percentage_points"] == [50., 50.]
    assert result["clusters_by_task"] == {"a": 2, "b": 6}
    single_source = [{**row, "cluster_id": "one-article"} if row["task"] == "b" else row for row in rows]
    unavailable = scoring.task_stratified_macro(single_source, [0.] * 8, [1.] * 8, repetitions=100)
    assert unavailable["ci95_percentage_points"] is None
    assert unavailable["ci_unavailable_reason"] == "fewer_than_two_source_clusters_in_at_least_one_task"
    shared = deepcopy(rows)
    shared[2]["cluster_id"] = "a1"
    assert scoring.task_stratified_macro(shared, [0.] * 8, [1.] * 8, repetitions=100)["ci_unavailable_reason"] == "source_cluster_shared_across_task_strata"


def test_predeclared_primary_composite_is_labelled_mixed_and_keeps_tasks():
    rows = []
    for task in scoring.LONG_BENCH_PRIMARY_TASKS:
        for i in range(2):
            rows.append(example(f"{task}{i}", f"{task}{i}", task=task, benchmark="longbench-v1", pilot_role="primary",
                        metric_name="rouge_score" if task == "gov_report" else "qa_f1_score", score_kind="longbench_v1"))
    metrics = scoring.Metrics([])
    metrics.longbench = SimpleNamespace(qa_f1_score=lambda p, a, **kw: float(p), rouge_score=lambda p, a, **kw: float(p))
    report, _ = scoring.summarize(rows, {"base": [prediction(r, "0") for r in rows],
                                       "long": [prediction(r, "1") for r in rows]}, metrics, repetitions=100)
    composite = report["exploratory_composites"][0]
    assert composite["mixed_metrics"] is True and composite["exploratory_composite"] is True
    assert composite["scores_percent"] == {"base": 0., "long": 100.}
    assert composite["comparisons"][0]["ci95_percentage_points"] == [100., 100.]
    assert len({g["task"] for g in report["groups"]["base"]}) == 4
    assert report["overall_score"] is None


def test_metric_module_tamper_fails_before_code_execution(tmp_path):
    path = tmp_path / "metrics.py"
    path.write_text("raise AssertionError('must never execute')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official metric file hash mismatch"):
        scoring.load_locked_module(path, scoring.LONG_BENCH_SHA256, scoring.LONG_BENCH_SHA256, "test_metric")
    with pytest.raises(ValueError, match="fixed official"):
        scoring.load_locked_module(path, file_hash(path), scoring.LONG_BENCH_SHA256, "test_metric")


def test_ruler_uses_the_real_locked_upstream_function_if_locally_available():
    path = Path(__file__).parents[1] / ".cache/pilot_eval_20261108_v1/RULER-db43f92ff634bbce1c33d1fb15858aec4bba509a/scripts/eval/synthetic/constants.py"
    if not path.is_file():
        pytest.skip("ignored official source is not required in a fresh clone")
    row = example(benchmark="ruler", task="multivalue", score_kind="ruler_all", answers=["key1", "key2", "key3"])
    metrics = scoring.Metrics([row], ruler_metrics=path, ruler_metrics_sha256=scoring.RULER_SHA256)
    assert metrics.one(row, prediction(row, "KEY1")) == .3333  # fractional recall, not all-or-nothing
    assert metrics.official_group_percent([row], [prediction(row, "KEY1")], [.3333]) == 33.33
    any_row = {**row, "score_kind": "ruler_any"}
    assert metrics.one(any_row, prediction(any_row, "KEY1")) == 1.


def test_score_file_outputs_ignore_record_level_artifacts_and_bind_hashes(tmp_path):
    rows = [example()]
    source = tmp_path / "records.jsonl"
    preds = tmp_path / "predictions.jsonl"
    write_jsonl(source, rows)
    write_jsonl(preds, [prediction(rows[0])])
    write_json(tmp_path / "eval_manifest.json", {"schema_version": "gemma-eval-input-v1", "records": 1,
        "records_sha256": file_hash(source), "selected_ids": ["one"], "source_manifests": [{"source": "toy"}]})
    report = scoring.score_files(source, {"base": preds}, tmp_path / "output", repetitions=100)
    assert report["inputs"]["predictions"]["base"]["sha256"] == file_hash(preds)
    assert (tmp_path / "output/.gitignore").read_text() == "record_scores.jsonl\n"
    assert json.loads((tmp_path / "output/summary.json").read_text())["overall_score"] is None
    # Real run summaries bind the file and checkpoint plan; mismatched adjacent
    # summaries cannot quietly attribute predictions to a different model run.
    write_json(tmp_path / "summary.json", {"predictions_sha256": "0" * 64, "selected_ids": ["one"]})
    with pytest.raises(ValueError, match="adjacent run summary"):
        scoring.score_files(source, {"base": preds}, tmp_path / "other", repetitions=100)
    write_json(tmp_path / "summary.json", {"predictions_sha256": file_hash(preds), "selected_ids": ["one"],
        "plan_sha256": "1" * 64, "evaluation_contract": {"adapter": None}})
    bound = scoring.score_files(source, {"base": preds}, tmp_path / "bound", repetitions=100)
    assert bound["inputs"]["predictions"]["base"]["run_summary"]["evaluation_contract"] == {"adapter": None}
