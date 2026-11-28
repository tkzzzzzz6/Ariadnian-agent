"""Score immutable pilot predictions without importing the application backend.

LongBench and RULER metric code is loaded only from exact upstream SHA-256 locks.
RULER per-record scores inherit the official function's two-decimal percentage
rounding; group scores call the official function over the complete group.
BoolQ is a custom generation control: the whole answer must be yes/no, allowing
surrounding whitespace and terminal punctuation, but no explanation or prefix.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re

from cpt.core import digest, file_hash, new_output, read_jsonl, write_json, write_jsonl
from gemma_eval.chat import PROTOCOL_VERSION, validate_input_manifest

LONG_BENCH_SHA256 = "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323"
RULER_SHA256 = "6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f"
LONG_BENCH_SOURCE = "https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench"
RULER_SOURCE = "https://github.com/NVIDIA/RULER/blob/db43f92ff634bbce1c33d1fb15858aec4bba509a/scripts/eval/synthetic/constants.py"
LONG_BENCH_PRIMARY_TASKS = ("hotpotqa", "multifieldqa_en", "narrativeqa", "gov_report")
ALLOWED_LB_METRICS = {"qa_f1_score", "qa_f1_zh_score", "rouge_score", "rouge_zh_score",
                      "classification_score", "retrieval_score", "retrieval_zh_score", "count_score", "code_sim_score"}
ALLOWED_STATUSES = {"ok", "success", "failed", "model_load_failed", "skipped_context_limit", "empty_prediction",
                    "interrupted_timeout", "not_attempted_after_timeout", "interrupted_worker_exit",
                    "not_attempted_after_worker_exit", "not_attempted", "missing_prediction", "worker_start_failed",
                    "not_attempted_after_start_failure", "interrupted_by_user", "not_attempted_after_interruption",
                    "missing_worker_prediction"}


def load_locked_module(path, expected_sha256, official_sha256, name):
    """User paths never select arbitrary Python code: the upstream digest is fixed."""
    if not path or expected_sha256 != official_sha256:
        raise ValueError(f"{name}: requires the fixed official source path and SHA-256")
    path = Path(path).resolve()
    if file_hash(path) != official_sha256:
        raise ValueError(f"{name}: official metric file hash mismatch")
    # Compile the bytes we hashed, avoiding stale or injected __pycache__ files.
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != official_sha256:
        raise ValueError("official source changed during read")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


def yes_no(text):
    match = re.fullmatch(r"\s*(yes|no)\s*[.!?。！？]*\s*", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def v2_answer(text):
    # The pinned LongBench v2 pred.py extraction, not permissive option matching.
    text = text.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", text)
    if not match:
        match = re.search(r"The correct answer is ([A-D])", text)
    return match.group(1) if match else None


def validate_records(records):
    if not records:
        raise ValueError("empty frozen evaluation suite")
    ids = set()
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("duplicate or invalid frozen input ID")
        ids.add(row["id"])
        if row.get("evaluation_only") is not True:
            raise ValueError("scoring requires evaluation-only input records")
        for key in ("task", "benchmark", "cluster_id", "score_kind"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"record requires {key}")
        if not isinstance(row.get("answers"), list) or not row["answers"] or any(not isinstance(a, str) or not a for a in row["answers"]):
            raise ValueError("references must be a nonempty list of nonempty strings")
        kind = row["score_kind"]
        if kind not in {"longbench", "longbench_v1", "longbench_v2", "ruler_all", "ruler_any", "boolq_yes_no"}:
            raise ValueError(f"unsupported score kind: {kind}")
        if kind in {"longbench", "longbench_v1"} and row.get("metric_name") not in ALLOWED_LB_METRICS:
            raise ValueError("LongBench requires a supported official metric_name")
        if kind == "boolq_yes_no" and (len(row["answers"]) != 1 or row["answers"][0] not in {"yes", "no"}):
            raise ValueError("BoolQ references must be exactly ['yes'] or ['no']")
        if kind == "longbench_v2" and (len(row["answers"]) != 1 or row["answers"][0] not in set("ABCD")):
            raise ValueError("LongBench v2 requires one uppercase option reference")


def align_predictions(records, predictions):
    """Join by fixed ID, rejecting silent duplicate overwrite or extra examples."""
    expected = {r["id"]: r for r in records}
    indexed = {}
    for prediction in predictions:
        eid = prediction.get("id")
        if eid not in expected or eid in indexed:
            raise ValueError("duplicate or unknown prediction ID")
        status = prediction.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{eid}: unrecognized prediction status")
        if not isinstance(prediction.get("prediction"), str):
            raise ValueError(f"{eid}: prediction must be a string, including failures")
        if prediction.get("example_sha256") != digest(expected[eid]):
            raise ValueError(f"{eid}: prediction source/example hash mismatch")
        if prediction.get("protocol") != PROTOCOL_VERSION:
            raise ValueError(f"{eid}: prediction was not generated with the fixed chat protocol")
        if prediction.get("benchmark", expected[eid]["benchmark"]) != expected[eid]["benchmark"] or prediction.get("task", expected[eid]["task"]) != expected[eid]["task"]:
            raise ValueError(f"{eid}: prediction benchmark/task mismatch")
        indexed[eid] = prediction
    return [indexed.get(row["id"], {"id": row["id"], "prediction": "", "status": "missing_prediction"}) for row in records]


def length_bucket(row):
    if row.get("length_bucket") is not None:
        return str(row["length_bucket"])
    for key in ("native_input_tokens", "input_tokens", "target_context_tokens", "length"):
        count = row.get(key)
        if type(count) is int and count > 0:
            return "<=4k" if count <= 4096 else "4k-8k" if count <= 8192 else "8k-16k" if count <= 16384 else ">16k"
    return "unspecified"


class Metrics:
    def __init__(self, records, *, longbench_metrics=None, longbench_metrics_sha256=None,
                 ruler_metrics=None, ruler_metrics_sha256=None):
        kinds = {row["score_kind"] for row in records}
        self.longbench = self.ruler = None
        self.provenance = {"boolq": {"protocol": "strict whole-response yes/no; whitespace and terminal punctuation allowed",
                                    "official_benchmark_score": False},
                           "longbench_v2": {"source": LONG_BENCH_SOURCE.rsplit("/LongBench", 1)[0] + "/pred.py",
                                            "source_sha256": "ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138"}}
        if kinds & {"longbench", "longbench_v1"}:
            self.longbench = load_locked_module(longbench_metrics, longbench_metrics_sha256, LONG_BENCH_SHA256, "ariadne_official_longbench_metrics")
            versions = {}
            for package, expected in {"jieba": "0.42.1", "rouge": "1.0.1", "fuzzywuzzy": "0.18.0"}.items():
                actual = importlib.metadata.version(package)
                if actual != expected:
                    raise ValueError(f"official LongBench scoring requires {package}=={expected}")
                versions[package] = actual
            self.provenance["longbench"] = {"source": LONG_BENCH_SOURCE, "metrics_sha256": LONG_BENCH_SHA256,
                                             "eval_sha256": "1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721",
                                             "versions": versions, "license": "upstream MIT"}
        if kinds & {"ruler_all", "ruler_any"}:
            self.ruler = load_locked_module(ruler_metrics, ruler_metrics_sha256, RULER_SHA256, "ariadne_official_ruler_metrics")
            self.provenance["ruler"] = {"source": RULER_SOURCE, "metrics_sha256": RULER_SHA256,
                                       "license": "upstream Apache-2.0",
                                       "per_record_rounding": "official function rounds percentage to two decimals",
                                       "all_semantics": "fraction of reference strings found, not all-or-nothing exact match"}

    def one(self, example, prediction):
        status, text = prediction["status"], prediction["prediction"]
        if status not in {"ok", "success"} or not text.strip():
            return 0.0
        kind, answers = example["score_kind"], example["answers"]
        if kind == "boolq_yes_no":
            value = float(yes_no(text) == answers[0])
        elif kind == "longbench_v2":
            value = float(v2_answer(text) == answers[0])
        elif kind in {"longbench", "longbench_v1"}:
            if example["task"] in {"trec", "triviaqa", "samsum", "lsht"}:
                text = text.lstrip("\n").split("\n")[0]
            metric = getattr(self.longbench, example["metric_name"])
            value = max(metric(text, answer, all_classes=example.get("all_classes", [])) for answer in answers)
        else:
            metric = self.ruler.string_match_all if kind == "ruler_all" else self.ruler.string_match_part
            text = re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip()
            value = metric([text], [answers]) / 100.0
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("metric returned an invalid/nonfinite score")
        return float(value)

    def official_group_percent(self, examples, predictions, scores):
        kind = examples[0]["score_kind"]
        if kind not in {"ruler_all", "ruler_any"}:
            return round(100 * sum(scores) / len(scores), 2)
        metric = self.ruler.string_match_all if kind == "ruler_all" else self.ruler.string_match_part
        texts = [re.sub(r"[\x00-\x1f]", "\n", p["prediction"].strip()).strip()
                 if p["status"] in {"ok", "success"} else "" for p in predictions]
        return metric(texts, [r["answers"] for r in examples])


def grouped_indices(records):
    groups = defaultdict(list)
    for index, row in enumerate(records):
        metric = row.get("metric_name", row["score_kind"])
        for bucket in dict.fromkeys(("all_lengths", length_bucket(row))):
            groups[(row["benchmark"], row["task"], metric, bucket)].append(index)
    return groups


def _quantile(values, p):
    values = sorted(values)
    location = (len(values) - 1) * p
    low, high = math.floor(location), math.ceil(location)
    return values[low] + (values[high] - values[low]) * (location - low)


def paired_cluster_bootstrap(examples, before, after, *, repetitions=2000, seed=17):
    if type(repetitions) is not int or not 100 <= repetitions <= 100000:
        raise ValueError("bootstrap repetitions must be in [100,100000]")
    if len(examples) != len(before) or len(before) != len(after) or not examples:
        raise ValueError("paired comparison requires all aligned input IDs")
    clusters = defaultdict(list)
    for row, first, second in zip(examples, before, after):
        clusters[row["cluster_id"]].append(second - first)
    observed = sum(b - a for a, b in zip(before, after)) / len(examples)
    result = {"delta_percentage_points": 100 * observed, "records": len(examples), "clusters": len(clusters),
              "ci95_percentage_points": None, "bootstrap_repetitions": repetitions, "seed": seed,
              "interpretation": "exploratory paired source-cluster bootstrap; not training-seed uncertainty or proof of general capability gain"}
    if len(clusters) < 2:
        result["ci_unavailable_reason"] = "fewer_than_two_source_clusters"
        return result
    values = [(sum(v), len(v)) for _, v in sorted(clusters.items())]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        selected = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(v[0] for v in selected) / sum(v[1] for v in selected))
    result["ci95_percentage_points"] = [100 * _quantile(estimates, .025), 100 * _quantile(estimates, .975)]
    return result


def task_stratified_macro(examples, before, after, *, repetitions=2000, seed=17):
    """Equal task weight, then paired resampling of whole source clusters per task.

    A cluster shared by tasks prevents independent task-stratified resampling;
    in that case the interval is unavailable, not silently based on wrong units.
    A task with one source cluster likewise cannot estimate source uncertainty.
    """
    if type(repetitions) is not int or not 100 <= repetitions <= 100000:
        raise ValueError("bootstrap repetitions must be in [100,100000]")
    if not examples or len(examples) != len(before) or len(before) != len(after):
        raise ValueError("macro comparison requires aligned records")
    tasks, cluster_tasks = defaultdict(list), defaultdict(set)
    for row, first, second in zip(examples, before, after):
        tasks[row["task"]].append((row["cluster_id"], first, second))
        cluster_tasks[row["cluster_id"]].add(row["task"])
    clusters = {}
    for task, values in sorted(tasks.items()):
        clustered = defaultdict(list)
        for cluster, first, second in values:
            clustered[cluster].append(second - first)
        clusters[task] = [(sum(v), len(v)) for _, v in sorted(clustered.items())]
    before_macro = sum(sum(v[1] for v in values) / len(values) for values in tasks.values()) / len(tasks)
    after_macro = sum(sum(v[2] for v in values) / len(values) for values in tasks.values()) / len(tasks)
    result = {"before_score_percent": 100 * before_macro, "after_score_percent": 100 * after_macro,
              "delta_percentage_points": 100 * (after_macro - before_macro), "records": len(examples),
              "tasks": sorted(tasks), "clusters": len(cluster_tasks),
              "clusters_by_task": {task: len(values) for task, values in clusters.items()},
              "ci95_percentage_points": None, "bootstrap_repetitions": repetitions, "seed": seed,
              "method": "equal task means; paired whole-source-cluster resampling stratified by task",
              "interpretation": "exploratory composite only; not proof of improvement or uncertainty over training seeds"}
    if any(len(values) < 2 for values in clusters.values()):
        result["ci_unavailable_reason"] = "fewer_than_two_source_clusters_in_at_least_one_task"
        return result
    if any(len(names) > 1 for names in cluster_tasks.values()):
        result["ci_unavailable_reason"] = "source_cluster_shared_across_task_strata"
        return result
    rng, estimates = random.Random(seed), []
    for _ in range(repetitions):
        task_estimates = []
        for values in clusters.values():
            selected = [values[rng.randrange(len(values))] for _ in values]
            task_estimates.append(sum(v[0] for v in selected) / sum(v[1] for v in selected))
        estimates.append(sum(task_estimates) / len(task_estimates))
    result["ci95_percentage_points"] = [100 * _quantile(estimates, .025), 100 * _quantile(estimates, .975)]
    return result


def summarize(records, prediction_sets, metrics, *, repetitions=2000, seed=17):
    validate_records(records)
    if not prediction_sets:
        raise ValueError("at least one prediction file is required")
    aligned, scores, details, group_reports, benchmark_reports = {}, {}, [], {}, {}
    groups = grouped_indices(records)
    for label, raw in prediction_sets.items():
        aligned[label] = align_predictions(records, raw)
        scores[label] = [metrics.one(row, pred) for row, pred in zip(records, aligned[label])]
        group_reports[label], benchmark_reports[label] = [], {}
        for index, (row, pred) in enumerate(zip(records, aligned[label])):
            details.append({"label": label, "id": row["id"], "example_sha256": digest(row), "cluster_id": row["cluster_id"],
                            "benchmark": row["benchmark"], "task": row["task"], "length_bucket": length_bucket(row),
                            "score_kind": row["score_kind"], "status": pred["status"], "score": scores[label][index]})
        for benchmark in sorted({row["benchmark"] for row in records}):
            indices = [i for i, row in enumerate(records) if row["benchmark"] == benchmark]
            benchmark_reports[label][benchmark] = {"records": len(indices),
                "statuses": dict(Counter(aligned[label][i]["status"] for i in indices)),
                "score": None, "note": "No pooled score across heterogeneous tasks/metrics."}
        for (benchmark, task, metric, bucket), indices in sorted(groups.items()):
            examples = [records[i] for i in indices]
            if len({r["score_kind"] for r in examples}) != 1:
                raise ValueError("a task/metric group mixes scoring protocols")
            predictions = [aligned[label][i] for i in indices]
            values = [scores[label][i] for i in indices]
            group_reports[label].append({"benchmark": benchmark, "task": task, "metric": metric, "length_bucket": bucket,
                "records": len(indices), "clusters": len({row["cluster_id"] for row in examples}),
                "statuses": dict(Counter(p["status"] for p in predictions)),
                "score_percent": metrics.official_group_percent(examples, predictions, values),
                "all_ids_in_denominator": True})
    comparisons = []
    for before, after in combinations(prediction_sets, 2):
        for (benchmark, task, metric, bucket), indices in sorted(groups.items()):
            comparison = paired_cluster_bootstrap([records[i] for i in indices], [scores[before][i] for i in indices],
                                                   [scores[after][i] for i in indices], repetitions=repetitions, seed=seed)
            comparisons.append({"before": before, "after": after, "benchmark": benchmark, "task": task,
                                "metric": metric, "length_bucket": bucket, **comparison})
    report = {"schema_version": "gemma-pilot-scoring-v1", "protocol": PROTOCOL_VERSION,
              "total_frozen_records": len(records), "labels": list(prediction_sets),
              "overall_score": None, "benchmarks": benchmark_reports, "groups": group_reports,
              "comparisons": comparisons, "metric_provenance": metrics.provenance,
              "limitations": ["This frozen pilot uses a custom common chat protocol; it is not an official leaderboard submission.",
                              "Missing, failed, interrupted, and overlength cases score zero in every denominator.",
                              "Single training seed and small selected subsets cannot establish robust general long-context gains.",
                              "Bootstrap intervals are exploratory, source-cluster paired, and uncorrected for multiple comparisons.",
                              "RULER per-record bootstrap values inherit official rounding; group scores use the full official metric call."]}
    primary_indices = [i for i, row in enumerate(records)
                       if row["benchmark"] == "longbench-v1" and row.get("pilot_role") == "primary"]
    report["exploratory_composites"] = []
    if primary_indices:
        primary = [records[i] for i in primary_indices]
        if {r["task"] for r in primary} != set(LONG_BENCH_PRIMARY_TASKS):
            raise ValueError("LongBench primary composite requires all four predeclared tasks")
        for row in primary:
            expected = "rouge_score" if row["task"] == "gov_report" else "qa_f1_score"
            if row.get("metric_name") != expected or row["score_kind"] not in {"longbench", "longbench_v1"}:
                raise ValueError("primary composite metric differs from the predeclared protocol")
        composite = {"name": "longbench_primary_four_task_macro", "benchmark": "longbench-v1",
                     "exploratory_composite": True, "mixed_metrics": True,
                     "definition": "equal mean of hotpotqa F1, multifieldqa_en F1, narrativeqa F1, and gov_report ROUGE-L",
                     "warning": "Mixed-metric exploratory composite, not a homogeneous accuracy or formal success criterion; inspect every task.",
                     "scores_percent": {}, "comparisons": []}
        for label, values in scores.items():
            selected = [values[i] for i in primary_indices]
            observed = task_stratified_macro(primary, selected, selected, repetitions=repetitions, seed=seed)
            composite["scores_percent"][label] = observed["before_score_percent"]
        for before, after in combinations(prediction_sets, 2):
            comparison = task_stratified_macro(primary, [scores[before][i] for i in primary_indices],
                                               [scores[after][i] for i in primary_indices], repetitions=repetitions, seed=seed)
            composite["comparisons"].append({"before": before, "after": after, **comparison})
        report["exploratory_composites"].append(composite)
    return report, details


def score_files(records_path, prediction_paths, output, *, repetitions=2000, seed=17, **metric_options):
    records_path = Path(records_path).resolve()
    records = read_jsonl(records_path)
    manifest = validate_input_manifest(records_path, records)
    validate_records(records)
    metrics = Metrics(records, **metric_options)
    inputs = {"records": {"path": str(records_path), "sha256": file_hash(records_path)}, "manifest": manifest,
              "predictions": {label: {"path": str(Path(path).resolve()), "sha256": file_hash(path)} for label, path in prediction_paths.items()}}
    predictions = {label: read_jsonl(path) for label, path in prediction_paths.items()}
    for label, path in prediction_paths.items():
        summary_path = Path(path).parent / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (summary.get("predictions_sha256") != inputs["predictions"][label]["sha256"]
                    or summary.get("selected_ids") != [row["id"] for row in records]):
                raise ValueError(f"{label}: adjacent run summary disagrees with predictions or frozen suite")
            evaluation_contract = summary.get("evaluation_contract")
            if summary.get("schema_version") == "gemma-bounded-summary-v1" and summary.get("child_summary_sha256"):
                child_path = summary_path.parent / "run/summary.json"
                if file_hash(child_path) != summary["child_summary_sha256"]:
                    raise ValueError(f"{label}: bounded child summary changed")
                child = json.loads(child_path.read_text(encoding="utf-8"))
                if child.get("plan_sha256") != summary.get("plan_sha256"):
                    raise ValueError(f"{label}: bounded child plan differs")
                evaluation_contract = child.get("evaluation_contract")
            inputs["predictions"][label]["run_summary"] = {"path": str(summary_path.resolve()), "sha256": file_hash(summary_path),
                "plan_sha256": summary.get("plan_sha256"), "evaluation_contract": evaluation_contract,
                "runtime_weights_sha256": (summary.get("runtime") or {}).get("weights_sha256")}
        else:
            inputs["predictions"][label]["run_summary"] = None
    report, details = summarize(records, predictions, metrics, repetitions=repetitions, seed=seed)
    report["inputs"] = inputs
    report["scoring_implementation_sha256"] = file_hash(__file__)
    out = new_output(output)
    # Standalone output directories remain safe to place under a versioned report:
    # only this per-record artifact is ignored, aggregates remain reviewable.
    (out / ".gitignore").write_text("record_scores.jsonl\n", encoding="utf-8")
    write_jsonl(out / "record_scores.jsonl", details)
    write_json(out / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--longbench-metrics", type=Path)
    parser.add_argument("--longbench-metrics-sha256")
    parser.add_argument("--ruler-metrics", type=Path)
    parser.add_argument("--ruler-metrics-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = vars(parser.parse_args(argv))
    try:
        paths = {}
        for entry in args.pop("predictions"):
            label, separator, path = entry.partition("=")
            if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in paths or not path:
                raise ValueError("prediction arguments must be unique LABEL=PATH values")
            paths[label] = Path(path)
        result = score_files(args.pop("records"), paths, args.pop("output"), **args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Pilot scoring: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
