"""Pinned official schemas and metric semantics; authored fixtures, no model/API calls."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from publicbench import data, metrics  # noqa: E402


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def cached(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    raw = cache / "raw"
    raw.mkdir(parents=True)
    template = "Source: {context}\nQuestion: {input}\nAnswer:"
    asset_values = {"LongBench/config/dataset2prompt.json": json.dumps({task: template for task in data.DEFAULT_V1_TASKS}),
                    "LongBench/config/dataset2maxlen.json": json.dumps({task: 128 for task in data.DEFAULT_V1_TASKS}),
                    "prompts/0shot.txt": "$DOC$\n$Q$\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\nThe correct answer is (insert answer here)",
                    "LICENSE": "MIT fixture notice; authored test data only"}
    assets = {}
    for relative, value in asset_values.items():
        path = cache / "official" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        assets[relative] = sha(path.read_bytes())
    monkeypatch.setattr(data, "OFFICIAL_ASSETS", assets)
    with ZipFile(raw / "data.zip", "w") as archive:
        archive.writestr("../must_not_extract.py", "raise RuntimeError('remote source must never execute')")
        archive.writestr("dataset_script.py", "raise RuntimeError('dataset source must never execute')")
        for task in data.DEFAULT_V1_TASKS:
            rows = [{"input": f"Original question {index}", "context": f"Original unchanged context {task} {index}",
                     "answers": ["ANSWER_ONLY_CANARY"], "length": 9999, "dataset": task,
                     "language": "zh" if task.endswith("_zh") else "en", "all_classes": ["Food", "Animal"] if task == "trec" else None,
                     "_id": f"official-{task}-{index}"} for index in range(5)]
            archive.writestr(f"data/{task}.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    v2 = []
    for domain in ("Single-Document QA", "Multi-Document QA", "Long In-context Learning", "Long-dialogue History Understanding", "Code Repo Understanding", "Long Structured Data Understanding"):
        for index in range(4):
            v2.append({"_id": f"official-v2-{domain}-{index}", "domain": domain, "sub_domain": "fixture",
                       "difficulty": "hard", "length": "long" if index == 3 else "short",
                       "context": "X" * 1100000 if index == 3 else f"Original context $Q$ for {domain} {index}",
                       "question": "The original question?", "choice_A": "First", "choice_B": "Second",
                       "choice_C": "Third", "choice_D": "Fourth", "answer": "C"})
    (raw / "data.json").write_text(json.dumps(v2), encoding="utf-8")
    specs = copy.deepcopy(data.SNAPSHOTS)
    for benchmark, spec in specs.items():
        path = raw / spec["filename"]
        spec["sha256"] = sha(path.read_bytes())
        (raw / f"{benchmark}.README.md").write_text("Fixture card only", encoding="utf-8")
        path.with_name(path.name + ".source.json").write_text(json.dumps({"repo": spec["repository"], "revision": spec["revision"],
                                                                         "filename": spec["filename"], "sha256": spec["sha256"]}), encoding="utf-8")
    monkeypatch.setattr(data, "SNAPSHOTS", specs)
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("offline fixture must not access network")))
    return cache


def test_default_pins_are_real_immutable_artifact_hashes():
    assert data.SNAPSHOTS["longbench-v1"]["revision"] == "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
    assert data.SNAPSHOTS["longbench-v1"]["sha256"] == "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
    assert data.SNAPSHOTS["longbench-v2"]["sha256"] == "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"


def test_v1_original_schema_fixed_sampling_no_scripts_and_no_gold_in_prompt(cached):
    before = sha((cached / "raw/data.zip").read_bytes())
    first, manifest = data.fetch_benchmark("v1", cached, n_per_task=2, seed=13, allow_download=False)
    again, same_manifest = data.fetch_benchmark("longbench", cached, n_per_task=2, seed=13, allow_download=False)
    other, _ = data.fetch_benchmark("v1", cached, n_per_task=2, seed=29, allow_download=False)
    assert first == again and manifest == same_manifest
    assert len(first) == 12 and {r["task"] for r in first} == set(data.DEFAULT_V1_TASKS)
    assert [r["id"] for r in first] != [r["id"] for r in other]
    for row in first:
        assert row["official_id"] == row["raw_record"]["_id"]
        assert row["context"] == row["raw_record"]["context"]
        assert row["answers"] == row["raw_record"]["answers"]
        assert "ANSWER_ONLY_CANARY" not in row["prompt"]
        assert row["evaluation_only"] and row["split_role"] == "evaluation_only_never_training"
    assert not manifest["remote_dataset_scripts_executed"] and not manifest["source_python_files_executed"]
    assert not (cached.parent / "must_not_extract.py").exists()
    assert not (cached / "dataset_script.py").exists()
    assert sha((cached / "raw/data.zip").read_bytes()) == before


def test_v2_streams_long_records_and_stratifies_before_inference(cached):
    rows, manifest = data.fetch_benchmark("v2", cached, n_per_task=2, seed=13,
                                         max_context_tokens=200, length_counter=len,
                                         tokenizer_metadata={"unit": "test_characters_not_real_model_tokens"}, allow_download=False)
    assert len(rows) == 12 and len(manifest["source_counts_before_filter"]) == 6
    assert all(value == 4 for value in manifest["source_counts_before_filter"].values())
    assert all(value == 3 for value in manifest["eligible_counts"].values())
    assert all(value == 1 for value in manifest["length_rejected_counts"].values())
    assert manifest["selection_frozen_before_inference"] and not manifest["resampling_after_failure"]
    for row in rows:
        assert row["reference_context_tokens"] <= 200
        assert row["context"] == row["raw_record"]["context"]
        assert "$Q$" in row["context"] and "$Q$" not in row["prompt"]
        assert row["prompt_rendering"]["potential_secondary_context_substitutions"] == 1
        assert row["prompt_rendering"]["warning"]
        assert row["reference_answers"] == ["C"]
        assert row["official_source_split"] == "train" and row["evaluation_only"]
    assert manifest["placeholder_literal_audit"]["source_context_counts_in_selected_examples"]["$Q$"] == 12


def test_cached_hash_and_sidecar_mismatch_fail_closed(cached):
    source = cached / "raw/data.zip"
    sidecar = source.with_name(source.name + ".source.json")
    changed = json.loads(sidecar.read_text())
    changed["revision"] = "0" * 40
    sidecar.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="sidecar"):
        data.fetch_benchmark("v1", cached, allow_download=False)
    sidecar.unlink()
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.fetch_benchmark("v1", cached, allow_download=False)


def test_unknown_revision_and_unmeasured_token_filter_rejected(cached):
    with pytest.raises(ValueError, match="pinned"):
        data.fetch_benchmark("v1", cached, revision="a" * 40, allow_download=False)
    with pytest.raises(ValueError, match="tokenizer"):
        data.fetch_benchmark("v1", cached, max_context_tokens=8192, allow_download=False)


@pytest.mark.parametrize("contents", ['[{"a":1},]', '[{"a":1}] extra', '[1]', '{"a":1}', '[{"a":'])
def test_streaming_json_rejects_non_arrays_invalid_records_and_trailing_data(tmp_path, contents):
    path = tmp_path / "array.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        list(data.iter_json_array(path))


def example(task="qasper", answers=None, benchmark="longbench-v1"):
    return {"id": f"fixture-{task}", "benchmark": benchmark, "task": task, "domain": "fixture",
            "language": "en", "reference_answers": answers or ["correct answer"], "all_classes": ["Food", "Animal", "Number", "Number of something"]}


def test_official_english_f1_articles_punctuation_and_empty_semantics():
    assert metrics.qa_f1_score("The, correct answer!", "a correct answer") == 1
    assert metrics.qa_f1_score("", "") == 0
    assert metrics.qa_f1_score("correct", "correct answer") == pytest.approx(2 / 3)
    assert metrics.score_prediction(example(answers=["wrong", "The correct answer"]), "correct answer")["score"] == 1


def test_official_chinese_uses_jieba_words_not_generic_character_f1():
    assert metrics.qa_f1_zh_score("中华人民共和国。", "中华人民共和国") == 1
    assert metrics.qa_f1_zh_score("中华人民共和国", "中华") == 0


def test_official_classification_first_line_and_multi_class_penalty():
    row = example("trec", ["Food"])
    assert metrics.score_prediction(row, "Food Animal")["score"] == 0.5
    assert metrics.score_prediction(row, "\nFood\nAnimal")["score"] == 1
    assert metrics.classification_score("Number of something", "Number of something", all_classes=row["all_classes"]) == 1


def test_official_retrieval_penalizes_every_extra_number():
    row = example("passage_retrieval_en", ["Paragraph 7"])
    assert metrics.score_prediction(row, "Paragraph 7 or 12")["score"] == 0.5
    assert metrics.score_prediction(row, "Paragraph 17")["score"] == 0
    assert metrics.retrieval_zh_score("段落7或段落12", "段落7") == 0.5


def test_official_rouge_l_is_order_sensitive_not_token_f1():
    assert metrics.rouge_score("a b c", "a b c") == pytest.approx(1)
    assert metrics.rouge_score("a c b", "a b c") == pytest.approx(2 / 3)


def test_missing_rouge_raises_instead_of_substituting_another_metric(monkeypatch):
    monkeypatch.setitem(sys.modules, "rouge", None)
    with pytest.raises(RuntimeError, match="no surrogate"):
        metrics.score_prediction(example("gov_report"), "summary")


def test_v2_accuracy_preserves_strict_official_answer_parser():
    row = example("longbench-v2", ["B"], benchmark="longbench-v2")
    assert metrics.score_prediction(row, "The correct answer is (B)")["score"] == 1
    assert metrics.score_prediction(row, "**The correct answer is B**")["score"] == 1
    assert metrics.score_prediction(row, "B")["score"] == 0
    assert metrics.score_prediction(row, "the correct answer is (B)")["score"] == 0
    assert metrics.score_prediction(row, "The correct answer is (B)", status="timeout")["score"] == 0


def test_failure_denominator_and_cross_benchmark_means_are_explicit():
    one = metrics.score_prediction(example(), "correct answer")
    two = metrics.score_prediction({**example(), "id": "failed"}, None, status="skipped_context_limit")
    result = metrics.aggregate_scores([one, two])
    assert result["mean_score"] == 0.5 and result["coverage"] == 0.5
    v2 = metrics.score_prediction(example("longbench-v2", ["B"], benchmark="longbench-v2"), "The correct answer is (B)")
    mixed = metrics.aggregate_scores([one, two, v2])
    assert mixed["mean_score"] is None
    assert mixed["by_benchmark"]["longbench-v1"]["macro_task_score"] == 0.5
    assert mixed["by_benchmark"]["longbench-v2"]["mean_score"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        metrics.aggregate_scores([one, one])
