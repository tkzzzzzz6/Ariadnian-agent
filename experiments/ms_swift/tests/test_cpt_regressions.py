"""Regression cases from independent admission/split/launch review; no training."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cpt import core, launch, schema  # noqa: E402
from cpt.assets import NativeTokenizer  # noqa: E402


def document(did, split, text, **extra):
    return {"id": did, "family_id": did, "title": did, "source_url": "https://example.test/" + did,
            "license": "CC0-1.0", "revision": "authored-fixture", "language": "en", "domain": "test",
            "kind": "report", "split": split, "blocks": [{"id": "p", "type": "paragraph", "text": text,
                                                            "section_path": [], "page": None}], **extra}


def save(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def receipt(doc):
    return {"document_id": doc["id"], "document_sha256": core.digest(doc), "decision": "approved",
            "reviewer_type": "human", "reviewer": "SYNTHETIC TEST FIXTURE - not a real reviewer",
            "reason": "Authored regression fixture; no real-world human review claim.",
            "license_decision": "approved", "license_evidence": ["https://example.test/license"]}


@pytest.mark.parametrize("use_cleaned", [False, True])
def test_identical_selected_content_cannot_cross_splits_under_new_family_ids(tmp_path, use_cleaned):
    docs = [document("a", "train", "Identical full source document."),
            document("b", "dev", "Different raw text." if use_cleaned else "Identical full source document.")]
    if use_cleaned:
        for doc in docs:
            doc["blocks"][0]["cleaned_text"] = "The exact selected training text is identical."
    source = tmp_path / "duplicate-source.jsonl"
    save(source, docs)
    with pytest.raises(ValueError, match="conflicting declared splits"):
        core._documents(source)


@pytest.mark.parametrize("marker", [{"evaluation_only": True}, {"kind": "benchmark_context"},
                                   {"benchmark": "longbench-v2"}, {"metadata": {"benchmark": "longbench-v1"}}])
def test_benchmark_is_denied_even_with_train_tag_license_and_review_receipt(tmp_path, marker):
    doc = document("evaluation", "train", "Held-out benchmark text.", **marker)
    review_path = tmp_path / "synthetic-review.jsonl"
    save(review_path, [receipt(doc)])
    _, report = core._admission([doc], review_path, ["CC0-1.0"])
    assert report[0]["eligible"] is False
    assert report[0]["reasons"] == ["reserved_benchmark_or_evaluation_only"]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Only asset-loading is mocked. All document/receipt/export/lineage/plan
    # validation logic executes unchanged and never invokes Torch or SWIFT.
    docs = [document("train", "train", "Distinct training-only source."),
            document("dev", "dev", "Separate development-only source.")]
    source, reviews, output = tmp_path / "source.jsonl", tmp_path / "reviews.jsonl", tmp_path / "prepared"
    save(source, docs); save(reviews, [receipt(doc) for doc in docs])
    metadata = {"special_tokens_budget": 2, "model_lock_sha256": "fixture-lock-hash",
                "repository": schema.MODEL_REPO, "revision": schema.MODEL_REVISION}
    monkeypatch.setattr(core, "load_native_tokenizer", lambda _: NativeTokenizer(len, metadata))
    manifest = core.prepare(source, tmp_path / "assets", output, reviews_path=reviews,
                            allowed_licenses=["CC0-1.0"], mode="training")
    monkeypatch.setattr(launch, "_locked_assets", lambda _: ({"repository": schema.MODEL_REPO,
                                                             "revision": schema.MODEL_REVISION}, "fixture-lock-hash"))
    assert launch._build(output, schema.default_recipe(), tmp_path / "run", "ordered")["training_ready"]
    return output, manifest


@pytest.mark.parametrize("target,replacement", [("train_ordered", "dev_ordered"), ("train_shuffled", "dev_shuffled"),
                                               ("dev_ordered", "train_ordered"), ("dev_shuffled", "train_shuffled")])
def test_valid_other_arm_path_and_hash_cannot_replace_its_bound_data(prepared, tmp_path, target, replacement):
    output, manifest = prepared
    manifest["artifacts"][target] = copy.deepcopy(manifest["artifacts"][replacement])
    (output / "prepare_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split/strategy lineage|share a path"):
        launch._build(output, schema.default_recipe(), tmp_path / "run", "ordered")


@pytest.mark.parametrize("mutation", ["lineage_document_hash", "lineage_span_document", "admission_split", "assignment_split"])
def test_manifest_file_hash_alone_cannot_hide_cross_source_lineage_changes(prepared, tmp_path, mutation):
    output, manifest = prepared
    filename = ("sequence_lineage.jsonl" if mutation.startswith("lineage") else
                "admission_report.jsonl" if mutation == "admission_split" else "split_assignments.jsonl")
    path = output / filename
    rows = core.read_jsonl(path)
    item = next(r for r in rows if r["document_id"] == "train")
    if mutation == "lineage_document_hash":
        item["document_sha256"] = "0" * 64
    elif mutation == "lineage_span_document":
        item["spans"][0]["document_id"] = "dev"
    else:
        item["split"] = "dev"
    save(path, rows)
    manifest["artifacts"][filename]["sha256"] = core.file_hash(path)
    (output / "prepare_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source admission or split assignment"):
        launch._build(output, schema.default_recipe(), tmp_path / "run", "ordered")
