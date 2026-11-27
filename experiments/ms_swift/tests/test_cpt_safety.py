"""CPT admission invariants using local synthetic sources, never model downloads."""

from __future__ import annotations

import copy
from collections import Counter
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


from cpt import core  # noqa: E402


def _document(did: str, split: str = "train", **extra) -> dict:
    return {
        "id": did, "family_id": f"family-{did}", "title": f"Synthetic source {did}",
        "source_url": f"https://example.org/source/{did}", "license": "CC0-1.0",
        "license_evidence": ["https://example.org/license/cc0"], "revision": "fixture-v1",
        "language": "en", "domain": "fixture", "kind": "report", "split": split,
        "review_status": "unreviewed", "annotation_status": "synthetic_fixture",
        "blocks": [{"id": "p1", "type": "paragraph", "section_path": ["Introduction"],
                    "page": 1, "text": f"Original source content for document {did}."}],
        **extra,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _admission(tmp_path: Path, documents: list[dict]) -> tuple[Path, list[dict]]:
    source = _write_jsonl(tmp_path / "sources.jsonl", documents)
    core.audit(source, tmp_path / "audit", allowed_licenses=("CC0-1.0",))
    return source, _rows(tmp_path / "audit" / "admission_review.jsonl")


def _approved(rows: list[dict], *, reviewer_type: str = "human") -> list[dict]:
    reviews = copy.deepcopy(rows)
    for row in reviews:
        row.update(decision="approved", license_decision="approved", reviewer_type=reviewer_type,
                   reviewer="Synthetic test reviewer", reason="Fixture content and permission reviewed.",
                   license_evidence=["https://example.org/license/cc0"])
    return reviews


@pytest.fixture
def native_assets(tmp_path, monkeypatch):
    """A synthetic token counter, not a downloaded or validated Gemma tokenizer."""
    assets = tmp_path / "fake-model-assets"
    assets.mkdir()
    calls = []

    def load(path):
        assert Path(path) == assets
        calls.append(Path(path))
        return SimpleNamespace(
            count=lambda text: len(text.split()),
            metadata={"kind": "test-only-word-counter", "unit": "model_tokens",
                      "special_tokens_budget": 2, "model_id": "google/gemma-4-E2B",
                      "native_context_length": 131072},
        )

    monkeypatch.setattr(core, "load_native_tokenizer", load)
    return assets, calls


def test_audit_exports_hash_bound_pending_queue_without_approving_or_mutating_sources(tmp_path):
    documents = [_document("train"), _document("heldout", "dev")]
    source = _write_jsonl(tmp_path / "sources.jsonl", documents)
    original = source.read_bytes()

    manifest = core.audit(source, tmp_path / "audit", allowed_licenses=("CC0-1.0",))
    rows = _rows(tmp_path / "audit" / "admission_review.jsonl")

    assert source.read_bytes() == original
    assert {row["document_id"] for row in rows} == {"train", "heldout"}
    assert all(row["decision"] == "pending" for row in rows)
    assert all(row["license_decision"] == "pending" for row in rows)
    assert all(re.fullmatch(r"[a-f0-9]{64}", row["document_sha256"]) for row in rows)
    assert len({row["document_sha256"] for row in rows}) == len(documents)
    assert manifest.get("training_ready") is not True
    assert (tmp_path / "audit" / "audit_manifest.json").is_file()
    assert all(row["review_status"] == "unreviewed" for row in _rows(source))


@pytest.mark.parametrize("group_field", ["family_id", "cluster_id", "duplicate_cluster_id", "near_duplicate_cluster_id"])
def test_audit_rejects_conflicting_existing_splits_in_connected_sources(tmp_path, group_field):
    documents = [_document("train", **{group_field: "same-group"}),
                 _document("heldout", "test", **{group_field: "same-group"})]
    source = _write_jsonl(tmp_path / "sources.jsonl", documents)

    with pytest.raises(ValueError):
        core.audit(source, tmp_path / "audit", allowed_licenses=("CC0-1.0",))


@pytest.mark.parametrize("constraints", [
    {"source_document_ids": ["parent"], "source_splits": {"parent": "test"}},
    {"parent_family_ids": ["upstream"], "parent_family_splits": {"upstream": "dev"}},
])
def test_audit_rejects_explicit_upstream_holdout_conflicting_with_train(tmp_path, constraints):
    source = _write_jsonl(tmp_path / "sources.jsonl", [_document("derived", **constraints)])
    with pytest.raises(ValueError):
        core.audit(source, tmp_path / "audit", allowed_licenses=("CC0-1.0",))


def test_default_prepare_is_preview_and_does_not_promote_unreviewed_sources(tmp_path, native_assets):
    assets, calls = native_assets
    source = _write_jsonl(tmp_path / "sources.jsonl", [_document("train"), _document("heldout", "dev")])
    original = source.read_bytes()

    manifest = core.prepare(source, assets, tmp_path / "preview", allowed_licenses=("CC0-1.0",))

    assert calls
    assert manifest["training_ready"] is False
    assert source.read_bytes() == original
    assert all(row["review_status"] == "unreviewed" for row in _rows(source))


def test_native_tokenizer_rejection_never_falls_back_to_reference_or_byte_counts(tmp_path, monkeypatch):
    source = _write_jsonl(tmp_path / "sources.jsonl", [_document("train"), _document("heldout", "dev")])

    def rejected(_):
        raise ValueError("native tokenizer lock rejected")

    monkeypatch.setattr(core, "load_native_tokenizer", rejected)
    with pytest.raises(ValueError, match="native tokenizer lock rejected"):
        core.prepare(source, tmp_path / "bad-assets", tmp_path / "preview")
    assert not (tmp_path / "preview").exists()


@pytest.mark.parametrize("review_selection", ["missing", "pending", "train_only"])
def test_training_prepare_requires_reviewed_train_and_validation_sources(tmp_path, native_assets, review_selection):
    assets, _ = native_assets
    source, pending = _admission(tmp_path, [_document("train"), _document("heldout", "dev")])
    reviews_path = None
    if review_selection != "missing":
        reviews = pending if review_selection == "pending" else [
            row for row in _approved(pending) if row["document_id"] == "train"
        ]
        reviews_path = _write_jsonl(tmp_path / "reviews.jsonl", reviews)

    with pytest.raises(ValueError):
        core.prepare(source, assets, tmp_path / "training", reviews_path=reviews_path,
                     allowed_licenses=("CC0-1.0",), mode="training")


def test_inherited_approved_flags_cannot_replace_the_bound_admission_review(tmp_path, native_assets):
    assets, _ = native_assets
    documents = [_document("train", review_status="approved", training_eligible=True),
                 _document("heldout", "dev", review_status="approved", training_eligible=True)]
    source = _write_jsonl(tmp_path / "sources.jsonl", documents)

    with pytest.raises(ValueError):
        core.prepare(source, assets, tmp_path / "training", allowed_licenses=("CC0-1.0",), mode="training")


@pytest.mark.parametrize("reviewer_type", ["ai", "automated", "", "ai_generated"])
def test_automated_or_blank_review_cannot_authorize_training(tmp_path, native_assets, reviewer_type):
    assets, _ = native_assets
    source, pending = _admission(tmp_path, [_document("train"), _document("heldout", "dev")])
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", _approved(pending, reviewer_type=reviewer_type))

    with pytest.raises(ValueError):
        core.prepare(source, assets, tmp_path / "training", reviews_path=reviews,
                     allowed_licenses=("CC0-1.0",), mode="training")


@pytest.mark.parametrize("invalid_review", [
    {"reviewer": ""}, {"reason": ""}, {"license_evidence": []},
    {"license_evidence": [""]}, {"license_decision": "pending"}, {"decision": "pending"},
])
def test_incomplete_admission_cannot_be_treated_as_training_approval(tmp_path, native_assets, invalid_review):
    assets, _ = native_assets
    source, pending = _admission(tmp_path, [_document("train"), _document("heldout", "dev")])
    approved = _approved(pending)
    for row in approved:
        row.update(invalid_review)
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", approved)

    with pytest.raises(ValueError):
        core.prepare(source, assets, tmp_path / "training", reviews_path=reviews,
                     allowed_licenses=("CC0-1.0",), mode="training")


def test_review_for_older_source_content_does_not_authorize_changed_document(tmp_path, native_assets):
    assets, _ = native_assets
    documents = [_document("train"), _document("heldout", "dev")]
    source, pending = _admission(tmp_path, documents)
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", _approved(pending))
    documents[0]["blocks"][0]["text"] += " New content introduced after review."
    _write_jsonl(source, documents)
    changed_bytes = source.read_bytes()

    with pytest.raises(ValueError):
        core.prepare(source, assets, tmp_path / "training", reviews_path=reviews,
                     allowed_licenses=("CC0-1.0",), mode="training")
    assert source.read_bytes() == changed_bytes


@pytest.mark.parametrize("licenses", [(), ("CC0",), ("cc0-1.0",), ("CC0-1.0 OR MIT",)])
def test_license_admission_requires_an_exact_explicit_allowlist_match(tmp_path, native_assets, licenses):
    assets, _ = native_assets
    source, pending = _admission(tmp_path, [_document("train"), _document("heldout", "dev")])
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", _approved(pending))

    with pytest.raises(ValueError):
        core.prepare(source, assets, tmp_path / "training", reviews_path=reviews,
                     allowed_licenses=licenses, mode="training")


@pytest.mark.parametrize("reviewer_type", ["human", "ai_assisted_human"])
def test_complete_hash_bound_admission_can_prepare_training_without_mutating_source_reviews(
    tmp_path, native_assets, reviewer_type,
):
    assets, calls = native_assets
    source, pending = _admission(tmp_path, [_document("train"), _document("heldout", "dev"),
                                           _document("test-source", "test")])
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", _approved(pending, reviewer_type=reviewer_type))
    original = source.read_bytes()

    manifest = core.prepare(source, assets, tmp_path / "training", reviews_path=reviews,
                            allowed_licenses=("CC0-1.0",), mode="training", sequence_length=64)

    assert calls
    assert manifest["training_ready"] is True
    assert source.read_bytes() == original
    assert all(row["review_status"] == "unreviewed" for row in _rows(source))
    output = tmp_path / "training"
    assert manifest["test_documents_exported"] == 0
    for strategy in ("ordered", "shuffled"):
        train = _rows(output / manifest["artifacts"][f"train_{strategy}"]["path"])
        dev = _rows(output / manifest["artifacts"][f"dev_{strategy}"]["path"])
        assert train and dev
        assert all("document heldout." not in row["text"] and "document test-source." not in row["text"] for row in train)
        assert all("document train." not in row["text"] and "document test-source." not in row["text"] for row in dev)
    assignments = _rows(output / "split_assignments.jsonl")
    assert {row["document_id"]: row["split"] for row in assignments} == {
        "train": "train", "heldout": "dev", "test-source": "test",
    }


@pytest.mark.parametrize("atomic_kind", ["code", "table", "formula"])
def test_oversized_atomic_block_quarantines_its_whole_document_in_both_conditions(
    tmp_path, native_assets, atomic_kind,
):
    assets, _ = native_assets
    bad = _document("oversized")
    bad["blocks"][0]["text"] = "SHORT_PART_OF_QUARANTINED_DOCUMENT"
    bad["blocks"].append({"id": "atomic", "type": atomic_kind,
                          "section_path": ["Protected source"], "page": 1,
                          "text": "ATOMIC_REJECT_MARKER " + " ".join(f"token{i}" for i in range(100)),
                          "token_count": 1})  # A stale reference count must not bypass native counting.
    source, pending = _admission(tmp_path, [_document("train"), _document("heldout", "dev"), bad])
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", _approved(pending))

    manifest = core.prepare(source, assets, tmp_path / "training", reviews_path=reviews,
                            allowed_licenses=("CC0-1.0",), mode="training", sequence_length=64)

    assert manifest["training_ready"] is True
    texts = [row["text"] for path in (tmp_path / "training").rglob("*.jsonl") for row in _rows(path)
             if isinstance(row.get("text"), str)]
    assert texts, "Prepared SWIFT datasets must expose reviewable text rows."
    assert not any("ATOMIC_REJECT_MARKER" in text or "SHORT_PART_OF_QUARANTINED_DOCUMENT" in text for text in texts)
    assert any("Original source content for document train." in text for text in texts)
    quarantined = _rows(tmp_path / "training" / "quarantine.jsonl")
    assert manifest["quarantined_documents"] == 1
    assert any(row["document_id"] == "oversized" and row["block_id"] == "atomic" for row in quarantined)
    lineage = _rows(tmp_path / "training" / "sequence_lineage.jsonl")
    assert all(row["document_id"] != "oversized" for row in lineage)


def test_native_repacking_preserves_exact_source_spans_in_both_conditions_and_audits_paragraph_cuts(
    tmp_path, native_assets,
):
    assets, _ = native_assets
    train = _document("train")
    train["blocks"] = [
        {"id": "long", "type": "paragraph", "section_path": ["Long source"], "page": 1,
         "text": " ".join(f"source{i}" for i in range(100)), "token_count": 1},
        {"id": "short", "type": "paragraph", "section_path": ["Short source"], "page": 2,
         "text": "Original kept text.", "cleaned_text": "Selected cleaned text."},
    ]
    documents = [train, _document("heldout", "dev"), _document("test-source", "test")]
    source, pending = _admission(tmp_path, documents)
    reviews = _write_jsonl(tmp_path / "reviews.jsonl", _approved(pending))
    output = tmp_path / "training"

    manifest = core.prepare(source, assets, output, reviews_path=reviews,
                            allowed_licenses=("CC0-1.0",), mode="training", sequence_length=16)

    by_id = {document["id"]: document for document in documents}
    lineage = _rows(output / "sequence_lineage.jsonl")
    span_counters = {}
    for strategy in ("ordered", "shuffled"):
        spans = []
        train_lineage = [row for row in lineage if row["split"] == "train" and row["strategy"] == strategy]
        rows = _rows(output / manifest["artifacts"][f"train_{strategy}"]["path"])
        for record in train_lineage:
            slices = []
            for span in record["spans"]:
                document = by_id[span["document_id"]]
                block = next(block for block in document["blocks"] if block["id"] == span["block_id"])
                text = block[span["text_field"]]
                assert 0 <= span["start"] < span["end"] <= len(text)
                slices.append(text[span["start"]:span["end"]])
                spans.append((span["document_id"], span["block_id"], span["start"], span["end"], span["text_field"]))
            emitted = rows[record["row"]]["text"]
            assert emitted == "\n\n".join(slices)
            assert record["body_tokens"] == len(emitted.split()) <= 14
        span_counters[strategy] = Counter(spans)
        cuts = sorted((start, end) for _, block_id, start, end, _ in spans if block_id == "long")
        assert len(cuts) > 1  # A stale reference count must not suppress paragraph splitting.
        assert cuts[0][0] == 0 and cuts[-1][1] == len(train["blocks"][0]["text"])
        assert all(left[1] == right[0] for left, right in zip(cuts, cuts[1:]))
        assert manifest["counts"][f"train_{strategy}"]["body_tokens"] == sum(row["body_tokens"] for row in train_lineage)
    assert span_counters["ordered"] == span_counters["shuffled"]
    assert manifest["comparison"]["strict_equal_token_budget"] is False

