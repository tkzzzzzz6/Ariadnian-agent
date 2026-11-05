"""Research-data invariants: provenance, leakage, exact limits and export gates."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from longdoc.pipeline import (  # noqa: E402
    BuildConfig, Tokenizer, UnionFind, assign_splits, build, clean_document,
    deduplicate, mix_documents, pack, read_documents,
)
from longdoc.assessment import assess, score_review  # noqa: E402
from longdoc.__main__ import main  # noqa: E402


def document(did="one", text="A real paragraph with source information.", **extra):
    return {"id": did, "family_id": did, "title": "Source title", "source_url": None,
            "license": None, "revision": "v1", "language": "en", "domain": "science", "kind": "report",
            "blocks": [{"id": "p1", "type": "paragraph", "text": text, "section_path": ["Introduction"], "page": 1}], **extra}


def write_input(tmp_path, docs):
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    return path


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def clean(doc):
    return clean_document(doc, BuildConfig(), Tokenizer())[0]


def test_schema_rejects_ambiguous_provenance_and_invalid_pages(tmp_path):
    doc = document()
    del doc["family_id"]
    with pytest.raises(ValueError, match="family_id"):
        read_documents(write_input(tmp_path, [doc]))
    doc = document()
    doc["blocks"][0]["page"] = True
    with pytest.raises(ValueError, match="positive integer"):
        read_documents(write_input(tmp_path, [doc]))
    with pytest.raises(ValueError, match="duplicate document id"):
        read_documents(write_input(tmp_path, [document(), document()]))


def test_cleaning_preserves_original_and_structured_spacing():
    doc = document(text="  This  is a hyphen-\nation and soft\u00adhyphen.  ")
    doc["blocks"] += [{"id": kind, "type": kind, "text": "  x\t  y\n  z  ", "section_path": [], "page": None}
                      for kind in ("heading", "code", "table", "formula")]
    original = copy.deepcopy(doc)
    result, audit = clean_document(doc, BuildConfig(), Tokenizer())
    assert doc == original
    assert result["blocks"][0]["text"] == original["blocks"][0]["text"]
    assert result["blocks"][0]["cleaned_text"] == "This is a hyphenation and softhyphen."
    assert len(audit) == 1 and audit[0]["original_sha256"] != audit[0]["cleaned_sha256"]
    for block in result["blocks"][1:]:
        assert block["cleaned_text"] == block["text"]


def test_near_duplicates_audited_and_family_transitivity_prevents_leakage():
    base = " ".join(f"word{i}" for i in range(150))
    docs = [clean(document("a", base, family_id="family-a")),
            clean(document("b", base + " extra", family_id="family-b")),
            clean(document("c", "Completely different third text", family_id="family-b"))]
    retained, candidates, removed, unions = deduplicate(docs, BuildConfig(near_action="remove"))
    assert len(removed) == 1 and removed[0]["review_status"] == "unreviewed_heuristic"
    assert any(c["decision"] == "duplicate" and c["method"].endswith("jaccard5") for c in candidates)
    assignments = assign_splits(docs, retained, unions, 17)
    assert len({a["cluster_id"] for a in assignments}) == 1
    assert len({a["split"] for a in assignments}) == 1


def test_default_near_candidates_are_kept_but_linked_for_leakage_prevention():
    base = " ".join(f"word{i}" for i in range(150))
    docs = [clean(document("a", base)), clean(document("b", base + " extra"))]
    retained, candidates, removed, unions = deduplicate(docs, BuildConfig())
    assert len(retained) == 2 and not removed
    assert candidates[0]["decision"] == "near_duplicate_candidate"
    assert unions.find("a") == unions.find("b")
    assignments = assign_splits(docs, retained, unions, 17)
    assert len({r["split"] for r in assignments}) == 1


def test_source_compilations_preserve_declared_parent_splits_and_connect_all_parents(tmp_path):
    parent_a = document("source-a", "Independent original alpha source.", family_id="family-a", split="test")
    parent_b = document("source-b", "An unrelated beta original source.", family_id="family-b", split="test")
    derived = document("compilation", "A derivative combining alpha and beta.", split="test",
                       parent_family_ids=["family-a", "family-b"], source_document_ids=["source-a", "source-b"])
    cleaned = [clean(d) for d in (parent_a, parent_b, derived)]
    assert all(d["split"] == "test" for d in cleaned)
    _, _, _, unions = deduplicate(cleaned, BuildConfig())
    assigned = assign_splits(cleaned, cleaned, unions, 999)
    assert {row["split"] for row in assigned} == {"test"}
    assert len({row["cluster_id"] for row in assigned}) == 1
    assert all(row["split_basis"] == "declared_component_split" for row in assigned)
    parent_b["split"] = "train"
    path = write_input(tmp_path, [parent_a, parent_b, derived])
    with pytest.raises(ValueError, match="conflicting declared splits"):
        build(path, tmp_path / "conflicting")
    assert not (tmp_path / "conflicting").exists()


def test_missing_parents_link_derivatives_and_expose_unverifiable_upstream_splits(tmp_path):
    first = document("derived-a", "Derived alpha with different body.", split="dev",
                     parent_family_ids=["outside-family"], source_document_ids=["outside-source-a"])
    second = document("derived-b", "Derived beta with another body.", split="dev",
                      parent_family_ids=["outside-family"], source_document_ids=["outside-source-b"])
    cleaned = [clean(d) for d in (first, second)]
    _, _, _, unions = deduplicate(cleaned, BuildConfig())
    rows = assign_splits(cleaned, cleaned, unions, 17)
    assert len({r["cluster_id"] for r in rows}) == 1
    assert all(r["missing_parent_family_ids"] == ["outside-family"] for r in rows)
    assert all(r["split"] == "dev" and r["missing_source_document_ids"] for r in rows)
    second["split"] = "test"
    with pytest.raises(ValueError, match="conflicting declared splits"):
        build(write_input(tmp_path, [first, second]), tmp_path / "denied-missing")
    first["source_splits"] = {"outside-source-a": "test"}
    with pytest.raises(ValueError, match="source_splits"):
        build(write_input(tmp_path, [first]), tmp_path / "denied-source-constraint")


def test_declared_duplicate_cluster_and_ancestry_schema_are_not_discarded(tmp_path):
    docs = [document("a", "The first independent prose source.", split="train", cluster_id="upstream-cluster"),
            document("b", "Completely separate text about another subject.", split="test", cluster_id="upstream-cluster")]
    with pytest.raises(ValueError, match="conflicting declared splits"):
        build(write_input(tmp_path, docs), tmp_path / "denied-cluster")
    bad = document(parent_family_ids="not-a-list")
    with pytest.raises(ValueError, match="parent_family_ids"):
        read_documents(write_input(tmp_path, [bad]))
    bad = document(source_document_ids=["source"], source_splits={"undeclared-source": "test"})
    with pytest.raises(ValueError, match="source_splits"):
        read_documents(write_input(tmp_path, [bad]))


def test_similar_prose_never_auto_removes_different_code_or_formula():
    base = " ".join(f"word{i}" for i in range(150))
    left, right = document("a", base), document("b", base)
    for doc, code in ((left, "if a:\n    return True"), (right, "if a:\nreturn True")):
        doc["blocks"].append({"id": "code", "type": "code", "text": code,
                              "section_path": [], "page": None})
    kept, candidates, removed, _ = deduplicate([clean(left), clean(right)], BuildConfig())
    assert len(kept) == 2 and not removed
    assert candidates and all(not c["protected_structure_identical"] for c in candidates)


def test_packing_covers_every_nonoversized_block_once_and_keeps_splits():
    doc = clean(document(text="中文字节"))
    doc["split"] = "train"
    doc["blocks"] += [dict(doc["blocks"][0], id="p2", text="middle", cleaned_text="middle"),
                      dict(doc["blocks"][0], id="p3", text="Z" * 50, cleaned_text="Z" * 50),
                      dict(doc["blocks"][0], id="p4", text="tail", cleaned_text="tail")]
    other = clean(document("other", "test only"))
    other["split"] = "test"
    tokenizer = Tokenizer()
    for strategy in ("document", "random"):
        packed, rejected = pack([doc, other], tokenizer, 20, 17, strategy)
        assert [r["block_id"] for r in rejected] == ["p3"]
        seen = []
        for row in packed:
            assert row["token_count"] == len(row["text"].encode("utf-8")) <= 20
            for span in row["spans"]:
                seen.append((span["document_id"], span["block_id"]))
                source = doc if span["document_id"] == "one" else other
                assert row["split"] == source["split"]
                block = next(b for b in source["blocks"] if b["id"] == span["block_id"])
                assert row["text"][span["start_char"]:span["end_char"]] == block["cleaned_text"]
        assert sorted(seen) == [("one", "p1"), ("one", "p2"), ("one", "p4"), ("other", "p1")]


def test_token_mixing_is_train_only_without_replacement_or_budget_overshoot():
    docs = [dict(clean(document(str(i), "X" * length, domain=domain)), split=split)
            for i, (length, domain, split) in enumerate([(10, "a", "train"), (20, "a", "train"),
                                                       (30, "b", "train"), (10, "b", "test")])]
    selected, report = mix_documents(docs, BuildConfig(mix_weights={"a": 1, "b": 1}, mix_token_budget=50))
    assert all(d["split"] == "train" for d in selected)
    assert len({d["id"] for d in selected}) == len(selected)
    assert report["used_units"] == sum(d["token_count"] for d in selected) <= 50
    assert sum(report["realized_units_by_domain"].values()) == report["used_units"]
    with pytest.raises(ValueError, match="no train documents"):
        mix_documents(docs, BuildConfig(mix_weights={"absent": 1}))


def test_build_reproducible_and_default_is_explicit_byte_assessment(tmp_path):
    source = write_input(tmp_path, [document(), document("two", "Different content and distinct subject")])
    one = build(source, tmp_path / "one", BuildConfig(sequence_length=100))
    two = build(source, tmp_path / "two", BuildConfig(sequence_length=100))
    assert one == two
    import hashlib

    package_dir = Path(__file__).resolve().parents[1] / "longdoc"
    assert one["implementation_sha256"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in sorted(package_dir.glob("*.py"))}
    assert one["tokenizer"]["unit"] == "utf8_bytes" and not one["training_export_allowed"]
    assert "NOT model tokens" in one["tokenizer"]["note"]
    assert all(not row["training_eligible"] for row in records(tmp_path / "one" / "training_gate.jsonl"))
    with pytest.raises(ValueError, match="never overwritten"):
        build(source, tmp_path / "one")


def make_local_tokenizer(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "A": 1, "B": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.enable_truncation(max_length=1)
    tok.enable_padding(length=20)
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def test_local_model_tokenizer_disables_embedded_truncation_padding(tmp_path):
    tokenizer = Tokenizer(str(make_local_tokenizer(tmp_path)))
    assert tokenizer.count("A B A B") == 4
    assert tokenizer.metadata["unit"] == "model_tokens"
    assert tokenizer.metadata["truncation"] is False


def test_training_requires_license_review_and_actual_model_tokenizer(tmp_path):
    source = write_input(tmp_path, [document()])
    with pytest.raises(ValueError, match="allowlist"):
        build(source, tmp_path / "denied", BuildConfig(purpose="training"))
    with pytest.raises(ValueError, match="assessment-only"):
        build(source, tmp_path / "denied", BuildConfig(purpose="training", allowed_licenses=("CC0-1.0",)))
    model_path = str(make_local_tokenizer(tmp_path))
    config = BuildConfig(purpose="training", allowed_licenses=("CC0-1.0",), tokenizer_path=model_path)
    with pytest.raises(ValueError, match="license/review gate"):
        build(source, tmp_path / "denied", config)
    approved = document(license="CC0-1.0", review_status="approved")
    source = write_input(tmp_path, [approved])
    manifest = build(source, tmp_path / "approved", config)
    assert manifest["training_export_allowed"]


def test_controlled_assessment_exposes_template_false_removal_and_keeps_natural_reviews_blank(tmp_path):
    text = " ".join(f"unique_source_word_{i}" for i in range(200))
    source = write_input(tmp_path, [document("a", text), document("b", text)])
    result = assess(source, tmp_path / "assessment", cases_per_type=2, sequence_length=200)
    assert "pipeline.py" in result["implementation_sha256"]
    assert "assessment.py" in result["implementation_sha256"]
    assert result["controlled_dedup"]["cases"] == 8
    by_type = result["controlled_dedup"]["by_case_type"]
    assert by_type["exact_copy"]["correct"] == 2
    assert by_type["different_protected_table"]["correct"] == 2
    # Honest diagnostic: distinct core records sharing a large template can be
    # falsely removed by a lexical near-duplicate heuristic. Never hide this.
    assert by_type["shared_template_distinct_records"]["correct"] == 0
    assert result["controlled_dedup"]["negative_false_candidate_rate"] == 0.5
    assert result["controlled_dedup"]["negative_false_removal_rate"] == 0
    assert result["controlled_quality"]["sensitive_cases"] == 4
    assert result["structure_integrity"]["integrity_errors"] == 0
    assert result["structure_integrity"]["oversized_blocks_explicitly_excluded"] == 2
    assert result["natural_review"]["reviewed_pairs"] == 0
    reviews = records(tmp_path / "assessment" / "dedup_review_template.jsonl")
    assert len(reviews) == 1
    assert all(r["label"] == r["review_status"] == r["review_method"] == r["reviewer"] == "" for r in reviews)
    controlled = records(tmp_path / "assessment" / "controlled_dedup_cases.jsonl")
    assert all(r["label_source"] == "controlled_construction" for r in controlled)
    risky = assess(source, tmp_path / "assessment-remove", cases_per_type=2, near_action="remove")
    assert risky["controlled_dedup"]["negative_false_removal_rate"] == 0.5


def test_review_scoring_does_not_turn_blank_uncertain_or_ai_assistance_into_human_passes(tmp_path):
    rows = []
    for i in range(4):
        rows.append({"item_id": str(i), "label_source": "natural_unreviewed", "predicted_label": "duplicate",
                     "label": "", "review_status": "", "reviewer": "", "review_method": "", "rationale": ""})
    rows[1].update(label="distinct", review_status="reviewed", reviewer="reviewer-A", review_method="ai_assisted", rationale="Distinct record IDs")
    rows[2].update(label="uncertain", review_status="reviewed", reviewer="reviewer-B", review_method="independent_human", rationale="Insufficient source")
    rows[3].update(label="duplicate", review_status="reviewed", reviewer="reviewer-C", review_method="independent_human")
    path = tmp_path / "review.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = score_review(path, tmp_path / "score.json", "dedup")
    assert result["reviewed_items"] == 2 and result["review_coverage"] == 0.5
    assert result["all_review_methods"]["binary_scored"] == 1
    assert result["all_review_methods"]["fp"] == 1
    assert result["by_review_method"]["independent_human"]["binary_scored"] == 0
    assert result["by_review_method"]["independent_human"]["accuracy"] is None
    assert result["by_review_method"]["ai_assisted"]["accuracy"] == 0
    rows[0]["label_source"] = "controlled_construction"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="controlled gold must remain separate"):
        score_review(path, tmp_path / "invalid.json", "dedup")


def test_cli_build_assess_and_blank_review_score(tmp_path, capsys):
    source = write_input(tmp_path, [document("a"), document("b")])
    assert main(["build", "--input", str(source), "--output", str(tmp_path / "build")]) == 0
    assert json.loads(capsys.readouterr().out)["tokenizer"]["unit"] == "utf8_bytes"
    assert main(["assess", "--input", str(source), "--output", str(tmp_path / "audit"), "--cases-per-type", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["natural_review"]["reviewed_pairs"] == 0
    assert main(["score-review", "--review", str(tmp_path / "audit" / "dedup_review_template.jsonl"),
                 "--kind", "dedup", "--output", str(tmp_path / "blank-score.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reviewed_items"] == 0 and result["all_review_methods"]["accuracy"] is None
    assert main(["build", "--input", str(source), "--output", str(tmp_path / "invalid"), "--mix", "domain=NaN"]) == 2
    assert "finite" in capsys.readouterr().err
