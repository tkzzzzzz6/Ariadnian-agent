"""Shared contracts can run independently of the application and model stack."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ariadne_data.documents import UnionFind, assign_splits, read_documents, validate_document


def document(did="one", **extra):
    return {"id": did, "family_id": did, "title": "Source title", "source_url": None,
            "license": None, "revision": "v1", "language": "en", "domain": "science", "kind": "report",
            "blocks": [{"id": "p1", "type": "paragraph", "text": "A source paragraph.",
                        "section_path": ["Introduction"], "page": 1}], **extra}


def input_file(tmp_path, docs):
    path = tmp_path / "input.jsonl"
    path.write_text("\ufeff\n" + "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    return path


def test_package_imports_and_validates_with_only_standard_library():
    # -I/-S excludes the active environment's packages and PYTHONPATH. Supply
    # only this package root, never the backend or its application environment.
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from ariadne_data.documents import validate_document, UnionFind, assign_splits
doc = json.loads(sys.argv[2])
validate_document(doc)
assert assign_splits([doc], [], UnionFind([doc['id']]), 17)[0]['split'] == 'test'
assert not {'longdoc', 'app', 'torch', 'transformers', 'tokenizers', 'fastapi'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(root),
                             json.dumps(document(split="test"))], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reader_accepts_bom_blank_lines_and_sorts_without_mutating_file(tmp_path):
    path = input_file(tmp_path, [document("z"), document("a")])
    original = path.read_bytes()
    assert [d["id"] for d in read_documents(path)] == ["a", "z"]
    assert path.read_bytes() == original


def test_reader_rejects_duplicate_ids_and_empty_inputs(tmp_path):
    with pytest.raises(ValueError, match="duplicate document id"):
        read_documents(input_file(tmp_path, [document(), document()]))
    with pytest.raises(ValueError, match="no documents"):
        read_documents(input_file(tmp_path, []))


@pytest.mark.parametrize("changes,match", [
    ({"source_document_ids": ["parent"], "source_splits": {"unknown": "train"}}, "source_splits"),
    ({"parent_family_ids": ["family", "family"]}, "parent_family_ids"),
    ({"cluster_id": " "}, "cluster_id"),
    ({"split": "validation"}, "split"),
])
def test_provenance_constraints_are_validated(changes, match):
    with pytest.raises(ValueError, match=match):
        validate_document(document(**changes))


def test_schema_rejects_boolean_page_and_duplicate_blocks():
    doc = document()
    doc["blocks"][0]["page"] = True
    with pytest.raises(ValueError, match="positive integer"):
        validate_document(doc)
    doc = document()
    doc["blocks"].append(copy.deepcopy(doc["blocks"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_document(doc)


@pytest.mark.parametrize("field", ["family_id", "cluster_id", "duplicate_cluster_id", "near_duplicate_cluster_id"])
def test_connected_holdout_conflicts_are_rejected(field):
    docs = [document("a", split="train", **{field: "shared"}),
            document("b", split="test", **{field: "shared"})]
    with pytest.raises(ValueError, match="conflicting declared splits"):
        assign_splits(docs, [], UnionFind([d["id"] for d in docs]), 17)


def test_missing_ancestors_link_siblings_and_preserve_reviewable_unknowns():
    docs = [document("a", split="dev", parent_family_ids=["missing-family"],
                     source_document_ids=["missing-a"]),
            document("b", split="dev", parent_family_ids=["missing-family"],
                     source_document_ids=["missing-b"])]
    original = copy.deepcopy(docs)
    assigned = assign_splits(docs, [], UnionFind(["a", "b"]), 999)
    assert docs == original
    assert {row["split"] for row in assigned} == {"dev"}
    assert {row["cluster_id"] for row in assigned} == {"a"}
    assert all(row["missing_parent_family_ids"] == ["missing-family"] for row in assigned)
    assert assigned[0]["missing_source_document_ids"] == ["missing-a"]
    docs[0]["source_splits"] = {"missing-a": "test"}
    with pytest.raises(ValueError, match="conflicting declared splits"):
        assign_splits(docs, [], UnionFind(["a", "b"]), 999)


def test_prior_duplicate_unions_are_preserved_and_retained_rows_receive_only_split_fields():
    docs = [document("z", split="test"), document("a")]
    retained = [copy.deepcopy(docs[1])]
    unions = UnionFind(["z", "a"])
    unions.union("z", "a")
    assigned = assign_splits(docs, retained, unions, 17)
    assert {row["split"] for row in assigned} == {"test"}
    assert {row["cluster_id"] for row in assigned} == {"a"}
    assert retained == [{**docs[1], "split": "test", "cluster_id": "a"}]


def test_seeded_fallback_stays_deterministic_and_order_independent():
    docs = [document("a"), document("b"), document("c")]
    first = assign_splits(docs, [], UnionFind(["a", "b", "c"]), 17)
    second = assign_splits(list(reversed(docs)), [], UnionFind(["c", "b", "a"]), 17)
    result = {row["document_id"]: (row["split"], row["split_basis"]) for row in first}
    assert result == {row["document_id"]: (row["split"], row["split_basis"]) for row in second}
    assert all(basis == "seeded_component_hash_no_declared_split" for _, basis in result.values())
