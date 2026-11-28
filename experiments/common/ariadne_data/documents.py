"""Canonical longdoc-v1 schema, JSONL loading and split provenance.

Extracted from the assessment pipeline without changing validation or assignment
behavior. This package is independent of backend, tokenizers and training tools.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "longdoc-v1"


BLOCK_TYPES = {"heading", "paragraph", "table", "formula", "code"}


REQUIRED_TEXT = ("id", "family_id", "title", "language", "domain", "kind")


SEPARATOR = "\n\n"


def _hash(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read_documents(path: str | Path) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                validate_document(doc)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if doc["id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate document id {doc['id']!r}")
            seen.add(doc["id"])
            docs.append(doc)
    if not docs:
        raise ValueError("input contains no documents")
    return sorted(docs, key=lambda d: d["id"])


def validate_document(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("document must be an object")
    for name in REQUIRED_TEXT:
        if not isinstance(doc.get(name), str) or not doc[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("source_url", "license", "revision"):
        if name not in doc or (doc[name] is not None and not isinstance(doc[name], str)):
            raise ValueError(f"{name} is required and must be a string or null")
    if "review_status" in doc and not isinstance(doc["review_status"], str):
        raise ValueError("review_status must be a string when provided")
    if "split" in doc and doc["split"] not in {"train", "dev", "test"}:
        raise ValueError("declared split must be train, dev or test")
    for name in ("parent_family_ids", "source_document_ids"):
        if name in doc and (not isinstance(doc[name], list)
                            or any(not isinstance(v, str) or not v.strip() for v in doc[name])
                            or len(set(doc[name])) != len(doc[name])):
            raise ValueError(f"{name} must be a list of unique nonempty strings")
    for name in ("cluster_id", "duplicate_cluster_id", "near_duplicate_cluster_id"):
        if name in doc and (not isinstance(doc[name], str) or not doc[name].strip()):
            raise ValueError(f"{name} must be a nonempty string when provided")
    for name, references in (("source_splits", "source_document_ids"), ("parent_family_splits", "parent_family_ids")):
        if name in doc:
            mapping = doc[name]
            if (not isinstance(mapping, dict) or any(not isinstance(k, str) or k not in doc.get(references, [])
                                                    or v not in {"train", "dev", "test"} for k, v in mapping.items())):
                raise ValueError(f"{name} must map declared {references} to train/dev/test")
    blocks = doc.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("blocks must be a nonempty list")
    ids: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("block must be an object")
        bid = block.get("id")
        if not isinstance(bid, str) or not bid.strip() or bid in ids:
            raise ValueError("block ids must be nonempty and unique within a document")
        ids.add(bid)
        if block.get("type") not in BLOCK_TYPES:
            raise ValueError(f"unsupported block type: {block.get('type')!r}")
        if not isinstance(block.get("text"), str) or not block["text"].strip():
            raise ValueError(f"block {bid}: text must be a nonempty string")
        section = block.get("section_path")
        if not isinstance(section, list) or any(not isinstance(s, str) or not s.strip() for s in section):
            raise ValueError(f"block {bid}: section_path must be a list of nonempty titles")
        page = block.get("page")
        if "page" not in block or (page is not None and (type(page) is not int or page < 1)):
            raise ValueError(f"block {bid}: page must be null or a positive integer")


class UnionFind:
    def __init__(self, values: list[str]):
        self.parents = {v: v for v in values}

    def find(self, value: str) -> str:
        if self.parents[value] != value:
            self.parents[value] = self.find(self.parents[value])
        return self.parents[value]

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def assign_splits(all_docs: list[dict], retained: list[dict], duplicate_unions: UnionFind, seed: int) -> list[dict]:
    """Connect source ancestry before accepting or inferring component splits.

    Missing sources cannot be verified, but their declared family/source keys
    still link sibling derivatives. Explicit source split declarations must
    agree with every document in the connected component.
    """
    owners: dict[str, str] = {}
    available_ids = {d["id"] for d in all_docs}
    available_families = {d["family_id"] for d in all_docs}
    for doc in all_docs:
        keys = ["family:" + doc["family_id"], "document:" + doc["id"]]
        keys.extend("family:" + family for family in doc.get("parent_family_ids", []))
        keys.extend("document:" + did for did in doc.get("source_document_ids", []))
        for field in ("cluster_id", "duplicate_cluster_id", "near_duplicate_cluster_id"):
            if doc.get(field):
                keys.append(f"declared-{field}:" + doc[field])
        for key in keys:
            if key in owners:
                duplicate_unions.union(doc["id"], owners[key])
            else:
                owners[key] = doc["id"]
    components: dict[str, list[dict]] = defaultdict(list)
    for doc in all_docs:
        components[duplicate_unions.find(doc["id"])].append(doc)
    decisions: dict[str, tuple[str, str]] = {}
    for cluster, members in components.items():
        declarations = []
        for doc in members:
            if doc.get("split"):
                declarations.append((f"document:{doc['id']}", doc["split"]))
            for field in ("source_splits", "parent_family_splits"):
                declarations.extend((f"{doc['id']}:{field}:{source}", split) for source, split in doc.get(field, {}).items())
        values = {split for _, split in declarations}
        if len(values) > 1:
            detail = ", ".join(f"{source}={split}" for source, split in sorted(declarations))
            raise ValueError(f"conflicting declared splits within family/duplicate/source-ancestry component {cluster}: {detail}")
        if values:
            split, basis = next(iter(values)), "declared_component_split"
        else:
            bucket = int(_hash(f"{seed}:{cluster}")[:16], 16) / 2**64
            split = "train" if bucket < 0.8 else "dev" if bucket < 0.9 else "test"
            basis = "seeded_component_hash_no_declared_split"
        decisions[cluster] = split, basis
    assignments = []
    for doc in all_docs:
        cluster = duplicate_unions.find(doc["id"])
        split, basis = decisions[cluster]
        assignments.append({"document_id": doc["id"], "family_id": doc["family_id"],
                            "cluster_id": cluster, "split": split, "input_split": doc.get("split"),
                            "split_basis": basis, "parent_family_ids": doc.get("parent_family_ids", []),
                            "source_document_ids": doc.get("source_document_ids", []),
                            "missing_parent_family_ids": [v for v in doc.get("parent_family_ids", []) if v not in available_families],
                            "missing_source_document_ids": [v for v in doc.get("source_document_ids", []) if v not in available_ids],
                            "ancestry_note": "Missing source documents/families are linked by declared keys; their upstream split cannot be independently checked unless supplied as a split constraint."})
    lookup = {a["document_id"]: a for a in assignments}
    for doc in retained:
        doc.update({k: lookup[doc["id"]][k] for k in ("cluster_id", "split")})
    return assignments
