"""A deliberately conservative, standard-library-first corpus pipeline.

``block.text`` is immutable source content. ``cleaned_text`` is a separate,
audited view. This package never reads application retrieval chunks. Near
deduplication and quality scores are heuristic research candidates, not human
verification. Default UTF-8 byte counts are not model-token counts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "longdoc-v1"
BLOCK_TYPES = {"heading", "paragraph", "table", "formula", "code"}
REQUIRED_TEXT = ("id", "family_id", "title", "language", "domain", "kind")
SEPARATOR = "\n\n"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _implementation_snapshot() -> dict[str, str]:
    """Hash package source files at entry, without runtime settings or secrets."""
    return {path.name: _hash(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))}


@dataclass
class BuildConfig:
    seed: int = 17
    sequence_length: int = 8192
    tokenizer_path: str | None = None
    near_threshold: float = 0.92
    near_action: str = "candidate-only"
    min_quality_score: float = 0.0
    purpose: str = "assessment"
    allowed_licenses: tuple[str, ...] = ()
    mix_weights: dict[str, float] = field(default_factory=dict)
    mix_token_budget: int | None = None

    def validate(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not 0 < self.near_threshold <= 1:
            raise ValueError("near_threshold must be in (0,1]")
        if self.near_action not in {"candidate-only", "remove"}:
            raise ValueError("near_action must be candidate-only or remove")
        if not 0 <= self.min_quality_score <= 1:
            raise ValueError("min_quality_score must be in [0,1]")
        if self.purpose not in {"assessment", "training"}:
            raise ValueError("purpose must be assessment or training")
        if self.purpose == "training" and not self.allowed_licenses:
            raise ValueError("training export requires an explicit --allow-license allowlist")
        if self.mix_token_budget is not None and self.mix_token_budget < 1:
            raise ValueError("mix_token_budget must be positive")
        if any(not k or not math.isfinite(v) or v <= 0 for k, v in self.mix_weights.items()):
            raise ValueError("mix weights must be finite, positive and have nonempty domain names")


class Tokenizer:
    def __init__(self, path: str | None = None):
        self.impl = None
        if path:
            p = Path(path)
            if not p.is_file():
                raise ValueError(f"local tokenizer file does not exist: {p}")
            from tokenizers import Tokenizer as HFTokenizer

            self.impl = HFTokenizer.from_file(str(p))
            # Caller-supplied tokenizer files can otherwise silently truncate
            # or pad the input, corrupting corpus counts and packing limits.
            self.impl.no_truncation()
            self.impl.no_padding()
            self.metadata = {"kind": "huggingface-tokenizers-json", "unit": "model_tokens",
                             "path": str(p.resolve()), "sha256": _hash(p.read_bytes()),
                             "add_special_tokens": False, "truncation": False, "padding": False,
                             "note": "Counts apply to this exact local tokenizer, without BOS/EOS. Training must budget special tokens separately."}
        else:
            self.metadata = {"kind": "utf8-byte-diagnostic", "unit": "utf8_bytes",
                             "note": "Diagnostic bytes, NOT model tokens; no 4K/8K model-context claim is valid."}

    def encode(self, text: str) -> list[int]:
        return self.impl.encode(text, add_special_tokens=False).ids if self.impl else list(text.encode("utf-8"))

    def count(self, text: str) -> int:
        return len(self.encode(text))


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


def document_text(doc: dict, *, cleaned: bool = True) -> str:
    return SEPARATOR.join(b.get("cleaned_text", b["text"]) if cleaned else b["text"] for b in doc["blocks"])


def clean_document(doc: dict, config: BuildConfig, tokenizer: Tokenizer) -> tuple[dict, list[dict]]:
    out = copy.deepcopy(doc)
    audits = []
    # These fields belong to this build, never trust pre-existing computed values.
    # Existing source splits and duplicate clusters are provenance constraints,
    # not disposable statistics. assign_splits validates and preserves them.
    for key in ("lineage", "quality", "token_count", "training_eligible"):
        out.pop(key, None)
    for block in out["blocks"]:
        original = block["text"]
        cleaned = original
        operations = []
        # Code/table/formula spacing and heading formatting remain intact.
        if block["type"] == "paragraph":
            for name, transform in (
                ("remove_soft_hyphen", lambda t: t.replace("\u00ad", "")),
                ("join_lowercase_line_hyphenation", lambda t: re.sub(r"-\r?\n(?=[a-z])", "", t)),
                ("normalize_inline_spaces", lambda t: re.sub(r"[ \t]+", " ", t)),
                ("trim_paragraph_edges", lambda t: t.strip()),
            ):
                changed = transform(cleaned)
                if changed != cleaned:
                    operations.append(name)
                cleaned = changed
        if not cleaned.strip():
            # A pathological soft-hyphen-only paragraph is preserved rather
            # than silently dropping a source block.
            cleaned = original
            operations = ["cleaning_would_empty_block_preserved_original"]
        block["cleaned_text"] = cleaned
        block["token_count"] = tokenizer.count(cleaned)
        if operations:
            audits.append({"document_id": doc["id"], "block_id": block["id"],
                           "operations": operations, "original_sha256": _hash(original),
                           "cleaned_sha256": _hash(cleaned), "original_chars": len(original),
                           "cleaned_chars": len(cleaned)})
    out["lineage"] = {"original_document_sha256": _hash(_json(doc)),
                      "original_text_sha256": _hash(document_text(doc, cleaned=False)),
                      "cleaned_text_sha256": _hash(document_text(out)), "schema_version": SCHEMA_VERSION}
    out["training_eligible"] = doc.get("license") in config.allowed_licenses and doc.get("review_status") == "approved"
    out["token_count"] = tokenizer.count(document_text(out))
    if out["token_count"] == 0:
        raise ValueError(f"document {doc['id']!r} encodes to zero units with the selected tokenizer")
    out["token_unit"] = tokenizer.metadata["unit"]
    return out, audits


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


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _shingles(text: str) -> array:
    words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]|[^\s]", _normalized(text))
    pieces = [" ".join(words)] if len(words) < 5 else (" ".join(words[i:i + 5]) for i in range(len(words) - 4))
    # Compressed, sorted uint64 fingerprints avoid retaining millions of Python
    # shingle strings for real long-document corpora. No randomized Python hash.
    return array("Q", sorted({int.from_bytes(hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest(), "big")
                              for piece in pieces}))


def _jaccard(left: array, right: array) -> float:
    i, j, intersection = 0, 0, 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            intersection += 1
            i, j = i + 1, j + 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return intersection / max(1, len(left) + len(right) - intersection)


def _content_signature(doc: dict, *, protected_only: bool = False) -> str:
    # Do not equate whitespace/case changes in code, formulas or tables with
    # exact duplicates, and do not erase structural boundaries for exact dedup.
    return _hash(_json([{"type": b["type"], "text": b["cleaned_text"], "section_path": b["section_path"]}
                        for b in doc["blocks"] if not protected_only or b["type"] != "paragraph"]))


def deduplicate(docs: list[dict], config: BuildConfig) -> tuple[list[dict], list[dict], list[dict], UnionFind]:
    """Deterministic bottom-32 shingle-hash candidates + exact Jaccard.

    Candidate generation is approximate; near-duplicate recall is not guaranteed.
    Audits expose every compared candidate and every automatic removal.
    """
    ordered = sorted(docs, key=lambda d: (not d["training_eligible"], d["id"]))
    unions = UnionFind([d["id"] for d in docs])
    removals = UnionFind([d["id"] for d in docs])
    exact: dict[str, str] = {}
    index: dict[int, list[str]] = defaultdict(list)
    features: dict[str, array] = {}
    lengths: dict[str, int] = {}
    protected_signatures: dict[str, str] = {}
    candidates, decisions = [], []
    for doc in ordered:
        did = doc["id"]
        text = _normalized(document_text(doc))
        digest = _content_signature(doc)
        protected_signatures[did] = _content_signature(doc, protected_only=True)
        lengths[did] = len(text)
        shingles = _shingles(text)
        features[did] = shingles
        keys = shingles[:32]
        if digest in exact:
            previous = exact[digest]
            unions.union(did, previous)
            removals.union(did, previous)
            candidates.append({"left": previous, "right": did, "method": "cleaned_blocks_exact",
                               "similarity": 1.0, "decision": "duplicate"})
        else:
            neighbors = sorted({other for key in keys for other in index[key]})
            for previous in neighbors:
                similarity = _jaccard(shingles, features[previous])
                length_ratio = min(lengths[did], lengths[previous]) / max(1, lengths[did], lengths[previous])
                protected_equal = protected_signatures[did] == protected_signatures[previous]
                suspected = similarity >= config.near_threshold and length_ratio >= 0.9
                duplicate = suspected and protected_equal and config.near_action == "remove"
                candidates.append({"left": previous, "right": did, "method": "bottom32_shingle_candidate_jaccard5",
                                   "similarity": round(similarity, 8), "length_ratio": round(length_ratio, 8),
                                   "threshold": config.near_threshold, "minimum_length_ratio": 0.9,
                                   "protected_structure_identical": protected_equal,
                                   "similarity_threshold_passed": suspected, "auto_remove_eligible": suspected and protected_equal,
                                   "split_link_due_to_similarity": suspected,
                                   "decision": "duplicate" if duplicate else "near_duplicate_candidate" if suspected and protected_equal else "keep_both",
                                   "review_status": "unreviewed_heuristic"})
                if suspected:
                    unions.union(did, previous)
                if duplicate:
                    removals.union(did, previous)
            exact[digest] = did
        for key in keys:
            index[key].append(did)
    keep: dict[str, dict] = {}
    for doc in ordered:
        cluster = removals.find(doc["id"])
        if cluster not in keep:
            keep[cluster] = doc
        else:
            decisions.append({"document_id": doc["id"], "kept_document_id": keep[cluster]["id"],
                              "duplicate_cluster": cluster, "reason": "cleaned_blocks_exact_or_near_duplicate_cluster",
                              "review_status": "unreviewed_heuristic", "reversible": True})
    return sorted(keep.values(), key=lambda d: d["id"]), candidates, decisions, unions


def quality(doc: dict) -> dict:
    text = document_text(doc)
    paragraph_texts = [_normalized(b["cleaned_text"]) for b in doc["blocks"] if b["type"] == "paragraph"]
    repeated = len(paragraph_texts) - len(set(paragraph_texts))
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    repeated_ratio = repeated / max(1, len(paragraph_texts))
    control_ratio = sum(ord(c) < 32 and c not in "\n\r\t" for c in text) / max(1, len(text))
    score = max(0.0, 1.0 - min(1.0, replacement_ratio * 20) - repeated_ratio - min(1.0, control_ratio * 20))
    return {"method": "explainable_heuristic_v1_not_human_verified", "score": round(score, 6),
            "replacement_character_ratio": replacement_ratio, "control_character_ratio": control_ratio,
            "duplicate_paragraph_ratio": repeated_ratio, "characters": len(text),
            "block_types": dict(sorted(Counter(b["type"] for b in doc["blocks"]).items())),
            "section_path_coverage": sum(bool(b["section_path"]) for b in doc["blocks"]) / len(doc["blocks"]),
            "page_coverage": sum(b["page"] is not None for b in doc["blocks"]) / len(doc["blocks"]),
            "note": "Length and structure coverage are diagnostics, not proof of semantic quality, coherence or OCR correctness."}


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


def _segments(doc: dict, tokenizer: Tokenizer, limit: int) -> tuple[list[dict], list[dict]]:
    """Split only at block boundaries. Oversized atomic blocks are explicit rejects."""
    segments, oversized = [], []
    current: list[dict] = []

    def emit() -> None:
        if current:
            segments.append({"document_id": doc["id"], "family_id": doc["family_id"],
                             "domain": doc["domain"], "split": doc["split"],
                             "segment_index": len(segments), "blocks": list(current)})
            current.clear()

    for block in doc["blocks"]:
        text = block["cleaned_text"]
        # Recompute for the selected tokenizer rather than trusting counts from
        # a prior run or a caller-mutated block.
        count = tokenizer.count(text)
        if count > limit:
            emit()
            oversized.append({"document_id": doc["id"], "block_id": block["id"], "token_count": count,
                              "token_unit": tokenizer.metadata["unit"], "limit": limit,
                              "reason": "atomic_block_exceeds_limit_no_truncation", "chars": len(text)})
            continue
        if current and tokenizer.count(SEPARATOR.join([b["cleaned_text"] for b in current] + [text])) > limit:
            emit()
        current.append(block)
    emit()
    return segments, oversized


def pack(docs: list[dict], tokenizer: Tokenizer, limit: int, seed: int, strategy: str) -> tuple[list[dict], list[dict]]:
    segments, oversize = [], []
    for doc in sorted(docs, key=lambda d: d["id"]):
        parts, rejected = _segments(doc, tokenizer, limit)
        segments.extend(parts)
        oversize.extend(rejected)
    if strategy == "random":
        random.Random(seed).shuffle(segments)
    elif strategy != "document":
        raise ValueError(f"unknown packing strategy: {strategy}")
    outputs = []
    # A sequence can never contain documents from different data splits.
    for split in ("train", "dev", "test"):
        text = ""
        spans: list[dict] = []

        def emit() -> None:
            nonlocal text, spans
            if spans:
                count = tokenizer.count(text)
                outputs.append({"id": f"{strategy}-{split}-{len(outputs):06d}", "split": split,
                                "strategy": strategy, "text": text, "spans": spans,
                                "token_count": count, "token_unit": tokenizer.metadata["unit"],
                                "sequence_limit": limit, "fill_ratio": count / limit,
                                "special_tokens_added": False})
                text, spans = "", []

        for part in [s for s in segments if s["split"] == split]:
            body = SEPARATOR.join(b["cleaned_text"] for b in part["blocks"])
            proposed = text + (SEPARATOR if text else "") + body
            if spans and tokenizer.count(proposed) > limit:
                emit()
            offset = len(text) + (len(SEPARATOR) if text else 0)
            text += (SEPARATOR if text else "") + body
            for block in part["blocks"]:
                end = offset + len(block["cleaned_text"])
                spans.append({"document_id": part["document_id"], "family_id": part["family_id"],
                              "block_id": block["id"], "type": block["type"],
                              "section_path": block["section_path"], "page": block["page"],
                              "segment_index": part["segment_index"], "start_char": offset, "end_char": end})
                offset = end + len(SEPARATOR)
        emit()
    return outputs, oversize


def mix_documents(docs: list[dict], config: BuildConfig) -> tuple[list[dict], dict]:
    """Greedy remaining-token-deficit sampler; no replacement or document cuts."""
    train = [d for d in docs if d["split"] == "train"]
    available: dict[str, list[dict]] = defaultdict(list)
    for doc in train:
        available[doc["domain"]].append(doc)
    weights = config.mix_weights or {domain: 1.0 for domain in sorted(available)}
    missing = set(weights) - set(available)
    if missing:
        raise ValueError(f"mix domains have no train documents: {', '.join(sorted(missing))}")
    total_available = sum(d["token_count"] for domain in weights for d in available[domain])
    budget = config.mix_token_budget if config.mix_token_budget is not None else total_available
    total_weight = sum(weights.values())
    targets = {domain: budget * weight / total_weight for domain, weight in sorted(weights.items())} if weights else {}
    rng = random.Random(config.seed)
    for domain in sorted(available):
        rng.shuffle(available[domain])
    selected = []
    counts = Counter({domain: 0 for domain in weights})
    used = 0
    while used < budget:
        feasible = {domain: [d for d in available[domain] if d["token_count"] <= budget - used]
                    for domain in weights}
        feasible = {domain: candidates for domain, candidates in feasible.items() if candidates}
        if not feasible:
            break
        domain = max(sorted(feasible), key=lambda k: (targets[k] - counts[k]) / max(1.0, targets[k]))
        candidates = feasible[domain]
        # Closest to remaining domain target, deterministic after seeded shuffle.
        choice = min(candidates, key=lambda d: abs(d["token_count"] - max(0, targets[domain] - counts[domain])))
        available[domain].remove(choice)
        selected.append(choice)
        counts[domain] += choice["token_count"]
        used += choice["token_count"]
    return selected, {"method": "remaining_token_deficit_whole_documents_without_replacement",
                      "scope": "train_only", "budget": budget, "available_units": total_available,
                      "used_units": used, "unused_budget": budget - used,
                      "target_units_by_domain": targets, "realized_units_by_domain": dict(sorted(counts.items())),
                      "selected_document_ids": [d["id"] for d in selected],
                      "note": "Whole-document granularity and source scarcity can prevent exact target proportions. Counts exclude inter-document separators and training special tokens."}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(_json(record) + "\n")


def packing_metrics(docs: list[dict], packs: list[dict], oversize: list[dict], tokenizer: Tokenizer) -> dict:
    positions = {(d["id"], b["id"]): i for d in docs for i, b in enumerate(d["blocks"])}
    seen: dict[str, int] = {}
    breaks = boundaries = multiple = emitted_blocks = 0
    emitted_keys = set()
    for sequence in packs:
        previous_doc = None
        sequence_docs = set()
        for span in sequence["spans"]:
            did = span["document_id"]
            index = positions[(did, span["block_id"])]
            if did in seen and index < seen[did]:
                breaks += 1
            if previous_doc is not None and previous_doc != did:
                boundaries += 1
            previous_doc, seen[did] = did, index
            sequence_docs.add(did)
            emitted_keys.add((did, span["block_id"]))
            emitted_blocks += 1
        multiple += len(sequence_docs) > 1
    block_counts = {(d["id"], b["id"]): b["token_count"] for d in docs for b in d["blocks"]}
    fills = [p["fill_ratio"] for p in packs]
    return {"token_unit": tokenizer.metadata["unit"], "sequences": len(packs),
            "mean_fill_ratio": sum(fills) / len(fills) if fills else None,
            "min_fill_ratio": min(fills) if fills else None, "max_fill_ratio": max(fills) if fills else None,
            "emitted_sequence_units_including_separators": sum(p["token_count"] for p in packs),
            "source_document_units_before_packing": sum(d["token_count"] for d in docs),
            "source_block_units_before_packing": sum(block_counts.values()),
            "retained_source_block_units": sum(block_counts[k] for k in emitted_keys),
            "oversized_atomic_block_units": sum(r["token_count"] for r in oversize),
            "emitted_blocks": emitted_blocks, "oversized_blocks": len(oversize),
            "within_sequence_document_boundaries": boundaries, "sequences_with_multiple_documents": multiple,
            "source_block_order_breaks": breaks,
            "note": "Source block/document counts and emitted counts have different separators and may not be additive under a model tokenizer. Order breaks count backward source-block moves across emitted sequences."}


def build(input_path: str | Path, output_dir: str | Path, config: BuildConfig | None = None) -> dict:
    implementation_sha256 = _implementation_snapshot()
    config = config or BuildConfig()
    config.validate()
    input_path, output = Path(input_path), Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be absent or an empty directory; existing runs are never overwritten")
    tokenizer = Tokenizer(config.tokenizer_path)
    if config.purpose == "training" and tokenizer.impl is None:
        raise ValueError("training export requires a local model tokenizer; UTF-8 bytes are assessment-only")
    raw = read_documents(input_path)
    cleaned, cleaning_audit = [], []
    for doc in raw:
        item, audit = clean_document(doc, config, tokenizer)
        item["quality"] = quality(item)
        cleaned.append(item)
        cleaning_audit.extend(audit)
    kept, candidate_audit, duplicate_audit, unions = deduplicate(cleaned, config)
    quality_rejected = [{"document_id": d["id"], "reason": "heuristic_quality_below_configured_threshold",
                         "quality_score": d["quality"]["score"], "threshold": config.min_quality_score}
                        for d in kept if d["quality"]["score"] < config.min_quality_score]
    retained = [d for d in kept if d["quality"]["score"] >= config.min_quality_score]
    assignments = assign_splits(cleaned, retained, unions, config.seed)
    gate = [{"document_id": d["id"], "license": d["license"], "review_status": d.get("review_status", "unreviewed"),
             "training_eligible": d["training_eligible"],
             "reasons": ([] if d["license"] in config.allowed_licenses else ["license_not_explicitly_allowlisted"])
                        + ([] if d.get("review_status") == "approved" else ["review_not_approved"])} for d in cleaned]
    if config.purpose == "training":
        denied = [d["id"] for d in retained if not d["training_eligible"]]
        if denied:
            raise ValueError(f"training export denied for {len(denied)} retained documents (license/review gate): {', '.join(denied[:10])}")
    document_packs, oversize = pack(retained, tokenizer, config.sequence_length, config.seed, "document")
    random_packs, _ = pack(retained, tokenizer, config.sequence_length, config.seed, "random")
    if config.purpose == "training" and oversize:
        raise ValueError("training export denied: oversized atomic blocks; run assessment first to inspect oversize audit")
    mixed, mixing = mix_documents(retained, config)
    mixing["token_unit"] = tokenizer.metadata["unit"]
    mixed_packs, mixed_oversize = pack(mixed, tokenizer, config.sequence_length, config.seed, "document")
    files = {
        "raw.jsonl": raw, "cleaned.jsonl": cleaned, "deduped.jsonl": kept, "corpus.jsonl": retained,
        "cleaning_audit.jsonl": cleaning_audit, "dedup_candidates.jsonl": candidate_audit,
        "dedup_removed.jsonl": duplicate_audit, "quality_rejected.jsonl": quality_rejected,
        "split_assignments.jsonl": assignments, "training_gate.jsonl": gate,
        "oversize_blocks.jsonl": oversize, "packed_document.jsonl": document_packs,
        "packed_random.jsonl": random_packs, "mixed_documents.jsonl": mixed,
        "packed_mixed.jsonl": mixed_packs, "mixed_oversize_blocks.jsonl": mixed_oversize,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, records in files.items():
        _write_jsonl(output / name, records)
    (output / "mixing_report.json").write_text(json.dumps(mixing, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION, "input_sha256": _hash(input_path.read_bytes()),
        "implementation_sha256": implementation_sha256,
        "config": asdict(config), "tokenizer": tokenizer.metadata,
        "counts": {"raw_documents": len(raw), "cleaned_documents": len(cleaned),
                   "changed_blocks": len(cleaning_audit), "compared_duplicate_candidates": len(candidate_audit),
                   "duplicates_removed": len(duplicate_audit), "quality_rejected": len(quality_rejected),
                   "retained_documents": len(retained), "oversize_blocks": len(oversize),
                   "document_sequences": len(document_packs), "random_sequences": len(random_packs),
                   "mixed_documents": len(mixed), "mixed_sequences": len(mixed_packs)},
        "retained_counts_by_split": dict(sorted(Counter(d["split"] for d in retained).items())),
        "retained_units_by_domain": dict(sorted(_sum_by(retained, "domain").items())),
        "retained_units_by_language": dict(sorted(_sum_by(retained, "language").items())),
        "retained_units_by_split": dict(sorted(_sum_by(retained, "split").items())),
        "split_policy": "family/duplicate/declared-cluster/source-ancestry connected components; declared document/source splits preserved and conflicts rejected; otherwise SHA256(seed:component), expected 80/10/10; missing parents linked by declared identity and audited; no exact counts or stratification",
        "dedup_policy": "literal cleaned block content/type/section-path exact; normalized bottom-32 uint64 BLAKE2 shingle candidates then fingerprint Jaccard; near matches are candidate-only by default but always linked for split assignment; protected structural differences prevent automatic removal even in explicit remove mode; near recall unmeasured",
        "packing_policy": "block-boundary splits only; explicit oversized-block rejects; per-split; no BOS/EOS or padding; exact re-tokenization of emitted text",
        "packing_metrics": {"document": packing_metrics(retained, document_packs, oversize, tokenizer),
                            "random": packing_metrics(retained, random_packs, oversize, tokenizer),
                            "mixed": packing_metrics(mixed, mixed_packs, mixed_oversize, tokenizer)},
        "purpose": config.purpose,
        "training_export_allowed": config.purpose == "training",
        "limitations": ["Heuristic quality is not independent human verification.",
                        "Assessment outputs are not permission to train or redistribute source documents.",
                        "Near-dedup removal candidates require human audit; no accuracy claim has been measured.",
                        "Splitting an oversized document at block boundaries can shorten long-range dependencies.",
                        "Random packs shuffle document segments; document packs preserve source-block order. Neither implements semantic document ordering.",
                        "No continued-pretraining or benchmark improvement is claimed by this pipeline."],
        "files": {name: {"sha256": _hash((output / name).read_bytes()), "records": len(records)}
                  for name, records in files.items()},
    }
    manifest["files"]["mixing_report.json"] = {"sha256": _hash((output / "mixing_report.json").read_bytes())}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _sum_by(docs: list[dict], key: str) -> Counter:
    counts: Counter = Counter()
    for doc in docs:
        counts[doc[key]] += doc["token_count"]
    return counts
