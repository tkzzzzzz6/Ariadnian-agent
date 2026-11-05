"""Controlled data diagnostics and explicitly separate natural-data review.

Constructed labels diagnose known transformations, not natural-document semantic
gold. Review templates are blank and do not approve any source for training.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

from .pipeline import (
    BLOCK_TYPES, BuildConfig, Tokenizer, _hash, _implementation_snapshot, _json, _write_jsonl,
    clean_document, deduplicate, document_text, pack, quality, read_documents,
)


def _prepared(doc: dict, tokenizer: Tokenizer) -> dict:
    result = clean_document(doc, BuildConfig(), tokenizer)[0]
    result["quality"] = quality(result)
    return result


def _pair(source: dict, kind: str, index: int, seed: int) -> tuple[dict, dict, str]:
    left, right = copy.deepcopy(source), copy.deepcopy(source)
    prefix = f"controlled-{kind}-{index:04d}"
    left["id"], right["id"] = prefix + "-left", prefix + "-right"
    left["family_id"], right["family_id"] = left["id"], right["id"]
    for doc in (left, right):
        doc["diagnostic_source_id"] = source["id"]
        doc["review_status"] = "controlled_diagnostic_not_human_reviewed"
    label = "duplicate"
    if kind == "near_copy_small_typo":
        paragraphs = [b for b in right["blocks"] if b["type"] == "paragraph"]
        block = max(paragraphs or right["blocks"], key=lambda b: len(b["text"]))
        if block["type"] == "paragraph":
            # One known small textual edit. This is lineage-based near-copy
            # gold, not a claim that every edit preserves scientific semantics.
            midpoint = len(block["text"]) // 2
            block["text"] = block["text"][:midpoint] + f" diagnostic_typo_{seed}_{index} " + block["text"][midpoint:]
        else:
            # No prose exists: add a short synthetic paragraph. Source
            # scarcity remains visible through source length in each case.
            right["blocks"].append({"id": "controlled-added", "type": "paragraph", "text": "diagnostic_typo",
                                    "section_path": [], "page": None})
    elif kind in {"shared_template_distinct_records", "different_protected_table"}:
        label = "distinct"
        template = " ".join(f"Standard report item {i} follows the shared collection and measurement protocol."
                            for i in range(100))
        for doc, record_id, value in ((left, f"A-{index}", 11), (right, f"B-{index}", 99)):
            payload_type = "table" if kind == "different_protected_table" else "paragraph"
            doc["blocks"] = [
                {"id": "template", "type": "paragraph", "text": template, "section_path": [], "page": None},
                {"id": "record", "type": payload_type,
                 "text": f"Synthetic record ID: {record_id}. Measured value: {value}.",
                 "section_path": ["Record"], "page": None},
            ]
            # This is explicitly synthetic diagnostic material, not a source
            # derivative with inherited licensing or natural-document gold.
            doc["source_url"], doc["license"], doc["revision"] = None, "CC0-1.0", "synthetic-diagnostic-v1"
            doc["kind"] = "controlled_synthetic_record"
            doc["title"], doc["language"], doc["domain"] = "Controlled synthetic record diagnostic", "en", "diagnostic"
    return left, right, label


def _structure_diagnostics(docs: list[dict], tokenizer: Tokenizer, limit: int, seed: int) -> dict:
    prepared = [_prepared(d, tokenizer) for d in docs]
    for doc in prepared:
        doc["split"] = "train"  # Diagnostic packing only; no training artifact or split assignment.
    packs, oversize = pack(prepared, tokenizer, limit, seed, "document")
    lookup = {(d["id"], b["id"]): b for d in prepared for b in d["blocks"]}
    original_lookup = {(d["id"], b["id"]): b for d in docs for b in d["blocks"]}
    emitted = Counter()
    errors = []
    for row in packs:
        for span in row["spans"]:
            key = (span["document_id"], span["block_id"])
            emitted[key] += 1
            block = lookup[key]
            if row["text"][span["start_char"]:span["end_char"]] != block["cleaned_text"]:
                errors.append({"document_id": key[0], "block_id": key[1], "reason": "serialized_span_mismatch"})
            if span["section_path"] != block["section_path"] or span["page"] != block["page"]:
                errors.append({"document_id": key[0], "block_id": key[1], "reason": "source_metadata_mismatch"})
    rejected = {(r["document_id"], r["block_id"]) for r in oversize}
    for key, block in lookup.items():
        if block["text"] != original_lookup[key]["text"]:
            errors.append({"document_id": key[0], "block_id": key[1], "reason": "source_text_mutated"})
        if block["type"] != "paragraph" and block["cleaned_text"] != block["text"]:
            errors.append({"document_id": key[0], "block_id": key[1], "reason": "protected_content_mutated"})
        if emitted[key] != (0 if key in rejected else 1):
            errors.append({"document_id": key[0], "block_id": key[1], "reason": "missing_or_repeated_emission"})
    return {"scope": "structured-input preservation and serialized source spans; NOT real PDF extraction gold",
            "source_blocks": len(lookup), "emitted_blocks": sum(emitted.values()),
            "oversized_blocks_explicitly_excluded": len(oversize), "integrity_errors": len(errors),
            "errors": errors, "oversize_audit": oversize,
            "source_blocks_by_type": dict(sorted(Counter(b["type"] for b in lookup.values()).items())),
            "supported_types_observed": sorted({b["type"] for b in lookup.values()}),
            "supported_types_unobserved": sorted(BLOCK_TYPES - {b["type"] for b in lookup.values()}),
            "token_unit": tokenizer.metadata["unit"], "sequence_limit": limit}


def assess(input_path: str | Path, output_dir: str | Path, *, seed: int = 17,
           cases_per_type: int = 20, near_threshold: float = 0.92,
           near_action: str = "candidate-only",
           tokenizer_path: str | None = None, sequence_length: int = 8192) -> dict:
    implementation_sha256 = _implementation_snapshot()
    if cases_per_type < 1:
        raise ValueError("cases_per_type must be positive")
    config = BuildConfig(seed=seed, near_threshold=near_threshold, near_action=near_action, sequence_length=sequence_length)
    config.validate()
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("assessment output must be absent or empty; existing runs are never overwritten")
    docs = read_documents(input_path)
    tokenizer = Tokenizer(tokenizer_path)
    sampled = list(docs)
    random.Random(seed).shuffle(sampled)
    sampled = sampled[:cases_per_type]
    dedup_cases, fixtures = [], []
    kinds = ("exact_copy", "near_copy_small_typo", "shared_template_distinct_records", "different_protected_table")
    for kind in kinds:
        for index, source in enumerate(sampled):
            left, right, label = _pair(source, kind, index, seed)
            fixtures.extend((left, right))
            pair = [_prepared(left, tokenizer), _prepared(right, tokenizer)]
            kept, candidates, removed, _ = deduplicate(pair, config)
            prediction = "duplicate" if any(c["decision"] in {"duplicate", "near_duplicate_candidate"} for c in candidates) else "distinct"
            dedup_cases.append({"case_id": f"{kind}-{index:04d}", "case_type": kind,
                                "label_source": "controlled_construction", "source_document_id": source["id"],
                                "left_id": left["id"], "right_id": right["id"], "label": label,
                                "predicted_label": prediction, "candidate_generated": bool(candidates),
                                "removed_documents": len(removed), "auto_removed": len(kept) == 1, "correct": prediction == label,
                                "source_characters": len(document_text(source, cleaned=False)),
                                "max_candidate_similarity": max((c["similarity"] for c in candidates), default=None)})
    quality_cases = []
    for source in sampled:
        baseline = _prepared(source, tokenizer)
        for kind in ("replacement_characters_injected", "repeated_paragraphs_injected"):
            variant = copy.deepcopy(source)
            paragraphs = [b for b in variant["blocks"] if b["type"] == "paragraph"]
            if not paragraphs:
                paragraphs = [{"id": "diagnostic-prose", "type": "paragraph", "text": "A controlled diagnostic paragraph.",
                               "section_path": [], "page": None}]
                variant["blocks"].extend(paragraphs)
            if kind == "replacement_characters_injected":
                paragraphs[0]["text"] += "\ufffd" * max(20, len(document_text(source, cleaned=False)) // 10)
            else:
                for i, block in enumerate(list(paragraphs)):
                    variant["blocks"].append({**block, "id": f"diagnostic-repeat-{i}-{_hash(block['id'])[:8]}"})
            degraded = _prepared(variant, tokenizer)
            before, after = baseline["quality"]["score"], degraded["quality"]["score"]
            evaluable = before > 0
            quality_cases.append({"case_id": f"{kind}-{source['id']}", "case_type": kind,
                                  "label_source": "known_injected_corruption_only", "source_document_id": source["id"],
                                  "baseline_score": before, "perturbed_score": after, "score_delta": after - before,
                                  "evaluable": evaluable, "sensitive": after < before if evaluable else None,
                                  "note": "The source baseline is unreviewed, not gold high-quality prose."})
    natural = [_prepared(doc, tokenizer) for doc in docs]
    _, candidates, _, _ = deduplicate(natural, config)
    pair_review = [{"item_id": f"natural-pair-{i:06d}", "left_id": c["left"], "right_id": c["right"],
                    "method": c["method"], "similarity": c["similarity"],
                    "predicted_label": "duplicate" if c["decision"] in {"duplicate", "near_duplicate_candidate"} else "distinct",
                    "auto_removed": c["decision"] == "duplicate",
                    "label_source": "natural_unreviewed", "label": "", "review_status": "", "reviewer": "",
                    "review_method": "", "rationale": ""} for i, c in enumerate(candidates)]
    quality_review = [{"item_id": f"quality-{d['id']}", "document_id": d["id"], "quality": d["quality"],
                       "prediction_threshold": 0.7, "predicted_label": "degraded" if d["quality"]["score"] < 0.7 else "adequate",
                       "label_source": "natural_unreviewed", "label": "", "review_status": "", "reviewer": "",
                       "review_method": "", "rationale": ""} for d in natural]
    positive = [r for r in dedup_cases if r["label"] == "duplicate"]
    negative = [r for r in dedup_cases if r["label"] == "distinct"]
    structure = _structure_diagnostics(docs, tokenizer, sequence_length, seed)
    result = {
        "schema_version": "longdoc-assessment-v1", "seed": seed,
        "implementation_sha256": implementation_sha256,
        "input_sha256": _hash(Path(input_path).read_bytes()), "tokenizer": tokenizer.metadata,
        "config": {"cases_per_type_requested": cases_per_type, "near_threshold": near_threshold, "near_action": near_action,
                   "sequence_length": sequence_length},
        "controlled_dedup": {
            "scope": "constructed textual copies and synthetic record identities; not natural-pair gold",
            "source_documents_used": len(sampled), "cases": len(dedup_cases),
            "positive_candidate_recall": sum(r["candidate_generated"] for r in positive) / len(positive),
            "positive_threshold_detection_recall": sum(r["predicted_label"] == "duplicate" for r in positive) / len(positive),
            "positive_auto_removal_recall": sum(r["auto_removed"] for r in positive) / len(positive),
            "negative_false_candidate_rate": sum(r["predicted_label"] == "duplicate" for r in negative) / len(negative),
            "negative_false_removal_rate": sum(r["auto_removed"] for r in negative) / len(negative),
            "by_case_type": {kind: {"cases": sum(r["case_type"] == kind for r in dedup_cases),
                                     "correct": sum(r["correct"] for r in dedup_cases if r["case_type"] == kind),
                                     "auto_removed": sum(r["auto_removed"] for r in dedup_cases if r["case_type"] == kind),
                                     "candidate_generated": sum(r["candidate_generated"] for r in dedup_cases if r["case_type"] == kind)}
                             for kind in kinds},
            "limitations": ["Controlled variants sharing source documents are not independent natural samples.",
                            "A small textual near-copy label denotes provenance similarity, not scientific equivalence.",
                            "Different record identities with shared templates stress false candidates even at high lexical similarity.",
                            "Default candidate-only mode does not automatically remove near matches; an explicit remove-mode run measures its removal risk."]},
        "controlled_quality": {
            "scope": "paired sensitivity to injected corruption; NOT semantic quality accuracy",
            "cases": len(quality_cases), "evaluable_cases": sum(r["evaluable"] for r in quality_cases),
            "score_floor_inconclusive_cases": sum(not r["evaluable"] for r in quality_cases),
            "sensitive_cases": sum(r["sensitive"] is True for r in quality_cases),
            "by_case_type": {kind: {"cases": sum(r["case_type"] == kind for r in quality_cases),
                                     "evaluable": sum(r["evaluable"] for r in quality_cases if r["case_type"] == kind),
                                     "sensitive": sum(r["sensitive"] is True for r in quality_cases if r["case_type"] == kind)}
                             for kind in ("replacement_characters_injected", "repeated_paragraphs_injected")}},
        "structure_integrity": structure,
        "natural_review": {"dedup_candidate_pairs": len(pair_review), "quality_documents": len(quality_review),
                           "reviewed_pairs": 0, "reviewed_quality_documents": 0, "coverage": 0.0,
                           "note": "Blank fields mean unreviewed. Natural candidates are not controlled gold. No source is automatically approved."},
        "limitations": ["Candidate-only review cannot measure missed natural duplicate pairs; add independently sampled noncandidate pairs for natural recall.",
                        "No PDF parsing accuracy, human quality accuracy, training effect or model-context capability is measured here."],
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"controlled_dedup_cases.jsonl": dedup_cases, "controlled_pair_documents.jsonl": fixtures,
                 "controlled_quality_cases.jsonl": quality_cases, "dedup_review_template.jsonl": pair_review,
                 "quality_review_template.jsonl": quality_review}
    for name, rows in artifacts.items():
        _write_jsonl(output / name, rows)
    result["files"] = {name: {"records": len(rows), "sha256": _hash((output / name).read_bytes())}
                       for name, rows in artifacts.items()}
    (output / "assessment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def score_review(review_path: str | Path, output_path: str | Path, kind: str) -> dict:
    if kind not in {"dedup", "quality"}:
        raise ValueError("review kind must be dedup or quality")
    output = Path(output_path)
    if output.exists():
        raise ValueError("review score output already exists; existing results are never overwritten")
    rows = []
    with Path(review_path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{review_path}:{line_no}: invalid JSON") from exc
                if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
                    raise ValueError(f"{review_path}:{line_no}: item_id is required")
                if row.get("label_source") != "natural_unreviewed":
                    raise ValueError("review scoring accepts natural-data templates only; controlled gold must remain separate")
                rows.append(row)
    if len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate review item_id")
    positive, negative = ("duplicate", "distinct") if kind == "dedup" else ("degraded", "adequate")
    allowed_methods = {"independent_human", "ai_assisted", "llm_only"}
    complete, incomplete = [], []
    for row in rows:
        ready = (row.get("review_status") == "reviewed" and row.get("label") in {positive, negative, "uncertain"}
                 and isinstance(row.get("reviewer"), str) and bool(row["reviewer"].strip())
                 and row.get("review_method") in allowed_methods
                 and isinstance(row.get("rationale"), str) and bool(row["rationale"].strip()))
        if ready:
            if row.get("predicted_label") not in {positive, negative}:
                raise ValueError(f"item {row['item_id']}: invalid predicted_label")
            complete.append(row)
        else:
            incomplete.append(row["item_id"])

    def metrics(items: list[dict]) -> dict:
        binary = [r for r in items if r["label"] != "uncertain"]
        tp = sum(r["label"] == positive and r["predicted_label"] == positive for r in binary)
        fp = sum(r["label"] == negative and r["predicted_label"] == positive for r in binary)
        tn = sum(r["label"] == negative and r["predicted_label"] == negative for r in binary)
        fn = sum(r["label"] == positive and r["predicted_label"] == negative for r in binary)
        return {"reviewed": len(items), "binary_scored": len(binary), "uncertain": len(items) - len(binary),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall_within_reviewed_candidates": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "accuracy": (tp + tn) / len(binary) if binary else None}

    result = {"schema_version": "longdoc-review-score-v1", "kind": kind,
              "input_sha256": _hash(Path(review_path).read_bytes()), "total_items": len(rows),
              "reviewed_items": len(complete), "unreviewed_or_incomplete_items": len(incomplete),
              "review_coverage": len(complete) / len(rows) if rows else 0.0,
              "incomplete_item_ids": incomplete, "all_review_methods": metrics(complete),
              "by_review_method": {method: metrics([r for r in complete if r["review_method"] == method])
                                   for method in sorted(allowed_methods)},
              "note": "Review methods are caller declarations, not independently authenticated. Blank/incomplete/uncertain labels never count as passed checks. Candidate-only recall is not full natural-corpus recall."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
