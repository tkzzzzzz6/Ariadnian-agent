"""Dataset admission and native-tokenizer exports. No weights or training here."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from ariadne_data.documents import UnionFind, assign_splits, read_documents
from .assets import load_native_tokenizer


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def new_output(path) -> Path:
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _documents(path):
    docs = read_documents(path)
    if any(d.get("split") not in {"train", "dev", "test"} for d in docs):
        raise ValueError("CPT requires existing explicit train/dev/test splits; no re-splitting is permitted")
    families = {d["family_id"] for d in docs}
    ids = {d["id"] for d in docs}
    for d in docs:
        for refs, mapping, known in (("parent_family_ids", "parent_family_splits", families),
                                     ("source_document_ids", "source_splits", ids)):
            if any(ref not in known and ref not in d.get(mapping, {}) for ref in d.get(refs, [])):
                raise ValueError(f"{d['id']}: missing upstream source split constraint")
    # Validate the complete input graph before admission filtering; never assign
    # a new split to an admitted subset, nor mutate the original documents.
    unions = UnionFind(d["id"] for d in docs)
    exact_owners = {}
    for doc in docs:
        body = "\n\n".join(b.get("cleaned_text", b["text"]) for b in doc["blocks"])
        sha = digest(body)
        if sha in exact_owners:
            unions.union(doc["id"], exact_owners[sha])
        else:
            exact_owners[sha] = doc["id"]
    assignments = assign_splits(docs, [], unions, 17)
    return docs, assignments


def _admission(docs, reviews_path, allowed_licenses):
    allowed = set(allowed_licenses)
    if any(not isinstance(v, str) or not v.strip() for v in allowed):
        raise ValueError("license allowlist must contain exact nonempty identifiers")
    reviews = {}
    known = {d["id"] for d in docs}
    for r in read_jsonl(reviews_path) if reviews_path else []:
        if not isinstance(r, dict):
            raise ValueError("review rows must be objects")
        did = r.get("document_id")
        if did not in known or did in reviews:
            raise ValueError("review contains an unknown or duplicate document_id")
        if r.get("decision") not in {"pending", "rejected", "approved"}:
            raise ValueError(f"{did}: invalid review decision")
        reviews[did] = r
    queue, report = [], []
    for d in docs:
        did, sha = d["id"], digest(d)
        queue.append({"document_id": did, "document_sha256": sha, "split": d["split"],
                      "license": d.get("license"), "decision": "pending", "reviewer_type": "",
                      "reviewer": "", "reason": "", "license_decision": "pending",
                      "license_evidence": d.get("license_evidence", [])})
        r = reviews.get(did, {})
        reasons = []
        metadata = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        if (d.get("evaluation_only") is True or d.get("benchmark") or metadata.get("benchmark")
                or d.get("kind") in {"benchmark", "benchmark_context", "benchmark_question", "benchmark_reference"}):
            reasons.append("reserved_benchmark_or_evaluation_only")
        if r and r.get("document_sha256") != sha:
            raise ValueError(f"{did}: stale review document_sha256 mismatch")
        if r.get("decision") != "approved":
            reasons.append("content_review_not_approved")
        if r.get("reviewer_type") not in {"human", "ai_assisted_human"}:
            reasons.append("explicit_human_review_receipt_required")
        if any(not isinstance(r.get(k), str) or not r[k].strip() for k in ("reviewer", "reason")):
            reasons.append("reviewer_and_reason_required")
        if d.get("license") not in allowed:
            reasons.append("license_not_explicitly_allowlisted")
        evidence = r.get("license_evidence")
        if (r.get("license_decision") != "approved" or not isinstance(evidence, list)
                or not evidence or any(not isinstance(v, str) or not v.strip() for v in evidence)):
            reasons.append("license_review_and_evidence_required")
        report.append({"document_id": did, "document_sha256": sha, "split": d["split"],
                       "eligible": not reasons, "reasons": reasons,
                       "review_receipt_sha256": digest(r) if r else None,
                       "reviewer_type": r.get("reviewer_type"),
                       "note": "Receipt records a supplied review claim; software cannot independently verify reviewer identity."})
    return queue, report


def audit(input_path, output_dir, *, reviews_path=None, allowed_licenses=()) -> dict:
    docs, assignments = _documents(input_path)
    queue, admission = _admission(docs, reviews_path, allowed_licenses)
    out = new_output(output_dir)
    write_jsonl(out / "admission_review.jsonl", queue)
    write_jsonl(out / "admission_report.jsonl", admission)
    write_jsonl(out / "split_assignments.jsonl", assignments)
    manifest = {"schema_version": "cpt-audit-v1", "training_ready": False,
                "input_sha256": file_hash(input_path), "documents": len(docs),
                "splits": dict(Counter(d["split"] for d in docs)),
                "approved_documents": sum(r["eligible"] for r in admission),
                "pending_queue_sha256": file_hash(out / "admission_review.jsonl"),
                "note": "Pending queue is not an approval or a training dataset."}
    write_json(out / "audit_manifest.json", manifest)
    return manifest


def _fragments(doc, count, limit):
    fragments, oversize = [], []
    for block in doc["blocks"]:
        # The filtered corpus's cleaned_text is explicitly selected without
        # changing the original JSON or performing any additional cleaning.
        text = block.get("cleaned_text", block["text"])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{doc['id']}:{block['id']}: selected source text is empty or invalid")
        if count(text) > limit and block["type"] != "paragraph":
            oversize.append({"document_id": doc["id"], "block_id": block["id"],
                             "block_type": block["type"], "body_tokens": count(text),
                             "reason": "oversized_atomic_block_entire_document_quarantined"})
            continue
        start = 0
        while start < len(text):
            end = len(text)
            if count(text[start:end]) > limit:
                lo, hi = start + 1, len(text)
                # Counts need not be strictly monotonic. Binary search only
                # proposes a cut; the exact emitted piece is checked below.
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if count(text[start:mid]) <= limit:
                        lo = mid
                    else:
                        hi = mid - 1
                end = lo
                while end > start and count(text[start:end]) > limit:
                    end -= 1
            if end <= start or count(text[start:end]) > limit:
                raise ValueError("sequence length cannot fit one source character")
            fragments.append({"document_id": doc["id"], "block_id": block["id"],
                              "block_type": block["type"], "start": start, "end": end,
                              "text": text[start:end], "text_field": "cleaned_text" if "cleaned_text" in block else "text"})
            start = end
    return fragments, oversize


def _sequences(fragments, count, limit):
    rows, lineage, current = [], [], []

    def flush():
        if not current:
            return
        text = "\n\n".join(f["text"] for f in current)
        tokens = count(text)
        if tokens > limit:
            raise ValueError("native-tokenizer overflow: no row was silently truncated")
        rows.append({"text": text})
        lineage.append({"row": len(rows) - 1, "text_sha256": digest(text), "body_tokens": tokens,
                        "spans": [{k: v for k, v in f.items() if k != "text"} for f in current]})
        current.clear()

    for fragment in fragments:
        proposed = "\n\n".join([*(f["text"] for f in current), fragment["text"]])
        if current and count(proposed) > limit:
            flush()
        current.append(fragment)
    flush()
    return rows, lineage


def prepare(input_path, model_assets, output_dir, *, reviews_path=None, allowed_licenses=(),
            sequence_length=8192, mode="preview", seed=17) -> dict:
    if mode not in {"preview", "training"}:
        raise ValueError("mode must be preview or training")
    if type(sequence_length) is not int or not 8 <= sequence_length <= 131072:
        raise ValueError("sequence_length must be an integer in [8,131072]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    docs, assignments = _documents(input_path)
    _, admission = _admission(docs, reviews_path, allowed_licenses)
    admitted = {r["document_id"] for r in admission if r["eligible"]}
    selected = [d for d in docs if d["split"] != "test" and (mode == "preview" or d["id"] in admitted)]
    if mode == "training" and not all(any(d["split"] == split for d in selected) for split in ("train", "dev")):
        raise ValueError("training export requires approved train and dev documents and exact license admission")
    tokenizer = load_native_tokenizer(model_assets)
    budget = tokenizer.metadata.get("special_tokens_budget", 2)
    if type(budget) is not int or not 0 < budget < sequence_length:
        raise ValueError("invalid conservative special-token budget")
    limit = sequence_length - budget
    all_rows, all_lineage, quarantined = {}, [], []
    kept = Counter()
    # Never cross document/split boundaries when forming a training sequence.
    for split in ("train", "dev"):
        ordered_rows, shuffled_rows = [], []
        ordered_lineage, shuffled_lineage = [], []
        for doc in selected:
            if doc["split"] != split:
                continue
            fragments, oversize = _fragments(doc, tokenizer.count, limit)
            if oversize:
                quarantined.extend(oversize)
                continue
            if not fragments:
                raise ValueError(f"{doc['id']}: no nonempty source fragments")
            kept[split] += 1
            # Shuffle original blocks as units: paragraph fragments stay
            # contiguous and in order, preserving content within each block.
            blocks = {}
            for f in fragments:
                blocks.setdefault(f["block_id"], []).append(f)
            keys = list(blocks)
            rng = random.Random(f"{seed}:{doc['id']}")
            rng.shuffle(keys)
            shuffled = [f for key in keys for f in blocks[key]]
            for strategy, parts, target_rows, target_lineage in (
                    ("ordered", fragments, ordered_rows, ordered_lineage),
                    ("shuffled", shuffled, shuffled_rows, shuffled_lineage)):
                rows, lineage = _sequences(parts, tokenizer.count, limit)
                for item in lineage:
                    item.update(row=item["row"] + len(target_rows), split=split, strategy=strategy,
                                document_id=doc["id"], document_sha256=digest(doc))
                target_rows.extend(rows)
                target_lineage.extend(lineage)
        all_rows[f"{split}_ordered"] = ordered_rows
        all_rows[f"{split}_shuffled"] = shuffled_rows
        all_lineage.extend(ordered_lineage + shuffled_lineage)
    if mode == "training" and (not kept["train"] or not kept["dev"]):
        raise ValueError("quarantine left no approved train/dev documents; cannot export training data")
    out = new_output(output_dir)
    artifacts, counts = {}, {}
    for key, rows in all_rows.items():
        name = key + (".preview.jsonl" if mode == "preview" else ".jsonl")
        write_jsonl(out / name, rows)
        artifacts[key] = {"path": name, "sha256": file_hash(out / name)}
        tokens = [r["body_tokens"] for r in all_lineage if f"{r['split']}_{r['strategy']}" == key]
        counts[key] = {"sequences": len(rows), "body_tokens": sum(tokens),
                       "max_body_tokens": max(tokens, default=0)}
    for name, rows in (("sequence_lineage.jsonl", all_lineage), ("quarantine.jsonl", quarantined),
                       ("admission_report.jsonl", admission), ("split_assignments.jsonl", assignments)):
        write_jsonl(out / name, rows)
        artifacts[name] = {"path": name, "sha256": file_hash(out / name)}
    manifest = {"schema_version": "cpt-prepared-v1", "mode": mode, "training_ready": mode == "training",
                "input_sha256": file_hash(input_path), "reviews_sha256": file_hash(reviews_path) if reviews_path else None,
                "allowed_licenses": sorted(set(allowed_licenses)), "tokenizer": tokenizer.metadata,
                "model_assets": str(Path(model_assets).resolve()), "sequence_length": sequence_length,
                "special_tokens_budget": budget, "seed": seed, "documents": len(docs),
                "approved_documents": len(admitted), "exported_documents": dict(kept),
                "test_documents_exported": 0, "quarantined_documents": len({r['document_id'] for r in quarantined}),
                "artifacts": artifacts, "counts": counts,
                "comparison": {"content_spans_equal": True, "strict_equal_token_budget": False,
                               "note": "Native BPE counts and sequence counts can change with block order. Report both; do not claim equal token/compute budgets."},
                "swift_template_verified": False,
                "note": "Native body counts reserve a conservative special-token budget. Exact SWIFT template verification is required at launch. Preview is never trainable."}
    write_json(out / "prepare_manifest.json", manifest)
    return manifest
