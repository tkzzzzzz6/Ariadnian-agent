"""Source-bound single-turn SFT preparation; no model imports or implicit review.

Records contain id, split, source_document_ids, license, kind and messages.
Both the answer record and every source document require human review for a
training export. Preview exports cannot be executed. Test/benchmark rows fail
closed rather than being repurposed as training examples.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from .core import (_documents, _admission, digest, file_hash, new_output,
                   read_jsonl, write_json, write_jsonl)
from .assets import load_native_tokenizer


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) not in {2, 3}:
        raise ValueError("SFT requires one user/assistant turn with an optional system message")
    expected = ["user", "assistant"] if len(messages) == 2 else ["system", "user", "assistant"]
    for item, role in zip(messages, expected):
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] != role or not isinstance(item["content"], str)
                or not item["content"].strip()):
            raise ValueError("SFT messages require exact role/content fields, ordered roles and nonempty text")


def _review_rows(path, rows):
    reviews = {}
    known = {r["id"] for r in rows}
    for r in read_jsonl(path) if path else []:
        if not isinstance(r, dict):
            raise ValueError("SFT review rows must be objects")
        rid = r.get("record_id")
        if rid not in known or rid in reviews:
            raise ValueError("unknown or duplicate SFT review record_id")
        if r.get("decision") not in {"pending", "approved", "rejected"}:
            raise ValueError("invalid SFT review decision")
        reviews[rid] = r
    return reviews


def prepare_sft(input_path, source_corpus, model_assets, output_dir, *, reviews_path=None,
                source_reviews_path=None, allowed_licenses=(), sequence_length=4096, mode="preview"):
    if mode not in {"preview", "training"} or type(sequence_length) is not int or not 8 <= sequence_length <= 131072:
        raise ValueError("invalid SFT mode or sequence length")
    bound_paths = {"input_sha256": input_path, "source_corpus_sha256": source_corpus,
                   "reviews_sha256": reviews_path, "source_reviews_sha256": source_reviews_path}
    # Bind the bytes before reading records or receipts. Hashing only at the end
    # could attest to new file contents while exporting the previously read data.
    source_hashes = {key: file_hash(path) if path is not None else None
                     for key, path in bound_paths.items()}
    docs, assignments = _documents(source_corpus)
    _, source_admission = _admission(docs, source_reviews_path, allowed_licenses)
    sources = {d["id"]: d for d in docs}
    source_eligibility = {r["document_id"]: r for r in source_admission}
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("SFT input must not be empty")
    ids, content_splits = set(), {}
    for row in rows:
        allowed = {"id", "split", "source_document_ids", "license", "kind", "messages", "evaluation_only", "benchmark"}
        required = allowed - {"evaluation_only", "benchmark"}
        if not isinstance(row, dict) or not required <= set(row) or set(row) - allowed:
            raise ValueError("SFT row has missing/unknown fields")
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("SFT ids must be nonempty and unique")
        ids.add(row["id"])
        if not isinstance(row["license"], str) or not row["license"].strip():
            raise ValueError("SFT records require a nonempty exact license identifier")
        if (row["split"] not in {"train", "dev"} or row.get("evaluation_only") is True
                or row.get("benchmark") or row["kind"] not in {"human_qa", "reviewed_synthetic_qa"}):
            raise ValueError("SFT rejects test/benchmark/evaluation-only or unknown-kind samples")
        validate_messages(row["messages"])
        source_ids = row["source_document_ids"]
        if (not isinstance(source_ids, list) or not source_ids or any(not isinstance(x, str) for x in source_ids)
                or len(set(source_ids)) != len(source_ids) or any(x not in sources for x in source_ids)):
            raise ValueError("SFT requires unique known source_document_ids")
        for sid in source_ids:
            source = sources[sid]
            if source["split"] != row["split"]:
                raise ValueError("SFT source split differs from record split")
            if "reserved_benchmark_or_evaluation_only" in source_eligibility[sid]["reasons"]:
                raise ValueError("SFT source is benchmark/evaluation-only")
        # Same prompt with a changed answer must not cross train/dev either.
        for content in (row["messages"], row["messages"][:-1]):
            sha = digest(content)
            if sha in content_splits and content_splits[sha] != row["split"]:
                raise ValueError("SFT train/dev share an exact conversation or prompt")
            content_splits[sha] = row["split"]
    reviews = _review_rows(reviews_path, rows)
    tokenizer = load_native_tokenizer(model_assets)
    reports, queue, exports, lineage = [], [], {"train": [], "dev": []}, []
    for row in rows:
        sha, review = digest(row), reviews.get(row["id"], {})
        if review and review.get("record_sha256") != sha:
            raise ValueError("stale SFT record review hash")
        bound_sources = {sid: digest(sources[sid]) for sid in row["source_document_ids"]}
        if review and review.get("source_sha256") != bound_sources:
            raise ValueError("stale/missing SFT review source hashes")
        reasons = []
        if review.get("decision") != "approved": reasons.append("answer_content_review_not_approved")
        if review.get("reviewer_type") not in {"human", "ai_assisted_human"}: reasons.append("human_review_required")
        if any(not isinstance(review.get(k), str) or not review[k].strip() for k in ("reviewer", "reason")):
            reasons.append("reviewer_and_reason_required")
        evidence = review.get("license_evidence")
        if (row["license"] not in allowed_licenses or review.get("license_decision") != "approved"
                or not isinstance(evidence, list) or not evidence
                or any(not isinstance(e, str) or not e.strip() for e in evidence)):
            reasons.append("license_review_required")
        if any(not source_eligibility[sid]["eligible"] for sid in row["source_document_ids"]):
            reasons.append("source_document_not_admitted")
        # A lower bound only: real SWIFT template encoding must still fit at launch.
        content_tokens = sum(tokenizer.count(m["content"]) for m in row["messages"])
        if content_tokens > sequence_length: reasons.append("content_exceeds_sequence_length")
        reports.append({"record_id": row["id"], "record_sha256": sha, "source_sha256": bound_sources,
                        "split": row["split"], "eligible": not reasons, "reasons": reasons,
                        "review_receipt_sha256": digest(review) if review else None,
                        "content_tokens_without_template": content_tokens})
        queue.append({"record_id": row["id"], "record_sha256": sha, "source_sha256": bound_sources,
                      "decision": "pending", "reviewer_type": "", "reviewer": "", "reason": "",
                      "license_decision": "pending", "license_evidence": []})
        if mode == "preview" or not reasons:
            item = {"messages": row["messages"]}
            lineage.append({"record_id": row["id"], "record_sha256": sha, "source_sha256": bound_sources,
                            "split": row["split"], "row": len(exports[row["split"]]), "messages_sha256": digest(item)})
            exports[row["split"]].append(item)
    if mode == "training" and any(not exports[s] for s in exports):
        raise ValueError("SFT training requires approved train and dev records and their sources")
    for key, path in bound_paths.items():
        if path is None:
            continue
        try:
            unchanged = file_hash(path) == source_hashes[key]
        except OSError as exc:
            raise ValueError(f"source file changed during SFT preparation: {key}") from exc
        if not unchanged:
            raise ValueError(f"source file changed during SFT preparation: {key}")
    out = new_output(output_dir)
    artifacts = {}
    values = {"train": exports["train"], "dev": exports["dev"], "admission": reports,
              "reviews_pending": queue, "lineage": lineage, "source_admission": source_admission,
              "source_splits": assignments}
    for key, records in values.items():
        path = out / f"{key}.jsonl"
        write_jsonl(path, records)
        artifacts[key] = {"path": path.name, "sha256": file_hash(path)}
    manifest = {"schema_version": "sft-prepared-v1", "mode": mode, "training_ready": mode == "training",
                **source_hashes,
                "model_assets": str(Path(model_assets).resolve()), "tokenizer": tokenizer.metadata,
                "sequence_length": sequence_length, "artifacts": artifacts,
                "counts": {s: len(v) for s, v in exports.items()}, "approved_records": sum(r["eligible"] for r in reports),
                "test_records_exported": 0,
                "limitations": ["Content-only token counts exclude chat-template tokens; real template verification is mandatory.",
                                "Human receipts are supplied claims, not independently verified identities."]}
    write_json(out / "prepare_manifest.json", manifest)
    return manifest


def build_sft(prepared_dir, recipe, output_dir, *, init_adapter=None):
    from .launch import _locked_assets, _artifact, _argv, _adapter_provenance
    from .schema import validate_recipe, recipe_sha256
    validate_recipe(recipe)
    root = Path(prepared_dir).resolve()
    path = root / "prepare_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sft-prepared-v1":
        raise ValueError("SFT recipe requires an SFT prepared manifest")
    assets = Path(manifest["model_assets"]).resolve()
    lock, lock_sha = _locked_assets(assets)
    if recipe["model"] != {"repo": lock["repository"], "revision": lock["revision"]} or manifest["tokenizer"]["model_lock_sha256"] != lock_sha:
        raise ValueError("SFT tokenizer/model lock mismatch")
    if manifest["sequence_length"] != recipe["parameters"]["max_length"]:
        raise ValueError("SFT recipe max_length differs from preparation")
    for key in manifest["artifacts"]: _artifact(root, manifest, key)
    admission_rows = read_jsonl(_artifact(root, manifest, "admission"))
    admission = {r["record_id"]: r for r in admission_rows}
    lineage = read_jsonl(_artifact(root, manifest, "lineage"))
    if len(admission) != len(admission_rows) or len({x["record_id"] for x in lineage}) != len(lineage):
        raise ValueError("duplicate SFT admission or lineage id")
    blockers = []
    if manifest.get("mode") != "training" or manifest.get("training_ready") is not True:
        blockers.append("preview_or_unapproved_data")
    datasets = {s: str(_artifact(root, manifest, s)) for s in ("train", "dev")}
    seen_prompts = {}
    split_rows = read_jsonl(_artifact(root, manifest, "source_splits"))
    source_rows = read_jsonl(_artifact(root, manifest, "source_admission"))
    source_splits = {x["document_id"]: x["split"] for x in split_rows}
    sources = {x["document_id"]: x for x in source_rows}
    if len(source_splits) != len(split_rows) or len(sources) != len(source_rows):
        raise ValueError("duplicate SFT source admission or split record")
    if manifest.get("test_records_exported") != 0 or any(x["split"] not in datasets for x in lineage):
        raise ValueError("SFT test/unknown split export rejected")
    for split, dataset in datasets.items():
        rows = read_jsonl(dataset)
        expected = sorted((x for x in lineage if x["split"] == split), key=lambda x: x["row"])
        if len(rows) != len(expected) or [x["row"] for x in expected] != list(range(len(rows))):
            raise ValueError("SFT lineage row mismatch")
        if not rows: blockers.append(f"empty_{split}_dataset")
        for row, item in zip(rows, expected):
            if set(row) != {"messages"}: raise ValueError("SFT export must contain only messages")
            validate_messages(row["messages"])
            if digest(row) != item["messages_sha256"]: raise ValueError("SFT conversation hash mismatch")
            admitted = admission.get(item["record_id"], {})
            if (admitted.get("record_sha256") != item["record_sha256"] or admitted.get("split") != split
                    or admitted.get("source_sha256") != item["source_sha256"]):
                raise ValueError("SFT admission/lineage mismatch")
            if not admitted.get("eligible"): blockers.append("exported_record_not_admitted")
            if not isinstance(item["source_sha256"], dict) or not item["source_sha256"]:
                raise ValueError("SFT lineage must bind at least one source document")
            for sid, sha in item["source_sha256"].items():
                if source_splits.get(sid) != split or sources.get(sid, {}).get("document_sha256") != sha:
                    raise ValueError("SFT source lineage/split mismatch")
                if not sources[sid].get("eligible"): blockers.append("source_document_not_admitted")
            prompt = digest(row["messages"][:-1])
            if prompt in seen_prompts and seen_prompts[prompt] != split: raise ValueError("SFT train/dev prompt overlap")
            seen_prompts[prompt] = split
    checkpoint = str(Path(output_dir).resolve() / "checkpoints")
    result = {"schema_version": "cpt-launch-plan-v1", "recipe": recipe, "recipe_sha256": recipe_sha256(recipe),
              "prepared_dir": str(root), "prepare_manifest_sha256": file_hash(path), "model_assets": str(assets),
              "model_lock_sha256": lock_sha, "strategy": "sft", "datasets": datasets,
              "dataset_sha256": {s: file_hash(p) for s, p in datasets.items()},
              "output_dir": str(Path(output_dir).resolve()), "checkpoint_dir": checkpoint,
              "argv": _argv(recipe, datasets, checkpoint), "blockers": sorted(set(blockers)), "training_ready": not blockers,
              "note": "SFT assistant-only loss; real SWIFT template verification required. No claim of CPT gain."}
    if init_adapter:
        provenance = _adapter_provenance(init_adapter, recipe)
        result["init_adapter"] = provenance
        result["argv"].extend(["--adapters", provenance["path"]])
    return result
