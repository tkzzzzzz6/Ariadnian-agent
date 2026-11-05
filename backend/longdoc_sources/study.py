"""Reproducible cross-source diagnostics, review queues and extractive compilations.

The outputs are explicitly controlled or unreviewed. This module never invents
human labels, writes training approval, or treats an extractive compilation as
reasoning supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from .__main__ import digest, write_json, write_jsonl


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def balanced(documents: list[dict], seed: int = 17) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for doc in documents:
        groups[doc["language"], doc["domain"], doc["kind"], doc["split"]].append(doc)
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda d: d["id"])
        rng.shuffle(groups[key])
    ordered = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def eligible(doc: dict) -> list[dict]:
    return [b for b in doc["blocks"] if b["type"] == "paragraph" and 100 <= len(b["text"]) <= 1800
            and not b["text"].lstrip().startswith(("..", ":", "<!--", "!["))]


def evidence(doc: dict, block: dict, length: int = 220) -> dict:
    return {"document_id": doc["id"], "block_id": block["id"], "quote": block["text"][:length]}


def extra_tasks(documents: list[dict], limit: int = 180) -> list[dict]:
    ordered = balanced(documents)
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for doc in ordered:
        if eligible(doc):
            pool[doc["domain"], doc["language"], doc["split"]].append(doc)
    tasks = []
    for doc in ordered:
        paras = eligible(doc)
        if not paras:
            continue
        base = {"language": doc["language"], "split": doc["split"], "answerable": True,
                "annotation_status": "controlled_synthetic", "diagnostic_only": True, "reference_answer": ""}
        partners = [d for d in pool[doc["domain"], doc["language"], doc["split"]] if d["family_id"] != doc["family_id"]]
        if partners:
            other = partners[0]
            ev = [evidence(doc, paras[len(paras) // 2]), evidence(other, eligible(other)[len(eligible(other)) // 2])]
            tasks.append({**base, "id": "crossdoc-" + digest(doc["id"].encode())[:16],
                          "task_type": "cross_document_evidence", "document_ids": [doc["id"], other["id"]],
                          "question": "Find BOTH quoted passages and attribute each to its source document. This checks evidence location, not inference: " + " | ".join(e["quote"] for e in ev),
                          "evidence": ev, "label_origin": "deterministic_two_source_exact_quotes"})
        table = next((b for b in doc["blocks"] if b["type"] == "table" and len(b["text"]) > 50), None)
        if table:
            para = next((b for b in paras if b["section_path"] == table["section_path"]), paras[0])
            ev = [evidence(doc, table), evidence(doc, para)]
            tasks.append({**base, "id": "table-" + digest(doc["id"].encode())[:16], "task_type": "table_text_evidence",
                          "document_ids": [doc["id"]], "question": "Locate the table fragment AND prose fragment: " + " | ".join(e["quote"] for e in ev),
                          "evidence": ev, "label_origin": "deterministic_table_and_prose_quotes"})
        if len(tasks) >= limit:
            break
    return tasks[:limit]


def qa_candidates(documents: list[dict], limit: int = 60) -> list[dict]:
    """Section-grounded extractive candidates; independent review remains pending."""
    tasks = []
    for doc in balanced(documents):
        paras = [b for b in eligible(doc) if b["section_path"]]
        if not paras:
            continue
        block = paras[len(paras) // 2]
        section = " / ".join(block["section_path"])
        zh = doc["language"] == "zh"
        question = (f"根据《{doc['title']}》（文档 {doc['id']}）中“{section}”一节，该节具体说明了什么？请给出该节的一段原文并注明出处。" if zh else
                    f"According to the section '{section}' in '{doc['title']}' (document {doc['id']}), what does that section explain? Return one paragraph from that section and cite its source.")
        tasks.append({"id": "candidate-" + digest(doc["id"].encode())[:16], "task_type": "extractive_qa",
                      "document_ids": [doc["id"]], "question": question, "reference_answer": block["text"],
                      "reference_answers": [b["text"] for b in paras if b["section_path"] == block["section_path"]],
                      "evidence": [], "candidate_evidence_options": [[evidence(doc, b, len(b["text"]))] for b in paras if b["section_path"] == block["section_path"]],
                      "evidence_review_pending": True, "answerable": True,
                      "language": doc["language"], "split": doc["split"], "annotation_status": "ai_generated",
                      "review_status": "unreviewed", "reviewer": "", "label_origin": "template_section_extractive_candidate",
                      "limitations": ["not_human_gold", "multiple_valid_paragraphs_possible", "not_a_reasoning_benchmark"]})
        if len(tasks) >= limit:
            break
    return tasks


def compile_sources(documents: list[dict], limit: int = 20) -> list[dict]:
    """Create attributed long extractive compilations within one split only."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for doc in balanced(documents):
        groups[doc["domain"], doc["language"], doc["split"]].append(doc)
    output = []
    pairs = {key: list(zip(docs[::2], docs[1::2])) for key, docs in groups.items()}
    while any(pairs.values()):
        for key in sorted(pairs):
            if not pairs[key]:
                continue
            left, right = pairs[key].pop(0)
            if left["family_id"] == right["family_id"]:
                continue
            blocks = []
            provenance = []
            for source in (left, right):
                blocks.append({"id": f"b{len(blocks):05d}", "type": "heading", "text": source["title"], "section_path": [source["title"]], "page": None})
                for original in source["blocks"]:
                    block = {**original, "id": f"b{len(blocks):05d}", "section_path": [source["title"], *original["section_path"]],
                             "source_document_id": source["id"], "source_block_id": original["id"]}
                    blocks.append(block)
                    provenance.append({"block_id": block["id"], "source_document_id": source["id"], "source_block_id": original["id"], "source_text_sha256": digest(original["text"].encode())})
            sid = "compilation-" + digest((left["id"] + right["id"]).encode())[:16]
            output.append({"id": sid, "family_id": sid, "parent_family_ids": [left["family_id"], right["family_id"]],
                           "source_document_ids": [left["id"], right["id"]], "title": "Attributed source compilation: " + left["domain"],
                           "source_splits": {d["id"]: d["split"] for d in (left, right)},
                           "parent_family_splits": {d["family_id"]: d["split"] for d in (left, right)},
                           "source_url": left["source_url"], "source_urls": [left["source_url"], right["source_url"]],
                           "license": "mixed-source-see-provenance", "source_licenses": {d["id"]: d["license"] for d in (left, right)},
                           "revision": sid, "language": key[1], "domain": key[0], "kind": "extractive_compilation",
                           "split": key[2], "blocks": blocks, "provenance": provenance, "review_status": "unreviewed",
                           "annotation_status": "controlled_synthetic", "synthesis_method": "attributed_extractive_compilation_not_generated_reasoning",
                           "training_eligible": False, "limitations": ["same_domain_does_not_guarantee_semantic_coherence", "no_claim_of_generated_long_distance_reasoning"]})
            if len(output) >= limit:
                return output
    return output


def build(input_path: Path, output: Path) -> dict:
    docs = load(input_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("study output must be empty; never overwrite review data")
    output.mkdir(parents=True, exist_ok=True)
    extra = extra_tasks(docs)
    candidates = qa_candidates(docs)
    synth = compile_sources(docs)
    review = [{"document_id": d["id"], "source_url": d["source_url"], "revision": d["revision"],
               "sample_block_ids": [b["id"] for b in d["blocks"] if b["type"] in {"table", "formula"}][:3] + [b["id"] for b in eligible(d)[:2]],
               "review_status": "unreviewed", "reviewer": "", "review_method": "", "reading_order_correct": None,
               "headings_preserved": None, "table_formula_preserved": None, "body_complete": None, "notes": ""}
              for d in balanced(docs)[:100]]
    write_jsonl(output / "additional_diagnostics.jsonl", extra)
    write_jsonl(output / "qa_candidates.jsonl", candidates)
    write_jsonl(output / "extractive_compilations.jsonl", synth)
    write_jsonl(output / "structure_review_queue.jsonl", review)
    manifest = {"input_sha256": digest(input_path.read_bytes()), "implementation_sha256": digest(Path(__file__).read_bytes()), "additional_diagnostics": len(extra),
                "additional_task_types": dict(Counter(t["task_type"] for t in extra)), "qa_candidates": len(candidates),
                "extractive_compilations": len(synth), "structure_review_items": len(review), "human_verified": 0,
                "limitations": ["All natural QA candidates and document reviews await independent review.", "Compilations are source-linked extractive data, not generated reasoning or approved training data."]}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.documents, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
