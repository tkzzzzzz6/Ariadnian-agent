"""Build local, pending human-review packets and lexical overlap diagnostics.

No model calls, approval decisions, split reassignment or training occurs here.
Run from experiments/ms_swift: python -m cpt.review --help.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import html
import itertools
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlparse

from .core import _documents, audit, digest, file_hash, new_output, read_jsonl, write_json, write_jsonl


LEXICAL = re.compile(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", re.UNICODE)
MASK = (1 << 64) - 1
ROLL_BASE = 1000003


def lexical_tokens(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Case/NFKC normalize matched tokens; retain original code-point offsets."""
    matches = list(LEXICAL.finditer(text))
    return ([unicodedata.normalize("NFKC", m.group()).casefold() for m in matches],
            [m.span() for m in matches])


def fingerprints(tokens: list[str], ngram: int = 20, window: int = 12) -> dict[int, int]:
    """Rightmost-minimum winnowing; one representative position per hash.

    Hash equality is only a candidate; compare token sequences before reporting.
    All windows are scanned. Repeated positions of one identical shingle in the
    same document collapse to one representative, so counts are not occurrences.
    """
    if type(ngram) is not int or ngram < 2 or type(window) is not int or window < 1:
        raise ValueError("ngram >= 2 and window >= 1 are required")
    if len(tokens) < ngram:
        return {}
    vocabulary = {t: int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "big") for t in set(tokens)}
    values = [vocabulary[t] for t in tokens]
    power = pow(ROLL_BASE, ngram - 1, 1 << 64)
    rolling = 0
    for value in values[:ngram]:
        rolling = (rolling * ROLL_BASE + value) & MASK
    queue = deque()
    selected = {}
    windows = min(window, len(tokens) - ngram + 1)
    for index in range(len(tokens) - ngram + 1):
        if index:
            rolling = (((rolling - values[index - 1] * power) & MASK) * ROLL_BASE + values[index + ngram - 1]) & MASK
        while queue and queue[-1][1] >= rolling:
            queue.pop()
        queue.append((index, rolling))
        while queue[0][0] <= index - windows:
            queue.popleft()
        if index >= windows - 1:
            position, value = queue[0]
            selected.setdefault(value, position)
    return selected


@dataclass
class TextRecord:
    key: str
    source_id: str
    split: str
    category: str
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int]]


def text_record(key: str, source_id: str, split: str, category: str, text: str) -> TextRecord:
    tokens, offsets = lexical_tokens(text)
    return TextRecord(key, source_id, split, category, text, tokens, offsets)


def compare_records(records: list[TextRecord], *, ngram: int = 20, window: int = 12,
                    examples_per_pair: int = 3) -> tuple[list[dict], dict]:
    if len({r.key for r in records}) != len(records):
        raise ValueError("overlap record keys must be unique")
    if type(examples_per_pair) is not int or examples_per_pair < 1:
        raise ValueError("examples_per_pair must be positive")
    index = defaultdict(list)
    for i, row in enumerate(records):
        for value, position in fingerprints(row.tokens, ngram, window).items():
            index[value].append((i, position))
    pairs = {}
    # Include every distinct verified shared fingerprint; cap only displayed
    # evidence examples, never the detection pass or number of candidate pairs.
    for value, owners in index.items():
        for (ai, ap), (bi, bp) in itertools.combinations(owners, 2):
            a, b = records[ai], records[bi]
            if a.category == b.category == "benchmark":
                continue
            if a.category == b.category == "corpus" and a.split == b.split:
                continue
            if a.tokens[ap:ap + ngram] != b.tokens[bp:bp + ngram]:
                continue
            key = (ai, bi)
            pair = pairs.setdefault(key, {
                "left_id": a.source_id, "right_id": b.source_id,
                "left_kind": a.category, "right_kind": b.category,
                "left_split": a.split, "right_split": b.split,
                "comparison": "public_benchmark" if "benchmark" in {a.category, b.category} else "cross_split",
                "shared_distinct_fingerprints": 0,
                "evidence": [], "review_status": "pending", "automatic_training_exclusion": False,
            })
            pair["shared_distinct_fingerprints"] += 1
            if len(pair["evidence"]) >= examples_per_pair:
                continue
            # Evidence is exactly the verified ngram. Do not imply it is the
            # longest common span or reconstruct frequency from sampled hashes.
            left = (a.offsets[ap][0], a.offsets[ap + ngram - 1][1])
            right = (b.offsets[bp][0], b.offsets[bp + ngram - 1][1])
            pair["evidence"].append({
                "normalized_tokens": ngram,
                "left_start_char": left[0], "left_end_char": left[1],
                "right_start_char": right[0], "right_end_char": right[1],
                "left_excerpt": a.text[left[0]:left[1]], "right_excerpt": b.text[right[0]:right[1]],
                "normalized_sha256": digest(a.tokens[ap:ap + ngram]),
            })
    rows = sorted(pairs.values(), key=lambda p: (p["comparison"], p["left_id"], p["right_id"]))
    summary = {
        "method": "lexical_winnowing_v1", "ngram_tokens": ngram, "window_shingles": window,
        "guaranteed_detectable_exact_normalized_span_tokens": ngram + window - 1,
        "token_definition": "Individual U+3400-U+9FFF Han characters; other Unicode alphanumeric runs. NFKC and casefold per token. Punctuation/whitespace ignored. Not Gemma tokens.",
        "records": len(records), "normalized_tokens": sum(len(r.tokens) for r in records),
        "distinct_index_hashes": len(index), "candidate_pairs": len(rows),
        "pairs_by_comparison": dict(Counter(p["comparison"] for p in rows)),
        "pairs_by_split": dict(Counter(f"{p['left_split']}:{p['right_split']}" for p in rows)),
        "stored_evidence_per_pair_limit": examples_per_pair,
        "candidate_pair_or_scan_limit": None,
        "hash_matches_verified_by_token_equality": True,
        "winnowing_detection_guarantee_assumptions": "Aligned normalized tokens and no 64-bit fingerprint collisions; different shingles sharing a hash can collapse to one representative.",
        "limitations": [
            "Candidate overlap can be quotations, licenses or boilerplate; human disposition is required.",
            "Not semantic deduplication or proof of uncontaminated data; shorter spans can be missed.",
            "Only supplied benchmark contexts are scanned; neither the full benchmark nor unknown base-model pretraining data are covered.",
            "Within-split corpus pairs are intentionally not compared. Counts are distinct selected shingles, not all repeated occurrences.",
            "Evidence offsets refer to the cleaned block text joined by two newlines, or to the supplied benchmark context.",
        ],
    }
    return rows, summary


def _benchmark_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("benchmark input is empty")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("benchmark records need nonempty IDs")
        if row["id"] in seen:
            raise ValueError("duplicate benchmark ID")
        seen.add(row["id"])
        if row.get("evaluation_only") is not True or not isinstance(row.get("context"), str) or not row["context"].strip():
            raise ValueError("benchmark records require evaluation_only=true and nonempty context")
        if "raw_record_sha256" in row:
            if row.get("raw_record_hash_kind") != "canonical_JSON_values" or digest(row.get("raw_record")) != row["raw_record_sha256"]:
                raise ValueError(f"{row['id']}: benchmark raw record hash mismatch")
            if row["context"] != row["raw_record"].get("context"):
                raise ValueError(f"{row['id']}: benchmark context differs from raw source")
    return rows


def _body(doc: dict) -> str:
    return "\n\n".join(b.get("cleaned_text", b["text"]) for b in doc["blocks"])


def select_candidates(docs: list[dict], overlaps: list[dict], *, per_stratum: int = 3,
                      dev_per_stratum: int = 1, seed: int = 17) -> tuple[list[dict], dict]:
    for value in (per_stratum, dev_per_stratum):
        if type(value) is not int or value < 1:
            raise ValueError("per-stratum train/dev quotas must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    flags = Counter()
    for row in overlaps:
        if row["left_kind"] == "corpus":
            flags[row["left_id"]] += 1
        if row["right_kind"] == "corpus":
            flags[row["right_id"]] += 1
    groups = defaultdict(list)
    for doc in docs:
        if doc["split"] == "test":
            continue
        groups[(doc["split"], doc["domain"], doc["language"])].append(doc)
    selected = []
    strata = []
    for key, candidates in sorted(groups.items()):
        def rank(d):
            evidence = isinstance(d.get("license_evidence"), list) and bool(d["license_evidence"])
            return (flags[d["id"]], not bool(d.get("license")), not evidence,
                    hashlib.sha256(f"{seed}:{d['id']}".encode()).hexdigest())
        ordered = sorted(candidates, key=rank)
        quota = per_stratum if key[0] == "train" else dev_per_stratum
        chosen = ordered[:quota]
        selected.extend(chosen)
        strata.append({"split": key[0], "domain": key[1], "language": key[2],
                       "available": len(ordered), "selected": len(chosen), "quota": quota})
    return selected, {"seed": seed, "strata": strata,
                      "ranking": "Fewer lexical overlap candidates, then present license metadata/evidence, then seeded ID hash. Not a quality/license approval or a representative statistical sample.",
                      "test_documents_selected": 0}


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _url(value) -> str:
    value = str(value or "")
    if urlparse(value).scheme not in {"https", "http"}:
        return _escape(value)
    return f'<a href="{_escape(value)}" rel="noreferrer">{_escape(value)}</a>'


def _page(title: str, body: str) -> str:
    return ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            f'<title>{_escape(title)}</title><style>body{{font:16px/1.65 system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem}}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:1rem}table{border-collapse:collapse}'
            'th,td{border:1px solid #ddd;padding:.5rem}small{color:#555}.warning{background:#fff2d5;padding:1rem}</style>'
            f'<h1>{_escape(title)}</h1>{body}</html>')


def _write_pages(output: Path, selected: list[dict], overlaps: list[dict]) -> None:
    pages = output / "documents"
    pages.mkdir()
    rows = []
    for doc in selected:
        name = hashlib.sha256(doc["id"].encode()).hexdigest()[:24] + ".html"
        hits = [o for o in overlaps if doc["id"] in (o["left_id"], o["right_id"])]
        rows.append(f'<tr><td><a href="documents/{name}">{_escape(doc["id"])}</a></td>'
                    f'<td>{_escape(doc["split"])}</td><td>{_escape(doc["domain"])}</td>'
                    f'<td>{_escape(doc["language"])}</td><td>{_escape(doc.get("license"))}</td><td>{len(hits)}</td></tr>')
        metadata = {k: doc.get(k) for k in ("id", "family_id", "split", "language", "domain", "kind", "license", "license_text", "license_evidence", "revision", "quality", "parser_limits", "review_status", "token_count", "tokenizer", "source_document_ids", "source_splits", "parent_family_ids", "parent_family_splits")}
        metadata["document_sha256"] = digest(doc)
        metadata["reference_token_count_is_not_gemma_budget"] = True
        body = ('<p><a href="../index.html">返回审核索引</a></p>'
                '<p class="warning">待人工审核。来源许可文本、启发式质量分数与重合候选均不是通过结论。请阅读完整正文，核对来源、第三方内容、表格/公式/代码与适用许可。</p>'
                f'<p>来源：{_url(doc.get("source_url"))}</p>'
                f'<h2>来源与元数据</h2><pre>{_escape(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre>'
                f'<h2>重合候选（{len(hits)} 对）</h2><pre>{_escape(json.dumps(hits, ensure_ascii=False, indent=2))}</pre><h2>完整清洗正文</h2>')
        for block in doc["blocks"]:
            label = f"{block['id']} | {block['type']} | {' / '.join(block.get('section_path', []))}"
            body += f'<h3>{_escape(label)}</h3><pre>{_escape(block.get("cleaned_text", block["text"]))}</pre>'
            if block.get("cleaned_text", block["text"]) != block["text"]:
                body += f'<details><summary>原始文本</summary><pre>{_escape(block["text"])}</pre></details>'
        (pages / name).write_text(_page(doc["title"], body), encoding="utf-8")
    instructions = ('<p class="warning">此包仅为待审材料，不可直接训练。全部审核字段保持 pending/空白；公开基准与 test 文档不进入候选训练导出。</p>'
                    '<ol><li>打开每篇文档，核对原文、来源许可、正文和重合候选。</li>'
                    '<li>复制 selected_reviews.pending.jsonl 为独立 reviewed.jsonl；逐项填写 decision、license_decision、reviewer_type、reviewer、reason 和 license_evidence。</li>'
                    '<li>reviewer_type 仅在真实人工复核后填写 human 或 ai_assisted_human。AI 独立审查不能填为人工。</li>'
                    '<li>reason 中记录每项重要重合候选的处置依据。跨划分共享正文/代码或翻译章节须先处理来源家族关系；公式渲染前导、许可等重复也需要明确判断。不要删除 hash/split，也不要改原始输入。</li>'
                    '<li>将凭据交给 cpt audit/prepare，仍使用完整冻结语料作为 input，明确传递允许的精确 license 标识；未批准行保持排除。</li></ol>'
                    '<p>审批并不自动等于无污染；本工具不会替审核者接受或剔除重合内容。分层候选按诊断和元数据排序，不是语义质量排名。</p>'
                    '<p><a href="overlap_candidates.jsonl">所有重合候选及原文证据</a> · <a href="family_review_candidates.jsonl">来源家族关系待审建议</a> · <a href="selection_review.jsonl">入选文档的重合计数与选择理由</a> · <a href="selected_reviews.pending.jsonl">待填写审核凭据</a></p>')
    table = '<table><tr><th>文档</th><th>划分</th><th>领域</th><th>语言</th><th>许可元数据</th><th>重合对</th></tr>' + ''.join(rows) + '</table>'
    (output / "index.html").write_text(_page("长文训练数据：第一批待审材料", instructions + table), encoding="utf-8")


def build(input_path: Path, benchmark_path: Path, output_dir: Path, *, per_stratum=3,
          dev_per_stratum=1, seed=17, ngram=20, window=12) -> dict:
    started = time.monotonic()
    input_hash, benchmark_hash = file_hash(input_path), file_hash(benchmark_path)
    docs, assignments = _documents(input_path)
    benchmark = _benchmark_rows(benchmark_path)
    # Validate parameters before claiming an output directory.
    select_candidates(docs, [], per_stratum=per_stratum, dev_per_stratum=dev_per_stratum, seed=seed)
    fingerprints([], ngram, window)
    output = new_output(output_dir)
    records = [text_record("corpus:" + d["id"], d["id"], d["split"], "corpus", _body(d)) for d in docs]
    records.extend(text_record("benchmark:" + d["id"], d["id"], "evaluation_only", "benchmark", d["context"]) for d in benchmark)
    overlaps, overlap_summary = compare_records(records, ngram=ngram, window=window)
    selected, selection = select_candidates(docs, overlaps, per_stratum=per_stratum, dev_per_stratum=dev_per_stratum, seed=seed)
    audit_manifest = audit(input_path, output / "admission")
    queue = read_jsonl(output / "admission" / "admission_review.jsonl")
    selected_ids = {d["id"] for d in selected}
    write_jsonl(output / "selected_reviews.pending.jsonl", [r for r in queue if r["document_id"] in selected_ids])
    write_jsonl(output / "candidate_documents.jsonl", selected)
    write_jsonl(output / "overlap_candidates.jsonl", overlaps)
    lookup = {d["id"]: d for d in docs}
    flags = Counter()
    for row in overlaps:
        for side in ("left", "right"):
            if row[f"{side}_kind"] == "corpus":
                flags[row[f"{side}_id"]] += 1
    selection_review = [{"document_id": d["id"], "document_sha256": digest(d), "split": d["split"],
                         "domain": d["domain"], "language": d["language"], "overlap_candidate_pairs": flags[d["id"]],
                         "selection_reason": "Within stratum quota after sorting by overlap candidate count, license metadata presence and seeded ID hash; no quality approval.",
                         "overlap_disposition": "pending" if flags[d["id"]] else "no_hit_at_current_threshold_not_proof_of_no_contamination"}
                        for d in selected]
    write_jsonl(output / "selection_review.jsonl", selection_review)
    family_candidates = []
    for row in overlaps:
        if row["comparison"] != "cross_split":
            continue
        left, right = lookup[row["left_id"]], lookup[row["right_id"]]
        same_filename = Path(left["id"].split(":", 1)[-1]).name == Path(right["id"].split(":", 1)[-1]).name
        cross_language = left["language"] != right["language"]
        family_candidates.append({"left_id": left["id"], "right_id": right["id"],
                                  "left_family_id": left["family_id"], "right_family_id": right["family_id"],
                                  "left_split": left["split"], "right_split": right["split"],
                                  "shared_distinct_fingerprints": row["shared_distinct_fingerprints"],
                                  "hint": "cross_language_same_filename_shared_lexical_content" if cross_language and same_filename else "cross_split_shared_lexical_content",
                                  "decision": "pending", "merge_families": None,
                                  "instruction": "Review full sources and overlap evidence. If related family is confirmed, create a new version with source-family constraints; preserve historical splits. Do not merge all hits automatically."})
    write_jsonl(output / "family_review_candidates.jsonl", family_candidates)
    write_jsonl(output / "frozen_corpus.jsonl", [{"document_id": d["id"], "document_sha256": digest(d), "family_id": d["family_id"], "split": d["split"]} for d in docs])
    write_jsonl(output / "frozen_benchmark.jsonl", [{"id": r["id"], "record_sha256": digest(r), "context_sha256": hashlib.sha256(r["context"].encode()).hexdigest(), "evaluation_only": True} for r in benchmark])
    _write_pages(output, selected, overlaps)
    files = {str(p.relative_to(output)).replace("\\", "/"): {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file()}
    # Detect concurrent source changes rather than freeze mismatched provenance.
    if input_hash != file_hash(input_path) or benchmark_hash != file_hash(benchmark_path):
        raise ValueError("source changed while building review packet; discard this incomplete packet")
    summary = {
        "schema_version": "training-data-review-v1", "training_ready": False, "approved_documents": 0,
        "source": {"path": str(Path(input_path).resolve()), "sha256": input_hash, "documents": len(docs),
                   "splits": dict(Counter(d["split"] for d in docs)), "domains": dict(Counter(d["domain"] for d in docs)),
                   "languages": dict(Counter(d["language"] for d in docs)), "licenses": dict(Counter(d.get("license") for d in docs))},
        "benchmark": {"path": str(Path(benchmark_path).resolve()), "sha256": benchmark_hash, "records": len(benchmark), "evaluation_only": True,
                      "coverage": "Only the frozen supplied contexts; not the complete public benchmark."},
        "selected": {"documents": len(selected), "splits": dict(Counter(d["split"] for d in selected)),
                     "domains": dict(Counter(d["domain"] for d in selected)), "languages": dict(Counter(d["language"] for d in selected)),
                     "reference_tokens": sum(d.get("token_count", 0) for d in selected), "native_gemma_token_budget_verified": False,
                     "documents_with_overlap_candidates": sum(bool(flags[d["id"]]) for d in selected),
                     "documents_without_overlap_candidates": sum(not flags[d["id"]] for d in selected),
                     "flagged_documents_by_split": dict(Counter(d["split"] for d in selected if flags[d["id"]])),
                     "missing_dev_domains": sorted({d["domain"] for d in docs} - {d["domain"] for d in docs if d["split"] == "dev"}),
                     "selection": selection},
        "overlap_diagnostic": {**overlap_summary,
                               "public_benchmark_candidate_pairs": sum(r["comparison"] == "public_benchmark" for r in overlaps),
                               "corpus_cross_split_candidate_pairs": sum(r["comparison"] == "cross_split" for r in overlaps),
                               "corpus_documents_with_candidates": len(flags),
                               "flagged_corpus_documents_by_split": dict(Counter(lookup[did]["split"] for did in flags)),
                               "cross_language_same_filename_family_candidates": sum(r["hint"] == "cross_language_same_filename_shared_lexical_content" for r in family_candidates)},
        "admission": audit_manifest, "original_splits_preserved": True, "split_assignments": len(assignments),
        "artifacts": files, "elapsed_seconds": round(time.monotonic() - started, 3),
        "implementation_sha256": file_hash(Path(__file__)),
        "next_steps": ["Human content/license review and overlap disposition", "Run cpt audit with copied completed receipts against the full unchanged source", "Create a new training export and recheck native loss-token budgets"],
    }
    write_json(output / "review_manifest.json", summary)
    return summary


def aggregate_summary(manifest: dict, packet_dir: Path) -> dict:
    """Small versionable record; text, document IDs and human receipts stay local."""
    out = {key: manifest[key] for key in ("schema_version", "training_ready", "approved_documents", "source", "benchmark", "selected", "overlap_diagnostic", "admission", "original_splits_preserved", "split_assignments", "elapsed_seconds", "implementation_sha256", "next_steps")}
    out["packet_manifest_sha256"] = file_hash(Path(packet_dir) / "review_manifest.json")
    out["local_packet"] = str(Path(packet_dir).resolve())
    out["artifacts"] = {name: spec for name, spec in manifest["artifacts"].items() if not name.startswith("documents/")}
    out["document_pages"] = sum(name.startswith("documents/") for name in manifest["artifacts"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Build pending review pages and verified lexical overlap candidates; never approves or trains")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--per-stratum", type=int, default=3)
    p.add_argument("--dev-per-stratum", type=int, default=1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--ngram", type=int, default=20)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--summary-output", type=Path, help="Optional new aggregate JSON file for version control; never contains document text")
    args = p.parse_args(argv)
    try:
        if args.summary_output and args.summary_output.exists():
            raise ValueError("summary output already exists; use a new path")
        result = build(args.input, args.benchmark, args.output, per_stratum=args.per_stratum,
                       dev_per_stratum=args.dev_per_stratum, seed=args.seed, ngram=args.ngram, window=args.window)
        if args.summary_output:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            with args.summary_output.open("x", encoding="utf-8") as stream:
                json.dump(aggregate_summary(result, args.output), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
    except (ValueError, OSError, TypeError) as exc:
        p.exit(2, f"Data review: {exc}\n")
    # Full content/paths remain in the local packet; CLI emits a small summary.
    print(json.dumps({"manifest": str((args.output / "review_manifest.json").resolve()),
                      "documents": result["source"]["documents"], "selected": result["selected"]["documents"],
                      "candidate_pairs": result["overlap_diagnostic"]["candidate_pairs"],
                      "approved_documents": 0, "training_ready": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
