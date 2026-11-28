"""Build paired review candidates from explicitly authored, ignored QA specs.

This module never writes an approval or a training export. Original corpus rows
are copied unchanged. Evidence anchors are exact block substrings, not generated
summaries; long contexts contain distinct original blocks in document order.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
from pathlib import Path

from cpt.assets import load_native_tokenizer
from cpt.core import digest, file_hash, new_output, read_jsonl, write_json, write_jsonl

DIRECTIONS = {"evidence_local_qa", "multihop", "crossdoc", "table_condition"}
INSTRUCTION = "Answer the question using only the supplied source excerpts. Explain necessary reasoning and cite the relevant [D#:block] evidence. Do not infer facts absent from the excerpts. Respond in the question's language."


def block_text(block):
    return block.get("cleaned_text", block.get("text", ""))


def render_context(docs, selected, quotes=None):
    parts = []
    for n, doc in enumerate(docs, 1):
        parts.append(f"Document D{n}: {doc['title']}\nSource ID: {doc['id']}")
        previous = -1
        for index in sorted(selected[doc["id"]]):
            if previous >= 0 and index != previous + 1:
                parts.append("[Non-selected source blocks omitted]")
            block = doc["blocks"][index]
            text = (quotes or {}).get((doc["id"], index), block_text(block))
            parts.append(f"[D{n}:{block['id']}]\n{text}")
            previous = index
    return "\n\n".join(parts)


def prompt(context, question):
    return f"{INSTRUCTION}\n\n<source_excerpts>\n{context}\n</source_excerpts>\n\nQuestion: {question}"


def select_long(docs, evidence, question, count, target=5200, maximum=7800):
    selected = {d["id"]: set(evidence[d["id"]]) for d in docs}
    # First include the span between anchors when that complete span fits.
    spans = {sid: set(range(min(v), max(v) + 1)) for sid, v in selected.items()}
    if count(prompt(render_context(docs, spans), question)) <= maximum:
        selected = spans
    total = count(prompt(render_context(docs, selected), question))
    if total > maximum:
        raise ValueError("evidence blocks alone exceed long-context maximum")
    remaining = []
    for doc in docs:
        anchors = evidence[doc["id"]]
        for i in range(len(doc["blocks"])):
            if i not in selected[doc["id"]]:
                remaining.append((min(abs(i - j) for j in anchors), i, doc["id"]))
    for _, i, sid in sorted(remaining):
        if total >= target:
            break
        selected[sid].add(i)
        candidate = count(prompt(render_context(docs, selected), question))
        if candidate > maximum:
            selected[sid].remove(i)
        else:
            total = candidate
    if total < 4096:
        raise ValueError(f"real unique source material too short for long condition: {total}")
    return selected, total


def connected_families(specs, sources):
    parent = {d["family_id"]: d["family_id"] for d in sources.values()}
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for row in specs:
        families = [sources[s]["family_id"] for s in row["source_document_ids"]]
        for family in families[1:]:
            parent[root(family)] = root(families[0])
    groups = collections.defaultdict(set)
    for family in parent:
        groups[root(family)].add(family)
    return {f: "component-" + digest(sorted(g))[:16] for g in groups.values() for f in g}


def pending_answer(row, sources):
    return {"record_id": row["id"], "record_sha256": digest(row),
            "source_sha256": {s: digest(sources[s]) for s in row["source_document_ids"]},
            "decision": "pending", "reviewer_type": "", "reviewer": "", "reason": "",
            "license_decision": "pending", "license_evidence": []}


STYLE = "body{font:16px/1.6 system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 20px;color:#16304a;background:#f6f8fb}article,section{background:white;padding:24px;margin:20px 0;border:1px solid #d5dfe9;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}a{color:#1268a3}.pending{color:#a54813;font-weight:700}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd8e3;text-align:left}summary{cursor:pointer;font-weight:700}small{overflow-wrap:anywhere}"


def write_review(out, docs, pairs, arms, summary):
    h = html.escape
    head = f'<!doctype html><meta charset="utf-8"><style>{STYLE}</style>'
    (out / "sources").mkdir(exist_ok=True)
    pages = {}
    for doc in docs:
        name = digest(doc["id"])[:16] + ".html"
        pages[doc["id"]] = name
        body = head + f'<h1>{h(doc["title"])}</h1><p class="pending">Source and license review PENDING</p><p>{h(doc["id"])}</p><p>Original split: {h(doc["split"])} · {h(doc["license"])}</p><small>Original document SHA256: {digest(doc)}</small><p><a href="{h(doc["source_url"], quote=True)}">Original source</a></p><pre>{h(json.dumps({k: doc.get(k) for k in ["family_id", "revision", "license_evidence", "license_text", "lineage"]}, ensure_ascii=False, indent=2))}</pre>'
        for b in doc["blocks"]:
            body += f'<section id="{h(b["id"])}"><b>{h(b["id"])} · {h(b["type"])}</b><pre>{h(block_text(b))}</pre></section>'
        (out / "sources" / name).write_text(body, encoding="utf-8")
    body = head + '<h1>长短配对 SFT：逐条人工审阅包</h1><p class="pending">AI-authored candidates · 全部待人工审核 · 不可用于训练</p><p>每条短、长条件保持同一个问题、答案及来源。短上下文为精确证据摘录；长上下文为原始语料中的不同段落，保留顺序及省略标识。没有复制填充。答案及证据正确性尚未获独立人工验证。</p><p><a href="review_focus.html">先阅读本轮审阅重点与数据局限</a></p>'
    body += f'<pre>{h(json.dumps(summary, ensure_ascii=False, indent=2))}</pre><p>打开每个来源页面可见完整原文和许可证据；展开每条的两个上下文可检查实际训练输入。原文可能含排版/公式错误，不能把“忠于来源”误当成事实正确。</p><ol>'
    for pair in pairs:
        body += f'<li><a href="#{h(pair["id"])}">{h(pair["id"])} · {h(pair["split"])} · {h(pair["direction"])} · {h(pair["question"])}</a></li>'
    body += '</ol>'
    for i, pair in enumerate(pairs):
        body += f'<article id="{h(pair["id"])}"><h2>{h(pair["id"])}</h2><p class="pending">Answer/content, source, license: PENDING</p><p>{h(pair["split"])} · {h(pair["direction"])} · {h(pair["task_subtype"])} · short {pair["short_content_tokens"]} / long {pair["long_content_tokens"]} native content tokens</p><h3>问题</h3><pre>{h(pair["question"])}</pre><h3>候选答案</h3><pre>{h(pair["answer"])}</pre><h3>推理需求及证据</h3><pre>{h(pair["rationale"])}</pre>'
        for e in pair["evidence"]:
            body += f'<section><a href="sources/{pages[e["document_id"]]}#{h(e["block_id"])}">{h(e["document_id"])} · {h(e["block_id"])}</a><pre>{h(e["quote"])}</pre></section>'
        for arm in ("short", "long"):
            body += f'<details><summary>{arm} 完整上下文、问题及答案</summary><pre>{h(json.dumps(arms[arm][i]["messages"], ensure_ascii=False, indent=2))}</pre></details>'
        body += f'<small>short record SHA256 {pair["short_sha256"]}<br>long record SHA256 {pair["long_sha256"]}</small></article>'
    (out / "index.html").write_text(body, encoding="utf-8")
    details = [
        "检查每个答案的事实、计算、条件、限制和引用；字符串存在不代表答案正确。",
        "来源页面展示完整文档和许可证据，内容批准与内部训练用途许可均须实际人工确认。",
        f"当前 {summary['counts'].get('train', 0)} train / {summary['counts'].get('dev', 0)} dev；目标128/32，缺额 {summary['deficit_counts']}。",
        f"各split语言分布：{summary['languages_by_split']}；跨文档连接来源族：{summary['crossdoc_families']}。",
        "连接来源族按章节、文件、DOI并合并跨文档边；不等于整本书或整个软件仓库留出。",
        "multi_evidence_comparison 表示比较或汇总，不自动等于多跳推理；请检查每份证据是否必要。",
        "短输入保留精确证据，长输入添加真实不同来源块；输入token与FLOPs没有配平。",
        "当前长度为固定原生tokenizer消息内容计数，实际SWIFT完整模板仍须另行核验。",
    ]
    focus = head + '<h1>本轮人工审阅重点</h1><p class="pending">候选检查不等于人工批准。</p><ul>'
    focus += ''.join('<li>' + h(item) + '</li>' for item in details)
    focus += f'</ul><p><a href="index.html">返回 {len(pairs)} 条完整审阅包</a></p>'
    (out / "review_focus.html").write_text(focus, encoding="utf-8")


def build(corpus, specs_path, quarantine, assets, output, aggregate=None):
    corpus, specs_path, quarantine = map(Path, (corpus, specs_path, quarantine))
    all_docs = {d["id"]: d for d in read_jsonl(corpus)}
    specs = read_jsonl(specs_path)
    if not specs:
        raise ValueError("authored specs required")
    banned = {row[k] for row in read_jsonl(quarantine) for k in ("left_id", "right_id")}
    selected_ids = {sid for row in specs for sid in row["source_document_ids"]}
    if selected_ids & banned:
        raise ValueError("quarantined overlap source selected")
    sources = {sid: all_docs[sid] for sid in sorted(selected_ids)}
    components = connected_families(specs, sources)
    tokenizer = load_native_tokenizer(assets)
    seen_ids, seen_qa, family_splits, family_counts = set(), set(), {}, collections.Counter()
    arms, pairs = {"short": [], "long": []}, []
    for spec in specs:
        if spec["id"] in seen_ids or spec["direction"] not in DIRECTIONS:
            raise ValueError("duplicate pair id or unknown direction")
        qa_key = tuple(' '.join(spec[k].split()).casefold() for k in ('question', 'answer'))
        if qa_key in seen_qa:
            raise ValueError("duplicate question/answer cannot inflate paired coverage")
        seen_qa.add(qa_key)
        seen_ids.add(spec["id"])
        docs = [sources[sid] for sid in spec["source_document_ids"]]
        split = docs[0]["split"]
        if split not in ("train", "dev") or any(d["split"] != split for d in docs):
            raise ValueError("source split conflict or test source selected")
        if any(d.get("benchmark") or d.get("evaluation_only") for d in docs):
            raise ValueError("benchmark/evaluation source selected")
        family = components[docs[0]["family_id"]]
        if family in family_splits and family_splits[family] != split:
            raise ValueError("connected source family crosses splits")
        family_splits[family] = split
        family_counts[family] += 1
        if family_counts[family] > 4:
            raise ValueError("more than four QA per connected source family")
        evidence, quotes, required = [], {}, {d["id"]: set() for d in docs}
        for entry in spec["evidence"]:
            sid, index = entry["document_id"], entry["block_index"]
            if sid not in required:
                raise ValueError("evidence outside declared sources")
            b = sources[sid]["blocks"][index]
            text = block_text(b)
            quote = entry.get("quote", text)
            if not quote or quote not in text or (sid, index) in quotes:
                raise ValueError("missing/non-exact/duplicate evidence quote")
            quotes[sid, index] = quote
            required[sid].add(index)
            start = text.index(quote)
            evidence.append({"document_id": sid, "block_id": b["id"], "block_index": index,
                             "start_char": start, "end_char": start + len(quote), "quote": quote,
                             "block_sha256": digest(b), "document_sha256": digest(sources[sid])})
        if any(not v for v in required.values()):
            raise ValueError("every source must provide necessary evidence")
        if spec["direction"] == "crossdoc" and len(docs) < 2:
            raise ValueError("crossdoc needs multiple sources")
        if spec["direction"] == "table_condition" and not any(sources[e["document_id"]]["blocks"][e["block_index"]]["type"] == "table" for e in evidence):
            raise ValueError("table_condition requires an actual text table")
        short_prompt = prompt(render_context(docs, required, quotes), spec["question"])
        short_tokens = tokenizer.count(short_prompt) + tokenizer.count(spec["answer"])
        if short_tokens > 1950:
            raise ValueError(f'{spec["id"]}: short content leaves insufficient template room ({short_tokens})')
        chosen, long_input = select_long(docs, required, spec["question"], tokenizer.count,
                                        target=spec.get("target_long_tokens", 5200))
        long_prompt = prompt(render_context(docs, chosen), spec["question"])
        long_tokens = long_input + tokenizer.count(spec["answer"])
        if long_tokens > 8000:
            raise ValueError("long content leaves insufficient template room")
        licenses = sorted({d["license"] for d in docs})
        license_id = licenses[0] if len(licenses) == 1 else " AND ".join(f"({x})" for x in licenses)
        for arm, content in (("short", short_prompt), ("long", long_prompt)):
            arms[arm].append({"id": spec["id"], "split": split, "source_document_ids": spec["source_document_ids"],
                              "license": license_id, "kind": "reviewed_synthetic_qa",
                              "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": spec["answer"]}]})
        pairs.append({"id": spec["id"], "split": split, "direction": spec["direction"], "language": spec["language"],
                      "task_subtype": spec.get("task_subtype", spec["direction"]),
                      "question": spec["question"], "answer": spec["answer"], "rationale": spec["rationale"],
                      "source_document_ids": spec["source_document_ids"], "source_families": [d["family_id"] for d in docs],
                      "connected_family_id": family, "evidence": evidence, "long_blocks": {sid: sorted(v) for sid,v in chosen.items()},
                      "short_content_tokens": short_tokens, "long_content_tokens": long_tokens,
                      "long_input_content_tokens": long_input, "native_chat_tokens_verified": False,
                      "short_sha256": digest(arms["short"][-1]), "long_sha256": digest(arms["long"][-1]),
                      "annotation_status": "ai_authored_pending_human_review", "training_eligible": False})
    counts = collections.Counter(p["split"] for p in pairs)
    directions = {s: dict(collections.Counter(p["direction"] for p in pairs if p["split"] == s)) for s in ("train", "dev")}
    summary = {"schema_version": "paired-review-candidates-v1", "counts": dict(counts), "directions": directions,
               "connected_families": dict(collections.Counter(family_splits.values())), "source_documents": len(sources),
               "languages": dict(collections.Counter(p["language"] for p in pairs)),
               "languages_by_split": {s: dict(collections.Counter(p["language"] for p in pairs if p["split"] == s)) for s in ("train", "dev")},
               "crossdoc_families": {s: len({p["connected_family_id"] for p in pairs if p["split"] == s and p["direction"] == "crossdoc"}) for s in ("train", "dev")},
               "domains": dict(collections.Counter(d["domain"] for d in sources.values())),
               "target_counts": {"train": 128, "dev": 32}, "deficit_counts": {"train": max(0,128-counts["train"]), "dev": max(0,32-counts["dev"])},
               "approved_pairs": 0, "training_ready": False, "test_sources_selected": 0,
               "quarantined_source_count": len(banned), "selected_quarantined_sources": 0,
               "native_chat_tokens_verified": False,
               "token_count_scope": "Pinned Gemma native tokenizer, message content only; full MS-SWIFT template verification is still mandatory.",
               "token_ranges": {arm: [min(p[f"{arm}_content_tokens"] for p in pairs), max(p[f"{arm}_content_tokens"] for p in pairs)] for arm in arms},
               "limitations": ["AI-authored candidate answers are not human-verified gold.", "Direction labels describe the intended task, not independently demonstrated reasoning validity.", "Chapter/file and DOI family isolation does not mean entire books or software repositories are held out.", "Real source windows preserve repeated framework examples already in the source; no text was duplicated to pad length.", "Task proportions and deficits are reported from actual records; target coverage is not a quality claim."]}
    out = new_output(output)
    write_jsonl(out / "sources.jsonl", list(sources.values()))
    write_jsonl(out / "pairs.jsonl", pairs)
    for arm, rows in arms.items():
        write_jsonl(out / f"{arm}.jsonl", rows)
        write_jsonl(out / f"{arm}_reviews.pending.jsonl", [pending_answer(r, sources) for r in rows])
    source_reviews = [{"document_id": d["id"], "document_sha256": digest(d), "split": d["split"], "license": d["license"],
                       "decision": "pending", "reviewer_type": "", "reviewer": "", "reason": "", "license_decision": "pending", "license_evidence": []} for d in sources.values()]
    write_jsonl(out / "source_reviews.pending.jsonl", source_reviews)
    write_json(out / "summary.json", summary)
    lock = {"schema_version": "paired-data-lock-v1", "corpus_sha256": file_hash(corpus), "authored_specs_sha256": file_hash(specs_path),
            "quarantine_sha256": file_hash(quarantine), "model_lock_sha256": tokenizer.metadata["model_lock_sha256"],
            "artifacts": {p.name: file_hash(p) for p in sorted(out.glob("*.json*"))},
            "sources": [{"id":d["id"], "family_id":d["family_id"], "split":d["split"], "sha256":digest(d), "connected_family_id":components[d["family_id"]]} for d in sources.values()]}
    write_json(out / "data.lock.json", lock)
    write_review(out, list(sources.values()), pairs, arms, summary)
    packet = {"schema_version": "paired-review-packet-v1", "training_ready": False,
              "files": {str(p.relative_to(out)).replace("\\", "/"): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "review_packet.lock.json", packet)
    if aggregate:
        target = Path(aggregate); target.mkdir(parents=True, exist_ok=True)
        write_json(target / "data_summary.json", summary)
        write_json(target / "data_source_lock.json", lock)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "specs", "quarantine", "assets", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--aggregate")
    args = parser.parse_args()
    print(json.dumps(build(args.corpus,args.specs,args.quarantine,args.assets,args.output,args.aggregate), ensure_ascii=False, indent=2))
