"""测评任务。每个任务调用的都是系统里真实在用的函数（assist / retrieval），不另写一套逻辑。

  judge       引用支撑判断：给定论断和文献片段，判断 supports / partial / unrelated
  fake_cite   假引用检出：把正确引用换成随机或“看起来相关”的另一篇文献，看页边批注能否标出
  retrieval   检索：给论断找到能支撑它的文献（英文 / 中文 / 中文 + 跨语言关键词扩展）
  missing     缺引用检测：中文段落里哪些句子需要引用
  latency     行内续写延迟 + 本次测评所有模型调用的延迟与花费汇总
"""
from __future__ import annotations

import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .datasets import Bundle, Pair, sample
from .metrics import f1, pct, prf, rate, ranking_metrics
from .provenance import digest, evaluation_case, file_record, safe_text, stable_id

VERDICTS = ("supports", "partial", "unrelated", "unknown")
VALID_VERDICTS = {"supports", "partial", "unrelated"}


def _pmap(fn: Callable, items: list, workers: int, label: str) -> list:
    """并发执行并在终端显示进度。模型调用是 IO 密集，线程池足够。"""
    out: list = [None] * len(items)
    done = 0
    step = max(1, len(items) // 10)

    def run(i: int) -> None:
        out[i] = fn(items[i])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _ in ex.map(run, range(len(items))):
            done += 1
            if done % step == 0 or done == len(items):
                print(f"  {label}: {done}/{len(items)}", file=sys.stderr, flush=True)
    return out


@dataclass
class _PaperStub:
    """judge_support 只用到标题和年份；测评数据没有标题，用中性的占位"""
    title: str = "（测评文献）"
    year: int | None = None
    id: str = "eval-judge-evidence"


# ---------------------------------------------------------------------------
# 文献库：把所有证据段落作为“论文”存入测评专用的数据库
# ---------------------------------------------------------------------------
class Library:
    def __init__(self, bundle: Bundle):
        import app.ingest as ingest
        from app.db import get_conn
        from app.embeddings import embed_texts, embeddings_enabled, to_blob
        from app.library import PaperIn, add_paper
        from app.retrieval import invalidate_index

        self.passages = bundle.passages
        self.pid: dict[str, str] = {}
        self.text: dict[str, str] = {}
        t0 = time.perf_counter()
        use_emb = embeddings_enabled()
        orig = ingest.embeddings_enabled
        ingest.embeddings_enabled = lambda: False  # 先只写文本，向量最后批量算
        try:
            for i, text in enumerate(self.passages):
                paper, _ = add_paper(PaperIn(title=f"Evidence E{i:04d}", abstract=text, source="eval"))
                ingest.store_chunks(paper.id, [ingest.Chunk(text, "Abstract", None)], fulltext=False)
                self.pid[text] = paper.id
                self.text[paper.id] = text
        finally:
            ingest.embeddings_enabled = orig
        if use_emb:
            print(f"  计算 {len(self.passages)} 段向量…", file=sys.stderr, flush=True)
            with get_conn() as conn:
                rows = [(r["id"], r["text"]) for r in conn.execute("SELECT id, text FROM chunks")]
            vecs = embed_texts([t for _, t in rows], batch=64)
            with get_conn() as conn:
                conn.executemany("UPDATE chunks SET embedding=? WHERE id=?", [(to_blob(v), cid) for (cid, _), v in zip(rows, vecs)])
            invalidate_index()
        self.build_seconds = round(time.perf_counter() - t0, 1)


# ---------------------------------------------------------------------------
# 1. 引用支撑判断
# ---------------------------------------------------------------------------
def _verdict_call(fn: Callable, case_id: str) -> dict:
    """Keep abstentions, malformed outputs and exceptions out of correct counts."""
    try:
        with evaluation_case(case_id):
            response = fn()
        if not isinstance(response, dict) or not isinstance(response.get("verdict"), str):
            return {"verdict": "invalid", "status": "failure", "error_type": "InvalidOutput",
                    "error": "Missing or non-string verdict"}
        verdict = response["verdict"]
        reason = safe_text(str(response.get("reason") or ""))
        if verdict in VALID_VERDICTS:
            return {"verdict": verdict, "status": "valid", "reason": reason}
        if verdict == "unknown":
            failed = reason.startswith("判断失败")
            return {"verdict": verdict, "status": "failure" if failed else "abstention", "reason": reason,
                    **({"error_type": "InvalidOutput" if "eval_invalid_output:" in reason else "HandledFailure",
                        "error": reason} if failed else {})}
        return {"verdict": "invalid", "status": "failure", "error_type": "InvalidOutput",
                "error": safe_text(f"Unexpected verdict: {verdict}")}
    except Exception as exc:
        return {"verdict": "unknown", "status": "failure", "error_type": type(exc).__name__,
                "error": safe_text(str(exc))}


def _status_metrics(outcomes: list[dict]) -> dict:
    n = len(outcomes)
    valid = sum(o["status"] == "valid" for o in outcomes)
    abstained = sum(o["status"] == "abstention" for o in outcomes)
    failed = sum(o["status"] == "failure" for o in outcomes)
    return {"n_valid": valid, "n_abstained": abstained, "n_failed": failed,
            "coverage": rate(valid, n), "abstention_rate": rate(abstained, n), "failure_rate": rate(failed, n)}


def _judge_metrics(golds: list[bool], preds: list[str | dict]) -> dict[str, Any]:
    if len(golds) != len(preds):
        raise ValueError("gold and prediction counts differ")
    outcomes = [p if isinstance(p, dict) else {"verdict": p, "status": "valid" if p in VALID_VERDICTS
                 else "abstention" if p == "unknown" else "failure"} for p in preds]
    pos = [p["verdict"] for g, p in zip(golds, outcomes) if g]
    neg = [p["verdict"] for g, p in zip(golds, outcomes) if not g]
    tp = sum(p == "supports" for p in pos)
    fp = sum(p == "supports" for p in neg)
    fn = len(pos) - tp
    correct = sum(p["status"] == "valid" and ((g and p["verdict"] == "supports") or
                  (not g and p["verdict"] in {"partial", "unrelated"})) for g, p in zip(golds, outcomes))
    status = _status_metrics(outcomes)
    return {
        "n": len(golds), "n_supported": len(pos), "n_not_supported": len(neg),
        "n_correct": correct,
        # 二分类：partial/unrelated 均为有效拒绝；unknown/非法/异常均不正确。
        "accuracy": rate(correct, len(golds)), "total_accuracy": rate(correct, len(golds)),
        "valid_accuracy": rate(correct, status["n_valid"]), **status,
        "support_precision": rate(tp, tp + fp),
        "support_recall": rate(tp, tp + fn),
        "support_f1": f1(rate(tp, tp + fp), rate(tp, tp + fn)),
        # 最要紧的错误：证据撑不起论断，却判成“支持”（放过假引用）
        "false_support_rate": rate(fp, len(neg)),
        # 宽松口径：partial 也算没否定，看真正支持的论断有多少被判成 unrelated
        "missed_support_rate": rate(sum(p == "unrelated" for p in pos), len(pos)),
        "unknown_rate": rate(sum(p["verdict"] == "unknown" for p in outcomes), len(outcomes)),
        "confusion": {
            "supported": dict(Counter(pos)),
            "not_supported": dict(Counter(neg)),
        },
    }


def run_judge(bundle: Bundle, limit: int | None, workers: int, seed: int = 13) -> dict[str, Any]:
    from app.assist import judge_support

    def evaluate(pairs: list[Pair], label: str) -> dict:
        inputs = [{"case_id": stable_id(label, p.claim, p.evidence), "claim": p.claim,
                   "evidence": p.evidence[:1500], "supported": p.supported,
                   "paper": vars(_PaperStub(id=stable_id("evidence", p.evidence)))} for p in pairs]
        def one(case: dict) -> dict:
            return {**case, **_verdict_call(lambda: judge_support(case["claim"], _PaperStub(**case["paper"]),
                                                               case["evidence"]), case["case_id"])}
        cases = _pmap(one, inputs, workers, label)
        return _judge_metrics([p.supported for p in pairs], cases) | {"cases": cases, "inputs_sha256": digest(inputs)}

    pairs = sample(bundle.pairs, limit, seed)
    res = evaluate(pairs, "judge")

    # 中文论断 + 英文文献（真实使用场景：用户写中文、文献库是英文论文）
    # 译文评分只使用支持正例；随机错配作为未复核候选单独导出。
    rng = random.Random(seed)
    related: dict[str, set[str]] = {}
    for p in bundle.pairs:
        related.setdefault(p.claim, set()).add(p.evidence)
    pool = list(dict.fromkeys(p.evidence for p in bundle.pairs))
    zh_pairs: list[Pair] = []
    for p in sample([p for p in bundle.pairs if p.supported and p.claim in bundle.zh_claims], limit, seed):
        zh = bundle.zh_claims[p.claim]
        zh_pairs.append(Pair(zh, p.evidence, True))
        others = [e for e in pool if e not in related[p.claim]]
        if others:
            zh_pairs.append(Pair(zh, rng.choice(others), False))
    if zh_pairs:
        # Randomly mismatched pairs are candidates, not independently verified
        # negative gold. Only translated dataset positives are scored here.
        res["zh"] = evaluate([p for p in zh_pairs if p.supported], "judge_zh")
        res["zh"]["label_scope"] = "translated positives only; negative candidates are unverified"
        res["zh_negative_candidates"] = [dict(candidate_id=stable_id("negative", p.claim, p.evidence),
                                             claim=p.claim, evidence=p.evidence, label=None,
                                             candidate_kinds=["zh_mismatch"],
                                             review_status="unreviewed", reviewer="", review_note="",
                                             independent_review=False) for p in zh_pairs if not p.supported]
    return res


# ---------------------------------------------------------------------------
# 2. 假引用检出（走页边批注同一条路径：verify_citation = 在被引文献里检索片段 + 判断）
# ---------------------------------------------------------------------------
def run_fake_cite(bundle: Bundle, lib: Library, limit: int | None, workers: int, seed: int = 13) -> dict[str, Any]:
    from app.assist import verify_citation
    from app.retrieval import search

    rng = random.Random(seed)
    related: dict[str, set[str]] = {}
    for p in bundle.pairs:  # 同一论断的所有证据（不论标签）都不能拿来当“假引用”
        related.setdefault(p.claim, set()).add(p.evidence)
    pos = sample([p for p in bundle.pairs if p.supported], limit, seed)
    pool = [e for e in dict.fromkeys(p.evidence for p in bundle.pairs)]

    inputs: list[dict] = []
    candidates: dict[str, dict] = {}
    generation_errors = []
    all_reviewed = {r["candidate_id"]: r for r in bundle.reviewed_negatives}
    reviewed = sample(bundle.reviewed_negatives, limit, seed)

    def add_candidate(claim: str, evidence: str, kind: str) -> None:
        cid = stable_id("negative", claim, evidence)
        if cid not in candidates:
            candidates[cid] = {"candidate_id": cid, "claim": claim, "evidence": evidence,
                               "candidate_kinds": [], "review_status": "unreviewed", "label": None,
                               "reviewer": "", "review_note": "", "independent_review": False}
        if kind not in candidates[cid]["candidate_kinds"]:
            candidates[cid]["candidate_kinds"].append(kind)

    for p in pos:
        inputs.append({"kind": "true", "case_id": stable_id("true", p.claim, p.evidence),
                       "claim": p.claim, "evidence": p.evidence, "paper_id": lib.pid[p.evidence]})
        cands = [e for e in pool if e not in related[p.claim]]
        if cands:
            add_candidate(p.claim, rng.choice(cands), "random")
        # Not in known gold is insufficient to assert a negative label.
        try:
            hard = next((h.text for h in search(p.claim, top_k=30) if h.text not in related[p.claim]), None)
        except Exception as exc:
            hard = None
            generation_errors.append({"claim_sha256": digest(p.claim), "error_type": type(exc).__name__,
                                      "error": safe_text(str(exc))})
        if hard:
            add_candidate(p.claim, hard, "hard")
    for review in reviewed:
        add_candidate(review["claim"], review["evidence"], "imported_review")
    for cid, candidate in candidates.items():
        if cid in all_reviewed:
            candidate.update(all_reviewed[cid])
        inputs.append({"kind": "reviewed_negative" if cid in all_reviewed else "unreviewed_candidate",
                       "case_id": cid, "claim": candidate["claim"], "evidence": candidate["evidence"],
                       "paper_id": lib.pid[candidate["evidence"]]})

    def one(case: dict) -> dict:
        return {**case, **_verdict_call(lambda: verify_citation(case["claim"], case["paper_id"]), case["case_id"])}
    cases = _pmap(one, inputs, workers, "fake_cite")
    by: dict[str, list[dict]] = {}
    for case in cases:
        by.setdefault(case["kind"], []).append(case)

    def summary(rows: list[dict]) -> dict[str, Any]:
        vs = [r["verdict"] for r in rows]
        return {
            "n": len(vs),
            "flagged_red": rate(sum(v == "unrelated" for v in vs), len(vs)),      # 页边卡片标红：不相关
            "flagged_any": rate(sum(v in {"partial", "unrelated"} for v in vs), len(vs)),
            "verdicts": dict(Counter(vs)), **_status_metrics(rows),
        }

    t = summary(by.get("true", []))
    reviewed_summary = summary(by.get("reviewed_negative", []))
    return {"true": t, "reviewed_negative": reviewed_summary,
            "candidate_diagnostics": {"label_scope": "unreviewed; descriptive only, not gold performance",
                                      **summary(by.get("unreviewed_candidate", []))},
            "headline": {
        "detect_reviewed": reviewed_summary["flagged_red"],
        "detect_reviewed_any": reviewed_summary["flagged_any"],
        "n_reviewed": reviewed_summary["n"],
        "false_alarm_red": t.get("flagged_red"),
        "false_alarm_any": t.get("flagged_any"),
        }, "cases": cases, "negative_candidates": list(candidates.values()),
        "generation_errors": generation_errors,
        "inputs_sha256": digest([{k: v for k, v in c.items() if k != "paper_id"} for c in inputs])}


# ---------------------------------------------------------------------------
# 3. 检索
# ---------------------------------------------------------------------------
def run_retrieval(bundle: Bundle, lib: Library, limit: int | None, fake: bool, seed: int = 13) -> dict[str, Any]:
    import app.retrieval as R
    from app.assist import retrieval_query
    from app.embeddings import embeddings_enabled

    gold: dict[str, set[str]] = {}
    for p in bundle.pairs:
        if p.supported:
            gold.setdefault(p.claim, set()).add(lib.pid[p.evidence])
    claims = sample(sorted(gold), limit, seed)
    zh = [c for c in claims if c in bundle.zh_claims]

    def evaluate(queries: list[tuple[str, str]], hybrid: bool) -> dict[str, Any]:
        orig = R.embeddings_enabled
        if not hybrid:
            R.embeddings_enabled = lambda: False  # 只用 BM25
        try:
            ranked = []
            cases = []
            for claim, q in queries:
                cid = stable_id("retrieval", claim, q)
                error = None
                try:
                    with evaluation_case(cid):
                        hits = list(dict.fromkeys(h.paper_id for h in R.search(q, top_k=10)))
                except Exception as exc:
                    hits = []
                    error = {"error_type": type(exc).__name__, "error": safe_text(str(exc))}
                ranked.append((hits, gold[claim]))
                cases.append({"case_id": cid, "claim": claim, "query": q,
                              "gold_evidence_ids": sorted(stable_id("evidence", lib.text[pid]) for pid in gold[claim]),
                              "ranked_evidence_ids": [stable_id("evidence", lib.text[pid]) for pid in hits],
                              "status": "failure" if error else "valid", **(error or {})})
        finally:
            R.embeddings_enabled = orig
        return ranking_metrics(ranked) | {"n": len(queries), "cases": cases,
                                         "failure_rate": rate(sum(c["status"] == "failure" for c in cases), len(cases)),
                                         "inputs_sha256": digest(queries)}

    res: dict[str, Any] = {"corpus_size": len(lib.passages), "embeddings": embeddings_enabled()}
    res["en_bm25"] = evaluate([(c, c) for c in claims], hybrid=False)
    res["zh_bm25"] = evaluate([(c, bundle.zh_claims[c]) for c in zh], hybrid=False)
    if not fake:  # 关键词扩展要调用模型
        expansion_cases = []
        def expand(claim: str) -> tuple[str, str]:
            cid = stable_id("zh_expand", claim)
            try:
                with evaluation_case(cid):
                    query = retrieval_query(bundle.zh_claims[claim])
                failed = query == bundle.zh_claims[claim]
                detail = {"status": "failure" if failed else "valid",
                          **({"error_type": "NoExpansion", "error": "No English keyword expansion returned"} if failed else {})}
            except Exception as exc:
                query = bundle.zh_claims[claim]
                detail = {"status": "failure", "error_type": type(exc).__name__, "error": safe_text(str(exc))}
            expansion_cases.append({"case_id": cid, "input": bundle.zh_claims[claim], "query": query, **detail})
            return claim, query
        expanded = _pmap(expand, zh, 4, "zh_expand")
        res["zh_bm25_expand"] = evaluate(expanded, hybrid=False)
        res["zh_bm25_expand"]["expansion_cases"] = sorted(expansion_cases, key=lambda c: c["case_id"])
        # 扩展失败（模型返回空）时查询不变，这个比例高说明关键词扩展没生效
        res["zh_bm25_expand"]["expand_failed_rate"] = rate(sum(q == bundle.zh_claims[c] for c, q in expanded), len(expanded))
        if embeddings_enabled():
            res["en_hybrid"] = evaluate([(c, c) for c in claims], hybrid=True)
            res["zh_hybrid_expand"] = evaluate(expanded, hybrid=True)
    return res


# ---------------------------------------------------------------------------
# 4. 缺引用检测（页边批注的“这句话缺少引用”）
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s，。；！？、：,.;!?:\"'“”‘’（）()]")


def _norm(s: str) -> str:
    return _PUNCT.sub("", s)


def run_missing(bundle: Bundle, limit: int | None, workers: int, seed: int = 13) -> dict[str, Any]:
    from app.assist import _review, strip_cites

    paras = sample(bundle.citeworthy, limit, seed)

    def one(para: dict) -> dict:
        sents = [s["text"] for s in para["sentences"]]
        plain = "".join(sents)
        cid = stable_id("missing", plain, para.get("title", ""))
        try:
            with evaluation_case(cid):
                data = _review(plain, plain, para.get("title", ""))
            if not isinstance(data, dict) or not isinstance(data.get("missing"), list):
                raise ValueError("Invalid missing-citation response schema")
            if any(not isinstance(m, dict) or not isinstance(m.get("sentence"), str) for m in data["missing"]):
                raise ValueError("Invalid missing-citation item schema")
        except Exception as exc:
            return {"case_id": cid, "status": "failure", "picked": [], "error_type": type(exc).__name__,
                    "error": safe_text(str(exc))}
        picked: list[int] = []
        # 与 annotate() 完全相同的过滤：最多 2 句，必须原样出自段落
        for m in (data.get("missing") or [])[:2]:
            sent = strip_cites(str(m.get("sentence", ""))).replace("[引用]", "").strip()
            if len(sent) < 6 or sent not in plain:
                continue
            ns = _norm(sent)
            for i, s in enumerate(sents):
                if ns and (ns in _norm(s) or _norm(s) in ns):
                    picked.append(i)
                    break
        return {"case_id": cid, "status": "valid", "picked": sorted(set(picked))}

    preds = _pmap(one, paras, workers, "missing")
    tp = fp = fn = 0
    for para, prediction in zip(paras, preds):
        picked = prediction["picked"]
        golds = {i for i, s in enumerate(para["sentences"]) if s["cite"]}
        tp += len(golds & set(picked))
        fp += len(set(picked) - golds)
        fn += len(golds - set(picked))
    n_sent = sum(len(p["sentences"]) for p in paras)
    return {"paragraphs": len(paras), "sentences": n_sent, "gold_positive": tp + fn, "flagged": tp + fp,
            "cases": preds, "inputs_sha256": digest(paras), **_status_metrics(preds)} | prf(tp, fp, fn)


# ---------------------------------------------------------------------------
# 5. 延迟与花费
# ---------------------------------------------------------------------------
def run_locate_audited(bundle: Bundle, limit: int | None) -> dict:
    """Keep hashes of the actual rendered PDFs before the locate task removes them."""
    from . import locate

    original_render = locate.render_pdf
    pdf_inputs: list[dict] = []
    items = [vars(item) for item in locate.build_items(bundle, limit)]

    def render(path, *args, **kwargs):
        original_render(path, *args, **kwargs)
        pdf_inputs.append(file_record(path, name=path.name))

    locate.render_pdf = render
    try:
        result = locate.run_locate(bundle, limit)
    finally:
        locate.render_pdf = original_render
    return result | {"pdf_inputs": pdf_inputs, "inputs_sha256": digest({"items": items, "pdfs": pdf_inputs}),
                     "seeds": {"builder": 7, "layout": 11}}


def run_latency(bundle: Bundle, limit: int | None, seed: int = 13) -> dict[str, Any]:
    from app.assist import complete

    paras = sample(bundle.citeworthy, limit or 20, seed)
    times = []
    valid_times = []
    cases = []
    empty = 0
    for p in paras:
        before = "".join(s["text"] for s in p["sentences"][:2])
        cid = stable_id("latency", before, p.get("title", ""))
        case = {"case_id": cid, "before": before, "title": p.get("title", "")}
        t0 = time.perf_counter()
        try:
            with evaluation_case(cid):
                text = complete(before, title=p.get("title", ""), use_library=False)
            is_empty = not text.strip()
            empty += is_empty
            case.update(status="abstention" if is_empty else "valid", output_sha256=digest(text))
        except Exception as exc:
            empty += 1
            case.update(status="failure", error_type=type(exc).__name__, error=safe_text(str(exc)))
        times.append((time.perf_counter() - t0) * 1000)
        case["latency_ms"] = round(times[-1])
        cases.append(case)
        if case["status"] == "valid":
            valid_times.append(times[-1])
    return {"complete": {"n": len(times), "p50_ms": pct(times, 50), "p95_ms": pct(times, 95),
                         "valid_p50_ms": pct(valid_times, 50), "valid_p95_ms": pct(valid_times, 95),
                         "empty_rate": rate(empty, len(times)), "cases": cases, **_status_metrics(cases)},
            "inputs_sha256": digest([{"before": c["before"], "title": c["title"]} for c in cases])}


def llm_usage() -> dict[str, Any]:
    """汇总本次测评期间所有模型调用（测评用的是独立数据库，不会混入平时的使用记录）"""
    from app import pricing
    from app.db import get_conn

    out: dict[str, Any] = {}
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT ts, tier, model, ok, latency_ms, prompt_tokens, completion_tokens, cached_tokens, cost_usd FROM llm_calls")]
    for r in rows:
        r["cost_cny"] = pricing.call_cost(r, pricing.price_for(r["model"])) if r["ok"] else None
    for tier in sorted({r["tier"] for r in rows}):
        rs = [r for r in rows if r["tier"] == tier]
        lat = [r["latency_ms"] for r in rs if r["ok"] and r["latency_ms"] is not None]
        cost = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
        out[tier] = {
            "models": sorted({r["model"] for r in rs}),
            "calls": len(rs),
            "error_rate": rate(sum(not r["ok"] for r in rs), len(rs)),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in rs),
            "cost_usd": round(sum(cost), 4) if cost else None,
            "cost_cny": round(sum(c for c in cny), 4) if (cny := [r["cost_cny"] for r in rs if r["cost_cny"] is not None]) else None,
        }
    return out


def iter_tasks(names: Iterable[str]) -> list[str]:
    allowed = ["judge", "fake_cite", "retrieval", "missing", "locate", "latency"]
    names = list(names)
    bad = [n for n in names if n not in allowed]
    if bad:
        raise SystemExit(f"未知任务：{', '.join(bad)}；可选：{', '.join(allowed)}")
    return [n for n in allowed if n in names]
