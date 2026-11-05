"""把测评结果整理成终端摘要和 evals/results/README.md 汇总表。"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

from .provenance import PROTOCOL_VERSION


def _aggregate_metrics(value: Any) -> Any:
    """Fail closed on record arrays and free-form text; keep numeric aggregates."""
    if not isinstance(value, dict):
        return value if value is None or isinstance(value, (int, float, bool)) else None
    omitted = {"cases", "predictions", "inputs", "evidence", "paper", "pdf_inputs", "source_lineage",
               "negative_candidates", "zh_negative_candidates", "generation_errors", "expansion_cases",
               "messages", "requests", "paths", "raw_output", "claim", "question", "reference_answer",
               "text", "query", "before", "after", "reason", "error", "snippet"}
    result = {}
    for key, item in value.items():
        if key in omitted or key.endswith(("_cases", "_predictions", "_candidates", "_inputs")):
            continue
        if item is None or isinstance(item, (int, float, bool)):
            result[key] = item
        elif isinstance(item, dict):
            nested = _aggregate_metrics(item)
            if nested:
                result[key] = nested
        elif isinstance(item, str):
            if key.endswith("_sha256") and re.fullmatch(r"[0-9a-f]{64}", item):
                result[key] = item
            elif key in {"status", "error_type"} and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,79}", item):
                result[key] = item
    return result


def aggregate_summary(result: dict) -> dict:
    """Public aggregate artifact: no per-case content, requests or local paths.

    Provenance uses an explicit allowlist. Hashes bind the summary to the local
    details without publishing configuration, filenames or dirty-file listings.
    This helper never imports application settings or accesses credentials.
    """
    summary = {key: result[key] for key in ("protocol_version", "created", "mode", "models", "embeddings",
               "source", "seed", "limit", "seconds") if key in result}
    summary.update(artifact_kind="aggregate_summary", data=_aggregate_metrics(result.get("data", {})),
                   tasks=_aggregate_metrics(result.get("tasks", {})),
                   llm_usage=_aggregate_metrics(result.get("llm_usage", {})),
                   local_details="Per-case results, requests and negative-review candidates remain local and are not part of this artifact.")
    provenance = result.get("provenance") or {}
    safe = {key: provenance[key] for key in ("protocol_version", "runtime", "seed", "seeds", "tasks", "limit", "workers",
            "bundle_sha256", "model_requests_sha256", "prompt_sources_sha256") if key in provenance}
    for key in ("git", "git_at_end"):
        if isinstance(provenance.get(key), dict):
            safe[key] = {name: provenance[key][name] for name in ("commit", "dirty", "tracked_diff_sha256", "unavailable")
                         if name in provenance[key]}
    for key in ("sources", "dataset_files", "config"):
        if key in provenance:
            payload = json.dumps(provenance[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            safe[key + "_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    safe["model_request_count"] = len(provenance.get("model_requests") or [])
    summary["provenance"] = safe
    return summary


def write_summary(full_result_path: Path) -> Path:
    if full_result_path.name.endswith(".summary.json"):
        raise ValueError("write_summary expects a full result JSON")
    raw = full_result_path.read_bytes()
    result = json.loads(raw)
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Only v2 results are exported; historical JSON stays unchanged")
    summary = aggregate_summary(result)
    summary["full_result_sha256"] = hashlib.sha256(raw).hexdigest()
    target = full_result_path.with_suffix(".summary.json")
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _p(v: Any) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _ms(v: Any) -> str:
    return "—" if v is None else f"{v / 1000:.2f}s" if v >= 1000 else f"{v}ms"


def _g(d: dict, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)  # type: ignore[assignment]
    return d


def _name(r: dict) -> str:
    if r["mode"] == "offline":
        return "离线基线（关键词规则）"
    return r["models"].get("judge", "?")


def _cost(r: dict) -> str:
    """LiteLLM 没有价格表的模型（自定义接口、新模型）算不出花费，显示未知而不是 $0"""
    usage = list((r.get("llm_usage") or {}).values())
    cny = [u.get("cost_cny") for u in usage]
    if any(c is not None for c in cny):  # 优先用官网价格表算的人民币
        known = [c for c in cny if c is not None]
        return f"¥{sum(known):.3f}" + ("（部分未知）" if len(known) < len(cny) else "")
    costs = [u.get("cost_usd") for u in usage]
    known = [c for c in costs if c is not None]
    if not known:
        return "未知"
    return f"${sum(known):.4f}" + ("（部分未知）" if len(known) < len(costs) else "")


def summarize(r: dict) -> str:
    t = {k: v for k, v in r["tasks"].items() if v.get("status") != "failure"}
    modern = r.get("protocol_version") == PROTOCOL_VERSION
    lines = [f"\n== 测评结果：{_name(r)}（{r['created']}，用时 {r['seconds']}s）",
             f"协议：{PROTOCOL_VERSION if modern else 'v1 历史口径；unknown 可能计为正确，检索 R@k 实为 Hit@k'}"]
    for name, task in r["tasks"].items():
        if task.get("status") == "failure":
            lines.append(f"! {name} 任务失败：{task.get('error_type', '')} {task.get('error', '')}")
    if "judge" in t:
        j = t["judge"]
        lines.append(f"引用判断  n={j['n']}  准确率 {_p(j['accuracy'])}  支持 F1 {_p(j['support_f1'])}  "
                     f"假支持率 {_p(j['false_support_rate'])}  漏判支持 {_p(j['missed_support_rate'])}")
        if modern:
            lines.append(f"  有效判定准确率 {_p(j.get('valid_accuracy'))}  覆盖 {_p(j.get('coverage'))}  "
                         f"弃答 {_p(j.get('abstention_rate'))}  失败 {_p(j.get('failure_rate'))}")
        if "zh" in j:
            lines.append(f"  中文{'正例' if modern else '论断'} n={j['zh']['n']}  "
                         f"准确率 {_p(j['zh']['accuracy'])}  覆盖 {_p(j['zh'].get('coverage'))}")
    if "fake_cite" in t:
        h = t["fake_cite"]["headline"]
        if modern:
            lines.append(f"假引用检出  人工复核负例 n={h['n_reviewed']}  检出 {_p(h['detect_reviewed'])}"
                         f"（含黄色 {_p(h['detect_reviewed_any'])}）  正确引用误报 {_p(h['false_alarm_red'])}")
            lines.append("  自动错配与检索难例仅为未复核候选，不计入 gold 检出率；候选内容和标注模板随结果保存。")
        else:
            lines.append(f"假引用检出（历史未复核候选）  随机 {_p(h['detect_random'])}  难例 {_p(h['detect_hard'])}  "
                         f"正确引用误报 {_p(h['false_alarm_red'])}")
    if "retrieval" in t:
        rt = t["retrieval"]
        parts = [f"{k} {'宏 R@5' if modern else '旧 Hit@5'} {_p(_g(rt, k, 'recall@5'))}"
                 + (f" Hit@5 {_p(_g(rt, k, 'hit@5'))} 完整@5 {_p(_g(rt, k, 'complete@5'))}" if modern else "")
                 + f" MRR {_g(rt, k, 'mrr@10')}"
                 + (f"（扩展失败 {_p(_g(rt, k, 'expand_failed_rate'))}）" if _g(rt, k, "expand_failed_rate") else "") for k in
                 ("en_bm25", "en_hybrid", "zh_bm25", "zh_bm25_expand", "zh_hybrid_expand") if k in rt]
        lines.append(f"检索（语料 {rt['corpus_size']} 段）  " + " | ".join(parts))
    if "missing" in t:
        m = t["missing"]
        lines.append(f"缺引用检测  {m['sentences']} 句  P {_p(m['precision'])}  R {_p(m['recall'])}  F1 {_p(m['f1'])}")
    if "locate" in t and "locate" in t["locate"]:
        lc = t["locate"]
        lo = lc["locate"]
        lines.append(f"原文定位  n={lc['n']}  定位准确率 {_p(_g(lo, 'claim', 'hit'))}（只取首句 {_p(_g(lo, 'first_sentence', 'hit'))}）  "
                     f"高亮精确率 {_p(_g(lo, 'claim', 'precision'))}（整块高亮 {_p(_g(lo, 'whole_chunk', 'precision'))}）  "
                     f"证据覆盖 {_p(_g(lo, 'claim', 'coverage'))}  P50 {_ms(lc.get('latency_p50_ms'))}")
        lines.append(f"  相关段落 R@1 {_p(_g(lc, 'passages', 'recall@1'))}  R@3 {_p(_g(lc, 'passages', 'recall@3'))}  |  "
                     f"中文论断 n={_g(lc, 'locate_zh', 'n')}  定位 {_p(_g(lc, 'locate_zh', 'hit'))}  段落 R@3 {_p(_g(lc, 'passages_zh', 'recall@3'))}")
    if "latency" in t and "complete" in t["latency"]:
        c = t["latency"]["complete"]
        lines.append(f"行内续写  P50 {_ms(c['p50_ms'])}  P95 {_ms(c['p95_ms'])}  空结果 {_p(c['empty_rate'])}")
    usage = r.get("llm_usage", {}).values()
    calls = sum(u["calls"] for u in usage)
    if calls:
        tokens = sum(u["prompt_tokens"] + u["completion_tokens"] for u in usage)
        lines.append(f"模型调用 {calls} 次，{tokens / 1000:.0f}k tokens，花费 {_cost(r)}")
        for tier in ("judge", "complete"):
            u = r["llm_usage"].get(tier)
            if u and u["calls"] and u["completion_tokens"] / u["calls"] > 200:
                lines.append(f"! {tier} 档平均每次输出 {u['completion_tokens'] // u['calls']} tokens，远超正常的几十个，"
                             f"疑似推理模型（先思考再回答）：延迟高、容易输出为空。建议在设置的「按任务分档」里给 {tier} 档换非推理模型。")
    return "\n".join(lines)


HEADER = """# 测评结果

由 `npm run eval`（或 `cd backend && uv run python -m evals`）生成。说明见 [../README.md](../README.md)。

v2 与历史 v1 分开呈现：v1 的负例 unknown 可能计入正确，`Recall@k` 实为 `Hit@k`，自动负例未独立核实。
v2 的 unknown / 非法 / 异常均不正确；中文只评分已有支持标签的译文正例；假引用检出只评分明确人工复核的负例。
历史 JSON 保持原样，不能把两种协议的数字直接当作提升对比。

## v2

检索列为英文 / 中文 / 中文+扩展的宏平均证据召回率；Hit@k 与完整证据集命中率见每次 JSON。
v2 行链接可提交的 `.summary.json` 聚合摘要；含逐条证据的完整 JSON、请求记录和负例模板仅保留本机，未上传 Git。

| 运行 | 判断准确率<br/>总量 / 有效 | 判断覆盖<br/>弃答 / 失败 | 假支持率 ↓ | 中文正例<br/>支持率 | 人工负例<br/>检出率（n） | 正确引用<br/>误报 ↓ | 宏 R@5<br/>英 / 中 / 扩展 | 缺引用 F1 | 定位<br/>准确 / 段落命中@3 | 续写 P50 | 花费 |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""

LEGACY_HEADER = """
## v1 历史结果（旧口径）

下列“随机 / 难例”是未人工核实候选的旧统计；不是 gold 假引用检出率。旧 R@5 列实际为 Hit@5。

| 运行 | 判断旧准确率 | 假支持率 ↓ | 中文旧准确率 | 未复核候选旧标红率<br/>随机 / 难例 | 正确引用误报 ↓ | 旧 Hit@5<br/>英 / 中 / 扩展 | 缺引用 F1 | 定位<br/>准确 / 段落命中@3 | 续写 P50 | 花费 |
|---|---|---|---|---|---|---|---|---|---|---|
"""


def write_index(results_dir: Path) -> None:
    rows = []
    legacy_rows = []
    runs: dict[str, tuple[Path, dict]] = {}
    for f in sorted(results_dir.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "tasks" not in r or "models" not in r:
            continue
        is_summary = f.name.endswith(".summary.json") and r.get("artifact_kind") == "aggregate_summary"
        run_id = f.name.removesuffix(".summary.json") if is_summary else f.stem
        if run_id not in runs or is_summary:
            runs[run_id] = (f, r)
    for _, (f, r) in sorted(runs.items()):
        t = r["tasks"]
        fk = _g(t, "fake_cite", "headline") or {}
        rt = t.get("retrieval") or {}
        limit = f"，每项 ≤{r['limit']} 条" if r.get("limit") else ""
        name = f"[{_name(r)}]({f.name})<br/><sub>{r['created'][:16].replace('T', ' ')}{limit}</sub>"
        if r.get("protocol_version") == PROTOCOL_VERSION:
            rows.append("| " + " | ".join([
                name,
                f"{_p(_g(t, 'judge', 'accuracy'))} / {_p(_g(t, 'judge', 'valid_accuracy'))}",
                f"{_p(_g(t, 'judge', 'coverage'))}<br/>{_p(_g(t, 'judge', 'abstention_rate'))} / {_p(_g(t, 'judge', 'failure_rate'))}",
                _p(_g(t, "judge", "false_support_rate")), _p(_g(t, "judge", "zh", "accuracy")),
                f"{_p(fk.get('detect_reviewed'))} ({fk.get('n_reviewed', 0)})" if fk else "—",
                _p(fk.get("false_alarm_red")),
                " / ".join(_p(_g(rt, k, "recall@5")) for k in ("en_bm25", "zh_bm25", "zh_bm25_expand")) if rt else "—",
                _p(_g(t, "missing", "f1")),
                f"{_p(_g(t, 'locate', 'locate', 'claim', 'hit'))} / {_p(_g(t, 'locate', 'passages', 'recall@3'))}" if _g(t, "locate", "locate") else "—",
                _ms(_g(t, "latency", "complete", "p50_ms")), _cost(r) if r["mode"] == "model" else "—",
            ]) + " |")
            continue
        legacy_rows.append("| " + " | ".join([
            name,
            _p(_g(t, "judge", "accuracy")),
            _p(_g(t, "judge", "false_support_rate")),
            _p(_g(t, "judge", "zh", "accuracy")),
            f"{_p(fk.get('detect_random'))} / {_p(fk.get('detect_hard'))}" if fk else "—",
            _p(fk.get("false_alarm_red")) if fk else "—",
            " / ".join(_p(_g(rt, k, "recall@5")) for k in ("en_bm25", "zh_bm25", "zh_bm25_expand")) if rt else "—",
            _p(_g(t, "missing", "f1")),
            f"{_p(_g(t, 'locate', 'locate', 'claim', 'hit'))} / {_p(_g(t, 'locate', 'passages', 'recall@3'))}" if _g(t, "locate", "locate") else "—",
            _ms(_g(t, "latency", "complete", "p50_ms")),
            _cost(r) if r["mode"] == "model" else "—",
        ]) + " |")
    (results_dir / "README.md").write_text(HEADER + "\n".join(rows) + "\n" + LEGACY_HEADER +
                                          "\n".join(legacy_rows) + "\n", encoding="utf-8")
