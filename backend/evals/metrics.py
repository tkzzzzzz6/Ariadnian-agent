"""测评指标。全部是比例或毫秒数，保留 3 位小数。"""
from __future__ import annotations

import math


def rate(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r == 0:
        return None if p is None or r is None else 0.0
    return round(2 * p * r / (p + r), 3)


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p, r = rate(tp, tp + fp), rate(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": f1(p, r), "tp": tp, "fp": fp, "fn": fn}


def pct(values: list[float], p: float) -> int | None:
    """最近秩法百分位数"""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return int(round(s[k]))


def ranking_metrics(ranked: list[tuple[list[str], set[str]]]) -> dict[str, float | None]:
    """文档级检索：宏平均证据召回、至少命中一项、完整证据集命中分别报告。

    无 gold 的查询不可评估，单列计数；重复文档只保留首次出现。历史 v1
    的 recall@k 实际是 hit@k，不能与这里的 recall@k 直接比较。
    """
    scored = [(list(dict.fromkeys(r)), g) for r, g in ranked if g]
    n = len(scored)
    out: dict = {"n_scored": n, "n_unscorable": len(ranked) - n}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = rate(sum(len(set(r[:k]) & g) / len(g) for r, g in scored), n)
        out[f"hit@{k}"] = rate(sum(bool(set(r[:k]) & g) for r, g in scored), n)
        out[f"complete@{k}"] = rate(sum(g <= set(r[:k]) for r, g in scored), n)
    rr = 0.0
    for r, g in scored:
        for i, pid in enumerate(r[:10], start=1):
            if pid in g:
                rr += 1 / i
                break
    out["mrr@10"] = round(rr / n, 3) if n else None
    return out
