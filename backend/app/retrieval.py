"""混合检索：BM25（中英文分词）+ 可选向量检索，用 RRF 融合。"""
from __future__ import annotations

from . import tracing

import re
import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .db import get_conn
from .embeddings import embed_texts, embeddings_enabled, from_blob

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_EN_STOP = set(
    "a an the of to in on for and or with is are was were be been by as at from that this these those it its "
    "we our you they their which can may also into than then there such using use used based".split()
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = [w for w in _WORD.findall(text) if w not in _EN_STOP and len(w) > 1]
    if _CJK.search(text):
        import jieba

        jieba.setLogLevel(60)
        cjk_only = "".join(ch if _CJK.match(ch) else " " for ch in text)
        tokens += [w for w in jieba.lcut(cjk_only) if w.strip() and len(w.strip()) > 1]
    return tokens


@dataclass
class Hit:
    chunk_id: int
    paper_id: str
    text: str
    section: str | None
    page: int | None
    score: float


class _Index:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.version = -1
        self.rows: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.vecs: np.ndarray | None = None

    def ensure(self) -> None:
        with get_conn() as conn:
            version = conn.execute("SELECT coalesce(max(id),0) + count(*) FROM chunks").fetchone()[0]
            if version == self.version and self.bm25 is not None:
                return
            rows = [dict(r) for r in conn.execute("SELECT id, paper_id, section, page, text, embedding FROM chunks")]
        with self.lock:
            self.rows = rows
            corpus = [tokenize(r["text"]) or ["_"] for r in rows]
            self.bm25 = BM25Okapi(corpus) if corpus else None
            embs = [r["embedding"] for r in rows]
            if rows and all(e is not None for e in embs):
                m = np.stack([from_blob(e) for e in embs])
                self.vecs = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
            else:
                self.vecs = None
            self.version = version


_index = _Index()


def invalidate_index() -> None:
    _index.version = -1


@tracing.traced("tool", "检索文献证据")
def search(query: str, top_k: int = 6, paper_ids: list[str] | None = None) -> list[Hit]:
    _index.ensure()
    rows = _index.rows
    if not rows or _index.bm25 is None:
        return []
    allowed = set(paper_ids) if paper_ids else None
    mask = np.array([allowed is None or r["paper_id"] in allowed for r in rows])

    rankings: list[list[int]] = []
    q_tokens = tokenize(query)
    if q_tokens:
        bm = np.asarray(_index.bm25.get_scores(q_tokens))
        bm = np.where(mask, bm, -np.inf)
        order = [i for i in np.argsort(-bm) if bm[i] > 0][: top_k * 4]
        rankings.append(order)
    if _index.vecs is not None and embeddings_enabled():
        try:
            qv = embed_texts([query])[0]
            qv = qv / (np.linalg.norm(qv) + 1e-8)
            sims = np.where(mask, _index.vecs @ qv, -np.inf)
            rankings.append(list(np.argsort(-sims)[: top_k * 4]))
        except Exception:
            pass  # 向量服务挂了就退化为纯 BM25

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank)
    best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [
        Hit(
            chunk_id=rows[i]["id"], paper_id=rows[i]["paper_id"], text=rows[i]["text"],
            section=rows[i]["section"], page=rows[i]["page"], score=round(s, 5),
        )
        for i, s in best
    ]
