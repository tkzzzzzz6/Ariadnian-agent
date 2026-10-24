"""外部学术检索：Semantic Scholar + arXiv + OpenAlex，结果统一成 PaperIn，并按 DOI / arXiv / 标题去重。

这些公开接口都有频率限制，这里统一处理：
- 每个来源一个限速器，同一来源的请求之间保持最小间隔（S2 带 key 时 1 次/秒，arXiv 官方建议 3 秒一次）；
- 429 / 5xx 自动重试，优先遵守服务端返回的 Retry-After，否则指数退避；
- Semantic Scholar 重试后仍失败（典型情况：没有 API key、被共享配额限流），自动改用 OpenAlex 补位；
- 检索结果在 SQLite 里缓存若干天，同一个检索式不会重复请求。
"""
from __future__ import annotations

from . import tracing

import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Callable

import httpx

from .config import get_settings
from .db import get_conn
from .library import PaperIn, normalize_title

_UA = {"User-Agent": "Ariadne-paper-assistant/0.1 (local research tool)"}
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

SOURCE_NAMES = {"semantic_scholar": "Semantic Scholar", "arxiv": "arXiv", "openalex": "OpenAlex"}

# 测试时替换成 httpx.MockTransport / 假的 sleep
_TRANSPORT: httpx.BaseTransport | None = None
_sleep: Callable[[float], None] = time.sleep


_no_retry: ContextVar[bool] = ContextVar("scholar_no_retry", default=False)


class SourceError(RuntimeError):
    """某个来源检索失败，message 是给用户看的中文说明"""


# ---------------------------------------------------------------------------
# 限速 + 重试
# ---------------------------------------------------------------------------
class _Throttle:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.min_interval
        if delay > 0:
            _sleep(delay)


# 某个来源连续重试仍被限流后，冷却一段时间直接跳过，避免每个检索式都白等一轮重试
_COOLDOWN_SECONDS = 60.0
_cooldown_until: dict[str, float] = {}

_THROTTLES = {
    "semantic_scholar": _Throttle(1.05),
    "arxiv": _Throttle(3.0),
    "openalex": _Throttle(0.15),
}


def _retry_after(r: httpx.Response, attempt: int) -> float:
    ra = r.headers.get("retry-after")
    if ra and ra.strip().isdigit():
        return min(float(ra), 30.0)
    return min(2.0 ** attempt, 16.0)  # 1, 2, 4, 8 …


def _get(source: str, url: str, params: dict, headers: dict | None = None, retries: int = 3) -> httpx.Response:
    left = _cooldown_until.get(source, 0.0) - time.monotonic()
    if left > 0:
        raise SourceError(f"{SOURCE_NAMES[source]} 刚被限流，{int(left)} 秒内暂停使用")
    hdrs = dict(_UA, **(headers or {}))
    if _no_retry.get():
        retries = 0
    with httpx.Client(transport=_TRANSPORT, timeout=30, follow_redirects=True) as client:
        for attempt in range(retries + 1):
            _THROTTLES[source].wait()
            try:
                r = client.get(url, params=params, headers=hdrs)
            except httpx.TransportError as e:
                if attempt == retries:
                    raise SourceError(f"{SOURCE_NAMES[source]} 连接失败：{e}") from e
                _sleep(min(2.0 ** attempt, 8.0))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == retries:
                    if r.status_code == 429:
                        _cooldown_until[source] = time.monotonic() + _COOLDOWN_SECONDS
                        raise SourceError(f"{SOURCE_NAMES[source]} 请求太频繁被限流（429），已重试 {retries} 次")
                    raise SourceError(f"{SOURCE_NAMES[source]} 服务暂时不可用（{r.status_code}）")
                _sleep(_retry_after(r, attempt))
                continue
            if r.status_code >= 400:
                raise SourceError(f"{SOURCE_NAMES[source]} 请求失败（{r.status_code}）")
            return r
    raise SourceError(f"{SOURCE_NAMES[source]} 请求失败")  # pragma: no cover


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def _cache_get(key: str) -> list[PaperIn] | None:
    days = get_settings().search_cache_days
    if days <= 0:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT results FROM search_cache WHERE key=? AND created_at >= datetime('now', ?)", (key, f"-{days} days")
        ).fetchone()
    return [PaperIn(**d) for d in json.loads(row["results"])] if row else None


def _cache_put(key: str, papers: list[PaperIn]) -> None:
    if get_settings().search_cache_days <= 0:
        return
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO search_cache (key, results, created_at) VALUES (?, ?, datetime('now'))",
            (key, json.dumps([p.model_dump() for p in papers], ensure_ascii=False)),
        )


@tracing.traced("tool", "查询学术来源")
def _cached(source: str, fn: Callable[[str, int], list[PaperIn]], query: str, limit: int) -> list[PaperIn]:
    tracing.annotate(source=source)
    key = f"{source}|{limit}|{' '.join(query.lower().split())}"
    hit = _cache_get(key)
    if hit is not None:
        tracing.annotate(cache_hit=True)
        return hit
    tracing.annotate(cache_hit=False)
    res = fn(query, limit)
    _cache_put(key, res)
    return res


# ---------------------------------------------------------------------------
# 各来源
# ---------------------------------------------------------------------------
def search_arxiv(query: str, limit: int = 10) -> list[PaperIn]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit, "sortBy": "relevance"}
    r = _get("arxiv", "https://export.arxiv.org/api/query", params)
    root = ET.fromstring(r.text)
    out: list[PaperIn] = []
    for e in root.findall("a:entry", _ATOM):
        abs_url = (e.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
        arxiv_id = abs_url.rsplit("/abs/", 1)[-1] if "/abs/" in abs_url else None
        pdf_url = None
        for link in e.findall("a:link", _ATOM):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        published = e.findtext("a:published", default="", namespaces=_ATOM) or ""
        out.append(
            PaperIn(
                title=re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=_ATOM)).strip(),
                authors=[a.findtext("a:name", default="", namespaces=_ATOM) for a in e.findall("a:author", _ATOM)],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=e.findtext("arxiv:journal_ref", default=None, namespaces=_ATOM),
                doi=e.findtext("arxiv:doi", default=None, namespaces=_ATOM),
                arxiv_id=arxiv_id,
                url=abs_url or None,
                pdf_url=pdf_url,
                abstract=re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=_ATOM)).strip(),
                source="arxiv",
            )
        )
    return out


def search_semantic_scholar(query: str, limit: int = 10) -> list[PaperIn]:
    fields = "title,authors,year,venue,externalIds,url,abstract,openAccessPdf,citationCount"
    key = get_settings().semantic_scholar_api_key
    r = _get(
        "semantic_scholar",
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": query, "limit": limit, "fields": fields},
        {"x-api-key": key} if key else None,
    )
    out: list[PaperIn] = []
    for d in r.json().get("data", []) or []:
        ext = d.get("externalIds") or {}
        pdf = (d.get("openAccessPdf") or {}).get("url")
        arxiv_id = ext.get("ArXiv")
        out.append(
            PaperIn(
                title=d.get("title") or "",
                authors=[a.get("name", "") for a in d.get("authors") or []],
                year=d.get("year"),
                venue=d.get("venue") or None,
                doi=ext.get("DOI"),
                arxiv_id=arxiv_id,
                url=d.get("url"),
                pdf_url=pdf or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
                abstract=d.get("abstract"),
                source="semantic_scholar",
                citation_count=d.get("citationCount"),
            )
        )
    return out


def _openalex_abstract(inv: dict | None) -> str | None:
    """OpenAlex 的摘要是倒排索引 {词: [位置…]}，还原成原文"""
    if not inv:
        return None
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos)) or None


def search_openalex(query: str, limit: int = 10) -> list[PaperIn]:
    params: dict = {"search": query, "per-page": limit}
    key = get_settings().openalex_api_key
    if key:
        params["api_key"] = key
    r = _get("openalex", "https://api.openalex.org/works", params)
    out: list[PaperIn] = []
    for w in r.json().get("results", []) or []:
        doi = (w.get("doi") or "").removeprefix("https://doi.org/") or None
        arxiv_id = None
        if doi and doi.lower().startswith("10.48550/arxiv."):
            arxiv_id, doi = doi.split(".", 2)[-1].removeprefix("arXiv.").removeprefix("arxiv."), None
        loc = w.get("primary_location") or {}
        best = w.get("best_oa_location") or {}
        out.append(
            PaperIn(
                title=w.get("display_name") or w.get("title") or "",
                authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
                year=w.get("publication_year"),
                venue=((loc.get("source") or {}).get("display_name")) or None,
                doi=doi,
                arxiv_id=arxiv_id,
                url=loc.get("landing_page_url") or w.get("id"),
                pdf_url=best.get("pdf_url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
                abstract=_openalex_abstract(w.get("abstract_inverted_index")),
                source="openalex",
                citation_count=w.get("cited_by_count"),
            )
        )
    return out


_FNS: dict[str, Callable[[str, int], list[PaperIn]]] = {
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "openalex": search_openalex,
}


def _merge(a: PaperIn, b: PaperIn) -> PaperIn:
    data = a.model_dump()
    for k, v in b.model_dump().items():
        if data.get(k) in (None, "", []) and v not in (None, "", []):
            data[k] = v
    return PaperIn(**data)


def dedupe(papers: list[PaperIn]) -> list[PaperIn]:
    out: list[PaperIn] = []
    index: dict[str, int] = {}
    for p in papers:
        if not p.title:
            continue
        keys = [f"t:{normalize_title(p.title)}"]
        if p.doi:
            keys.append(f"d:{p.doi.lower()}")
        if p.arxiv_id:
            keys.append(f"a:{p.arxiv_id.split('v')[0]}")
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            out.append(p)
            hit = len(out) - 1
        else:
            out[hit] = _merge(out[hit], p)
        for k in keys:
            index[k] = hit
    return out


@tracing.traced("tool", "检索学术文献")
def search(query: str, sources: list[str] | None = None, limit: int = 8) -> tuple[list[PaperIn], list[str]]:
    """并发检索多个来源；返回 (结果, 提示信息列表)。单个来源失败不影响其他来源，
    Semantic Scholar 失败时自动用 OpenAlex 补位。"""
    if get_settings().llm_fake:
        return demo_search(query, limit), []
    sources = [s for s in (sources or get_settings().search_sources) if s in _FNS]
    results: list[PaperIn] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as ex:
        futures = {src: ex.submit(copy_context().run, _cached, src, _FNS[src], query, limit) for src in sources}
        for src, fut in futures.items():
            try:
                results.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                errors[src] = str(e) if isinstance(e, SourceError) else f"{SOURCE_NAMES.get(src, src)}：{e}"
    notes: list[str] = []
    if "semantic_scholar" in errors and "openalex" not in sources:
        try:
            results.extend(_cached("openalex", search_openalex, query, limit))
            errors["semantic_scholar"] += "，已改用 OpenAlex 补位"
        except Exception as e:  # noqa: BLE001
            notes.append(f"OpenAlex 补位也失败了：{e}")
    notes = list(errors.values()) + notes
    if "semantic_scholar" in errors and not get_settings().semantic_scholar_api_key:
        notes.append("提示：在 backend/.env 里配置 SEMANTIC_SCHOLAR_API_KEY（免费申请）可以避免共享配额限流")
    return dedupe(results), list(dict.fromkeys(notes))


def test_source(source: str) -> dict:
    """设置页「测试」：不走缓存、不重试，直接发一次检索"""
    if source not in _FNS:
        raise ValueError(f"未知来源：{source}")
    _cooldown_until.pop(source, None)
    start = time.monotonic()
    key = get_settings().semantic_scholar_api_key if source == "semantic_scholar" else get_settings().openalex_api_key if source == "openalex" else None
    try:
        if source == "arxiv":
            res = search_arxiv("transformer", 1)
        elif source == "semantic_scholar":
            res = _probe(lambda: search_semantic_scholar("transformer", 1))
        else:
            res = _probe(lambda: search_openalex("transformer", 1))
        return {"ok": True, "latency_ms": int((time.monotonic() - start) * 1000), "sample": res[0].title if res else None,
                "authenticated": bool(key)}
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ("被限流：建议配置 API Key" if "429" in msg else "key 无效或没有权限" if ("401" in msg or "403" in msg)
                else "连不上服务，检查网络或代理" if "连接失败" in msg else "检索失败")
        return {"ok": False, "latency_ms": int((time.monotonic() - start) * 1000), "hint": hint, "error": msg, "authenticated": bool(key)}


def _probe(fn: Callable[[], list[PaperIn]]) -> list[PaperIn]:
    """测试时不重试：直接把限流/鉴权错误暴露出来"""
    token = _no_retry.set(True)
    try:
        return fn()
    finally:
        _no_retry.reset(token)


def download_pdf(url: str, dest: Path, max_bytes: int | None = None) -> Path:
    """下载 PDF。链接来自文献元数据（用户可编辑），所以每一跳（含重定向）都检查不能指向内网 / 本机。"""
    from .security import check_public_url, max_upload_bytes

    limit = max_bytes or max_upload_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    check_public_url(url)
    hooks = {"request": [lambda req: check_public_url(str(req.url))]}
    tmp = dest.with_suffix(".part")
    try:
        with httpx.Client(transport=_TRANSPORT, timeout=60, follow_redirects=True, event_hooks=hooks) as client:
            with client.stream("GET", url, headers=_UA) as r:
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                size = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes():
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"PDF 超过 {limit // (1024 * 1024)} MB 上限")
                        f.write(chunk)
        with open(tmp, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError(f"下载的不是 PDF（content-type: {ctype}）")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


# ---------------------------------------------------------------------------
# 离线演示数据（LLM_FAKE=1 时使用；均为真实论文的元数据）
# ---------------------------------------------------------------------------
_DEMO = [
    PaperIn(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones",
                 "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        year=2017, venue="NeurIPS", arxiv_id="1706.03762", url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural "
        "networks. We propose a new simple network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely.",
        source="demo", citation_count=100000,
    ),
    PaperIn(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin",
                 "Naman Goyal", "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel",
                 "Sebastian Riedel", "Douwe Kiela"],
        year=2020, venue="NeurIPS", arxiv_id="2005.11401", url="https://arxiv.org/abs/2005.11401",
        pdf_url="https://arxiv.org/pdf/2005.11401",
        abstract="Large pre-trained language models store factual knowledge in their parameters, but their ability "
        "to access and precisely manipulate knowledge is limited. We explore retrieval-augmented generation (RAG) "
        "models which combine pre-trained parametric and non-parametric memory for language generation.",
        source="demo",
    ),
    PaperIn(
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        authors=["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran", "Karthik Narasimhan", "Yuan Cao"],
        year=2023, venue="ICLR", arxiv_id="2210.03629", url="https://arxiv.org/abs/2210.03629",
        pdf_url="https://arxiv.org/pdf/2210.03629",
        abstract="We explore the use of LLMs to generate both reasoning traces and task-specific actions in an "
        "interleaved manner, allowing for greater synergy between the two.",
        source="demo",
    ),
    PaperIn(
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        authors=["Akari Asai", "Zeqiu Wu", "Yizhong Wang", "Avirup Sil", "Hannaneh Hajishirzi"],
        year=2024, venue="ICLR", arxiv_id="2310.11511", url="https://arxiv.org/abs/2310.11511",
        pdf_url="https://arxiv.org/pdf/2310.11511",
        abstract="We introduce Self-Reflective Retrieval-Augmented Generation (Self-RAG), which enhances an LM's "
        "quality and factuality through retrieval and self-reflection.",
        source="demo",
    ),
    PaperIn(
        title="Dense Passage Retrieval for Open-Domain Question Answering",
        authors=["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis", "Ledell Wu", "Sergey Edunov",
                 "Danqi Chen", "Wen-tau Yih"],
        year=2020, venue="EMNLP", arxiv_id="2004.04906", doi="10.18653/v1/2020.emnlp-main.550",
        url="https://arxiv.org/abs/2004.04906", pdf_url="https://arxiv.org/pdf/2004.04906",
        abstract="We show that retrieval can be practically implemented using dense representations alone, where "
        "embeddings are learned from a small number of questions and passages by a simple dual-encoder framework.",
        source="demo",
    ),
    PaperIn(
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        year=2019, venue="NAACL", arxiv_id="1810.04805", doi="10.18653/v1/N19-1423",
        url="https://arxiv.org/abs/1810.04805", pdf_url="https://arxiv.org/pdf/1810.04805",
        abstract="We introduce a new language representation model called BERT, designed to pre-train deep "
        "bidirectional representations from unlabeled text by jointly conditioning on both left and right context.",
        source="demo",
    ),
]


def demo_search(query: str, limit: int) -> list[PaperIn]:
    q = set(re.findall(r"[a-z]+", query.lower()))
    scored = sorted(_DEMO, key=lambda p: -len(q & set(re.findall(r"[a-z]+", (p.title + " " + (p.abstract or "")).lower()))))
    return [p.model_copy() for p in scored[:limit]]
