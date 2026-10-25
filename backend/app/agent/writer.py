"""分节写作 Agent（LangGraph）：按确认过的大纲写选中的章节，边写边流式输出，写完自查引用，最后可由总编统稿。

三种节奏：
  逐节确认  prepare → write_section → review_section ⏸ ─ accept / skip ─→ advance → … → [polish] → finish
                          ▲                │
                          └── regenerate ──┘（带上用户意见重写本节）
  连续写完  同上，但 review_section 不暂停
  并行写    prepare ─Send→ write_one × N（每节一个子 Agent，同时写）→ collect → [polish 总编统稿] → finish

每一节（无论谁来写）都走同一套流程：
  1. 检索：用「章节标题 + 要点」在文献库里找证据（优先大纲里给这一节分配的文献），再带上用户在原文上记的笔记；
  2. 写作：流式输出（token 事件实时发给前端，直接写进文档里这一节的位置），只允许引用证据里出现的文献；
  3. 自查：逐句核对引用的原文是否支撑论断，撑不起的句子让模型改写一次；
  4. 再跑一遍完整批注，结果随草稿一起返回，前端直接显示成页边批注。
总编（polish）通读所有新写的章节，只改衔接、重复和前后不一致的地方，每处修改单独发给前端，在文档里逐处呈现。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from .. import llm, tracing
from ..assist import CITE_TOKEN, annotate, cite_keys, retrieval_query, strip_cites
from ..config import get_settings
from ..db import get_conn
from ..library import get_paper, list_papers
from ..retrieval import search

LENGTHS = {
    "short": ("1–2 段，共约 250 字", "1-2 paragraphs, about 180 words in total"),
    "medium": ("2–3 段，共约 500 字", "2-3 paragraphs, about 350 words in total"),
    "long": ("3–5 段，共约 900 字", "3-5 paragraphs, about 650 words in total"),
}


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class WriterState(TypedDict, total=False):
    topic: str
    doc_title: str
    language: Literal["zh", "en"]
    paper_type: str
    outline: list[dict]  # [{title, goal, bibkeys}]
    mode: Literal["confirm", "auto", "parallel"]
    length: Literal["short", "medium", "long"]
    selected: list[int] | None  # 要写的章节（大纲下标）；None = 全部
    polish: bool  # 写完后由总编统稿
    order: list[int]  # 实际要写的章节，按大纲顺序
    pos: int  # 逐节模式：当前写到 order 的第几个
    index: int  # 当前章节在大纲里的下标
    drafts: list[dict]  # 每节一个：{title, paragraphs, annotations, fixed, issues, evidence, status}
    parts: Annotated[dict, _merge]  # 并行模式：各子 Agent 的结果 {"下标": draft}
    edits: list[dict]  # 总编的修改
    polish_summary: str
    polish_done: bool
    action: str
    feedback: str
    errors: list[str]


def _progress(message: str, **extra: Any) -> None:
    tracing.progress(message, **extra)
    try:
        get_stream_writer()({"message": message, **extra})
    except Exception:  # 不在 stream 上下文时忽略
        pass


# ---------------------------------------------------------------------------
# 证据：检索片段 + 用户笔记
# ---------------------------------------------------------------------------
@tracing.traced("tool", "收集证据与用户笔记")
def gather_evidence(section: dict, topic: str, limit: int = 8) -> list[dict]:
    by_key = {p.bibkey: p for p in list_papers()}
    allowed = [by_key[k].id for k in section.get("bibkeys") or [] if k in by_key]
    query = retrieval_query(f"{section.get('title', '')} {section.get('goal', '')} {topic}".strip())
    hits = search(query, top_k=limit, paper_ids=allowed or None)
    if allowed and len(hits) < 4:  # 分配的文献太少：再从整个文献库补几条
        seen = {h.chunk_id for h in hits}
        hits += [h for h in search(query, top_k=4) if h.chunk_id not in seen]
    ev: list[dict] = []
    for h in hits:
        p = get_paper(h.paper_id)
        if p:
            ev.append({"bibkey": p.bibkey, "paper_id": p.id, "title": p.title, "year": p.year,
                       "page": h.page, "section": h.section, "text": h.text[:700], "source": "chunk"})
    # 用户在原文上记的笔记：比检索片段更能代表「用户想怎么用这篇文献」，排在前面
    pids = {e["paper_id"] for e in ev} | set(allowed)
    if pids:
        marks = ",".join("?" * len(pids))
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT paper_id, page, quote, comment FROM notes WHERE paper_id IN ({marks}) ORDER BY updated_at DESC LIMIT 6",
                tuple(pids),
            ).fetchall()
        for r in rows:
            p = get_paper(r["paper_id"])
            if p:
                text = r["quote"] + (f"（用户笔记：{r['comment']}）" if r["comment"] else "")
                ev.insert(0, {"bibkey": p.bibkey, "paper_id": p.id, "title": p.title, "year": p.year,
                              "page": r["page"], "section": None, "text": text[:700], "source": "note"})
    return ev


def _evidence_block(ev: list[dict]) -> str:
    lines = []
    for i, e in enumerate(ev, 1):
        where = f"p.{e['page']}" if e.get("page") else "页码未知"
        tag = "用户笔记" if e["source"] == "note" else (e.get("section") or "片段")
        lines.append(f"[E{i}] @{e['bibkey']}（{e['title'][:60]}, {e.get('year') or 'n.d.'}）{where} · {tag}\n{e['text']}")
    return "\n\n".join(lines)


def clean_citations(text: str, allowed: set[str]) -> str:
    """只保留证据里出现过的 bibkey；全部不合法的引用整段去掉（绝不留下编造的引用）"""

    def fix(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
        keep = [p for p in parts if p.split(",")[0].strip().lstrip("@") in allowed]
        return f"[{'; '.join(keep)}]" if keep else ""

    out = CITE_TOKEN.sub(fix, text)
    return re.sub(r"[ \t]+([，。；：,.;:])", r"\1", out).strip()


# ---------------------------------------------------------------------------
# 写作（流式）与自查
# ---------------------------------------------------------------------------
def _prompt(state: WriterState, sec: dict, ev: list[dict], feedback: str) -> list[dict[str, str]]:
    zh = state.get("language", "zh") == "zh"
    outline = "\n".join(f"{i + 1}. {s['title']}：{s.get('goal', '')}" for i, s in enumerate(state["outline"]))
    prev = [d for d in state.get("drafts") or [] if d.get("status") == "accepted" and d.get("paragraphs")]
    prev_txt = ""
    if prev and state.get("mode") != "parallel":
        last = prev[-1]
        prev_txt = f"\n上一节《{last['title']}》的开头：{strip_cites(' '.join(last['paragraphs']))[:300]}…"
    length = LENGTHS.get(state.get("length", "medium"), LENGTHS["medium"])[0 if zh else 1]
    rule = (
        "引用规则：只能引用证据里出现的文献，写成 [@bibkey]；证据给出了页码时写成 [@bibkey, p. 页码]。"
        "每条引用都必须能在对应证据里找到依据；证据不足以支撑的说法不要写，或者写成不带引用的一般性表述。"
        "不要编造数据、实验结果和文献。"
        if zh
        else "Citation rules: cite only papers that appear in the evidence, as [@bibkey], or [@bibkey, p. N] when the evidence gives a page. "
        "Every citation must be supported by the cited evidence; drop claims the evidence cannot support. Never invent numbers, results or papers."
    )
    fb = f"\n\n用户对上一版的意见：{feedback}" if feedback else ""
    user = (
        f"论文题目：{state.get('doc_title') or state.get('topic')}\n论文类型：{state.get('paper_type') or '学术论文'}\n"
        f"全文大纲：\n{outline}\n\n现在写：《{sec['title']}》——{sec.get('goal', '')}{prev_txt}\n\n"
        f"证据：\n{_evidence_block(ev) or '（文献库里没有找到相关证据）'}\n\n{rule}\n"
        f"篇幅：{length}。用{'中文' if zh else 'English'}写，学术语体，段落之间逻辑连贯。{fb}\n"
        "直接输出正文：段落之间空一行；不要写章节标题，不要 Markdown，不要 JSON，不要任何解释。"
    )
    system = "你是严谨的学术写作者，根据给定证据写论文的一节。" if zh else "You are a careful academic writer."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _fake_draft(sec: dict, ev: list[dict]) -> str:
    keys = sorted({e["bibkey"] for e in ev})
    k0 = keys[0] if keys else None
    page0 = next((e["page"] for e in ev if e["bibkey"] == k0 and e.get("page")), None)
    p1 = (
        f"{sec['title']}部分围绕“{sec.get('goal') or sec['title']}”展开。"
        + (f"已有研究在生成答案之前先检索外部知识，从而减少大语言模型的事实错误[@{k0}{', p. %s' % page0 if page0 else ''}]。" if k0 else "")
        + "（离线演示）"
    )
    p2 = "本节随后讨论该方向仍然存在的问题，并说明本文的切入点。" + (f"相关方法同样依赖检索质量[@{keys[-1]}]。" if keys else "")
    return f"{p1}\n\n{p2}"


def _split_paragraphs(text: str, title: str) -> list[str]:
    out = []
    for line in re.split(r"\n\s*\n|\n", text.strip()):
        t = line.strip().strip("`").strip()
        if not t or t.startswith("#") or t.replace(" ", "") == title.replace(" ", ""):
            continue
        out.append(t)
    return out


def _stream_draft(state: WriterState, i: int, sec: dict, ev: list[dict], feedback: str) -> list[str]:
    """流式写一节：边生成边把增量发给前端（攒够一小段再发，避免事件过多）"""
    buf, full, last = "", "", time.perf_counter()
    for delta in llm.stream_chat(_prompt(state, sec, ev, feedback), tier="agent", fake=_fake_draft(sec, ev)):
        buf += delta
        full += delta
        if len(buf) >= 16 or time.perf_counter() - last > 0.08:
            _progress("", section=i, phase="token", delta=buf)
            buf, last = "", time.perf_counter()
    if buf:
        _progress("", section=i, phase="token", delta=buf)
    allowed = {e["bibkey"] for e in ev}
    return [p for p in (clean_citations(p, allowed) for p in _split_paragraphs(full, sec["title"])) if p]


def _bad_items(anns: list[dict]) -> list[dict]:
    out = []
    for a in anns:
        if a.get("kind") != "cite_check":
            continue
        for it in a.get("items") or []:
            if it.get("verdict") in {"unrelated", "missing"}:
                out.append({"sentence": a["anchor"], "bibkey": it.get("bibkey"), "reason": it.get("reason", "")})
    return out


def _repair(state: WriterState, paragraph: str, bad: list[dict], ev: list[dict]) -> str:
    zh = state.get("language", "zh") == "zh"
    problems = "\n".join(f"- “{b['sentence']}” 引用 @{b['bibkey']}：{b['reason']}" for b in bad)
    user = (
        f"下面这段论文里，有的引用经核查撑不起对应的句子：\n{problems}\n\n段落：\n{paragraph}\n\n"
        f"证据：\n{_evidence_block(ev)}\n\n"
        "请只修改有问题的句子：换成证据里确实能支撑它的文献，或者收窄说法使其与证据一致，"
        "实在没有依据就去掉这条引用并改成一般性表述。其他句子原样保留，引用格式不变。"
        '只输出 JSON：{"paragraph": "修改后的段落"}'
    )
    fake = json.dumps({"paragraph": CITE_TOKEN.sub(lambda m: m.group(0) if all(b["bibkey"] not in cite_keys(m.group(1)) for b in bad) else "", paragraph)}, ensure_ascii=False)
    data = llm.chat_json(
        [{"role": "system", "content": "你是严谨的学术编辑。" if zh else "You are a careful academic editor."}, {"role": "user", "content": user}],
        tier="agent", fake=fake, max_tokens=2000,
    )
    fixed = str(data.get("paragraph") or "").strip()
    return clean_citations(fixed, {e["bibkey"] for e in ev}) if fixed else paragraph


def _write(state: WriterState, i: int) -> dict:
    """写第 i 节：检索 → 流式写作 → 自查引用 → 批注。返回草稿。"""
    sec = state["outline"][i]
    title = sec["title"]
    _progress(f"《{title}》：检索证据…", section=i, phase="retrieve")
    ev = gather_evidence(sec, state.get("topic", ""))
    notes_n = sum(1 for e in ev if e["source"] == "note")
    _progress(f"《{title}》：找到 {len(ev)} 条证据" + (f"（含 {notes_n} 条你的笔记）" if notes_n else "") + "，开始写作…", section=i, phase="draft")
    paragraphs = _stream_draft(state, i, sec, ev, state.get("feedback") or "")

    _progress(f"《{title}》：自查引用…", section=i, phase="check")
    fixed = 0
    title_ctx = state.get("doc_title") or state.get("topic", "")
    for j, para in enumerate(paragraphs):
        bad = _bad_items(annotate(para, title_ctx, check_citations=True, review=False))
        if bad:
            _progress(f"《{title}》：第 {j + 1} 段有 {len(bad)} 处引用撑不起论断，正在修改…", section=i, phase="fix")
            new = _repair(state, para, bad, ev)
            after = _bad_items(annotate(new, title_ctx, check_citations=True, review=False))
            if len(after) < len(bad):
                fixed += len(bad) - len(after)
                paragraphs[j] = new

    _progress(f"《{title}》：生成页边批注…", section=i, phase="annotate")
    annotations = [annotate(p, title_ctx, check_citations=True, review=True) for p in paragraphs]
    issues = sum(1 for anns in annotations for a in anns if a["kind"] != "cite_check" or any(it.get("verdict") != "supports" for it in a.get("items") or []))
    draft = {
        "title": title, "paragraphs": paragraphs, "annotations": annotations, "fixed": fixed, "issues": issues,
        "evidence": [{k: e[k] for k in ("bibkey", "page", "source")} for e in ev], "status": "drafted",
    }
    # 自查后的定稿（可能改过句子、去掉了引用）：前端用它替换流式写进文档的初稿
    _progress(f"《{title}》写完了", section=i, phase="done", draft=draft)
    return draft


# ---------------------------------------------------------------------------
# 总编统稿
# ---------------------------------------------------------------------------
def _fake_polish(items: list[tuple[int, dict]]) -> dict:
    edits = []
    for n, (_, d) in enumerate(items[1:3], start=2):
        p = d["paragraphs"][0]
        if not p.startswith("承接上一节"):
            edits.append({"section": n, "paragraph": 1, "revised": "承接上一节的讨论，" + p, "reason": "补上与上一节的过渡（离线演示）"})
    return {"summary": "各节内容基本衔接，补了几处过渡。（离线演示）", "edits": edits}


@tracing.traced("node", "总编统稿")
def polish(state: WriterState) -> dict:
    drafts = [dict(d) for d in state.get("drafts") or []]
    items = [(i, d) for i, d in enumerate(drafts) if d.get("status") in ("accepted", "drafted") and d.get("paragraphs")]
    if not items:
        return {"polish_done": True}
    zh = state.get("language", "zh") == "zh"
    _progress(f"总编 Agent：通读新写的 {len(items)} 节，检查衔接、重复和前后一致…", phase="polish")
    body = "\n\n".join(
        f"§{n}《{d['title']}》\n" + "\n".join(f"¶{j + 1} {p}" for j, p in enumerate(d["paragraphs"]))
        for n, (_, d) in enumerate(items, start=1)
    )
    outline = "\n".join(f"{i + 1}. {s['title']}：{s.get('goal', '')}" for i, s in enumerate(state["outline"]))
    user = (
        f"论文题目：{state.get('doc_title') or state.get('topic')}\n全文大纲：\n{outline}\n\n"
        f"下面这些章节是几位作者分头写的（§ 是章节序号，¶ 是段落序号，引用写成 [@bibkey]）：\n\n{body}\n\n"
        "你是这篇论文的总编。请只做必要的修改：章节之间的衔接与过渡、重复的内容、术语和说法前后不一致、偏离大纲目标的段落。"
        "不要为了改而改，最多 6 处；每处给出修改后的整段，保留原有引用，不要新增文献。"
        '只输出 JSON：{"summary": "一两句总体意见", "edits": [{"section": 章节序号, "paragraph": 段落序号, "revised": "修改后的整段", "reason": "为什么改"}]}'
    )
    try:
        data = llm.chat_json(
            [{"role": "system", "content": "你是严谨的学术期刊编辑。" if zh else "You are a careful journal editor."}, {"role": "user", "content": user}],
            tier="agent", fake=lambda _m: json.dumps(_fake_polish(items), ensure_ascii=False), max_tokens=4000,
        ) or {}
    except Exception as e:  # noqa: BLE001
        return {"errors": list(state.get("errors") or []) + [f"总编统稿失败：{type(e).__name__}: {e}"]}

    title_ctx = state.get("doc_title") or state.get("topic", "")
    edits = []
    for e in (data.get("edits") or [])[:6]:
        try:
            n, j = int(e.get("section")) - 1, int(e.get("paragraph")) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= n < len(items)):
            continue
        i, d = items[n]
        d = drafts[i]
        if not (0 <= j < len(d["paragraphs"])):
            continue
        before = d["paragraphs"][j]
        # 只允许用这一段原有的、或这一节证据里的文献
        allowed = {ev["bibkey"] for ev in d.get("evidence") or []} | {k for m in CITE_TOKEN.finditer(before) for k in cite_keys(m.group(1))}
        after = clean_citations(str(e.get("revised") or "").strip(), allowed)
        if not after or after == before:
            continue
        anns = annotate(after, title_ctx, check_citations=True, review=False)
        paragraphs = list(d["paragraphs"])
        paragraphs[j] = after
        annotations = list(d.get("annotations") or [[] for _ in paragraphs])
        if j < len(annotations):
            annotations[j] = anns
        drafts[i] = {**d, "paragraphs": paragraphs, "annotations": annotations}
        edit = {"section": i, "title": d["title"], "para": j, "before": before, "after": after, "reason": str(e.get("reason") or "").strip(), "annotations": anns}
        edits.append(edit)
        _progress(f"总编 Agent：修改《{d['title']}》第 {j + 1} 段", phase="edit", **edit)
    summary = str(data.get("summary") or "").strip()
    _progress(f"总编 Agent：改了 {len(edits)} 处" if edits else "总编 Agent：通读完毕，没有需要修改的地方", phase="polished")
    return {"drafts": drafts, "edits": edits, "polish_summary": summary, "polish_done": True}


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
@tracing.traced("node", "准备写作")
def prepare(state: WriterState) -> dict:
    outline = state.get("outline") or []
    sel = state.get("selected")
    order = sorted({i for i in sel if 0 <= i < len(outline)}) if sel else list(range(len(outline)))
    drafts = [{"title": s["title"], "status": "pending" if i in order else "excluded"} for i, s in enumerate(outline)]
    how = {"parallel": f"{len(order)} 个子 Agent 并行写", "auto": "连续写完"}.get(state.get("mode", ""), "逐节确认")
    _progress(f"共 {len(order)} 节，{how}" + ("，最后由总编统稿" if state.get("polish") else ""))
    return {"order": order, "pos": 0, "index": order[0] if order else 0, "drafts": drafts, "errors": [], "parts": {}, "edits": [], "polish_done": False}


@tracing.traced("node", "写作本节")
def write_section(state: WriterState) -> dict:
    i = state["index"]
    drafts = list(state["drafts"])
    drafts[i] = _write(state, i)
    return {"drafts": drafts, "feedback": ""}


@tracing.traced("node", "并行章节写作")
def write_one(state: WriterState) -> dict:
    """并行模式的子 Agent：只写一节。出错只影响这一节"""
    i = state["index"]
    try:
        return {"parts": {str(i): _write(state, i)}}
    except Exception as e:  # noqa: BLE001
        msg = f"《{state['outline'][i]['title']}》写作失败：{type(e).__name__}: {e}"
        _progress(msg, section=i, phase="error")
        return {"parts": {str(i): {"title": state["outline"][i]["title"], "status": "error", "error": msg}}}


@tracing.traced("node", "汇总章节")
def collect(state: WriterState) -> dict:
    drafts = list(state["drafts"])
    errors = list(state.get("errors") or [])
    for k, d in (state.get("parts") or {}).items():
        drafts[int(k)] = d
        if d.get("status") == "error":
            errors.append(d.get("error", ""))
    _progress(f"子 Agent 都写完了：{sum(1 for d in drafts if d.get('status') == 'drafted')} 节")
    return {"drafts": drafts, "errors": errors}


@tracing.traced("node", "确认本节")
def review_section(state: WriterState) -> dict:
    i = state["index"]
    drafts = list(state["drafts"])
    if state.get("mode") == "auto":
        return {"drafts": drafts, "action": "accept"}  # 草稿留在文档里，最后统一保留或删除
    order = state.get("order") or [i]
    answer = interrupt({
        "kind": "review_section",
        "message": f"第 {state.get('pos', 0) + 1}/{len(order)} 节写好了，请确认",
        "index": i,
        "pos": state.get("pos", 0),
        "total": len(order),
        "draft": drafts[i],
    }) or {}
    action = answer.get("action", "accept")
    if action == "regenerate":
        return {"action": "regenerate", "feedback": answer.get("feedback") or "换一种写法"}
    drafts[i] = {**drafts[i], "status": "accepted" if action == "accept" else "skipped"}
    return {"drafts": drafts, "action": action}


@tracing.traced("node", "下一节")
def advance(state: WriterState) -> dict:
    pos = state.get("pos", 0) + 1
    return {"pos": pos, "index": state["order"][pos]}


@tracing.traced("node", "完成")
def finish(state: WriterState) -> dict:
    drafts = state.get("drafts") or []
    n = sum(1 for d in drafts if d.get("status") in ("accepted", "drafted"))
    _progress(f"完成：写了 {n} 节" + (f"，总编改了 {len(state.get('edits') or [])} 处" if state.get("polish") else ""))
    return {}


def _after_write(state: WriterState) -> str:
    return "polish" if state.get("polish") else "finish"


def _route_start(state: WriterState):
    if not state.get("order"):
        return "finish"
    if state.get("mode") == "parallel":
        return [Send("write_one", {**state, "index": i}) for i in state["order"]]
    return "write_section"


def _route_review(state: WriterState) -> str:
    a = state.get("action")
    if a == "regenerate":
        return "write_section"
    if a == "stop" or state.get("pos", 0) + 1 >= len(state.get("order") or []):
        return _after_write(state)
    return "advance"


def build_writer(checkpointer: Any = None):
    g = StateGraph(WriterState)
    g.add_node("prepare", prepare)
    g.add_node("write_section", write_section)
    g.add_node("write_one", write_one)
    g.add_node("collect", collect)
    g.add_node("review_section", review_section)
    g.add_node("advance", advance)
    g.add_node("polish", polish)
    g.add_node("finish", finish)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", _route_start, ["write_section", "write_one", "finish"])
    g.add_edge("write_one", "collect")
    g.add_conditional_edges("collect", _after_write, ["polish", "finish"])
    g.add_edge("write_section", "review_section")
    g.add_conditional_edges("review_section", _route_review, ["write_section", "advance", "polish", "finish"])
    g.add_edge("advance", "write_section")
    g.add_edge("polish", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


_writer = None


def get_writer():
    global _writer
    if _writer is None:
        conn = sqlite3.connect(str(get_settings().data_dir / "checkpoints.sqlite"), check_same_thread=False)
        _writer = build_writer(SqliteSaver(conn))
    return _writer
