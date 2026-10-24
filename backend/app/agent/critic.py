"""审稿 Agent（LangGraph）：模拟审稿人挑逻辑和论据上的问题，作者修改后再审，最多若干轮。

    review ──(还有问题且没到轮数上限)──→ await_revision ⏸ ── again ──→ review …
       │                                       └── stop ──→ finish
       └──(没有问题 / 到达上限)──→ finish

- review：第 1 轮通读全文，按「逻辑 / 论据 / 结论过度 / 结构 / 表述」挑问题，每条都摘录原文中的一句话作为定位，
  前端据此把意见挂到正文旁边的页边批注里；论据类问题顺带在文献库里检索几篇可能用得上的文献。
  第 2 轮起先逐条判断上一轮的问题是否解决（结合作者的修改和回复），再只补充新的主要问题，保证收敛。
- await_revision：interrupt() 暂停，作者在正文里修改、逐条标记「已修改 / 不改（附理由）」后点「再审一轮」，
  前端把最新的全文和回复一起发回来；也可以直接结束。
- 引用是否真的支撑论断由页边批注逐句核查，审稿人只看论证本身，不重复做这件事。
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .. import llm, tracing
from ..assist import CITE_TOKEN, cite_keys, split_sentences, strip_cites, suggest_citations
from ..config import get_settings
from ..library import list_papers

CATEGORIES = {
    "logic": "逻辑",
    "evidence": "论据",
    "overclaim": "结论过度",
    "structure": "结构",
    "clarity": "表述",
}
MAX_DOC_CHARS = 16000


class CritiqueState(TypedDict, total=False):
    doc_title: str
    language: Literal["zh", "en"]
    paper_type: str
    focus: str
    sections: list[dict]  # [{title, text}]，text 里的引用写成 [@bibkey]，段落之间空一行
    max_rounds: int
    round: int
    issues: list[dict]  # 当前仍需处理的问题
    resolved: list[dict]  # 已解决（审稿人确认）的问题
    summary: str
    verdict: str
    scores: dict
    history: list[dict]  # 每轮：{round, summary, verdict, scores, raised, resolved}
    responses: dict  # 作者对每条意见的回复 {id: {status: fixed|rebut|open, note}}
    action: str
    errors: list[str]


def _progress(message: str, **extra: Any) -> None:
    tracing.progress(message, **extra)
    try:
        get_stream_writer()({"message": message, **extra})
    except Exception:  # 不在 stream 上下文时忽略
        pass


# ---------------------------------------------------------------------------
# 定位：审稿人摘录的句子 → 文档里某一段中逐字存在的片段
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s)


def _paragraphs(sections: list[dict]) -> list[tuple[str, str]]:
    """[(章节标题, 去掉引用后的段落)]"""
    out = []
    for s in sections:
        for p in re.split(r"\n\s*\n", s.get("text") or ""):
            plain = strip_cites(p).strip()
            if plain:
                out.append((s.get("title", ""), plain))
    return out


def _bigrams(s: str) -> set[str]:
    s = _norm(s).lower()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def locate(quote: str, sections: list[dict], section: str = "") -> tuple[str, str]:
    """返回 (anchor, 所在章节)。anchor 是某个段落里逐字存在的句子；找不到时 anchor 为空（作为整节意见）。"""
    q = _norm(strip_cites(quote or "")).strip("“”\"'")
    paras = _paragraphs(sections)
    if not q:
        return "", section
    for title, p in paras:
        if q in _norm(p):
            # 用原段落里的片段（保留原有空格），前端按忽略空白的方式匹配
            return _slice_by_norm(p, q), title
    # 模型改写了原句：找重合度最高的句子
    qb = _bigrams(q)
    best, best_t, score = "", section, 0.0
    for title, p in paras:
        for sent in split_sentences(p):
            sb = _bigrams(sent)
            if not sb or not qb:
                continue
            sc = len(qb & sb) / max(len(qb), 1)
            if sc > score:
                best, best_t, score = sent.strip(), title, sc
    return (best, best_t) if score >= 0.55 else ("", section)


def _slice_by_norm(text: str, q: str) -> str:
    """在 text 里找到去空白后等于 q 的那一段，返回原文片段"""
    idx = []
    flat = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            idx.append(i)
            flat.append(ch)
    k = "".join(flat).find(q)
    if k < 0:
        return q
    return text[idx[k] : idx[k + len(q) - 1] + 1]


# ---------------------------------------------------------------------------
# 审稿
# ---------------------------------------------------------------------------
def _doc_block(sections: list[dict]) -> str:
    parts, total = [], 0
    for s in sections:
        t = f"## {s.get('title') or '（无标题）'}\n{(s.get('text') or '').strip()}\n"
        if total + len(t) > MAX_DOC_CHARS:
            parts.append(t[: max(0, MAX_DOC_CHARS - total)] + "\n（后文过长，已截断）")
            break
        parts.append(t)
        total += len(t)
    return "\n".join(parts)


def _refs_block(sections: list[dict]) -> str:
    keys: list[str] = []
    for s in sections:
        for m in CITE_TOKEN.finditer(s.get("text") or ""):
            for k in cite_keys(m.group(1)):
                if k not in keys:
                    keys.append(k)
    by_key = {p.bibkey: p for p in list_papers()}
    lines = [f"[@{k}] {by_key[k].title}（{by_key[k].year or 'n.d.'}）" for k in keys if k in by_key]
    return "\n".join(lines)


def _issue_line(i: dict) -> str:
    r = i.get("response") or {}
    resp = {"fixed": "作者：已修改", "rebut": f"作者：不改，理由是「{r.get('note') or '未说明'}」", "open": "作者：暂未处理"}.get(
        r.get("status", "open"), "作者：暂未处理"
    )
    if r.get("status") == "fixed" and r.get("note"):
        resp += f"（{r['note']}）"
    return f"- id={i['id']}｜{CATEGORIES.get(i['category'], i['category'])}｜《{i.get('section', '')}》“{i.get('anchor') or '整节'}”：{i['comment']}｜{resp}"


_OVERCLAIM = re.compile(r"显著|首次|证明了|完全|所有|一定能|彻底|毫无疑问|\balways\b|\bprove[sd]?\b|\bsignificantly\b|\bfirst\b", re.I)
_CLAIM = re.compile(r"表明|发现|提高|降低|减少|优于|导致|研究|方法|\bshow|\bimprove|\breduce|\boutperform", re.I)


def _fake_review(state: CritiqueState, prev: list[dict]) -> dict:
    """离线演示：用规则挑几处问题，第二轮起按作者的修改判断是否解决"""
    sections = state.get("sections") or []
    if prev:
        cur = {_norm(p) for _, p in _paragraphs(sections)}
        text = "".join(cur)
        ups = []
        for i in prev:
            r = (i.get("response") or {}).get("status")
            gone = bool(i.get("anchor")) and _norm(i["anchor"]) not in text
            if r == "rebut":
                ups.append({"id": i["id"], "status": "resolved", "comment": "作者的解释成立，接受。（离线演示）"})
            elif gone or r == "fixed":
                ups.append({"id": i["id"], "status": "resolved" if gone else "partial",
                            "comment": "已按意见修改。（离线演示）" if gone else "原句还在，改动不够明显。（离线演示）"})
            else:
                ups.append({"id": i["id"], "status": "unresolved", "comment": "这一条还没有处理。（离线演示）"})
        return {"summary": "作者回应了大部分意见。（离线演示）", "verdict": "minor", "scores": {"logic": 4, "evidence": 3, "structure": 4},
                "followups": ups, "issues": []}
    issues: list[dict] = []
    sents: list[tuple[str, str]] = []
    for s in sections:
        for para in re.split(r"\n\s*\n", s.get("text") or ""):
            for sent in split_sentences(para):
                plain = strip_cites(sent).strip()
                sents.append((s.get("title", ""), plain))
                if _OVERCLAIM.search(plain) and not any(x["category"] == "overclaim" for x in issues):
                    issues.append({"section": s.get("title", ""), "quote": plain, "category": "overclaim", "severity": "major",
                                   "comment": "结论的措辞超出了文中给出的证据。（离线演示）", "suggestion": "收窄论断，或补充实验 / 文献支撑。"})
                elif _CLAIM.search(plain) and not CITE_TOKEN.search(sent) and not any(x["category"] == "evidence" for x in issues):
                    issues.append({"section": s.get("title", ""), "quote": plain, "category": "evidence", "severity": "major",
                                   "comment": "这是一个需要论据支撑的判断，但没有给出引用或数据。（离线演示）", "suggestion": "补充文献或实验结果，或改为有限定的表述。"})
        body = strip_cites(s.get("text") or "").strip()
        if s.get("title") and len(body) < 60:
            issues.append({"section": s["title"], "quote": "", "category": "structure", "severity": "minor",
                           "comment": "这一节内容过少，和其他章节不成比例。（离线演示）", "suggestion": "补充本节要点，或并入相邻章节。"})
    used = {x["quote"] for x in issues}
    longest = max((x for x in sents if x[1] not in used), key=lambda x: len(x[1]), default=("", ""))
    if longest[1]:
        issues.insert(0, {"section": longest[0], "quote": longest[1], "category": "logic", "severity": "major",
                          "comment": "这句话一次跨了好几步推理，中间缺少过渡，读者很难跟上。（离线演示）", "suggestion": "拆成两句，补上中间的推理或依据。"})
    return {"summary": "选题有意义，结构基本完整；主要问题在于部分论断缺少论据、个别推理跳跃。（离线演示）", "verdict": "major",
            "scores": {"logic": 3, "evidence": 2, "structure": 3}, "issues": issues[:5]}


def _ask(state: CritiqueState, prev: list[dict], rnd: int) -> dict:
    zh = state.get("language", "zh") == "zh"
    sections = state.get("sections") or []
    max_new = 8 if rnd == 1 else 4
    cats = "logic=逻辑跳跃、因果不成立、前后矛盾；evidence=论据不足、缺少数据或文献支撑、以偏概全；overclaim=结论超出证据；structure=结构缺失、章节衔接差；clarity=概念不清、术语未定义"
    refs = _refs_block(sections)
    focus = f"\n作者希望重点看：{state['focus']}" if state.get("focus") else ""
    follow = ""
    if prev:
        follow = (
            "\n\n上一轮你提出、尚未确认解决的意见，以及作者的回复：\n" + "\n".join(_issue_line(i) for i in prev)
            + "\n请先逐条判断（结合作者修改后的全文和回复）："
            "resolved=已解决或作者的理由成立；partial=有改进但不够；unresolved=没有处理或回复不成立。"
            f"然后只补充新发现的**主要**问题（最多 {max_new} 条），不要重复已有意见，不要吹毛求疵。"
        )
    user = (
        f"论文题目：{state.get('doc_title') or '（未命名）'}\n论文类型：{state.get('paper_type') or '学术论文'}\n这是第 {rnd} 轮审稿。{focus}\n\n"
        f"全文（引用写成 [@bibkey]）：\n{_doc_block(sections)}\n\n"
        + (f"文中引用的文献：\n{refs}\n\n" if refs else "")
        + f"只挑**逻辑和论据**上的问题，不管错别字、格式和文风；引用与原文是否一致由其他工具检查，你不用核对。"
        f"问题类别：{cats}。严重程度：major=影响结论成立，minor=值得改进。{follow}\n\n"
        f"每条问题必须摘录原文中的一句话作为定位（quote 逐字复制，不含 [@...] 引用标记；针对整节的问题 quote 留空），"
        f"comment 说清楚问题是什么、为什么是问题，suggestion 给出具体可操作的改法。用{'中文' if zh else 'English'}写意见。\n"
        '只输出 JSON：{"summary": "总体评价，2-3 句", "verdict": "accept|minor|major", '
        '"scores": {"logic": 1-5, "evidence": 1-5, "structure": 1-5}, '
        + ('"followups": [{"id": "上一轮的 id", "status": "resolved|partial|unresolved", "comment": "一句话说明"}], ' if prev else "")
        + f'"issues": [{{"section": "章节标题", "quote": "原文句子", "category": "logic|evidence|overclaim|structure|clarity", "severity": "major|minor", "comment": "…", "suggestion": "…"}}]}}'
        f"（issues 最多 {max_new} 条，按严重程度排序）"
    )
    system = (
        f"你是{state.get('paper_type') or '学术论文'}的审稿人：严格但建设性，只提有理有据、作者能据此修改的意见。"
        if zh else "You are a rigorous but constructive peer reviewer."
    )
    return llm.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tier="agent", fake=lambda _m: json.dumps(_fake_review(state, prev), ensure_ascii=False), max_tokens=4000,
    )


def _candidates(anchor: str) -> list[dict]:
    """论据类问题：在文献库里找几篇可能用得上的文献（不调用模型判断，只给检索结果）"""
    if len(anchor) < 8:
        return []
    try:
        hits = suggest_citations(anchor, k=3, judge=False)
    except Exception:  # noqa: BLE001
        return []
    return [
        {"paper_id": h["paper"]["id"], "bibkey": h["paper"]["bibkey"], "title": h["paper"]["title"], "year": h["paper"].get("year"),
         "snippet": (h.get("snippet") or "")[:240], "page": h.get("page")}
        for h in hits
    ]


@tracing.traced("node", "审稿")
def review(state: CritiqueState) -> dict:
    rnd = int(state.get("round") or 0) + 1
    sections = state.get("sections") or []
    prev = [i for i in state.get("issues") or []]
    errors = list(state.get("errors") or [])
    _progress(f"第 {rnd} 轮：审稿人正在通读全文…" if rnd == 1 else f"第 {rnd} 轮：核对上一轮的 {len(prev)} 条意见，并检查修改后的全文…", phase="read")
    try:
        data = _ask(state, prev, rnd) or {}
    except Exception as e:  # noqa: BLE001
        errors.append(f"第 {rnd} 轮审稿失败：{type(e).__name__}: {e}")
        return {"round": rnd, "errors": errors, "issues": prev}

    # 上一轮的问题：按审稿人的判断更新
    ups = {str(u.get("id")): u for u in data.get("followups") or [] if isinstance(u, dict)}
    still, resolved = [], list(state.get("resolved") or [])
    for i in prev:
        u = ups.get(i["id"])
        st = (u or {}).get("status", "unresolved")
        i = {**i, "followup": {"round": rnd, "status": st, "comment": (u or {}).get("comment", "")}}
        # 作者修改后原句可能变了：重新定位，找不到就挂在整节上
        anchor, sec = locate(i.get("anchor", ""), sections, i.get("section", ""))
        i.update(anchor=anchor, section=sec, response=None)
        (resolved if st == "resolved" else still).append(i)

    # 新问题
    _progress(f"第 {rnd} 轮：整理意见并定位到原文…", phase="locate")
    fresh = []
    for k, it in enumerate(data.get("issues") or []):
        if not isinstance(it, dict) or not str(it.get("comment") or "").strip():
            continue
        cat = it.get("category") if it.get("category") in CATEGORIES else "logic"
        anchor, sec = locate(str(it.get("quote") or ""), sections, str(it.get("section") or ""))
        issue = {
            "id": f"r{rnd}-{k + 1}", "round": rnd, "section": sec, "anchor": anchor, "category": cat,
            "severity": "minor" if it.get("severity") == "minor" else "major",
            "comment": str(it["comment"]).strip(), "suggestion": str(it.get("suggestion") or "").strip(),
            "candidates": [],
        }
        if cat == "evidence" and anchor:
            issue["candidates"] = _candidates(anchor)
        fresh.append(issue)
    order = {"major": 0, "minor": 1}
    issues = sorted(still, key=lambda x: order[x["severity"]]) + sorted(fresh, key=lambda x: order[x["severity"]])

    scores = {k: int(v) for k, v in (data.get("scores") or {}).items() if isinstance(v, (int, float)) and k in ("logic", "evidence", "structure")}
    verdict = data.get("verdict") if data.get("verdict") in ("accept", "minor", "major") else ("minor" if issues else "accept")
    if not issues:
        verdict = "accept"
    history = list(state.get("history") or []) + [{
        "round": rnd, "summary": str(data.get("summary") or "").strip(), "verdict": verdict, "scores": scores,
        "raised": len(fresh), "resolved": len(resolved) - len(state.get("resolved") or []), "open": len(issues),
    }]
    major = sum(1 for i in issues if i["severity"] == "major")
    _progress(f"第 {rnd} 轮完成：{len(issues)} 条意见待处理（主要 {major} 条）" + (f"，上一轮解决了 {history[-1]['resolved']} 条" if rnd > 1 else ""), phase="done")
    return {"round": rnd, "issues": issues, "resolved": resolved, "summary": history[-1]["summary"], "verdict": verdict,
            "scores": scores, "history": history, "errors": errors, "responses": {}}


@tracing.traced("node", "等待修改")
def await_revision(state: CritiqueState) -> dict:
    answer = interrupt({
        "kind": "revise",
        "message": f"第 {state['round']} 轮审稿意见已挂在正文旁边，改完后可以再审一轮",
        "round": state["round"],
        "max_rounds": state.get("max_rounds", 3),
        "summary": state.get("summary", ""),
        "verdict": state.get("verdict", ""),
        "scores": state.get("scores", {}),
        "issues": state.get("issues", []),
        "resolved": state.get("resolved", []),
        "history": state.get("history", []),
    }) or {}
    action = "again" if answer.get("action") == "again" else "stop"
    responses = answer.get("responses") or {}
    issues = [{**i, "response": responses.get(i["id"])} for i in state.get("issues") or []]
    upd: dict[str, Any] = {"action": action, "issues": issues, "responses": responses}
    if answer.get("sections"):
        upd["sections"] = [{"title": str(s.get("title") or ""), "text": str(s.get("text") or "")} for s in answer["sections"]]
    return upd


@tracing.traced("node", "完成")
def finish(state: CritiqueState) -> dict:
    n = len(state.get("issues") or [])
    _progress(f"审稿结束：共 {state.get('round', 0)} 轮" + (f"，还有 {n} 条意见未解决" if n else "，意见都已解决"))
    return {}


def _route_review(state: CritiqueState) -> str:
    if not state.get("issues") or int(state.get("round") or 0) >= int(state.get("max_rounds") or 3):
        return "finish"
    return "await_revision"


def _route_revision(state: CritiqueState) -> str:
    return "review" if state.get("action") == "again" else "finish"


def build_critic(checkpointer: Any = None):
    g = StateGraph(CritiqueState)
    g.add_node("review", review)
    g.add_node("await_revision", await_revision)
    g.add_node("finish", finish)
    g.add_edge(START, "review")
    g.add_conditional_edges("review", _route_review, ["await_revision", "finish"])
    g.add_conditional_edges("await_revision", _route_revision, ["review", "finish"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


_critic = None


def get_critic():
    global _critic
    if _critic is None:
        conn = sqlite3.connect(str(get_settings().data_dir / "checkpoints.sqlite"), check_same_thread=False)
        _critic = build_critic(SqliteSaver(conn))
    return _critic
