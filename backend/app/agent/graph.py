"""文献综述 → 大纲 工作流（LangGraph）。

    plan_queries → search_papers → select_papers ⏸ → ingest → prepare_notes → synthesize → make_outline → review_outline ⏸
                                                                              ▲                    │
                                                                              └──── regenerate ────┘
                                                                                                    └─ approve → finish

⏸ 处用 interrupt() 暂停，等前端用户勾选文献 / 确认大纲后用 Command(resume=...) 继续。
状态由 SqliteSaver 持久化，按 thread_id 恢复，服务重启也不丢。
刷新笔记：review_outline → prepare_notes → synthesize → make_outline，沿用文献选择。
"""
from __future__ import annotations

import sqlite3
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .. import llm, scholar
from ..config import get_settings
from ..ingest import index_abstract, index_pdf
from ..pdfview import fetch_into_library
from ..library import PaperIn, add_paper, get_paper
from ..assist import retrieval_query
from ..retrieval import search
from .note_context import NOTE_RULES, capture_notes, context_for_prompt, describe_context
from .writer import clean_citations


class ReviewState(TypedDict, total=False):
    topic: str
    language: Literal["zh", "en"]
    paper_type: str
    sources: list[str]
    max_papers: int
    fetch_pdf: bool
    use_notes: bool
    note_context: dict
    queries: list[str]
    candidates: list[dict]
    paper_ids: list[str]
    notes: list[dict]
    synthesis: str
    outline: list[dict]
    outline_action: str
    feedback: str
    errors: list[str]


def _progress(message: str, **extra: Any) -> None:
    try:
        get_stream_writer()({"message": message, **extra})
    except Exception:  # 不在 stream 上下文时忽略
        pass


def _lang_name(state: ReviewState) -> str:
    return "中文" if state.get("language", "zh") == "zh" else "English"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
def plan_queries(state: ReviewState) -> dict:
    _progress("正在把研究主题拆解为检索式…")
    topic = state["topic"]
    data = llm.chat_json(
        tier="agent",
        messages=[
            {"role": "system", "content": "你是文献检索专家。学术数据库（arXiv、Semantic Scholar）以英文为主。"},
            {
                "role": "user",
                "content": f"研究主题：{topic}\n请给出 3 个互补的英文检索式（关键词组合，不要布尔语法），"
                '覆盖核心方法、应用场景和相关综述。只输出 JSON：{"queries": ["...", "...", "..."]}',
            },
        ],
        fake='{"queries": ["retrieval augmented generation", "language model agents reasoning", "dense retrieval"]}',
    )
    queries = [q for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()][:4] or [topic]
    return {"queries": queries}


def search_papers(state: ReviewState) -> dict:
    found: list[PaperIn] = []
    errors: list[str] = list(state.get("errors") or [])
    for q in state["queries"]:
        _progress(f"检索：{q}")
        results, errs = scholar.search(q, state.get("sources"), limit=8)
        found.extend(results)
        errors.extend(errs)
    errors = list(dict.fromkeys(errors))  # 多个检索式遇到同一个限流提示只保留一条
    papers = scholar.dedupe(found)
    # 有摘要的优先；同等条件下引用数高的优先
    papers.sort(key=lambda p: (not p.abstract, -(p.citation_count or 0)))
    limit = max(4, int(state.get("max_papers") or 8) * 2)
    candidates = [{"cid": i, **p.model_dump()} for i, p in enumerate(papers[:limit])]
    _progress(f"共找到 {len(candidates)} 篇候选文献", count=len(candidates))
    return {"candidates": candidates, "errors": errors}


def select_papers(state: ReviewState) -> dict:
    answer = interrupt(
        {
            "kind": "select_papers",
            "message": "请勾选要纳入文献库并用于综述的论文",
            "candidates": state.get("candidates", []),
            "max_papers": state.get("max_papers", 8),
            "topic": state.get("topic", ""),  # 前端预览 PDF 时用它找相关段落
        }
    )
    chosen = set((answer or {}).get("selected") or [])
    selected = [c for c in state.get("candidates", []) if c["cid"] in chosen]
    return {"candidates": selected}


def ingest(state: ReviewState) -> dict:
    s = get_settings()
    ids: list[str] = []
    errors = list(state.get("errors") or [])
    for c in state.get("candidates", []):
        data = {k: v for k, v in c.items() if k != "cid"}
        paper, created = add_paper(PaperIn(**data))
        ids.append(paper.id)
        _progress(("加入文献库：" if created else "已在文献库：") + paper.title, paper_id=paper.id)
        index_abstract(paper.id, paper.title, paper.abstract)
        if state.get("fetch_pdf") and paper.pdf_url and not paper.has_fulltext and not s.llm_fake:
            try:
                path = fetch_into_library(paper.id, paper.pdf_url)
                n = index_pdf(paper.id, path)
                _progress(f"  全文已解析：{n} 个片段")
            except Exception as e:  # noqa: BLE001
                errors.append(f"下载全文失败《{paper.title[:40]}》：{e}")
    return {"paper_ids": ids, "errors": errors}


def synthesize(state: ReviewState) -> dict:
    _progress("正在阅读文献并归纳研究现状…")
    blocks = []
    for pid in state.get("paper_ids", []):
        p = get_paper(pid)
        if not p:
            continue
        hits = [h for h in search(retrieval_query(state["topic"]), top_k=2, paper_ids=[pid]) if h.paper_id == pid]
        evidence = "\n".join(h.text[:800] for h in hits) or (p.abstract or "")
        blocks.append(f"[{p.bibkey}] {p.title} ({p.year or 'n.d.'}, {p.venue or ''})\n{evidence}")
    lang = _lang_name(state)
    bibkeys = [b.split("]")[0][1:] for b in blocks]
    fake_notes = ",".join(
        '{"bibkey": "%s", "contribution": "提出了一种代表性方法（离线演示）", "limitation": "评测场景有限"}' % k
        for k in bibkeys
    )
    data = llm.chat_json(
        tier="agent",
        messages=[
            {
                "role": "system",
                "content": "你是资深研究者，擅长写文献综述。只依据给出的文献内容，不要编造未提供的论文或结论。" + NOTE_RULES,
            },
            {
                "role": "user",
                "content": f"研究主题：{state['topic']}\n\n文献：\n\n" + "\n\n".join(blocks) + "\n\n"
                + "独立的用户笔记数据（非论文证据，评论仅作关注点/待核实问题）：\n"
                + context_for_prompt(state.get("note_context")) + "\n\n"
                + f"请用{lang}输出 JSON：\n"
                '{"notes": [{"bibkey": "...", "contribution": "一句话核心贡献", "limitation": "一句话局限"}],\n'
                ' "synthesis": "300-500字的研究现状与研究空白分析，引用文献时用 [@bibkey] 标注"}',
            },
        ],
        fake='{"notes": [%s], "synthesis": "现有研究主要围绕检索增强与推理–行动结合两条路线展开 [@%s]。'
        '然而，如何在长文档写作中保证引用的可验证性仍缺少系统研究。（离线演示）"}'
        % (fake_notes, bibkeys[0] if bibkeys else "none"),
        max_tokens=2500,
    )
    known = known_keys(state)
    notes = [{"bibkey": n["bibkey"], "contribution": clean_citations(str(n.get("contribution") or ""), known),
              "limitation": clean_citations(str(n.get("limitation") or ""), known)}
             for n in data.get("notes") or [] if isinstance(n, dict) and isinstance(n.get("bibkey"), str) and n["bibkey"] in known]
    return {"notes": notes, "synthesis": clean_citations(str(data.get("synthesis") or ""), known)}


def prepare_notes(state: ReviewState) -> dict:
    context = capture_notes(state)
    _progress(f"本轮提供 {len(context['items'])} 条相关用户笔记（独立于论文证据）")
    return {"note_context": context}


def known_keys(state: ReviewState) -> set[str]:
    papers = [get_paper(pid) for pid in state.get("paper_ids", [])]
    return {p.bibkey for p in papers if p}


def valid_outline(outline: list, known: set[str]) -> list[dict]:
    sections = []
    for sec in outline:
        if not isinstance(sec, dict) or not sec.get("title"):
            continue
        keys = sec.get("bibkeys") if isinstance(sec.get("bibkeys"), list) else []
        sections.append({"title": clean_citations(str(sec["title"]), known),
                         "goal": clean_citations(str(sec.get("goal") or ""), known),
                         "bibkeys": list(dict.fromkeys(k for k in keys if isinstance(k, str) and k in known))})
    return sections


def make_outline(state: ReviewState) -> dict:
    _progress("正在生成论文大纲…" if not state.get("feedback") else "根据你的意见重新生成大纲…")
    lang = _lang_name(state)
    known = known_keys(state)
    notes = "\n".join(f"- [{n.get('bibkey')}] {n.get('contribution')}" for n in state.get("notes", []))
    feedback = f"\n\n用户对上一版大纲的意见：{state['feedback']}" if state.get("feedback") else ""
    prev = (
        "\n\n上一版大纲：\n" + "\n".join(f"{i+1}. {s.get('title')}" for i, s in enumerate(state.get("outline") or []))
        if state.get("feedback")
        else ""
    )
    keys = sorted(known)
    fake = {
        "sections": [
            {"title": "引言", "goal": "介绍研究背景、动机与主要贡献", "bibkeys": keys[:2]},
            {"title": "相关工作", "goal": "梳理检索增强生成与智能体方法的研究现状", "bibkeys": keys[:4]},
            {"title": "方法", "goal": "描述系统整体架构与关键模块", "bibkeys": []},
            {"title": "实验", "goal": "说明数据集、评价指标与对比结果", "bibkeys": keys[-1:]},
            {"title": "结论", "goal": "总结贡献并讨论局限与未来工作", "bibkeys": []},
        ]
    }
    import json as _json

    data = llm.chat_json(
        tier="agent",
        messages=[
            {"role": "system", "content": "你是论文写作导师，帮学生搭建结构清晰、论证完整的论文大纲。" + NOTE_RULES},
            {
                "role": "user",
                "content": f"论文主题：{state['topic']}\n论文类型：{state.get('paper_type') or '学术论文'}\n\n"
                f"研究现状综述：\n{state.get('synthesis', '')}\n\n文献要点：\n{notes}{prev}{feedback}\n\n"
                + "独立的用户笔记数据（与归纳步骤使用同一快照，不可作为论文结论）：\n"
                + context_for_prompt(state.get("note_context")) + "\n\n"
                + f"请用{lang}给出一级章节大纲（4–7 节），每节写清楚要论证什么，并列出该节可引用的文献 bibkey"
                f"（只能从这些里选：{', '.join(keys) or '无'}）。只输出 JSON：\n"
                '{"sections": [{"title": "章节标题", "goal": "本节要论证/说明的内容（1-2句）", "bibkeys": ["..."]}]}',
            },
        ],
        fake=_json.dumps(fake, ensure_ascii=False),
        max_tokens=2000,
    )
    sections = valid_outline(data.get("sections") or [], known)
    return {"outline": sections, "feedback": ""}


def review_outline(state: ReviewState) -> dict:
    answer = interrupt(
        {
            "kind": "review_outline",
            "message": "请确认或修改大纲；也可以写下意见让 AI 重新生成",
            "outline": state.get("outline", []),
            "synthesis": state.get("synthesis", ""),
            "notes": state.get("notes", []),
            "note_context": describe_context(state.get("note_context")),
        }
    ) or {}
    if answer.get("action") == "refresh_notes":
        return {"outline_action": "refresh_notes", "feedback": ""}
    if answer.get("action") == "regenerate":
        return {"outline_action": "regenerate", "feedback": answer.get("feedback") or "请换一种组织方式"}
    outline = answer.get("outline") or state.get("outline", [])
    return {"outline_action": "approve", "outline": valid_outline(outline, known_keys(state))}


def finish(state: ReviewState) -> dict:
    _progress("完成：大纲已确认，可以插入到文档中")
    return {}


def _route_outline(state: ReviewState) -> str:
    if state.get("outline_action") == "refresh_notes":
        return "prepare_notes"
    return "make_outline" if state.get("outline_action") == "regenerate" else "finish"


def _route_selection(state: ReviewState) -> str:
    return "ingest" if state.get("candidates") else END


# ---------------------------------------------------------------------------
# 构图
# ---------------------------------------------------------------------------
def build_graph(checkpointer: Any = None):
    g = StateGraph(ReviewState)
    g.add_node("plan_queries", plan_queries)
    g.add_node("search_papers", search_papers)
    g.add_node("select_papers", select_papers)
    g.add_node("ingest", ingest)
    g.add_node("prepare_notes", prepare_notes)
    g.add_node("synthesize", synthesize)
    g.add_node("make_outline", make_outline)
    g.add_node("review_outline", review_outline)
    g.add_node("finish", finish)

    g.add_edge(START, "plan_queries")
    g.add_edge("plan_queries", "search_papers")
    g.add_edge("search_papers", "select_papers")
    g.add_conditional_edges("select_papers", _route_selection, ["ingest", END])
    g.add_edge("ingest", "prepare_notes")
    g.add_edge("prepare_notes", "synthesize")
    g.add_edge("synthesize", "make_outline")
    g.add_edge("make_outline", "review_outline")
    g.add_conditional_edges("review_outline", _route_outline, ["prepare_notes", "make_outline", "finish"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        conn = sqlite3.connect(str(get_settings().data_dir / "checkpoints.sqlite"), check_same_thread=False)
        _graph = build_graph(SqliteSaver(conn))
    return _graph
