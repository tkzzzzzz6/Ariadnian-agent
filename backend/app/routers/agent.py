from __future__ import annotations

import json
import time
import uuid
from contextlib import closing
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..agent.graph import get_graph, known_keys, valid_outline
from ..agent.note_context import describe_context
from ..agent.writer import clean_citations
from ..agent.critic import get_critic
from ..agent.writer import get_writer
from ..agent.runtime import AgentRun, active, release, reserve
from .. import runstats, tracing
from ..db import get_conn

router = APIRouter(prefix="/api/agent", tags=["agent"])

NODE_LABELS = {
    "plan_queries": "拆解检索式",
    "search_papers": "检索文献",
    "select_papers": "勾选文献",
    "ingest": "加入文献库",
    "prepare_notes": "准备相关用户笔记",
    "synthesize": "归纳研究现状",
    "make_outline": "生成大纲",
    "review_outline": "确认大纲",
    "finish": "完成",
    # 分节写作
    "prepare": "准备",
    "write_section": "写作本节",
    "review_section": "确认本节",
    "advance": "下一节",
    "write_one": "子 Agent 写作",
    "collect": "汇总",
    "polish": "总编统稿",
    # 审稿
    "review": "审稿",
    "await_revision": "等你修改",
}


class StartIn(BaseModel):
    topic: str = Field(..., min_length=2)
    doc_id: str | None = None
    language: str = "zh"
    paper_type: str = "学术论文"
    sources: list[str] | None = None  # 不传则使用设置里的默认来源
    max_papers: int = 8
    fetch_pdf: bool = True
    use_notes: bool = True


class SectionIn(BaseModel):
    title: str = Field(..., min_length=1)
    goal: str = ""
    bibkeys: list[str] = Field(default_factory=list)


class WriteIn(BaseModel):
    doc_id: str | None = None
    topic: str = ""
    doc_title: str = ""
    language: str = "zh"
    paper_type: str = "学术论文"
    outline: list[SectionIn] = Field(..., min_length=1, max_length=20)
    mode: str = "confirm"  # confirm=每节写完停下来确认；auto=连续写完；parallel=子 Agent 并行写
    length: str = "medium"  # short / medium / long
    selected: list[int] | None = None  # 要写的章节（大纲下标），不传 = 全部
    polish: bool = False  # 写完后由总编统稿


class DocSectionIn(BaseModel):
    title: str = ""
    text: str = ""


class CritiqueIn(BaseModel):
    doc_id: str | None = None
    doc_title: str = ""
    language: str = "zh"
    paper_type: str = "学术论文"
    focus: str = ""
    sections: list[DocSectionIn] = Field(..., min_length=1, max_length=60)
    max_rounds: int = Field(3, ge=1, le=5)


class ResumeIn(BaseModel):
    value: dict[str, Any]


def _set_status(tid: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE agent_threads SET status=?, updated_at=datetime('now') WHERE thread_id=?", (status, tid))


def _kind(tid: str) -> str:
    with get_conn() as conn:
        row = conn.execute("SELECT kind FROM agent_threads WHERE thread_id=?", (tid,)).fetchone()
    return row["kind"] if row else "review"


def _graph_for(tid: str):
    k = _kind(tid)
    return get_writer() if k == "write" else get_critic() if k == "critique" else get_graph()


def _result(values: dict) -> dict:
    if "max_rounds" in values:  # 审稿
        keys = ("round", "max_rounds", "summary", "verdict", "scores", "issues", "resolved", "history", "errors")
        return {k: values.get(k) for k in keys} | {"kind": "critique"}
    if "drafts" in values:  # 分节写作
        keys = ("topic", "outline", "drafts", "mode", "order", "polish", "polish_done", "edits", "polish_summary", "errors")
        return {k: values.get(k) for k in keys} | {"kind": "write"}
    known = known_keys(values)
    return {k: values.get(k) for k in ("topic", "paper_ids", "errors")} | {
        "kind": "review", "outline": valid_outline(values.get("outline") or [], known),
        "synthesis": clean_citations(values.get("synthesis") or "", known),
        "notes": [n for n in values.get("notes") or [] if isinstance(n, dict) and n.get("bibkey") in known],
        "note_context": describe_context(values.get("note_context")),
    }


def _run(run: AgentRun, graph: Any, payload: Any) -> None:
    tid = run.tid
    config = {"configurable": {"thread_id": tid}}
    meter = runstats.RunMeter(tid, runstats.load(tid))
    trace = tracing.start(tid)
    status, trace_error = "interrupted", None
    last = [time.perf_counter(), meter.version]

    def tick(force: bool = False) -> dict | None:
        now = time.perf_counter()
        if force or (meter.version != last[1] and now - last[0] >= 0.5):
            last[0], last[1] = now, meter.version
            return {"type": "stats", "stats": meter.snapshot()}
        return None

    try:
        _set_status(tid, "running")
        run.emit({"type": "status", "status": "running", "thread_id": tid, "run_id": trace.id if trace else None})
        run.emit(tick(force=True))
        while True:
            # Fully drain one superstep before considering cancellation. This
            # persists completed work, including parallel nodes, and closes the
            # iterator on its owner thread rather than during SSE finalization.
            with closing(graph.stream(payload, config, stream_mode=["updates", "custom"],
                                      interrupt_after="*", durability="sync")) as stream:
                for mode, chunk in tracing.bind(trace, runstats.bind(meter, stream)):
                    if mode == "custom":
                        run.emit({"type": "trace", **chunk["trace"]} if "trace" in chunk else {"type": "progress", **chunk})
                    else:
                        for node, update in chunk.items():
                            if node == "__interrupt__":
                                if update:  # static checkpoint boundaries have no user input
                                    run.emit({"type": "interrupt", "value": update[0].value})
                            else:
                                meter.step()
                                run.emit({"type": "node", "node": node, "label": NODE_LABELS.get(node, node)})
                    if (event := tick()) is not None:
                        run.emit(event)
            snap = graph.get_state(config)
            if not snap.next:
                status = "done"
            elif any(task.interrupts for task in snap.tasks):
                status = "waiting"
            elif run.stop_requested.is_set():
                status = "stopped"
            else:
                payload = None
                continue
            meter.stop_clock()
            run.emit(tick(force=True))
            _set_status(tid, status)
            run.emit({"type": "done", "result": _result(snap.values)} if status == "done"
                     else {"type": "status", "status": status})
            return
    except Exception as e:  # noqa: BLE001
        status, trace_error = "error", f"{type(e).__name__}: {e}"
        meter.stop_clock()
        run.emit(tick(force=True))
        _set_status(tid, "error")
        run.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        tracing.finish(trace, status, trace_error)
        if trace and trace.incomplete:
            run.emit({"type": "trace_warning", "message": "部分运行记录未能保存，任务执行不受影响"})
        meter.stop_clock()
        runstats.save(meter)


def _response(tid: str, payload: Any = None, *, resuming: bool = False):
    run = reserve(tid)  # reserve before inspecting the checkpoint (no resume race)
    try:
        graph = _graph_for(tid)
        if resuming:
            snap = graph.get_state({"configurable": {"thread_id": tid}})
            if not snap.next:
                raise HTTPException(409, "该任务没有在等待输入")
            waiting = any(task.interrupts for task in snap.tasks)
            if waiting and not payload:
                raise HTTPException(409, "任务在等你确认，请先在卡片上做出选择")
            if waiting and snap.values.get("note_context") and isinstance(payload, dict):
                context = describe_context(snap.values["note_context"])
                if context["changed"] and payload.get("action") != "refresh_notes" and payload.get("confirm_stale_notes") is not True:
                    raise HTTPException(409, "本轮笔记来源已修改或删除。请刷新笔记并重新生成，或明确确认沿用旧快照。")
            payload = Command(resume=payload) if waiting else None
        response = run.response()
        run.start(lambda worker: _run(worker, graph, payload))
        return response
    except BaseException:
        release(run)
        raise


@router.post("/start")
def start(body: StartIn):
    tid = uuid.uuid4().hex[:16]
    with get_conn() as conn:
        conn.execute("INSERT INTO agent_threads (thread_id, doc_id, topic) VALUES (?,?,?)", (tid, body.doc_id, body.topic))
    state = body.model_dump(exclude={"doc_id"})
    return _response(tid, state)


@router.post("/write")
def write(body: WriteIn):
    """分节写作：写选中的章节（逐节确认 / 连续写完 / 子 Agent 并行），可选总编统稿；草稿流式输出"""
    picked = sorted({i for i in body.selected if 0 <= i < len(body.outline)}) if body.selected is not None else list(range(len(body.outline)))
    if not picked:
        raise HTTPException(422, "至少选择一节")
    tid = uuid.uuid4().hex[:16]
    titles = "、".join(body.outline[i].title for i in picked[:3]) + ("…" if len(picked) > 3 else "")
    with get_conn() as conn:
        conn.execute("INSERT INTO agent_threads (thread_id, doc_id, topic, kind) VALUES (?,?,?,?)",
                     (tid, body.doc_id, f"分节写作：{titles}", "write"))
    state = body.model_dump(exclude={"doc_id"}) | {"selected": picked}
    state["mode"] = body.mode if body.mode in ("auto", "parallel") else "confirm"
    state["length"] = body.length if body.length in ("short", "medium", "long") else "medium"
    return _response(tid, state)


@router.post("/critique")
def critique(body: CritiqueIn):
    """审稿：审全文 → 作者修改并回复 → 再审，最多 max_rounds 轮"""
    sections = [s.model_dump() for s in body.sections if s.text.strip()]
    if not sections:
        raise HTTPException(422, "文档还没有正文，写一些内容后再审稿")
    tid = uuid.uuid4().hex[:16]
    with get_conn() as conn:
        conn.execute("INSERT INTO agent_threads (thread_id, doc_id, topic, kind) VALUES (?,?,?,?)",
                     (tid, body.doc_id, f"审稿：{body.doc_title or '未命名论文'}", "critique"))
    state = body.model_dump(exclude={"doc_id"}) | {"sections": sections, "round": 0}
    return _response(tid, state)


@router.post("/{tid}/resume")
def resume(tid: str, body: ResumeIn):
    return _response(tid, body.value, resuming=True)


@router.post("/{tid}/stop")
def stop(tid: str):
    """请求在当前图步骤保存后停止；执行释放之前仍禁止恢复。"""
    if run := active(tid):
        run.stop_requested.set()
        return {"ok": True, "stopping": True}
    with get_conn() as conn:
        conn.execute("UPDATE agent_threads SET status='stopped', updated_at=datetime('now') WHERE thread_id=? AND status='running'", (tid,))
    return {"ok": True, "stopping": False}


@router.get("")
def list_threads(doc_id: str | None = None):
    with get_conn() as conn:
        if doc_id:
            rows = conn.execute(
                "SELECT * FROM agent_threads WHERE doc_id=? ORDER BY created_at DESC LIMIT 20", (doc_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM agent_threads ORDER BY created_at DESC LIMIT 20").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        st = json.loads(d.pop("stats") or "null") if d.get("stats") else None
        d["tokens"] = (st or {}).get("total_tokens")  # 历史列表只需要总量
        if d["status"] == "running":
            a = active(d["thread_id"])
            d["status"] = "stopping" if a and a.stop_requested.is_set() else "running" if a else "interrupted"
        out.append(d)
    return out


@router.get("/{tid}/stats")
def thread_stats(tid: str):
    """任务的运行统计（历史任务回看时用）；没有记录时返回 null"""
    return runstats.load(tid)


@router.get("/{tid}/traces")
def trace_runs(tid: str):
    try:
        return {"runs": tracing.runs(tid), "policy": tracing.policy()}
    except Exception as e:
        raise HTTPException(503, "运行记录暂时不可读取，任务执行不受影响") from e


@router.get("/{tid}/traces/{rid}")
def trace_run(tid: str, rid: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    try:
        data = tracing.read(tid, rid, offset, limit)
    except Exception as e:
        raise HTTPException(503, "运行记录暂时不可读取") from e
    if data is None:
        raise HTTPException(404, "运行记录不存在或已清理")
    return data


@router.get("/{tid}/traces/{rid}/spans/{sid}")
def trace_span(tid: str, rid: str, sid: str):
    try:
        data = tracing.detail(tid, rid, sid)
    except Exception as e:
        raise HTTPException(503, "步骤记录暂时不可读取") from e
    if data is None:
        raise HTTPException(404, "步骤不存在或已清理")
    return data


@router.get("/{tid}/traces/{rid}/overview")
def trace_overview(tid: str, rid: str):
    try:
        data = tracing.overview(tid, rid)
    except Exception as e:
        raise HTTPException(503, "运行概览暂时不可读取") from e
    if data is None:
        raise HTTPException(404, "运行记录不存在或已清理")
    return data


@router.get("/{tid}/traces/{rid}/nodes/{nid}")
def trace_node_calls(tid: str, rid: str, nid: str, offset: int = Query(0, ge=0), focus: str | None = Query(None, max_length=64)):
    try:
        data = tracing.node_calls(tid, rid, nid, offset, focus=focus)
    except Exception as e:
        raise HTTPException(503, "调用记录暂时不可读取") from e
    if data is None:
        raise HTTPException(404, "运行记录不存在或已清理")
    return data


@router.get("/{tid}/traces/{rid}/diagnostic")
def trace_diagnostic(tid: str, rid: str):
    try:
        data = tracing.diagnostic(tid, rid)
    except Exception as e:
        raise HTTPException(503, "诊断记录暂时不可读取") from e
    if data is None:
        raise HTTPException(404, "运行记录不存在或已清理")
    return data


@router.delete("/{tid}/traces")
def trace_clear(tid: str):
    from ..security import settings_locked

    if settings_locked():
        raise HTTPException(403, "管理员已锁定设置，无法清理记录")
    if active(tid):
        raise HTTPException(409, "请等待任务结束或停止后再清理")
    tracing.clear(tid)
    return {"ok": True}


@router.get("/{tid}")
def get_thread(tid: str):
    graph = _graph_for(tid)
    snap = graph.get_state({"configurable": {"thread_id": tid}})
    if not snap.values:
        raise HTTPException(404, "任务不存在")
    pending = None
    for task in snap.tasks:
        if task.interrupts:
            pending = task.interrupts[0].value
            break
    if isinstance(pending, dict) and pending.get("kind") == "review_outline":
        current = _result(snap.values)
        pending = {**pending, **{k: current[k] for k in ("outline", "synthesis", "notes", "note_context")}}
    with get_conn() as conn:
        row = conn.execute("SELECT status, kind FROM agent_threads WHERE thread_id=?", (tid,)).fetchone()
    return {
        "stats": runstats.load(tid),
        "thread_id": tid,
        "kind": row["kind"] if row else "review",
        "status": row["status"] if row else ("waiting" if pending else "done"),
        "stopping": bool((run := active(tid)) and run.stop_requested.is_set()),
        "interrupt": pending,
        "result": _result(snap.values),
    }
