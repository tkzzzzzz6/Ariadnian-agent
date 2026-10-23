"""Local, best-effort Agent traces. Recording never owns graph execution.

A bounded queue and short SQLite transactions isolate trace IO from checkpoints.
Only explicit content capture stores inputs/outputs/reasoning; no token events
are written. ContextVars link parallel graph nodes, tools and model attempts.
"""
from __future__ import annotations

import inspect
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Iterator

from .config import SECRET_KEYS, get_settings

_run: ContextVar[Recorder | None] = ContextVar("trace_run", default=None)
_span: ContextVar[Span | None] = ContextVar("trace_span", default=None)
_live: dict[str, Recorder] = {}
_lock = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, started REAL NOT NULL,
 ended REAL, status TEXT NOT NULL, capture INTEGER NOT NULL, incomplete INTEGER NOT NULL DEFAULT 0,
 error TEXT, version TEXT NOT NULL DEFAULT '0.1.0'
);
CREATE INDEX IF NOT EXISTS runs_thread ON runs(thread_id, started);
CREATE TABLE IF NOT EXISTS spans (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
 parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL, label TEXT NOT NULL,
 status TEXT NOT NULL, started REAL NOT NULL, ended REAL, duration_ms INTEGER,
 data TEXT NOT NULL, detail TEXT
);
CREATE INDEX IF NOT EXISTS spans_run ON spans(run_id, started, id);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
 parent_id TEXT, at REAL NOT NULL, message TEXT NOT NULL, phase TEXT
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
"""


def path():
    return get_settings().data_dir / "traces.sqlite"


def _secrets() -> set[str]:
    settings = get_settings()
    from .config import _env

    secrets = [t.api_key for t in settings.tiers.values()]
    secrets += [_env(k) for k in SECRET_KEYS]
    secrets += [os.getenv("OPENAI_API_KEY"), os.getenv("ARIA_PASSWORD")]
    return {s for s in secrets if s}


def scrub(value: Any, limit: int = 2000, secrets: set[str] | None = None) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    # Keep the run's original keys redacted even after a settings change.
    r = _run.get()
    known = _secrets() | (secrets or set()) | (r.secrets if r else set())
    for secret in sorted(known, key=len, reverse=True):
        text = text.replace(secret, "[已脱敏]")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"',;]+", r"\1[已脱敏]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password|secret|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;}]+",
                  r"\1[已脱敏]", text)
    text = re.sub(r"https?://[^\s/@]+:[^\s/@]+@", "https://[已脱敏]@", text)
    return text if len(text) <= limit else text[:limit] + "…[已截断]"


def policy() -> dict:
    from .config import _env

    def days(key, default):
        try:
            return max(1, min(365, int(_env(key, str(default)) or default)))
        except ValueError:
            return default
    return {"capture": (_env("TRACE_CAPTURE", "0") or "0").lower() in ("1", "true", "yes"),
            "retention_days": days("TRACE_RETENTION_DAYS", 30),
            "detail_days": days("TRACE_DETAIL_DAYS", 7)}


@contextmanager
def connect():
    conn = sqlite3.connect(path(), timeout=0.15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _initialize(conn):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)


def _prune(conn):
    p = policy()
    now = time.time()
    conn.execute("UPDATE spans SET detail=NULL WHERE run_id IN (SELECT id FROM runs WHERE ended IS NOT NULL AND started < ?)",
                 (now - p["detail_days"] * 86400,))
    conn.execute("DELETE FROM runs WHERE ended IS NOT NULL AND started < ?", (now - p["retention_days"] * 86400,))
    with _lock:
        live_ids = set(_live)
    stale = [r["id"] for r in conn.execute("SELECT id FROM runs WHERE ended IS NULL AND started < ?",
                                         (now - p["retention_days"] * 86400,)) if r["id"] not in live_ids]
    conn.executemany("DELETE FROM runs WHERE id=?", [(rid,) for rid in stale])


class Recorder:
    def __init__(self, thread_id: str):
        self.id = uuid.uuid4().hex
        self.thread_id = thread_id
        self.capture = policy()["capture"]
        self.secrets = _secrets()
        self.started = time.time()
        self.last_activity = self.started
        self.incomplete = False
        self.status = "running"
        self.events: queue.Queue = queue.Queue(maxsize=2048)
        self.count = 0
        self.count_lock = threading.Lock()
        self.closed = threading.Event()
        self.ready = threading.Event()
        self.writer = threading.Thread(target=self._write, name=f"trace-{self.id[:8]}", daemon=True)
        self.writer.start()

    def put(self, kind: str, values: tuple):
        with self.count_lock:
            self.count += 1
            if self.count > 20000:
                self.incomplete = True
                return
        try:
            self.events.put_nowait((kind, values))
        except queue.Full:
            self.incomplete = True

    def _write(self):
        try:
            with connect() as conn:
                _initialize(conn)
                conn.execute("INSERT INTO runs(id,thread_id,started,status,capture) VALUES (?,?,?,?,?)",
                             (self.id, self.thread_id, self.started, "running", int(self.capture)))
                _prune(conn)
                conn.commit()
                self.ready.set()
                while not self.closed.is_set() or not self.events.empty():
                    batch = []
                    try:
                        batch.append(self.events.get(timeout=0.2))
                    except queue.Empty:
                        continue
                    while len(batch) < 50:
                        try:
                            batch.append(self.events.get_nowait())
                        except queue.Empty:
                            break
                    try:
                        for kind, v in batch:
                            if kind == "span":
                                conn.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                                             "status=excluded.status, ended=excluded.ended, duration_ms=excluded.duration_ms, "
                                             "data=excluded.data, detail=excluded.detail", v)
                            elif kind == "event":
                                conn.execute("INSERT INTO events(run_id,parent_id,at,message,phase) VALUES (?,?,?,?,?)", v)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        self.incomplete = True
                conn.execute("UPDATE runs SET ended=?,status=?,incomplete=?,error=? WHERE id=?",
                             (getattr(self, "ended", time.time()), self.status, int(self.incomplete), getattr(self, "error", None), self.id))
                conn.commit()
        except Exception:
            self.incomplete = True
        finally:
            self.ready.set()

    def finish(self, status: str, error: str | None = None):
        self.status = status
        self.ended = time.time()
        self.error = scrub(error, 600, self.secrets) if error else None
        self.closed.set()
        self.put("flush", ())
        # Bounded flush; a broken disk must not hold the Agent worker indefinitely.
        self.writer.join(timeout=0.5)
        if self.writer.is_alive():
            self.incomplete = True


def start(thread_id: str) -> Recorder | None:
    try:
        r = Recorder(thread_id)
        with _lock:
            _live[r.id] = r
        return r
    except Exception:
        return None


def cleanup():
    """Run off the request path at startup; no read endpoint mutates history."""
    if path().exists():
        try:
            with connect() as conn:
                _initialize(conn)
                _prune(conn)
        except Exception:
            pass


def finish(r: Recorder | None, status: str, error: str | None = None):
    if r:
        try:
            r.finish(status, error)
        except Exception:
            r.incomplete = True
            r.closed.set()
        finally:
            with _lock:
                _live.pop(r.id, None)


def bind(r: Recorder | None, iterator: Iterator) -> Iterator:
    while True:
        token = _run.set(r)
        try:
            item = next(iterator)
        except StopIteration:
            return
        finally:
            _run.reset(token)
        yield item


def ids() -> tuple[str | None, str | None]:
    r, s = _run.get(), _span.get()
    return (r.id if r else None, s.id if s else None)


class Span:
    def __init__(self, kind: str, name: str, label: str, inputs: Any = None):
        self.recorder = _run.get()
        self.id = uuid.uuid4().hex
        parent = _span.get()
        self.parent_id = parent.id if parent else None
        self.kind, self.name, self.label = kind, name, label
        self.started, self.clock = time.time(), time.perf_counter()
        self.status, self.ended, self.duration = "running", None, None
        self.data: dict = {}
        if kind != "node" and parent:
            self.data["node_id"] = parent.id if parent.kind == "node" else parent.data.get("node_id")
            self.data["node_label"] = parent.label if parent.kind == "node" else parent.data.get("node_label")
        self.detail: dict = {}
        if inputs is not None:
            self.content("input", inputs)

    def content(self, key: str, value: Any):
        if self.recorder and self.recorder.capture and value is not None:
            try:
                self.detail[key] = scrub(value, 2000)
            except Exception:
                self.recorder.incomplete = True

    def save(self):
        if self.recorder:
            try:
                self.data["updated_at"] = time.time()
                self.recorder.last_activity = self.data["updated_at"]
                self.recorder.put("span", (self.id, self.recorder.id, self.parent_id, self.kind, self.name,
                    scrub(self.label, 180), self.status, self.started, self.ended, self.duration,
                    json.dumps(self.data, ensure_ascii=False), json.dumps(self.detail, ensure_ascii=False) if self.detail else None))
                # Tiny live notices share the existing stream; persistence does
                # not depend on a connected HTTP consumer.
                from langgraph.config import get_stream_writer

                try:
                    get_stream_writer()({"trace": {"kind": self.kind, "label": self.label,
                        "status": self.status, "phase": self.data.get("phase"), "run_id": self.recorder.id}})
                except RuntimeError:
                    pass
            except Exception:
                self.recorder.incomplete = True


@contextmanager
def span(kind: str, name: str, label: str, inputs: Any = None):
    if _run.get() is None:
        yield None
        return
    try:
        s = Span(kind, name, label, inputs)
    except Exception:
        _run.get().incomplete = True
        yield None
        return
    s.save()
    token = _span.set(s)
    try:
        yield s
    except BaseException as e:
        from langgraph.errors import GraphInterrupt

        s.status = "waiting" if isinstance(e, GraphInterrupt) else "cancelled" if isinstance(e, GeneratorExit) else "error"
        if s.status == "error":
            s.data["error"] = scrub(f"{type(e).__name__}: {e}", 600)
        raise
    else:
        s.status = "done"
    finally:
        _span.reset(token)
        s.ended = time.time()
        s.duration = max(0, int((time.perf_counter() - s.clock) * 1000))
        s.save()


def annotate(**data):
    if s := _span.get():
        try:
            s.data.update(data)
            s.save()
        except Exception:
            if s.recorder:
                s.recorder.incomplete = True


def content(**data):
    if s := _span.get():
        for key, value in data.items():
            s.content(key, value)


def activity():
    """In-memory heartbeat only; never write a row for every streamed token."""
    if r := _run.get():
        r.last_activity = time.time()


def snapshot(**data):
    """Publish bounded partial content at most once a second per model call."""
    s = _span.get()
    if not s or not s.recorder.capture:
        return
    now = time.monotonic()
    if now - getattr(s, "last_snapshot", 0) < 1:
        return
    signature = tuple(data.items())
    if signature == getattr(s, "snapshot_signature", None):
        return
    s.last_snapshot, s.snapshot_signature = now, signature
    content(**data)
    s.data["content_revision"] = s.data.get("content_revision", 0) + 1
    s.save()


def progress(message: str, **extra):
    r, s = _run.get(), _span.get()
    if not r:
        return
    activity()
    try:
        # Deliberately ignore draft, paragraphs, streamed tokens and raw results.
        if extra.get("delta") is not None or extra.get("phase") == "token":
            return
        r.put("event", (r.id, s.id if s else None, time.time(), scrub(message, 500), extra.get("phase")))
        if extra.get("phase") == "error" and s:
            s.data["handled_error"] = scrub(message, 600)
    except Exception:
        r.incomplete = True


def traced(kind: str, label: str):
    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            if _run.get() is None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            inputs = None
            title = label
            if kind == "node":
                state = args[0] if args else {}
                index = state.get("index")
                outline = state.get("outline") or []
                if isinstance(index, int) and 0 <= index < len(outline) and fn.__name__ in ("write_one", "write_section", "review_section"):
                    title += " · " + str(outline[index].get("title", index + 1))
            else:
                inputs = {k: v for k, v in bound.items() if k in (
                    "query", "sources", "limit", "top_k", "paper_ids", "paper_id", "url", "section", "topic",
                    "paragraph", "title", "check_citations", "review")}
            with span(kind, fn.__name__, title, inputs) as s:
                if s and kind == "node" and isinstance(index, int) and 0 <= index < len(outline):
                    s.data["section_index"] = index
                    s.data["section_title"] = scrub(outline[index].get("title", ""), 180)
                result = fn(*args, **kwargs)
                if s:
                    if kind == "node" and isinstance(result, dict):
                        if fn.__name__ == "write_one" and any(p.get("status") == "error" for p in result.get("parts", {}).values()):
                            s.data["outcome"] = "error"
                        if fn.__name__ == "prepare":
                            s.data["planned_sections"] = len(result.get("order", []))
                            s.data["mode"] = state.get("mode")
                        before = (args[0].get("errors") or []) if args else []
                        errors = result.get("errors") or []
                        if len(errors) > len(before):
                            s.data["outcome"] = "partial"
                            s.data["handled_error"] = scrub("；".join(str(x) for x in errors[len(before):]), 600)
                    if isinstance(result, (list, tuple)):
                        s.data["result_count"] = len(result[0]) if isinstance(result, tuple) and isinstance(result[0], list) else len(result)
                        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], list) and result[1]:
                            s.data["warnings"] = [scrub(x, 300) for x in result[1][:5]]
                    elif isinstance(result, int):
                        s.data["result_count"] = result
                    if kind == "tool":
                        try:
                            _tool_summary(s, fn.__name__, bound, result)
                        except Exception:
                            s.recorder.incomplete = True
                        s.content("output", result)
                return result
        return wrapped
    return decorate


def _tool_summary(s: Span, name: str, bound: dict, result: Any):
    """Readable result projections are opt-in, bounded, valid JSON, never repr."""
    if name == "annotate" and isinstance(result, list):
        citations = [x for a in result for x in a.get("items", []) if isinstance(a, dict)]
        s.data.update(checked_count=len(citations), issue_count=sum(x.get("verdict") != "supports" for x in citations)
                      + sum(a.get("kind") != "cite_check" for a in result), annotation_count=len(result))
    if name == "gather_evidence" and isinstance(result, list):
        s.data["notes_count"] = sum(x.get("source") == "note" for x in result)
    if not s.recorder.capture:
        return
    values = result[0] if isinstance(result, tuple) and result and isinstance(result[0], list) else result
    if not isinstance(values, list) and not bound.get("query") and not bound.get("paragraph"):
        return
    structured: dict = {"query": str(bound.get("query", ""))[:240], "items": []}
    fields = ("paper_id", "title", "bibkey", "page", "year", "source", "url", "pdf_url", "text", "anchor", "reason", "verdict", "kind")
    if isinstance(values, list):
        for item in values[:6]:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "__dataclass_fields__"):
                from dataclasses import asdict
                item = asdict(item)
            if not isinstance(item, dict):
                continue
            entry = {k: (str(item[k])[:140] if isinstance(item[k], str) and k not in ("url", "pdf_url") else item[k])
                     for k in fields if item.get(k) is not None and not (k in ("url", "pdf_url") and len(str(item[k])) > 600)}
            if name == "annotate":
                entry["checks"] = [{k: str(v)[:100] for k, v in check.items() if k in ("title", "verdict", "reason", "paper_id", "page")}
                                   for check in item.get("items", [])[:2]]
            structured["items"].append(entry)
    if bound.get("paragraph"):
        structured["anchor"] = str(bound["paragraph"])[:160]
    structured["total"] = s.data.get("result_count", 0)
    structured["type"] = "citations" if name == "annotate" else "evidence" if name in ("gather_evidence", "search") and not isinstance(result, tuple) else "papers"
    while len(json.dumps(structured, ensure_ascii=False)) > 1800 and structured["items"]:
        structured["items"].pop()
    structured["sampled"] = len(structured["items"]) < structured["total"]
    s.content("structured", structured)


def overview(thread_id: str, run_id: str):
    base = read(thread_id, run_id, limit=1)
    if base is None:
        return None
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT id,parent_id,kind,name,label,status,started,ended,duration_ms,data,detail IS NOT NULL AS has_detail "
                                             "FROM spans WHERE run_id=? ORDER BY started,id", (run_id,))]
    for row in rows:
        row["data"] = json.loads(row["data"])
        row["has_detail"] = bool(row["has_detail"])
        if row["status"] == "running" and base["run"]["status"] not in ("running", "stopping"):
            row["status"] = "interrupted"
    nodes = [s for s in rows if s["kind"] == "node"]
    good_ops = {s["data"].get("operation") for s in rows if s["kind"] == "llm" and s["status"] == "done"} - {None}
    failures = [s for s in rows if s["status"] == "error" or s["data"].get("handled_error") or s["data"].get("warnings")]
    parents = {s["parent_id"] for s in failures if s["status"] == "error"}
    issues = []
    for s in failures:
        if s["id"] in parents:
            continue
        recovered = s["status"] == "error" and s["data"].get("operation") in good_ops
        issues.append({"id": s["id"], "node_id": s["data"].get("node_id", s["id"] if s["kind"] == "node" else None),
                       "label": s["label"], "state": "recovered" if recovered else "warning" if s["data"].get("warnings") else "error",
                       "message": s["data"].get("error") or s["data"].get("handled_error") or "；".join(s["data"].get("warnings", []))})
    grouped: dict[str, list] = {}
    for s in rows:
        grouped.setdefault(s["data"].get("node_id"), []).append(s)
    for n in nodes:
        children = grouped.get(n["id"], [])
        n["data"]["call_count"] = sum(s["kind"] == "llm" for s in children)
        n["data"]["tool_count"] = sum(s["kind"] == "tool" for s in children)
        n["data"]["result_count"] = next((s["data"]["result_count"] for s in children if s["kind"] == "tool" and "result_count" in s["data"]), None)
    active = [s for s in rows if s["status"] == "running" and s["kind"] != "node"]
    last_activity = max([base["run"]["started"]] + [s["data"].get("updated_at", s["ended"] or s["started"]) for s in rows]
                       + [e["at"] for e in base["events"]])
    with _lock:
        live = _live.get(run_id)
    if live:
        last_activity = max(last_activity, live.last_activity)
    from .db import get_conn

    with get_conn() as conn:
        task = conn.execute("SELECT thread_id,doc_id,topic,kind,status FROM agent_threads WHERE thread_id=?", (thread_id,)).fetchone()
    return {**base, "spans": nodes[:500], "nodes_truncated": len(nodes) > 500, "issues": issues[:100],
            "issues_truncated": len(issues) > 100, "active_calls": active[:40], "last_activity": last_activity,
            "task": dict(task) if task else None, "has_more": False}


def node_calls(thread_id: str, run_id: str, node_id: str, offset: int = 0, limit: int = 50, focus: str | None = None):
    run = next((r for r in runs(thread_id) if r["id"] == run_id), None)
    if not run:
        return None
    with connect() as conn:
        if focus:
            target = conn.execute("SELECT started,id FROM spans WHERE run_id=? AND id=? AND json_extract(data,'$.node_id')=?",
                                  (run_id, focus, node_id)).fetchone()
            if target:
                rank = conn.execute("SELECT COUNT(*) FROM spans WHERE run_id=? AND json_extract(data,'$.node_id')=? "
                    "AND (started < ? OR (started=? AND id < ?))", (run_id, node_id, target["started"], target["started"], target["id"])).fetchone()[0]
                offset = (rank // limit) * limit
        rows = [dict(r) for r in conn.execute("SELECT id,parent_id,kind,name,label,status,started,ended,duration_ms,data,detail IS NOT NULL AS has_detail "
            "FROM spans WHERE run_id=? AND json_extract(data,'$.node_id')=? ORDER BY started,id LIMIT ? OFFSET ?",
            (run_id, node_id, limit + 1, offset))]
    for row in rows:
        row["data"] = json.loads(row["data"])
        row["has_detail"] = bool(row["has_detail"])
        if row["status"] == "running" and run["status"] not in ("running", "stopping"):
            row["status"] = "interrupted"
    return {"spans": rows[:limit], "has_more": len(rows) > limit, "offset": offset}


def runs(thread_id: str) -> list[dict]:
    if not path().exists():
        return []
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM runs WHERE thread_id=? ORDER BY started DESC LIMIT 100", (thread_id,))]
    from .agent.runtime import active
    with _lock:
        live = dict(_live)

    for row in rows:
        row["capture"], row["incomplete"] = bool(row["capture"]), bool(row["incomplete"])
        if row["status"] == "running" and row["id"] not in live:
            row["status"] = "interrupted"
            row["incomplete"] = True
        elif row["id"] in live:
            row["incomplete"] = live[row["id"]].incomplete
            a = active(thread_id)
            if a and a.stop_requested.is_set():
                row["status"] = "stopping"
    return rows


def read(thread_id: str, run_id: str, offset: int = 0, limit: int = 50) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=? AND thread_id=?", (run_id, thread_id)).fetchone()
        if not row:
            return None
        # Detail is fetched separately, never sent in the timeline response.
        items = [dict(r) for r in conn.execute("SELECT id,parent_id,kind,name,label,status,started,ended,duration_ms,data,detail IS NOT NULL AS has_detail "
                 "FROM spans WHERE run_id=? ORDER BY started,id LIMIT ? OFFSET ?", (run_id, limit + 1, offset))]
        events = [dict(r) for r in conn.execute("SELECT * FROM events WHERE run_id=? ORDER BY id DESC LIMIT 100", (run_id,))]
        counts = {r["kind"]: r["n"] for r in conn.execute("SELECT kind,COUNT(*) AS n FROM spans WHERE run_id=? GROUP BY kind", (run_id,))}
        call_counts = {r["node_id"]: r["n"] for r in conn.execute("SELECT json_extract(data,'$.node_id') AS node_id,COUNT(*) AS n FROM spans "
                       "WHERE run_id=? AND kind='llm' GROUP BY node_id", (run_id,))}
        usage = dict(conn.execute("SELECT COUNT(*) AS calls, "
                     "SUM(CASE WHEN json_extract(data,'$.prompt_tokens') IS NULL OR json_extract(data,'$.completion_tokens') IS NULL THEN 1 ELSE 0 END) AS unknown_calls, "
                     "SUM(COALESCE(json_extract(data,'$.prompt_tokens'),0)+COALESCE(json_extract(data,'$.completion_tokens'),0)) AS known_tokens, "
                     "MAX(COALESCE(json_extract(data,'$.estimated'),0)) AS estimated FROM spans WHERE run_id=? AND kind='llm'", (run_id,)).fetchone())
    status = next((r for r in runs(thread_id) if r["id"] == run_id), dict(row))
    for item in items:
        item["has_detail"] = bool(item["has_detail"])
        item["data"] = json.loads(item["data"])
        if item["kind"] == "node":
            item["data"]["call_count"] = call_counts.get(item["id"], 0)
        if item["status"] == "running" and status["status"] == "interrupted":
            item["status"] = "interrupted"
    return {"run": status, "spans": items[:limit], "has_more": len(items) > limit,
            "events": list(reversed(events)), "summary": {**usage, "tools": counts.get("tool", 0), "steps": counts.get("node", 0)}}


def detail(thread_id: str, run_id: str, span_id: str):
    with connect() as conn:
        row = conn.execute("SELECT s.detail FROM spans s JOIN runs r ON r.id=s.run_id WHERE r.thread_id=? AND r.id=? AND s.id=?",
                           (thread_id, run_id, span_id)).fetchone()
    return None if not row else {"detail": json.loads(row["detail"]) if row["detail"] else None}


def clear(thread_id: str):
    if path().exists():
        with connect() as conn:
            conn.execute("DELETE FROM runs WHERE thread_id=?", (thread_id,))


def diagnostic(thread_id: str, run_id: str) -> dict | None:
    data = read(thread_id, run_id, limit=200)
    if data is None:
        return None
    # A compact allowlist; no prompts, model output, paper text or reasoning.
    r = data["run"]
    with connect() as conn:
        failures = [dict(s) for s in conn.execute("SELECT id,parent_id,kind,name,status,data FROM spans WHERE run_id=? "
                   "AND (status='error' OR json_extract(data, '$.handled_error') IS NOT NULL) ORDER BY started LIMIT 100", (run_id,))]
    for s in failures:
        s["data"] = json.loads(s["data"])
    return {"thread_id": thread_id, "run_id": run_id, "version": r["version"], "status": r["status"],
            "started": r["started"], "ended": r["ended"], "incomplete": r["incomplete"], "error": r["error"],
            "failures": failures}
