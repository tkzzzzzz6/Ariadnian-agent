from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault("LLM_FAKE", "1")
os.environ.setdefault("ARIA_DATA_DIR", tempfile.mkdtemp(prefix="aria-trace-tests-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app import llm, tracing
from app.db import get_conn
from app.main import app

client = TestClient(app)


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "path", lambda: tmp_path / "traces.sqlite")
    p = {"capture": False, "retention_days": 30, "detail_days": 7}
    monkeypatch.setattr(tracing, "policy", lambda: dict(p))
    return p


@contextmanager
def recorded(tid="test-task"):
    r = tracing.start(tid)
    assert r is not None
    token = tracing._run.set(r)
    try:
        yield r
    finally:
        tracing._run.reset(token)
        tracing.finish(r, "done")
        r.writer.join(2)


def events(response):
    return [json.loads(x[6:]) for x in response.text.splitlines() if x.startswith("data: ")]


def wait_runs(tid, count):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        rows = tracing.runs(tid)
        if len(rows) == count and all(r["ended"] is not None for r in rows):
            return rows
        time.sleep(0.01)
    raise AssertionError("trace writer did not finish")


def test_review_interrupt_resume_and_old_task(trace_db):
    first = events(client.post("/api/agent/start", json={"topic": "agent observability", "fetch_pdf": False, "max_papers": 2}))
    tid = next(e["thread_id"] for e in first if e["type"] == "status")
    rows = wait_runs(tid, 1)
    assert rows[0]["status"] == "waiting"
    page = client.get(f"/api/agent/{tid}/traces/{rows[0]['id']}").json()
    assert any(s["name"] == "select_papers" and s["status"] == "waiting" for s in page["spans"])
    assert any(s["kind"] == "llm" and s["data"]["estimated"] for s in page["spans"])
    assert any(s["kind"] == "tool" for s in page["spans"])
    assert not any(s["has_detail"] for s in page["spans"])
    resumed = events(client.post(f"/api/agent/{tid}/resume", json={"value": {"selected": []}}))
    assert resumed[-1]["type"] == "done"
    rows = wait_runs(tid, 2)
    assert len({r["id"] for r in rows}) == 2
    assert rows[0]["status"] == "done"
    old = client.get("/api/agent/legacy-task/traces").json()
    assert old["runs"] == []
    assert client.get(f"/api/agent/other-task/traces/{rows[0]['id']}").status_code == 404
    assert client.get(f"/api/agent/{tid}/traces/{rows[0]['id']}?limit=999999").status_code == 422


def test_parallel_writer_children_are_linked_to_own_nodes(trace_db):
    ev = events(client.post("/api/agent/write", json={"topic": "test", "mode": "parallel", "outline": [
        {"title": "Introduction"}, {"title": "Related Work"}]}))
    tid = next(e["thread_id"] for e in ev if e["type"] == "status")
    row = wait_runs(tid, 1)[0]
    assert row["status"] == "done"
    data = tracing.read(tid, row["id"], limit=200)
    by_id = {s["id"]: s for s in data["spans"]}
    nodes = [s for s in data["spans"] if s["name"] == "write_one"]
    assert len(nodes) == 2 and len({s["label"] for s in nodes}) == 2
    for node in nodes:
        children = [s for s in by_id.values() if s["parent_id"] == node["id"]]
        assert any(s["kind"] == "llm" for s in children)
        assert any(s["name"] == "gather_evidence" for s in children)
    assert not row["incomplete"]


def test_capture_redacts_truncates_and_diagnostic_excludes_content(trace_db, monkeypatch):
    trace_db["capture"] = True
    monkeypatch.setenv("OPENAI_API_KEY", "secret-for-trace-test")
    with recorded() as r:
        with tracing.span("node", "write", "写作"):
            with pytest.raises(RuntimeError):
                with tracing.span("tool", "search", "检索", {"query": "private-paper-content", "api_key": "secret-for-trace-test"}):
                    tracing.content(output="private-result" * 1000)
                    raise RuntimeError("Authorization: Bearer secret-for-trace-test password=hunter2")
    page = tracing.read(r.thread_id, r.id)
    s = next(s for s in page["spans"] if s["kind"] == "tool")
    d = tracing.detail(r.thread_id, r.id, s["id"])["detail"]
    assert "已截断" in d["output"]
    combined = json.dumps([page, d], ensure_ascii=False)
    assert "secret-for-trace-test" not in combined and "hunter2" not in combined
    diagnostic = json.dumps(tracing.diagnostic(r.thread_id, r.id))
    assert "private-paper-content" not in diagnostic and "private-result" not in diagnostic
    assert len(json.loads(diagnostic)["failures"]) == 1
    assert tracing.detail("wrong-task", r.id, s["id"]) is None


def test_secret_rotation_during_run_stays_redacted(trace_db, monkeypatch):
    trace_db["capture"] = True
    monkeypatch.setenv("OPENAI_API_KEY", "old-private-key")
    with recorded() as r:
        monkeypatch.setenv("OPENAI_API_KEY", "new-private-key")
        with tracing.span("tool", "request", "执行", "old-private-key new-private-key"):
            pass
    row = tracing.read(r.thread_id, r.id)["spans"][0]
    content = tracing.detail(r.thread_id, r.id, row["id"])["detail"]["input"]
    assert "old-private-key" not in content and "new-private-key" not in content


def test_model_reasoning_is_separate_from_document_and_logs_link(trace_db, monkeypatch):
    trace_db["capture"] = True
    settings = llm.get_settings()
    tiers = {k: dataclasses.replace(v, api_key="test-key") for k, v in settings.tiers.items()}
    monkeypatch.setattr(llm, "get_settings", lambda: dataclasses.replace(settings, llm_fake=False, tiers=tiers))
    response = NS(choices=[NS(message=NS(content="document-body", reasoning_content="provider-reasoning"))],
                  usage=NS(prompt_tokens=9, completion_tokens=5, completion_tokens_details={"reasoning_tokens": 2}))
    monkeypatch.setattr(llm.litellm, "completion", lambda **kw: response)
    with recorded() as r:
        with tracing.span("node", "draft", "写作"):
            assert llm.chat([{"role": "system", "content": "private-system"}, {"role": "user", "content": "question"}]) == "document-body"
    s = next(s for s in tracing.read(r.thread_id, r.id)["spans"] if s["kind"] == "llm")
    assert s["data"]["reasoning_available"] and s["data"]["reasoning_tokens"] == 2
    detail = tracing.detail(r.thread_id, r.id, s["id"])["detail"]
    assert detail["reasoning"] == "provider-reasoning" and "private-system" not in detail["input"]
    with get_conn() as conn:
        call = conn.execute("SELECT run_id,span_id FROM llm_calls WHERE run_id=?", (r.id,)).fetchone()
    assert call["span_id"] == s["id"]


def test_stream_reasoning_first_body_and_unknown_usage(trace_db, monkeypatch):
    trace_db["capture"] = True
    settings = llm.get_settings()
    tiers = {k: dataclasses.replace(v, api_key="test-key") for k, v in settings.tiers.items()}
    monkeypatch.setattr(llm, "get_settings", lambda: dataclasses.replace(settings, llm_fake=False, tiers=tiers))
    def chunks(**kw):
        yield NS(choices=[NS(delta=NS(content=None, reasoning_content="reason"))])
        yield NS(choices=[NS(delta=NS(content="body"))])
    monkeypatch.setattr(llm.litellm, "completion", chunks)
    with recorded() as r:
        assert "".join(llm.stream_chat([{"role": "user", "content": "question"}])) == "body"
    s = tracing.read(r.thread_id, r.id)["spans"][0]
    assert s["data"]["first_reasoning_ms"] is not None and s["data"]["first_token_ms"] is not None
    assert s["data"]["prompt_tokens"] is None
    assert tracing.detail(r.thread_id, r.id, s["id"])["detail"]["reasoning"] == "reason"


def test_writer_failure_is_nonblocking_and_context_does_not_leak(trace_db, monkeypatch):
    def broken():
        raise OSError("disk unavailable")
    monkeypatch.setattr(tracing, "connect", broken)
    started = time.monotonic()
    with recorded() as r:
        with tracing.span("tool", "work", "执行"):
            result = 42
    assert result == 42 and r.incomplete
    assert time.monotonic() - started < 1
    assert tracing.ids() == (None, None)


def test_retention_and_clear_preserve_task_and_checkpoint(trace_db):
    trace_db["capture"] = True
    with recorded() as old:
        with tracing.span("tool", "work", "执行", "input"):
            pass
    with tracing.connect() as conn:
        conn.execute("UPDATE runs SET started=? WHERE id=?", (time.time() - 8 * 86400, old.id))
        tracing._prune(conn)
    assert not tracing.read(old.thread_id, old.id)["spans"][0]["has_detail"]
    with tracing.connect() as conn:
        conn.execute("UPDATE runs SET started=? WHERE id=?", (time.time() - 31 * 86400, old.id))
        tracing._prune(conn)
    assert tracing.runs(old.thread_id) == []
    ev = events(client.post("/api/agent/start", json={"topic": "clear trace", "fetch_pdf": False}))
    tid = next(e["thread_id"] for e in ev if e["type"] == "status")
    wait_runs(tid, 1)
    assert client.delete(f"/api/agent/{tid}/traces").status_code == 200
    assert tracing.runs(tid) == []
    assert client.get(f"/api/agent/{tid}").json()["status"] == "waiting"


def test_partial_run_after_process_loss_is_marked_interrupted(trace_db):
    with recorded() as r:
        with tracing.span("node", "work", "执行"):
            pass
    with tracing.connect() as conn:
        conn.execute("UPDATE runs SET status='running',ended=NULL WHERE id=?", (r.id,))
        conn.execute("UPDATE spans SET status='running',ended=NULL,duration_ms=NULL WHERE run_id=?", (r.id,))
    data = tracing.read(r.thread_id, r.id)
    assert data["run"]["status"] == "interrupted" and data["run"]["incomplete"]
    assert data["spans"][0]["status"] == "interrupted"


def test_visible_fallback_attempts_and_queue_limit(trace_db, monkeypatch):
    settings = llm.get_settings()
    tiers = {k: dataclasses.replace(v, model="openai/bad", fallbacks=("openai/good",), api_key="test-key") for k, v in settings.tiers.items()}
    monkeypatch.setattr(llm, "get_settings", lambda: dataclasses.replace(settings, llm_fake=False, tiers=tiers))
    def complete(**kw):
        if kw["model"].endswith("bad"):
            raise RuntimeError("temporary failure")
        return NS(choices=[NS(message=NS(content="ok"))], usage=None)
    monkeypatch.setattr(llm.litellm, "completion", complete)
    with recorded() as r:
        with tracing.span("node", "work", "执行"):
            assert llm.chat([{"role": "user", "content": "question"}]) == "ok"
    data = tracing.read(r.thread_id, r.id)
    calls = [s for s in data["spans"] if s["kind"] == "llm"]
    assert [s["status"] for s in calls] == ["error", "done"]
    assert data["summary"]["calls"] == 2 and data["summary"]["unknown_calls"] == 2
    node = next(s for s in data["spans"] if s["kind"] == "node")
    assert node["data"]["call_count"] == 2
    with recorded("bounded") as bounded:
        bounded.count = 20000
        bounded.put("event", (bounded.id, None, time.time(), "over limit", None))
    assert bounded.incomplete


def test_scholar_source_threads_inherit_trace_context(trace_db, monkeypatch):
    from app import scholar

    settings = scholar.get_settings()
    monkeypatch.setattr(scholar, "get_settings", lambda: dataclasses.replace(settings, llm_fake=False))
    monkeypatch.setattr(scholar, "_cache_get", lambda key: [])
    with recorded() as r:
        assert scholar.search("query", ["arxiv", "openalex"])[0] == []
    data = tracing.read(r.thread_id, r.id)
    sources = [s for s in data["spans"] if s["name"] == "_cached"]
    assert len(sources) == 2 and all(s["data"]["cache_hit"] for s in sources)
    assert len({s["parent_id"] for s in sources}) == 1
    assert {s["data"]["source"] for s in sources} == {"arxiv", "openalex"}


def test_overview_recovered_attempts_are_distinct_from_unresolved_errors(trace_db, monkeypatch):
    settings = llm.get_settings()
    tiers = {k: dataclasses.replace(v, model="openai/bad", fallbacks=("openai/good",), api_key="test-key") for k, v in settings.tiers.items()}
    monkeypatch.setattr(llm, "get_settings", lambda: dataclasses.replace(settings, llm_fake=False, tiers=tiers))
    def complete(**kw):
        if kw["model"].endswith("bad"):
            raise RuntimeError("temporary failure")
        return NS(choices=[NS(message=NS(content="ok"))], usage=None)
    monkeypatch.setattr(llm.litellm, "completion", complete)
    with recorded() as r:
        with tracing.span("node", "work", "执行"):
            assert llm.chat([{"role": "user", "content": "question"}]) == "ok"
            try:
                with tracing.span("tool", "download", "下载"):
                    raise ValueError("unresolved download")
            except ValueError:
                pass
    data = tracing.overview(r.thread_id, r.id)
    assert len(data["spans"]) == 1  # compact overview contains stages, not every call
    assert data["spans"][0]["data"]["call_count"] == 2
    assert sorted(i["state"] for i in data["issues"]) == ["error", "recovered"]
    assert "structured" not in json.dumps(data)


def test_global_issue_navigation_finds_call_beyond_first_page(trace_db):
    with recorded() as r:
        with tracing.span("node", "work", "执行") as node:
            for _ in range(65):
                with tracing.span("tool", "search", "检索"):
                    pass
            try:
                with tracing.span("tool", "last", "最后一个失败调用") as target:
                    raise RuntimeError("last call failed")
            except RuntimeError:
                pass
    data = tracing.overview(r.thread_id, r.id)
    assert data["issues"][0]["id"] == target.id
    page = tracing.node_calls(r.thread_id, r.id, node.id, focus=target.id)
    assert page["offset"] == 50 and any(s["id"] == target.id for s in page["spans"])
    assert tracing.node_calls("different-task", r.id, node.id) is None


def test_tool_projections_are_valid_bounded_and_opt_in(trace_db):
    @tracing.traced("tool", "检索")
    def search(query):
        return ([{"title": "论文" * 80, "url": "https://example.org/paper", "text": "evidence" * 100} for _ in range(10)], [])
    with recorded("no-capture") as a:
        search("private-query")
    row = tracing.read(a.thread_id, a.id)["spans"][0]
    assert not row["has_detail"]
    trace_db["capture"] = True
    with recorded("capture") as b:
        search("private-query")
    row = tracing.read(b.thread_id, b.id)["spans"][0]
    detail = tracing.detail(b.thread_id, b.id, row["id"])["detail"]
    projection = json.loads(detail["structured"])
    assert projection["query"] == "private-query" and projection["total"] == 10 and projection["sampled"]
    assert 0 < len(projection["items"]) <= 6 and len(detail["structured"]) < 2000
    assert "private-query" not in json.dumps(tracing.overview(b.thread_id, b.id))


def test_partial_content_is_throttled_and_activity_does_not_write(trace_db):
    trace_db["capture"] = True
    with recorded() as r:
        with tracing.span("llm", "model", "模型") as call:
            tracing.snapshot(reasoning="first")
            tracing.snapshot(reasoning="second")
            assert call.data["content_revision"] == 1
            assert call.detail["reasoning"] == "first"
            call.last_snapshot -= 2
            tracing.snapshot(reasoning="first, then a longer chunk")
            assert call.data["content_revision"] == 2 and call.detail["reasoning"] == "first, then a longer chunk"
            before = r.count
            for _ in range(1000):
                tracing.activity()
            assert r.count == before


def test_handled_parallel_failure_is_not_counted_as_a_successful_chapter(trace_db, monkeypatch):
    from app.agent import writer

    original = writer._write
    def fail_one(state, index):
        if index == 0:
            raise RuntimeError("chapter failed")
        return original(state, index)
    monkeypatch.setattr(writer, "_write", fail_one)
    ev = events(client.post("/api/agent/write", json={"mode": "parallel", "outline": [{"title": "One"}, {"title": "Two"}]}))
    tid = next(e["thread_id"] for e in ev if e["type"] == "status")
    row = wait_runs(tid, 1)[0]
    data = tracing.overview(tid, row["id"])
    failed = [n for n in data["spans"] if n["name"] == "write_one" and n["data"].get("outcome") == "error"]
    assert len(failed) == 1 and data["issues"]


def test_live_preview_keeps_newest_chunks_after_reaching_content_limit():
    preview = ""
    for i in range(30):
        preview = llm._append_preview(preview, f"chunk-{i}-" + "x" * 200)
    assert len(preview) <= 1800
    assert preview.startswith("chunk-0-") and "chunk-29-" in preview
    assert "中间内容已省略" in preview
