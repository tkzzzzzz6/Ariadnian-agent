"""Exercise disconnect/stop while a real LangGraph node is still running.

The HTTP body must never own or finalize the graph iterator. Synchronization
uses events so the regressions do not depend on model latency or paid calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("LLM_FAKE", "1")
os.environ.setdefault("ARIA_DATA_DIR", tempfile.mkdtemp(prefix="aria-runtime-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from starlette.requests import ClientDisconnect

from app.agent.runtime import active
from app import runstats, tracing
from app.routers import agent


class State(TypedDict, total=False):
    topic: str
    count: int


@pytest.fixture
def blocked_graph(monkeypatch):
    entered = threading.Event()
    proceed = threading.Event()
    calls = []
    owners = []

    def first(state):
        calls.append("first")
        get_stream_writer()({"message": "first node is running"})
        entered.set()
        assert proceed.wait(10), "test did not release its node"
        return {"count": 1}

    def second(state):
        calls.append("second")
        return {"count": state["count"] + 1}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    class TrackedGraph:
        def stream(self, *args, **kwargs):
            owner = threading.get_ident()
            try:
                yield from graph.stream(*args, **kwargs)
            finally:
                owners.append((owner, threading.get_ident()))

        def get_state(self, config):
            return graph.get_state(config)

    monkeypatch.setattr(agent, "_graph_for", lambda tid: TrackedGraph())
    yield entered, proceed, calls, owners, graph
    proceed.set()


def state_for(graph, tid):
    return graph.get_state({"configurable": {"thread_id": tid}})


def test_disconnect_closes_only_http_body_and_preserves_checkpoint(blocked_graph):
    entered, proceed, calls, owners, graph = blocked_graph
    response = agent.start(agent.StartIn(topic="disconnect regression"))

    async def disconnect():
        body = response.body_iterator
        first = json.loads((await anext(body)).removeprefix("data: "))
        await body.aclose()
        return first["thread_id"]

    tid = asyncio.run(disconnect())
    run = active(tid)
    assert run is not None and run.stop_requested.is_set()
    assert entered.wait(5)
    assert not owners  # closing HTTP did not try to close the live graph
    with pytest.raises(HTTPException) as exc:
        agent.resume(tid, agent.ResumeIn(value={}))
    assert exc.value.status_code == 409
    proceed.set()
    assert run.finished.wait(5), "worker deadlocked while closing its graph"
    assert active(tid) is None
    assert calls == ["first"]
    assert state_for(graph, tid).values["count"] == 1
    assert agent.get_thread(tid)["status"] == "stopped"
    assert tracing.runs(tid)[0]["status"] == "stopped"
    assert owners and all(a == b for a, b in owners)

    async def resume():
        response = agent.resume(tid, agent.ResumeIn(value={}))
        return [json.loads(event.removeprefix("data: ")) async for event in response.body_iterator
                if event.startswith("data: ")]

    events = asyncio.run(resume())
    assert events[-1]["type"] == "done"
    assert calls == ["first", "second"]  # completed node wasn't replayed
    assert [r["status"] for r in tracing.runs(tid)] == ["done", "stopped"]
    assert state_for(graph, tid).values["count"] == 2
    assert active(tid) is None


def test_stop_keeps_stream_open_and_blocks_resume_until_checkpoint(blocked_graph):
    entered, proceed, calls, owners, graph = blocked_graph
    response = agent.start(agent.StartIn(topic="stop regression"))

    async def stop_and_drain():
        body = response.body_iterator
        initial = json.loads((await anext(body)).removeprefix("data: "))
        tid = initial["thread_id"]
        assert entered.wait(5)
        assert agent.stop(tid) == {"ok": True, "stopping": True}
        assert agent.get_thread(tid)["stopping"] is True
        assert agent.get_thread(tid)["status"] == "running"
        with pytest.raises(HTTPException) as exc:
            agent.resume(tid, agent.ResumeIn(value={}))
        assert exc.value.status_code == 409
        proceed.set()
        events = [json.loads(event.removeprefix("data: ")) async for event in body
                  if event.startswith("data: ")]
        return tid, events

    tid, events = asyncio.run(stop_and_drain())
    assert tracing.runs(tid)[0]["status"] == "stopped"
    assert events[-1] == {"type": "status", "status": "stopped"}
    assert calls == ["first"]
    assert state_for(graph, tid).values["count"] == 1
    assert active(tid) is None
    assert owners and all(a == b for a, b in owners)


def test_failure_releases_task_for_recovery(monkeypatch):
    class BrokenGraph:
        def stream(self, *args, **kwargs):
            raise RuntimeError("node failed")
            yield  # make this a generator, matching LangGraph's interface

    monkeypatch.setattr(agent, "_graph_for", lambda tid: BrokenGraph())
    response = agent.start(agent.StartIn(topic="error regression"))

    async def drain():
        return [json.loads(event.removeprefix("data: ")) async for event in response.body_iterator
                if event.startswith("data: ")]

    events = asyncio.run(drain())
    tid = events[0]["thread_id"]
    assert events[-1] == {"type": "error", "message": "RuntimeError: node failed"}
    assert active(tid) is None


def test_stop_resume_preserves_worker_model_usage_and_streamed_statistics(monkeypatch):
    entered = threading.Event()
    proceed = threading.Event()
    monkeypatch.setattr(runstats, "context_window", lambda model: 8192)

    def record_call():
        meter = runstats.current()
        assert meter is not None, "worker lost its task's model-usage context"
        meter.record("test-model", ok=True, latency=1.0, first_token=0.1,
                     prompt_tokens=10, completion_tokens=5, cached_tokens=2, cost=0.001)

    def first(state):
        record_call()
        entered.set()
        assert proceed.wait(10)
        return {"count": 1}

    def second(state):
        record_call()
        return {"count": state["count"] + 1}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(agent, "_graph_for", lambda tid: graph)

    async def drain(response):
        return [json.loads(event.removeprefix("data: ")) async for event in response.body_iterator
                if event.startswith("data: ")]

    response = agent.start(agent.StartIn(topic="worker statistics regression"))
    assert entered.wait(5)
    try:
        assert agent.stop(response.run.tid)["stopping"] is True
    finally:
        proceed.set()
    events = asyncio.run(drain(response))
    assert events[-1] == {"type": "status", "status": "stopped"}
    first_stats = runstats.load(response.run.tid)
    assert first_stats["calls"] == first_stats["steps"] == 1
    assert first_stats["total_tokens"] == 15
    assert [event["stats"] for event in events if event["type"] == "stats"][-1]["calls"] == 1

    resumed = asyncio.run(drain(agent.resume(response.run.tid, agent.ResumeIn(value={}))))
    assert resumed[-1]["type"] == "done"
    final_stats = agent.get_thread(response.run.tid)["stats"]
    assert final_stats["calls"] == final_stats["steps"] == 2
    assert final_stats["total_tokens"] == 30
    assert [event["stats"] for event in resumed if event["type"] == "stats"][-1]["calls"] == 2
    assert next(row for row in agent.list_threads() if row["thread_id"] == response.run.tid)["tokens"] == 30


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
def test_asgi_disconnect_requests_stop_without_finalizing_graph(blocked_graph, spec):
    entered, proceed, calls, owners, graph = blocked_graph
    response = agent.start(agent.StartIn(topic="ASGI disconnect regression"))
    assert entered.wait(5)

    async def disconnect():
        sent = asyncio.Event()
        initial = {}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body", b"").startswith(b"data: "):
                initial.update(json.loads(message["body"].decode().removeprefix("data: ")))
                sent.set()
                if spec == "2.4":
                    raise OSError("client disconnected during send")

        async def receive():
            await sent.wait()
            return {"type": "http.disconnect"}

        try:
            await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": spec}}, receive, send), 2)
        except ClientDisconnect:
            assert spec == "2.4"
        return initial["thread_id"]

    tid = asyncio.run(disconnect())
    run = active(tid)
    assert run is not None and run.stop_requested.is_set()
    assert not owners
    proceed.set()
    assert run.finished.wait(5)
    assert calls == ["first"]
    assert state_for(graph, tid).values["count"] == 1
    assert agent.get_thread(tid)["status"] == "stopped"
