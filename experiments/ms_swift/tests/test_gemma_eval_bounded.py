"""Bounded-worker failure contracts; no model, weights or GPU is launched."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cpt.core import digest, write_json, write_jsonl
from gemma_eval import bounded


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    plan = {"kind": "profile", "counts": {"ready": 3}, "model": {"repo": "fixture", "revision": "f" * 40}}
    requests = [{"id": str(i), "task": "synthetic_resource_probe", "status": "ready",
                 "input_ids": [2, 3], "input_tokens": 2, "max_new_tokens": 2} for i in range(3)]
    path = tmp_path / "plan.json"
    write_json(path, plan)
    monkeypatch.setattr(bounded, "load_plan", lambda _: (plan, requests))
    return path, plan, requests


def child_factory(plan, predictions, *, timeout=False, returncode=0):
    def create(command, **kwargs):
        assert command[1:4] == ["-m", "gemma_eval", "run"]
        assert "bounded" not in command
        assert kwargs["shell"] is False
        child = Path(command[command.index("--output") + 1])
        child.mkdir()
        write_jsonl(child / "predictions.jsonl", predictions)
        if not timeout:
            write_json(child / "summary.json", {"plan_sha256": digest(plan), "runtime": {"fixture": True}})
        proc = Mock(pid=12345)
        if timeout:
            proc.wait.side_effect = subprocess.TimeoutExpired(command, 1)
            proc.poll.return_value = None
        else:
            proc.wait.return_value = returncode
            proc.poll.return_value = returncode
        return proc
    return create


def test_timeout_preserves_completed_rows_and_every_unfinished_id(tmp_path, prepared):
    path, plan, requests = prepared
    real_row = {"id": "0", "status": "ok", "prediction": "observed fixture",
                "generation_seconds": 4.0, "example_sha256": None}
    def stopped(proc):
        proc.poll.return_value = -9
        return {"cleanup_requested": True, "cleanup_success": True}
    result = bounded.run_bounded(path, tmp_path / "out", max_seconds=1, execute=True,
        popen_factory=child_factory(plan, [real_row], timeout=True), stopper=stopped)
    rows = [json.loads(line) for line in (tmp_path / "out/predictions.jsonl").read_text().splitlines()]
    assert rows[0] == real_row
    assert [row["status"] for row in rows] == ["ok", "interrupted_timeout", "not_attempted_after_timeout"]
    assert rows[1]["generation_seconds"] is None
    assert result["total"] == 3 and result["successful"] == 1
    assert result["status"] == "timeout"
    assert json.loads((tmp_path / "out/run/predictions.jsonl").read_text()) == real_row


def test_zero_exit_with_failed_predictions_is_not_full_success(tmp_path, prepared):
    path, plan, _ = prepared
    rows = [{"id": str(i), "status": "failed" if i == 1 else "ok", "prediction": "" if i == 1 else "fixture"} for i in range(3)]
    result = bounded.run_bounded(path, tmp_path / "out", max_seconds=3, execute=True,
                                 popen_factory=child_factory(plan, rows))
    assert result["status"] == "completed_with_failures"
    assert result["successful"] == 2 and result["failed_or_skipped"] == 1
    assert result["worker_returncode"] == 0


def test_worker_start_exception_retains_full_denominator(tmp_path, prepared):
    path, _, _ = prepared
    result = bounded.run_bounded(path, tmp_path / "out", max_seconds=3, execute=True,
                                 popen_factory=Mock(side_effect=OSError("cannot start")))
    assert result["status"] == "start_failed"
    assert result["total"] == 3 and result["successful"] == 0
    assert result["statuses"] == {"worker_start_failed": 1, "not_attempted_after_start_failure": 2}
    assert result["error"] == {"type": "OSError", "stage": "worker_start"}


def test_failed_cleanup_is_not_reported_as_success(tmp_path, prepared):
    path, plan, _ = prepared
    result = bounded.run_bounded(path, tmp_path / "out", max_seconds=1, execute=True,
        popen_factory=child_factory(plan, [], timeout=True),
        stopper=lambda _: {"cleanup_requested": True, "cleanup_success": False, "cleanup_error_type": "TimeoutExpired"})
    assert result["cleanup"]["cleanup_success"] is False
    assert result["status"] == "timeout"


def test_inspection_never_spawns_or_creates_results(tmp_path, prepared):
    path, _, _ = prepared
    factory = Mock(side_effect=AssertionError("must not spawn"))
    result = bounded.run_bounded(path, tmp_path / "out", max_seconds=3, popen_factory=factory)
    assert result["executed"] is False
    assert not factory.called and not (tmp_path / "out").exists()


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, None, 86401])
def test_invalid_budget_rejected_before_loading_plan(tmp_path, budget):
    with pytest.raises(ValueError, match="max_seconds"):
        bounded.run_bounded(tmp_path / "absent", None, max_seconds=budget)


def test_windows_cleanup_targets_only_created_pid_tree(monkeypatch):
    proc = Mock(pid=12345)
    proc.poll.side_effect = [None, 0, 0]
    proc.wait.return_value = 0
    command = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(bounded.platform, "system", lambda: "Windows")
    monkeypatch.setattr(bounded.subprocess, "run", command)
    result = bounded.stop_owned_worker(proc)
    assert command.call_args.args[0] == ["taskkill", "/PID", "12345", "/T", "/F"]
    assert command.call_args.kwargs["shell"] is False
    assert result["cleanup_success"] is True


def test_existing_output_preserved(tmp_path, prepared):
    path, _, _ = prepared
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("existing evidence")
    with pytest.raises(ValueError, match="new or empty"):
        bounded.run_bounded(path, out, max_seconds=2, execute=True)
    assert (out / "summary.json").read_text() == "existing evidence"


@pytest.mark.parametrize("skip_index,expected", [
    (0, ["skipped_context_limit", "interrupted_timeout", "not_attempted_after_timeout"]),
    (1, ["interrupted_timeout", "skipped_context_limit", "not_attempted_after_timeout"]),
])
def test_timeout_preserves_prepared_overflow_without_consuming_interrupted_case(tmp_path, prepared, skip_index, expected):
    path, plan, requests = prepared
    requests[skip_index]["status"] = "skipped_context_limit"
    bounded.run_bounded(path, tmp_path / "out", max_seconds=1, execute=True,
        popen_factory=child_factory(plan, [], timeout=True), stopper=lambda _: {"cleanup_success": True})
    rows = [json.loads(line) for line in (tmp_path / "out/predictions.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == expected
    assert rows[skip_index]["transport_called"] is False


def test_child_summary_plan_mismatch_downgrades_successful_exit(tmp_path, prepared):
    path, plan, _ = prepared
    rows = [{"id": str(i), "status": "ok", "prediction": "fixture"} for i in range(3)]
    factory = child_factory(plan, rows)
    def mismatched_summary(command, **kwargs):
        proc = factory(command, **kwargs)
        child = Path(command[command.index("--output") + 1])
        write_json(child / "summary.json", {"plan_sha256": "wrong", "runtime": {"fixture": True}})
        return proc
    result = bounded.run_bounded(path, tmp_path / "out", max_seconds=3, execute=True, popen_factory=mismatched_summary)
    assert result["status"] == "incomplete_worker_output"
    assert result["runtime"] is None
    assert result["child_output_errors"] == [{"type": "child_summary_plan_mismatch"}]
