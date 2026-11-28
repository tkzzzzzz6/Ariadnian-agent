"""Run one fixed Gemma evaluation worker with an explicit wall-clock limit.

The wrapper never accepts arbitrary commands, alters driver policies, or deletes
partial evidence. Timeout is an interruption, not an OOM or measured latency.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from cpt.core import digest, file_hash, new_output, read_jsonl
from .core import load_plan
from .runtime import _atomic_json, _atomic_jsonl, implementation_hashes


def validate_budget(value):
    if type(value) is not int or not 1 <= value <= 86400:
        raise ValueError("max_seconds must be an explicit integer in [1,86400]")


def stop_owned_worker(proc):
    """Only the process this wrapper created and its descendants are targeted."""
    evidence = {"worker_pid": proc.pid, "cleanup_requested": False, "cleanup_success": False,
                "method": "taskkill_tree" if platform.system() == "Windows" else "owned_process_group"}
    if proc.poll() is not None:
        evidence.update(worker_exit_observed=True, note="Worker already exited; no tree cleanup attempted.")
        return evidence
    evidence["cleanup_requested"] = True
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                    shell=False, capture_output=True, timeout=15, check=False)
            evidence["cleanup_command_returncode"] = result.returncode
            if result.returncode == 0:
                proc.wait(timeout=10)
                evidence["cleanup_success"] = proc.poll() is not None
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # A group may still contain descendants after its leader exits.
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                evidence["cleanup_success"] = True
            else:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    evidence["cleanup_success"] = True
                else:
                    evidence["cleanup_success"] = False
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence["cleanup_error_type"] = type(exc).__name__
    evidence["worker_exit_observed"] = proc.poll() is not None
    if not evidence["cleanup_success"]:
        evidence["note"] = "Process-tree cleanup was not confirmed; no claim that all worker descendants stopped."
    return evidence


def _completed_rows(child_dir, requests):
    path = child_dir / "predictions.jsonl"
    if not path.exists():
        return [], []
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError) as exc:
        return [], [{"type": type(exc).__name__, "stage": "child_prediction_read"}]
    completed, errors = [], []
    allowed = {"ok", "failed", "model_load_failed", "skipped_context_limit", "empty_prediction"}
    for index, row in enumerate(rows):
        if (index >= len(requests) or not isinstance(row, dict) or row.get("id") != requests[index]["id"]
                or row.get("status") not in allowed or not isinstance(row.get("prediction"), str)
                or row.get("example_sha256") != requests[index].get("example_sha256")):
            errors.append({"type": "invalid_child_prediction_prefix", "row": index})
            break
        completed.append(row)
    return completed, errors


def _final_rows(requests, completed, reason):
    rows = list(completed)
    statuses = {
        "timeout": ("interrupted_timeout", "not_attempted_after_timeout"),
        "start_failed": ("worker_start_failed", "not_attempted_after_start_failure"),
        "interrupted": ("interrupted_by_user", "not_attempted_after_interruption"),
        "worker_exited": ("missing_worker_prediction", "not_attempted_after_worker_exit"),
    }
    first, rest = statuses[reason]
    unfinished_ready = 0
    for request in requests[len(completed):]:
        row = {key: value for key, value in request.items() if key not in {"input_ids", "prompt", "status"}}
        if request["status"] == "skipped_context_limit":
            row.update(status="skipped_context_limit", prediction="", evaluation_only=True,
                       prepared_status=request["status"], transport_called=False, generation_seconds=None,
                       note="The frozen CPU preparation already established overflow; no worker completion is required to retain this skip.")
            rows.append(row)
            continue
        row.update(status=first if unfinished_ready == 0 else rest, prediction="", evaluation_only=True,
                   prepared_status=request["status"], transport_called=None,
                   interruption_reason=reason, generation_seconds=None,
                   note="No completion measurement is available; interruption does not establish OOM or latency.")
        rows.append(row)
        unfinished_ready += 1
    return rows


def run_bounded(plan_path, output, *, max_seconds, execute=False, popen_factory=None, stopper=None):
    validate_budget(max_seconds)
    plan_path = Path(plan_path).resolve()
    plan, requests = load_plan(plan_path)
    if not execute:
        return {"executed": False, "max_seconds": max_seconds, "kind": plan["kind"],
                "plan_sha256": digest(plan), "plan_file_sha256": file_hash(plan_path),
                "total": len(requests), "counts": plan["counts"],
                "message": "Inspection only; --execute starts a bounded local worker with the current Python."}
    if output is None:
        raise ValueError("execution requires a fresh --output directory")
    out = new_output(output)
    child_dir = out / "run"
    command = [sys.executable, "-m", "gemma_eval", "run", "--plan", str(plan_path),
               "--output", str(child_dir), "--execute"]
    implementations = implementation_hashes()
    initial = {"schema_version": "gemma-bounded-run-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "starting", "executed": True, "plan_sha256": digest(plan),
               "plan_file_sha256": file_hash(plan_path), "implementation_sha256": implementations,
               "command": command, "max_seconds": max_seconds, "total": len(requests),
               "selected_ids": [r["id"] for r in requests], "child_directory": str(child_dir)}
    _atomic_json(out / "run_manifest.json", initial)
    proc, cleanup, error, reason = None, None, None, "worker_exited"
    started = time.monotonic()
    factory, stop = popen_factory or subprocess.Popen, stopper or stop_owned_worker
    returncode = None
    try:
        with (out / "worker.stdout.log").open("wb") as stdout, (out / "worker.stderr.log").open("wb") as stderr:
            proc = factory(command, stdout=stdout, stderr=stderr, shell=False,
                           start_new_session=platform.system() != "Windows")
            _atomic_json(out / "run_manifest.json", {**initial, "status": "running", "worker_pid": proc.pid})
            try:
                returncode = proc.wait(timeout=max_seconds)
            except subprocess.TimeoutExpired:
                reason = "timeout"
                cleanup = stop(proc)
                returncode = proc.poll()
    except KeyboardInterrupt:
        reason = "interrupted"
        if proc is not None:
            cleanup = stop(proc)
            returncode = proc.poll()
    except (OSError, ValueError) as exc:
        reason = "start_failed" if proc is None else "interrupted"
        error = {"type": type(exc).__name__, "stage": "worker_start" if proc is None else "worker_monitor"}
        if proc is not None:
            cleanup = stop(proc)
            returncode = proc.poll()
    elapsed = time.monotonic() - started
    completed, read_errors = _completed_rows(child_dir, requests)
    rows = _final_rows(requests, completed, reason)
    statuses = dict(Counter(row["status"] for row in rows))
    if reason == "worker_exited":
        if len(completed) != len(requests) or read_errors:
            status = "incomplete_worker_output"
        elif returncode == 0 and all(row["status"] == "ok" for row in rows):
            status = "completed"
        else:
            status = "completed_with_failures"
    else:
        status = reason
    child_summary = None
    summary_path = child_dir / "summary.json"
    if summary_path.is_file():
        try:
            candidate = json.loads(summary_path.read_text(encoding="utf-8"))
            if candidate.get("plan_sha256") == digest(plan):
                child_summary = candidate
            else:
                read_errors.append({"type": "child_summary_plan_mismatch"})
        except (OSError, ValueError) as exc:
            read_errors.append({"type": type(exc).__name__, "stage": "child_summary_read"})
    if read_errors and status in {"completed", "completed_with_failures"}:
        status = "incomplete_worker_output"
    _atomic_jsonl(out / "predictions.jsonl", rows)
    summary = {"schema_version": "gemma-bounded-summary-v1", "status": status, "kind": plan["kind"],
               "executed": True, "plan_sha256": digest(plan), "plan_file_sha256": file_hash(plan_path),
               "implementation_sha256": implementations, "model": plan["model"],
               "max_seconds": max_seconds, "wrapper_wall_seconds": elapsed,
               "worker_returncode": returncode, "cleanup": cleanup, "error": error,
               "child_output_errors": read_errors, "child_completed_records": len(completed),
               "selected_ids": [row["id"] for row in rows], "total": len(rows), "statuses": statuses,
               "successful": statuses.get("ok", 0), "failed_or_skipped": len(rows) - statuses.get("ok", 0),
               "predictions_sha256": file_hash(out / "predictions.jsonl"),
               "runtime": child_summary.get("runtime") if child_summary else None,
               "child_summary_sha256": file_hash(summary_path) if child_summary else None,
               "notes": ["Completed child predictions are preserved; all planned IDs remain in the denominator.",
                         "Timeout is a wall-clock interruption, not an OOM or a per-case latency measurement.",
                         "The first unfinished ID may have been in model loading or generation; its exact stage is not assumed.",
                         "Cleanup reports only observed command/process evidence; unsuccessful cleanup is explicit.",
                         "The wrapper never changes driver policies or deletes child run evidence."]}
    if plan["kind"] == "nll":
        successful = [row for row in rows if row["status"] == "ok"]
        tokens = sum(row["loss_tokens"] for row in successful)
        total_nll = sum(row["nll_sum"] for row in successful)
        summary["nll"] = {"complete": len(successful) == len(rows),
            "planned_loss_tokens": sum(row["loss_tokens"] for row in requests), "scored_loss_tokens": tokens,
            "mean_nll": total_nll / tokens if tokens and len(successful) == len(rows) else None,
            "conditional_mean_nll_successful_documents": total_nll / tokens if tokens else None}
    _atomic_json(out / "summary.json", summary)
    _atomic_json(out / "run_manifest.json", {**initial, "status": status,
        "worker_pid": proc.pid if proc else None, "summary_sha256": file_hash(out / "summary.json"),
        "child_completed_records": len(completed), "cleanup": cleanup})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_bounded(args.plan, args.output, max_seconds=args.max_seconds, execute=args.execute)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Bounded Gemma evaluation: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("executed") and result.get("status") != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
