"""Resource inventory and bounded monitoring of the guarded training launcher.

Board-level nvidia-smi samples include other applications and are not allocator
peaks. No Torch import is required for inventory or planning. Never executes an
arbitrary command read from a plan, changes GPU settings, or shuts down a host.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from .core import file_hash, new_output, write_json

GPU_FIELDS = ("index", "uuid", "name", "driver_version", "memory.total",
              "memory.used", "memory.free", "utilization.gpu", "power.draw")
PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "peft",
            "bitsandbytes", "ms-swift", "datasets", "trl", "tokenizers")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_gpu_csv(value: str) -> list[dict]:
    """Keep unsupported counters null; zero would falsely claim no usage."""
    result = []
    for row in csv.reader(io.StringIO(value), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(GPU_FIELDS):
            raise ValueError("unexpected nvidia-smi column count")
        item = {}
        for key, raw in zip(GPU_FIELDS, row):
            raw = raw.strip()
            if key in {"uuid", "name", "driver_version"}:
                item[key] = raw
            else:
                try:
                    number = float(raw)
                    item[key] = number if math.isfinite(number) else None
                except ValueError:
                    item[key] = None
        result.append(item)
    return result


def gpu_sample(*, timeout=10) -> dict:
    stamp = utc_now()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode:
            return {"time_utc": stamp, "gpus": [], "error": "nvidia_smi_failed",
                    "returncode": result.returncode}
        return {"time_utc": stamp, "gpus": parse_gpu_csv(result.stdout), "error": None}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"time_utc": stamp, "gpus": [], "error": type(exc).__name__}


def inventory() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": "resource-inventory-v1", "time_utc": utc_now(),
        "platform": platform.system(), "python": platform.python_version(),
        "interpreter": sys.executable, "packages": versions,
        "gpu_sample": gpu_sample(), "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "training_tested": False, "inference_tested": False,
        "note": "Inventory only. Installed versions do not prove CUDA kernels or model compatibility.",
    }


def summarize_samples(samples: list[dict]) -> dict:
    boards = {}
    for sample in samples:
        for gpu in sample["gpus"]:
            key = gpu["uuid"]
            board = boards.setdefault(key, {"name": gpu["name"], "samples": 0,
                                            "sampled_peak_board_used_mib": None,
                                            "minimum_board_free_mib": None})
            board["samples"] += 1
            for field, output, function in (("memory.used", "sampled_peak_board_used_mib", max),
                                             ("memory.free", "minimum_board_free_mib", min)):
                value = gpu[field]
                if value is not None:
                    previous = board[output]
                    board[output] = value if previous is None else function(previous, value)
    return {"boards": boards, "samples": len(samples),
            "failed_samples": sum(bool(s["error"]) for s in samples),
            "scope": "whole GPU board, includes other processes; discrete samples can miss peaks",
            "training_peak_allocated_bytes": None,
            "effective_loss_tokens_per_second": None}


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _stop_owned_process(proc, *, grace_seconds=10):
    # Training is Linux-only. Each child is launched into its own session, so
    # timeout cleanup cannot signal an unrelated user process group.
    pgid = proc.pid
    forced = False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        proc.poll()
        return {"forced": False, "group_alive_after_cleanup": False}
    deadline = time.monotonic() + grace_seconds
    # Reap the direct child, but keep watching its owned group: a trainer can
    # outlive the wrapper or ignore SIGTERM. Waiting only on proc misses that.
    while time.monotonic() < deadline:
        proc.poll()
        if not _group_exists(pgid):
            break
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            forced = True
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    return {"forced": forced, "group_alive_after_cleanup": _group_exists(pgid)}


def profile_training(plan_path, output, *, execute=False, max_seconds=900, interval=2.0):
    if type(max_seconds) is not int or not 1 <= max_seconds <= 604800:
        raise ValueError("max_seconds must be an integer in [1,604800]")
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or not 0.25 <= interval <= 60:
        raise ValueError("interval must be finite and in [0.25,60]")
    from .launch import launch
    plan = Path(plan_path).resolve()
    checked = launch(plan, execute=False)
    out = new_output(output)
    manifest = {"schema_version": "training-resource-profile-v1", "created_utc": utc_now(),
                "plan": str(plan), "plan_sha256": file_hash(plan),
                "executed": False, "status": "planned", "max_seconds": max_seconds,
                "sample_interval_seconds": interval, "preflight": checked,
                "inventory": inventory(), "training_completed": False,
                "checkpoint_reload_verified": False}
    report = out / "resource_summary.json"
    write_json(report, manifest)
    if not execute:
        return manifest
    if checked.get("training_ready") is not True or checked.get("blockers"):
        manifest["status"] = "blocked_by_data_or_plan"
        write_json(report, manifest)
        return manifest
    if platform.system() != "Linux":
        manifest["status"] = "blocked_requires_linux_training_environment"
        write_json(report, manifest)
        return manifest
    # The model launcher validates the plan again immediately before execution.
    # Reconstruct the command here; never execute saved argv or shell strings.
    command = [sys.executable, "-m", "cpt", "run", "--plan", str(plan), "--execute"]
    samples, proc = [], None
    started = time.monotonic()
    manifest.update(status="starting", command=command)
    write_json(report, manifest)
    try:
        with (out / "training.stdout.log").open("wb") as stdout, \
             (out / "training.stderr.log").open("wb") as stderr, \
             (out / "gpu_samples.jsonl").open("w", encoding="utf-8") as sample_file:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=False,
                                    start_new_session=True)
            manifest.update(status="running", executed=True)
            write_json(report, manifest)
            while True:
                remaining = max(0.1, max_seconds - (time.monotonic() - started))
                sample = gpu_sample(timeout=min(10, remaining))
                samples.append(sample)
                sample_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sample_file.flush()
                code = proc.poll()
                if code is not None:
                    manifest.update(status="process_exited" if code == 0 else "process_failed", returncode=code)
                    break
                if time.monotonic() - started >= max_seconds:
                    manifest["status"] = "timeout"
                    manifest["cleanup"] = _stop_owned_process(proc)
                    manifest["returncode"] = proc.returncode
                    break
                time.sleep(min(interval, max(0.0, max_seconds - (time.monotonic() - started))))
    except BaseException as exc:
        manifest.update(status="interrupted_or_failed", error_type=type(exc).__name__)
        if proc is not None:
            manifest["cleanup"] = _stop_owned_process(proc)
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic() - started,
                        resource_measurement=summarize_samples(samples), finished_utc=utc_now())
        manifest["note"] = ("A zero exit code is not validated model quality or checkpoint recovery. "
                            "Counters are sampled board memory, not Torch allocator peaks. "
                            "Effective loss-token throughput stays null until measured by the trainer. "
                            "The runtime limit excludes up to 20 seconds of termination/reaping grace.")
        write_json(report, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="read-only driver/package/disk inventory")
    inspect.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="monitor a validated training plan; planning by default")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--max-seconds", type=int, default=900)
    train.add_argument("--interval", type=float, default=2.0)
    train.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inventory()
            write_json(new_output(args.output) / "resource_summary.json", result)
        else:
            result = profile_training(args.plan, args.output, execute=args.execute,
                                      max_seconds=args.max_seconds, interval=args.interval)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Resource profile: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") in {"blocked_by_data_or_plan", "blocked_requires_linux_training_environment",
                                        "process_failed", "timeout", "interrupted_or_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
