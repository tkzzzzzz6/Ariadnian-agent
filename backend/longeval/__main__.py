"""Run with ``python -m longeval``; standard library only by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .adapters import import_benchmark
from .core import build_diagnostics, evaluate, validate


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number}: record must be a JSON object")
        records.append(record)
    return records


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _format(value) -> str:
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown(summary: dict) -> str:
    columns = ("n", "prediction_coverage", "evidence_recall", "evidence_precision", "evidence_complete_hit", "answer_em", "answer_f1", "answer_coverage", "unanswerable_rejection_accuracy")
    lines = ["# Long-document evaluation", "", f"Mode: `{summary['mode']}`. Length unit: `{summary['length_unit']}`.", ""]
    if summary.get("tokenizer"):
        lines.extend([f"Reference tokenizer: `{summary['tokenizer']['path']}`. Its counts are not the target model's native input tokens.", ""])
    if summary.get("live"):
        lines.extend([f"Live model: `{summary['live']['model']}`. Prompt: `{summary['live']['prompt_version']}`. Thinking setting: `{summary['live'].get('thinking', 'default')}`.", "", "Provider-reported usage (unknown values remain distinct from measured zeros):", "", "```json", json.dumps(summary["live"]["usage"], ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["Metrics use [0, 1]. Missing/failed predictions count as evidence and eligible answer failures; answer coverage remains explicit.", "", "| Slice | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)])
    def add(name, data):
        safe_name = str(name).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + safe_name + " | " + " | ".join(_format(data.get(c)) for c in columns) + " |")
    add("Overall (mixed labels; inspect slices)", summary["overall"])
    for group, values in summary["groups"].items():
        for name, data in values.items():
            add(f"{group}: {name}", data)
    lines.extend(["", "## Interpretation", ""] + ["- " + x for x in summary["limitations"]])
    if summary["warnings"]:
        lines.extend(["", "## Dataset warnings", ""] + ["- " + x for x in summary["warnings"]])
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Provenance-aware offline long-document evaluation (no model calls)")
    commands = p.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Validate contracts, quotes, family splits and leakage")
    check.add_argument("--documents", required=True)
    check.add_argument("--tasks", required=True)
    check.add_argument("--output-dir")
    build = commands.add_parser("build-diagnostics", help="Generate explicitly controlled location diagnostics")
    build.add_argument("--documents", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--max-tasks", type=int, default=1000)
    build.add_argument("--seed", type=int, default=13, help="Deterministic stratified round-robin sampling seed")
    run = commands.add_parser("run", help="Evaluate context selection or score imported predictions")
    run.add_argument("--documents", required=True)
    run.add_argument("--tasks", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--strategy", choices=("bm25", "structure", "truncate"), default="bm25")
    run.add_argument("--budget-bytes", type=int, default=16384)
    run.add_argument("--budget-tokens", type=int, help="Context text token budget; requires a local tokenizer")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--predictions", help="JSONL: id, answer, evidence, predicted_answerable/abstain, status")
    mode.add_argument("--model", help="Explicit opt-in live model, e.g. deepseek/deepseek-chat; uses configured credentials")
    mode.add_argument("--use-configured-model", action="store_true", help="Explicit opt-in live model from app settings")
    run.add_argument("--tier", choices=("write", "judge", "agent"), default="write")
    run.add_argument("--max-output-tokens", type=int, default=2048)
    run.add_argument("--timeout", type=float, default=60)
    run.add_argument("--thinking", choices=("default", "disabled", "enabled"), default="default", help="Explicit provider thinking mode; default preserves provider behavior")
    run.add_argument("--limit", type=int, help="Maximum cases; required explicitly for a live run")
    run.add_argument("--split", choices=("train", "dev", "test"))
    run.add_argument("--tokenizer", "--tokenizer-path", dest="tokenizer_path", help="Local tokenizer.json or Transformers directory; never downloads")
    imp = commands.add_parser("import-benchmark", help="Import local RULER/LongBench v2 records, not an official run")
    imp.add_argument("--benchmark", choices=("ruler", "longbench-v2"), required=True)
    imp.add_argument("--input", required=True)
    imp.add_argument("--documents-out", required=True)
    imp.add_argument("--tasks-out", required=True)
    imp.add_argument("--license", required=True, dest="license_name")
    imp.add_argument("--source-url", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    implementation_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    try:
        if args.command == "import-benchmark":
            source = Path(args.input)
            if source.suffix.lower() == ".json":
                records = json.loads(source.read_text(encoding="utf-8-sig"))
                if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                    raise ValueError("benchmark JSON must contain a list of record objects")
            else:
                records = read_jsonl(source)
            docs, tasks = import_benchmark(records, args.benchmark, license_name=args.license_name, source_url=args.source_url)
            result = validate(docs, tasks)
            if not result["valid"]:
                raise ValueError("; ".join(result["errors"]))
            write_jsonl(args.documents_out, docs)
            write_jsonl(args.tasks_out, tasks)
            print(f"Imported {len(tasks)} {args.benchmark} samples. This is not an official full benchmark result.")
            return 0
        documents = read_jsonl(args.documents)
        if args.command == "build-diagnostics":
            if args.max_tasks < 1:
                raise ValueError("max-tasks must be positive")
            result = validate(documents, [])
            if not result["valid"]:
                raise ValueError("; ".join(result["errors"]))
            tasks = build_diagnostics(documents, args.max_tasks, args.seed)
            write_jsonl(args.output, tasks)
            print(f"Wrote {len(tasks)} controlled synthetic location diagnostics; no human/reasoning gold claimed.")
            return 0
        tasks = read_jsonl(args.tasks)
        if args.command == "validate":
            result = validate(documents, tasks)
            if args.output_dir:
                write_json(Path(args.output_dir) / "validation.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["valid"] else 2
        full_validation = validate(documents, tasks)
        if not full_validation["valid"]:
            raise ValueError("; ".join(full_validation["errors"]))
        all_task_ids = {t.get("id") for t in tasks}
        if args.split:
            tasks = [t for t in tasks if t.get("split") == args.split]
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("limit must be positive")
            tasks = tasks[:args.limit]
        if not tasks:
            raise ValueError("no tasks selected")
        predictions = read_jsonl(args.predictions) if args.predictions else None
        if predictions is not None:
            unknown = {p.get("id") for p in predictions} - all_task_ids
            if unknown:
                raise ValueError("unknown prediction IDs: " + ", ".join(str(x) for x in unknown))
            selected_ids = {t["id"] for t in tasks}
            predictions = [p for p in predictions if p.get("id") in selected_ids]
        counter, unit, tokenizer_metadata = None, "utf8_bytes", None
        if args.tokenizer_path:
            from .tokenizer import load_tokenizer
            counter, tokenizer_metadata = load_tokenizer(args.tokenizer_path)
            unit = "reference_tokenizer_tokens:" + str(Path(args.tokenizer_path).resolve())
        if args.budget_tokens is not None and (counter is None or args.budget_tokens < 1):
            raise ValueError("--budget-tokens requires a positive value and --tokenizer with a local tokenizer")
        live_metadata = None
        if args.model or args.use_configured_model:
            if args.limit is None:
                raise ValueError("Live model runs require explicit --limit to bound the number of calls")
            from .live import run_live
            predictions, live_metadata = run_live(documents, tasks, model=args.model, tier=args.tier, strategy=args.strategy, budget_bytes=args.budget_bytes, budget_tokens=args.budget_tokens, length_counter=counter, max_output_tokens=args.max_output_tokens, timeout=args.timeout, thinking=args.thinking, progress=lambda event: print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True))
        rows, summary = evaluate(documents, tasks, strategy=args.strategy, budget_bytes=args.budget_bytes, budget_tokens=args.budget_tokens, predictions=predictions, length_counter=counter, length_unit=unit)
        summary["tokenizer"] = tokenizer_metadata
        if tokenizer_metadata:
            summary["limitations"].append("Local/reference tokenizer counts are not the target model's actual input token usage; provider-reported usage is separate.")
        if live_metadata:
            summary.update(mode="live_model", live=live_metadata, strategy=args.strategy, budget_tokens=args.budget_tokens, budget_bytes=args.budget_bytes if args.budget_tokens is None else None)
            summary["limitations"].append("Model answers and diagnostic location scores are inference results; no CPT or weight update was performed.")
        summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["inputs"] = {"documents": str(Path(args.documents).resolve()), "tasks": str(Path(args.tasks).resolve()), "predictions": str(Path(args.predictions).resolve()) if args.predictions else None}
        out = Path(args.output_dir)
        summary["input_sha256"] = {"documents": hashlib.sha256(Path(args.documents).read_bytes()).hexdigest(), "tasks": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(), "selected_task_ids": hashlib.sha256(json.dumps([t["id"] for t in tasks], ensure_ascii=False).encode("utf-8")).hexdigest()}
        if args.predictions:
            summary["input_sha256"]["predictions"] = hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest()
        summary["implementation_sha256"] = implementation_hashes
        summary["full_input_validation"] = {key: full_validation[key] for key in ("valid", "documents", "tasks")}
        if live_metadata:
            # Score raw answer/quotes first. Then recursively redact every outgoing
            # string using the secret set captured inside the live app context.
            artifacts = live_metadata.redact_for_storage({"rows": rows, "summary": summary, "model_predictions": predictions})
            rows, summary, predictions = artifacts["rows"], artifacts["summary"], artifacts["model_predictions"]
            summary["storage_redaction"] = {"applied": True, "scope": "all prediction and metadata strings after scoring", "scored_before_redaction": True}
        if live_metadata:
            write_jsonl(out / "model-predictions.jsonl", predictions)
        write_jsonl(out / "predictions.jsonl", rows)
        write_json(out / "summary.json", summary)
        (out / "report.md").write_text(markdown(summary), encoding="utf-8")
        print(f"Evaluated {len(rows)} tasks ({summary['mode']}); results: {out.resolve()}")
        print(f"Evidence recall={_format(summary['overall']['evidence_recall'])}; answer coverage={_format(summary['overall']['answer_coverage'])}")
        return 1 if summary["overall"]["failed_or_missing"] else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"longeval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
