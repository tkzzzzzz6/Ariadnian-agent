"""Fetch, independently evaluate, or rescore pinned LongBench subsets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from longeval.__main__ import read_jsonl, write_json, write_jsonl
from longeval.tokenizer import load_tokenizer

from . import runner


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch source-fixed official prompts/data and deterministically sample")
    fetch.add_argument("--output", required=True, type=Path)
    fetch.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / ".cache")
    fetch.add_argument("--benchmark", choices=("v1", "v2", "both"), default="both")
    fetch.add_argument("--tasks", help="Comma-separated LongBench v1 tasks")
    fetch.add_argument("--per-task", type=int, default=5)
    fetch.add_argument("--v2-per-domain", type=int, default=2)
    fetch.add_argument("--seed", type=int, default=13)
    fetch.add_argument("--revision", help="Explicit dataset revision (use one benchmark at a time)")
    fetch.add_argument("--max-context-tokens", type=int, help="Explicit pre-sampling reference-token length filter; changes subset protocol")
    fetch.add_argument("--tokenizer", help="Local reference tokenizer for the optional pre-sampling length filter")
    run = sub.add_parser("run", help="Explicit bounded model run using official pure-text task prompts")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--use-configured-model", action="store_true")
    mode.add_argument("--model")
    run.add_argument("--limit", required=True, type=int, help="Maximum actual examples, including skipped/failed ones")
    run.add_argument("--tier", choices=("write", "judge", "agent"), default="write")
    run.add_argument("--thinking", choices=("default", "disabled", "enabled"), default="default")
    run.add_argument("--strategy", choices=("full", "bm25", "structure", "truncate"), default="full")
    run.add_argument("--context-tokens", type=int, default=65536)
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--max-output-tokens", type=int, help="Optional lower cap; reduced official caps label a modified protocol")
    run.add_argument("--timeout", type=float, default=60)
    run.add_argument("--chunk-chars", type=int, default=2048)
    rescore = sub.add_parser("rescore", help="Score previously saved plain-text answers without model requests")
    rescore.add_argument("--input", required=True, type=Path)
    rescore.add_argument("--predictions", required=True, type=Path)
    rescore.add_argument("--output", required=True, type=Path)
    rescore.add_argument("--limit", type=int, help="Score the same fixed prefix as a bounded original run")
    return p


def _empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("output directory must be empty; do not overwrite an existing evaluation")
    path.mkdir(parents=True, exist_ok=True)


def _checkpoint(output: Path, row: dict | None, state: dict) -> None:
    if row is not None:
        with (output / "predictions.partial.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    temporary = output / "inprogress_manifest.tmp"
    write_json(temporary, {**state, "updated_utc": datetime.now(timezone.utc).isoformat()})
    temporary.replace(output / "inprogress_manifest.json")


def _finish_inprogress(args, status: str) -> None:
    if args.command != "run" or not getattr(args, "_owns_output", False):
        return
    path = args.output / "inprogress_manifest.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        temporary = args.output / "inprogress_manifest.tmp"
        write_json(temporary, {**state, "status": status, "updated_utc": datetime.now(timezone.utc).isoformat()})
        temporary.replace(path)


def _verify_sampling_manifest(manifest: dict, examples: list[dict], file_sha256: str) -> None:
    if not isinstance(manifest, dict) or manifest.get("selected_records_sha256") != file_sha256:
        raise ValueError("fetch manifest does not match selected_records input")
    if type(manifest.get("records")) is not int or manifest["records"] != len(examples):
        raise ValueError("fetch manifest record count does not match input")
    ids = manifest.get("selected_ids")
    sources = manifest.get("source_manifests")
    source_ids = None
    if sources is not None:
        if not isinstance(sources, list) or not sources or any(not isinstance(source, dict) or not isinstance(source.get("selected_ids"), list) for source in sources):
            raise ValueError("fetch manifest source selection contract is invalid")
        source_ids = [did for source in sources for did in source["selected_ids"]]
        if any(source.get("selected_examples") != len(source["selected_ids"]) for source in sources):
            raise ValueError("source manifest selected count/IDs mismatch")
    if ids is None:
        ids = source_ids
    if not isinstance(ids, list) or any(not isinstance(did, str) for did in ids) or ids != [example["id"] for example in examples]:
        raise ValueError("fetch manifest selected IDs/order do not match input")
    if source_ids is not None and source_ids != ids:
        raise ValueError("fetch manifest top-level/source selected IDs disagree")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch":
            from .data import fetch_benchmark
            if args.per_task < 1 or args.v2_per_domain < 1:
                raise ValueError("per-task and v2-per-domain must be positive")
            if args.revision and args.benchmark == "both":
                raise ValueError("--revision applies to one benchmark; fetch v1/v2 separately")
            counter, tokenizer_metadata = None, None
            if args.tokenizer:
                counter, tokenizer_metadata = load_tokenizer(args.tokenizer)
            if args.max_context_tokens is not None and (args.max_context_tokens < 1 or counter is None):
                raise ValueError("positive --max-context-tokens requires --tokenizer")
            examples, manifests = [], []
            for benchmark in (["v1", "v2"] if args.benchmark == "both" else [args.benchmark]):
                selected, manifest = fetch_benchmark(benchmark, args.cache_dir, tasks=(args.tasks.split(",") if args.tasks else runner.DEFAULT_TASKS) if benchmark == "v1" else None, n_per_task=args.per_task if benchmark == "v1" else args.v2_per_domain, seed=args.seed, revision=args.revision, v2_stratify=True, max_context_tokens=args.max_context_tokens, length_counter=counter, tokenizer_metadata=tokenizer_metadata)
                examples.extend(selected)
                manifests.append(manifest)
            runner.validate_records(examples)
            _empty_output(args.output)
            path = args.output / "selected_records.jsonl"
            write_jsonl(path, examples)
            manifest = {"protocol_version": runner.VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "records": len(examples), "selected_ids": [example["id"] for example in examples], "selected_records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_manifests": manifests, "official_full_dataset": False, "length_filter_requested": args.max_context_tokens, "reference_tokenizer": tokenizer_metadata}
            write_json(args.output / "fetch_manifest.json", manifest)
            print(f"Selected {len(examples)} public benchmark records: {path.resolve()}")
            return 0
        all_examples = read_jsonl(args.input)
        runner.validate_records(all_examples)
        input_file_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
        sampling_manifest = args.input.parent / "fetch_manifest.json"
        observed_sampling, sampling_sha256 = None, None
        if sampling_manifest.exists():
            observed_sampling = json.loads(sampling_manifest.read_text(encoding="utf-8"))
            _verify_sampling_manifest(observed_sampling, all_examples, input_file_sha256)
            sampling_sha256 = hashlib.sha256(sampling_manifest.read_bytes()).hexdigest()
        elif args.command == "run":
            raise ValueError("live public benchmark run requires an adjacent fetch_manifest.json with verified sample provenance")
        if args.command == "rescore":
            if args.limit is not None and args.limit < 1:
                raise ValueError("limit must be positive")
            examples = all_examples if args.limit is None else all_examples[:args.limit]
            selected_ids = {example["id"] for example in examples}
            all_predictions = read_jsonl(args.predictions)
            if any(row.get("id") not in {e["id"] for e in all_examples} for row in all_predictions):
                raise ValueError("unknown prediction IDs")
            rows, summary = runner.rescore_records(examples, [row for row in all_predictions if row.get("id") in selected_ids])
            summary["prediction_file_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
            summary["provenance_status"] = "verified_fetch_manifest" if observed_sampling is not None else "unverified_input_without_fetch_manifest"
            summary["official_subset_provenance_verified"] = observed_sampling is not None
            if observed_sampling is None:
                summary["limitations"] = ["No matching fetch manifest is available: this rescore has unverified input provenance and cannot be promoted to a verified official-prompt subset result."]
                for row in rows:
                    row["source_run_protocol"] = row.get("protocol")
                    row["protocol"] = "unverified_provenance_rescore"
        else:
            if args.limit < 1:
                raise ValueError("limit must be positive")
            examples = all_examples[:args.limit]
            counter, tokenizer_metadata = load_tokenizer(args.tokenizer)
            _empty_output(args.output)
            args._owns_output = True
            write_json(args.output / "inprogress_manifest.json", {"protocol_version": runner.VERSION, "status": "starting", "completed": 0, "total": len(examples), "input_file_sha256": input_file_sha256, "strategy": args.strategy, "thinking": args.thinking, "started_utc": datetime.now(timezone.utc).isoformat()})
            rows, summary = runner.run_records(examples, model=args.model, tier=args.tier, strategy=args.strategy, context_tokens=args.context_tokens, length_counter=counter, tokenizer_metadata=tokenizer_metadata, thinking=args.thinking, max_output_tokens=args.max_output_tokens, timeout=args.timeout, chunk_chars=args.chunk_chars, progress=lambda event: print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True), checkpoint=lambda row, state: _checkpoint(args.output, row, {**state, "input_file_sha256": input_file_sha256}))
            summary["provenance_status"] = "verified_fetch_manifest"
            summary["official_subset_provenance_verified"] = True
        summary.update(created_utc=datetime.now(timezone.utc).isoformat(), input_file_sha256=input_file_sha256, input_records_total=len(all_examples), selected_records=len(examples))
        if observed_sampling is not None:
            summary["sampling_manifest"] = observed_sampling
            summary["sampling_manifest_sha256"] = sampling_sha256
        if hasattr(summary, "redact_for_storage"):
            storage = summary.redact_for_storage({"rows": rows, "summary": summary, "examples": examples})
            rows, summary, examples = storage["rows"], storage["summary"], storage["examples"]
            summary["storage_redaction"] = {"applied_after_scoring": True, "scope": "all outgoing prediction/metadata/example strings"}
        if args.command == "rescore":
            _empty_output(args.output)
        write_jsonl(args.output / "predictions.jsonl", rows)
        write_jsonl(args.output / "selected_records.jsonl", examples)
        write_json(args.output / "summary.json", summary)
        (args.output / "report.md").write_text(runner.markdown(summary), encoding="utf-8")
        _finish_inprogress(args, "complete" if all(row["status"] == "ok" for row in rows) else "complete_with_failed_or_skipped_cases")
        print(f"Scored {len(rows)} public benchmark examples; {args.output.resolve()}")
        return 1 if any(row["status"] != "ok" for row in rows) else 0
    except KeyboardInterrupt:
        _finish_inprogress(args, "interrupted")
        print("publicbench: interrupted; completed cases remain in predictions.partial.jsonl", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        _finish_inprogress(args, "failed")
        print(f"publicbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
