"""Independent public-benchmark inference, with official text prompts and metrics.

This runner never places reference answers in requests and never uses Ariadne's
citation-JSON prompt. Full-context and modified-context protocols remain distinct.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

from longeval.core import select_context
from longeval.live import LiveMetadata, app_context

VERSION = "ariadne-publicbench-v1"
DEFAULT_TASKS = ["qasper", "multifieldqa_zh", "hotpotqa", "gov_report", "trec", "passage_retrieval_en"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def render_prompt(example: dict, context_override: str | None = None) -> str:
    from .data import render_prompt as render
    return render(example, context_override=context_override)


def score_prediction(example: dict, prediction: str, status: str = "ok") -> dict:
    from .metrics import score_prediction as score
    return score(example, prediction, status=status)


def aggregate_scores(rows: list[dict]) -> dict:
    from .metrics import aggregate_scores as aggregate
    return aggregate(rows)


def implementation_hashes() -> dict:
    backend = Path(__file__).resolve().parent.parent
    files = list(Path(__file__).parent.glob("*.py"))
    files += [backend / p for p in ("longeval/live.py", "longeval/core.py", "longeval/tokenizer.py", "app/llm.py", "app/config.py")]
    return {p.relative_to(backend).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def validate_records(examples: list[dict]) -> None:
    if not isinstance(examples, list) or any(not isinstance(e, dict) for e in examples):
        raise ValueError("benchmark input must be a list of record objects")
    seen = set()
    for example in examples:
        for name in ("id", "benchmark", "task", "context", "prompt_template"):
            if not isinstance(example.get(name), str) or not example[name]:
                raise ValueError(f"benchmark record requires nonempty string {name}")
        if example["id"] in seen:
            raise ValueError("duplicate benchmark example ID: " + example["id"])
        seen.add(example["id"])
        answers = example.get("reference_answers", example.get("answers"))
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
            raise ValueError(f"{example['id']}: reference answers must be a nonempty string list")
        cap = example.get("official_max_new_tokens")
        if type(cap) is not int or cap < 1:
            raise ValueError(f"{example['id']}: missing positive official_max_new_tokens")
        if not isinstance(example.get("question", ""), str):
            raise ValueError(f"{example['id']}: question must be a string")
        if "raw_record_sha256" in example:
            raw = example.get("raw_record")
            if not isinstance(raw, dict):
                raise ValueError(f"{example['id']}: source-backed example requires raw_record")
            canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != example["raw_record_sha256"]:
                raise ValueError(f"{example['id']}: raw_record canonical hash mismatch")
            raw_answers = raw.get("answers") if example["benchmark"] == "longbench-v1" else [raw.get("answer")]
            raw_question = raw.get("input") if example["benchmark"] == "longbench-v1" else raw.get("question")
            if example["context"] != raw.get("context") or example["question"] != raw_question or answers != raw_answers:
                raise ValueError(f"{example['id']}: context/question/reference differs from source raw_record")
        if "prompt_sha256" in example or "prompt" in example:
            prompt = example.get("prompt")
            if not isinstance(prompt, str) or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != example.get("prompt_sha256"):
                raise ValueError(f"{example['id']}: stored prompt byte hash mismatch")
            if render_prompt(example) != prompt:
                raise ValueError(f"{example['id']}: current official rendering differs from stored prompt")


def _blocks(context: str, chunk_chars: int) -> list[dict]:
    """Source-aligned paragraphs/windows; heading paths are heuristic metadata."""
    blocks = []
    section = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", context):
        start, end = match.span()
        value = context[start:end]
        first_line = value.splitlines()[0].strip()
        heading = re.match(r"^(?:#{1,6}\s+|(?:Passage|Document|Chapter|Section)\s+\S+[:.]?\s*)(.*)", first_line, re.I)
        if heading:
            section = [first_line[:200]]
        for offset in range(0, len(value), chunk_chars):
            text = value[offset:offset + chunk_chars]
            blocks.append({"id": f"c{len(blocks):06d}", "type": "paragraph", "text": text, "section_path": list(section), "page": None, "source_start_char": start + offset, "source_end_char": start + offset + len(text)})
    return blocks


def prepare_context(example: dict, *, strategy: str, context_tokens: int, length_counter: Callable[[str], int], chunk_chars: int = 2048) -> tuple[str, dict]:
    original = example["context"]
    original_tokens = length_counter(original)
    if strategy == "full":
        return original, {"strategy": "full", "original_context_reference_tokens": original_tokens, "selected_context_reference_tokens": original_tokens, "selected_context_characters": len(original), "modified": False, "truncated": False, "fits_reference_context_limit": original_tokens <= context_tokens, "source_spans": [{"start_char": 0, "end_char": len(original)}], "selection_sha256": digest(original)}
    if strategy not in {"bm25", "structure", "truncate"}:
        raise ValueError("strategy must be full, bm25, structure, or truncate")
    blocks = _blocks(original, chunk_chars)
    if not blocks:
        raise ValueError("context has no usable source text")
    did = example["id"]
    title = str(example.get("title") or example.get("raw_record", {}).get("title") or "")
    document = {"id": did, "title": title, "blocks": blocks}
    query = example.get("question") or example["prompt_template"].replace("{context}", "").replace("$DOC$", "")
    task = {"id": did, "question": query, "document_ids": [did]}
    selected = select_context(task, {did: document}, strategy, 1, budget_tokens=context_tokens, length_counter=length_counter)
    lookup = {b["id"]: b for b in blocks}
    # Rank for selection, then restore source order for the outgoing text.
    evidence = sorted(selected["evidence"], key=lambda e: lookup[e["block_id"]]["source_start_char"])
    text = "\n\n".join(ev["quote"] for ev in evidence)
    # Token merges can change when ranking order is restored. Enforce the final,
    # actually transmitted text budget too, without rescanning the whole corpus.
    if length_counter(text) > context_tokens:
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if length_counter(text[:middle]) <= context_tokens:
                low = middle
            else:
                high = middle - 1
        text = text[:low]
    spans = []
    outgoing_cursor = 0
    for ev in evidence:
        block = lookup[ev["block_id"]]
        consumed = min(len(ev["quote"]), max(0, len(text) - outgoing_cursor))
        if consumed:
            spans.append({"start_char": block["source_start_char"] + ev.get("start_char", 0), "end_char": block["source_start_char"] + ev.get("start_char", 0) + consumed})
        outgoing_cursor += len(ev["quote"]) + 2
    count = length_counter(text)
    if count > context_tokens:
        raise ValueError("selected public-benchmark context exceeded the reference tokenizer budget")
    return text, {"strategy": strategy, "original_context_reference_tokens": original_tokens, "selected_context_reference_tokens": count, "selected_context_characters": len(text), "modified": text != original, "truncated": text != original, "fits_reference_context_limit": True, "source_spans": spans, "selected_source_characters": sum(s["end_char"] - s["start_char"] for s in spans), "source_character_coverage": sum(s["end_char"] - s["start_char"] for s in spans) / max(len(original), 1), "chunk_chars": chunk_chars, "chunking": "blank-line paragraphs split into non-overlapping character windows", "ordering": "ranked selection restored to source order", "structure_weighting": "BM25 plus inferred heading/title lexical weighting; no neighboring-block expansion", "selection_sha256": digest(text)}


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _usage(response, llm) -> dict:
    usage = _field(response, "usage")
    details = _field(usage, "prompt_tokens_details", {})
    completion_details = _field(usage, "completion_tokens_details", {})
    cached = _field(usage, "prompt_cache_hit_tokens")
    if cached is None:
        cached = _field(details, "cached_tokens")
    return {"prompt_tokens": _field(usage, "prompt_tokens"), "completion_tokens": _field(usage, "completion_tokens"), "total_tokens": _field(usage, "total_tokens"), "cached_tokens": cached, "reasoning_tokens": _field(completion_details, "reasoning_tokens"), "cost_usd": llm._cost(response)}


def _base_row(example: dict) -> dict:
    return {"id": example["id"], "official_id": example.get("official_id"), "benchmark": example["benchmark"], "task": example["task"], "domain": example.get("domain", example["task"]), "language": example.get("language", "unknown"), "source_revision": example.get("source_revision"), "source_sha256": example.get("source_sha256") or example.get("source_file_sha256"), "example_sha256": digest(example), "prompt_template_sha256": digest(example["prompt_template"]), "metric_name": example.get("metric_name"), "evaluation_only": True, "status": "not_requested", "prediction": "", "usage": {k: None for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens", "cost_usd")}}


def _score_row(example: dict, row: dict) -> None:
    try:
        metrics = score_prediction(example, row["prediction"], status=row["status"])
        row["metric_result"] = metrics
        if row["status"] in {"ok", "success"} and metrics.get("status", "ok") not in {"ok", "success"}:
            row["status"] = metrics["status"]
        for key in ("score", "score_percent", "metric", "official_answer_extraction"):
            if key in metrics:
                row[key] = metrics[key]
    except Exception as exc:
        row.update(status="scoring_failed", score=0.0, score_percent=0.0, scoring_error_type=type(exc).__name__, scoring_error=str(exc)[:600])


def run_records(examples: list[dict], *, model: str | None = None, tier: str = "write", strategy: str = "full", context_tokens: int = 65536, length_counter: Callable[[str], int], tokenizer_metadata: dict, thinking: str = "default", max_output_tokens: int | None = None, timeout: float = 60, chunk_chars: int = 2048, progress: Callable[[dict], None] | None = None, checkpoint: Callable[[dict | None, dict], None] | None = None) -> tuple[list[dict], LiveMetadata]:
    validate_records(examples)
    if not examples:
        raise ValueError("no benchmark records selected")
    if context_tokens < 1 or timeout <= 0 or chunk_chars < 1 or max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("context/output budgets, timeout and chunk_chars must be positive")
    if thinking not in {"default", "disabled", "enabled"}:
        raise ValueError("thinking must be default, disabled, or enabled")
    if strategy not in {"full", "bm25", "structure", "truncate"}:
        raise ValueError("unknown context strategy")
    hashes = implementation_hashes()
    selected_records_sha256 = digest(examples)
    rows = []
    with app_context(model, tier) as settings:
        from app import llm, tracing
        selected_tier = replace(settings.tiers[tier], timeout=timeout, retries=0, fallbacks=())
        llm._check_ready(selected_tier)
        resolved_model = llm.resolve_model(selected_tier.model)
        for index, example in enumerate(examples):
            row = _base_row(example)
            official_cap = example["official_max_new_tokens"]
            cap = min(official_cap, max_output_tokens) if max_output_tokens is not None else official_cap
            row.update(model=resolved_model, official_max_new_tokens=official_cap, parameters={"temperature": 0, "max_output_tokens": cap, "thinking": thinking, "timeout": timeout, "transport_retries": 0, "fallbacks": []}, transport_called=False)
            row["protocol"] = "official_prompt_full_context_subset" if strategy == "full" and cap == official_cap else "modified_context_or_generation_budget_subset"
            started = time.perf_counter()
            try:
                context, selection = prepare_context(example, strategy=strategy, context_tokens=context_tokens, length_counter=length_counter, chunk_chars=chunk_chars)
                row["context_selection"] = selection
                if not selection["fits_reference_context_limit"]:
                    row.update(status="skipped_context_limit", error="Original context exceeds the declared reference-token limit; no model call and no silent truncation")
                elif not context:
                    row.update(status="skipped_empty_selection", error="Context strategy selected no text")
                else:
                    prompt = render_prompt(example, context_override=context)
                    # Exact source-fixed benchmark prompt, one user message, no
                    # citation JSON/system prompt or hidden reference answers.
                    messages = [{"role": "user", "content": prompt}]
                    row.update(prompt_sha256=digest(prompt), messages_sha256=digest(messages), serialized_prompt_reference_tokens=length_counter(prompt), prompt_wrapper="one user message; no added system instructions")
                    kwargs = llm._kwargs(selected_tier, resolved_model) | {"messages": messages, "temperature": 0, "max_tokens": cap}
                    if thinking != "default":
                        kwargs["extra_body"] = {"thinking": {"type": thinking}}
                    row["transport_called"] = True
                    if checkpoint:
                        snapshot = LiveMetadata({}, secrets=tracing._secrets())
                        state = snapshot.redact_for_storage({"protocol_version": VERSION, "status": "waiting_for_model", "model": resolved_model, "strategy": strategy, "thinking": thinking, "completed": len(rows), "total": len(examples), "active_id": example["id"], "active_request_usage_unknown": True, "transport_calls_attempted": sum(item["transport_called"] for item in rows) + 1, "usage": usage_summary(rows), "selected_records_sha256": selected_records_sha256})
                        checkpoint(None, state)
                    response = llm._completion(kwargs, tier, "公开长上下文基准", operation=f"publicbench-{example['id']}", attempt=1)
                    row["usage"] = _usage(response, llm)
                    choices = _field(response, "choices", [])
                    if not choices:
                        raise ValueError("provider returned no completion choices")
                    row["finish_reason"] = _field(choices[0], "finish_reason")
                    row["response_model"] = _field(response, "model", resolved_model)
                    prediction = _field(_field(choices[0], "message"), "content", "")
                    if not isinstance(prediction, str) or not prediction.strip():
                        raise ValueError("empty model answer; a length-limited empty output is a failure")
                    row.update(status="ok", prediction=prediction)
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=tracing.scrub(str(exc), 600))
            row["latency_ms"] = round((time.perf_counter() - started) * 1000)
            _score_row(example, row)
            rows.append(row)
            if checkpoint:
                snapshot = LiveMetadata({}, secrets=tracing._secrets())
                safe = snapshot.redact_for_storage({"row": row, "state": {"protocol_version": VERSION, "mode": "live_public_benchmark", "status": "in_progress", "model": resolved_model, "thinking": thinking, "strategy": strategy, "completed": len(rows), "total": len(examples), "active_id": None, "active_request_usage_unknown": False, "transport_calls_attempted": sum(item["transport_called"] for item in rows), "usage": usage_summary(rows), "last_id": row["id"], "selected_records_sha256": selected_records_sha256}})
                checkpoint(safe["row"], safe["state"])
            if progress:
                progress({"completed": index + 1, "total": len(examples), "benchmark": example["benchmark"], "task": example["task"], "id": example["id"], "status": row["status"]})
        summary = LiveMetadata({"protocol_version": VERSION, "mode": "live_public_benchmark", "model": resolved_model, "tier": tier, "thinking": thinking, "strategy": strategy, "context_tokens": context_tokens, "tokenizer": tokenizer_metadata, "selected_ids": [e["id"] for e in examples], "selected_records_sha256": selected_records_sha256, "implementation_sha256": hashes, "protocols": dict(Counter(row["protocol"] for row in rows)), "metrics": aggregate_scores(rows), "usage": usage_summary(rows), "statuses": dict(Counter(row["status"] for row in rows)), "transport_calls_attempted": sum(row["transport_called"] for row in rows), "official_full_dataset_run": False, "limits": {"max_output_tokens_override": max_output_tokens, "timeout": timeout, "one_transport_call_per_case": True, "transport_retries": 0, "fallbacks": []}, "limitations": ["A source-fixed official prompt and metric on a sampled subset is not a full official leaderboard result.", "Full-context cases exceeding the declared reference-token guard remain skipped in the denominator; original text is never silently truncated.", "BM25/structure/truncate replace the original context and are a modified budget protocol, reported separately from full context.", "Reference-tokenizer counts are not this API model's native tokens; provider usage is recorded separately.", "The reference context budget excludes question/instruction/chat-template overhead; provider window errors are failures, not hidden truncations.", "Benchmark references are never added to the request; official few-shot examples already in source context remain intact.", "No evidence-span gold is assumed and no citation-JSON output is required.", "No training or weight updates were performed."]}, secrets=tracing._secrets())
    summary["limitations"].append("Temperature is fixed at 0 here; the pinned upstream v2 non-CoT script uses 0.1. Official task prompts/max lengths/metrics do not imply identical upstream generation settings.")
    return rows, summary


def usage_summary(rows: list[dict]) -> dict:
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens", "cost_usd"):
        values = [row.get("usage", {}).get(key) for row in rows]
        known = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        result[key] = sum(known) if known else None
        result[key + "_known_calls"] = len(known)
    return result


def rescore_records(examples: list[dict], predictions: list[dict]) -> tuple[list[dict], dict]:
    validate_records(examples)
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in predictions):
        raise ValueError("predictions require string IDs")
    ids = [row["id"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prediction IDs")
    unknown = set(ids) - {e["id"] for e in examples}
    if unknown:
        raise ValueError("unknown prediction IDs: " + ", ".join(sorted(unknown)))
    by_id = {row["id"]: row for row in predictions}
    rows = []
    for example in examples:
        stored = by_id.get(example["id"], {})
        base = _base_row(example)
        row = {**base, **stored}
        if example["id"] not in by_id:
            row.update(status="missing_prediction", prediction="")
        if row.get("example_sha256") and row["example_sha256"] != digest(example):
            raise ValueError("prediction was produced for a different benchmark example: " + example["id"])
        for key in ("id", "official_id", "benchmark", "task", "domain", "language", "source_revision", "source_sha256", "metric_name"):
            row[key] = base[key]
        _score_row(example, row)
        rows.append(row)
    return rows, {"protocol_version": VERSION, "mode": "rescore_without_api", "metrics": aggregate_scores(rows), "statuses": dict(Counter(row["status"] for row in rows)), "selected_ids": [e["id"] for e in examples], "selected_records_sha256": digest(examples), "implementation_sha256": implementation_hashes(), "transport_calls_attempted": 0, "official_full_dataset_run": False}


def markdown(summary: dict) -> str:
    metrics = summary["metrics"]
    lines = ["# Public long-context benchmark subset", "", f"Mode: `{summary['mode']}`. Full official dataset run: **no**.", ""]
    if summary.get("model"):
        lines += [f"Model: `{summary['model']}`. Strategy: `{summary['strategy']}`. Thinking: `{summary['thinking']}`.", ""]
    sampling = summary.get("sampling_manifest", {})
    if sampling.get("length_filter_requested") is not None:
        lines += [f"Sampling uses seed `{sampling.get('seed')}` and a declared pre-sampling reference-token context filter of `{sampling['length_filter_requested']}`. This is a length-filtered subset, not a uniform sample of the entire benchmark.", ""]
    lines += ["Different benchmark protocols/task metrics are shown separately; no combined v1+v2 score is claimed.", "", "| Slice | N | Score (%) | Coverage |", "|---|---:|---:|---:|"]
    for group in ("by_benchmark", "by_task", "by_domain"):
        for key, values in metrics.get(group, {}).items():
            name = f"{group}: {key}".replace("|", "\\|")
            score = values.get("score_percent")
            if score is None and values.get("mean_score") is not None:
                score = 100 * values["mean_score"]
            score_text = "—" if score is None else f"{score:.2f}"
            coverage = values.get("coverage", 0)
            lines.append(f"| {name} | {values.get('n', 0)} | {score_text} | {coverage:.3f} |")
    lines += ["", "Statuses: `" + json.dumps(summary.get("statuses", {}), ensure_ascii=False) + "`.", ""]
    if summary.get("usage"):
        lines += ["Provider-reported usage (null means unknown):", "", "```json", json.dumps(summary["usage"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Interpretation", ""] + ["- " + value for value in summary.get("limitations", [])]
    return "\n".join(lines) + "\n"
