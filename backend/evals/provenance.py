"""可复现测评的版本、输入散列与脱敏配置；绝不序列化原始设置文件。"""
from __future__ import annotations

import hashlib
import inspect
import json
import platform
import re
import subprocess
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROTOCOL_VERSION = "ariadne-eval-v2"
BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
_case_id: ContextVar[str | None] = ContextVar("eval_case_id", default=None)


def digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(kind: str, *values: str) -> str:
    return f"{kind}-{digest(list(values))[:24]}"


def file_record(path: Path, *, name: str | None = None, origin: str | None = None) -> dict:
    data = path.read_bytes()
    if name is None:
        try:
            name = path.resolve().relative_to(REPO).as_posix()
        except ValueError:
            name = path.name
    return {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            **({"origin": origin} if origin else {})}


def safe_text(value: Any, limit: int = 600) -> str:
    # Reuse the application's configured-secret and credential-pattern scrubbing.
    from app.tracing import scrub

    cleaned = scrub(value, max(limit * 2, 1200))
    cleaned = re.sub(r"([?&][\w.-]+=)[^&\s\"']+", r"\1[已脱敏]", cleaned)
    return cleaned[:limit]


@contextmanager
def evaluation_case(case_id: str):
    token = _case_id.set(case_id)
    try:
        yield
    finally:
        _case_id.reset(token)


@contextmanager
def audit_requests():
    """Hash actual LLM requests and reject malformed judge verdicts before app coercion.

    The application normalizes missing/invalid verdicts to partial. Evaluation
    validates the raw parsed object, otherwise an invalid response could score as
    a correct negative. Only the evaluation process is instrumented.
    """
    from app import assist, llm

    original_chat, original_json = llm.chat, llm.chat_json
    records: list[dict] = []
    lock = threading.Lock()

    def chat(messages, **kwargs):
        record = {"case_id": _case_id.get(), "messages_sha256": digest(messages),
                  "system_prompt_sha256": digest([m for m in messages if m.get("role") == "system"]),
                  "parameters": {k: kwargs[k] for k in ("tier", "temperature", "max_tokens", "json_mode") if k in kwargs}}
        try:
            result = original_chat(messages, **kwargs)
            record.update(status="returned", output_sha256=digest(result))
            return result
        except Exception as exc:
            record.update(status="failure", error_type=type(exc).__name__, error=safe_text(str(exc)))
            raise
        finally:
            with lock:
                records.append(record)

    def chat_json(messages, **kwargs):
        result = original_json(messages, **kwargs)
        if any(m.get("role") == "system" and m.get("content") == assist.JUDGE_SYSTEM for m in messages):
            if (not isinstance(result, dict) or not isinstance(result.get("verdict"), str)
                    or result["verdict"] not in {"supports", "partial", "unrelated"}):
                raise ValueError("eval_invalid_output: judge verdict is missing or outside the schema")
        return result

    llm.chat, llm.chat_json = chat, chat_json
    # Cached results from interactive use / previous evaluations must not bypass
    # raw-output validation or change the request manifest.
    assist._cached_support.cache_clear()
    assist._english_keywords.cache_clear()
    try:
        yield records
    finally:
        llm.chat, llm.chat_json = original_chat, original_json
        assist._cached_support.cache_clear()
        assist._english_keywords.cache_clear()


def _endpoint(value: str | None) -> dict | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        # Paths may contain credentials too. Keep only origin + a path hash;
        # exclude userinfo, query values and fragments, even if not configured keys.
        return {"origin": safe_text(f"{parsed.scheme}://{parsed.hostname or ''}" +
                                   (f":{parsed.port}" if parsed.port else "")),
                "path_sha256": text_digest(parsed.path)}
    except ValueError:
        return {"invalid_endpoint": True}


def configuration(settings: Any) -> dict:
    return {
        "provider": safe_text(settings.provider), "offline": settings.llm_fake,
        "tiers": {name: {"model": safe_text(tier.model), "timeout": tier.timeout,
                         "retries": tier.retries, "fallbacks": [safe_text(x) for x in tier.fallbacks],
                         "api_key_configured": bool(tier.api_key), "endpoint": _endpoint(tier.api_base)}
                  for name, tier in settings.tiers.items()},
        "embedding": {"provider": safe_text(settings.embedding_provider),
                      "model": safe_text(settings.embedding_model),
                      "api_key_configured": bool(settings.embedding_api_key),
                      "endpoint": _endpoint(settings.embedding_base_url)},
        "pdf_parser": settings.pdf_parser,
    }


def git_state() -> dict:
    def run(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=REPO, stderr=subprocess.DEVNULL, timeout=15)

    try:
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        return {"commit": run("rev-parse", "HEAD").decode().strip(), "dirty": bool(status.strip()),
                "status": status.decode("utf-8", errors="replace").splitlines(),
                "tracked_diff_sha256": hashlib.sha256(run("diff", "HEAD", "--no-ext-diff", "--binary")).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "unavailable": True}


def capture(settings: Any, bundle: Any, seed: int, task_names: list[str], limit: int | None) -> dict:
    from app import assist

    sources = sorted((BACKEND / "evals").glob("*.py")) + [BACKEND / "app" / x for x in
              ("assist.py", "llm.py", "config.py", "retrieval.py", "ingest.py", "pdfview.py", "embeddings.py")]
    prompts = {"JUDGE_SYSTEM": text_digest(assist.JUDGE_SYSTEM),
               "COMPLETE_SYSTEM": text_digest(assist.COMPLETE_SYSTEM),
               "_review_source": text_digest(inspect.getsource(assist._review)),
               "_english_keywords_source": text_digest(inspect.getsource(assist._english_keywords))}
    return {
        "protocol_version": PROTOCOL_VERSION, "git": git_state(),
        "runtime": {"python": platform.python_version(), "platform": platform.system()},
        "seed": seed, "seeds": {"sampling_and_negative_generation": seed, "locate_builder": 7, "locate_layout": 11},
        "tasks": task_names, "limit": limit, "config": configuration(settings),
        "sources": [file_record(p) for p in sources], "prompt_sources_sha256": prompts,
        "dataset_files": bundle.source_files,
        "bundle_sha256": digest({"pairs": [vars(p) for p in bundle.pairs], "passages": bundle.passages,
                                 "zh_claims": bundle.zh_claims, "citeworthy": bundle.citeworthy,
                                 "reviewed_negatives": bundle.reviewed_negatives}),
        "input_hash_scope": "Each task hashes its exact sampled inputs; model requests are audited separately.",
        "request_audit_scope": "app.llm.chat messages, including JSON repair requests; provider transport retries and embedding calls are not individually intercepted",
    }
