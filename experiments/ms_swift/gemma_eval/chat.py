"""Fixed text-only Gemma4 SFT comparison protocol; CPU preparation loads no Torch.

The wrapper and stop words follow Gemma4TemplateMeta at the pinned MS-SWIFT
commit. GPU execution additionally verifies EVERY frozen input against the real
SWIFT inference template. HF Base tokenizer metadata has no chat template; it is
never used as an implicit formatting policy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from cpt.core import digest, file_hash, read_jsonl
from cpt.schema import MODEL_MAX_LENGTH, SWIFT_COMMIT, validate_model_lock

PROTOCOL_VERSION = "gemma4-nothinking-eval-v1"
TEMPLATE_SOURCE = f"https://github.com/modelscope/ms-swift/blob/{SWIFT_COMMIT}/swift/template/templates/gemma.py"
STOP_TOKEN_IDS = [1, 106, 50]
SPECIAL_TOKENS = {"<bos>": 2, "<eos>": 1, "<|turn>": 105, "<turn|>": 106, "<|tool_response>": 50}
PROTOCOL = {
    "version": PROTOCOL_VERSION, "template": "gemma4_nothinking", "template_source": TEMPLATE_SOURCE,
    "swift_commit": SWIFT_COMMIT, "mode": "transformers", "system": None,
    "turns": "one user message, open assistant turn; no answer included",
    "bos_token_id": 2, "stop_token_ids": STOP_TOKEN_IDS, "pad_token_id": 0,
    "do_sample": False, "num_beams": 1, "seed": 17,
    "truncation": "none; retain over-budget records as skipped_context_limit",
    "budget": "native chat input tokens plus source max_new_tokens <= context_length",
    "runtime_verification": "all inputs must equal pinned SWIFT inference template before weight loading",
}


def _ids(tokenizer, text):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return encoded.ids if hasattr(encoded, "ids") else encoded


def encode_chat(tokenizer, prompt):
    """One full rendered string, exactly one format BOS; no automatic BOS/EOS."""
    for token, expected in SPECIAL_TOKENS.items():
        if _ids(tokenizer, token) != [expected]:
            raise ValueError(f"Gemma4 protocol special token mismatch: {token}")
    return _ids(tokenizer, "<bos><|turn>user\n" + prompt + "<turn|>\n<|turn>model\n")


def _positive(value, label):
    if type(value) is not int or not 1 <= value <= MODEL_MAX_LENGTH:
        raise ValueError(f"{label} must be a positive integer <= {MODEL_MAX_LENGTH}")


def chat_rows(examples, tokenizer, context_length, limit=None):
    _positive(context_length, "context_length")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1000000):
        raise ValueError("limit must be a positive integer <= 1000000")
    selected = examples if limit is None else examples[:limit]
    if not selected:
        raise ValueError("no chat evaluation records")
    seen, rows = set(), []
    for example in selected:
        eid, prompt = example.get("id"), example.get("prompt")
        if not isinstance(eid, str) or not eid or eid in seen:
            raise ValueError("chat evaluation IDs must be unique nonempty strings")
        seen.add(eid)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{eid}: requires nonempty frozen prompt")
        if example.get("evaluation_only") is not True:
            raise ValueError(f"{eid}: requires explicit evaluation_only=true")
        for field in ("benchmark", "task"):
            if not isinstance(example.get(field), str) or not example[field]:
                raise ValueError(f"{eid}: requires {field}")
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if example.get("prompt_sha256", prompt_sha) != prompt_sha:
            raise ValueError(f"{eid}: frozen prompt hash mismatch")
        cap = example.get("official_max_new_tokens", example.get("max_new_tokens"))
        _positive(cap, "official_max_new_tokens/max_new_tokens")
        if "max_new_tokens" in example and example["max_new_tokens"] != cap:
            raise ValueError(f"{eid}: generation reserves disagree")
        ids = encode_chat(tokenizer, prompt)
        rows.append({"id": eid, "benchmark": example["benchmark"], "task": example["task"],
                     "protocol": PROTOCOL_VERSION,
                     "domain": example.get("domain", example["task"]), "language": example.get("language", "unknown"),
                     "example_sha256": digest(example), "prompt": prompt, "prompt_sha256": prompt_sha,
                     "input_ids": ids, "input_tokens": len(ids), "max_new_tokens": cap,
                     "status": "ready" if len(ids) + cap <= context_length else "skipped_context_limit"})
    return rows


def validate_input_manifest(input_path, examples):
    """A producer owns prompt/score construction; its frozen file binds all labels.

    An optional source_manifest.file names local evidence that is rehashed on
    every load. Relative source paths resolve next to the manifest, never cwd.
    """
    path = Path(input_path).resolve().parent / "eval_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "gemma-eval-input-v1"
            or manifest.get("records_sha256") != file_hash(input_path)
            or type(manifest.get("records")) is not int or manifest["records"] != len(examples)
            or manifest.get("selected_ids") != [x["id"] for x in examples]
            or not isinstance(manifest.get("source_manifests"), list) or not manifest["source_manifests"]):
        raise ValueError("chat input differs from adjacent eval manifest or has no source provenance")
    for source in manifest["source_manifests"]:
        if not isinstance(source, dict) or not source:
            raise ValueError("source manifests must be nonempty objects")
        if "file" in source:
            spec = source["file"]
            candidate = Path(spec["path"])
            candidate = candidate if candidate.is_absolute() else path.parent / candidate
            if file_hash(candidate) != spec["sha256"]:
                raise ValueError("chat source evidence file changed")
    return {"path": str(path), "sha256": file_hash(path)}


def verify_swift_rows(plan, rows):
    """Verify the real, pinned inference template without loading any weights.

    The training environment is intentionally required here, not at import or
    preparation time. Isolated metadata and offline mode forbid a model download.
    max_length=None keeps even over-budget requests intact for comparison.
    """
    from cpt.swift_verify import _offline_environment, _runtime_provenance, _load_backend
    assets = Path(plan["tokenizer"]["asset_directory"])
    lock_path = assets / "model.lock.json"
    if file_hash(lock_path) != plan["tokenizer"]["model_lock_sha256"]:
        raise ValueError("chat tokenizer lock changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_model_lock(lock, assets)
    if {"repo": lock["repository"], "revision": lock["revision"]} != plan["model"]:
        raise ValueError("chat tokenizer and model mismatch")
    with _offline_environment(), tempfile.TemporaryDirectory(prefix="ariadne-chat-verify-") as temporary:
        isolated = Path(temporary)
        for name, spec in lock["files"].items():
            shutil.copyfile(assets / name, isolated / name)
            if file_hash(isolated / name) != spec["sha256"]:
                raise ValueError("chat metadata changed during copy")
        provenance = _runtime_provenance()
        get_processor, get_template, _ = _load_backend()
        processor = get_processor(str(isolated), model_type="gemma4", task_type="causal_lm", use_hf=True,
                                  revision=lock["revision"], download_model=False)
        if processor.__class__.__name__ != "Gemma4Processor":
            raise ValueError("chat verification requires Gemma4Processor")
        template = get_template(processor, template_type="gemma4_nothinking", max_length=None,
                                use_chat_template=True, truncation_strategy="raise", padding_free=False,
                                template_backend="swift", enable_thinking=False)
        template.set_mode("transformers")
        fingerprints = []
        for row in rows:
            encoded = template.encode({"messages": [{"role": "user", "content": row["prompt"]}]})
            if not isinstance(encoded, dict) or encoded.get("input_ids") != row["input_ids"]:
                raise ValueError(f"{row['id']}: frozen encoding differs from real SWIFT chat template")
            fingerprints.append({"id": row["id"], "input_ids_sha256": digest(row["input_ids"])})
    return {"schema_version": "gemma-chat-template-verification-v1", "verified": True,
            "protocol": PROTOCOL, "runtime": provenance, "records": len(rows),
            "input_ids_sha256": digest(fingerprints), "model_lock_sha256": file_hash(lock_path),
            "weights_loaded": False, "offline": True}


def verify_plan(path):
    from .core import load_plan
    plan, rows = load_plan(path)
    if plan["kind"] != "chat":
        raise ValueError("verify-chat requires a chat plan")
    return verify_swift_rows(plan, rows)
