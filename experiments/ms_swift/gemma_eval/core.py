"""Frozen inputs for local Gemma evaluation. Preparation never loads weights."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from cpt.assets import load_native_tokenizer
from cpt.core import _documents, digest, file_hash, new_output, read_jsonl, write_json, write_jsonl
from cpt.schema import MODEL_MAX_LENGTH, MODEL_REPO, MODEL_REVISION, SPECIAL_TOKEN_IDS
from .chat import PROTOCOL, PROTOCOL_VERSION, chat_rows, validate_input_manifest

VERSION = "gemma-local-eval-v1"


def positive(value, name, maximum=MODEL_MAX_LENGTH):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")


def tokenizer_assets(directory):
    from tokenizers import Tokenizer
    metadata = load_native_tokenizer(directory).metadata
    tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer, metadata


def encode_prompt(tokenizer, prompt):
    # Explicit Base completion: exactly one prefix BOS, no chat template or EOS.
    return [SPECIAL_TOKEN_IDS["bos_token_id"]] + tokenizer.encode(prompt, add_special_tokens=False).ids


def window_tokens(ids, length):
    """Each target after BOS is scored once; one-token overlap retains boundary target."""
    positive(length, "window_length")
    if length < 2:
        raise ValueError("window_length must be at least 2")
    for start in range(0, len(ids) - 1, length - 1):
        piece = ids[start:start + length]
        yield {"input_ids": piece, "labels": [-100] + piece[1:], "start_token": start,
               "loss_tokens": len(piece) - 1}


def benchmark_rows(examples, tokenizer, context_length, limit=None):
    positive(context_length, "context_length")
    if limit is not None:
        positive(limit, "limit", 1000000)
    selected = examples if limit is None else examples[:limit]
    if not selected:
        raise ValueError("no benchmark records")
    seen, rows = set(), []
    for example in selected:
        eid, prompt = example.get("id"), example.get("prompt")
        if not isinstance(eid, str) or not eid or eid in seen:
            raise ValueError("benchmark IDs must be unique nonempty strings")
        seen.add(eid)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{eid}: requires the producer's frozen rendered prompt")
        if hashlib.sha256(prompt.encode()).hexdigest() != example.get("prompt_sha256"):
            raise ValueError(f"{eid}: frozen prompt hash mismatch")
        cap = example.get("official_max_new_tokens")
        positive(cap, "official_max_new_tokens")
        ids = encode_prompt(tokenizer, prompt)
        rows.append({"id": eid, "benchmark": example["benchmark"], "task": example["task"],
                     "domain": example.get("domain", example["task"]), "language": example.get("language", "unknown"),
                     "example_sha256": digest(example), "prompt": prompt, "prompt_sha256": example["prompt_sha256"],
                     "input_ids": ids, "input_tokens": len(ids), "max_new_tokens": cap,
                     "status": "ready" if len(ids) + cap <= context_length else "skipped_context_limit"})
    return rows


def adapter_base_matches(value):
    """SWIFT/PEFT can persist the pinned HF snapshot path instead of repo ID.

    This recognizes a declaration only. Training plan/admission validation and
    the runtime's independently hashed, pinned base weights remain mandatory.
    """
    if value == MODEL_REPO:
        return True
    if not isinstance(value, str):
        return False
    for path in (PurePosixPath(value), PureWindowsPath(value)):
        if (path.is_absolute() and ".." not in path.parts
                and path.parts[-3:] == ("models--google--gemma-4-E2B", "snapshots", MODEL_REVISION)):
            return True
    return False


def adapter_files(path):
    if path is None:
        return None
    path = Path(path).resolve()
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not adapter_base_matches(config.get("base_model_name_or_path")) or config.get("peft_type") != "LORA":
        raise ValueError("adapter must declare the same Gemma Base and LORA method")
    if config.get("revision") not in (None, MODEL_REVISION):
        raise ValueError("adapter base revision mismatch")
    weights = sorted(path.glob("adapter_model*.safetensors"))
    if not weights:
        raise ValueError("adapter requires safetensors weights; pickle adapters are not accepted")
    owner = next((p for p in [path, *list(path.parents)[:3]] if (p / "launch_plan.json").is_file()), None)
    if owner is None:
        raise ValueError("adapter requires its bound training launch plan and receipt")
    plan_path, receipt_path = owner / "launch_plan.json", owner / "launch_receipt.json"
    source = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (source.get("recipe", {}).get("model") != {"repo": MODEL_REPO, "revision": MODEL_REVISION}
            or not path.is_relative_to(Path(source["checkpoint_dir"]).resolve())
            or receipt.get("plan_sha256") != file_hash(plan_path)
            or receipt.get("status") != "process_exited" or receipt.get("returncode") != 0):
        raise ValueError("adapter training source/revision/process receipt mismatch")
    from cpt.launch import launch
    if not launch(plan_path, execute=False).get("training_ready"):
        raise ValueError("adapter source training plan is not valid or admitted")
    files = [config_path, *weights]
    return {"path": str(path), "files": {p.name: file_hash(p) for p in files},
            "source_plan": {"path": str(plan_path), "sha256": file_hash(plan_path)},
            "source_receipt_sha256": file_hash(receipt_path), "revision_binding": "validated_training_plan",
            "note": "Source hashes bind an adapter; runtime loading and model quality remain separate checks."}


def prepare(input_path, model_assets, output, *, kind="benchmark", context_length=8192,
            quantization="nf4", limit=None, lengths=(2048, 4096, 8192), max_new_tokens=32,
            split="dev", adapter=None, cache_dir=None):
    if kind not in {"benchmark", "profile", "nll", "chat"} or quantization not in {"nf4", "bf16"}:
        raise ValueError("unsupported kind or quantization")
    positive(context_length, "context_length")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if limit is not None:
        positive(limit, "limit", 1000000)
    tokenizer, metadata = tokenizer_assets(model_assets)
    input_path = Path(input_path).resolve()
    provenance = {"path": str(input_path), "sha256": file_hash(input_path)}
    if kind == "chat":
        examples = read_jsonl(input_path)
        provenance["eval_manifest"] = validate_input_manifest(input_path, examples)
        rows = chat_rows(examples, tokenizer, context_length, limit)
        protocol = PROTOCOL_VERSION
    elif kind == "benchmark":
        examples = read_jsonl(input_path)
        fetch_path = input_path.parent / "fetch_manifest.json"
        fetch = json.loads(fetch_path.read_text(encoding="utf-8"))
        if fetch.get("selected_records_sha256") != provenance["sha256"] or fetch.get("records") != len(examples):
            raise ValueError("benchmark input differs from adjacent fetch manifest")
        if fetch.get("selected_ids") is not None and fetch["selected_ids"] != [x["id"] for x in examples]:
            raise ValueError("benchmark ID order differs from fetch manifest")
        if not fetch.get("source_manifests"):
            raise ValueError("benchmark fetch manifest requires source manifests")
        source_ids = [eid for source in fetch["source_manifests"] for eid in source.get("selected_ids", [])]
        if source_ids != [x["id"] for x in examples]:
            raise ValueError("source manifest ID order/count differs from benchmark records")
        provenance["fetch_manifest"] = {"path": str(fetch_path), "sha256": file_hash(fetch_path)}
        rows = benchmark_rows(examples, tokenizer, context_length, limit)
        protocol = "frozen_public_prompts_base_completion_no_chat; overflow kept as failure; no truncation"
    elif kind == "profile":
        positive(max_new_tokens, "max_new_tokens")
        if not lengths or len(set(lengths)) != len(lengths):
            raise ValueError("profile lengths must be nonempty and unique")
        text = input_path.read_text(encoding="utf-8-sig")
        seed_ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not seed_ids:
            raise ValueError("profile seed text is empty")
        rows = []
        for length in lengths:
            positive(length, "profile length")
            if length < 2 or length + max_new_tokens > MODEL_MAX_LENGTH:
                raise ValueError("profile length plus output exceeds model window")
            count = length - 1
            body = (seed_ids * ((count + len(seed_ids) - 1) // len(seed_ids)))[:count]
            rows.append({"id": f"profile-{length}", "task": "synthetic_resource_probe", "input_ids": [2] + body,
                         "input_tokens": length, "max_new_tokens": max_new_tokens, "status": "ready"})
        protocol = "synthetic_repeated_seed_token_ids_exact_input_lengths; no quality metric; cold first case"
    else:
        if split not in {"dev", "test"}:
            raise ValueError("NLL accepts only explicit dev/test splits")
        docs, _ = _documents(input_path)
        selected = [d for d in docs if d["split"] == split]
        selected = selected if limit is None else selected[:limit]
        rows = []
        for doc in selected:
            text = "\n\n".join(b.get("cleaned_text", b["text"]) for b in doc["blocks"])
            ids = encode_prompt(tokenizer, text) + [1]
            if len(ids) < 3:
                raise ValueError("empty NLL document")
            rows.append({"id": doc["id"], "task": "heldout_nll", "split": split,
                         "document_sha256": digest(doc), "input_ids": ids, "input_tokens": len(ids),
                         "loss_tokens": len(ids) - 1, "status": "ready"})
        if not rows:
            raise ValueError("no documents in requested held-out split")
        protocol = "nonoverlapping_targets_with_one_previous_token; reset_KV_each_window; BOS_and_EOS; no_padding"
    out = new_output(output)
    write_jsonl(out / "requests.jsonl", rows)
    plan = {"schema_version": VERSION, "kind": kind, "input": provenance,
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION}, "tokenizer": metadata,
            "adapter": adapter_files(adapter), "cache_dir": str(Path(cache_dir).resolve()) if cache_dir else None,
            "quantization": quantization, "compute_dtype": "bfloat16", "device_policy": "single_cuda_no_cpu_offload",
            "context_length": context_length, "protocol": protocol, "do_sample": False,
            "selection": {"limit": limit, "split": split, "lengths": list(lengths), "max_new_tokens": max_new_tokens},
            "requests": {"path": str((out / "requests.jsonl").resolve()), "sha256": file_hash(out / "requests.jsonl")},
            "selected_ids": [r["id"] for r in rows], "counts": dict(Counter(r["status"] for r in rows)),
            "total_records": len(rows), "executed": False,
            "limitations": ["Preparation is not inference success or measured GPU fit.",
                            "Base completion intentionally uses no instruction model or chat wrapper.",
                            "Public benchmark scores must be computed with backend publicbench rescore."]}
    if kind == "chat":
        plan["chat_protocol"] = deepcopy(PROTOCOL)
        plan["limitations"] = ["Preparation is not inference success or measured GPU fit.",
                               "The same fixed no-thinking chat protocol applies to Base and all SFT adapters.",
                               "Scoring must use the frozen producer's metric implementation; failures remain in the denominator.",
                               "Existing legacy Base completion plans are a different protocol and cannot be compared directly."]
    write_json(out / "plan.json", plan)
    return plan


def load_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != VERSION or plan.get("model") != {"repo": MODEL_REPO, "revision": MODEL_REVISION}:
        raise ValueError("unsupported model or plan")
    if plan.get("kind") not in {"benchmark", "profile", "nll", "chat"} or plan.get("quantization") not in {"nf4", "bf16"}:
        raise ValueError("unsupported evaluation protocol")
    if plan.get("device_policy") != "single_cuda_no_cpu_offload" or plan.get("compute_dtype") != "bfloat16" or plan.get("do_sample") is not False:
        raise ValueError("evaluation runtime contract changed")
    positive(plan["context_length"], "context_length")
    if plan["kind"] == "chat" and (plan.get("protocol") != PROTOCOL_VERSION or plan.get("chat_protocol") != PROTOCOL):
        raise ValueError("fixed Gemma4 chat protocol changed")
    bound_files = [plan["input"], plan["requests"]]
    bound_files.extend(plan["input"][key] for key in ("fetch_manifest", "eval_manifest") if key in plan["input"])
    for item in bound_files:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("evaluation source/request file changed")
    tokenizer, metadata = tokenizer_assets(plan["tokenizer"]["asset_directory"])
    if metadata != plan["tokenizer"]:
        raise ValueError("model tokenizer assets or paths changed; prepare a new plan")
    if plan.get("adapter") and adapter_files(plan["adapter"]["path"]) != plan["adapter"]:
        raise ValueError("adapter changed since plan creation")
    rows = read_jsonl(plan["requests"]["path"])
    if [r["id"] for r in rows] != plan["selected_ids"] or len(rows) != plan["total_records"]:
        raise ValueError("request order/count differs from plan")
    if dict(Counter(r["status"] for r in rows)) != plan["counts"]:
        raise ValueError("request status counts differ from plan")
    if plan["context_length"] < 2:
        raise ValueError("context_length must be at least 2")
    for row in rows:
        ids = row.get("input_ids")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < 262144 for x in ids) or len(ids) != row.get("input_tokens"):
            raise ValueError("invalid request token IDs")
        if row.get("status") not in {"ready", "skipped_context_limit"}:
            raise ValueError("invalid prepared request status")
        if plan["kind"] != "nll":
            positive(row.get("max_new_tokens"), "max_new_tokens")
            cap = plan["context_length"] if plan["kind"] in {"benchmark", "chat"} else MODEL_MAX_LENGTH
            fits = len(ids) + row["max_new_tokens"] <= cap
            if (row["status"] == "ready") != fits:
                raise ValueError("request budget/status inconsistent")
        elif row["status"] != "ready" or row.get("loss_tokens") != len(ids) - 1:
            raise ValueError("NLL loss-token accounting differs from the request")
    # Re-derive tokens from the bound sources. Updating both request file and its
    # hash cannot silently swap questions, labels, prompts or profile lengths.
    selection = plan["selection"]
    if plan["kind"] == "chat":
        examples = read_jsonl(plan["input"]["path"])
        if validate_input_manifest(plan["input"]["path"], examples) != plan["input"].get("eval_manifest"):
            raise ValueError("chat source manifest binding changed")
        expected = chat_rows(examples, tokenizer, plan["context_length"], selection["limit"])
    elif plan["kind"] == "benchmark":
        expected = benchmark_rows(read_jsonl(plan["input"]["path"]), tokenizer,
                                  plan["context_length"], selection["limit"])
    elif plan["kind"] == "profile":
        seed = tokenizer.encode(Path(plan["input"]["path"]).read_text(encoding="utf-8-sig"), add_special_tokens=False).ids
        if not seed:
            raise ValueError("profile seed became empty")
        expected = []
        for length in selection["lengths"]:
            positive(length, "profile length")
            positive(selection["max_new_tokens"], "max_new_tokens")
            count = length - 1
            if count < 1 or length + selection["max_new_tokens"] > MODEL_MAX_LENGTH:
                raise ValueError("invalid profile token budget")
            body = (seed * ((count + len(seed) - 1) // len(seed)))[:count]
            expected.append({"id": f"profile-{length}", "task": "synthetic_resource_probe", "input_ids": [2] + body,
                             "input_tokens": length, "max_new_tokens": selection["max_new_tokens"], "status": "ready"})
    else:
        docs, _ = _documents(plan["input"]["path"])
        if selection["split"] not in {"dev", "test"}:
            raise ValueError("NLL requires a held-out split")
        selected = [d for d in docs if d["split"] == selection["split"]]
        selected = selected if selection["limit"] is None else selected[:selection["limit"]]
        expected = []
        for doc in selected:
            ids = encode_prompt(tokenizer, "\n\n".join(b.get("cleaned_text", b["text"]) for b in doc["blocks"])) + [1]
            expected.append({"id": doc["id"], "task": "heldout_nll", "split": selection["split"],
                             "document_sha256": digest(doc), "input_ids": ids, "input_tokens": len(ids),
                             "loss_tokens": len(ids) - 1, "status": "ready"})
    if expected != rows:
        raise ValueError("frozen requests differ from native encoding of the bound sources")
    return plan, rows
