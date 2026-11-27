"""Explicit local-only CUDA execution. Heavy packages are imported only in Runtime."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gc
from importlib import resources
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

from cpt.core import digest, file_hash, new_output, read_jsonl, write_json, write_jsonl
from .core import load_plan, window_tokens
from .chat import STOP_TOKEN_IDS, encode_chat, verify_swift_rows


def implementation_hashes():
    """Snapshot only this package's implementation and declared requirements."""
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("__init__.py", "__main__.py", "core.py", "chat.py", "portable.py", "runtime.py", "bounded.py", "model_weights.lock.json")]
    requirements = root.parent / "requirements-inference.txt"
    if requirements.is_file():
        paths.append(requirements)
    return {path.relative_to(root.parent).as_posix(): file_hash(path) for path in paths if path.is_file()}


def request_generation_config(base, max_new_tokens):
    """Per-case copy; never mix GenerationConfig with duplicate generate kwargs."""
    configured = deepcopy(base)
    configured.max_new_tokens = max_new_tokens
    configured.use_cache = True
    return configured


def _verify_weight_content(path, expected):
    path = Path(path)
    if path.stat().st_size != expected["bytes"]:
        raise ValueError("cached weight size differs from the pinned LFS size")
    actual = file_hash(path)
    if actual != expected["sha256"]:
        raise ValueError("cached weight content differs from the pinned LFS SHA-256")
    return actual


def verify_cached_weights(plan, resolver):
    lock_text = resources.files("gemma_eval").joinpath("model_weights.lock.json").read_text(encoding="utf-8")
    lock = json.loads(lock_text)
    if (lock.get("schema_version") != "gemma-pinned-weights-v1"
            or {"repo": lock.get("repository"), "revision": lock.get("revision")} != plan["model"]
            or lock.get("filename") != "model.safetensors"
            or type(lock.get("bytes")) is not int or lock["bytes"] < 1
            or not re.fullmatch(r"[a-f0-9]{64}", lock.get("sha256", ""))):
        raise ValueError("invalid or mismatched pinned model-weight lock")
    cached = resolver(lock["repository"], lock["filename"], revision=lock["revision"], cache_dir=plan.get("cache_dir"))
    if not isinstance(cached, str) or not Path(cached).is_file():
        raise FileNotFoundError("pinned model.safetensors is not present in the local cache")
    actual = _verify_weight_content(cached, lock)
    return cached, {"sha256": actual, "bytes": lock["bytes"], "lock_sha256": digest(lock), "source": lock["source"]}


def verify_cached_metadata(plan, resolver):
    """The cache read by from_pretrained must match the plan's inert assets."""
    assets = Path(plan["tokenizer"]["asset_directory"])
    lock_path = assets / "model.lock.json"
    if file_hash(lock_path) != plan["tokenizer"]["model_lock_sha256"]:
        raise ValueError("model lock changed before runtime metadata verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    fingerprints = {}
    for name in ("config.json", "generation_config.json"):
        expected = lock["files"][name]
        if file_hash(assets / name) != expected["sha256"]:
            raise ValueError(f"locked runtime metadata changed: {name}")
        cached = resolver(plan["model"]["repo"], name,
                          revision=plan["model"]["revision"], cache_dir=plan.get("cache_dir"))
        if not isinstance(cached, str) or not Path(cached).is_file():
            raise FileNotFoundError(f"pinned cached runtime metadata is absent: {name}")
        if Path(cached).stat().st_size != expected["bytes"] or file_hash(cached) != expected["sha256"]:
            raise ValueError(f"cached runtime metadata differs from the model lock: {name}")
        fingerprints[name] = expected["sha256"]
    return fingerprints


def greedy_generation_config(config_class, *, chat=False):
    """Fresh settings, never inherited from model/adapter generation_config.

    In particular no forced/blocked tokens, stop strings, minimum output length,
    beam search, repetition penalties, or sampling policy can arrive from cache.
    Defaults not named here come solely from pinned Transformers 5.12.1.
    """
    return config_class(do_sample=False, num_beams=1, num_return_sequences=1,
        temperature=1.0, top_p=1.0, typical_p=1.0,
        min_length=0, min_new_tokens=None, max_time=None, stop_strings=None,
        forced_bos_token_id=None, forced_eos_token_id=None, bad_words_ids=None,
        suppress_tokens=None, begin_suppress_tokens=None, sequence_bias=None,
        repetition_penalty=1.0, encoder_repetition_penalty=1.0,
        no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
        token_healing=False, renormalize_logits=False, watermarking_config=None,
        bos_token_id=2, eos_token_id=list(STOP_TOKEN_IDS) if chat else 1, pad_token_id=0, use_cache=True)


class Runtime:
    def __init__(self, plan):
        self.stage = "dependencies"
        try:
            self._initialize(plan)
        except Exception as exc:
            exc.gemma_stage = self.stage
            raise

    def _initialize(self, plan):
        self.chat = plan["kind"] == "chat"
        chat_verification = None
        if self.chat:
            self.stage = "chat_template_verification"
            chat_verification = verify_swift_rows(plan, read_jsonl(plan["requests"]["path"]))
        self.stage = "dependencies"
        import torch
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration, GenerationConfig
        for package, expected in {"torch": "2.8.0", "transformers": "5.12.1", "accelerate": "1.12.0", "peft": "0.20.0"}.items():
            if importlib.metadata.version(package).split("+")[0] != expected:
                raise RuntimeError(f"requires pinned {package} {expected}")
        self.stage = "cuda_initialization"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("requires CUDA with native BF16; CPU fallback is disabled")
        self.torch = torch
        torch.cuda.set_device(0)
        torch.manual_seed(17)
        torch.cuda.reset_peak_memory_stats(0)
        self.stage = "tokenizer_load"
        self.tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"]["asset_directory"], local_files_only=True,
                                                       trust_remote_code=False)
        self.stage = "weight_fingerprint"
        from huggingface_hub import try_to_load_from_cache
        self.stage = "cached_metadata_verification"
        metadata_hashes = verify_cached_metadata(plan, try_to_load_from_cache)
        locked_config = AutoConfig.from_pretrained(plan["tokenizer"]["asset_directory"],
                                                   local_files_only=True, trust_remote_code=False)
        self.generation_config = greedy_generation_config(GenerationConfig, chat=self.chat)
        self.stage = "weight_fingerprint"
        cached_weight, weights = verify_cached_weights(plan, try_to_load_from_cache)
        options = {"revision": plan["model"]["revision"], "dtype": torch.bfloat16, "config": locked_config,
                   "device_map": {"": 0}, "attn_implementation": "sdpa", "local_files_only": True,
                   "trust_remote_code": False, "cache_dir": plan.get("cache_dir")}
        if plan["quantization"] == "nf4":
            self.stage = "quantization_dependencies"
            if importlib.metadata.version("bitsandbytes") != "0.50.0":
                raise RuntimeError("requires pinned bitsandbytes 0.50.0")
            options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                bnb_4bit_use_double_quant=True,
                                                                bnb_4bit_compute_dtype=torch.bfloat16)
        self.stage = "model_load"
        self.model = Gemma4ForConditionalGeneration.from_pretrained(plan["model"]["repo"], **options)
        if plan.get("adapter"):
            self.stage = "adapter_load"
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, plan["adapter"]["path"], is_trainable=False,
                                                 local_files_only=True)
        self.stage = "device_verification"
        devices = {str(p.device) for p in self.model.parameters()}
        if any(not d.startswith("cuda") for d in devices):
            raise RuntimeError("model contains offloaded parameters; single-GPU validation refused")
        self.model.eval()
        self.metadata = {"gpu": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                         "cuda": torch.version.cuda, "devices": sorted(devices), "cpu_offload": False,
                         "platform": platform.system(),
                         "cpu_offload_scope": "framework parameter-device check only; no Hugging Face CPU offload",
                         "physical_vram_residency_verified": False,
                         "driver_or_os_paging_measured": False,
                         "memory_residency_note": "CUDA parameter devices do not prove exclusive physical VRAM residency; WDDM/OS-managed paging is not measured. Slow generation alone does not establish paging.",
                         "versions": {p: importlib.metadata.version(p) for p in
                                      ("torch", "transformers", "accelerate", "tokenizers", "peft")},
                         "quantization": plan["quantization"], "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                         "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                         "free_memory_after_load_bytes": torch.cuda.mem_get_info(0)[0],
                         "weights_sha256": weights["sha256"], "weights_bytes": weights["bytes"],
                         "weights_lock_sha256": weights["lock_sha256"], "weights_lock_source": weights["source"],
                         "cached_metadata_sha256": metadata_hashes,
                         "base_generation_config_sha256": digest(self.generation_config.to_dict()),
                         "generation_settings_source": "fresh explicit greedy config; no model/adapter settings inherited",
                         "quantization_note": "NF4 applies to supported linear layers; embeddings/PLE are not assumed four-bit."}
        if plan["quantization"] == "nf4":
            self.metadata["versions"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        if self.chat:
            self.metadata["chat_template_verification"] = chat_verification
        self.stage = "ready"

    def generate(self, row):
        torch = self.torch
        if "prompt" in row:
            actual = (encode_chat(self.tokenizer, row["prompt"]) if getattr(self, "chat", False)
                      else [2] + self.tokenizer.encode(row["prompt"], add_special_tokens=False))
            if actual != row["input_ids"]:
                raise ValueError("HF runtime encoding differs from frozen native-tokenizer IDs")
        ids = torch.tensor([row["input_ids"]], device="cuda:0", dtype=torch.long)
        mask = torch.ones_like(ids)
        from transformers.generation.streamers import BaseStreamer

        class Timer(BaseStreamer):
            def __init__(self):
                self.prompt_seen = False
                self.first = None
                self.last = None

            def put(self, value):
                if not self.prompt_seen:
                    self.prompt_seen = True
                    return
                torch.cuda.synchronize(0)
                now = time.perf_counter()
                self.first = now if self.first is None else self.first
                self.last = now

            def end(self):
                pass

        timer = Timer()
        generation_config = request_generation_config(self.generation_config, row["max_new_tokens"])
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=generation_config,
                                         streamer=timer, logits_to_keep=1)
        torch.cuda.synchronize(0)
        end = time.perf_counter()
        output = result[0, len(row["input_ids"]):].tolist()
        prediction = self.tokenizer.decode(output, skip_special_tokens=True)
        ttft = timer.first - start if timer.first is not None else None
        decode_time = timer.last - timer.first if timer.first is not None and timer.last is not None else None
        return {"prediction": prediction, "status": "ok" if prediction.strip() else "empty_prediction",
                "generation_config_sha256": digest(generation_config.to_dict()),
                "generated_tokens": len(output), "generation_seconds": end - start,
                "time_to_first_token_seconds": ttft,
                "decode_tokens_per_second": (len(output) - 1) / decode_time if len(output) > 1 and decode_time and decode_time > 0 else None,
                "input_tokens_per_first_token_second": len(row["input_ids"]) / ttft if ttft else None,
                "end_to_end_generated_tokens_per_second": len(output) / (end - start),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "finish_reason": ("eos" if output and output[-1] == 1 else
                                  "turn_stop" if getattr(self, "chat", False) and output and output[-1] in STOP_TOKEN_IDS else "length"),
                "terminal_token_id": output[-1] if output else None,
                "latency_protocol": "synchronized CUDA; streamer observes token delivery; first case cold; no warmup",
                "usage": {"prompt_tokens": len(row["input_ids"]), "completion_tokens": len(output),
                          "total_tokens": len(row["input_ids"]) + len(output), "cost_usd": None}}

    def nll(self, row, length):
        torch = self.torch
        total, tokens, windows = 0.0, 0, 0
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(0)
        for window in window_tokens(row["input_ids"], length):
            ids = torch.tensor([window["input_ids"]], device="cuda:0", dtype=torch.long)
            labels = torch.tensor([window["labels"]], device="cuda:0", dtype=torch.long)
            with torch.inference_mode():
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
            loss = float(output.loss.item())
            if not math.isfinite(loss):
                raise ValueError("nonfinite held-out loss")
            total += loss * window["loss_tokens"]
            tokens += window["loss_tokens"]
            windows += 1
            del output, ids, labels
        torch.cuda.synchronize(0)
        if tokens != row["loss_tokens"]:
            raise ValueError("NLL target accounting differs from plan")
        return {"status": "ok", "nll_sum": total, "loss_tokens": tokens, "mean_nll": total / tokens,
                "windows": windows, "generation_seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}

    def recover(self):
        gc.collect()
        self.torch.cuda.empty_cache()


def failure(exc, stage=None):
    # Exception text may contain network credentials or machine paths. Store type
    # and a bounded category; do not serialize environment or provider responses.
    name = type(exc).__name__
    message = str(exc).lower()
    category = ("cuda_out_of_memory" if "OutOfMemory" in name or "out of memory" in message else
                "missing_dependency" if isinstance(exc, (ModuleNotFoundError, importlib.metadata.PackageNotFoundError)) else
                "dependency_version_mismatch" if "requires pinned" in message else
                "tokenizer_mismatch" if "encoding differs" in message else
                "missing_cached_weight" if isinstance(exc, FileNotFoundError) and getattr(exc, "gemma_stage", "") == "weight_fingerprint" else "runtime_error")
    result = {"error_type": name, "error_category": category,
              "error_stage": getattr(exc, "gemma_stage", stage or "unknown"),
              "error_detail": "Structured diagnostic only; raw exceptions and environment variables are not exported."}
    module = getattr(exc, "name", None)
    if isinstance(module, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", module):
        result["missing_module"] = module
    versions = {}
    for package in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result["installed_versions"] = versions
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                result["cuda_memory_at_failure"] = {"free_bytes": free, "total_bytes": total,
                    "allocated_bytes": torch.cuda.memory_allocated(0), "reserved_bytes": torch.cuda.memory_reserved(0),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        except Exception:
            result["cuda_memory_at_failure"] = None
    return result


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _atomic_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def execute_rows(plan, requests, output, runtime_factory=Runtime):
    out = new_output(output)
    implementations = implementation_hashes()
    _atomic_json(out / "run_manifest.json", {"schema_version": "gemma-local-run-v1", "status": "loading_model",
                                         "plan": plan, "executed": True, "completed": 0,
                                         "implementation_sha256": implementations})
    runtime, load_error = None, None
    if any(r["status"] == "ready" for r in requests):
        try:
            runtime = runtime_factory(plan)
        except Exception as exc:
            load_error = failure(exc)
    rows = []
    for request in requests:
        row = {k: v for k, v in request.items() if k not in {"input_ids", "prompt", "status"}}
        row.update(status=request["status"], prediction="", protocol=plan["protocol"],
                   model=plan["model"]["repo"], model_revision=plan["model"]["revision"],
                   quantization=plan["quantization"], evaluation_only=True, transport_called=False,
                   example_sha256=request.get("example_sha256"))
        if request["status"] == "ready":
            if load_error:
                row.update(status="model_load_failed", **load_error)
            else:
                try:
                    row["transport_called"] = True
                    result = runtime.nll(request, plan["context_length"]) if plan["kind"] == "nll" else runtime.generate(request)
                    row.update(result)
                except Exception as exc:
                    row.update(status="failed", **failure(exc, stage=plan["kind"]))
                    try:
                        runtime.recover()
                    except Exception as recovery:
                        row["recovery_error"] = failure(recovery, stage="cuda_recovery")
        rows.append(row)
        # Checkpoint after each case; output is local/ignored. Interrupted runs
        # retain completed cases and never get a completed success summary.
        _atomic_jsonl(out / "predictions.jsonl", rows)
        _atomic_json(out / "run_manifest.json", {"schema_version": "gemma-local-run-v1", "status": "in_progress",
                                              "plan_sha256": digest(plan), "executed": True,
                                              "completed": len(rows), "total": len(requests),
                                              "implementation_sha256": implementations})
    statuses = dict(Counter(r["status"] for r in rows))
    summary = {"schema_version": "gemma-local-summary-v1", "kind": plan["kind"], "executed": True,
               "model_loaded": runtime is not None, "runtime": runtime.metadata if runtime else None,
               "plan_sha256": digest(plan), "selected_ids": [r["id"] for r in rows], "total": len(rows),
               "statuses": statuses, "successful": statuses.get("ok", 0), "failed_or_skipped": len(rows) - statuses.get("ok", 0),
               "by_task": {task: dict(Counter(r["status"] for r in rows if r["task"] == task)) for task in sorted({r["task"] for r in rows})},
               "predictions_sha256": file_hash(out / "predictions.jsonl"), "load_error": load_error,
               "quality_scores": None, "notes": ["All planned IDs remain in the denominator, including overflow and load failures.",
                                                "Use publicbench rescore for official task scores; resource probes do not measure model quality.",
                                                "PyTorch allocation peaks exclude unrelated processes and driver allocations.",
                                                "CUDA tensor placement and disabled framework CPU offload do not verify physical VRAM residency or rule out driver/OS-managed paging."]}
    summary["implementation_sha256"] = implementations
    if plan["kind"] == "chat":
        summary["evaluation_contract"] = {"protocol": plan["protocol"], "model": plan["model"],
            "context_length": plan["context_length"], "quantization": plan["quantization"],
            "adapter": ({"files": plan["adapter"]["files"], "source_plan_sha256": plan["adapter"]["source_plan"]["sha256"]}
                        if plan.get("adapter") else None)}
    if plan["kind"] == "nll":
        good = [r for r in rows if r["status"] == "ok"]
        tokens = sum(r["loss_tokens"] for r in good)
        nll_sum = sum(r["nll_sum"] for r in good)
        complete = len(good) == len(rows)
        summary["nll"] = {"planned_loss_tokens": sum(r["loss_tokens"] for r in requests), "scored_loss_tokens": tokens,
                          "complete": complete, "mean_nll": nll_sum / tokens if complete and tokens else None,
                          "conditional_mean_nll_successful_documents": nll_sum / tokens if tokens else None,
                          "protocol": plan["protocol"]}
    _atomic_json(out / "summary.json", summary)
    _atomic_json(out / "run_manifest.json", {"schema_version": "gemma-local-run-v1", "status": "completed",
                                          "plan": plan, "executed": True, "completed": len(rows),
                                          "implementation_sha256": implementations,
                                          "summary_sha256": file_hash(out / "summary.json")})
    return summary


def run(plan_path, output=None, *, execute=False):
    plan, rows = load_plan(plan_path)
    if not execute:
        return {"executed": False, "plan_sha256": digest(plan), "kind": plan["kind"],
                "total_records": len(rows), "counts": plan["counts"], "message": "Inspection only; --execute explicitly loads local cached model weights."}
    if output is None:
        raise ValueError("execution requires a fresh --output directory")
    return execute_rows(plan, rows, output)
