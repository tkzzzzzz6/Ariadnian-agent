"""Pure-stdlib recipe and native-model provenance validation; never loads weights.

Verified sources: google/gemma-4-E2B config/tokenizer metadata at MODEL_REVISION;
MS-SWIFT v4.6.0 (ad818a2cdfafaf4305ac40c4acd892a211978bc2).
Validation is not a GPU/runtime test, training completion, or data-review approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

RECIPE_SCHEMA_VERSION = "cpt-recipe-v1"
MODEL_LOCK_SCHEMA_VERSION = "cpt-model-lock-v1"
MODEL_REPO = "google/gemma-4-E2B"
MODEL_REVISION = "433813a669639fd7eaa83155828e9000ed357d53"
SWIFT_VERSION = "4.6.0"
SWIFT_COMMIT = "ad818a2cdfafaf4305ac40c4acd892a211978bc2"
BITSANDBYTES_VERSION = "0.50.0"
METHODS = {"pt_lora", "pt_qlora", "sft_lora", "sft_qlora"}
QUANT_PARAMETERS = {"quant_method": "bnb", "quant_bits": 4,
                    "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True}
MODEL_MAX_LENGTH = 131072
MODEL_VOCAB_SIZE = 262144
SPECIAL_TOKEN_IDS = {"bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0}
SPECIAL_TOKEN_TEXT = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}
REQUIRED_MODEL_FILES = frozenset({"tokenizer.json", "tokenizer_config.json", "config.json"})
ALLOWED_MODEL_FILES = REQUIRED_MODEL_FILES | frozenset({
    "generation_config.json", "processor_config.json", "preprocessor_config.json",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "spiece.model",
    "vocab.json", "merges.txt",
})
MAX_METADATA_FILE_BYTES = 128 * 1024 * 1024

# These are explicit CLI argument names from the pinned SWIFT/HF contract.
# model/dataset/output paths, adapters, hubs, plugins and executable config files
# are deliberately outside this recipe; the owning launcher must bind them.
FIXED_PARAMETERS = {
    "model_type": "gemma4", "torch_dtype": "bfloat16", "tuner_type": "lora",
    "template": "gemma4_nothinking", "use_chat_template": False, "loss_scale": "all",
    "target_modules": ["all-linear"], "freeze_vit": True, "freeze_aligner": True, "freeze_llm": False,
    "strict": True, "truncation_strategy": "delete", "packing": False, "padding_free": False,
    "streaming": False, "load_args": False, "split_dataset_ratio": 0.0,
}
INTEGER_RANGES = {
    "max_length": (2, MODEL_MAX_LENGTH), "lora_rank": (1, 1024), "lora_alpha": (1, 65536),
    "per_device_train_batch_size": (1, 4096), "per_device_eval_batch_size": (1, 4096),
    "gradient_accumulation_steps": (1, 1048576), "max_steps": (1, 1000000000),
    "logging_steps": (1, 1000000000), "save_steps": (1, 1000000000),
    "eval_steps": (1, 1000000000), "save_total_limit": (1, 1000000),
    "seed": (0, 2**32 - 1), "data_seed": (0, 2**32 - 1),
    "dataloader_num_workers": (0, 256),
}
FLOAT_RANGES = {
    "learning_rate": (0.0, 1.0, False), "weight_decay": (0.0, 1.0, True),
    "warmup_ratio": (0.0, 1.0, True), "lora_dropout": (0.0, 1.0, True),
    "num_train_epochs": (0.0, 1000000.0, False), "max_grad_norm": (0.0, 1000000.0, True),
}
BOOL_PARAMETERS = frozenset({"gradient_checkpointing", "load_best_model_at_end", "greater_is_better",
                             "add_version", "save_only_model", "use_flash_ckpt"})
CHOICE_PARAMETERS = {"attn_impl": {"sdpa", "eager"}, "lr_scheduler_type": {"linear", "cosine", "constant", "constant_with_warmup"},
                     "save_strategy": {"steps", "epoch", "no", "best"}, "eval_strategy": {"steps", "epoch", "no"},
                     "metric_for_best_model": {"eval_loss"}}
ALLOWED_PARAMETERS = frozenset(FIXED_PARAMETERS) | frozenset(INTEGER_RANGES) | frozenset(FLOAT_RANGES) | BOOL_PARAMETERS | frozenset(CHOICE_PARAMETERS) | {"report_to"}
REQUIRED_PARAMETERS = frozenset(FIXED_PARAMETERS) | {"max_length", "lora_rank", "lora_alpha", "per_device_train_batch_size", "seed"}


def default_recipe(method="pt_lora", *, sequence_length=None, smoke=False) -> dict:
    """Return an editable plan, with no readiness or training-success claims."""
    result = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "trainer": {"name": "ms-swift", "version": SWIFT_VERSION}, "method": "pt_lora",
        "parameters": {
            **{k: list(v) if isinstance(v, list) else v for k, v in FIXED_PARAMETERS.items()},
            "max_length": 8192, "lora_rank": 16, "lora_alpha": 32,
            "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8, "gradient_checkpointing": True,
            "learning_rate": 1e-4, "max_steps": 100, "seed": 17,
            "attn_impl": "sdpa", "report_to": ["none"],
        },
    }
    if method not in METHODS:
        raise ValueError("unsupported training method")
    result["method"] = method
    if method.startswith("sft_"):
        result["parameters"].update(use_chat_template=True, loss_scale="default")
    if method.endswith("qlora"):
        result["parameters"].update(QUANT_PARAMETERS)
    if sequence_length is not None:
        result["parameters"]["max_length"] = sequence_length
    if smoke:
        result["parameters"].update(max_steps=20, logging_steps=1, save_steps=20,
                                    eval_steps=20, save_total_limit=1, data_seed=17,
                                    save_strategy="steps", eval_strategy="steps")
    validate_recipe(result)
    return result


def _object(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _keys(value: dict, required: set | frozenset, allowed: set | frozenset, name: str) -> None:
    if set(value) - allowed:
        raise ValueError(f"{name}: unknown fields/parameters: {', '.join(sorted(str(x) for x in set(value) - allowed))}")
    if required - set(value):
        raise ValueError(f"{name}: missing required fields/parameters: {', '.join(sorted(required - set(value)))}")


def _revision(value, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{name} must be a lowercase immutable 40-hex Hugging Face revision, not main/a tag")


def validate_recipe(recipe: dict) -> None:
    recipe = _object(recipe, "recipe")
    fields = {"schema_version", "model", "trainer", "method", "parameters"}
    _keys(recipe, fields, fields, "recipe")
    if recipe["schema_version"] != RECIPE_SCHEMA_VERSION:
        raise ValueError("unsupported CPT recipe schema_version")
    model = _object(recipe["model"], "recipe.model")
    _keys(model, {"repo", "revision"}, {"repo", "revision"}, "recipe.model")
    if model["repo"] != MODEL_REPO:
        raise ValueError("this recipe supports only Gemma 4 E2B Base; an instruction-tuned/Qwen/quantized checkpoint is not interchangeable")
    _revision(model["revision"], "recipe.model.revision")
    trainer = _object(recipe["trainer"], "recipe.trainer")
    _keys(trainer, {"name", "version"}, {"name", "version"}, "recipe.trainer")
    if trainer != {"name": "ms-swift", "version": SWIFT_VERSION}:
        raise ValueError(f"trainer must be pinned to ms-swift {SWIFT_VERSION}")
    if recipe["method"] not in METHODS:
        raise ValueError("unsupported training method")
    fixed = dict(FIXED_PARAMETERS)
    if recipe["method"].startswith("sft_"):
        fixed.update(use_chat_template=True, loss_scale="default")
    if recipe["method"].endswith("qlora"):
        fixed.update(QUANT_PARAMETERS)
    parameters = _object(recipe["parameters"], "recipe.parameters")
    _keys(parameters, REQUIRED_PARAMETERS | set(fixed), ALLOWED_PARAMETERS | set(fixed), "recipe.parameters")
    for name, required in fixed.items():
        actual = parameters[name]
        if isinstance(required, bool):
            valid = type(actual) is bool and actual is required
        elif name == "split_dataset_ratio":
            valid = type(actual) in {int, float} and math.isfinite(actual) and actual == 0
        else:
            valid = type(actual) is type(required) and actual == required
        if not valid:
            raise ValueError(f"{name} must remain {required!r} for the {recipe['method']} contract")
    for name, (minimum, maximum) in INTEGER_RANGES.items():
        if name in parameters and (type(parameters[name]) is not int or not minimum <= parameters[name] <= maximum):
            raise ValueError(f"{name} must be an integer in [{minimum},{maximum}] (bool is not accepted)")
    for name, (minimum, maximum, include_minimum) in FLOAT_RANGES.items():
        if name not in parameters:
            continue
        value = parameters[name]
        if type(value) not in {int, float} or not math.isfinite(value) or value > maximum or (value < minimum if include_minimum else value <= minimum):
            raise ValueError(f"{name} must be finite and in its supported numeric range")
        if name in {"warmup_ratio", "lora_dropout"} and value == 1:
            raise ValueError(f"{name} must be less than 1")
    for name in BOOL_PARAMETERS:
        if name in parameters and type(parameters[name]) is not bool:
            raise ValueError(f"{name} must be a JSON boolean")
    for name, choices in CHOICE_PARAMETERS.items():
        if name in parameters and (not isinstance(parameters[name], str) or parameters[name] not in choices):
            raise ValueError(f"{name} must be one of {sorted(choices)}")
    if "report_to" in parameters and parameters["report_to"] != ["none"]:
        raise ValueError("report_to must be ['none']; remote experiment reporting is outside this recipe")
    if parameters.get("save_strategy") == "best":
        if (parameters.get("eval_strategy") != "steps" or not parameters.get("eval_steps")
                or parameters.get("metric_for_best_model") != "eval_loss"
                or parameters.get("greater_is_better") is not False
                or parameters.get("load_best_model_at_end") is not True
                or any(parameters.get(k) is not False for k in ("add_version", "save_only_model", "use_flash_ckpt"))):
            raise ValueError("best requires step eval, eval_loss minimization and load_best_model_at_end")
    # Reject non-JSON/nonfinite metadata even if Python called us directly.
    json.dumps(recipe, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON number is not accepted: {value}")

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def load_recipe(path: str | Path) -> dict:
    recipe = _load_json(Path(path))
    validate_recipe(recipe)
    return recipe


def recipe_sha256(recipe: dict) -> str:
    validate_recipe(recipe)
    return hashlib.sha256(json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_native_metadata(config: dict, tokenizer_config: dict, tokenizer: dict) -> None:
    """Prevent the earlier Qwen reference tokenizer from masquerading as native."""
    config = _object(config, "native config.json")
    text = _object(config.get("text_config"), "native config.text_config")
    if config.get("model_type") != "gemma4" or text.get("model_type") != "gemma4_text":
        raise ValueError("native config must describe gemma4/gemma4_text, not a reference tokenizer's model")
    if "Gemma4ForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("native config has an unexpected Gemma 4 architecture")
    if text.get("max_position_embeddings") != MODEL_MAX_LENGTH or text.get("vocab_size") != MODEL_VOCAB_SIZE:
        raise ValueError("native Gemma E2B context/vocabulary metadata differs from the verified model contract")
    for name, expected in SPECIAL_TOKEN_IDS.items():
        if type(text.get(name)) is not int or text[name] != expected:
            raise ValueError(f"native config {name} must be {expected}")
    tokenizer_config = _object(tokenizer_config, "native tokenizer_config.json")
    if tokenizer_config.get("tokenizer_class") != "GemmaTokenizer" or tokenizer_config.get("processor_class") != "Gemma4Processor":
        raise ValueError("native tokenizer metadata must identify GemmaTokenizer/Gemma4Processor")
    if tokenizer_config.get("add_bos_token") is not True:
        raise ValueError("native tokenizer must preserve its declared add_bos_token=true behavior")
    for name, expected in SPECIAL_TOKEN_TEXT.items():
        if tokenizer_config.get(name) != expected:
            raise ValueError(f"native tokenizer {name} must be {expected!r}")
    tokenizer = _object(tokenizer, "native tokenizer.json")
    model = _object(tokenizer.get("model"), "native tokenizer.model")
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        items = list(vocab.items())
    elif isinstance(vocab, list) and all(isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) for pair in vocab):
        items = [(pair[0], index) for index, pair in enumerate(vocab)]
    else:
        raise ValueError("native tokenizer.json must contain a complete vocabulary, not a diagnostic/placeholder file")
    ids = {}
    for token, tid in items:
        if not isinstance(token, str) or type(tid) is not int or tid < 0:
            raise ValueError("native tokenizer vocabulary entries must map text to nonnegative integer IDs")
        if tid in ids and ids[tid] != token:
            raise ValueError("native tokenizer vocabulary contains conflicting token IDs")
        ids[tid] = token
    for item in tokenizer.get("added_tokens", []):
        if not isinstance(item, dict) or type(item.get("id")) is not int or not isinstance(item.get("content"), str):
            raise ValueError("native added_tokens entries have invalid metadata")
        tid, token = item["id"], item["content"]
        if tid in ids and ids[tid] != token:
            raise ValueError("native special token ID conflicts with its vocabulary")
        ids[tid] = token
    if set(ids) != set(range(MODEL_VOCAB_SIZE)):
        raise ValueError("native tokenizer vocabulary is incomplete or differs from Gemma's 262144-token model")
    for name, expected in SPECIAL_TOKEN_IDS.items():
        token = SPECIAL_TOKEN_TEXT[name.removesuffix("_id")]
        if ids.get(expected) != token:
            raise ValueError(f"native tokenizer ID {expected} must identify {token!r}")


def validate_model_lock(lock: dict, directory: Path | None = None) -> None:
    """Validate immutable origin and inert tokenizer/config files only.

    Repository+revision identifies the weight source. The listed local files are
    not a weight snapshot, and successful validation does not claim weights were
    downloaded, training works, or an adapter has been applied for inference.
    """
    lock = _object(lock, "model lock")
    required = {"schema_version", "repository", "revision", "files"}
    _keys(lock, required, required | {"model_metadata"}, "model lock")
    if lock["schema_version"] != MODEL_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported model lock schema_version")
    if lock["repository"] != MODEL_REPO:
        raise ValueError("model lock must identify google/gemma-4-E2B Base; training/inference sources cannot silently change")
    _revision(lock["revision"], "model lock revision")
    files = _object(lock["files"], "model lock files")
    if not REQUIRED_MODEL_FILES <= set(files):
        raise ValueError("model lock requires config.json, tokenizer.json and tokenizer_config.json")
    for filename, metadata in files.items():
        if not isinstance(filename, str) or filename not in ALLOWED_MODEL_FILES:
            raise ValueError(f"model lock contains a forbidden/non-tokenizer file: {filename!r}; weights, scripts and paths are prohibited")
        metadata = _object(metadata, f"model lock file {filename}")
        _keys(metadata, {"sha256", "bytes"}, {"sha256", "bytes"}, f"model lock file {filename}")
        if not isinstance(metadata["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]):
            raise ValueError(f"{filename}: sha256 must be 64 lowercase hex characters")
        if type(metadata["bytes"]) is not int or not 0 < metadata["bytes"] <= MAX_METADATA_FILE_BYTES:
            raise ValueError(f"{filename}: bytes must be a positive integer no larger than 128MiB")
    if "model_metadata" in lock:
        _object(lock["model_metadata"], "model_metadata")
    json.dumps(lock, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if directory is None:
        return
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model lock directory must be a directory")
    for filename, metadata in files.items():
        path = (root / filename).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"model file resolves outside the locked directory or is not a file: {filename}")
        if path.stat().st_size != metadata["bytes"] or _file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"model lock hash/size mismatch: {filename}")
    _validate_native_metadata(_load_json(root / "config.json"), _load_json(root / "tokenizer_config.json"), _load_json(root / "tokenizer.json"))
