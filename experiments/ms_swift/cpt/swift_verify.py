"""Validate every prepared PT/SFT record with the pinned SWIFT template, without weights.

This module has no training-stack imports at import time. Run it in the dedicated
environment from experiments/ms_swift/requirements-train.txt, not backend/.venv.
The API is verified against MS-SWIFT v4.6.0 / SWIFT_COMMIT in schema.py:
swift.get_processor(..., download_model=False) wraps load_model=False;
swift.get_template(...).set_mode('train') performs the actual PT encoding.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .schema import (
    MODEL_VOCAB_SIZE,
    SWIFT_COMMIT,
    SWIFT_VERSION,
    validate_model_lock,
    validate_recipe,
)

_CORE_VERSIONS = {
    "ms-swift": SWIFT_VERSION,
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchaudio": "2.8.0",
    "transformers": "5.12.1",
    "peft": "0.20.0",
    "accelerate": "1.12.0",
    "datasets": "4.8.4",
    "trl": "0.29.1",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def _absolute_file(value, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute local file path")
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} must identify a local file")
    return path


def _read_records(path: Path, *, sft=False) -> list[dict]:
    records = []
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number}: blank JSONL records are not permitted")
        try:
            row = _json(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON record") from exc
        if sft:
            from .sft import validate_messages
            if not isinstance(row, dict) or set(row) != {"messages"}:
                raise ValueError(f"{path.name}:{line_number}: SFT requires messages-only records")
            validate_messages(row["messages"])
            records.append(row)
            continue
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{path.name}:{line_number}: a nonempty original text field is required")
        if any(key in row for key in ("messages", "response", "query", "system", "input_ids", "labels")):
            raise ValueError(f"{path.name}:{line_number}: mixed raw/encoded/chat input is not permitted")
        if "id" in row:
            if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
                raise ValueError(f"{path.name}:{line_number}: invalid or duplicate record id")
            ids.add(row["id"])
        records.append(row)
    if not records:
        raise ValueError(f"{path.name}: the dataset has no records")
    return records


@contextmanager
def _offline_environment():
    """Keep temporary offline settings local to this verification call."""
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime_provenance() -> dict:
    versions = {}
    for package, expected in _CORE_VERSIONS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"missing training dependency {package}; use the dedicated SWIFT environment") from exc
        # CUDA wheel local suffixes (e.g. 2.8.0+cu128) are legitimate.
        if actual.split("+", 1)[0] != expected:
            raise ValueError(f"{package} must be {expected}; found {actual}")
        versions[package] = actual
    distribution = metadata.distribution("ms-swift")
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise ValueError("MS-SWIFT source commit is unverified; install the pinned Git requirement")
    direct_url = _json(direct_url_text)
    if not isinstance(direct_url, dict):
        raise ValueError("MS-SWIFT direct_url.json must be an object")
    vcs = direct_url.get("vcs_info")
    origin = str(direct_url.get("url", "")).removesuffix(".git").rstrip("/")
    if origin != "https://github.com/modelscope/ms-swift" or not isinstance(vcs, dict) or vcs.get("commit_id") != SWIFT_COMMIT:
        raise ValueError("MS-SWIFT must originate from the fixed upstream repository and commit")
    return {"versions": versions, "swift_commit": SWIFT_COMMIT, "swift_origin": origin}


def _load_backend():
    """Return the real pinned APIs; isolated for dependency-free contract tests."""
    from datasets import Dataset
    from swift import get_processor, get_template
    from swift.dataset.preprocessor import ResponsePreprocessor, MessagesPreprocessor

    def preprocess(records):
        # Use the upstream column mapping and preprocessor, rather than manually
        # synthesizing messages or assuming special-token insertion rules.
        if "messages" in records[0]:
            dataset = Dataset.from_list(records)
            processor = MessagesPreprocessor()
        else:
            dataset = Dataset.from_list([{"text": row["text"]} for row in records])
            processor = ResponsePreprocessor(columns={"text": "response"})
        return processor(dataset, strict=True, num_proc=1, load_from_cache_file=False)

    return get_processor, get_template, preprocess


def _integer_list(value, name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"template {name} must be a nonempty one-dimensional integer sequence")
    return value


def _encode_split(path: Path, records: list[dict], template, preprocess, max_length: int) -> dict:
    processed = preprocess(records)
    if len(processed) != len(records):
        raise ValueError(f"{path.name}: upstream preprocessing dropped or added records")
    encoded_digest = hashlib.sha256()
    loss_digest = hashlib.sha256()
    body_digest = hashlib.sha256()
    loss_counts = Counter()
    lengths = []
    loss_lengths = []
    body_lengths = []
    for index, original in enumerate(records):
        row = processed[index]
        expected_messages = [{"role": "assistant", "content": original["text"]}]
        if not isinstance(row, dict) or row.get("messages") != expected_messages:
            raise ValueError(f"{path.name}:{index + 1}: text-to-response preprocessing changed the original text or role")
        try:
            encoded = template.encode(row, return_length=True)
        except Exception as exc:
            raise ValueError(f"{path.name}:{index + 1}: strict SWIFT PT template encoding failed") from exc
        if not isinstance(encoded, dict):
            raise ValueError(f"{path.name}:{index + 1}: template dropped or split a record")
        input_ids = _integer_list(encoded.get("input_ids"), "input_ids")
        labels = _integer_list(encoded.get("labels"), "labels")
        if len(input_ids) != len(labels) or len(input_ids) > max_length:
            raise ValueError(f"{path.name}:{index + 1}: mismatched labels or overlength encoded sequence")
        if any(token < 0 or token >= MODEL_VOCAB_SIZE for token in input_ids):
            raise ValueError(f"{path.name}:{index + 1}: input token is outside the locked model vocabulary")
        body_ids = _integer_list(template.tokenizer.encode(original["text"], add_special_tokens=False), "body input_ids")
        body_offset = next((offset for offset in range(len(input_ids) - len(body_ids) + 1)
                            if input_ids[offset:offset + len(body_ids)] == body_ids), None)
        if body_offset is None:
            raise ValueError(f"{path.name}:{index + 1}: the template encoding lost or changed original body tokens")
        # PT/all must supervise every token after the first. This also rejects
        # accidental answer-only SFT masks or a record with empty supervision.
        if labels[0] != -100 or len(labels) < 2 or any(label != token for label, token in zip(labels[1:], input_ids[1:])):
            raise ValueError(f"{path.name}:{index + 1}: labels do not implement the full-text PT/all loss contract")
        if encoded.get("loss_scale") is not None:
            scales = encoded["loss_scale"]
            if hasattr(scales, "tolist"):
                scales = scales.tolist()
            if not isinstance(scales, list) or len(scales) != len(labels) or any(scale != 1 for scale in scales[1:]):
                raise ValueError(f"{path.name}:{index + 1}: unexpected PT loss weights")
        loss_ids = labels[1:]
        lengths.append(len(input_ids))
        loss_lengths.append(len(loss_ids))
        body_lengths.append(len(body_ids))
        encoded_digest.update(json.dumps([input_ids, labels], separators=(",", ":")).encode("ascii") + b"\n")
        loss_digest.update(json.dumps(loss_ids, separators=(",", ":")).encode("ascii") + b"\n")
        body_digest.update(json.dumps(body_ids, separators=(",", ":")).encode("ascii") + b"\n")
        loss_counts.update(loss_ids)
    histogram = [[token, count] for token, count in sorted(loss_counts.items())]
    return {
        "path": str(path), "sha256": _sha256(path), "records": len(records), "encoded_records": len(lengths),
        "dropped_records": 0, "input_tokens": sum(lengths), "loss_tokens": sum(loss_lengths),
        "body_tokens": sum(body_lengths), "added_template_tokens": sum(lengths) - sum(body_lengths),
        "min_input_length": min(lengths), "max_input_length": max(lengths),
        "input_and_labels_sha256": encoded_digest.hexdigest(), "ordered_loss_token_ids_sha256": loss_digest.hexdigest(),
        "ordered_body_token_ids_sha256": body_digest.hexdigest(),
        "loss_token_histogram_sha256": hashlib.sha256(json.dumps(histogram, separators=(",", ":")).encode("ascii")).hexdigest(),
    }


def _sft_expected_ids(messages, tokenizer):
    """Independent expected mask for the pinned single-turn Gemma4 template.

    This verifies the real SWIFT output, not a replacement training encoder.
    Upstream Gemma4TemplateMeta and default loss scaling merge the masked prompt
    and the supervised answer+suffix as two separately tokenized contexts.
    """
    prefix = "<bos>"
    if messages[0]["role"] == "system":
        prefix += "<|turn>system\n" + messages[0]["content"] + "<turn|>\n"
    prefix += "<|turn>user\n" + messages[-2]["content"] + "<turn|>\n<|turn>model\n"
    response = messages[-1]["content"]
    if not response.endswith(("<eos>", "<turn|>", "<|tool_response>")):
        response += "<turn|>\n"
    prompt_ids = _integer_list(tokenizer.encode(prefix, add_special_tokens=False), "SFT prompt input_ids")
    response_ids = _integer_list(tokenizer.encode(response, add_special_tokens=False), "SFT response input_ids")
    return prompt_ids, response_ids


def _encode_sft_split(path, records, template, preprocess, max_length):
    processed = preprocess(records)
    if len(processed) != len(records):
        raise ValueError(f"{path.name}: upstream SFT preprocessing dropped or added records")
    encoded_digest, loss_digest = hashlib.sha256(), hashlib.sha256()
    loss_counts, lengths, loss_lengths = Counter(), [], []
    for index, original in enumerate(records):
        row = processed[index]
        if not isinstance(row, dict) or row.get("messages") != original["messages"]:
            raise ValueError(f"{path.name}:{index + 1}: SFT preprocessing changed original messages")
        try:
            encoded = template.encode(row, return_length=True)
        except Exception as exc:
            raise ValueError(f"{path.name}:{index + 1}: strict SWIFT SFT template encoding failed") from exc
        if not isinstance(encoded, dict):
            raise ValueError(f"{path.name}:{index + 1}: SFT template dropped or split a record")
        ids = _integer_list(encoded.get("input_ids"), "input_ids")
        labels = _integer_list(encoded.get("labels"), "labels")
        if len(ids) != len(labels) or len(ids) > max_length:
            raise ValueError(f"{path.name}:{index + 1}: mismatched labels or overlength SFT sequence")
        if any(token < 0 or token >= MODEL_VOCAB_SIZE for token in ids):
            raise ValueError(f"{path.name}:{index + 1}: SFT input token is outside the locked vocabulary")
        prompt_ids, response_ids = _sft_expected_ids(original["messages"], template.tokenizer)
        if ids != prompt_ids + response_ids:
            raise ValueError(f"{path.name}:{index + 1}: SFT template changed the pinned prompt/answer encoding")
        if labels != [-100] * len(prompt_ids) + response_ids:
            raise ValueError(f"{path.name}:{index + 1}: SFT labels must mask the whole prompt and supervise the answer plus suffix")
        scales = encoded.get("loss_scale")
        if scales is not None:
            if hasattr(scales, "tolist"): scales = scales.tolist()
            if scales != [0] * len(prompt_ids) + [1] * len(response_ids):
                raise ValueError(f"{path.name}:{index + 1}: unexpected SFT loss weights")
        lengths.append(len(ids)); loss_lengths.append(len(response_ids))
        encoded_digest.update(json.dumps([ids, labels], separators=(",", ":")).encode("ascii") + b"\n")
        loss_digest.update(json.dumps(response_ids, separators=(",", ":")).encode("ascii") + b"\n")
        loss_counts.update(response_ids)
    histogram = [[token, count] for token, count in sorted(loss_counts.items())]
    return {"path": str(path), "sha256": _sha256(path), "records": len(records), "encoded_records": len(lengths),
            "dropped_records": 0, "input_tokens": sum(lengths), "loss_tokens": sum(loss_lengths),
            "masked_prompt_tokens": sum(lengths) - sum(loss_lengths),
            "min_input_length": min(lengths), "max_input_length": max(lengths),
            "input_and_labels_sha256": encoded_digest.hexdigest(),
            "ordered_loss_token_ids_sha256": loss_digest.hexdigest(),
            "loss_token_histogram_sha256": hashlib.sha256(json.dumps(histogram, separators=(",", ":")).encode("ascii")).hexdigest(),
            "loss_contract": "assistant answer and turn suffix only; system/user prompt masked"}


def verify_template(plan: dict) -> dict:
    """Verify all train/dev rows of a cpt-launch-plan-v1; never load weights.

    Required plan fields: recipe, absolute model_assets directory (containing
    model.lock.json), datasets={train,dev}, and dataset_sha256={train,dev}.
    Additional launch fields are deliberately ignored. All checks are fail-closed.
    The result establishes template/data compatibility, not training readiness,
    GPU feasibility, training eligibility, cross-arm equality, or model quality.
    """
    if not isinstance(plan, dict) or plan.get("schema_version") != "cpt-launch-plan-v1":
        raise ValueError("verify_template requires a cpt-launch-plan-v1 object")
    recipe = plan.get("recipe")
    validate_recipe(recipe)
    sft = recipe["method"].startswith("sft_")
    datasets, hashes = plan.get("datasets"), plan.get("dataset_sha256")
    if not isinstance(datasets, dict) or set(datasets) != {"train", "dev"} or not isinstance(hashes, dict) or set(hashes) != {"train", "dev"}:
        raise ValueError("plan requires exactly train/dev datasets and corresponding SHA-256 values")
    paths, records = {}, {}
    for split in ("train", "dev"):
        if not isinstance(hashes[split], str) or not _SHA256.fullmatch(hashes[split]):
            raise ValueError(f"{split}: invalid dataset SHA-256")
        paths[split] = _absolute_file(datasets[split], f"datasets.{split}")
        if _sha256(paths[split]) != hashes[split]:
            raise ValueError(f"{split}: dataset SHA-256 mismatch before template loading")
        records[split] = _read_records(paths[split], sft=sft)
    if paths["train"] == paths["dev"]:
        raise ValueError("train and dev datasets must be different files")
    assets_value = plan.get("model_assets")
    if not isinstance(assets_value, (str, Path)) or not Path(assets_value).is_absolute():
        raise ValueError("model_assets must be an absolute local directory")
    assets = Path(assets_value).resolve(strict=True)
    if not assets.is_dir():
        raise ValueError("model_assets must be a directory")
    lock_path = assets / "model.lock.json"
    lock = _json(lock_path.read_text(encoding="utf-8-sig"))
    lock_sha256 = _sha256(lock_path)
    if "model_lock_sha256" in plan and plan["model_lock_sha256"] != lock_sha256:
        raise ValueError("model lock SHA-256 differs from the launch plan")
    validate_model_lock(lock, assets)
    if lock["repository"] != recipe["model"]["repo"] or lock["revision"] != recipe["model"]["revision"]:
        raise ValueError("model lock and recipe refer to different model origins")
    if not {"processor_config.json", "preprocessor_config.json"} & set(lock["files"]):
        raise ValueError("locked Gemma4 processor metadata is missing; tokenizer-only assets cannot verify the multimodal template")
    for filename in lock["files"]:
        if filename.endswith(".json"):
            content = _json((assets / filename).read_text(encoding="utf-8-sig"))
            if isinstance(content, dict) and content.get("auto_map"):
                raise ValueError(f"{filename}: remote-code auto_map is outside the inert metadata contract")
    parameters = recipe["parameters"]
    with _offline_environment(), tempfile.TemporaryDirectory(prefix="ariadne-swift-verify-") as temporary:
        # Only locked inert files enter this directory. Weight files, args.json
        # and any unlisted local code are never visible to the upstream loader.
        isolated = Path(temporary)
        for filename in lock["files"]:
            shutil.copyfile(assets / filename, isolated / filename)
            if _sha256(isolated / filename) != lock["files"][filename]["sha256"]:
                raise ValueError(f"{filename}: metadata changed during isolated copy")
        provenance = _runtime_provenance()
        get_processor, get_template, preprocess = _load_backend()
        processor = get_processor(str(isolated), model_type="gemma4", task_type="causal_lm", use_hf=True,
                                  revision=lock["revision"], download_model=False)
        if processor.__class__.__name__ != "Gemma4Processor":
            raise ValueError("the pinned metadata did not load Gemma4Processor")
        template = get_template(processor, template_type=parameters["template"], max_length=parameters["max_length"],
                                use_chat_template=parameters["use_chat_template"], loss_scale=parameters["loss_scale"],
                                truncation_strategy="raise", padding_free=False,
                                template_backend="swift", enable_thinking=False)
        template.set_mode("train")
        encode_split = _encode_sft_split if sft else _encode_split
        results = {split: encode_split(paths[split], records[split], template, preprocess, parameters["max_length"])
                   for split in ("train", "dev")}
    for split in results:
        if results[split]["sha256"] != hashes[split]:
            raise ValueError(f"{split}: dataset changed during template verification")
    if _sha256(lock_path) != lock_sha256:
        raise ValueError("model lock changed during template verification")
    return {
        "schema_version": "cpt-swift-template-verification-v1", "verified": True,
        "model": dict(recipe["model"]), "model_lock_sha256": lock_sha256, "runtime": provenance,
        "template": parameters["template"], "use_chat_template": parameters["use_chat_template"],
        "loss_scale": parameters["loss_scale"], "mode": "train",
        "max_length": parameters["max_length"], "weights_loaded": False, "offline": True, "datasets": results,
        "limitations": ["Template verification does not establish data training eligibility or training completion.",
                        "No model weights are loaded; GPU kernels, memory usage and training behavior remain untested.",
                        "Cross-arm fairness requires comparing verified source coverage, loss-token identities and budgets separately."],
    }
