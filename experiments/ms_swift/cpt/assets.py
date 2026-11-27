"""Fetch and verify native tokenizer/config bytes, never model weights or code."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .schema import (
    MAX_METADATA_FILE_BYTES, MODEL_LOCK_SCHEMA_VERSION, MODEL_MAX_LENGTH, MODEL_REPO,
    MODEL_REVISION, MODEL_VOCAB_SIZE, REQUIRED_MODEL_FILES, SPECIAL_TOKEN_IDS,
    _file_sha256, _load_json, validate_model_lock,
)

# Verified using the HF tree API at the exact pinned revision. Ordinary file
# oids are Git blob SHA1; an LFS oid is SHA256 of the actual resolved bytes.
PINNED_FILES = {
    "config.json": {"git_blob_sha1": "5f0662eb398ede0ab1515c4760e13bcd6bf13f2f", "bytes": 4914},
    "generation_config.json": {"git_blob_sha1": "a07eb97486e34532ffd3cf64e44cba33af40127c", "bytes": 181},
    "processor_config.json": {"git_blob_sha1": "5465974d23e1eca2c46c2809b26c997946ce0d90", "bytes": 1689},
    "tokenizer.json": {"git_blob_sha1": "98de2092a89a0c7068f5181bebba69cca2afc92b", "bytes": 32157288,
                       "lfs_sha256": "9cad58a1ba9824a35574be51e8c9dcc2164a662cdfe56b1b008e06a231f80f0b"},
    "tokenizer_config.json": {"git_blob_sha1": "d06ea6ad5c6c809c73019583496d0fe63af692d0", "bytes": 906},
}


@dataclass(frozen=True)
class NativeTokenizer:
    count: Callable[[str], int]
    metadata: dict


def _git_blob_sha1(path: Path) -> str:
    hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_source_file(path: Path, expected: dict) -> None:
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"pinned model metadata size mismatch: {path.name}")
    if "lfs_sha256" in expected:
        matches = _file_sha256(path) == expected["lfs_sha256"]
    else:
        matches = _git_blob_sha1(path) == expected["git_blob_sha1"]
    if not matches:
        raise ValueError(f"pinned HF source hash mismatch: {path.name}")


def _processor_metadata(directory: Path) -> dict:
    path = directory / "processor_config.json"
    if not path.is_file():
        raise ValueError("native Gemma4Processor metadata is missing from the asset snapshot")
    config = _load_json(path)
    if not isinstance(config, dict) or config.get("processor_class") != "Gemma4Processor":
        raise ValueError("processor_config must identify Gemma4Processor")
    expected = {"feature_extractor": "feature_extractor_type", "image_processor": "image_processor_type",
                "video_processor": "video_processor_type"}
    for name, type_key in expected.items():
        if not isinstance(config.get(name), dict) or not isinstance(config[name].get(type_key), str):
            raise ValueError(f"native processor inline component is incomplete: {name}")
    references = []

    def scan(value):
        if isinstance(value, dict):
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, str) and value.endswith((".json", ".model", ".txt")):
            references.append(value)

    scan(config)
    missing = [name for name in references if name not in PINNED_FILES or not (directory / name).is_file()]
    if missing:
        raise ValueError("processor references additional unverified metadata files: " + ", ".join(missing))
    return {"processor_class": "Gemma4Processor", "inline_components": sorted(expected),
            "referenced_metadata_files": references,
            "separate_preprocessor_config_in_pinned_snapshot": False}


def fetch_tokenizer(output_dir: str | Path) -> dict:
    """Download only the five known inert files into an absent/empty directory.

    Public HF endpoints are used without reading or exposing local credentials.
    Model metadata/config validation does not instantiate a processor or model.
    """
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("native asset output must be absent or empty; existing assets are never overwritten")
    api_url = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}"
    response = requests.get(api_url, timeout=(15, 60))
    response.raise_for_status()
    info = response.json()
    if not isinstance(info, dict) or info.get("id") != MODEL_REPO or info.get("sha") != MODEL_REVISION:
        raise ValueError("HF API metadata does not match the pinned model repository/revision")
    siblings = {item.get("rfilename") for item in info.get("siblings", []) if isinstance(item, dict)}
    if not REQUIRED_MODEL_FILES <= siblings or not set(PINNED_FILES) <= siblings:
        raise ValueError("HF pinned snapshot is missing required tokenizer/processor metadata")
    unexpected = [name for name in siblings if isinstance(name, str) and name.endswith(".json") and name not in PINNED_FILES]
    if unexpected:
        raise ValueError("pinned snapshot has additional unreviewed config dependencies: " + ", ".join(sorted(unexpected)))
    output.mkdir(parents=True, exist_ok=True)
    source_files = {}
    for filename, expected in PINNED_FILES.items():
        if not 0 < expected["bytes"] <= MAX_METADATA_FILE_BYTES:
            raise ValueError("model metadata exceeds the explicit inert-file size bound")
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}"
        target = output / filename
        temporary = output / (filename + ".part")
        try:
            with requests.get(url, timeout=(15, 180), stream=True) as response:
                response.raise_for_status()
                count = 0
                with temporary.open("xb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if count > expected["bytes"]:
                            raise ValueError(f"model metadata download exceeds pinned size: {filename}")
                        stream.write(chunk)
            _verify_source_file(temporary, expected)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        source_files[filename] = {"url": url, **expected}
    files = {name: {"sha256": _file_sha256(output / name), "bytes": (output / name).stat().st_size}
             for name in PINNED_FILES}
    lock = {
        "schema_version": MODEL_LOCK_SCHEMA_VERSION, "repository": MODEL_REPO,
        "revision": MODEL_REVISION, "files": files,
        "model_metadata": {
            "verified_hf_api_url": api_url, "verified_api_revision": info["sha"],
            "source_file_verification": source_files,
            "native_context_tokens": MODEL_MAX_LENGTH, "vocab_size": MODEL_VOCAB_SIZE,
            "special_token_ids": dict(SPECIAL_TOKEN_IDS), "processor": _processor_metadata(output),
            "scope": "native tokenizer and inert configs only; no weight download or runtime training verification",
        },
    }
    validate_model_lock(lock, output)
    (output / "model.lock.json").write_text(json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
    return lock


def load_native_tokenizer(path: str | Path) -> NativeTokenizer:
    """Load only a source-verified native snapshot, not a standalone reference JSON."""
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError("native tokenizer requires the complete locked asset directory, not a standalone tokenizer.json")
    lock_path = directory / "model.lock.json"
    lock = _load_json(lock_path)
    validate_model_lock(lock, directory)
    if lock["revision"] != MODEL_REVISION or set(lock["files"]) != set(PINNED_FILES):
        raise ValueError("native asset loader accepts only the complete source-verified pinned snapshot")
    for filename, expected in PINNED_FILES.items():
        _verify_source_file(directory / filename, expected)
    processor = _processor_metadata(directory)
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("native tokenizer counting requires the tokenizers package; Torch/Transformers are not needed") from exc
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count(text: str) -> int:
        if not isinstance(text, str):
            raise ValueError("native tokenizer input must be text")
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    metadata = {
        "kind": "native-model-tokenizer", "repository": MODEL_REPO, "revision": MODEL_REVISION,
        "unit": "gemma4_native_tokens", "native_context_tokens": MODEL_MAX_LENGTH,
        "vocab_size": MODEL_VOCAB_SIZE, "bos_token_id": SPECIAL_TOKEN_IDS["bos_token_id"],
        "eos_token_id": SPECIAL_TOKEN_IDS["eos_token_id"], "pad_token_id": SPECIAL_TOKEN_IDS["pad_token_id"],
        "model_lock_sha256": _file_sha256(lock_path), "tokenizer_sha256": lock["files"]["tokenizer.json"]["sha256"],
        "config_sha256": lock["files"]["config.json"]["sha256"],
        "tokenizer_config_sha256": lock["files"]["tokenizer_config.json"]["sha256"],
        "asset_directory": str(directory.resolve()), "processor_metadata": processor,
        "add_special_tokens": False, "truncation": False, "padding": False,
        "special_tokens_budget": 2,
        "special_tokens_budget_note": "Conservative export allowance, not a claim that every SWIFT PT template adds exactly two tokens; check actual runtime encoding.",
        "scope": "source/native tokenizer verified; no model weights loaded or training/inference runtime verified",
    }
    return NativeTokenizer(count, metadata)
