"""Load a local tokenizer without downloading code or model artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable


def load_tokenizer(path: str) -> tuple[Callable[[str], int], dict]:
    source = Path(path).resolve()
    if not source.exists():
        raise ValueError(f"tokenizer path is not local or does not exist: {source}")
    if source.is_file():
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ValueError("Loading tokenizer.json requires the optional tokenizers package") from exc
        try:
            tokenizer = Tokenizer.from_file(str(source))
        except Exception as exc:
            raise ValueError(f"cannot load local tokenizer JSON: {exc}") from exc
        tokenizer.no_truncation()
        tokenizer.no_padding()
        count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
        hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ValueError("Loading a tokenizer directory requires the optional Transformers package") from exc
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        count = lambda text: len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file() and p.suffix in {".json", ".txt", ".model"}}
    return count, {"path": str(source), "artifact_sha256": hashes, "add_special_tokens": False, "network_downloads": False}
