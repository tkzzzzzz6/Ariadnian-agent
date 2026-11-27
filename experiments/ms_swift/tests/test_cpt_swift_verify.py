"""Weight-free SWIFT contract tests; these are not real SWIFT integration runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from cpt import swift_verify
from cpt.schema import default_recipe


OFFLINE_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _dataset(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _contract(tmp_path, monkeypatch):
    recipe = default_recipe()
    assets = tmp_path / "metadata"
    assets.mkdir()
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        path = _write(assets / name, {"processor_class": "Gemma4Processor"} if name == "processor_config.json" else {})
        files[name] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    lock = {"schema_version": "cpt-model-lock-v1", "repository": recipe["model"]["repo"],
            "revision": recipe["model"]["revision"], "files": files}
    _write(assets / "model.lock.json", lock)
    # Deliberately present but unlocked. The upstream processor must not see
    # weights, executable local plugins, or unrelated training arguments.
    (assets / "model.safetensors").write_bytes(b"synthetic sentinel, not model weights")
    (assets / "unlisted.py").write_text("# never imported\n", encoding="utf-8")
    _write(assets / "args.json", {"unlisted": True})
    paths = {
        "train": _dataset(tmp_path / "train.jsonl", [{"text": "First original paragraph.\n\nSecond paragraph."},
                                                      {"text": "中文原文：保留空格  与换行\n下一行。"}]),
        "dev": _dataset(tmp_path / "dev.jsonl", [{"text": "Held out original text."}]),
    }
    plan = {"schema_version": "cpt-launch-plan-v1", "recipe": recipe,
            "model_assets": str(assets.resolve()), "model_lock_sha256": _hash(assets / "model.lock.json"),
            "datasets": {split: str(path.resolve()) for split, path in paths.items()},
            "dataset_sha256": {split: _hash(path) for split, path in paths.items()}}
    observed = {"backend_loads": 0, "processor_calls": [], "template_calls": [],
                "encoded_messages": [], "preprocessed_texts": [], "set_mode": []}

    class Gemma4Processor:
        pass

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [ord(character) + 10 for character in text]

    class Template:
        tokenizer = Tokenizer()

        def set_mode(self, mode):
            observed["set_mode"].append(mode)

        def encode(self, row, *, return_length):
            assert return_length is True and observed["set_mode"][-1] == "train"
            observed["encoded_messages"].append(copy.deepcopy(row["messages"]))
            body = self.tokenizer.encode(row["messages"][0]["content"], add_special_tokens=False)
            ids = [2, *body, 1]
            encoded = {"input_ids": ids, "labels": [-100, *ids[1:]]}
            mutation = observed.get("encoded_mutation")
            return mutation(encoded) if mutation else encoded

    def get_processor(model_dir, **kwargs):
        assert all(os.environ.get(key) == "1" for key in OFFLINE_KEYS)
        isolated = Path(model_dir)
        assert isolated.resolve() != assets.resolve()
        assert {path.name for path in isolated.iterdir()} == set(files)
        assert all(_hash(isolated / name) == spec["sha256"] for name, spec in files.items())
        observed["isolated"] = isolated
        observed["processor_calls"].append(kwargs)
        return Gemma4Processor()

    def get_template(processor, **kwargs):
        assert processor.__class__.__name__ == "Gemma4Processor"
        observed["template_calls"].append(kwargs)
        return Template()

    def preprocess(records):
        observed["preprocessed_texts"].extend(row["text"] for row in records)
        processed = [{"messages": [{"role": "assistant", "content": row["text"]}]} for row in records]
        mutation = observed.get("processed_mutation")
        return mutation(processed) if mutation else processed

    def load_backend():
        assert all(os.environ.get(key) == "1" for key in OFFLINE_KEYS)
        observed["backend_loads"] += 1
        return get_processor, get_template, preprocess

    # Full asset validation has separate schema tests. These tiny inert fixtures
    # intentionally do not pretend to be a real 262K-vocabulary tokenizer.
    monkeypatch.setattr(swift_verify, "validate_model_lock", lambda lock, directory: None)
    monkeypatch.setattr(swift_verify, "_runtime_provenance", lambda: {"contract_mock": True})
    monkeypatch.setattr(swift_verify, "_load_backend", load_backend)
    return plan, observed, paths, assets


def test_verified_pt_mapping_preserves_every_original_record_and_disables_model_download(tmp_path, monkeypatch):
    plan, observed, paths, _ = _contract(tmp_path, monkeypatch)
    result = swift_verify.verify_template(plan)

    originals = [json.loads(line)["text"] for split in ("train", "dev")
                 for line in paths[split].read_text(encoding="utf-8").splitlines()]
    assert observed["preprocessed_texts"] == originals
    assert observed["encoded_messages"] == [[{"role": "assistant", "content": text}] for text in originals]
    assert observed["processor_calls"] == [{"model_type": "gemma4", "task_type": "causal_lm", "use_hf": True,
                                             "revision": plan["recipe"]["model"]["revision"], "download_model": False}]
    assert observed["set_mode"] == ["train"]
    options = observed["template_calls"][0]
    assert options["use_chat_template"] is False and options["loss_scale"] == "all"
    assert options["truncation_strategy"] == "raise" and options["padding_free"] is False
    assert options["template_backend"] == "swift"
    assert result["verified"] is True and result["weights_loaded"] is False
    assert result["runtime"] == {"contract_mock": True}
    assert result["datasets"]["train"]["records"] == result["datasets"]["train"]["encoded_records"] == 2
    assert result["datasets"]["dev"]["dropped_records"] == 0
    assert result["datasets"]["train"]["added_template_tokens"] == 4
    assert result["datasets"]["train"]["loss_tokens"] == result["datasets"]["train"]["input_tokens"] - 2
    assert not observed["isolated"].exists()


@pytest.mark.parametrize("target", ["train", "dev", "model_lock"])
def test_changed_bound_hash_is_rejected_before_loading_any_swift_backend(tmp_path, monkeypatch, target):
    plan, observed, paths, assets = _contract(tmp_path, monkeypatch)
    changed = assets / "model.lock.json" if target == "model_lock" else paths[target]
    changed.write_text(changed.read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setattr(swift_verify, "_runtime_provenance", lambda: pytest.fail("must reject before runtime loading"))

    with pytest.raises(ValueError, match="SHA-256"):
        swift_verify.verify_template(plan)
    assert observed["backend_loads"] == 0


@pytest.mark.parametrize("kind", ["drop", "append", "alter_text", "user_role"])
def test_upstream_preprocessing_cannot_drop_rewrite_or_relabel_original_documents(tmp_path, monkeypatch, kind):
    plan, observed, _, _ = _contract(tmp_path, monkeypatch)

    def mutate(rows):
        if kind == "drop":
            return rows[:-1]
        if kind == "append":
            return [*rows, copy.deepcopy(rows[0])]
        if kind == "alter_text":
            rows[0]["messages"][0]["content"] += " changed"
        else:
            rows[0]["messages"][0]["role"] = "user"
        return rows

    observed["processed_mutation"] = mutate
    with pytest.raises(ValueError, match="preprocessing"):
        swift_verify.verify_template(plan)
    assert not observed["encoded_messages"]


@pytest.mark.parametrize("kind", ["overflow", "empty_labels", "empty_ids", "sft_mask", "no_supervision",
                                  "misaligned", "lost_body", "invalid_vocabulary", "weighted_loss"])
def test_template_failures_never_receive_a_verified_receipt(tmp_path, monkeypatch, kind):
    plan, observed, _, _ = _contract(tmp_path, monkeypatch)
    limit = plan["recipe"]["parameters"]["max_length"]

    def mutate(encoded):
        if kind == "overflow":
            encoded = {"input_ids": [2] * (limit + 1), "labels": [-100, *([2] * limit)]}
        elif kind == "empty_labels":
            encoded["labels"] = []
        elif kind == "empty_ids":
            encoded["input_ids"] = []
        elif kind == "sft_mask":
            encoded["labels"][1] = -100
        elif kind == "no_supervision":
            encoded["labels"] = [-100] * len(encoded["input_ids"])
        elif kind == "misaligned":
            encoded["labels"].pop()
        elif kind == "lost_body":
            encoded["input_ids"][2] = 9
            encoded["labels"][2] = 9
        elif kind == "invalid_vocabulary":
            encoded["input_ids"][-1] = -1
            encoded["labels"][-1] = -1
        elif kind == "weighted_loss":
            encoded["loss_scale"] = [1] * len(encoded["input_ids"])
            encoded["loss_scale"][1] = 0.5
        return encoded

    observed["encoded_mutation"] = mutate
    with pytest.raises(ValueError):
        swift_verify.verify_template(plan)


@pytest.mark.parametrize("fail", [False, True])
def test_offline_environment_and_isolated_metadata_are_restored_after_success_or_failure(tmp_path, monkeypatch, fail):
    plan, observed, _, _ = _contract(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous-hf-value")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("MODELSCOPE_OFFLINE", "previous-ms-value")
    before = {name: os.environ.get(name) for name in OFFLINE_KEYS}
    if fail:
        observed["encoded_mutation"] = lambda encoded: {"input_ids": encoded["input_ids"], "labels": []}
        with pytest.raises(ValueError):
            swift_verify.verify_template(plan)
    else:
        swift_verify.verify_template(plan)
    assert {name: os.environ.get(name) for name in OFFLINE_KEYS} == before
    assert not observed["isolated"].exists()

