"""Native-asset download/lock tests use explicitly mocked, authored snapshots."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cpt import assets, schema  # noqa: E402


@pytest.fixture(scope="module")
def fixture_files():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokens = {0: "<pad>", 1: "<eos>", 2: "<bos>", 3: "<unk>"}
    vocabulary = {tokens.get(i, f"t{i}"): i for i in range(schema.MODEL_VOCAB_SIZE)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.enable_truncation(max_length=1)
    tokenizer.enable_padding(length=20)
    config = {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
              "text_config": {"model_type": "gemma4_text", "max_position_embeddings": 131072,
                              "vocab_size": 262144, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0}}
    tokenizer_config = {"tokenizer_class": "GemmaTokenizer", "processor_class": "Gemma4Processor",
                        "add_bos_token": True, "model_max_length": 10**30, **schema.SPECIAL_TOKEN_TEXT}
    processor = {"processor_class": "Gemma4Processor", "feature_extractor": {"feature_extractor_type": "Gemma4AudioFeatureExtractor"},
                 "image_processor": {"image_processor_type": "Gemma4ImageProcessor"},
                 "video_processor": {"video_processor_type": "Gemma4VideoProcessor"}}
    encode = lambda value: json.dumps(value).encode()
    return {"config.json": encode(config), "tokenizer_config.json": encode(tokenizer_config),
            "tokenizer.json": tokenizer.to_str().encode(), "processor_config.json": encode(processor),
            "generation_config.json": encode({"bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0})}


@pytest.fixture
def fake_hf(fixture_files, monkeypatch):
    calls = []
    pins = {}
    for name, contents in fixture_files.items():
        pin = {"bytes": len(contents), "git_blob_sha1": hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()}
        if name == "tokenizer.json":
            pin["lfs_sha256"] = hashlib.sha256(contents).hexdigest()
        pins[name] = pin
    monkeypatch.setattr(assets, "PINNED_FILES", pins)

    class Response:
        def __init__(self, *, value=None, contents=b""):
            self.value, self.contents = value, contents
        def raise_for_status(self):
            pass
        def json(self):
            return self.value
        def iter_content(self, size):
            for i in range(0, len(self.contents), size):
                yield self.contents[i:i + size]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def get(url, **kwargs):
        calls.append((url, kwargs))
        assert "headers" not in kwargs and "auth" not in kwargs
        if "/api/models/" in url:
            return Response(value={"id": schema.MODEL_REPO, "sha": schema.MODEL_REVISION,
                                   "siblings": [{"rfilename": name} for name in fixture_files]
                                               + [{"rfilename": "model.safetensors"}, {"rfilename": "modeling_remote.py"}]})
        filename = url.rsplit("/", 1)[-1]
        assert filename in fixture_files, "Never request weights, code, arbitrary files or a different model"
        assert f"/{schema.MODEL_REVISION}/" in url
        return Response(contents=fixture_files[filename])

    monkeypatch.setattr(assets.requests, "get", get)
    return calls, get, Response


def test_fetch_has_only_inert_files_and_load_disables_silent_tokenizer_limits(tmp_path, fake_hf):
    calls, _, _ = fake_hf
    output = tmp_path / "native"
    lock = assets.fetch_tokenizer(output)
    assert set(lock["files"]) == set(assets.PINNED_FILES)
    assert {p.name for p in output.iterdir()} == set(assets.PINNED_FILES) | {"model.lock.json"}
    assert all(not url.endswith((".safetensors", ".py", ".bin")) for url, _ in calls)
    native = assets.load_native_tokenizer(output)
    assert native.count("t4 t5 t6") == 3  # Saved fixture had truncation=1 and padding=20.
    assert native.count("") == 0 and native.metadata["add_special_tokens"] is False
    assert native.metadata["special_tokens_budget"] == 2
    assert native.metadata["native_context_tokens"] == 131072
    assert native.metadata["repository"] == schema.MODEL_REPO
    assert native.metadata["revision"] == schema.MODEL_REVISION
    assert native.metadata["model_lock_sha256"] == hashlib.sha256((output / "model.lock.json").read_bytes()).hexdigest()
    assert native.metadata["processor_metadata"]["inline_components"] == ["feature_extractor", "image_processor", "video_processor"]


def test_nonempty_output_never_overwrites_or_contacts_hf(tmp_path, fake_hf):
    calls, _, _ = fake_hf
    tmp_path.joinpath("existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="never overwritten"):
        assets.fetch_tokenizer(tmp_path)
    assert not calls and tmp_path.joinpath("existing.txt").read_text() == "keep"


def test_download_hash_mismatch_never_creates_a_verified_lock(tmp_path, fake_hf, monkeypatch):
    _, get, Response = fake_hf
    def tampered(url, **kwargs):
        if url.endswith("config.json") and not url.endswith(("processor_config.json", "generation_config.json", "tokenizer_config.json")):
            response = get(url, **kwargs)
            return Response(contents=response.contents.replace(b"gemma4", b"qwen42", 1))
        return get(url, **kwargs)
    monkeypatch.setattr(assets.requests, "get", tampered)
    with pytest.raises(ValueError, match="hash mismatch"):
        assets.fetch_tokenizer(tmp_path / "failed")
    assert not (tmp_path / "failed/model.lock.json").exists()
    assert not list((tmp_path / "failed").glob("*.part"))


def test_forged_new_file_hash_still_cannot_change_the_pinned_origin(tmp_path, fake_hf):
    output = tmp_path / "native"
    assets.fetch_tokenizer(output)
    file = output / "tokenizer.json"
    file.write_bytes(file.read_bytes().replace(b'"t4":4', b'"r4":4', 1))
    lock = json.loads((output / "model.lock.json").read_text())
    lock["files"]["tokenizer.json"] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}
    (output / "model.lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="pinned HF source hash"):
        assets.load_native_tokenizer(output)


def test_missing_processor_dependency_and_standalone_reference_are_rejected(tmp_path, fake_hf):
    reference = tmp_path / "tokenizer.json"
    reference.write_text('{}')
    with pytest.raises(ValueError, match="complete locked asset directory"):
        assets.load_native_tokenizer(reference)
    output = tmp_path / "native"
    lock = assets.fetch_tokenizer(output)
    del lock["files"]["processor_config.json"]
    (output / "model.lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="complete source-verified"):
        assets.load_native_tokenizer(output)


def test_wrong_api_revision_is_rejected_before_any_file_download(tmp_path, fake_hf, monkeypatch):
    calls, _, Response = fake_hf
    monkeypatch.setattr(assets.requests, "get", lambda *a, **kw: Response(value={"id": schema.MODEL_REPO, "sha": "0" * 40}))
    with pytest.raises(ValueError, match="pinned model repository/revision"):
        assets.fetch_tokenizer(tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
