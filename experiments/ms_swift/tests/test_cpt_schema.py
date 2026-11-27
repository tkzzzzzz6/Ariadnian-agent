"""Pure metadata safety checks; no model, GPU, network or Torch dependency."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cpt import schema  # noqa: E402


def lock_for(directory: Path) -> dict:
    return {"schema_version": schema.MODEL_LOCK_SCHEMA_VERSION, "repository": schema.MODEL_REPO,
            "revision": schema.MODEL_REVISION, "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                                    "bytes": path.stat().st_size}
                                                       for path in directory.iterdir() if path.is_file()}}


@pytest.fixture(scope="module")
def native_assets(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-gemma-schema")
    config = {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
              "text_config": {"model_type": "gemma4_text", "max_position_embeddings": 131072,
                              "vocab_size": 262144, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0}}
    tokenizer_config = {"tokenizer_class": "GemmaTokenizer", "processor_class": "Gemma4Processor", "add_bos_token": True,
                        "model_max_length": 10**30, **schema.SPECIAL_TOKEN_TEXT}
    special = {0: "<pad>", 1: "<eos>", 2: "<bos>", 3: "<unk>"}
    vocab = {special.get(i, f"fixture-token-{i}"): i for i in range(schema.MODEL_VOCAB_SIZE)}
    tokenizer = {"version": "1.0", "model": {"type": "BPE", "vocab": vocab}, "added_tokens": []}
    for filename, value in (("config.json", config), ("tokenizer_config.json", tokenizer_config), ("tokenizer.json", tokenizer)):
        (directory / filename).write_text(json.dumps(value), encoding="utf-8")
    return directory, lock_for(directory)


def test_defaults_are_explicit_native_full_text_lora_and_fresh_objects():
    recipe = schema.default_recipe()
    assert schema.validate_recipe(recipe) is None
    assert recipe["model"] == {"repo": schema.MODEL_REPO, "revision": schema.MODEL_REVISION}
    assert recipe["parameters"]["use_chat_template"] is False
    assert recipe["parameters"]["loss_scale"] == "all"
    assert recipe["parameters"]["torch_dtype"] == "bfloat16"
    assert recipe["parameters"]["freeze_vit"] and recipe["parameters"]["freeze_aligner"]
    assert not {"training_ready", "training_succeeded", "weights_downloaded"} & set(recipe)
    recipe["parameters"]["target_modules"].append("vision_tower")
    assert schema.default_recipe()["parameters"]["target_modules"] == ["all-linear"]


@pytest.mark.parametrize("name,value", [("use_chat_template", True), ("loss_scale", "last_round"),
                                        ("freeze_vit", False), ("freeze_aligner", False),
                                        ("freeze_llm", True), ("split_dataset_ratio", .1),
                                        ("packing", True), ("strict", 1), ("torch_dtype", "float16"),
                                        ("streaming", True), ("target_modules", "all-linear")])
def test_training_objective_and_scope_cannot_silently_change(name, value):
    recipe = schema.default_recipe()
    recipe["parameters"][name] = value
    with pytest.raises(ValueError, match=name):
        schema.validate_recipe(recipe)


@pytest.mark.parametrize("name,value", [("max_length", 131073), ("max_length", True), ("lora_rank", 0),
                                        ("learning_rate", float("nan")), ("max_steps", -1), ("seed", True)])
def test_invalid_numerics_never_serialize_into_a_training_plan(name, value):
    recipe = schema.default_recipe()
    recipe["parameters"][name] = value
    with pytest.raises(ValueError, match=name):
        schema.validate_recipe(recipe)


def test_unknown_flags_unpinned_sources_and_mislabeled_sft_are_rejected():
    recipe = schema.default_recipe()
    recipe["parameters"]["custom_register_path"] = "remote.py"
    with pytest.raises(ValueError, match="unknown"):
        schema.validate_recipe(recipe)
    for repository in ("Qwen/Qwen2.5-0.5B", "google/gemma-4-E2B-it"):
        recipe = schema.default_recipe(); recipe["model"]["repo"] = repository
        with pytest.raises(ValueError, match="Base"):
            schema.validate_recipe(recipe)
    recipe = schema.default_recipe(); recipe["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="40-hex"):
        schema.validate_recipe(recipe)
    recipe = schema.default_recipe(); recipe["method"] = "sft_lora"
    with pytest.raises(ValueError, match="use_chat_template"):
        schema.validate_recipe(recipe)


def test_recipe_hash_is_canonical_load_rejects_duplicate_keys_and_nonfinite_json(tmp_path):
    recipe = schema.default_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    assert schema.load_recipe(path) == recipe
    assert schema.recipe_sha256(recipe) == schema.recipe_sha256(dict(reversed(list(recipe.items()))))
    path.write_text('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        schema.load_recipe(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        schema.load_recipe(path)


def test_native_lock_validates_complete_vocabulary_and_ignores_hf_length_sentinel(native_assets):
    directory, lock = native_assets
    assert schema.validate_model_lock(lock) is None
    assert schema.validate_model_lock(lock, directory) is None


@pytest.mark.parametrize("filename", ["model.safetensors", "weights.bin", "../config.json", "C:\\config.json", "modeling_gemma4.py", "arbitrary.json"])
def test_model_lock_never_lists_weights_scripts_or_traversal(native_assets, filename):
    _, lock = native_assets
    lock = copy.deepcopy(lock)
    lock["files"][filename] = {"sha256": "a" * 64, "bytes": 16}
    with pytest.raises(ValueError, match="forbidden"):
        schema.validate_model_lock(lock)


def test_hash_and_size_mismatch_and_reference_tokenizer_are_rejected(native_assets, tmp_path):
    directory, lock = native_assets
    bad = copy.deepcopy(lock); bad["files"]["tokenizer.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/size"):
        schema.validate_model_lock(bad, directory)
    bad = copy.deepcopy(lock); bad["files"]["config.json"]["bytes"] += 1
    with pytest.raises(ValueError, match="hash/size"):
        schema.validate_model_lock(bad, directory)
    for filename in schema.REQUIRED_MODEL_FILES:
        (tmp_path / filename).write_text((directory / filename).read_text(), encoding="utf-8")
    config = json.loads((tmp_path / "config.json").read_text())
    config["model_type"] = "qwen2"
    (tmp_path / "config.json").write_text(json.dumps(config))
    forged_origin = lock_for(tmp_path)
    with pytest.raises(ValueError, match="gemma4"):
        schema.validate_model_lock(forged_origin, tmp_path)
    config["model_type"] = "gemma4"
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2}}}))
    with pytest.raises(ValueError, match="incomplete"):
        schema.validate_model_lock(lock_for(tmp_path), tmp_path)


def test_schema_import_and_validation_require_only_stdlib():
    experiment_root = Path(__file__).resolve().parents[1]
    script = "import sys;from cpt.schema import default_recipe,validate_recipe;validate_recipe(default_recipe());assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True, text=True, timeout=20, cwd=experiment_root)
    assert result.returncode == 0, result.stderr
