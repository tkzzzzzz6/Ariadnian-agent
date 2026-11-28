"""Authored fixtures only; no real reviews, model loading or GPU training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpt import core, launch, schema, sft, swift_verify


@pytest.fixture
def sft_sources(tmp_path, monkeypatch):
    docs = [{"id": split, "family_id": split, "title": split, "language": "en", "domain": "fixture",
             "kind": "report", "split": split, "license": "CC0-1.0", "source_url": "https://example.test/" + split,
             "revision": "synthetic-test-only", "blocks": [{"id": "p", "type": "paragraph",
             "text": "Synthetic source for " + split, "section_path": [], "page": None}]}
            for split in ("train", "dev", "test")]
    source = tmp_path / "sources.jsonl"
    core.write_jsonl(source, docs)
    source_reviews = [{"document_id": d["id"], "document_sha256": core.digest(d), "decision": "approved",
                       "reviewer_type": "human", "reviewer": "SYNTHETIC FIXTURE - no real human review",
                       "reason": "Authored unit-test data only", "license_decision": "approved",
                       "license_evidence": ["https://example.test/license"]} for d in docs]
    source_reviews_path = tmp_path / "source_reviews.jsonl"
    core.write_jsonl(source_reviews_path, source_reviews)
    rows = [{"id": split + "-qa", "split": split, "source_document_ids": [split], "license": "CC0-1.0",
             "kind": "human_qa", "messages": [{"role": "user", "content": "Which source? " + split},
                                               {"role": "assistant", "content": "Source: " + split}]}
            for split in ("train", "dev")]
    input_path = tmp_path / "qa.jsonl"
    core.write_jsonl(input_path, rows)
    reviews = [{"record_id": r["id"], "record_sha256": core.digest(r),
                "source_sha256": {r["split"]: core.digest(next(d for d in docs if d["id"] == r["split"]))},
                "decision": "approved", "reviewer_type": "human",
                "reviewer": "SYNTHETIC FIXTURE - no real human review", "reason": "Test only",
                "license_decision": "approved", "license_evidence": ["https://example.test/license"]} for r in rows]
    reviews_path = tmp_path / "qa_reviews.jsonl"
    core.write_jsonl(reviews_path, reviews)
    lock_sha = "a" * 64
    monkeypatch.setattr(sft, "load_native_tokenizer", lambda _: SimpleNamespace(
        count=len, metadata={"model_lock_sha256": lock_sha}))
    monkeypatch.setattr(launch, "_locked_assets", lambda _: (
        {"repository": schema.MODEL_REPO, "revision": schema.MODEL_REVISION}, lock_sha))
    return SimpleNamespace(docs=docs, source=source, source_reviews=source_reviews_path,
                           rows=rows, input=input_path, reviews=reviews_path, review_rows=reviews,
                           assets=tmp_path / "assets", out=tmp_path / "prepared")


def prepare(f, **kwargs):
    return sft.prepare_sft(f.input, f.source, f.assets, f.out, allowed_licenses=["CC0-1.0"], **kwargs)


@pytest.mark.parametrize("method", sorted(schema.METHODS))
def test_explicit_methods_keep_their_own_objective_and_quantization(method):
    recipe = schema.default_recipe(method, sequence_length=4096, smoke=True)
    schema.validate_recipe(recipe)
    assert recipe["parameters"]["max_steps"] == 20
    assert recipe["parameters"]["use_chat_template"] == method.startswith("sft_")
    assert recipe["parameters"]["loss_scale"] == ("default" if method.startswith("sft_") else "all")
    assert ("quant_method" in recipe["parameters"]) == method.endswith("qlora")
    if method.endswith("qlora"):
        for field, required in schema.QUANT_PARAMETERS.items():
            changed = copy.deepcopy(recipe)
            changed["parameters"].pop(field)
            with pytest.raises(ValueError, match="missing"):
                schema.validate_recipe(changed)
    else:
        recipe["parameters"].update(schema.QUANT_PARAMETERS)
        with pytest.raises(ValueError, match="unknown"):
            schema.validate_recipe(recipe)


def test_preview_sft_preserves_pending_reviews_and_cannot_launch(sft_sources, tmp_path, monkeypatch):
    f = sft_sources
    manifest = prepare(f)
    assert not manifest["training_ready"] and manifest["approved_records"] == 0
    assert manifest["reviews_sha256"] is None and manifest["source_reviews_sha256"] is None
    assert all(r["decision"] == "pending" and not r["reviewer"]
               for r in core.read_jsonl(f.out / "reviews_pending.jsonl"))
    recipe = tmp_path / "recipe.json"
    core.write_json(recipe, schema.default_recipe("sft_qlora", sequence_length=4096, smoke=True))
    result = launch.plan(f.out, recipe, tmp_path / "plan")
    assert result["strategy"] == "sft" and not result["training_ready"]
    assert "swift.cli.sft" in result["argv"] and "--columns" not in result["argv"]
    assert "preview_or_unapproved_data" in result["blockers"]
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("preview must never spawn"))
    with pytest.raises(ValueError, match="training denied"):
        launch.launch(tmp_path / "plan/launch_plan.json", execute=True)


def test_sft_requires_both_record_and_source_reviews(sft_sources):
    f = sft_sources
    with pytest.raises(ValueError, match="approved train and dev"):
        prepare(f, reviews_path=f.reviews, mode="training")
    with pytest.raises(ValueError, match="approved train and dev"):
        prepare(f, source_reviews_path=f.source_reviews, mode="training")
    result = prepare(f, reviews_path=f.reviews, source_reviews_path=f.source_reviews, mode="training")
    assert result["training_ready"] and result["counts"] == {"train": 1, "dev": 1}
    assert result["input_sha256"] == core.file_hash(f.input)
    assert result["source_corpus_sha256"] == core.file_hash(f.source)
    assert result["reviews_sha256"] == core.file_hash(f.reviews)
    assert result["source_reviews_sha256"] == core.file_hash(f.source_reviews)


@pytest.mark.parametrize("changed", ["input", "source", "reviews", "source_reviews"])
def test_sft_source_mutation_during_preparation_is_rejected_before_output(sft_sources, monkeypatch, changed):
    f = sft_sources
    original_loader = sft.load_native_tokenizer

    def mutate_and_load(path):
        # All four inputs have already been read when native-tokenizer loading
        # occurs. Appending whitespace changes bytes without changing JSON data.
        with getattr(f, changed).open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return original_loader(path)

    monkeypatch.setattr(sft, "load_native_tokenizer", mutate_and_load)
    with pytest.raises(ValueError, match="changed during SFT preparation"):
        prepare(f, reviews_path=f.reviews, source_reviews_path=f.source_reviews, mode="training")
    assert not f.out.exists()


@pytest.mark.parametrize("mutation", ["answer", "source", "benchmark", "test", "source_split", "prompt_leak"])
def test_sft_stale_reviews_and_leakage_are_rejected(sft_sources, mutation):
    f = sft_sources
    if mutation == "answer": f.rows[0]["messages"][-1]["content"] += " changed"
    elif mutation == "source":
        f.docs[0]["blocks"][0]["text"] += " changed"
        core.write_jsonl(f.source, f.docs)
    elif mutation == "benchmark": f.rows[0]["benchmark"] = "longbench"
    elif mutation == "test": f.rows[0]["split"] = "test"
    elif mutation == "source_split": f.rows[0]["source_document_ids"] = ["dev"]
    else: f.rows[1]["messages"][0] = copy.deepcopy(f.rows[0]["messages"][0])
    core.write_jsonl(f.input, f.rows)
    with pytest.raises(ValueError):
        prepare(f, reviews_path=f.reviews, source_reviews_path=f.source_reviews, mode="training")


def test_changed_sft_lineage_invalidates_existing_plan(sft_sources, tmp_path):
    f = sft_sources
    prepare(f, reviews_path=f.reviews, source_reviews_path=f.source_reviews, mode="training")
    recipe = tmp_path / "recipe.json"
    core.write_json(recipe, schema.default_recipe("sft_lora", sequence_length=4096))
    result = launch.plan(f.out, recipe, tmp_path / "plan")
    assert result["training_ready"] and not launch.launch(tmp_path / "plan/launch_plan.json")["executed"]
    with (f.out / "lineage.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        launch.launch(tmp_path / "plan/launch_plan.json")


def test_qlora_missing_extension_is_rejected_before_loading_model(monkeypatch):
    monkeypatch.setattr(launch.importlib.metadata, "version", lambda _: "0.49.0")
    with pytest.raises(ValueError, match="bitsandbytes must be"):
        launch._quant_environment(schema.default_recipe("pt_qlora"))
    assert launch._quant_environment(schema.default_recipe()) is None


def test_foreign_adapter_without_parent_provenance_is_rejected(tmp_path):
    config = {"peft_type": "LORA", "base_model_name_or_path": schema.MODEL_REPO, "r": 16, "lora_alpha": 32}
    core.write_json(tmp_path / "adapter_config.json", config)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"unit-test placeholder, not real weights")
    with pytest.raises(ValueError, match="source launch plan"):
        launch._adapter_provenance(tmp_path, schema.default_recipe("sft_lora"))


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) + 10 for c in text]


@pytest.mark.parametrize("mutation", [None, "prompt_loss", "missing_answer", "changed_input", "weighted", "overflow"])
def test_sft_template_verifies_whole_prompt_mask_and_supervised_answer(tmp_path, mutation):
    messages = [{"role": "system", "content": "Test system"}, {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}]
    records = [{"messages": messages}]
    path = tmp_path / "sft.jsonl"
    core.write_jsonl(path, records)
    tokenizer = CharacterTokenizer()
    prompt_ids, response_ids = swift_verify._sft_expected_ids(messages, tokenizer)
    encoded = {"input_ids": prompt_ids + response_ids, "labels": [-100] * len(prompt_ids) + response_ids}
    if mutation == "prompt_loss": encoded["labels"][5] = encoded["input_ids"][5]
    elif mutation == "missing_answer": encoded["labels"][-2] = -100
    elif mutation == "changed_input": encoded["input_ids"][3] += 1
    elif mutation == "weighted": encoded["loss_scale"] = [0] * len(prompt_ids) + [2] * len(response_ids)
    template = SimpleNamespace(tokenizer=tokenizer, encode=lambda *a, **kw: encoded)
    if mutation:
        with pytest.raises(ValueError):
            swift_verify._encode_sft_split(path, records, template, lambda x: x, 5 if mutation == "overflow" else 4096)
    else:
        result = swift_verify._encode_sft_split(path, records, template, lambda x: x, 4096)
        assert result["loss_tokens"] == len(response_ids)
        assert result["masked_prompt_tokens"] == len(prompt_ids)


def test_all_stored_recipes_match_their_explicit_method_and_are_valid():
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in configs.glob("*.json"):
        recipe = schema.load_recipe(path)
        assert recipe["method"] in path.name


def test_sft_template_integration_selects_chat_and_default_mask(tmp_path, monkeypatch):
    from test_cpt_swift_verify import _contract, _dataset, _hash
    plan, observed, paths, _ = _contract(tmp_path, monkeypatch)
    plan["recipe"] = schema.default_recipe("sft_qlora", sequence_length=4096)
    for split, path in paths.items():
        _dataset(path, [{"messages": [{"role": "user", "content": "Question " + split},
                                      {"role": "assistant", "content": "Answer " + split}]}])
        plan["dataset_sha256"][split] = _hash(path)
    with swift_verify._offline_environment():
        processor_loader, _, _ = swift_verify._load_backend()

    class Template:
        tokenizer = CharacterTokenizer()

        def set_mode(self, mode):
            assert mode == "train"

        def encode(self, row, **kwargs):
            user, answer = row["messages"]
            p = self.tokenizer.encode("<bos><|turn>user\n" + user["content"] + "<turn|>\n<|turn>model\n",
                                      add_special_tokens=False)
            a = self.tokenizer.encode(answer["content"] + "<turn|>\n", add_special_tokens=False)
            return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    def template_loader(processor, **kwargs):
        assert kwargs["use_chat_template"] is True
        assert kwargs["loss_scale"] == "default"
        assert kwargs["truncation_strategy"] == "raise"
        return Template()

    monkeypatch.setattr(swift_verify, "_load_backend", lambda: (processor_loader, template_loader, lambda x: x))
    result = swift_verify.verify_template(plan)
    assert result["verified"] and result["use_chat_template"]
    assert result["loss_scale"] == "default"
    assert result["datasets"]["train"]["loss_tokens"] < result["datasets"]["train"]["input_tokens"]


def test_cpt_to_sft_continuation_binds_adapter_and_parent_receipt(sft_sources, tmp_path, monkeypatch):
    from test_cpt_launch import _setup
    f = sft_sources
    prepare(f, reviews_path=f.reviews, source_reviews_path=f.source_reviews, mode="training")
    pt_root = tmp_path / "pt"
    pt_root.mkdir()
    _, pt_plan_path, pt_plan = _setup(pt_root, monkeypatch, "training")
    checkpoint = Path(pt_plan["checkpoint_dir"]) / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    core.write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": schema.MODEL_REPO,
                                                         "r": 16, "lora_alpha": 32})
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"Synthetic planning fixture, never load as real tensors")
    core.write_json(pt_plan_path.parent / "launch_receipt.json", {"status": "process_exited", "returncode": 0,
                                                                 "plan_sha256": core.file_hash(pt_plan_path)})
    recipe_path = tmp_path / "sft_recipe.json"
    core.write_json(recipe_path, schema.default_recipe("sft_lora", sequence_length=4096))
    sft_plan = launch.plan(f.out, recipe_path, tmp_path / "sft_plan", init_adapter=checkpoint)
    assert sft_plan["init_adapter"]["files"]["adapter_model.safetensors"] == core.file_hash(weights)
    assert sft_plan["argv"][-2:] == ["--adapters", str(checkpoint.resolve())]
    assert launch.launch(tmp_path / "sft_plan/launch_plan.json")["executed"] is False
    with pytest.raises(ValueError, match="quantization method"):
        launch._adapter_provenance(checkpoint, schema.default_recipe("sft_qlora"))
    weights.write_bytes(b"Changed synthetic checkpoint")
    with pytest.raises(ValueError, match="changed"):
        launch.launch(tmp_path / "sft_plan/launch_plan.json")
