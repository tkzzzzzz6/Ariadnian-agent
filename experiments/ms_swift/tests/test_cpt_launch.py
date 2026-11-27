"""Launch controls without installing SWIFT/CUDA or spawning training."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpt import core, launch
from cpt.schema import default_recipe


def _setup(tmp_path, monkeypatch, mode="preview"):
    assets = tmp_path / "assets"
    assets.mkdir()
    lock_sha = "a" * 64
    recipe = default_recipe()
    monkeypatch.setattr(core, "load_native_tokenizer", lambda _: SimpleNamespace(
        count=lambda text: len(text), metadata={"model_lock_sha256": lock_sha, "special_tokens_budget": 2}))
    monkeypatch.setattr(launch, "_locked_assets", lambda _: (
        {"repository": recipe["model"]["repo"], "revision": recipe["model"]["revision"]}, lock_sha))
    docs = []
    for split in ("train", "dev", "test"):
        docs.append({"id": split, "family_id": split, "title": split, "language": "en",
                     "domain": "fixture", "kind": "report", "split": split,
                     "license": "CC0-1.0", "source_url": "https://example.org", "revision": "v1",
                     "blocks": [{"id": "p", "type": "paragraph", "text": "source " + split,
                                 "section_path": [], "page": None}]})
    source = tmp_path / "source.jsonl"
    core.write_jsonl(source, docs)
    core.audit(source, tmp_path / "audit")
    reviews = core.read_jsonl(tmp_path / "audit" / "admission_review.jsonl")
    for row in reviews:
        row.update(decision="approved", reviewer_type="human", reviewer="Fixture reviewer",
                   reason="Fixture approved", license_decision="approved", license_evidence=["https://example.org/license"])
    review = tmp_path / "reviews.jsonl"
    core.write_jsonl(review, reviews)
    prepared = tmp_path / "prepared"
    core.prepare(source, assets, prepared, mode=mode, reviews_path=review,
                 allowed_licenses=["CC0-1.0"])
    recipe_path = tmp_path / "recipe.json"
    core.write_json(recipe_path, recipe)
    plan_dir = tmp_path / "plan"
    planned = launch.plan(prepared, recipe_path, plan_dir)
    return prepared, plan_dir / "launch_plan.json", planned


def test_preview_plan_is_reviewable_but_cannot_execute(tmp_path, monkeypatch):
    _, plan_path, result = _setup(tmp_path, monkeypatch)
    assert result["training_ready"] is False
    assert "preview_or_unapproved_data" in result["blockers"]
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    assert launch.launch(plan_path)["executed"] is False
    with pytest.raises(ValueError, match="training denied"):
        launch.launch(plan_path, execute=True)


def test_training_plan_requires_explicit_execute_and_emits_safe_argv(tmp_path, monkeypatch):
    _, plan_path, result = _setup(tmp_path, monkeypatch, "training")
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    assert result["training_ready"] is True
    assert launch.launch(plan_path)["executed"] is False
    args = result["argv"]
    assert args[1:3] == ["-m", "swift.cli.pt"]
    assert args[args.index("--model_revision") + 1] == default_recipe()["model"]["revision"]
    assert args[args.index("--use_chat_template") + 1] == "false"
    assert args[args.index("--push_to_hub") + 1] == "false"
    assert "--val_dataset" in args and "--strict" in args
    assert result["datasets"]["dev"].endswith("dev_ordered.jsonl")


@pytest.mark.parametrize("artifact", ["train_ordered.jsonl", "quarantine.jsonl", "admission_report.jsonl"])
def test_any_bound_artifact_mutation_invalidates_plan(tmp_path, monkeypatch, artifact):
    prepared, plan_path, _ = _setup(tmp_path, monkeypatch, "training")
    with (prepared / artifact).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        launch.launch(plan_path)


def test_saved_arbitrary_argv_is_never_trusted(tmp_path, monkeypatch):
    _, plan_path, result = _setup(tmp_path, monkeypatch, "training")
    altered = copy.deepcopy(result)
    altered["argv"].extend(["--external_plugins", "untrusted.py"])
    core.write_json(plan_path, altered)
    with pytest.raises(ValueError, match="changed"):
        launch.launch(plan_path, execute=True)


def test_environment_and_template_gate_precede_training(tmp_path, monkeypatch):
    _, plan_path, _ = _setup(tmp_path, monkeypatch, "training")
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    def unavailable():
        raise ValueError("fixture CUDA unavailable")
    monkeypatch.setattr(launch, "_environment", unavailable)
    with pytest.raises(ValueError, match="CUDA unavailable"):
        launch.launch(plan_path, execute=True)


def test_model_or_length_change_requires_fresh_preparation(tmp_path, monkeypatch):
    prepared, _, planned = _setup(tmp_path, monkeypatch, "training")
    recipe = planned["recipe"]
    recipe["parameters"]["max_length"] = 4096
    recipe_path = tmp_path / "different.json"
    core.write_json(recipe_path, recipe)
    with pytest.raises(ValueError, match="max_length"):
        launch.plan(prepared, recipe_path, tmp_path / "other")


def test_failed_template_verification_never_spawns_training(tmp_path, monkeypatch):
    from cpt import swift_verify
    _, plan_path, _ = _setup(tmp_path, monkeypatch, "training")
    monkeypatch.setattr(launch, "_environment", lambda: {"fixture": True})
    monkeypatch.setattr(swift_verify, "verify_template", lambda _: {"verified": False})
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="did not pass"):
        launch.launch(plan_path, execute=True)
    assert not (plan_path.parent / "launch_receipt.json").exists()


def test_explicit_execution_records_failure_and_disallows_accidental_restart(tmp_path, monkeypatch):
    from cpt import swift_verify
    _, plan_path, planned = _setup(tmp_path, monkeypatch, "training")
    monkeypatch.setattr(launch, "_environment", lambda: {"fixture": True})
    monkeypatch.setattr(swift_verify, "verify_template", lambda _: {"verified": True})
    calls = []
    def failed_process(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(launch.subprocess, "run", failed_process)
    with pytest.raises(ValueError, match="code 7"):
        launch.launch(plan_path, execute=True)
    assert len(calls) == 1 and calls[0][0] == planned["argv"]
    assert calls[0][1]["shell"] is False
    receipt = json.loads((plan_path.parent / "launch_receipt.json").read_text(encoding="utf-8"))
    assert receipt["returncode"] == 7 and receipt["training_completed"] is False
    with pytest.raises(ValueError, match="already started"):
        launch.launch(plan_path, execute=True)
    assert len(calls) == 1


def test_emulated_bf16_on_sm70_is_not_native_training_support(monkeypatch):
    monkeypatch.setattr(launch.platform, "system", lambda: "Linux")
    versions = {"ms-swift": "4.6.0", "torch": "2.8.0", "transformers": "5.12.1",
                "peft": "0.20.0", "accelerate": "1.12.0", "datasets": "4.8.4", "trl": "0.29.1"}
    monkeypatch.setattr(launch.importlib.metadata, "version", versions.__getitem__)
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda _: (7, 0),
                           is_bf16_supported=lambda **kwargs: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(ValueError, match="native BF16"):
        launch._environment()
