"""Audit CPT control plans and optional SWIFT encodings without claiming gains.

Matching body-token counts or optimizer steps cannot prove equal training
exposure. This preflight deliberately keeps that stronger claim unverified
until the trainer records the labels actually consumed, including partial
epochs, distributed sampling and discarded batches.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .core import digest, file_hash, new_output, read_jsonl, write_json
from .launch import launch


def source_coverage(rows, strategy):
    """Canonical coverage multiplicities, independent of sequence boundaries."""
    groups = defaultdict(lambda: defaultdict(int))
    documents = {}
    for row in rows:
        if row["split"] != "train" or row["strategy"] != strategy:
            continue
        did = row["document_id"]
        sha = row["document_sha256"]
        if did in documents and documents[did] != sha:
            raise ValueError("inconsistent source document hash")
        documents[did] = sha
        for span in row["spans"]:
            start, end = span["start"], span["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end or span["document_id"] != did:
                raise ValueError("invalid source span")
            key = (did, span["block_id"], span["text_field"])
            groups[key][start] += 1
            groups[key][end] -= 1
    segments = []
    for key, events in sorted(groups.items()):
        # Remove zero-delta joins so differing sequence cuts produce the same
        # fingerprint while overlapping/repeated source exposure still counts.
        events = {point: delta for point, delta in events.items() if delta}
        active, previous = 0, None
        for point, delta in sorted(events.items()):
            if active and previous is not None:
                segments.append([*key, previous, point, active])
            active += delta
            if active < 0:
                raise ValueError("negative source coverage")
            previous = point
        if active:
            raise ValueError("unclosed source coverage")
    if not segments:
        raise ValueError("no training source spans")
    return {"documents": len(documents), "document_hashes_sha256": digest(documents),
            "source_coverage_sha256": digest(segments),
            "source_character_exposures": sum((s[-2] - s[-3]) * s[-1] for s in segments)}


def _plan(path):
    path = Path(path).resolve()
    readiness = launch(path, execute=False)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not plan["recipe"]["method"].startswith("pt_"):
        raise ValueError("this structure comparison requires two CPT plans")
    prepared = Path(plan["prepared_dir"])
    manifest = json.loads((prepared / "prepare_manifest.json").read_text(encoding="utf-8"))
    spec = manifest["artifacts"]["sequence_lineage.jsonl"]
    lineage = read_jsonl(prepared / spec["path"])
    return plan, readiness, source_coverage(lineage, plan["strategy"])


def _template(path, plan):
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema_version") != "cpt-swift-template-verification-v1"
            or value.get("verified") is not True or value.get("model") != plan["recipe"]["model"]
            or value.get("model_lock_sha256") != plan["model_lock_sha256"]
            or value.get("max_length") != plan["recipe"]["parameters"]["max_length"]
            or value.get("use_chat_template") is not False or value.get("loss_scale") != "all"):
        raise ValueError("template audit is not bound to this CPT plan")
    for split in ("train", "dev"):
        data = value.get("datasets", {}).get(split, {})
        if (data.get("sha256") != plan["dataset_sha256"][split]
                or Path(data.get("path", "")).resolve() != Path(plan["datasets"][split]).resolve()
                or type(data.get("loss_tokens")) is not int or data["loss_tokens"] <= 0
                or data.get("dropped_records") != 0
                or not re.fullmatch(r"[a-f0-9]{64}", data.get("loss_token_histogram_sha256", ""))):
            raise ValueError("template audit has stale data or incomplete loss-token accounting")
    return {"sha256": file_hash(path), "train_loss_tokens_one_pass": value["datasets"]["train"]["loss_tokens"],
            "train_loss_histogram_sha256": value["datasets"]["train"]["loss_token_histogram_sha256"]}


def compare(left_path, right_path, *, left_template=None, right_template=None):
    left, left_ready, left_source = _plan(left_path)
    right, right_ready, right_source = _plan(right_path)
    checks = {
        "opposite_structure_arms": {left["strategy"], right["strategy"]} == {"ordered", "shuffled"},
        "same_model": left["recipe"]["model"] == right["recipe"]["model"],
        "same_training_recipe": left["recipe"] == right["recipe"],
        "same_source_coverage": left_source == right_source,
        "same_validation_data": left["dataset_sha256"]["dev"] == right["dataset_sha256"]["dev"],
        "both_training_admitted": left_ready["training_ready"] and right_ready["training_ready"],
    }
    encodings = [_template(left_template, left), _template(right_template, right)]
    have_both = all(x is not None for x in encodings)
    checks["both_native_template_encodings_verified"] = have_both
    checks["same_encoded_loss_budget_one_pass"] = (
        encodings[0]["train_loss_tokens_one_pass"] == encodings[1]["train_loss_tokens_one_pass"] if have_both else None)
    checks["same_encoded_loss_token_histogram_one_pass"] = (
        encodings[0]["train_loss_histogram_sha256"] == encodings[1]["train_loss_histogram_sha256"] if have_both else None)
    checks["actual_training_loss_token_exposure_verified"] = False
    return {"schema_version": "cpt-comparison-preflight-v1", "executed_training": False,
            "plans": [{"path": str(Path(p).resolve()), "sha256": file_hash(p)} for p in (left_path, right_path)],
            "source_coverage": {"left": left_source, "right": right_source},
            "template_audits": encodings, "checks": checks,
            "unresolved": [key for key, passed in checks.items() if passed is not True],
            "strict_equal_training_exposure": False,
            "limitations": ["One-pass encoding and source coverage are preparation evidence, not actual training exposure.",
                            "Equal optimizer steps can consume different labels. Do not infer equal-token causal effects.",
                            "This tool does not establish cross-split decontamination, model quality, or statistical significance."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--left-plan", type=Path, required=True)
    parser.add_argument("--right-plan", type=Path, required=True)
    parser.add_argument("--left-template", type=Path)
    parser.add_argument("--right-template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-strict", action="store_true", help="exit nonzero if equal actual training exposure is unproven")
    args = parser.parse_args(argv)
    try:
        result = compare(args.left_plan, args.right_plan, left_template=args.left_template, right_template=args.right_template)
        write_json(new_output(args.output) / "comparison_report.json", result)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Comparison preflight: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if args.require_strict and not result["strict_equal_training_exposure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
