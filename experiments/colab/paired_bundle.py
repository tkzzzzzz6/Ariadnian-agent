"""Allowlisted paired-pilot transport: explicit records, receipts and eval inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from bundle import ASSET_NAMES, sha256


ROLES = {"short_records": "short_records.jsonl", "long_records": "long_records.jsonl",
         "sources": "sources.jsonl", "short_reviews": "short_reviews.jsonl",
         "long_reviews": "long_reviews.jsonl", "source_reviews": "source_reviews.jsonl",
         "pair_manifest": "pair_manifest.json", "eval_records": "eval_records.jsonl",
         "eval_manifest": "eval_manifest.json", "human_approval": "approval.json"}


def build_bundle(repo, output, *, resource_only=False, **inputs):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("bundle output already exists")
    if set(inputs) - set(ROLES):
        raise ValueError("unknown input role")
    if not resource_only and any(not inputs.get(role) for role in ("short_records", "long_records", "sources")):
        raise ValueError("explicit short/long/source records are required")
    entries = {}

    def add(path, name=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required regular file missing: {path.name}")
        name = name or path.relative_to(repo).as_posix()
        if name in entries:
            raise ValueError("duplicate archive path")
        entries[name] = path

    for package in ("experiments/common/ariadne_data", "experiments/ms_swift/cpt", "experiments/ms_swift/gemma_eval"):
        for path in sorted((repo / package).rglob("*.py")):
            if "__pycache__" not in path.parts:
                add(path)
    for relative in ("experiments/common/pyproject.toml", "experiments/common/README.md",
                     "experiments/ms_swift/pyproject.toml", "experiments/ms_swift/requirements-train.txt",
                     "experiments/ms_swift/gemma_eval/model_weights.lock.json",
                     "experiments/colab/remote_job.py", "experiments/colab/paired_job.py",
                     "experiments/colab/training_audit.py", "experiments/colab/best_checkpoint.py",
                     "experiments/colab/tmux_runner.py", "experiments/colab/evaluation_outcome.py"):
        add(repo / relative)
    for name in ASSET_NAMES:
        add(repo / "experiments/ms_swift/.cache/model_assets/gemma4-e2b" / name)
    for role, path in inputs.items():
        if path is not None:
            add(path, "inputs/" + ROLES[role])
    manifest = {"schema_version": "colab-sft-bundle-v1", "experiment": "paired-long-context-pilot-v1",
                "resource_only": resource_only,
                "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for name, path in sorted(entries.items())},
                "scope": "Explicit source packages, inert model assets, supplied record/review/eval files only.",
                "data_admission": "Transport is not approval. Both arms must independently pass cpt training export."}
    if sum(item["bytes"] for item in manifest["files"].values()) > 512 * 1024 * 1024:
        raise ValueError("bundle exceeds bounded unpack size")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"archive": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
            "files": len(entries), "records_approved": False, "gpu_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resource-only", action="store_true")
    for name in ROLES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    print(json.dumps(build_bundle(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
