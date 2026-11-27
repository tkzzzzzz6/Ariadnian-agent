"""Build an explicit data/code transport ZIP; never walk the whole repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ASSET_NAMES = ("model.lock.json", "config.json", "generation_config.json",
               "processor_config.json", "tokenizer.json", "tokenizer_config.json")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build_bundle(repo, output, *, records, sources, reviews=None, source_reviews=None):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; use a new ZIP path")
    entries = {}

    def add(path, name=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required regular file missing: {path.name}")
        name = name or path.relative_to(repo).as_posix()
        if name in entries:
            raise ValueError(f"duplicate archive name: {name}")
        entries[name] = path

    for package in ("experiments/common/ariadne_data", "experiments/ms_swift/cpt",
                    "experiments/ms_swift/gemma_eval"):
        for path in sorted((repo / package).rglob("*.py")):
            if "__pycache__" not in path.parts:
                add(path)
    for name in ("experiments/common/pyproject.toml", "experiments/common/README.md",
                 "experiments/ms_swift/pyproject.toml", "experiments/ms_swift/requirements-train.txt",
                 "experiments/ms_swift/gemma_eval/model_weights.lock.json",
                 "experiments/ms_swift/configs/gemma4_e2b_sft_lora_4k_smoke.json",
                 "experiments/colab/remote_job.py"):
        add(repo / name)
    for name in ASSET_NAMES:
        add(repo / "experiments/ms_swift/.cache/model_assets/gemma4-e2b" / name)
    for role, path in (("records", records), ("sources", sources),
                       ("reviews", reviews), ("source_reviews", source_reviews)):
        if path is not None:
            add(path, f"inputs/{role}.jsonl")
    manifest = {
        "schema_version": "colab-sft-bundle-v1",
        "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for name, path in sorted(entries.items())},
        "data_admission": "pending until cpt sft-prepare --mode training succeeds with real human receipts",
        "scope": "explicit experiment package, one recipe, inert model assets, supplied data/reviews only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"archive": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
            "files": len(entries), "records_approved": False,
            "note": "Transport integrity only; this does not approve records or start a runtime."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--source-reviews", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_bundle(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
