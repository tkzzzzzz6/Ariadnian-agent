"""Acquire pinned upstream assets; run unchanged RULER generators in an isolated env.

The RULER launcher is adapted to subprocess argument lists for Windows safety.
No generator or task template is patched. The pinned generator's base prompt,
including its answer prefix, is subsequently wrapped by our common chat protocol.
This is a small custom protocol, not a published RULER leaderboard result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

RULER_REV = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
RULER_ZIP_SHA = "8423b2afaab612306924d079b3c365853cd0f9eeeed444b901ae25cee1ddea03"
BOOLQ_REV = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
TASKS = ["niah_multikey_1", "niah_multivalue", "vt", "cwe", "qa_1"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Ariadne-research-evaluation/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read()
        temp = path.with_suffix(path.suffix + ".partial")
        temp.write_bytes(raw)
        temp.replace(path)
    return {"url": url, "sha256": sha(path), "bytes": path.stat().st_size}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acquire(cache):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "ruler_source.zip"
    ruler = download(f"https://codeload.github.com/NVIDIA/RULER/zip/{RULER_REV}", archive)
    if ruler["sha256"] != RULER_ZIP_SHA:
        raise ValueError("RULER archive hash differs from inspected source")
    source = cache / f"RULER-{RULER_REV}"
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (cache / member.filename).resolve()
            if not target.is_relative_to(source.resolve()):
                raise ValueError("RULER archive traversal")
            if not member.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                data = zipped.read(member)
                if target.exists() and target.read_bytes() != data:
                    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                        expected = data.decode().split("oid sha256:")[1].splitlines()[0]
                        if sha(target) == expected:
                            continue
                    raise ValueError("upstream source modified locally")
                target.write_bytes(data)
    json_dir = source / "scripts/data/synthetic/json"
    word_file = json_dir / "english_words.json"
    if word_file.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
        word_asset = cache / "english_words.lfs.json"
        download(f"https://media.githubusercontent.com/media/NVIDIA/RULER/{RULER_REV}/scripts/data/synthetic/json/english_words.json", word_asset)
        if sha(word_asset) != "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca":
            raise ValueError("RULER English vocabulary LFS hash mismatch")
        word_file.write_bytes(word_asset.read_bytes())
    squad = download("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", json_dir / "squad.json")
    boolq = download(f"https://huggingface.co/datasets/google/boolq/resolve/{BOOLQ_REV}/data/validation-00000-of-00001.parquet",
                     cache / "boolq_validation.parquet")
    # Arrow's bundled OpenSSL DLL can conflict with urllib on Windows. Convert
    # in a separate process so networking and parquet never share that DLL state.
    converted = subprocess.run([sys.executable, "-c",
        "import json,sys,pyarrow.parquet as pq; print(json.dumps(pq.read_table(sys.argv[1]).to_pylist()))",
        str(cache / "boolq_validation.parquet")], capture_output=True, text=True, check=True)
    rows = json.loads(converted.stdout)
    if len(rows) != 3270 or any(type(r["answer"]) is not bool for r in rows):
        raise ValueError("unexpected BoolQ validation content")
    (cache / "boolq_dev.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # Match upstream concatenation: sorted repository text files, then sorted HTML
    # conversions. Cache each original response and every failure for audit.
    urls = (json_dir / "PaulGrahamEssays_URLs.txt").read_text().splitlines()
    import html2text
    from bs4 import BeautifulSoup
    def fetch_essay(url):
        name = url.rsplit("/", 1)[-1]
        is_html = name.endswith(".html")
        # HTTP endpoint may be blocked; HTTPS serves the same named source.
        fetched_url = url.replace("http://www.paulgraham.com/", "https://www.paulgraham.com/")
        path = cache / "essay_originals" / name
        try:
            receipt = download(fetched_url, path)
            if is_html:
                converter = html2text.HTML2Text()
                converter.ignore_images = converter.ignore_tables = converter.escape_all = True
                converter.reference_links = converter.mark_code = False
                parsed = converter.handle(str(BeautifulSoup(path.read_bytes().decode("unicode_escape", errors="ignore"), "html.parser").find("font")))
            else:
                parsed = path.read_text(encoding="utf-8")
            return {**receipt, "original_url": url, "name": name, "kind": "html" if is_html else "repo", "text": parsed, "status": "ok"}
        except Exception as exc:
            return {"original_url": url, "url": fetched_url, "name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=6) as pool:
        essays = list(pool.map(fetch_essay, urls))
    successful = sorted((e for e in essays if e["status"] == "ok"), key=lambda e: (e["kind"] != "repo", e["name"]))
    if not successful:
        raise ValueError("no RULER essay haystack available")
    (json_dir / "PaulGrahamEssays.json").write_text(json.dumps({"text": "".join(e["text"] for e in successful)}), encoding="utf-8")
    manifest = {"ruler": {**ruler, "revision": RULER_REV}, "squad": squad,
                "boolq": {**boolq, "revision": BOOLQ_REV, "records": len(rows), "jsonl_sha256": sha(cache / "boolq_dev.jsonl"),
                          "fallback_reason": "original GCS dev.jsonl returned HTTP 403; official Google HF parquet used"},
                "english_words_sha256": sha(word_file), "essay_haystack_sha256": sha(json_dir / "PaulGrahamEssays.json"),
                "essays": [{k: v for k, v in e.items() if k != "text"} for e in essays],
                "essay_acquisition": "upstream URL list; HTTPS transport; same parsing/order; failed sources retained in manifest",
                "source_scripts": {str(f.relative_to(source)): sha(f) for f in source.rglob("*.py")}}
    manifest_path = cache / "acquisition_manifest.json"
    if manifest_path.exists():
        previous = manifest_path.read_bytes()
        history = cache / ("acquisition_manifest." + hashlib.sha256(previous).hexdigest()[:16] + ".json")
        if not history.exists():
            history.write_bytes(previous)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate(cache, model_assets, generation_tag="v2"):
    import yaml
    import nltk
    nltk.data.path.insert(0, str(cache / "nltk_data"))
    for name in ("punkt", "punkt_tab"):
        nltk.download(name, download_dir=str(cache / "nltk_data"), quiet=True, raise_on_error=True)
    source = cache / f"RULER-{RULER_REV}"
    definitions = yaml.safe_load((source / "scripts/synthetic.yaml").read_text())
    constants = load_module(source / "scripts/data/synthetic/constants.py", "ruler_data_constants").TASKS
    import re
    if not re.fullmatch(r"[a-z0-9_-]+", generation_tag):
        raise ValueError("invalid generation tag")
    acquisition_sha = sha(cache / "acquisition_manifest.json")
    generation_root = cache / ("ruler_generated_" + generation_tag)
    input_lock = {"acquisition_manifest_sha256": acquisition_sha,
                  "model_lock_sha256": sha(model_assets / "model.lock.json"), "seed": 17011,
                  "task_grid": [(task, length) for length in (4096, 8192, 16384) for task in TASKS],
                  "generator_source_sha256": {str(f.relative_to(source)): sha(f) for f in source.rglob("*.py")},
                  "launcher_sha256": sha(Path(__file__))}
    bound = generation_root / "generation.inputs.json"
    serialized = json.dumps(input_lock, sort_keys=True, indent=2)
    if generation_root.exists():
        if not bound.exists() or bound.read_text(encoding="utf-8") != serialized:
            raise ValueError("generation tag already contains unbound/different inputs; choose a fresh tag")
    else:
        generation_root.mkdir()
        bound.write_text(serialized, encoding="utf-8")
    receipts = []
    for length in (4096, 8192, 16384):
        for task in TASKS:
            config = definitions[task]
            base = constants[config["task"]]
            out = generation_root / str(length)
            result = out / task / "validation.jsonl"
            command = [sys.executable, str(source / "scripts/data/synthetic" / (config["task"] + ".py")),
                       "--save_dir", str(out), "--save_name", task, "--subset", "validation",
                       "--tokenizer_path", str(model_assets), "--tokenizer_type", "hf",
                       "--max_seq_length", str(length), "--model_template_token", "16",
                       "--tokens_to_generate", str(base["tokens_to_generate"]), "--num_samples", "10",
                       "--random_seed", "17011", "--template", base["template"] + base.get("answer_prefix", "")]
            for key, value in config["args"].items():
                command += ["--" + key, str(value)]
            log_base = cache / "generator_logs" / f"{generation_tag}_{task}_{length}"
            attempt = 1
            while Path(str(log_base) + f".attempt{attempt}.log").exists():
                attempt += 1
            log = Path(str(log_base) + f".attempt{attempt}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONUTF8="1", NLTK_DATA=str(cache / "nltk_data"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            receipt_path = result.with_suffix(".receipt.json")
            if result.exists():
                if not receipt_path.exists():
                    raise ValueError("existing RULER output has no bound receipt; use a fresh generation tag")
                old = json.loads(receipt_path.read_text())
                if old.get("sha256") != sha(result) or old.get("input_lock_sha256") != sha(bound):
                    raise ValueError("existing RULER output changed or belongs to other inputs")
            else:
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
                if completed.returncode:
                    raise RuntimeError(f"RULER {task}/{length} failed; see {log}")
            rows = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
            if len(rows) != 10:
                raise ValueError("RULER generation count differs from frozen design")
            receipt = {"task": task, "length": length, "path": str(result), "sha256": sha(result),
                             "records": len(rows), "seed": 17011, "output_tokens": base["tokens_to_generate"],
                             "command": command, "generator_sha256": sha(Path(command[1])),
                             "input_lock_sha256": sha(bound)}
            if not receipt_path.exists():
                receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            receipts.append(receipt)
            print(f"generated {task}/{length}: {len(rows)}", flush=True)
    if sha(cache / "acquisition_manifest.json") != acquisition_sha:
        raise ValueError("acquisition inputs changed during generation")
    manifest = {"revision": RULER_REV, "acquisition_manifest_sha256": acquisition_sha,
                "input_lock_path": str(bound), "input_lock_sha256": sha(bound),
                "generation_tag": generation_tag, "protocol": "official unchanged generators; base prompt plus common Gemma chat wrapper",
                "chat_overhead_reserved": 16, "generation": receipts,
                "qa_cluster_note": "same official SQuAD query across lengths is one cluster; generator seed is paired across lengths"}
    manifest_path = cache / "ruler_generation_manifest.json"
    if manifest_path.exists():
        previous = manifest_path.read_bytes()
        history = cache / ("ruler_generation_manifest." + hashlib.sha256(previous).hexdigest()[:16] + ".json")
        if not history.exists():
            history.write_bytes(previous)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("acquire", "generate"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model-assets", type=Path)
    parser.add_argument("--generation-tag", default="v2")
    args = parser.parse_args()
    cache = args.cache.resolve()
    if args.action == "acquire":
        result = acquire(cache)
        print(json.dumps({"boolq": result["boolq"]["records"], "essays_ok": sum(e["status"] == "ok" for e in result["essays"])}))
    else:
        generate(cache, args.model_assets.resolve(), args.generation_tag)


if __name__ == "__main__":
    main()
