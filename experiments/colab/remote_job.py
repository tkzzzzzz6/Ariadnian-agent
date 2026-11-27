"""Separate Colab setup, preview, admitted training and artifact collection.

Run this file with an explicit action on the remote VM. It never provisions or
releases a VM; the calling operator owns that lifecycle, including failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_root(run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must contain 1-64 lowercase ASCII letters/digits/_/-")
    root = Path("/content") / ("ariadne_sft_" + run_id)
    if root.is_symlink():
        raise ValueError("run root cannot be a symbolic link")
    return root


def unpack(bundle, expected_sha256, root):
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256 or "") or sha256(bundle) != expected_sha256:
        raise ValueError("uploaded bundle SHA256 mismatch")
    if root.exists():
        raise ValueError("run directory already exists; choose a fresh run-id")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
            raise ValueError("duplicate archive names or missing manifest")
        if sum(info.file_size for info in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError("bundle exceeds the uncompressed transport size limit")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest.get("schema_version") != "colab-sft-bundle-v1":
            raise ValueError("unsupported bundle")
        if set(names) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("unlisted archive members")
        for name in names:
            pure = PurePosixPath(name)
            info = archive.getinfo(name)
            if (pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name
                    or pure.as_posix() != name or name == "."
                    or not pure.parts or (info.external_attr >> 16) & 0o170000 == 0o120000
                    or info.file_size > 100 * 1024 * 1024):
                raise ValueError("unsafe or oversized archive member")
            if name != "bundle_manifest.json":
                data = archive.read(name)
                spec = manifest["files"][name]
                if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                    raise ValueError("archive member hash mismatch: " + name)
        root.mkdir(parents=True)
        for name in names:
            target = root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return manifest


def environment(root):
    return {**os.environ, "HF_HOME": str(root / "hf_cache"),
            "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


def command(root, name, argv, *, timeout=1800, cwd=None):
    """Bound a process group; keep raw stdout/stderr locally for retrieval."""
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    logfile = evidence / (name + ".log")
    if logfile.exists():
        raise ValueError("stage log already exists; use a fresh run or stage")
    print(f"[ariadne] {name}: starting (limit {timeout}s)", flush=True)
    started = time.monotonic()
    with logfile.open("wb") as stream:
        proc = subprocess.Popen([str(x) for x in argv], cwd=cwd or root, env=environment(root),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
            raise
    write_json(evidence / (name + ".json"), {"argv": [str(x) for x in argv], "returncode": result,
               "wall_seconds": time.monotonic() - started, "log_sha256": sha256(logfile)})
    print(f"[ariadne] {name}: exit {result}", flush=True)
    if result:
        print(logfile.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
        raise RuntimeError(f"{name} failed with exit code {result}")


def check_gpu(root):
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=15, check=True)
    boards = [row for row in result.stdout.splitlines() if row.strip()]
    write_json(root / "evidence/gpu.json", {"boards": boards, "expected": "one NVIDIA A100"})
    if len(boards) != 1 or "A100" not in boards[0]:
        raise ValueError("requested A100 was not assigned; no GPU fallback is permitted")


def setup(root):
    check_gpu(root)
    candidate = shutil.which("python3.12")
    if not candidate and sys.version_info[:2] == (3, 12):
        candidate = sys.executable
    if not candidate:
        raise RuntimeError("Python 3.12 is required; create it before setup")
    uv = shutil.which("uv")
    creation = ([uv, "venv", "--python", candidate, "--seed", str(root / ".venv")]
                if uv else [candidate, "-m", "venv", str(root / ".venv")])
    command(root, "venv", creation, timeout=180)
    python = root / ".venv/bin/python"
    command(root, "pip", [python, "-m", "pip", "install", "--upgrade", "pip"], timeout=300)
    command(root, "torch", [python, "-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0",
                              "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
    command(root, "training_dependencies", [python, "-m", "pip", "install", "-r",
                                              root / "experiments/ms_swift/requirements-train.txt"])
    command(root, "project_packages", [python, "-m", "pip", "install", "-e", root / "experiments/common",
                                        "-e", root / "experiments/ms_swift"], timeout=300)
    command(root, "pip_check", [python, "-m", "pip", "check"], timeout=120)
    command(root, "pip_freeze", [python, "-m", "pip", "freeze"], timeout=120)
    command(root, "cuda", [python, "-c", "import json, torch; assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count()==1; assert 'A100' in torch.cuda.get_device_name(0); "
            "assert torch.cuda.is_bf16_supported(including_emulation=False); "
            "x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); y=x@x; "
            "torch.cuda.synchronize(); assert y[0,0].item()==16; "
            "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))"],
            timeout=120)
    # This is a separate child because the Colab kernel may have an unrelated Torch install.
    command(root, "model_download", [python, "-c", MODEL_DOWNLOAD], timeout=1800)


MODEL_DOWNLOAD = r'''
import hashlib, json
from pathlib import Path
from huggingface_hub import hf_hub_download
root = Path.cwd()
assets = root / 'experiments/ms_swift/.cache/model_assets/gemma4-e2b'
lock = json.loads((assets / 'model.lock.json').read_text())
weights = json.loads((root / 'experiments/ms_swift/gemma_eval/model_weights.lock.json').read_text())
files = {**lock['files'], weights['filename']: weights}
verified = {}
for name, spec in files.items():
    path = Path(hf_hub_download(repo_id=lock['repository'], revision=lock['revision'], filename=name, token=False))
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            sha.update(block)
    assert path.stat().st_size == spec['bytes'] and sha.hexdigest() == spec['sha256'], name
    verified[name] = {'sha256':sha.hexdigest(), 'bytes':path.stat().st_size}
(root / 'evidence/model_verification.json').write_text(json.dumps({'repository':lock['repository'], 'revision':lock['revision'], 'files':verified},indent=2))
print(json.dumps({'verified_files':len(verified),'revision':lock['revision']}))
'''


def prepare(root, *, training=False, licenses=()):
    check_gpu(root)
    python, work = root / ".venv/bin/python", root / "experiments/ms_swift"
    run = work / ".cache/colab" / ("training" if training else "preview")
    argv = [python, "-m", "cpt", "sft-prepare", "--input", root / "inputs/records.jsonl",
            "--source-corpus", root / "inputs/sources.jsonl", "--model-assets", work / ".cache/model_assets/gemma4-e2b",
            "--sequence-length", "4096", "--output", run / "prepared", "--mode", "training" if training else "preview"]
    for flag, filename in (("--reviews", "reviews.jsonl"), ("--source-reviews", "source_reviews.jsonl")):
        path = root / "inputs" / filename
        if training and not path.is_file():
            raise ValueError("real human review receipts are required: " + filename)
        if path.is_file():
            argv.extend([flag, path])
    for license_name in licenses:
        argv.extend(["--allow-license", license_name])
    stage = "training" if training else "preview"
    command(root, stage + "_prepare", argv, timeout=180, cwd=work)
    command(root, stage + "_plan", [python, "-m", "cpt", "plan", "--prepared", run / "prepared", "--recipe",
            work / "configs/gemma4_e2b_sft_lora_4k_smoke.json", "--output", run / "plan"], timeout=120, cwd=work)
    if not training:
        command(root, "preview_template", [python, "-c", PREVIEW_TEMPLATE], timeout=180)
    return run


def train(root, licenses):
    run = prepare(root, training=True, licenses=licenses)
    python, work = root / ".venv/bin/python", root / "experiments/ms_swift"
    command(root, "training", [python, "-m", "cpt.profile", "train", "--plan", run / "plan/launch_plan.json",
            "--output", run / "profile", "--max-seconds", "900", "--interval", "2", "--execute"],
            timeout=960, cwd=work)
    command(root, "checkpoint_reload", [python, "-c", RELOAD], timeout=300)


PREVIEW_TEMPLATE = r'''
import json
from pathlib import Path
from cpt.swift_verify import verify_template
root = Path.cwd()
plan_path = root / 'experiments/ms_swift/.cache/colab/preview/plan/launch_plan.json'
result = verify_template(json.loads(plan_path.read_text()))
result['data_admission_granted'] = False
result['scope_note'] = 'Real pinned SWIFT tokenizer/template check on pending preview data; no weight update or human review.'
(root / 'evidence/preview_template_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


RELOAD = r'''
import hashlib, json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration
root = Path.cwd()
run = root / 'experiments/ms_swift/.cache/colab/training'
plan = json.loads((run / 'plan/launch_plan.json').read_text())
adapters = sorted((run / 'plan/checkpoints').rglob('adapter_model.safetensors'))
if len(adapters) != 1:
    raise ValueError('expected exactly one retained adapter, found %d' % len(adapters))
checkpoint = adapters[0].parent
state = json.loads((checkpoint / 'trainer_state.json').read_text())
assert state['global_step'] == plan['recipe']['parameters']['max_steps'] == 20
assets = Path(plan['model_assets'])
config = AutoConfig.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
model = Gemma4ForConditionalGeneration.from_pretrained(plan['recipe']['model']['repo'],
    revision=plan['recipe']['model']['revision'], config=config, local_files_only=True,
    trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa')
model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
tokens = tokenizer('A short engineering reload test.', return_tensors='pt')['input_ids'].to('cuda')
with torch.inference_mode():
    logits = model(input_ids=tokens, use_cache=False).logits
torch.cuda.synchronize()
assert torch.isfinite(logits).all().item()
assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
result = {'schema_version':'colab-sft-reload-v1', 'checkpoint':str(checkpoint),
    'global_step':state['global_step'], 'adapter_sha256':hashlib.sha256(adapters[0].read_bytes()).hexdigest(),
    'checkpoint_reload_verified':True, 'finite_logits':True, 'logits_shape':list(logits.shape),
    'quality_improvement_measured':False,
    'note':'Successful reload and finite forward pass only; this is not an accuracy or long-context evaluation.'}
(root / 'evidence/checkpoint_reload_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


def archive_results(root):
    output = root / "artifacts.zip"
    # Never include environments, HF cache/base weights or CLI session credentials.
    files = [root / "bundle_manifest.json"]
    for relative in ("evidence", "inputs", "experiments/ms_swift/.cache/colab"):
        directory = root / relative
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink())
    files = [path for path in files if path.is_file()]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--action", choices=("setup", "preview", "train", "archive"), required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--allow-license", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this script runs on the Linux Colab VM")
    if args.action == "train" and not args.execute:
        parser.error("training additionally requires --execute and admitted human-reviewed data")
    root = run_root(args.run_id)
    if args.action == "setup":
        if args.bundle is None:
            parser.error("setup requires --bundle and --bundle-sha256")
        unpack(args.bundle, args.bundle_sha256, root)
    elif not (root / "bundle_manifest.json").is_file():
        parser.error("setup must create this run directory first")
    status = {"action": args.action, "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "training_requested": args.action == "train",
              "driver_sha256": sha256(__file__)}
    report = root / "evidence" / (args.action + "_status.json")
    write_json(report, status)
    try:
        if args.action == "setup":
            setup(root)
        elif args.action == "preview":
            prepare(root, licenses=args.allow_license)
        elif args.action == "train":
            train(root, args.allow_license)
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report, status)
        archive_results(root)


if __name__ == "__main__":
    main()
