# Colab A100 paired long-context pilot

These new drivers preserve the earlier 10-record smoke experiment unchanged.
They do not allocate a Colab runtime. The operator creates one named A100
session, retrieves artifacts after every phase, and stops that owned session.

## Independent resource validation

```powershell
experiments/ms_swift/.venv/Scripts/python.exe experiments/colab/paired_bundle.py `
  --repo D:/code/Ariadnian-agent --resource-only --output <fresh-resource-bundle.zip>
```

Upload the ZIP and the **same ZIP's copies** of `paired_job.py`, `remote_job.py`
and `training_audit.py` into one bootstrap directory on Colab. Do not mix a
historical bundle with later workspace versions of the bootstrap files.

```bash
python /content/paired_job.py --run-id resources_v1 --action setup \
  --bundle /content/resource-bundle.zip --bundle-sha256 <printed-sha256>
python /content/ariadne_paired_resources_v1/experiments/colab/paired_job.py \
  --run-id resources_v1 --action probe --execute
```

Setup verifies exactly one A100, creates a separate Python 3.12 environment,
installs the previously successful pinned CUDA 12.8 / Torch 2.8 / SWIFT commit,
and verifies the immutable Gemma Base weights and metadata. It does not change
the Colab notebook kernel's Python environment.

Resource training runs a fresh disposable LoRA for each of 4K, 8K and 16K
**actual input token** lengths. Each performs three real forward, backward and
AdamW updates, batch size one, with gradient checkpointing. Deterministic token
inputs and the final 32 next-token targets are engineering workloads. They are
not admitted QA, corpus learning, or task accuracy evidence. The direct HF/PEFT
language-model-linear target selection is recorded; it is not an assertion that
every SWIFT pilot allocation or objective is identical. No probe adapter is saved.

Inference probes run fresh Base at 4K, 8K, 16K and 32K and generate exactly 32
tokens, with KV cache enabled. Reported memory is the actual allocator peak;
separate two-second whole-board samples include loading and other overhead.
Timeouts, OOMs and failed lengths remain in the resource summary.

## Reviewed paired pilot

Build a fresh bundle with explicit `--short-records`, `--long-records`,
`--sources`, `--short-reviews`, `--long-reviews`, `--source-reviews`,
`--eval-records` and `--eval-manifest`. `--pair-manifest` is optional provenance.
Never edit an old bundle or migrate old absolute-path launch plans in place.

After setup of that run:

```bash
DRIVER=/content/ariadne_paired_pilot_v1/experiments/colab/paired_job.py
python "$DRIVER" --run-id pilot_v1 --action prepare --allow-license <exact-license>
python "$DRIVER" --run-id pilot_v1 --action train-short --execute
python "$DRIVER" --run-id pilot_v1 --action train-long --execute
python "$DRIVER" --run-id pilot_v1 --action eval --eval-arm all --execute
```

Repeat `--allow-license` for each approved exact identifier. A preview, missing
review, AI-only review, stale receipt or dropped pair prevents the paired run.
The real pinned SWIFT template must confirm short inputs at most 2K, long inputs
4–8K, identical answer labels/order and aligned sources/splits. Both plans are
fresh Base plus LoRA with seed 17, batch 1, accumulation 8, and two epochs.

The actual reviewed count determines `2 * ceil(train_pairs / 8)` optimizer
steps. For 32 train / 8 dev this is eight steps per arm, 64 consumed sequences
per arm. The earlier 128 / 32 proposal would be 32 steps; the runner records the
sample-size amendment and never duplicates pairs to fill that proposal.

The separately SHA-bound audit entrypoint runs the unchanged pinned SWIFT CLI
after the existing plan, data, runtime and template guards. An in-memory hook
observes each actual collated microbatch before SWIFT mutates it. It records
input-token counts, causal unmasked-label counts and hashes, backward completion,
optimizer steps, memory and timing. It never edits the upstream installation or
training tensors. CUDA synchronization adds measurement overhead to both arms.
Final comparison rejects different consumed answer labels/order or skipped
updates. Both adapters must reload and produce finite CUDA logits before eval.

`--action eval --eval-arm base` can run the frozen Base comparison independently;
`short` and `long` require successful lineage-bound adapter reloads. The evaluator
uses the same no-system `gemma4_nothinking` chat and frozen request manifest for
all arms, including failed calls in its denominator.

## Bounds and artifacts

- Per resource point: training at most 900 seconds, inference at most 600 seconds.
- Per SFT arm: 1,800 seconds, followed by a 600-second reload bound.
- Per evaluation arm: 2,700 seconds inside the existing bounded evaluator.
- Driver wall ceiling: six hours, including setup and gaps between phases.
- Successful phases can be skipped using their unchanged success receipts;
  failed attempts remain immutable and require a fresh run ID.

After every outer phase, `/content/ariadne_paired_<id>/artifacts.zip` contains
evidence, explicit inputs, prepared data, plans, audit JSONLs, evaluations and
checkpoints. It excludes the environment, base model cache and credentials.
Download it before `colab stop`. The six-hour guard does not itself release the
account runtime, and there is no automatic polling or rental in these scripts.

CPU validation:

```powershell
experiments/ms_swift/.venv/Scripts/python.exe -m unittest discover -s experiments/colab/tests -v
```

Passing CPU tests establishes guard/audit logic only. A real GPU report is
required for resource feasibility; a frozen held-out comparison is required for
any model-improvement claim.
