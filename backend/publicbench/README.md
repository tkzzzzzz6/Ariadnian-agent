# Public LongBench subset evaluation

This is an independent public-benchmark evaluation. It uses pinned LongBench
v1 task prompts, generation limits and task metrics, plus LongBench v2's official
non-CoT prompt and answer extraction. It does **not** reuse Ariadne's citation-JSON
prompt, invent evidence labels, update weights, or claim a full leaderboard run.

Run commands from `backend/`:

```powershell
uv run python -m publicbench fetch --benchmark both --per-task 5 --v2-per-domain 2 --seed 13 --tokenizer path/to/tokenizer.json --max-context-tokens 65536 --output publicbench/.cache/selected_example
uv run python -m publicbench run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_full --use-configured-model --limit 42 --thinking disabled --strategy full --context-tokens 65536 --tokenizer path/to/tokenizer.json
uv run python -m publicbench rescore --input publicbench/.cache/selected_example/selected_records.jsonl --predictions evals/results/publicbench_example_full/predictions.jsonl --output evals/results/publicbench_example_rescored
```

The default v1 tasks are `qasper`, `multifieldqa_zh`, `hotpotqa`, `gov_report`,
`trec`, and `passage_retrieval_en`: English/Chinese QA, multi-document QA,
summarization, classification and passage retrieval. `--tasks` overrides this list.
For v2, `--v2-per-domain 2` selects two cases from each of six official domains.

## Sampling and source provenance

`fetch` is the only command that downloads public artifacts. Raw archives, official
prompt/max-length configs, scorer code and source metadata are pinned to commits
and verified by SHA-256. The loader never executes downloaded code. Selection uses
a fixed seed and stable IDs before any inference, preserving original references
and raw records. Source/licensing observations accompany the fetch manifest;
benchmark data stays evaluation-only and is not approved for training.

The optional `--max-context-tokens` filter applies **before sampling**, using the
explicit local reference tokenizer. It is a declared length-filtered subset, not
a uniform full-benchmark sample. It never checks answers or model performance to
choose examples. Original text is retained. The manifest records pre/post-filter
counts, eligibility and selected IDs. No failed model case is resampled.

`run` requires an adjacent `fetch_manifest.json`, verifies file SHA, record count
and selected ID order, and validates available raw-record/prompt hashes and current
rendering before any model setup/request. It uses the first `--limit` fixed records without
resampling. It refuses a nonempty output directory. All input/source IDs and
record, prompt, request and implementation hashes are recorded.

## Separate evaluation protocols

`--strategy full` sends the unmodified context using the official source-fixed
task template. The request is one user message, with no added system instructions
or hidden reference answers. The API's normal chat wrapper still applies; this
is a sampled official-prompt protocol, not a promise of byte-identical model
execution with every upstream model-specific wrapper or an official submission.
Per-task official generation limits are used. An optional lower
`--max-output-tokens` labels the run as a modified generation-budget protocol.
The override cannot increase the official limit.

Full contexts exceeding `--context-tokens` are explicitly skipped without a model
call and remain in the denominator. There is no silent truncation or replacement
with a more convenient sample. The guard measures source context with the local
reference tokenizer, excluding question/instruction/chat-template overhead.
It does not certify native model-window fit. Provider window errors are failures;
provider-reported input/output/reasoning/cache usage is recorded separately.

`bm25`, `structure`, and `truncate` are **modified context-budget protocols**:

- Source paragraphs are split into non-overlapping windows of up to 2048 Unicode
  characters (`--chunk-chars`). They are not human-labeled semantic sections.
- BM25 selects text inside the already provided benchmark context, not documents
  from an external corpus. Structure adds lexical weight for inferred headings
  and available titles; it does not expand neighboring blocks.
- Ranked selections are restored to source order. Truncate keeps the prefix.
  Selected text is capped using the reference tokenizer and inserted into the
  otherwise unchanged official task template.
- Source spans, actual selected length, context modification and exact prompt
  hashes are retained. These results must be reported separately from full context.

For an 8K comparison on the same fixed input, run separate outputs:

```powershell
uv run python -m publicbench run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_bm25_8k --use-configured-model --limit 30 --thinking disabled --strategy bm25 --context-tokens 8192 --tokenizer path/to/tokenizer.json
```

Repeat with `structure` and `truncate`. A Qwen reference tokenizer used with a
DeepSeek API model is a reference measurement, not DeepSeek's native token count.
The tokenizer artifact hash/identity and native provider usage are separate.

## Scoring and bounded inference

Model calls require explicit `--use-configured-model` or `--model NAME` and
positive `--limit`. Each case makes at most one capped request with temperature
zero, no transport retry and no fallback model. `--thinking default` preserves
provider behavior; `disabled`/`enabled` passes the explicit thinking setting and
records it. `--timeout` defaults to 60 seconds. No persistent settings are changed.

Answers are plain text. V1 uses the official task-specific normalization/metrics,
including Chinese segmentation and applicable first-line extraction. V2 uses
the official `The correct answer is (A)` / `The correct answer is A` extraction:
an unsupported bare `A` scores zero. No citation-JSON schema is imposed on these
benchmarks. Empty answers, timeouts, malformed provider responses and skipped
cases remain scored zero with separate coverage/status counts. Nonempty but
incorrect answers remain valid completions with their official score.

Reports show per-benchmark/task/domain scores and coverage. V1 macro task scores
are retained; v1/v2 scores are not blended into a cross-benchmark average. A few
examples per task are exploratory estimates, not stable model rankings or proof
of general long-context reasoning gains.

`predictions.partial.jsonl` is flushed and synced after every completed case.
`inprogress_manifest.json` is atomically updated before requests and after cases,
including known usage and any active request whose usage remains unknown.
Interruptions preserve completed outputs and expenditure evidence; a pending
request may have incurred unreported cost. The final files are `predictions.jsonl`,
`summary.json`, `report.md` and the exact selected input records. All outgoing
strings are redacted from a captured credential snapshot **after** raw scoring.
Deliberately redacted text may prevent byte-identical rescoring of that rare case.

`rescore` performs no model calls. It recomputes official scores from saved plain
text, rejects unknown/duplicate IDs and mismatched example hashes, and counts
missing predictions as zero. Prefer the original input cache and matching fetch
manifest, with `--limit 30` when rescoring the same fixed 30-example prefix.
Without a matching fetch manifest, rescoring is explicitly unverified and cannot
be promoted to a verified official-prompt subset result. Use the original fixed input plus partial predictions
to audit an interrupted run, where unfinished cases remain missing.
