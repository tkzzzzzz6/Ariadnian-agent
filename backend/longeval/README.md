# Long-document evaluation

This package evaluates **context selection** offline and scores externally
provided model predictions. The default does not call models; an explicit live
mode is available. Neither mode updates weights or establishes continued-pretraining
gains. The offline mode has no default dependency beyond Python 3.11.

All context-selection candidates are restricted to each task's `document_ids`.
These results measure evidence retrieval **within given documents or a given
document group**, not document retrieval over the entire corpus. The `structure`
strategy adds lexical weights for document titles and `section_path` to BM25;
it does not perform neighboring-block expansion. Controlled quote-location
diagnostics show the exact source quotations in the question and cannot establish
long-distance reasoning, factual QA, or pretrained model capability gains.

From `backend/`:

```powershell
python -m longeval validate --documents data/documents.jsonl --tasks data/tasks.jsonl --output-dir results/check
python -m longeval build-diagnostics --documents data/documents.jsonl --output data/diagnostics.jsonl
python -m longeval run --documents data/documents.jsonl --tasks data/tasks.jsonl --strategy bm25 --budget-bytes 16384 --output-dir results/bm25
python -m longeval run --documents data/documents.jsonl --tasks data/tasks.jsonl --strategy structure --budget-bytes 16384 --output-dir results/structure
python -m longeval run --documents data/documents.jsonl --tasks data/tasks.jsonl --strategy truncate --budget-bytes 16384 --output-dir results/truncate
python -m longeval run --documents data/documents.jsonl --tasks data/tasks.jsonl --predictions data/model-predictions.jsonl --output-dir results/model
python -m longeval run --documents data/documents.jsonl --tasks data/tasks.jsonl --tokenizer path/to/tokenizer.json --budget-tokens 8192 --output-dir results/reference-token-budget
python -m longeval run --documents data/documents.jsonl --tasks data/tasks.jsonl --use-configured-model --limit 6 --max-output-tokens 2048 --output-dir results/live
```

## Contracts

Each `documents.jsonl` line is an object:

```json
{"id":"doc-1","family_id":"family-1","title":"Example","source_url":"https://example.org/source","license":"CC-BY-4.0","revision":"2026-11-05","language":"en","domain":"science","kind":"paper","split":"test","blocks":[{"id":"b1","type":"paragraph","text":"The measured temperature was 20 degrees.","section_path":["Results"],"page":1}]}
```

Each task line is an object:

```json
{"id":"task-1","task_type":"extractive_qa","document_ids":["doc-1"],"question":"What temperature was measured?","reference_answer":"20 degrees","evidence":[{"document_id":"doc-1","block_id":"b1","quote":"temperature was 20 degrees"}],"answerable":true,"language":"en","annotation_status":"unreviewed","split":"test"}
```

`annotation_status` is one of `human_verified`, `ai_generated`, `unreviewed`,
`controlled_synthetic`. A generated reference is not human-verified merely
because its quote exists. Only use `human_verified` after an actual human review.
`source_url`, `license`, and `revision` may be null for assessment-only inputs,
with explicit warnings; null does not establish source or training rights.
Additional provenance, confidence, annotator, source and difficulty metadata are
allowed. `reference_answers` optionally contains alternative acceptable answers.
Missing evidence gold is excluded from evidence metrics, never counted as a pass.
An unanswerable example has `answerable: false` and empty supporting `evidence`.

Task types with answer EM/F1: `qa`, `long_qa`, `cross_document_qa`,
`cross_section_qa`, `extractive_qa`, `multiple_choice`, `unanswerable`.
Other types are evidence-location diagnostics. Add task types deliberately;
do not convert reasoning tasks to quote lookup and report them as reasoning.

External predictions use the task ID and may omit the quote to indicate the
entire original block. A quote, when supplied, must be a literal substring:

```json
{"id":"task-1","answer":"20 degrees","evidence":[{"document_id":"doc-1","block_id":"b1","quote":"temperature was 20 degrees"}],"predicted_answerable":true,"status":"ok"}
```

Explicit abstention is `predicted_answerable: false` or `abstain: true`.
Blank output alone is not interpreted as a valid rejection. Failed or missing
predictions score zero on applicable evidence/answer metrics and lower coverage.
Contradictory imported predictions (for example, an answer combined with
`predicted_answerable: false`, or disagreement between the two abstention flags)
are schema failures, never correct rejections. The full document/task input is
validated before `--split` or `--limit` is applied, so a subset run cannot hide
invalid labels or cross-split leakage elsewhere in the supplied dataset.
Without imported predictions, no QA metric is reported: the three offline
strategies select source text only, with `answer: null`.

## Splits and measurement

All revisions in a `family_id`, declared duplicate cluster (`duplicate_cluster_id`,
`near_duplicate_cluster_id`, or `cluster_id`), and normalized exact text copies
must share a split. Derivatives are unioned with `source_document_ids` and
`parent_family_ids`, so a compilation cannot cross its included sources' splits.
Missing parents produce partial-lineage-verification warnings. Conflicts fail
validation. Missing splits are deterministically
inferred by family hash (80/10/10 expected proportions); small datasets need not
be balanced. `validation.json`/`summary.json` exposes resolved splits. Tasks must
match every document's split. Split **before** generating related examples.
Undeclared near-duplicates cannot be discovered by this evaluator; use the data
pipeline's duplicate clusters. Identical task questions across splits are rejected.

The default length is **UTF-8 bytes, not model tokens**. The budget counts selected
block text only, excluding metadata and prompt overhead. `--tokenizer` uses an
already local tokenizer for length analysis without downloading. With
`--budget-tokens`, selected context uses that tokenizer's budget; otherwise the
selection budget remains bytes. Neither count certifies a model fits a full prompt.
Distance bins use UTF-8 byte offsets within a document, with cross-document
evidence in its own category. Character span metrics use Unicode offsets.

Evidence recall/precision are macro averages over unique gold/predicted blocks.
A gold block is hit only when **all its gold quotation spans** are fully covered
by selected text. Selecting the right block but truncating away its evidence does
not count. Complete hit requires every gold block. Span precision/recall measures
the union of intersected selected/gold character spans, so entire-block selection
is distinguished from precise quotation. Duplicate selections do not inflate hits.
Gold or predicted evidence can include `start_char`/`end_char` to identify an
exact original-block Unicode character interval. Invalid offsets fail validation.
Without offsets, all literal occurrences are considered and overlap-based local
alignment chooses a matching occurrence, rather than always the first mention.
One ambiguous prediction is never multiplied into several separate occurrences.
Ambiguous distance annotations have their own bucket; precise position analysis
requires explicit offsets. `ambiguous_quote_entries` exposes each affected case.
Answer EM/F1 ignores punctuation/case and tokenizes English words plus Chinese
characters/bigrams; it is not a semantic or reasoning-quality judge.

Outputs are `predictions.jsonl` (per-example scores and actual selected text),
`summary.json`, and `report.md`. Scores are [0,1]. Slices cover annotation status,
language, domain, task type, split, length and evidence distance. The mixed-label
overall score must not be presented as independently verified gold accuracy.

## Controlled diagnostics

`build-diagnostics` generates exact-quote lookup, two-section joint location,
and verified absent-marker tasks. These are explicitly `controlled_synthetic`,
with visible source quotes in the query. They test source selection and bookkeeping,
not factual QA, human annotation quality, or long-distance reasoning.

## Local benchmark adapters

```powershell
python -m longeval import-benchmark --benchmark longbench-v2 --input local-benchmark.jsonl --documents-out data/benchmark-documents.jsonl --tasks-out data/benchmark-tasks.jsonl --license unknown --source-url https://huggingface.co/datasets/THUDM/LongBench-v2
```

`ruler` imports generated `input`/`outputs` JSONL; `longbench-v2` imports `context`,
`question`, `choice_A` through `choice_D`, and `answer`. These commands do not
download data. Both adapters retain original IDs/length metadata, mark all examples
as test, and set `official_full_run: false`. Original benchmark length is words
for LongBench v2 and generator-tokenizer tokens for RULER; this is distinct from
locally measured bytes/tokens. LongBench source labels remain `unreviewed` locally;
RULER labels are `controlled_synthetic`. Neither adapter invents evidence spans.
Do not train on imported test data, interpret a partial run as an official full
score, or use generic EM/F1 as a substitute for official RULER aggregation.

## Explicit live mode

Only `--model NAME` or `--use-configured-model` enables inference. Live mode also
requires explicit `--limit N`; `--max-output-tokens` (default 2048) and `--timeout`
(default 60 seconds) bound each request. It reuses the app's configured tier
credentials and transport in a temporary evaluation directory. It does not alter
the user's model settings, library, or documents. Configured fallback models and
transport retries are disabled: one capped request per case. Failed/unknown JSON,
invented block IDs, quotes absent from selected context, and timeout are failures,
retained in all applicable score denominators. Empty model output is not a valid
abstention. Only the question, task type and selected sources enter prompts; gold
answers/evidence/answerability do not. JSON must include `answer`,
`predicted_answerable` and `evidence`. Insufficient evidence requires empty answer
and evidence with `predicted_answerable: false`.

`--thinking default|disabled|enabled` defaults to preserving provider behavior.
An explicit mode adds `extra_body={"thinking":{"type":"disabled"}}` (or
`enabled`) to that bounded request and records the setting in per-case parameters
and the run manifest. It does not change saved model settings. Provider errors
remain failures; there is no uncapped retry or silent switch of thinking mode.

Results include actual raw outputs (redacted), source selection, model and prompt
versions/hashes, input/implementation hashes, finish reason, latency, and provider
token/cost usage when available. Answers and quotes are scored in memory before
every outgoing prediction/metadata string is recursively redacted with credentials
captured inside the live app configuration. Thus redaction cannot turn a legitimate
quote into an incorrect prediction during the original score. `storage_redaction`
records this ordering; secret values never enter serialized metadata. Normally
`model-predictions.jsonl` can be rescored offline; deliberately redacted quote text
may differ from its raw source and cannot support byte-identical rescoring.
The JSON usage summary states how many calls reported usage/cost; unavailable
provider usage/cost must not be interpreted as zero expenditure. Nothing here
turns controlled quote-location diagnostics into reasoning gold.

`--tokenizer PATH` accepts a local `tokenizer.json` through optional `tokenizers`,
or a local Transformers tokenizer directory. It never downloads. Use
`--budget-tokens N` to bound selected context text with that reference tokenizer;
inter-block separators count toward that budget, but question, source metadata,
system prompt and chat template overhead do not. `--budget-bytes` applies when no
token budget is requested. A Qwen reference tokenizer used with a DeepSeek model
does **not** measure DeepSeek's native input tokens. The manifest preserves
tokenizer artifact hashes and reports provider-native usage separately.

Diagnostics are sampled with deterministic (`--seed`, default 13) stratified
round-robin selection over language, domain, source kind and byte-length bucket,
instead of allowing the first domain in a sorted input file to fill the limit.
