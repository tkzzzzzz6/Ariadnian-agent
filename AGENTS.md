# Repository guidance

## Local assessment data

Downloaded corpora, generated tasks and individual predictions are intentionally
excluded from Git. An ignored file remains readable by local agents and scripts;
an empty `git status` or ordinary `rg --files` result does not establish that the
data are absent. Use the known data path directly, or restrict `rg --no-ignore`
to the relevant data directory.

Common local paths:

- `backend/longdoc_sources/.cache/public_v1/`: original public documents, raw
  sources and the reference tokenizer.
- `backend/longdoc_sources/.cache/pipeline_v1/`: processed corpus and data audits.
- Other `backend/longdoc_sources/.cache/` subdirectories: diagnostics, generated
  QA, AI review candidates and source-linked compilations.
- `backend/publicbench/.cache/`: downloaded public benchmarks and local imports.
- `backend/evals/results/longdoc_*/` and `publicbench_*/`: run summaries plus local
  record-level data and predictions. Their small manifests/summaries/reports and
  source locks are retained as versionable provenance.
- `backend/evals/results/*.negative-candidates.jsonl` and new timestamped P0 v2
  case JSONs: local evidence and review candidates.

For example, `rg --files --no-ignore backend/longdoc_sources/.cache -g '*.jsonl'`
finds local corpora without changing the ignore rules. Read a specific JSONL file
with a script or `Get-Content -LiteralPath <path> -TotalCount 1`. Limit inspection
to task-relevant dataset paths; do not automatically scan `.env`, credentials or
the application user's library/settings in `backend/data/`.

Do not remove local data to make Git clean, and do not force-add ignored data as
part of routine code work. Existing small offline fixtures in
`backend/evals/data/` remain versioned. A fresh clone or remote agent does not
receive ignored data: recreate it using the documented acquisition commands,
pinned source locks and hashes; report unavailable sources rather than assuming
that local cached data are present remotely. See `docs/local-data-access.md`.

## Evidence and labels

Keep controlled quote-location diagnostics, AI-generated/reviewed candidates and
independently human-verified gold distinct. Missing or blank review fields are
not passed checks. Quote/schema validity and engineering tests do not establish
reasoning accuracy or a pretraining-data improvement. Preserve failed calls and
their denominators when reporting model evaluations.
