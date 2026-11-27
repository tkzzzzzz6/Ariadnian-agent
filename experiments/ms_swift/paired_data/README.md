# Paired source-bound SFT candidates

This CPU-only package creates short/long pairs from explicitly authored local QA
specifications. It does not call a model, synthesize padding, grant human approval,
or produce an executable training export. Every source is copied unchanged from
the input corpus. Question, answer, source IDs, split and row order match across
the two arms. Evidence quotes are exact substrings of original source blocks.

Run from `experiments/ms_swift`:

```powershell
uv run python -m paired_data --corpus ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl --specs .cache/paired_pilot_20261108_v1/authored_specs_v2.jsonl --quarantine .cache/review_20261107_v2/overlap_candidates.jsonl --assets .cache/model_assets/gemma4-e2b --output .cache/paired_pilot_20261108_v1/new_review
uv run pytest tests/test_paired_data.py
```

The output directory must be new or empty. Keep historical packets intact. Source
IDs involved in the supplied overlap-candidate file are quarantined, even if a
candidate might later prove to be harmless boilerplate. Test sources are rejected.
Cross-document edges merge source families, translations with an existing shared
family stay together, and a connected family may supply at most four questions.
This is file/chapter/DOI isolation, not entire repository/book isolation.

Each ignored authored specification contains `id`, `source_document_ids`,
`direction`, `language`, `question`, `answer`, `rationale`, and `evidence` entries
with `document_id` and `block_index`. An optional exact `quote` shortens an evidence
block in the short arm; the long arm retains the full original block. Optional
`task_subtype` distinguishes multi-evidence comparison from intended multihop
reasoning. `target_long_tokens` is a desired message-input length, not permission
to fabricate additional source content.

The native tokenizer counts message contents; these are not full chat-template
lengths. Short content is capped at 1950 tokens and long content at 8000, leaving
room below the desired 2048/8192 sequence limits. The real MS-SWIFT template must
still verify lengths and equal answer-label token IDs before training.

`short.jsonl` and `long.jsonl` obey `cpt.sft`'s raw-record schema. The existing
`reviewed_synthetic_qa` kind is a schema category, **not** an approval statement:
all generated answer/source/license receipts remain pending. Both source review
and each arm's exact record hashes must be approved by a human before a separate
`cpt.sft.prepare_sft(..., mode="training")` call can admit data.

`review_packet.lock.json` binds the complete review presentation, source pages,
raw data, pending receipts, and source/data locks. Reviewers must inspect the
actual answers, necessary inference, full source/license evidence, and both
contexts. Mechanical substring and schema checks never establish reasoning
correctness or model improvement. Record-level exports remain ignored; only
small aggregate reports and source locks belong in Git.
