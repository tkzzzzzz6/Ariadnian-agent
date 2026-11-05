# Long-document evaluation

Mode: `live_model`. Length unit: `reference_tokenizer_tokens:D:\code\Ariadnian-agent\backend\longdoc_sources\.cache\public_v1\tokenizer.json`.

Reference tokenizer: `D:\code\Ariadnian-agent\backend\longdoc_sources\.cache\public_v1\tokenizer.json`. Its counts are not the target model's native input tokens.

Live model: `openai/deepseek-flash`. Prompt: `ariadne-longdoc-qa-v1`. Thinking setting: `disabled`.

Provider-reported usage (unknown values remain distinct from measured zeros):

```json
{
  "prompt_tokens": 71752,
  "completion_tokens": 3578,
  "total_tokens": 75330,
  "cached_tokens": 34944,
  "cost_usd": null,
  "usage_known_calls": 6,
  "cost_known_calls": 0
}
```

Metrics use [0, 1]. Missing/failed predictions count as evidence and eligible answer failures; answer coverage remains explicit.

| Slice | n | prediction_coverage | evidence_recall | evidence_precision | evidence_complete_hit | answer_em | answer_f1 | answer_coverage | unanswerable_rejection_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Overall (mixed labels; inspect slices) | 6 | 0.833 | 0.583 | 0.406 | 0.333 | 0.000 | 0.510 | 0.833 | — |
| annotation_status: ai_generated | 6 | 0.833 | 0.583 | 0.406 | 0.333 | 0.000 | 0.510 | 0.833 | — |
| language: en | 3 | 1.000 | 0.833 | 0.478 | 0.667 | 0.000 | 0.640 | 1.000 | — |
| language: zh | 3 | 0.667 | 0.333 | 0.333 | 0.000 | 0.000 | 0.381 | 0.667 | — |
| domain: agriculture | 1 | 1.000 | 0.500 | 0.333 | 0.000 | 0.000 | 0.620 | 1.000 | — |
| domain: data_science | 2 | 1.000 | 1.000 | 0.550 | 1.000 | 0.000 | 0.650 | 1.000 | — |
| domain: machine_learning | 3 | 0.667 | 0.333 | 0.333 | 0.000 | 0.000 | 0.381 | 0.667 | — |
| length_bucket: 8k-32k | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| length_bucket: <8k | 5 | 1.000 | 0.700 | 0.487 | 0.400 | 0.000 | 0.612 | 1.000 | — |
| evidence_distance_bucket: 8k-32k_bytes | 2 | 1.000 | 0.750 | 0.417 | 0.500 | 0.000 | 0.663 | 1.000 | — |
| evidence_distance_bucket: <8k_bytes | 3 | 1.000 | 0.667 | 0.533 | 0.333 | 0.000 | 0.578 | 1.000 | — |
| evidence_distance_bucket: cross_document | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| task_type: cross_document_qa | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| task_type: cross_section_qa | 5 | 1.000 | 0.700 | 0.487 | 0.400 | 0.000 | 0.612 | 1.000 | — |
| split: dev | 2 | 0.500 | 0.167 | 0.167 | 0.000 | 0.000 | 0.262 | 0.500 | — |
| split: test | 4 | 1.000 | 0.792 | 0.525 | 0.500 | 0.000 | 0.635 | 1.000 | — |

## Interpretation

- Offline context selection does not generate or evaluate model answers.
- Controlled synthetic location checks are not human-verified reasoning gold.
- Evidence block hits require full gold quote coverage; span metrics use Unicode character offsets.
- Context budget excludes metadata and prompt formatting; token budgets include inter-block separators.
- Human-verified, AI-generated and unreviewed labels are reported separately.
- Inferred family splits are deterministic and may be imbalanced in small datasets.
- Candidates are restricted to task.document_ids: within-given-document/group evidence retrieval, not full-corpus document retrieval.
- The structure strategy adds title/section_path lexical weighting to BM25; it does not expand neighboring blocks.
- Controlled synthetic quote-location questions expose the exact quotations and do not establish reasoning capability.
- Local/reference tokenizer counts are not the target model's actual input token usage; provider-reported usage is separate.
- Model answers and diagnostic location scores are inference results; no CPT or weight update was performed.
