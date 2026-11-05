# Long-document evaluation

Mode: `offline_context_selection_only`. Length unit: `reference_tokenizer_tokens:D:\code\Ariadnian-agent\backend\longdoc_sources\.cache\public_v1\tokenizer.json`.

Reference tokenizer: `D:\code\Ariadnian-agent\backend\longdoc_sources\.cache\public_v1\tokenizer.json`. Its counts are not the target model's native input tokens.

Metrics use [0, 1]. Missing/failed predictions count as evidence and eligible answer failures; answer coverage remains explicit.

| Slice | n | prediction_coverage | evidence_recall | evidence_precision | evidence_complete_hit | answer_em | answer_f1 | answer_coverage | unanswerable_rejection_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Overall (mixed labels; inspect slices) | 540 | 1.000 | 0.993 | 0.035 | 0.986 | — | — | 0.000 | 1.000 |
| annotation_status: controlled_synthetic | 540 | 1.000 | 0.993 | 0.035 | 0.986 | — | — | 0.000 | 1.000 |
| language: en | 472 | 1.000 | 0.992 | 0.035 | 0.984 | — | — | 0.000 | 1.000 |
| language: zh | 68 | 1.000 | 1.000 | 0.032 | 1.000 | — | — | 0.000 | 1.000 |
| domain: agriculture | 61 | 1.000 | 1.000 | 0.064 | 1.000 | — | — | 0.000 | 1.000 |
| domain: data_science | 15 | 1.000 | 1.000 | 0.016 | 1.000 | — | — | 0.000 | 1.000 |
| domain: life_sciences | 71 | 1.000 | 1.000 | 0.055 | 1.000 | — | — | 0.000 | 1.000 |
| domain: machine_learning | 214 | 1.000 | 1.000 | 0.027 | 1.000 | — | — | 0.000 | 1.000 |
| domain: mathematics | 52 | 1.000 | 0.963 | 0.030 | 0.925 | — | — | 0.000 | 1.000 |
| domain: numerical_computing | 46 | 1.000 | 0.957 | 0.047 | 0.914 | — | — | 0.000 | 1.000 |
| domain: programming | 81 | 1.000 | 1.000 | 0.016 | 1.000 | — | — | 0.000 | 1.000 |
| length_bucket: 32k-128k | 24 | 1.000 | 1.000 | 0.075 | 1.000 | — | — | 0.000 | 1.000 |
| length_bucket: 8k-32k | 325 | 1.000 | 0.993 | 0.035 | 0.985 | — | — | 0.000 | 1.000 |
| length_bucket: <8k | 191 | 1.000 | 0.992 | 0.029 | 0.985 | — | — | 0.000 | 1.000 |
| evidence_distance_bucket: 8k-32k_bytes | 63 | 1.000 | 0.984 | 0.039 | 0.968 | — | — | 0.000 | — |
| evidence_distance_bucket: <8k_bytes | 30 | 1.000 | 0.983 | 0.064 | 0.967 | — | — | 0.000 | — |
| evidence_distance_bucket: >=32k_bytes | 54 | 1.000 | 0.972 | 0.050 | 0.944 | — | — | 0.000 | — |
| evidence_distance_bucket: cross_document | 150 | 1.000 | 1.000 | 0.032 | 1.000 | — | — | 0.000 | — |
| evidence_distance_bucket: single_or_no_evidence | 243 | 1.000 | 1.000 | 0.023 | 1.000 | — | — | 0.000 | 1.000 |
| task_type: absent_marker | 117 | 1.000 | — | — | — | — | — | 0.000 | 1.000 |
| task_type: cross_document_evidence | 150 | 1.000 | 1.000 | 0.032 | 1.000 | — | — | 0.000 | — |
| task_type: cross_section_evidence | 117 | 1.000 | 0.979 | 0.044 | 0.957 | — | — | 0.000 | — |
| task_type: evidence_lookup | 126 | 1.000 | 1.000 | 0.023 | 1.000 | — | — | 0.000 | — |
| task_type: table_text_evidence | 30 | 1.000 | 0.983 | 0.062 | 0.967 | — | — | 0.000 | — |
| split: dev | 81 | 1.000 | 1.000 | 0.027 | 1.000 | — | — | 0.000 | 1.000 |
| split: test | 60 | 1.000 | 1.000 | 0.037 | 1.000 | — | — | 0.000 | 1.000 |
| split: train | 399 | 1.000 | 0.990 | 0.036 | 0.980 | — | — | 0.000 | 1.000 |

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
