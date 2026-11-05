# Long-document evaluation

Mode: `offline_context_selection_only`. Length unit: `reference_tokenizer_tokens:D:\code\Ariadnian-agent\backend\longdoc_sources\.cache\public_v1\tokenizer.json`.

Reference tokenizer: `D:\code\Ariadnian-agent\backend\longdoc_sources\.cache\public_v1\tokenizer.json`. Its counts are not the target model's native input tokens.

Metrics use [0, 1]. Missing/failed predictions count as evidence and eligible answer failures; answer coverage remains explicit.

| Slice | n | prediction_coverage | evidence_recall | evidence_precision | evidence_complete_hit | answer_em | answer_f1 | answer_coverage | unanswerable_rejection_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Overall (mixed labels; inspect slices) | 540 | 1.000 | 0.728 | 0.015 | 0.532 | — | — | 0.000 | 0.000 |
| annotation_status: controlled_synthetic | 540 | 1.000 | 0.728 | 0.015 | 0.532 | — | — | 0.000 | 0.000 |
| language: en | 472 | 1.000 | 0.717 | 0.015 | 0.520 | — | — | 0.000 | 0.000 |
| language: zh | 68 | 1.000 | 0.806 | 0.015 | 0.611 | — | — | 0.000 | 0.000 |
| domain: agriculture | 61 | 1.000 | 0.653 | 0.024 | 0.449 | — | — | 0.000 | 0.000 |
| domain: data_science | 15 | 1.000 | 0.955 | 0.011 | 0.909 | — | — | 0.000 | 0.000 |
| domain: life_sciences | 71 | 1.000 | 0.645 | 0.022 | 0.436 | — | — | 0.000 | 0.000 |
| domain: machine_learning | 214 | 1.000 | 0.774 | 0.014 | 0.571 | — | — | 0.000 | 0.000 |
| domain: mathematics | 52 | 1.000 | 0.725 | 0.014 | 0.550 | — | — | 0.000 | 0.000 |
| domain: numerical_computing | 46 | 1.000 | 0.729 | 0.014 | 0.543 | — | — | 0.000 | 0.000 |
| domain: programming | 81 | 1.000 | 0.700 | 0.006 | 0.492 | — | — | 0.000 | 0.000 |
| length_bucket: 32k-128k | 24 | 1.000 | 0.237 | 0.017 | 0.105 | — | — | 0.000 | 0.000 |
| length_bucket: 8k-32k | 325 | 1.000 | 0.632 | 0.013 | 0.337 | — | — | 0.000 | 0.000 |
| length_bucket: <8k | 191 | 1.000 | 1.000 | 0.019 | 1.000 | — | — | 0.000 | 0.000 |
| evidence_distance_bucket: 8k-32k_bytes | 63 | 1.000 | 0.929 | 0.022 | 0.873 | — | — | 0.000 | — |
| evidence_distance_bucket: <8k_bytes | 30 | 1.000 | 0.900 | 0.033 | 0.900 | — | — | 0.000 | — |
| evidence_distance_bucket: >=32k_bytes | 54 | 1.000 | 0.537 | 0.016 | 0.074 | — | — | 0.000 | — |
| evidence_distance_bucket: cross_document | 150 | 1.000 | 0.543 | 0.010 | 0.180 | — | — | 0.000 | — |
| evidence_distance_bucket: single_or_no_evidence | 243 | 1.000 | 0.889 | 0.011 | 0.889 | — | — | 0.000 | 0.000 |
| task_type: absent_marker | 117 | 1.000 | — | — | — | — | — | 0.000 | 0.000 |
| task_type: cross_document_evidence | 150 | 1.000 | 0.543 | 0.010 | 0.180 | — | — | 0.000 | — |
| task_type: cross_section_evidence | 117 | 1.000 | 0.756 | 0.020 | 0.513 | — | — | 0.000 | — |
| task_type: evidence_lookup | 126 | 1.000 | 0.889 | 0.011 | 0.889 | — | — | 0.000 | — |
| task_type: table_text_evidence | 30 | 1.000 | 0.867 | 0.030 | 0.867 | — | — | 0.000 | — |
| split: dev | 81 | 1.000 | 0.807 | 0.015 | 0.629 | — | — | 0.000 | 0.000 |
| split: test | 60 | 1.000 | 0.696 | 0.015 | 0.471 | — | — | 0.000 | 0.000 |
| split: train | 399 | 1.000 | 0.715 | 0.015 | 0.520 | — | — | 0.000 | 0.000 |

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
