# Public long-context benchmark subset

Mode: `live_public_benchmark`. Full official dataset run: **no**.

Model: `openai/deepseek-flash`. Strategy: `full`. Thinking: `disabled`.

Sampling uses seed `13` and a declared pre-sampling reference-token context filter of `65536`. This is a length-filtered subset, not a uniform sample of the entire benchmark.

Different benchmark protocols/task metrics are shown separately; no combined v1+v2 score is claimed.

| Slice | N | Score (%) | Coverage |
|---|---:|---:|---:|
| by_benchmark: longbench-v1 | 30 | 52.70 | 1.000 |
| by_benchmark: longbench-v2 | 12 | 41.67 | 1.000 |
| by_task: gov_report | 5 | 31.79 | 1.000 |
| by_task: hotpotqa | 5 | 49.33 | 1.000 |
| by_task: longbench-v2 | 12 | 41.67 | 1.000 |
| by_task: multifieldqa_zh | 5 | 73.33 | 1.000 |
| by_task: passage_retrieval_en | 5 | 100.00 | 1.000 |
| by_task: qasper | 5 | 61.71 | 1.000 |
| by_task: trec | 5 | 0.00 | 1.000 |
| by_domain: Code Repository Understanding | 2 | 0.00 | 1.000 |
| by_domain: Long In-context Learning | 2 | 50.00 | 1.000 |
| by_domain: Long Structured Data Understanding | 2 | 50.00 | 1.000 |
| by_domain: Long-dialogue History Understanding | 2 | 0.00 | 1.000 |
| by_domain: Multi-Document QA | 2 | 100.00 | 1.000 |
| by_domain: Single-Document QA | 2 | 50.00 | 1.000 |
| by_domain: gov_report | 5 | 31.79 | 1.000 |
| by_domain: hotpotqa | 5 | 49.33 | 1.000 |
| by_domain: multifieldqa_zh | 5 | 73.33 | 1.000 |
| by_domain: passage_retrieval_en | 5 | 100.00 | 1.000 |
| by_domain: qasper | 5 | 61.71 | 1.000 |
| by_domain: trec | 5 | 0.00 | 1.000 |

Statuses: `{"ok": 42}`.

Provider-reported usage (null means unknown):

```json
{
  "prompt_tokens": 580650,
  "prompt_tokens_known_calls": 42,
  "completion_tokens": 3482,
  "completion_tokens_known_calls": 42,
  "total_tokens": 584132,
  "total_tokens_known_calls": 42,
  "cached_tokens": 0,
  "cached_tokens_known_calls": 42,
  "reasoning_tokens": null,
  "reasoning_tokens_known_calls": 0,
  "cost_usd": null,
  "cost_usd_known_calls": 0
}
```

## Interpretation

- A source-fixed official prompt and metric on a sampled subset is not a full official leaderboard result.
- Full-context cases exceeding the declared reference-token guard remain skipped in the denominator; original text is never silently truncated.
- BM25/structure/truncate replace the original context and are a modified budget protocol, reported separately from full context.
- Reference-tokenizer counts are not this API model's native tokens; provider usage is recorded separately.
- The reference context budget excludes question/instruction/chat-template overhead; provider window errors are failures, not hidden truncations.
- Benchmark references are never added to the request; official few-shot examples already in source context remain intact.
- No evidence-span gold is assumed and no citation-JSON output is required.
- No training or weight updates were performed.
