# Public long-context benchmark subset

Mode: `live_public_benchmark`. Full official dataset run: **no**.

Model: `openai/deepseek-flash`. Strategy: `structure`. Thinking: `disabled`.

Sampling uses seed `13` and a declared pre-sampling reference-token context filter of `65536`. This is a length-filtered subset, not a uniform sample of the entire benchmark.

Different benchmark protocols/task metrics are shown separately; no combined v1+v2 score is claimed.

| Slice | N | Score (%) | Coverage |
|---|---:|---:|---:|
| by_benchmark: longbench-v1 | 30 | 56.17 | 1.000 |
| by_task: gov_report | 5 | 32.64 | 1.000 |
| by_task: hotpotqa | 5 | 49.33 | 1.000 |
| by_task: multifieldqa_zh | 5 | 73.33 | 1.000 |
| by_task: passage_retrieval_en | 5 | 100.00 | 1.000 |
| by_task: qasper | 5 | 61.71 | 1.000 |
| by_task: trec | 5 | 20.00 | 1.000 |
| by_domain: gov_report | 5 | 32.64 | 1.000 |
| by_domain: hotpotqa | 5 | 49.33 | 1.000 |
| by_domain: multifieldqa_zh | 5 | 73.33 | 1.000 |
| by_domain: passage_retrieval_en | 5 | 100.00 | 1.000 |
| by_domain: qasper | 5 | 61.71 | 1.000 |
| by_domain: trec | 5 | 20.00 | 1.000 |

Statuses: `{"ok": 30}`.

Provider-reported usage (null means unknown):

```json
{
  "prompt_tokens": 182730,
  "prompt_tokens_known_calls": 30,
  "completion_tokens": 2853,
  "completion_tokens_known_calls": 30,
  "total_tokens": 185583,
  "total_tokens_known_calls": 30,
  "cached_tokens": 140158,
  "cached_tokens_known_calls": 30,
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
