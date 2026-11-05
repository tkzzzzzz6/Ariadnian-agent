# Long-document data experiments

This package is a local corpus construction and audit tool. It performs no
network downloads, model calls, training, or benchmark evaluation. It does not
consume the application's overlapping retrieval chunks.

From `backend/`:

```powershell
.venv\Scripts\python.exe -m longdoc build --input corpus.jsonl --output runs\assessment --sequence-length 8192
```

The default unit is **UTF-8 bytes for diagnostics, not model tokens**. An 8,192
byte pack is not evidence of an 8K model context. Use an existing local Hugging
Face `tokenizer.json` for real token counts; its hash and options are recorded.
Embedded truncation and padding are disabled. No tokenizer is downloaded.

```powershell
.venv\Scripts\python.exe -m longdoc build --input corpus.jsonl --output runs\tokenized --tokenizer C:\models\example\tokenizer.json --sequence-length 8192 --seed 17 --mix science=2 --mix technology=1 --mix-token-budget 200000
```

Each build writes both `document` and `random` packing strategies. Model
tokenization excludes special tokens. Training must add and budget BOS/EOS,
attention masks, labels, and any document-boundary treatment separately.

## Input contract

One JSON object per line. All the fields below are required; `source_url`,
`license`, `revision` and `page` can be null. Metadata may be added. IDs must be
unique, and a document's `family_id` must group its versions and derivatives.

```json
{"id":"source-001-v1","family_id":"source-001","title":"Source title","source_url":"https://example.org/source","license":null,"revision":"v1","language":"en","domain":"science","kind":"report","review_status":"unreviewed","blocks":[{"id":"h1","type":"heading","text":"1 Introduction","section_path":["1 Introduction"],"page":1},{"id":"p1","type":"paragraph","text":"Original paragraph, kept exactly as supplied.","section_path":["1 Introduction"],"page":1}]}
```

Block types: `heading`, `paragraph`, `table`, `formula`, `code`. The original
`text` and block order remain unchanged in corpus exports. Cleanup adds a
separate `cleaned_text` view. Code, tables, formulas and headings retain their
spacing; paragraph cleanup conservatively removes soft hyphens, joins lowercase
line-end hyphenation, normalizes inline spacing and trims outer whitespace.
All changed blocks are audited. No short paragraph, reference section, appendix,
table or formula is automatically discarded.

## Processing and outputs

1. `raw.jsonl`, `cleaned.jsonl`: immutable source blocks, cleaned views,
   source/configuration hashes, selected-tokenizer counts, and quality dimensions.
2. `dedup_candidates.jsonl`, `dedup_removed.jsonl`, `deduped.jsonl`: exact
   matching of cleaned block content/type/section paths; deterministic bottom-32
   hashed 5-token-shingle candidate generation and full Jaccard comparison.
   Near matches are **candidate-only by default**, retaining both documents.
   The optional `--near-action remove` requires the configured similarity
   (default .92), length ratio at least .9, and identical protected structural
   content before automatic removal. This is a heuristic;
   candidate recall and false-removal rates are unmeasured. Removed originals
   remain available in `raw.jsonl` and `cleaned.jsonl` for review and recovery.
3. `quality_rejected.jsonl`, `corpus.jsonl`: optional explainable quality filter.
   The default threshold is zero, so it removes nothing. Replacement/control
   characters and repeated paragraph ratios are proxies, not semantic quality or
   independent review. Length, page and structural coverage are diagnostics.
4. `split_assignments.jsonl`: connected components of document families, exact
   duplicates and threshold-passing near candidates are assigned together with
   seeded hashing, even when candidates are retained for review. High-similarity
   pairs with protected content differences are also linked conservatively for
   split assignment, without removing their source documents. Optional
   `parent_family_ids` and `source_document_ids` connect compilations/derivatives
   to all their parents, including siblings sharing parent keys when parent files
   are absent. Existing `split` and declared duplicate cluster IDs are preserved
   as constraints. Optional `source_splits` and `parent_family_splits` map declared
   parent IDs to known upstream split constraints. Conflicting declared splits
   anywhere in a connected component fail the build before outputs are written.
   Missing parents and the assignment basis are recorded per document; an absent
   parent's upstream split cannot be independently checked without a declaration.
   Components with declared splits keep them; otherwise expected split
   proportions are 80/10/10, not exact counts. Small corpora may have empty splits.
   Check output counts before interpreting any result. Externally held-out
   evaluation sources need a separately enforced denylist; this package does not
   know benchmark provenance automatically.
5. `packed_document.jsonl`, `packed_random.jsonl`: sequences with exact counts,
   fill ratios and source block character spans. Document packing preserves
   source block order; random packing shuffles document segments. Neither
   implements semantic document ordering. No sequence crosses a data split.
   Oversized documents split only at block boundaries. An individual block over
   the limit is explicitly excluded and recorded in `oversize_blocks.jsonl`,
   never silently truncated. The corpus file still contains every source block.
6. `mixed_documents.jsonl`, `packed_mixed.jsonl`, `mixing_report.json`: train-only,
   seeded whole-document sampling without replacement, using remaining token
   deficits toward domain-weight targets. Sampling never overshoots its budget.
   Whole-document granularity and limited supply can prevent exact ratios;
   target/realized counts, selected IDs and unused budget are reported. Counts
   exclude inter-document separators and training special tokens. Subsequent
   packing counts the actual serialized sequence including separators.
7. `training_gate.jsonl`, `manifest.json`: license/review eligibility and build
   parameters, input/output hashes, stage counts, split distributions, units,
   limitations and exact output file inventory. Both build and assessment
   manifests include `implementation_sha256`, captured from this package's Python
   source files at run entry; runtime credentials/settings are not captured.
   `packing_metrics` compares each
   strategy's fill rates, emitted units including separators, source-block units
   retained, explicit oversized exclusions, within-sequence document boundaries,
   multi-document sequences and backward source-block order moves. Source block,
   source document and emitted token counts have different separators; counts
   can be nonadditive under real model tokenization.

Output directories must be absent or empty. Existing runs are not overwritten.
The same input bytes, tokenizer, settings and seed produce identical artifacts.

## Training export gate

Assessment is allowed with unknown license or unreviewed material, subject to
the user's existing access rights. Assessment output is not permission to train
or redistribute it. Training purpose requires all of:

- A real local model tokenizer; byte diagnostics cannot pass this gate.
- An explicit license allowlist supplied for the intended use.
- Every retained document has `review_status: "approved"` and an exactly matching
  license identifier. This field is a caller-supplied review record, not a review
  performed by this package.
- No oversized atomic blocks. Run assessment first to inspect rejects.

```powershell
.venv\Scripts\python.exe -m longdoc build --input reviewed.jsonl --output runs\training --purpose training --tokenizer C:\models\example\tokenizer.json --allow-license CC0-1.0 --allow-license CC-BY-4.0 --sequence-length 8192
```

For an exported training run, only sequences with `split == "train"` belong in
training. Dev and test sequences remain held out. This produces auditable text
sequences, not a ready-made training framework or proof of a training effect.

Run the invariants with:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_longdoc.py -q
```

## Controlled diagnostics and natural-data review

```powershell
.venv\Scripts\python.exe -m longdoc assess --input corpus.jsonl --output runs\assessment-diagnostics --cases-per-type 20 --seed 17
```

This runs three distinct checks and exports blank natural-data review templates:

- **Dedup controlled cases:** exact copies, small-typo near copies, synthetic
  records with shared templates but different core IDs/results, and differing
  protected tables. Each pair is tested independently. Reports include positive
  candidate recall, threshold detection recall, positive removal recall, false
  candidates and false removal on controlled negatives, per-type counts and the
  actual fixtures. Shared-template negatives reveal false candidate detections
  by lexical similarity; the report retains failures. Default candidate-only
  mode preserves near matches; run a separate assessment with
  `--near-action remove` to measure the removal risk of that optional policy.
  These constructed labels are not natural-document semantic gold, and repeated
  template variants are not independent natural samples.
- **Quality perturbation:** inject known replacement characters or duplicate
  paragraphs and measure paired quality-score degradation. A source baseline is
  unreviewed; this measures sensitivity to specific injected defects, not quality
  classification accuracy. Score-floor cases are explicitly inconclusive.
- **Structure integrity:** verify immutable original block text, protected
  content, section/page metadata, emitted character spans and exactly-once
  coverage of nonoversized blocks. Unsupported/unobserved type coverage and
  explicit oversized rejects are reported. This does not measure PDF extraction
  correctness against real-page gold.

Natural candidate pairs and per-document quality reviews are written separately
to `dedup_review_template.jsonl` and `quality_review_template.jsonl`. `label`,
`review_status`, `reviewer`, `review_method`, and `rationale` start empty. Fill
them after review; do not modify `label_source: "natural_unreviewed"`.

Dedup labels: `duplicate`, `distinct`, `uncertain`. Quality labels: `degraded`,
`adequate`, `uncertain`. A completed row needs `review_status: "reviewed"`, a
nonempty reviewer and rationale, and a declared `review_method` from
`independent_human`, `ai_assisted`, or `llm_only`.

```powershell
.venv\Scripts\python.exe -m longdoc score-review --review reviewed_pairs.jsonl --kind dedup --output runs\pair-review-score.json
.venv\Scripts\python.exe -m longdoc score-review --review reviewed_quality.jsonl --kind quality --output runs\quality-review-score.json
```

Coverage is reported over all template rows. Incomplete rows stay unreviewed;
uncertain rows count toward completed coverage but never binary accuracy. Review
methods have separate scores. A declared method is not independently
authenticated. Candidate-only reviews cannot measure missed natural duplicate
pairs; full natural recall requires an independently sampled pair set. These
reviews never automatically set training approval.
