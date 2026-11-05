# Public long-document sources

This package fetches a bounded assessment corpus without reading application API
keys. `catalog.json` describes source coverage. `sources.lock.json` pins Git
commits, paths and Git blob hashes, license observations, the tokenizer revision,
and observed Europe PMC XML body hashes. `--refresh-lock` is explicit.

```powershell
# Run from backend/. All later commands work offline against the cached files.
uv run python -m longdoc_sources fetch --output longdoc_sources/.cache/public_v1
```

Only tokenizer files are downloaded from Hugging Face, never model weights.
Lengths are measured with Qwen2.5-0.5B's pinned tokenizer, without special tokens.
They are reference-tokenizer counts, not the native input lengths of other API
models. `--min-tokens 4096` keeps natural documents, not repeated/padded text.

The output includes full `documents.jsonl`, raw downloads, the tokenizer,
`source_records.jsonl`, `acquisition_audit.jsonl` and `manifest.json`. Downloaded
full texts live in an ignored `.cache/` directory; source manifests and processing
code can be versioned separately. Source files are never executed.

## Source selection and provenance

- D2L English/Chinese chapters: textbook prose and code. The book's `config.ini`
  identifies CC-BY-SA-4.0 prose and MIT-0 code; the Chinese repository's top-level
  Apache license alone is not used as a prose license.
- CPython, scikit-learn, pandas, NumPy, SciPy and SymPy: original Markdown/RST
  documentation, retaining code/math/table directives and source line numbers.
- Europe PMC REST full-text XML: agriculture/life-science research articles.
  The per-article license is read from JATS, not inferred from “open access”.
  CC-BY/CC0 and CC-BY-SA variants in the acquisition allowlist are retained;
  other/unknown licenses are recorded as skips.

For third-party text, retain attribution and the source's own license conditions.
Repository-level observations do not assert that every embedded excerpt has the
same license. The corpus is **assessment-only and unreviewed**; this package never
sets document training approval.

If GitHub's anonymous REST quota is exhausted, source resolution uses read-only,
shallow, blobless Git metadata. Files are still checked against pinned Git blob
hashes. Raw URLs/PMC XML are cached; changed pinned data fails verification.
Failures and below-threshold documents are counted instead of silently included.

## Parsing scope

Markdown/RST headings, paragraph order, fenced code, math/table directives and
JATS nested sections are preserved. JATS tables/MathML retain XML. Bibliographies
are explicitly excluded from JATS; appendices are retained. Includes are not
expanded and RST source is not rendered. This is **not a real PDF visual-layout
benchmark** and has no human-reviewed extraction accuracy claim.

## Extra tasks and synthesis candidates

```powershell
uv run python -m longdoc_sources.study --documents longdoc_sources/.cache/pipeline_v1/corpus.jsonl --output longdoc_sources/.cache/study_v1
```

The study builder exports controlled cross-document/table-prose location checks,
section-based extractive QA candidates, blank structure-review queues and
attributed extractive long-document compilations. Compilations contain exact
source-block mappings, parent document/family IDs and original source licenses.
All parents must come from one split. These are source-linked compilations,
**not generated reasoning supervision or independently verified QA gold**.

For model-generated natural multi-evidence questions, use the separate
`python -m longdoc_sources.synthesize` command. Without the explicit
`--use-configured-model` flag it writes a dry-run plan only. Each selected anchor
allows one capped model request, with no retries or fallbacks. Quote/source/split
checks are automatic and do not establish semantic accuracy; accepted candidates
remain `ai_generated / unreviewed`. See `--help` and the root reproduction guide.
