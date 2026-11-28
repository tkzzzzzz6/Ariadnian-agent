# Shared research data contracts

`ariadne-data-core` supplies the canonical `longdoc-v1` document validation,
JSONL reading and family/duplicate/source-ancestry split logic. Both the
assessment pipeline and the independent training tools use
`ariadne_data.documents`; neither implementation is copied into the other.

The package has no runtime dependencies outside the Python standard library.
It does not read application settings, import the backend, instantiate a
tokenizer or load a model. Content/licensing approval and training eligibility
remain the responsibility of the calling workflow.

The existing `longdoc.pipeline` import names are retained as re-exports for
assessment callers. Document validation, declared split constraints and seeded
fallback assignments preserve their previous behavior. Missing ancestors are
linked and reported; their actual upstream split cannot be independently
verified without a supplied constraint.

Run its isolated checks with `uv run --project experiments/common --group dev
python -m pytest experiments/common/tests -q` from the repository root.
Its optional standalone development lock is local; the backend and MS-SWIFT
projects retain the two versioned consumer locks.
