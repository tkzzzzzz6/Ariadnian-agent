"""Meaningful source-admission and lexical-overlap checks, no GPU/network."""
import copy
import json
from pathlib import Path

import pytest

from cpt import core, review


def document(did, split="train", domain="science", language="en", text=None):
    return {"id": did, "family_id": f"family-{did}", "title": f"Title {did}",
            "source_url": "https://example.org/source", "license": "CC0-1.0", "revision": "fixture",
            "license_evidence": ["https://example.org/license"], "split": split,
            "domain": domain, "language": language, "kind": "research_article",
            "review_status": "unreviewed", "token_count": 100,
            "blocks": [{"id": "b1", "type": "paragraph", "section_path": [], "page": None,
                        "text": text or f"Unique {did} document content."}]}


def benchmark(text="A completely different public context."):
    raw = {"context": text}
    return {"id": "benchmark:fixture", "context": text, "evaluation_only": True,
            "raw_record": raw, "raw_record_sha256": core.digest(raw),
            "raw_record_hash_kind": "canonical_JSON_values"}


def save(path, rows):
    core.write_jsonl(path, rows)
    return path


def rec(key, text, split="train", category="corpus"):
    return review.text_record(key, key, split, category, text)


def test_winnowing_detects_shared_span_at_different_offsets_with_correct_source_evidence():
    shared = " ".join(f"word{i}" for i in range(35))
    a = rec("a", "Unique_prefix! " + shared + " suffixA")
    b = rec("b", "Other prefixes have DIFFERENT length. " + shared.upper().replace(" ", ", ") + " tailB", "dev")
    hits, summary = review.compare_records([a, b])
    assert len(hits) == 1
    assert summary["guaranteed_detectable_exact_normalized_span_tokens"] == 31
    evidence = hits[0]["evidence"][0]
    assert a.text[evidence["left_start_char"]:evidence["left_end_char"]] == evidence["left_excerpt"]
    assert b.text[evidence["right_start_char"]:evidence["right_end_char"]] == evidence["right_excerpt"]
    assert review.lexical_tokens(evidence["left_excerpt"])[0] == review.lexical_tokens(evidence["right_excerpt"])[0]


def test_chinese_char_tokens_and_short_span_limit_are_explicit():
    shared = "根据完整上下文进行资料分析而不是仅仅依靠某一个孤立的短句得出最终实验结论"
    a = rec("a", "甲" + shared + "结尾")
    b = rec("b", "乙丙" + shared + "最后", "evaluation_only", "benchmark")
    hits, _ = review.compare_records([a, b])
    assert len(hits) == 1 and hits[0]["comparison"] == "public_benchmark"
    assert review.fingerprints(["x"] * 19) == {}


def test_hash_collisions_are_not_reported_as_text_overlap(monkeypatch):
    monkeypatch.setattr(review, "fingerprints", lambda *args: {123: 0})
    hits, _ = review.compare_records([rec("a", "first phrase here"), rec("b", "different unrelated text", "dev")], ngram=2)
    assert hits == []


def test_same_split_and_benchmark_pairs_excluded_but_cross_split_kept():
    text = " ".join(f"token{i}" for i in range(40))
    records = [rec("a", text), rec("b", text), rec("c", text, "test"),
               rec("x", text, "evaluation_only", "benchmark"), rec("y", text, "evaluation_only", "benchmark")]
    hits, _ = review.compare_records(records)
    assert len(hits) == 8  # two train/test pairs plus three corpus x two public contexts
    assert not any({h["left_id"], h["right_id"]} in ({"a", "b"}, {"x", "y"}) for h in hits)
    assert all(h["review_status"] == "pending" and not h["automatic_training_exclusion"] for h in hits)


def test_sampling_preserves_splits_languages_domains_and_seed():
    docs = [document(f"{split}-{lang}-{i}", split, language=lang) for split in ("train", "dev", "test") for lang in ("en", "zh") for i in range(4)]
    original = copy.deepcopy(docs)
    selected, manifest = review.select_candidates(docs, [], per_stratum=2, dev_per_stratum=1)
    again, _ = review.select_candidates(list(reversed(docs)), [], per_stratum=2, dev_per_stratum=1)
    assert selected == again and docs == original
    assert len(selected) == 6 and {d["language"] for d in selected} == {"en", "zh"}
    assert all(d["split"] != "test" for d in selected)
    assert manifest["test_documents_selected"] == 0


def test_review_packet_pending_hash_bound_and_html_inert(tmp_path):
    docs = [document("train", text='<script>alert("bad")</script> Lorem ipsum'), document("dev", "dev"), document("test", "test")]
    source = save(tmp_path / "source.jsonl", docs)
    public = save(tmp_path / "public.jsonl", [benchmark()])
    source_bytes = source.read_bytes()
    output = tmp_path / "packet"
    summary = review.build(source, public, output, ngram=2, window=2)
    assert source.read_bytes() == source_bytes
    assert summary["approved_documents"] == 0 and summary["training_ready"] is False
    assert summary["source"]["documents"] == 3
    receipts = core.read_jsonl(output / "selected_reviews.pending.jsonl")
    assert {r["document_id"] for r in receipts} == {"train", "dev"}
    assert all(r["decision"] == r["license_decision"] == "pending" for r in receipts)
    assert all(not r["reviewer"] and not r["reviewer_type"] for r in receipts)
    assert receipts[0]["document_sha256"] == core.digest(next(d for d in docs if d["id"] == receipts[0]["document_id"]))
    # A packet cannot satisfy the existing admission contract by itself.
    audited = core.audit(source, tmp_path / "reaudit", reviews_path=output / "selected_reviews.pending.jsonl", allowed_licenses=["CC0-1.0"])
    assert audited["approved_documents"] == 0
    pages = "".join(p.read_text(encoding="utf-8") for p in (output / "documents").glob("*.html"))
    assert "<script>" not in pages and "&lt;script&gt;" in pages
    assert "Content-Security-Policy" in pages
    for name, artifact in summary["artifacts"].items():
        assert core.file_hash(output / name) == artifact["sha256"]


def test_conflicting_family_and_stale_benchmark_rejected_before_output(tmp_path):
    a, b = document("a"), document("b", "dev")
    b["family_id"] = a["family_id"]
    source = save(tmp_path / "source.jsonl", [a, b])
    public = save(tmp_path / "public.jsonl", [benchmark()])
    with pytest.raises(ValueError):
        review.build(source, public, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()
    save(source, [a, document("c", "dev")])
    stale = benchmark()
    stale["context"] += " Changed"
    save(public, [stale])
    with pytest.raises(ValueError, match="context differs"):
        review.build(source, public, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_refuses_overwrite_and_duplicate_benchmark_records(tmp_path):
    source = save(tmp_path / "source.jsonl", [document("a"), document("b", "dev")])
    public = save(tmp_path / "public.jsonl", [benchmark()])
    output = tmp_path / "packet"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        review.build(source, public, output)
    assert (output / "keep.txt").read_text() == "preserve"
    save(public, [benchmark(), benchmark()])
    with pytest.raises(ValueError, match="duplicate"):
        review.build(source, public, tmp_path / "second")


@pytest.mark.parametrize("options", [{"ngram": 1}, {"window": 0}, {"per_stratum": 0}, {"seed": -1}])
def test_bad_parameters_do_not_create_packet(tmp_path, options):
    source = save(tmp_path / "source.jsonl", [document("a")])
    public = save(tmp_path / "public.jsonl", [benchmark()])
    with pytest.raises(ValueError):
        review.build(source, public, tmp_path / "packet", **options)
    assert not (tmp_path / "packet").exists()


def test_flagged_candidates_get_explicit_counts_and_pending_family_suggestions(tmp_path):
    shared = " ".join(f"word{i}" for i in range(40))
    a = document("book-en:chapter/attention.md", text="alpha " + shared)
    b = document("book-zh:chapter/attention.md", split="dev", language="zh", text="beta " + shared)
    c = document("math-only-train", domain="mathematics")
    source = save(tmp_path / "source.jsonl", [a, b, c])
    public = save(tmp_path / "public.jsonl", [benchmark()])
    out = tmp_path / "packet"
    result = review.build(source, public, out)
    assert result["selected"]["documents_with_overlap_candidates"] == 2
    assert result["selected"]["documents_without_overlap_candidates"] == 1
    assert result["selected"]["missing_dev_domains"] == ["mathematics"]
    family = core.read_jsonl(out / "family_review_candidates.jsonl")
    assert len(family) == 1
    assert family[0]["hint"] == "cross_language_same_filename_shared_lexical_content"
    assert family[0]["decision"] == "pending" and family[0]["merge_families"] is None
    assert family[0]["left_split"] != family[0]["right_split"]
    counts = {r["document_id"]: r["overlap_candidate_pairs"] for r in core.read_jsonl(out / "selection_review.jsonl")}
    assert counts == {a["id"]: 1, b["id"]: 1, c["id"]: 0}
    assert 'family_review_candidates.jsonl' in (out / "index.html").read_text(encoding="utf-8")
    aggregate = review.aggregate_summary(result, out)
    serialized = json.dumps(aggregate)
    assert a["id"] not in serialized and shared not in serialized
    assert not any(name.startswith('documents/') for name in aggregate["artifacts"])
    assert aggregate["document_pages"] == 3


def test_existing_aggregate_report_is_preserved_before_any_packet_write(tmp_path):
    report = tmp_path / "data_summary.json"
    report.write_text('previous evidence', encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        review.main(['--input', str(tmp_path / 'missing'), '--benchmark', str(tmp_path / 'missing-benchmark'),
                     '--output', str(tmp_path / 'packet'), '--summary-output', str(report)])
    assert exc.value.code == 2
    assert report.read_text() == 'previous evidence'
    assert not (tmp_path / 'packet').exists()
