"""Source-preservation and fail-closed boundaries for paired review exports."""
import json
from types import SimpleNamespace

import pytest

from cpt.core import digest, file_hash, read_jsonl, write_jsonl
from paired_data.build import build, connected_families


@pytest.fixture
def packet_inputs(tmp_path, monkeypatch):
    docs = [{"id": sid, "family_id": sid, "split": split, "title": sid,
             "domain": "fixture", "license": "CC0-1.0", "source_url": "https://example.org/fixture",
             "blocks": [{"id": f"b{i:05}", "type": "paragraph",
                         "text": f"Unique fixture source {sid}, paragraph {i}: " + "a" * 180}
                        for i in range(40)]} for sid, split in (("a", "train"), ("b", "dev"))]
    specs = [{"id": f"qa-{d['id']}", "source_document_ids": [d["id"]],
              "direction": "evidence_local_qa", "language": "en",
              "question": "What is the fixture identifier?", "answer": d["id"],
              "rationale": "Fixture only, not research data.",
              "evidence": [{"document_id": d["id"], "block_index": 5}]}
             for d in docs]
    corpus, authored, quarantine = (tmp_path / name for name in ("corpus.jsonl", "specs.jsonl", "quarantine.jsonl"))
    write_jsonl(corpus, docs)
    write_jsonl(authored, specs)
    write_jsonl(quarantine, [])
    # Counting is deliberately a mock. Real native lengths are integration-audited
    # against the ignored locked asset snapshot, never claimed by this unit test.
    monkeypatch.setattr("paired_data.build.load_native_tokenizer",
                        lambda _: SimpleNamespace(count=len, metadata={"model_lock_sha256": "mock"}))
    return docs, specs, corpus, authored, quarantine


def test_exports_are_paired_pending_and_sources_unchanged(packet_inputs, tmp_path):
    docs, specs, corpus, authored, quarantine = packet_inputs
    out = tmp_path / "review"
    summary = build(corpus, authored, quarantine, "mock", out)
    assert summary["counts"] == {"train": 1, "dev": 1}
    assert not summary["training_ready"]
    assert read_jsonl(out / "sources.jsonl") == docs
    short, long = (read_jsonl(out / f"{arm}.jsonl") for arm in ("short", "long"))
    for s, l in zip(short, long):
        assert s["messages"][-1] == l["messages"][-1]
        assert s["id"] == l["id"]
        assert s["source_document_ids"] == l["source_document_ids"]
    for arm in ("short", "long"):
        for receipt in read_jsonl(out / f"{arm}_reviews.pending.jsonl"):
            assert receipt["decision"] == "pending"
            assert receipt["license_decision"] == "pending"
            assert receipt["reviewer_type"] == receipt["reviewer"] == ""
    packet = json.loads((out / "review_packet.lock.json").read_text(encoding="utf-8"))
    assert "index.html" in packet["files"]
    assert "review_focus.html" in packet["files"]
    assert all(file_hash(out / name) == sha for name, sha in packet["files"].items())
    for pair in read_jsonl(out / "pairs.jsonl"):
        assert pair["evidence"][0]["document_sha256"] == digest(next(d for d in docs if d["id"] == pair["source_document_ids"][0]))
        assert all(len(indices) == len(set(indices)) for indices in pair["long_blocks"].values())


def test_quarantined_sources_are_not_silently_reused(packet_inputs, tmp_path):
    _, _, corpus, authored, quarantine = packet_inputs
    write_jsonl(quarantine, [{"left_id": "a", "right_id": "external-test-source"}])
    with pytest.raises(ValueError, match="quarantined"):
        build(corpus, authored, quarantine, "mock", tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_original_test_source_cannot_become_train(packet_inputs, tmp_path):
    docs, _, corpus, authored, quarantine = packet_inputs
    docs[0]["split"] = "test"
    write_jsonl(corpus, docs)
    with pytest.raises(ValueError, match="test source"):
        build(corpus, authored, quarantine, "mock", tmp_path / "review")


def test_crossdoc_edges_merge_source_families():
    sources = {x: {"family_id": x} for x in "abcd"}
    specs = [{"source_document_ids": ["a", "b"]}, {"source_document_ids": ["b", "c"]}]
    families = connected_families(specs, sources)
    assert families["a"] == families["b"] == families["c"]
    assert families["a"] != families["d"]


def test_same_family_cannot_cross_splits(packet_inputs, tmp_path):
    docs, _, corpus, authored, quarantine = packet_inputs
    docs[1]["family_id"] = docs[0]["family_id"]
    write_jsonl(corpus, docs)
    with pytest.raises(ValueError, match="family crosses splits"):
        build(corpus, authored, quarantine, "mock", tmp_path / "review")


def test_missing_evidence_or_false_table_label_rejected(packet_inputs, tmp_path):
    _, specs, corpus, authored, quarantine = packet_inputs
    specs[0]["evidence"][0]["quote"] = "This quote never appeared in the original document."
    write_jsonl(authored, specs)
    with pytest.raises(ValueError, match="non-exact"):
        build(corpus, authored, quarantine, "mock", tmp_path / "bad_quote")
    del specs[0]["evidence"][0]["quote"]
    specs[0]["direction"] = "table_condition"
    write_jsonl(authored, specs)
    with pytest.raises(ValueError, match="actual text table"):
        build(corpus, authored, quarantine, "mock", tmp_path / "bad_table")


def test_too_short_real_source_is_not_padded(packet_inputs, tmp_path):
    docs, _, corpus, authored, quarantine = packet_inputs
    docs[0]["blocks"] = docs[0]["blocks"][:6]
    write_jsonl(corpus, docs)
    with pytest.raises(ValueError, match="real unique source material too short"):
        build(corpus, authored, quarantine, "mock", tmp_path / "review")


def test_same_qa_with_changed_id_or_spacing_cannot_inflate_coverage(packet_inputs, tmp_path):
    _, specs, corpus, authored, quarantine = packet_inputs
    specs.append({**specs[0], 'id': 'pretend-new', 'question': '  WHAT is the fixture identifier?  '})
    write_jsonl(authored, specs)
    with pytest.raises(ValueError, match='duplicate question/answer'):
        build(corpus, authored, quarantine, 'mock', tmp_path / 'review')


def test_review_focus_uses_current_counts_language_and_crossdoc_coverage(packet_inputs, tmp_path):
    _, _, corpus, authored, quarantine = packet_inputs
    out = tmp_path / 'review'
    summary = build(corpus, authored, quarantine, 'mock', out)
    focus = (out / 'review_focus.html').read_text(encoding='utf-8')
    assert '1 train / 1 dev' in focus
    assert '返回 2 条' in focus
    assert '32 train / 8 dev' not in focus and 'CSV+Decimal' not in focus
    assert summary['languages_by_split'] == {'train': {'en': 1}, 'dev': {'en': 1}}
    assert summary['crossdoc_families'] == {'train': 0, 'dev': 0}
