"""Do not let RULER's first ten SQuAD questions masquerade as ten sources."""
import copy
import hashlib

import pytest

from pilot_eval.ruler_lineage import _annotate, _answerable_records, annotate_ruler_qa


def fixture_document():
    def qa(qid, question, text, impossible=False):
        return {"id": qid, "question": question, "is_impossible": impossible,
                "answers": [{"text": text}, {"text": text}] if not impossible else []}
    return {"data": [{"title": "Fixture article", "paragraphs": [
        {"context": "First support paragraph.", "qas": [qa("skip", "Unanswerable?", "", True), qa("q1", "First question?", "First")]},
        {"context": "Second support paragraph.", "qas": [qa("q2", "Second question?", "Second")]}]}]}


def raw_for(index, row):
    return {"index": index, "input": "Official instructions and distractors\n\nQuestion: " + row["question"],
            "outputs": list(row["answers"]), "answer_prefix": " Answer:"}


def test_upstream_order_filter_and_article_clustering():
    records = _answerable_records(fixture_document())
    assert [r["source_qa_id"] for r in records] == ["q1", "q2"]
    first = _annotate(raw_for(0, records[0]), records)
    second = _annotate(raw_for(1, records[1]), records)
    assert first["cluster_id"] == second["cluster_id"]
    assert first["source_support_sha256"] != second["source_support_sha256"]
    assert first["source_support_sha256"] == hashlib.sha256(b"First support paragraph.").hexdigest()


def test_input_and_duplicate_references_are_preserved():
    records = _answerable_records(fixture_document())
    raw = raw_for(0, records[0])
    before = copy.deepcopy(raw)
    _annotate(raw, records)
    assert raw == before
    raw["outputs"] = ["First"]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        _annotate(raw, records)


def test_modified_question_and_wrong_source_order_rejected():
    records = _answerable_records(fixture_document())
    raw = raw_for(0, records[0])
    raw["input"] += " Answer:"
    with pytest.raises(ValueError, match="missing or ambiguous"):
        _annotate(raw, records)
    raw = raw_for(1, records[0])
    with pytest.raises(ValueError, match="first ten"):
        _annotate(raw, records)


def test_ambiguous_query_reference_pair_is_not_guessed():
    records = _answerable_records(fixture_document())
    duplicate = dict(records[0], source_qa_id="duplicate", source_article="Different article")
    with pytest.raises(ValueError, match="ambiguous"):
        _annotate(raw_for(0, records[0]), records + [duplicate])


def test_unpinned_squad_cannot_supply_official_lineage(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text('{"data":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="pinned SQuAD"):
        annotate_ruler_qa({}, path)
