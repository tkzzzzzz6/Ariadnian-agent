"""Meaningful holdout checks: sibling sources must never be split by row."""
import pytest

from pilot_eval.freeze import choose, source_components


def test_complete_shared_source_paragraph_links_variants_but_short_boilerplate_does_not():
    shared = "A sufficiently long source paragraph about the original experiment. " * 8
    rows = [{"id": "a", "context": shared + "\n\nquestion a"},
            {"id": "b", "context": "Introduction\n\n" + shared},
            {"id": "c", "context": "Introduction\n\nAn unrelated short context."}]
    source_components(rows)
    assert rows[0]["cluster_id"] == rows[1]["cluster_id"]
    assert rows[0]["cluster_id"] != rows[2]["cluster_id"]


def test_selection_is_answer_blind_and_unique_by_source():
    rows = [{"id": f"question-{i}", "cluster_id": f"source-{i // 2}", "answers": ["secret"]} for i in range(10)]
    first = choose(rows, 4)
    changed = [{**r, "answers": ["changed"]} for r in reversed(rows)]
    assert [r["id"] for r in first] == [r["id"] for r in choose(changed, 4)]
    assert len({r["cluster_id"] for r in first}) == 4


def test_insufficient_independent_families_is_a_failure_not_padding():
    with pytest.raises(ValueError, match="independent families"):
        choose([{"id": "a", "cluster_id": "same"}, {"id": "b", "cluster_id": "same"}], 2)
