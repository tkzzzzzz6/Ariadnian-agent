import copy

import pytest

from cpt.compare import source_coverage, _template


def row(start, end, strategy='ordered'):
    return {"document_id": "d", "document_sha256": "a" * 64, "strategy": strategy, "split": "train",
            "spans": [{"document_id": "d", "block_id": "b", "text_field": "text", "start": start, "end": end}]}


def test_coverage_ignores_sequence_boundaries_but_counts_repetitions():
    whole = source_coverage([row(0, 12)], 'ordered')
    cut = source_coverage([row(5, 12, 'shuffled'), row(0, 5, 'shuffled')], 'shuffled')
    assert whole == cut
    repeated = source_coverage([row(0, 12), row(0, 5)], 'ordered')
    assert repeated != whole
    assert repeated['source_character_exposures'] == 17


def test_source_change_is_not_equal_coverage():
    original = row(0, 5)
    changed = copy.deepcopy(original)
    changed['document_sha256'] = 'b' * 64
    assert source_coverage([original], 'ordered') != source_coverage([changed], 'ordered')


@pytest.mark.parametrize('start,end', [(1, 0), (0, 0), (-1, 2), (False, 3)])
def test_bad_spans_rejected(start, end):
    with pytest.raises(ValueError, match='invalid source span'):
        source_coverage([row(start, end)], 'ordered')


def test_missing_template_is_unknown_not_passed():
    assert _template(None, {}) is None


def test_empty_selected_arm_not_equal():
    with pytest.raises(ValueError, match='no training source spans'):
        source_coverage([row(0, 5)], 'shuffled')
