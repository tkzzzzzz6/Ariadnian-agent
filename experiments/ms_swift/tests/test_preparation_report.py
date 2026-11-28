from pilot_eval.preparation_report import render_report


def test_report_uses_aggregate_counts_escapes_labels_and_never_claims_quality_gain():
    data = {'counts': {'train': 128, 'dev': 32},
            'languages_by_split': {'train': {'zh': 68, 'en': 60}, 'dev': {'zh': 8, 'en': 24}},
            'directions': {'train': {'<unsafe>': 24}}, 'source_documents': 48,
            'connected_families': {'train': 32, 'dev': 8}, 'crossdoc_families': {'train': 6, 'dev': 2},
            'token_ranges': {'short': [226, 1103], 'long': [4394, 7463]}, 'training_ready': False}
    partition = {'selected_counts': {'development': 39, 'confirmation': 150, 'demonstration': 2},
                 'shortfalls': {'confirmation': {'longbench-v1:narrativeqa': 10}}, 'calibration_status': 'not_run'}
    result = render_report(data, partition)
    assert '<unsafe>' not in result and '&lt;unsafe&gt;' in result
    assert '128' in result and '150' in result and '4394' in result
    assert '未执行新训练' in result and '不是能力提升结果' in result
    assert '<svg' in result and 'https://cdn' not in result
