import copy

import pytest


def row(i, context=None, cluster=None):
    return {'id': str(i), 'benchmark': 'fixture', 'task': 'qa',
            'cluster_id': cluster or f'source-{i}', 'context': context or f'Original source {i}. ' * 30,
            'answers': ['private fixture answer'], 'evaluation_only': True}


def partition(pool, **kwargs):
    from pilot_eval.partition import partition_records
    return partition_records(pool, kwargs.pop('regression', []), kwargs.pop('sources', []),
                             kwargs.pop('quotas', {'development': {'fixture:qa': 1},
                                                 'confirmation': {'fixture:qa': 2}}), seed=17, **kwargs)


def test_partition_is_answer_blind_deterministic_and_never_mutates_pool():
    pool = [row(i) for i in range(12)]
    original = copy.deepcopy(pool)
    first = partition(pool)
    changed = list(reversed(copy.deepcopy(pool)))
    for item in changed:
        item['answers'] = ['changed']
    second = partition(changed)
    assert first['audit']['selection_complete']
    assert {k: [r['id'] for r in v] for k, v in first['selections'].items()} == {
        k: [r['id'] for r in v] for k, v in second['selections'].items()}
    assert pool == original
    selected = [r['cluster_id'] for rows in first['selections'].values() for r in rows]
    assert len(selected) == len(set(selected)) == 3


def test_renamed_id_and_cluster_cannot_reuse_regression_context():
    old = row('old')
    renamed = row('new', context=old['context'], cluster='pretend-new-source')
    result = partition([renamed, *[row(i) for i in range(8)]], regression=[old])
    assert 'new' not in [r['id'] for rows in result['selections'].values() for r in rows]
    assert result['audit']['blocked_regression_records'] == 1


def test_shared_paragraph_excludes_transitive_siblings():
    shared = 'Original shared scientific paragraph with source-specific content. ' * 12
    old = row('old', shared)
    sibling = row('sibling', shared + '\n\nSecond paragraph ' * 20)
    transitive = row('transitive', 'Unrelated intro\n\n' + sibling['context'].split('\n\n')[-1],
                     cluster=sibling['cluster_id'])
    result = partition([sibling, transitive, *[row(i) for i in range(8)]], regression=[old])
    assert result['audit']['blocked_regression_records'] == 2


def test_training_source_block_quarantines_overlapping_public_context():
    source = {'id': 'train-source', 'family_id': 'train-family', 'blocks': [{'text': row(0)['context']}]}
    result = partition([row(i) for i in range(8)], sources=[source])
    assert result['audit']['blocked_training_records'] == 1
    assert '0' not in [r['id'] for rows in result['selections'].values() for r in rows]


def test_capacity_shortage_is_reported_without_duplication():
    result = partition([row(0)], quotas={'development': {'fixture:qa': 1},
                                       'confirmation': {'fixture:qa': 2}})
    assert not result['audit']['selection_complete']
    assert result['audit']['shortfalls']['confirmation']['fixture:qa'] == 2
    assert result['selections']['confirmation'] == []


def test_duplicate_pool_ids_and_invalid_quota_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        partition([row(0), row(0)])
    with pytest.raises(ValueError, match='quota'):
        partition([row(0)], quotas={'confirmation': {'fixture:qa': -1}})


def test_ineligible_bridge_still_connects_eligible_context_to_training_source():
    from pilot_eval.partition import partition_records
    common = 'Shared substantial paragraph between public records with unique provenance. ' * 10
    train_text = 'Original training paragraph linked only through the ineligible public context. ' * 10
    a = row('a', common + '\n\nUnique A')
    b = row('b', common + '\n\n' + train_text)
    sources = [{'id': 'train', 'family_id': 'train', 'blocks': [{'text': train_text}]}]
    result = partition_records([a, b], [], sources, {'development': {'fixture:qa': 1}},
                               seed=17, eligible_ids={'a'})
    assert result['selections']['development'] == []
    assert result['audit']['blocked_eligible_training_records'] == 1
    assert result['audit']['eligible_pool_records'] == 1
