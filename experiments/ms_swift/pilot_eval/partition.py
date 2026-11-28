"""按完整来源连通分区，避免答案或模型表现影响抽样。"""
from __future__ import annotations

import hashlib

from .freeze import fingerprint, normalized, paragraphs

ROLES = ('demonstration', 'development', 'confirmation')


def context(row):
    raw = row.get('raw_record', {})
    return row.get('context') or raw.get('context') or raw.get('passage') or row.get('prompt', '')


def text_keys(text):
    return {'text:' + fingerprint(text)} | {'text:' + key for key in paragraphs(text)} if text.strip() else set()


def row_keys(row):
    keys = text_keys(context(row))
    keys.add('id:' + row['id'])
    if row.get('cluster_id'):
        keys.add('cluster:' + row['cluster_id'])
    for family in row.get('source_families', []):
        keys.add('family:' + family)
    return keys


def source_keys(source):
    blocks = [b.get('cleaned_text', b.get('text', '')) for b in source.get('blocks', [])]
    keys = text_keys('\n\n'.join(blocks))
    for block in blocks:
        if len(normalized(block)) >= 300:
            keys.add('text:' + fingerprint(block))
    if source.get('family_id'):
        keys.add('family:' + source['family_id'])
    return keys


def partition_records(pool, regression, training_sources, quotas, *, seed, eligible_ids=None):
    if not isinstance(quotas, dict) or set(quotas) - set(ROLES):
        raise ValueError('unknown partition quota role')
    for counts in quotas.values():
        if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('quota must be a non-negative integer')
    ids = [r['id'] for r in pool]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate pool IDs')
    eligible_ids = set(ids) if eligible_ids is None else set(eligible_ids)
    if not eligible_ids <= set(ids):
        raise ValueError('eligible ID outside the complete source pool')
    if any(r.get('evaluation_only') is not True or not context(r).strip() for r in pool):
        raise ValueError('pool must contain explicit evaluation-only source contexts')
    nodes = [(r['id'], row_keys(r)) for r in pool]
    nodes += [('regression:' + r['id'], row_keys(r)) for r in regression]
    nodes += [('training:' + r['id'], source_keys(r)) for r in training_sources]
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, (_, keys) in enumerate(nodes):
        for key in keys:
            if key in owners:
                a, b = find(i), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = i
    old_roots = {find(i) for i in range(len(pool), len(pool) + len(regression))}
    train_roots = {find(i) for i in range(len(pool) + len(regression), len(nodes))}
    roots = {r['id']: find(i) for i, r in enumerate(pool)}
    used = set(old_roots | train_roots)
    selections, shortfalls = {}, {}
    # 不合格记录也参与来源连通；仅在连通与隔离完成后过滤选题资格。
    ordered = sorted((r for r in pool if r['id'] in eligible_ids),
                     key=lambda r: hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())
    for role in ROLES:
        selected, shortfalls[role] = [], {}
        for task, required in sorted(quotas.get(role, {}).items()):
            matching = [r for r in ordered if r['benchmark'] + ':' + r['task'] == task]
            count = 0
            for row in matching:
                if count == required:
                    break
                root = roots[row['id']]
                if root in used:
                    continue
                used.add(root)
                selected.append({**row, 'split_role': role + '_evaluation_only_never_training'})
                count += 1
            shortfalls[role][task] = required - count
        selections[role] = selected
    return {'selections': selections, 'audit': {
        'schema_version': 'evaluation-source-partition-v2', 'seed': seed,
        'pool_records': len(pool), 'regression_records': len(regression),
        'eligible_pool_records': len(eligible_ids),
        'blocked_regression_records': sum(roots[r['id']] in old_roots for r in pool),
        'blocked_training_records': sum(roots[r['id']] in train_roots for r in pool),
        'blocked_eligible_training_records': sum(roots[eid] in train_roots for eid in eligible_ids),
        'selected_counts': {k: len(v) for k, v in selections.items()}, 'shortfalls': shortfalls,
        'selection_complete': not any(n for counts in shortfalls.values() for n in counts.values()),
        'selection_depends_on_answers_or_model_scores': False, 'model_predictions_observed': 0,
        'overlap_scope': 'IDs, declared clusters/families, normalized contexts and complete paragraphs/blocks >=300 chars; not semantic/pretraining contamination proof',
    }}
