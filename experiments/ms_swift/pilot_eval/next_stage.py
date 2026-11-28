"""准备独立开发与确认候选；本模块不执行模型调用。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import string
import zipfile

from cpt.core import digest, file_hash, new_output, read_jsonl, write_json, write_jsonl
from gemma_eval.core import tokenizer_assets
from .acquire import BOOLQ_REV
from .calibration import apply_profile, assess_run, build_calibration_records
from .freeze import (PRIMARY, SECONDARY, LB_REV, LB_ZIP_SHA, PROMPT_SHA, MAXLEN_SHA,
                     decorate, fingerprint, source_components)
from .partition import partition_records

SEED = 281126
BOOLQ_JSONL_SHA = 'a4b213c801e232d24576f40206770525b0e9954a9cb5f02007e908b25aba0afd'
REGRESSION_SHA = 'f22fb98c38a65f68abd8b2838de345ed7c0cda467ddb6c3acf73d9cfceef974e'


def eligible(row):
    if row['total_budget_tokens'] != row['native_input_tokens'] + row['max_new_tokens']:
        raise ValueError('native input plus official output budget disagrees')
    if row['benchmark'] == 'longbench-v1':
        return (row['total_budget_tokens'] <= 16384
                and (row['task'] not in PRIMARY or row['native_input_tokens'] >= 8192))
    return row['benchmark'] == 'boolq' and row['total_budget_tokens'] <= 2048


def bind_training_sources(scope, final_sources):
    ids = [r['id'] for r in final_sources]
    if not ids or len(ids) != len(set(ids)) or any(scope.get(r['id']) != digest(r) for r in final_sources):
        raise ValueError('final source is missing, duplicated, changed or outside audited source scope')
    return {'passed': True, 'scope_sha256': digest(scope),
            'selected_sources': {r['id']: digest(r) for r in final_sources},
            'note': 'New/changed source requires a new partition and confirmation seal.'}


def verify_preseal(prepared, final_sources):
    prepared = Path(prepared)
    preseal = json.loads((prepared / 'confirmation.preseal.json').read_text(encoding='utf-8'))
    if (preseal['source_lock_sha256'] != file_hash(prepared / 'source.lock.json')
            or preseal['records_sha256'] != file_hash(prepared / 'confirmation/records.jsonl')):
        raise ValueError('confirmation preseal or partition source lock changed')
    artifacts = preseal.get('partition_artifacts', {})
    if (set(artifacts) != {f'{role}/records.jsonl' for role in ('demonstration', 'development', 'confirmation', 'calibration')}
            or any(file_hash(prepared / name) != sha for name, sha in artifacts.items())):
        raise ValueError('partition development/demonstration/calibration preseal changed or incomplete')
    lock = json.loads((prepared / 'source.lock.json').read_text(encoding='utf-8'))
    return {**bind_training_sources(lock['training_source_scope'], final_sources),
            'source_lock_sha256': file_hash(prepared / 'source.lock.json'),
            'confirmation_sha256': preseal['records_sha256'], 'protocol_sealed': False}


def seal_confirmation(prepared, final_sources, plan, run, model_assets, output):
    prepared = Path(prepared)
    binding = verify_preseal(prepared, read_jsonl(final_sources))
    report = assess_run(prepared / 'calibration/records.jsonl', plan, run)
    if report.get('run_contract_verified') is not True or not report['calibration_ready']:
        raise ValueError('Base development calibration has no eligible profile; confirmation remains unsealed')
    demos = read_jsonl(prepared / 'demonstration/records.jsonl')
    records = read_jsonl(prepared / 'confirmation/records.jsonl')
    tokenizer, metadata = tokenizer_assets(model_assets)
    sealed = []
    for row in records:
        chosen = apply_profile(row, report['selected_profile'], demos) if row['benchmark'] == 'boolq' else dict(row)
        decorate(chosen, tokenizer)
        chosen['split_role'] = 'confirmation_evaluation_only_never_training'
        if not eligible(chosen):
            raise ValueError('selected calibration profile exceeds frozen confirmation length policy')
        sealed.append(chosen)
    out = new_output(output)
    lock = json.loads((prepared / 'source.lock.json').read_text(encoding='utf-8'))
    write_json(out / 'source.lock.json', lock)
    write_inputs(out / 'confirmation', sealed, lock)
    write_json(out / 'calibration_receipt.json', report)
    receipt = {'schema_version': 'confirmation-seal-v2', 'selected_profile': report['selected_profile'],
               'apply_to': 'all arms and seeds; no per-arm profile selection', 'final_sources': binding,
               'calibration_receipt_sha256': file_hash(out / 'calibration_receipt.json'),
               'records_sha256': file_hash(out / 'confirmation/records.jsonl'), 'records': len(sealed),
               'tokenizer': metadata, 'confirmation_results_observed': 0, 'executed': False,
               'formal_improvement_demonstrated': False,
               'limitation': 'Actual source-limited count is retained; no independent NarrativeQA/RULER claim if absent.'}
    write_json(out / 'confirmation.seal.json', receipt)
    return receipt


def write_inputs(out, records, source_lock):
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'records.jsonl', records)
    write_json(out / 'eval_manifest.json', {
        'schema_version': 'gemma-eval-input-v1', 'records': len(records),
        'records_sha256': file_hash(out / 'records.jsonl'), 'selected_ids': [r['id'] for r in records],
        'source_manifests': [{'stage': 'issue15-independent-partition', 'source_lock_sha256': digest(source_lock),
                             'file': {'path': '../source.lock.json', 'sha256': file_hash(out.parent / 'source.lock.json')}}],
        'evaluation_only': True, 'training_ready': False})


def prepare(*, cache, longbench_cache, regression, training_scope, model_assets, output, aggregate=None):
    cache, longbench_cache, regression, training_scope = map(Path, (cache, longbench_cache, regression, training_scope))
    paths = {'longbench_archive': longbench_cache / 'raw/data.zip',
             'official_prompts': longbench_cache / 'official/LongBench/config/dataset2prompt.json',
             'official_output_budgets': longbench_cache / 'official/LongBench/config/dataset2maxlen.json',
             'boolq': cache / 'boolq_dev.jsonl', 'regression': regression, 'training_scope': training_scope}
    expected = {'longbench_archive': LB_ZIP_SHA, 'official_prompts': PROMPT_SHA,
                'official_output_budgets': MAXLEN_SHA, 'boolq': BOOLQ_JSONL_SHA, 'regression': REGRESSION_SHA}
    hashes = {k: file_hash(p) for k, p in paths.items()}
    if any(hashes[k] != value for k, value in expected.items()):
        raise ValueError('pinned public source or historical regression hash mismatch')
    tokenizer, metadata = tokenizer_assets(model_assets)
    templates = json.loads(paths['official_prompts'].read_text(encoding='utf-8'))
    budgets = json.loads(paths['official_output_budgets'].read_text(encoding='utf-8'))
    pool = []
    with zipfile.ZipFile(paths['longbench_archive']) as archive:
        for task in PRIMARY + SECONDARY:
            if {f for _, f, _, _ in string.Formatter().parse(templates[task]) if f} - {'context', 'input'}:
                raise ValueError('official prompt interpolates non-input fields')
            name = next(n for n in archive.namelist() if n in (f'data/{task}.jsonl', f'{task}.jsonl'))
            for line in archive.read(name).decode('utf-8').splitlines():
                raw = json.loads(line)
                row = {'id': f"longbench-v1:{task}:{raw['_id']}", 'benchmark': 'longbench-v1', 'task': task,
                       'context': raw['context'], 'answers': raw['answers'], 'raw_record': raw,
                       'raw_record_sha256': digest(raw), 'language': raw.get('language', 'en'),
                       'source_provenance': {'repo': 'zai-org/LongBench', 'revision': LB_REV, 'archive_sha256': LB_ZIP_SHA},
                       'prompt': templates[task].format(context=raw['context'], input=raw['input']),
                       'max_new_tokens': budgets[task], 'score_kind': 'longbench_v1',
                       'metric_name': 'rouge_score' if task in ('gov_report', 'multi_news') else 'qa_f1_zh_score' if task == 'multifieldqa_zh' else 'qa_f1_score'}
                pool.append(decorate(row, tokenizer))
    source_components(pool)
    for i, raw in enumerate(read_jsonl(paths['boolq'])):
        row = {'id': f'boolq:validation:{i}', 'benchmark': 'boolq', 'task': 'yes_no',
               'cluster_id': 'boolq-passage:' + fingerprint(raw['passage']), 'raw_record': raw,
               'raw_record_sha256': digest(raw), 'answers': ['yes' if raw['answer'] else 'no'],
               'source_provenance': {'repo': 'google/boolq', 'revision': BOOLQ_REV, 'raw_sha256': BOOLQ_JSONL_SHA},
               'prompt': f"Read the passage and answer the question. Reply with exactly Yes or No.\n\nPassage: {raw['passage']}\n\nQuestion: {raw['question']}\nAnswer:",
               'max_new_tokens': 8, 'score_kind': 'boolq_yes_no', 'language': 'en'}
        pool.append(decorate(row, tokenizer))
    scope_rows = read_jsonl(training_scope)
    scope = {r['id']: digest(r) for r in scope_rows}
    if len(scope) != len(scope_rows):
        raise ValueError('duplicate training scope source IDs')
    quotas = {role: {**{f'longbench-v1:{t}': n for t in PRIMARY + SECONDARY}, 'boolq:yes_no': b}
              for role, n, b in [('development', 2, 24), ('confirmation', 10, 80)]}
    quotas['demonstration'] = {'boolq:yes_no': 2}
    result = partition_records(pool, read_jsonl(regression), scope_rows, quotas, seed=SEED,
                               eligible_ids={r['id'] for r in pool if eligible(r)})
    source_lock = {'schema_version': 'next-stage-sources-v2', 'input_hashes': hashes,
                   'tokenizer': metadata, 'training_source_scope': scope,
                   'source_scope_sha256': digest(scope), 'quotas': quotas, 'seed': SEED,
                   'length_policy': {'primary_min_input': 8192, 'longbench_max_total': 16384,
                                     'boolq_max_total': 2048, 'truncation': False}}
    out = new_output(output)
    write_json(out / 'source.lock.json', source_lock)
    for role, rows in result['selections'].items():
        write_inputs(out / role, rows, source_lock)
    calibration = build_calibration_records(result['selections']['development'], result['selections']['demonstration'])
    for row in calibration:
        decorate(row, tokenizer)
        row['split_role'] = 'development_format_calibration_only_never_training'
    write_inputs(out / 'calibration', calibration, source_lock)
    summary = {**result['audit'], 'raw_pool_counts': dict(Counter(r['benchmark'] + ':' + r['task'] for r in pool)),
               'eligible_pool_counts': dict(Counter(r['benchmark'] + ':' + r['task'] for r in pool if eligible(r))),
               'length_policy': source_lock['length_policy'], 'input_hashes': hashes,
               'source_scope_sha256': digest(scope), 'calibration_cases': len(calibration),
               'calibration_model_scope': 'fixed pinned Base only, bf16, no adapter; selected profile shared by all arms',
               'calibration_status': 'not_run', 'confirmation_sealed': False, 'training_ready': False,
               'regression_unchanged': file_hash(regression) == REGRESSION_SHA,
               'limitations': ['Exact contexts/paragraph overlap audit is not semantic or pretraining contamination proof.',
                               'Old RULER is regression only; independent new RULER is not included.',
                               'Final v2 selected sources must match the audited source scope before sealing.']}
    write_json(out / 'summary.json', summary)
    write_json(out / 'confirmation.preseal.json', {'schema_version': 'confirmation-preseal-v2',
        'source_lock_sha256': file_hash(out / 'source.lock.json'),
        'records_sha256': file_hash(out / 'confirmation/records.jsonl'), 'protocol_ready': False,
        'partition_artifacts': {f'{role}/records.jsonl': file_hash(out / role / 'records.jsonl')
                                for role in ('demonstration', 'development', 'confirmation', 'calibration')},
        'calibration_status': 'not_run', 'final_training_sources_bound': False})
    if aggregate:
        target = Path(aggregate); target.mkdir(parents=True, exist_ok=True)
        write_json(target / 'partition_summary.json', summary)
        write_json(target / 'partition_source.lock.json', source_lock)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'longbench-cache', 'regression', 'training-scope', 'model-assets', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--aggregate')
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
