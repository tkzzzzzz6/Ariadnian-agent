"""只用开发数据选择格式配置，保持历史评分函数不变。"""
from __future__ import annotations

from collections import Counter
import hashlib
import argparse
import json
from pathlib import Path

from cpt.core import digest, file_hash, read_jsonl, write_json
from cpt.schema import MODEL_REPO, MODEL_REVISION
from gemma_eval.chat import PROTOCOL, PROTOCOL_VERSION
from .scoring import ALLOWED_STATUSES, yes_no

PROFILES = (('zero_8', 8, False), ('zero_32', 32, False),
            ('zero_64', 64, False), ('two_shot_32', 32, True))


def question(row):
    raw = row['raw_record']
    return f"Passage: {raw['passage']}\n\nQuestion: {raw['question']}\nAnswer:"


def apply_profile(original, profile, demonstrations):
    _, budget, few_shot = next(p for p in PROFILES if p[0] == profile)
    prefix = 'Read the passage and answer the question. Reply with exactly Yes or No.\n\n'
    if few_shot:
        prefix += '\n\n'.join(question(r) + ' ' + r['answers'][0].title() for r in demonstrations) + '\n\n'
    prompt = prefix + question(original)
    return {**original, 'profile': profile, 'prompt': prompt, 'max_new_tokens': budget,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'demonstration_ids': [r['id'] for r in demonstrations] if few_shot else [],
            'profile_contract_sha256': digest(PROFILES)}


def build_calibration_records(development, demonstrations):
    development = [r for r in development if r['benchmark'] == 'boolq']
    if not development or len(demonstrations) != 2:
        raise ValueError('requires development BoolQ and exactly two demonstrations')
    if any('confirmation' in r.get('split_role', '') for r in development + demonstrations):
        raise ValueError('confirmation records cannot enter development calibration')
    if ({r['id'] for r in development} & {r['id'] for r in demonstrations}
            or {r['cluster_id'] for r in development} & {r['cluster_id'] for r in demonstrations}):
        raise ValueError('demonstration overlaps development')
    if len({r['cluster_id'] for r in demonstrations}) != 2:
        raise ValueError('demonstration source groups must be independent')
    rows = []
    for profile, _, _ in PROFILES:
        for original in development:
            if original.get('evaluation_only') is not True:
                raise ValueError('calibration requires evaluation-only inputs')
            rows.append({**apply_profile(original, profile, demonstrations),
                'id': original['id'] + ':calibration:' + profile, 'base_id': original['id'],
                'split_role': 'development_format_calibration_only_never_training',
                'score_kind': 'boolq_yes_no'})
    return rows


def assess_calibration(records, predictions, *, model_loaded):
    ids = [r['id'] for r in records]
    if not ids or len(ids) != len(set(ids)) or [r['id'] for r in predictions] != ids:
        raise ValueError('calibration ID coverage/order differs from frozen records')
    names = [p[0] for p in PROFILES]
    if {r.get('profile') for r in records} != set(names):
        raise ValueError('unknown or incomplete calibration profiles')
    id_sets = [{r['base_id'] for r in records if r['profile'] == name} for name in names]
    if any(s != id_sets[0] for s in id_sets):
        raise ValueError('profile base IDs differ')
    counts = {name: Counter(total=0, empty_or_failed=0, invalid_format=0, valid_wrong=0, valid_correct=0)
              for name in names}
    for record, prediction in zip(records, predictions):
        if 'confirmation' in record.get('split_role', ''):
            raise ValueError('confirmation results cannot select a calibration profile')
        status = prediction.get('status')
        if status not in ALLOWED_STATUSES:
            raise ValueError('unknown prediction status')
        count = counts[record['profile']]
        count['total'] += 1
        text = prediction.get('prediction', '')
        if status not in {'ok', 'success'} or not isinstance(text, str) or not text.strip():
            category = 'empty_or_failed'
        else:
            parsed = yes_no(text)
            category = 'invalid_format' if parsed is None else 'valid_correct' if parsed == record['answers'][0] else 'valid_wrong'
        count[category] += 1
    profiles = {}
    for name in names:
        count = dict(counts[name])
        count['valid_format_rate'] = (count['valid_wrong'] + count['valid_correct']) / count['total']
        count['eligible'] = model_loaded is True and count['empty_or_failed'] == 0 and count['valid_format_rate'] >= .9
        profiles[name] = count
    selected = next((name for name in names if profiles[name]['eligible']), None)
    return {'schema_version': 'development-format-calibration-v2', 'profiles': profiles,
        'selected_profile': selected, 'calibration_ready': selected is not None,
        'selection_rule': 'fixed profile priority; format>=90%, no failed/empty calls, model_loaded; never answer accuracy',
        'profile_contract_sha256': digest(PROFILES), 'confirmation_results_observed': 0,
        'formal_improvement_demonstrated': False, 'historical_scoring_changed': False}


def assess_run(records_path, plan_path, output):
    """诊断统计不能代替执行收据；绑定完整 Base worker 证据后才可选配置。"""
    records_path, plan_path, output = map(Path, (records_path, plan_path, output))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    model = {'repo': MODEL_REPO, 'revision': MODEL_REVISION}
    contract = {'protocol': PROTOCOL_VERSION, 'model': model, 'context_length': 16384,
                'quantization': 'bf16', 'adapter': None}
    if (any(plan.get(k) != v for k, v in contract.items()) or plan.get('kind') != 'chat'
            or plan.get('chat_protocol') != PROTOCOL or plan.get('do_sample') is not False):
        raise ValueError('calibration selection requires fixed Base bf16/no-adapter contract')
    if (plan['input']['sha256'] != file_hash(records_path)
            or plan['requests']['sha256'] != file_hash(plan['requests']['path'])):
        raise ValueError('calibration input/request hash changed')
    records = read_jsonl(records_path)
    ids = [r['id'] for r in records]
    requests = read_jsonl(plan['requests']['path'])
    if (plan['selected_ids'] != ids or [r['id'] for r in requests] != ids
            or any(r.get('example_sha256') != digest(e) or r.get('max_new_tokens') != e['max_new_tokens']
                   for r, e in zip(requests, records))):
        raise ValueError('calibration plan/requests differ from records')
    summary_path, child_path = output / 'summary.json', output / 'run/summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if (summary.get('schema_version') != 'gemma-bounded-summary-v1'
            or summary.get('status') not in {'completed', 'completed_with_failures'}
            or summary.get('executed') is not True or summary.get('kind') != 'chat'
            or summary.get('worker_returncode') not in (0, 1) or summary.get('error') is not None
            or summary.get('child_output_errors') or summary.get('plan_sha256') != digest(plan)
            or summary.get('plan_file_sha256') != file_hash(plan_path)):
        raise ValueError('calibration lacks complete matching run contract')
    if (summary.get('predictions_sha256') != file_hash(output / 'predictions.jsonl')
            or summary.get('child_summary_sha256') != file_hash(child_path)):
        raise ValueError('calibration prediction/child receipt hash differs')
    child = json.loads(child_path.read_text(encoding='utf-8'))
    runtime = child.get('runtime') or {}
    if (child.get('executed') is not True or child.get('model_loaded') is not True
            or child.get('load_error') is not None or child.get('plan_sha256') != digest(plan)
            or child.get('evaluation_contract') != contract
            or runtime.get('quantization') != 'bf16'
            or runtime.get('chat_template_verification', {}).get('verified') is not True
            or any(x.get('selected_ids') != ids or x.get('total') != len(ids) for x in (child, summary))
            or summary.get('child_completed_records') != len(ids)):
        raise ValueError('calibration loaded model/template or complete run contract differs')
    predictions = read_jsonl(output / 'predictions.jsonl')
    if [r.get('id') for r in predictions] != ids:
        raise ValueError('calibration prediction contract has wrong ordered IDs')
    for record, prediction in zip(records, predictions):
        expected = {'example_sha256': digest(record), 'model': MODEL_REPO, 'model_revision': MODEL_REVISION,
                    'quantization': 'bf16', 'protocol': PROTOCOL_VERSION, 'max_new_tokens': record['max_new_tokens']}
        if any(prediction.get(k) != v for k, v in expected.items()):
            raise ValueError('calibration prediction contract differs between profiles/records')
    original_path = output / 'run/predictions.jsonl'
    if (child.get('predictions_sha256') != file_hash(original_path)
            or read_jsonl(original_path) != predictions):
        raise ValueError('worker original predictions hash/content differs from canonical calibration output')
    if summary.get('statuses') != dict(Counter(r['status'] for r in predictions)):
        raise ValueError('calibration status denominator differs')
    report = assess_calibration(records, predictions, model_loaded=True)
    report.update(model_scope='pinned_base_bf16_no_adapter', apply_selected_profile_to='all future arms and seeds',
                  model=model, plan_file_sha256=file_hash(plan_path), plan_sha256=digest(plan),
                  records_sha256=file_hash(records_path), summary_sha256=file_hash(summary_path),
                  predictions_sha256=file_hash(output / 'predictions.jsonl'), run_contract_verified=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Assess the fixed Base development calibration, never confirmation.')
    for name in ('records', 'plan', 'run', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('use a new output; historical calibration receipts are immutable')
    report = assess_run(args.records, args.plan, args.run)
    write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
