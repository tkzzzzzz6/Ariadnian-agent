"""解释完整的测评样本结果，不把空答案变成成功，也不改写进程退出码。"""
import json
from collections import Counter
from pathlib import Path

from training_audit import file_hash


def completed_evaluation(output, expected_ids):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    child_path = output / 'run/summary.json'
    if (summary.get('schema_version') != 'gemma-bounded-summary-v1' or summary.get('kind') != 'chat'
            or summary.get('status') not in {'completed', 'completed_with_failures'}
            or summary.get('error') is not None or summary.get('worker_returncode') not in {0, 1}
            or summary.get('child_summary_sha256') != file_hash(child_path)
            or summary.get('predictions_sha256') != file_hash(output / 'predictions.jsonl')):
        raise ValueError('evaluation was not a complete verified worker outcome')
    child = json.loads(child_path.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if (child.get('model_loaded') is not True or child.get('total') != len(expected_ids)
            or summary.get('total') != len(expected_ids) or summary.get('child_completed_records') != len(expected_ids)
            or summary.get('selected_ids') != expected_ids or [r.get('id') for r in rows] != expected_ids
            or len(set(expected_ids)) != len(expected_ids)
            or any(r.get('status') not in {'ok', 'empty_prediction', 'failed', 'skipped_context_limit'} for r in rows)):
        raise ValueError('evaluation lacks a loaded model or the complete frozen case order')
    statuses = dict(Counter(r['status'] for r in rows))
    successful = statuses.get('ok', 0)
    if (summary.get('statuses') != statuses or summary.get('successful') != successful
            or summary.get('failed_or_skipped') != len(rows) - successful):
        raise ValueError('evaluation failure denominator disagrees with the records')
    return {'status': 'completed_with_case_failures' if successful != len(rows) else 'completed',
            'total': len(rows), 'successful': successful, 'failed_or_skipped': len(rows) - successful,
            'statuses': statuses, 'summary_sha256': file_hash(output / 'summary.json'),
            'note': 'Case failures remain failures in the frozen denominator; process receipts are unchanged.'}
