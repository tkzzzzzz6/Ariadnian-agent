"""样本失败可评分，加载失败与不完整运行仍须停止；保留非零进程收据。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OutcomeTests(unittest.TestCase):
    def test_complete_empty_predictions_keep_denominator(self):
        from evaluation_outcome import completed_evaluation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = completed_evaluation(root, ['one', 'two'])
            self.assertEqual((result['total'], result['successful'], result['failed_or_skipped']), (2, 1, 1))
            self.assertEqual(result['statuses']['empty_prediction'], 1)

    def fixture(self, root, *, loaded=True, status='completed_with_failures'):
        (root / 'run').mkdir()
        rows = [{'id': 'one', 'status': 'ok'}, {'id': 'two', 'status': 'empty_prediction'}]
        raw = ''.join(json.dumps(r) + '\n' for r in rows).encode()
        (root / 'predictions.jsonl').write_bytes(raw)
        child = {'model_loaded': loaded, 'total': 2}
        (root / 'run/summary.json').write_text(json.dumps(child))
        summary = {'schema_version': 'gemma-bounded-summary-v1', 'kind': 'chat', 'status': status,
                   'total': 2, 'selected_ids': ['one', 'two'], 'child_completed_records': 2,
                   'successful': 1, 'failed_or_skipped': 1, 'statuses': dict(Counter(r['status'] for r in rows)),
                   'predictions_sha256': hashlib.sha256(raw).hexdigest(),
                   'child_summary_sha256': hashlib.sha256((root / 'run/summary.json').read_bytes()).hexdigest(),
                   'error': None, 'worker_returncode': 1}
        (root / 'summary.json').write_text(json.dumps(summary))

    def test_load_failure_timeout_missing_or_changed_rows_are_rejected(self):
        from evaluation_outcome import completed_evaluation
        for change in ('load', 'timeout', 'corrupt', 'missing', 'wrong_ids'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, loaded=change != 'load', status='timeout' if change == 'timeout' else 'completed_with_failures')
                if change == 'corrupt':
                    (root / 'predictions.jsonl').write_text('corrupt')
                if change == 'missing':
                    (root / 'run/summary.json').unlink()
                with self.assertRaises((ValueError, OSError)):
                    completed_evaluation(root, ['wrong', 'two'] if change == 'wrong_ids' else ['one', 'two'])
