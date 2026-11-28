"""Portable tmux argv/lifecycle tests; no Colab allocation or GPU calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TmuxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'ariadne_paired_fixture'
        self.root.mkdir()
        (self.root / 'bundle_manifest.json').write_text('{"files": {}}')
        self.drive = Path(self.tmp.name) / 'drive'

    def tearDown(self):
        self.tmp.cleanup()

    def test_owned_session_name_and_quoted_argv(self):
        from tmux_runner import session_name, shell_command
        import shlex
        self.assertEqual(session_name('pilot_v2'), 'ariadne-sft-pilot_v2')
        with self.assertRaises(ValueError):
            session_name('../other')
        argv = ['python', 'path with space/file.py', '--allow-license', 'A AND B;$(bad)']
        self.assertEqual(shlex.split(shell_command(argv)), argv)

    def test_duplicate_start_does_not_create_second_session(self):
        import tmux_runner as mod
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 1 if 'has-session' in argv else 0, '', '')
        with patch.object(mod, 'tmux', side_effect=run), patch.object(mod.job, 'verify_bundle', return_value={'files': {}}):
            mod.start(self.root, self.drive, ['A AND B'])
            with self.assertRaises(ValueError):
                mod.start(self.root, self.drive, ['A AND B'])
        self.assertEqual(sum('new-session' in c for c in calls), 1)
        cmd = next(c[-1] for c in calls if 'new-session' in c)
        self.assertIn("'A AND B'", cmd)
        self.assertFalse(any('kill-server' in c for c in calls))

    def test_failed_start_has_exit_receipt(self):
        import tmux_runner as mod
        with patch.object(mod, 'tmux', side_effect=subprocess.CalledProcessError(7, ['tmux'])), patch.object(mod.job, 'verify_bundle'):
            with self.assertRaises(subprocess.CalledProcessError):
                mod.start(self.root, self.drive, [])
        self.assertEqual(json.loads((self.root / 'evidence/tmux_start.json').read_text())['status'], 'start_failed')

    def test_workflow_failure_persists_and_skips_later_stages(self):
        import tmux_runner as mod
        self.drive.mkdir()
        (self.root / 'evidence').mkdir()
        seen = []
        def first():
            seen.append('first')
            raise RuntimeError('resource validation failed')
        with patch.object(mod, 'persist', return_value=None):
            code = mod.execute_stages(self.root, self.drive, [('probe', first), ('train', lambda: seen.append('train'))])
        self.assertEqual(code, 1)
        self.assertEqual(seen, ['first'])
        receipt = json.loads((self.root / 'evidence/supervisor.json').read_text())
        self.assertEqual((receipt['status'], receipt['exit_code']), ('failed', 1))
        self.assertEqual(receipt['stage'], 'probe')
        self.assertEqual(json.loads((self.root / 'evidence/supervisor_exit.json').read_text())['exit_code'], 1)

    def test_resource_gate_rejects_missing_or_failed_lengths(self):
        import tmux_runner as mod
        (self.root / 'evidence').mkdir()
        path = self.root / 'evidence/resource_summary.json'
        points = [{'kind': k, 'length': n, 'process_status': 'succeeded', 'result_status': 'succeeded'}
                  for k, n in [('train', 8192), ('infer', 16384)]]
        path.write_text(json.dumps({'points': points}))
        mod.resource_gate(self.root)
        points[0]['process_status'] = 'failed'
        path.write_text(json.dumps({'points': points}))
        with self.assertRaises(ValueError):
            mod.resource_gate(self.root)

    def test_status_reports_whole_training_and_selected_best_separately(self):
        import tmux_runner as mod
        folder = self.root / 'experiments/ms_swift/.cache/paired/short/plan/training_audit'
        folder.mkdir(parents=True)
        (folder / 'optimizer_steps.jsonl').write_text('{"optimizer_step": 8}\n')
        (folder / 'best.json').write_text('{"step": 2, "eval_loss": 1.25}')
        result = mod.status(self.root)
        self.assertEqual(result['short']['optimizer_steps']['optimizer_step'], 8)
        self.assertEqual(result['short']['best']['step'], 2)

    def test_unpublished_local_checkpoint_prevents_runtime_release(self):
        import tmux_runner as mod
        self.drive.mkdir()
        (self.root / 'evidence').mkdir()
        plan = self.root / 'experiments/ms_swift/.cache/paired/short/plan/launch_plan.json'
        plan.parent.mkdir(parents=True)
        checkpoints = plan.parent / 'checkpoints'
        partial = checkpoints / 'checkpoint-4'
        partial.mkdir(parents=True)
        (partial / 'adapter_model.safetensors').write_bytes(b'only local new best')
        plan.write_text(json.dumps({'checkpoint_dir': str(checkpoints)}))
        def failed_save():
            raise OSError('best copy failed')
        with patch.object(mod, 'persist', return_value=None):
            code = mod.execute_stages(self.root, self.drive, [('train', failed_save)])
        self.assertEqual(code, 1)
        exit_receipt = json.loads((self.root / 'evidence/supervisor_exit.json').read_text())
        self.assertEqual(exit_receipt['status'], 'persistence_failed')
        self.assertTrue((partial / 'adapter_model.safetensors').exists())


if __name__ == '__main__':
    unittest.main()
