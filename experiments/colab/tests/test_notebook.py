import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class NotebookTests(unittest.TestCase):
    def test_unpublished_checkpoint_exit_never_flushes_or_unassigns(self):
        from sft_notebook import MONITOR
        import builtins
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'evidence').mkdir()
            state = {'status': 'persistence_failed', 'completed_stages': ['setup'], 'exit_code': 1}
            for name in ('supervisor', 'supervisor_exit'):
                (root / 'evidence' / (name + '.json')).write_text(json.dumps(state))
            lifecycle = []
            actual_import = builtins.__import__
            def importing(name, *args, **kwargs):
                if name == 'IPython.display':
                    return SimpleNamespace(HTML=lambda s: s, clear_output=lambda **kw: None, display=lambda x: None)
                if name == 'google.colab':
                    return SimpleNamespace(drive=SimpleNamespace(flush_and_unmount=lambda: lifecycle.append('flush')),
                                           runtime=SimpleNamespace(unassign=lambda: lifecycle.append('unassign')))
                return actual_import(name, *args, **kwargs)
            scope = {'root': root, 'drive_run': root, 'driver': root / 'driver.py', 'run_id': 'fixture',
                     'sys': sys, 'json': json, 'time': SimpleNamespace(monotonic=lambda: 0),
                     'subprocess': SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(stdout='{}'))}
            with patch('builtins.__import__', side_effect=importing), self.assertRaisesRegex(RuntimeError, '持久化失败'):
                exec(compile(MONITOR, '<monitor-test>', 'exec'), scope)
            self.assertEqual(lifecycle, [])

    def test_generated_notebook_is_executable_and_does_not_embed_secrets(self):
        from sft_notebook import build_notebook
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'SFT.ipynb'
            build_notebook('MyDrive/Ariadne/experiments/release/bundle.zip', 'a' * 64, path)
            data = json.loads(path.read_text(encoding='utf-8'))
            code = '\n'.join(''.join(cell['source']) for cell in data['cells'] if cell['cell_type'] == 'code')
            for cell in data['cells']:
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), '<notebook-cell>', 'exec')
            self.assertIn('drive.mount', code)
            self.assertIn('tmux_runner.py', code)
            self.assertIn('flush_and_unmount', code)
            self.assertIn('runtime.unassign', code)
            self.assertNotIn('GH_TOKEN', code)
            self.assertIn('a' * 64, code)
            with self.assertRaises(ValueError):
                build_notebook('../bundle.zip', 'not-a-hash', path)
            with self.assertRaises(FileExistsError):
                build_notebook('MyDrive/bundle.zip', 'a' * 64, path)
