"""CPU contracts; these fixtures are not an actual SWIFT/GPU save test."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


class BestTests(unittest.TestCase):
    def test_lost_drive_mount_is_not_mistaken_for_local_durable_storage(self):
        from best_checkpoint import require_mounted_drive
        with patch('best_checkpoint.os.path.ismount', return_value=False):
            with self.assertRaises(OSError):
                require_mounted_drive(Path('/content/drive/MyDrive/Ariadne/experiments/run'))
            require_mounted_drive(Path(tempfile.gettempdir()) / 'ordinary-cpu-fixture')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def checkpoint(self, step, loss):
        p = self.root / 'local' / f'checkpoint-{step}'
        p.mkdir(parents=True)
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            (p / name).write_bytes(b'fixture only')
        (p / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'best_global_step': step,
                                                       'best_metric': loss, 'best_model_checkpoint': str(p)}))
        return p

    def test_best_recipe_is_guarded_native_best(self):
        from paired_job import paired_recipe
        p = paired_recipe(32)['parameters']
        self.assertEqual((p['max_steps'], p['eval_steps']), (8, 2))
        self.assertEqual(p['save_strategy'], 'best')
        self.assertEqual(p['metric_for_best_model'], 'eval_loss')
        self.assertIs(p['greater_is_better'], False)
        self.assertIs(p['load_best_model_at_end'], True)
        # Temporarily retain old + new until the callback has verified publication.
        self.assertEqual(p['save_total_limit'], 2)
        self.assertIs(p['add_version'], False)
        self.assertIs(p['save_only_model'], False)
        self.assertIs(p['use_flash_ckpt'], False)

    def test_improvement_tie_and_nonfinite(self):
        from best_checkpoint import improves
        self.assertTrue(improves(2.0, None))
        self.assertTrue(improves(1.0, 2.0))
        self.assertFalse(improves(2.0, 2.0))
        for value in (None, True, float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                improves(value, 2.0)

    def test_publish_verify_then_prune_only_owned_best(self):
        from best_checkpoint import BestStore
        store = BestStore(self.root / 'drive', 'run-short-seed17')
        store.publish(self.checkpoint(2, 2.0), 2, 2.0)
        store.publish(self.checkpoint(4, 1.0), 4, 1.0)
        best = store.current()
        self.assertEqual((best['step'], best['eval_loss']), (4, 1.0))
        self.assertEqual(len(list((self.root / 'drive').glob('checkpoint-*'))), 1)
        with self.assertRaises(ValueError):
            BestStore(self.root / 'drive', 'another-run')
        with self.assertRaises(ValueError):
            store.publish(self.checkpoint(6, 1.0), 6, 1.0)

    def test_incomplete_or_failed_copy_preserves_old_best(self):
        import best_checkpoint as mod
        store = mod.BestStore(self.root / 'drive', 'run')
        store.publish(self.checkpoint(2, 2.0), 2, 2.0)
        incomplete = self.checkpoint(4, 1.0)
        (incomplete / 'optimizer.pt').unlink()
        with self.assertRaises(ValueError):
            store.publish(incomplete, 4, 1.0)
        good = self.checkpoint(6, 0.5)
        with patch.object(mod.shutil, 'copyfile', side_effect=OSError('Drive unavailable')):
            with self.assertRaises(OSError):
                store.publish(good, 6, 0.5)
        self.assertEqual(store.current()['step'], 2)
        self.assertTrue((self.root / 'drive/checkpoint-2/optimizer.pt').is_file())

    def test_corrupt_complete_copy_is_rejected(self):
        from best_checkpoint import BestStore
        store = BestStore(self.root / 'drive', 'run')
        store.publish(self.checkpoint(2, 2.0), 2, 2.0)
        (self.root / 'drive/checkpoint-2/optimizer.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            store.current()

    def test_exposure_distinguishes_best_from_total(self):
        from best_checkpoint import selected_exposure
        rows = [{'optimizer_step': i, 'sequences': 8, 'input_tokens': 100 * i,
                 'supervised_tokens': 20, 'ordered_microbatch_label_hashes': [str(i)]} for i in range(1, 9)]
        result = selected_exposure(rows, 2)
        self.assertEqual((result['total_completed_steps'], result['best_step']), (8, 2))
        self.assertEqual(result['selected_consumed_sequences'], 16)
        self.assertEqual(result['selected_input_tokens'], 300)
        with self.assertRaises(ValueError):
            selected_exposure(rows, 9)

    def test_callback_rejects_nan_and_sync_failure_keeps_old_local(self):
        from best_checkpoint import callback_for
        plan = self.root / 'short/plan/launch_plan.json'
        plan.parent.mkdir(parents=True)
        output = self.root / 'local'
        plan.write_text(json.dumps({'checkpoint_dir': str(output)}))
        audit = plan.parent / 'training_audit'
        audit.mkdir()
        cb = callback_for(plan, audit, self.root / 'drive', object)
        first = self.checkpoint(2, 2.0)
        state = SimpleNamespace(global_step=2, best_global_step=2, best_metric=2.0, best_model_checkpoint=str(first))
        cb.on_evaluate(None, state, None, metrics={'eval_loss': 2.0})
        cb.on_save(None, state, None)
        for loss in (float('nan'), -float('inf'), None):
            with self.assertRaises(ValueError):
                cb.on_evaluate(None, state, None, metrics={'eval_loss': loss})
        second = self.checkpoint(4, 1.0)
        state = SimpleNamespace(global_step=4, best_global_step=4, best_metric=1.0, best_model_checkpoint=str(second))
        with patch('best_checkpoint.BestStore.publish', side_effect=OSError('Drive unavailable')):
            with self.assertRaises(OSError):
                cb.on_save(None, state, None)
        self.assertTrue(first.is_dir())
        self.assertEqual(json.loads((audit / 'best.json').read_text())['step'], 2)
        cb.on_save(None, state, None)
        self.assertFalse(first.exists())
        self.assertEqual([p.name for p in output.glob('checkpoint-*')], ['checkpoint-4'])


if __name__ == '__main__':
    unittest.main()
