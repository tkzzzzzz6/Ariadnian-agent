"""完整 best 发布协议；stdlib 检验不等于模型可重载或云端 flush 完成。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

REQUIRED = {'adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
            'scheduler.pt', 'rng_state.pth', 'trainer_state.json'}


def require_mounted_drive(path):
    if Path(path).is_relative_to(Path('/content/drive')) and not os.path.ismount('/content/drive'):
        raise OSError('Drive mount lost; refusing to treat local fallback files as durable')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def improves(loss, previous):
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError('best requires a finite eval_loss')
    return previous is None or loss < previous


def inspect_checkpoint(path, step, loss):
    path = Path(path)
    improves(loss, None)
    if type(step) is not int or step < 1 or path.is_symlink() or not path.is_dir():
        raise ValueError('invalid complete checkpoint path/step')
    files = {}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():
            raise ValueError('checkpoint symlinks are not allowed')
        if p.is_file() and p.name != 'COMPLETE.json':
            if p.stat().st_size == 0:
                raise ValueError('checkpoint contains an empty file')
            files[p.relative_to(path).as_posix()] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    if not REQUIRED <= set(files):
        raise ValueError('incomplete checkpoint: adapter/optimizer/scheduler/RNG/trainer state required')
    state = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    if (state.get('global_step') != step or state.get('best_global_step') != step
            or state.get('best_metric') != loss):
        raise ValueError('checkpoint trainer state disagrees with native best')
    return files


class BestStore:
    """每个 run/arm/seed 独占目录；完成标记最后写，校验新代后才清理旧代。"""
    def __init__(self, root, owner):
        self.root = Path(root)
        require_mounted_drive(self.root)
        self.owner = owner
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}', owner) or self.root.is_symlink():
            raise ValueError('invalid best-store owner/path')
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / 'OWNER.json'
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text(encoding='utf-8')) != {'owner': owner}:
                raise ValueError('best-store belongs to another run')
        else:
            if any(self.root.iterdir()):
                raise ValueError('refusing to adopt nonempty best-store')
            # x avoids overwriting a concurrently claimed directory.
            with marker.open('x', encoding='utf-8') as stream:
                json.dump({'owner': owner}, stream)

    def _read(self, directory):
        if directory.is_symlink() or not re.fullmatch(r'checkpoint-[1-9][0-9]*', directory.name):
            raise ValueError('invalid owned generation')
        marker = directory / 'COMPLETE.json'
        if not marker.exists():
            return None
        if marker.is_symlink():
            raise ValueError('invalid completion marker')
        record = json.loads(marker.read_text(encoding='utf-8'))
        if (record.get('owner') != self.owner or directory.name != f"checkpoint-{record['step']}"
                or inspect_checkpoint(directory, record['step'], record['eval_loss']) != record['files']):
            raise ValueError('complete checkpoint hash/owner mismatch')
        return record

    def current(self):
        require_mounted_drive(self.root)
        records = [self._read(p) for p in self.root.glob('checkpoint-*')]
        records = sorted((r for r in records if r is not None), key=lambda r: r['step'])
        for earlier, later in zip(records, records[1:]):
            if not improves(later['eval_loss'], earlier['eval_loss']):
                raise ValueError('best generations do not strictly improve')
        return records[-1] if records else None

    def publish(self, source, step, loss, *, provenance=None):
        require_mounted_drive(self.root)
        files = inspect_checkpoint(source, step, loss)
        previous = self.current()
        if previous and (step <= previous['step'] or not improves(loss, previous['eval_loss'])):
            raise ValueError('only a later strict best can be published')
        target = self.root / f'checkpoint-{step}'
        if target.exists():
            raise ValueError('generation already exists; preserve it for diagnosis')
        target.mkdir()
        # A failed copy deliberately leaves an incomplete generation, never pruning old best.
        for name in files:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source) / name, dest)
        if inspect_checkpoint(target, step, loss) != files:
            raise ValueError('copied checkpoint failed readback verification')
        record = {'schema_version': 'complete-best-v1', 'owner': self.owner, 'step': step,
                  'eval_loss': loss, 'files': files, 'provenance': provenance or {},
                  'durability': 'readback verified; Drive flush still required before runtime release'}
        write_json(target / 'COMPLETE.json', record)
        if self._read(target) != record:
            raise ValueError('completion marker failed readback')
        # Only verified previous generations of THIS claimed store are eligible.
        for old in self.root.glob('checkpoint-*'):
            if old != target and self._read(old) is not None:
                shutil.rmtree(old)
        return record


def prune_local(root, best):
    root, best = Path(root).resolve(), Path(best).resolve()
    if best.parent != root:
        raise ValueError('best must be a direct owned output child')
    for path in root.glob('checkpoint-*'):
        if path == best:
            continue
        if path.is_symlink() or not re.fullmatch(r'checkpoint-[1-9][0-9]*', path.name):
            raise ValueError('refusing unexpected checkpoint cleanup target')
        shutil.rmtree(path)


def selected_exposure(rows, best_step):
    if (type(best_step) is not int or not 1 <= best_step <= len(rows)
            or [r['optimizer_step'] for r in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('best step is outside the contiguous observed training prefix')
    selected = rows[:best_step]
    hashes = [h for row in selected for h in row['ordered_microbatch_label_hashes']]
    return {'total_completed_steps': len(rows), 'best_step': best_step,
            **{'selected_' + key: sum(r[source] for r in selected) for key, source in
               [('consumed_sequences', 'sequences'), ('input_tokens', 'input_tokens'), ('supervised_tokens', 'supervised_tokens')]},
            'selected_ordered_label_hashes_sha256': hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode('ascii')).hexdigest(),
            'note': 'Equal full-run budgets do not imply equal selected checkpoint exposure.'}


def callback_for(plan_path, directory, drive_run, callback_base):
    """使用 Trainer 的公开 callback 接口；不替换保存实现或训练损失。"""
    plan_path, directory, drive_run = Path(plan_path), Path(directory), Path(drive_run)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    arm = plan_path.parent.parent.name
    if arm not in {'short', 'long'}:
        raise ValueError('best callback requires a paired short/long plan')
    owner = 'plan-' + digest(plan_path)
    store = BestStore(drive_run / 'best' / (arm + '-seed17'), owner)
    output = Path(plan['checkpoint_dir'])

    class BestCallback(callback_base):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # 仅记录数值指标，既不复制训练文本，也不复制环境变量。
            values = {k: v for k, v in (logs or {}).items() if type(v) in (int, float) and math.isfinite(v)}
            with (directory / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': state.global_step, **values}, allow_nan=False) + '\n')
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            loss = (metrics or {}).get('eval_loss')
            improves(loss, None)  # 比 native best 判定更早，拒绝 NaN 和 -inf。
            with (directory / 'dev_evaluations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': state.global_step, 'eval_loss': loss}, allow_nan=False) + '\n')
            return control

        def on_save(self, args, state, control, **kwargs):
            step, loss = state.global_step, state.best_metric
            checkpoint = output / f'checkpoint-{step}'
            if (state.best_global_step != step
                    or Path(state.best_model_checkpoint).resolve() != checkpoint.resolve()):
                raise ValueError('upstream saved a non-best checkpoint')
            files = inspect_checkpoint(checkpoint, step, loss)
            record = store.publish(checkpoint, step, loss, provenance={'plan_sha256': digest(plan_path)})
            write_json(directory / 'best.json', record)
            # Drive 成功且本地文件也核验后，才删除旧本地 best。
            prune_local(output, checkpoint)
            self.persist_logs()
            return control

        def persist_logs(self):
            require_mounted_drive(drive_run)
            target = drive_run / 'progress' / arm
            target.mkdir(parents=True, exist_ok=True)
            for file in directory.iterdir():
                if file.is_file() and file.suffix in {'.json', '.jsonl'}:
                    shutil.copyfile(file, target / file.name)
            for name in ('launch_plan.json', 'launch_receipt.json', 'audit_execution_contract.json'):
                source = plan_path.parent / name
                if source.is_file():
                    shutil.copyfile(source, target / name)

        def on_step_end(self, args, state, control, **kwargs):
            self.persist_logs()
            return control

    return BestCallback()
