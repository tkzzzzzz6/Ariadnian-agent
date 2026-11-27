"""有限时配对实验：tmux 可观察，Drive 分阶段留证；不分配/停止账号 VM。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import paired_job as job
from training_audit import file_hash, save


def session_name(run_id):
    job.run_root(run_id)
    return 'ariadne-sft-' + run_id


def shell_command(argv):
    return shlex.join([str(x) for x in argv])


def tmux(argv, **kwargs):
    return subprocess.run(['tmux', *argv], capture_output=True, text=True, timeout=15, **kwargs)


def start(root, drive_run, licenses):
    root, drive_run = Path(root), Path(drive_run)
    run_id = root.name.removeprefix('ariadne_paired_')
    name = session_name(run_id)
    job.verify_bundle(root)
    evidence = root / 'evidence'
    evidence.mkdir(exist_ok=True)
    receipt_path = evidence / 'tmux_start.json'
    if receipt_path.exists() or drive_run.exists():
        raise ValueError('run already claimed; inspect its status, do not duplicate training')
    save(receipt_path, {'status': 'starting', 'session': name, 'drive_run': str(drive_run)})
    try:
        if tmux(['has-session', '-t', '=' + name]).returncode == 0:
            raise ValueError('owned tmux session already exists')
        drive_run.mkdir(parents=True, exist_ok=False)
        save(drive_run / 'OWNER.json', {'run_id': run_id, 'bundle_manifest_sha256': file_hash(root / 'bundle_manifest.json')})
        shutil.copyfile(root / 'bundle_manifest.json', drive_run / 'bundle_manifest.json')
        if (root / 'inputs').is_dir():
            shutil.copytree(root / 'inputs', drive_run / 'inputs')
            for source in (root / 'inputs').iterdir():
                if file_hash(source) != file_hash(drive_run / 'inputs' / source.name):
                    raise ValueError('Drive input snapshot readback mismatch')
        argv = [sys.executable, Path(__file__).resolve(), '--action', 'supervise', '--run-id', run_id,
                '--drive-run', drive_run]
        for license_name in licenses:
            argv.extend(['--allow-license', license_name])
        # 留下退出后的 pane，便于检查退出码；不改用户的全局 tmux 设置。
        pipeline = shell_command(argv) + ' 2>&1 | tee -a ' + shlex.quote(str(evidence / 'runner.log'))
        launch = ('tmux set-option -t ' + shlex.quote(name) + ' remain-on-exit on; exec '
                  + shell_command(['bash', '-o', 'pipefail', '-c', pipeline]))
        tmux(['new-session', '-d', '-s', name, '-n', 'train', launch], check=True)
        status = [sys.executable, Path(__file__).resolve(), '--action', 'status', '--run-id', run_id]
        tmux(['new-window', '-d', '-t', name, '-n', 'status', shell_command(['watch', '-n', '2', shell_command(status)])], check=True)
        tmux(['new-window', '-d', '-t', name, '-n', 'gpu', 'watch -n 2 nvidia-smi'], check=True)
        receipt = {'status': 'started', 'session': name, 'drive_run': str(drive_run),
                   'attach': f'tmux attach-session -t {name}',
                   'switch': f'tmux switch-client -t {name}',
                   'capture': f'tmux capture-pane -p -t {name}:train -S -100',
                   'logs': str(evidence), 'note': 'tmux survives terminal disconnect, not VM reclamation.'}
        save(receipt_path, receipt)
        shutil.copyfile(receipt_path, drive_run / receipt_path.name)
        return receipt
    except BaseException as exc:
        save(receipt_path, {'status': 'start_failed', 'session': name, 'error_type': type(exc).__name__,
                            'note': 'Inspect this owned session before retry; never create a duplicate run.'})
        raise


def resource_gate(root):
    points = json.loads((root / 'evidence/resource_summary.json').read_text())['points']
    for kind, length in [('train', 8192), ('infer', 16384)]:
        found = [p for p in points if p['kind'] == kind and p['length'] == length]
        if len(found) != 1 or found[0]['process_status'] != 'succeeded' or found[0]['result_status'] != 'succeeded':
            raise ValueError(f'resource gate failed: {kind} {length}')


def persist(root, drive_run):
    from best_checkpoint import require_mounted_drive
    require_mounted_drive(drive_run)
    archive = job.archive_results(root)
    sha = file_hash(archive)
    destination = drive_run / 'artifacts'
    destination.mkdir(exist_ok=True)
    target = destination / (sha + '.zip')
    if not target.exists():
        shutil.copyfile(archive, target)
    if file_hash(target) != sha:
        raise ValueError('Drive artifact readback mismatch')
    save(drive_run / 'latest_archive.json', {'sha256': sha, 'bytes': target.stat().st_size,
         'relative_path': 'artifacts/' + target.name, 'best_stored_separately': True,
         'flush_required_before_release': True})
    for name in ('supervisor.json', 'runner.log'):
        source = root / 'evidence' / name
        if source.is_file():
            shutil.copyfile(source, drive_run / name)


def unpublished_checkpoints(root, drive_run):
    """轻量归档不含模型：释放前另验所有本地候选已完整发布，否则保留 VM。"""
    from best_checkpoint import BestStore, inspect_checkpoint
    problems = []
    for arm in ('short', 'long'):
        plan_path = root / f'experiments/ms_swift/.cache/paired/{arm}/plan/launch_plan.json'
        if not plan_path.exists():
            continue
        try:
            plan = json.loads(plan_path.read_text())
            local = Path(plan['checkpoint_dir'])
            if not local.resolve().is_relative_to(root.resolve()):
                raise ValueError('checkpoint path outside this run')
            candidates = list(local.glob('checkpoint-*'))
            if not candidates:
                continue
            folder = drive_run / 'best' / (arm + '-seed17')
            if not (folder / 'OWNER.json').is_file():
                raise ValueError('local checkpoint has no published Drive store')
            best = BestStore(folder, 'plan-' + file_hash(plan_path)).current()
            if not best:
                raise ValueError('Drive has no complete best')
            for candidate in candidates:
                if (candidate.name != f"checkpoint-{best['step']}"
                        or inspect_checkpoint(candidate, best['step'], best['eval_loss']) != best['files']):
                    raise ValueError('local best/candidate is not fully published')
        except (OSError, ValueError, KeyError) as exc:
            problems.append({'arm': arm, 'error_type': type(exc).__name__, 'error': str(exc)})
    return problems


def execute_stages(root, drive_run, stages):
    state = {'status': 'running', 'stage': None, 'completed_stages': [], 'started_unix': time.time(), 'pid': os.getpid(),
             'exit_code': None, 'quality_improvement_measured': False}
    code = 0
    try:
        for name, function in stages:
            state['stage'] = name
            save(root / 'evidence/supervisor.json', state)
            print('[supervisor] ' + name, flush=True)
            function()
            state['completed_stages'].append(name)
            save(root / 'evidence/supervisor.json', state)
            persist(root, drive_run)
        state['status'] = 'succeeded'
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        code = 1
        print(f'[supervisor] failed: {type(exc).__name__}: {exc}', flush=True)
    finally:
        unpublished = unpublished_checkpoints(root, drive_run)
        if unpublished:
            code = 1
            state.update(status='persistence_failed', unpublished_checkpoints=unpublished)
        state.update(exit_code=code, finished_unix=time.time())
        save(root / 'evidence/supervisor.json', state)
        try:
            persist(root, drive_run)
        except BaseException as exc:
            code = 1
            state.update(status='persistence_failed', exit_code=1, persistence_error_type=type(exc).__name__)
            save(root / 'evidence/supervisor.json', state)
            print('[supervisor] persistence failed; keep VM for artifact retrieval', flush=True)
        # 只有此收据出现，观察器才可 flush/release，避免与 finally 归档竞争。
        exit_receipt = {'status': state['status'], 'exit_code': code, 'finished_unix': time.time()}
        if state['status'] != 'persistence_failed':
            try:
                save(drive_run / 'supervisor_exit.json', exit_receipt)
                if json.loads((drive_run / 'supervisor_exit.json').read_text()) != exit_receipt:
                    raise ValueError('exit receipt readback mismatch')
            except BaseException:
                code = 1
                exit_receipt.update(status='persistence_failed', exit_code=1)
        save(root / 'evidence/supervisor_exit.json', exit_receipt)
    return code


def supervise(root, drive_run, licenses):
    job.verify_bundle(root)
    owner = json.loads((drive_run / 'OWNER.json').read_text())
    if owner != {'run_id': root.name.removeprefix('ariadne_paired_'), 'bundle_manifest_sha256': file_hash(root / 'bundle_manifest.json')}:
        raise ValueError('Drive run owner mismatch')
    os.environ['ARIADNE_DRIVE_RUN'] = str(drive_run)
    os.environ.pop('GH_TOKEN', None)
    os.environ.pop('GITHUB_TOKEN', None)
    save(root / 'evidence/runtime_budget.json', {'started_unix': time.time(), 'maximum_wall_seconds': 21600})
    def interrupted(signum, frame):
        raise TimeoutError('owned supervisor wall budget exhausted') if signum == signal.SIGALRM else InterruptedError('owned supervisor interrupted')
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    signal.alarm(21600)
    def prepare():
        argv = [root / '.venv/bin/python', root / 'experiments/colab/paired_job.py', '--run-id', owner['run_id'], '--action', 'prepare-child']
        for item in licenses:
            argv.extend(['--allow-license', item])
        job.command(root, 'paired_prepare', argv, timeout=600)
    stages = [('setup', lambda: job.setup(root)), ('probe', lambda: job.probe(root)),
              ('resource_gate', lambda: resource_gate(root)), ('prepare', prepare),
              ('eval_base', lambda: job.evaluate(root, 'base')),
              ('train_short', lambda: job.train(root, 'short')), ('train_long', lambda: job.train(root, 'long')),
              ('eval_short', lambda: job.evaluate(root, 'short')), ('eval_long', lambda: job.evaluate(root, 'long'))]
    try:
        return execute_stages(root, drive_run, stages)
    finally:
        signal.alarm(0)


def status(root):
    output = {'run': root.name}
    for name in ('tmux_start', 'supervisor'):
        path = root / 'evidence' / (name + '.json')
        if path.exists():
            output[name] = json.loads(path.read_text())
    for arm in ('short', 'long'):
        folder = root / 'experiments/ms_swift/.cache/paired' / arm / 'plan/training_audit'
        output[arm] = {}
        for name in ('optimizer_steps', 'dev_evaluations', 'metrics'):
            path = folder / (name + '.jsonl')
            if path.exists():
                try:
                    output[arm][name] = json.loads(path.read_text().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    output[arm][name] = 'write in progress'
        if (folder / 'best.json').exists():
            best = json.loads((folder / 'best.json').read_text())
            output[arm]['best'] = {k: best[k] for k in ('step', 'eval_loss')}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--action', choices=['start', 'supervise', 'status', 'capture'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--drive-run', type=Path)
    parser.add_argument('--allow-license', action='append', default=[])
    args = parser.parse_args()
    root = job.run_root(args.run_id)
    if args.action in {'start', 'supervise'}:
        if sys.platform != 'linux' or not os.path.ismount('/content/drive'):
            parser.error('Linux and genuinely mounted /content/drive required')
        prefix = Path('/content/drive/MyDrive/Ariadne/experiments').resolve()
        if not args.drive_run or args.drive_run.resolve().parent != prefix:
            parser.error('Drive run must be a new direct child of MyDrive/Ariadne/experiments')
    if args.action == 'start':
        print(json.dumps(start(root, args.drive_run, args.allow_license), indent=2))
    elif args.action == 'supervise':
        raise SystemExit(supervise(root, args.drive_run, args.allow_license))
    elif args.action == 'status':
        print(json.dumps(status(root), ensure_ascii=False, indent=2))
    else:
        print(tmux(['capture-pane', '-p', '-t', session_name(args.run_id) + ':train', '-S', '-100'], check=True).stdout)


if __name__ == '__main__':
    main()
