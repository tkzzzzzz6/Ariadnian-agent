"""生成不含密钥、绑定精确输入 ZIP 的 Colab A100 入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re

BOOTSTRAP = '''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, uuid, zipfile
from datetime import datetime, timezone
assert os.path.ismount('/content/drive'), '请先完成 Drive 挂载'
bundle = Path('/content/drive') / RELEASE['drive_bundle']
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == RELEASE['sha256'], '输入包 SHA256 不符'
if not shutil.which('tmux'):
    subprocess.run(['apt-get', 'update', '-qq'], check=True, timeout=180)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'tmux'], check=True, timeout=180)
run_id = 'sft_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
root = Path('/content') / ('ariadne_paired_' + run_id)
bootstrap = Path('/content') / ('ariadne_bootstrap_' + run_id)
bootstrap.mkdir()
# 已验证整个 ZIP 的指纹；先取出 stdlib 校验器，再用它安全解包并逐文件核验。
with zipfile.ZipFile(bundle) as z:
    (bootstrap / 'remote_job.py').write_bytes(z.read('experiments/colab/remote_job.py'))
sys.path.insert(0, str(bootstrap))
import remote_job
remote_job.unpack(bundle, RELEASE['sha256'], root)
drive_run = Path('/content/drive/MyDrive/Ariadne/experiments') / run_id
driver = root / 'experiments/colab/tmux_runner.py'
argv = [sys.executable, str(driver), '--action', 'start', '--run-id', run_id, '--drive-run', str(drive_run)]
for item in ['PSF-2.0', 'CC-BY-4.0', 'CC-BY-SA-4.0 AND MIT-0', 'BSD-3-Clause']:
    argv += ['--allow-license', item]
subprocess.run(argv, check=True, timeout=90)
print('run_id:', run_id)
print('Drive:', drive_run)
print('attach:', 'tmux attach-session -t ariadne-sft-' + run_id)
'''

MONITOR = '''from IPython.display import HTML, clear_output, display
import html
deadline = time.monotonic() + 21800
state = None
while time.monotonic() < deadline:
    status_file = root / 'evidence/supervisor.json'
    if status_file.exists():
        try:
            state = json.loads(status_file.read_text())
        except json.JSONDecodeError:
            time.sleep(2)
            continue
        observed = subprocess.run([sys.executable, str(driver), '--action', 'status', '--run-id', run_id],
                                  capture_output=True, text=True, timeout=15)
        clear_output(wait=True)
        completed = len(state['completed_stages'])
        display(HTML('<h3>Gemma SFT · ' + html.escape(run_id) + '</h3><progress max="9" value="' + str(completed) +
                     '"></progress><pre>' + html.escape(observed.stdout) + '</pre>'))
        exit_file = root / 'evidence/supervisor_exit.json'
        if exit_file.exists():
            state.update(json.loads(exit_file.read_text()))
            break
    time.sleep(20)  # 观察实际训练任务，不作空闲保活。
else:
    raise TimeoutError('supervisor 超时；请检查 owned tmux 与本地归档后停止本轮 runtime')
print('真实退出状态:', state['status'], 'exit code:', state['exit_code'])
print('归档:', drive_run / 'latest_archive.json')
print('完整 best:', drive_run / 'best')
if state['status'] == 'persistence_failed':
    raise RuntimeError('Drive 持久化失败：暂不释放，先取回本地 artifacts.zip / best')
receipt = json.loads((drive_run / 'latest_archive.json').read_text())
artifact = drive_run / receipt['relative_path']
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == receipt['sha256']
print('归档读回 SHA256 已验证；训练成功与否以上方真实状态为准。')
# 结束前 flush；失败则保留 runtime，不能把 FUSE 读回当成云端已落盘。
from google.colab import drive, runtime
drive.flush_and_unmount()
subprocess.run(['tmux', 'kill-session', '-t', 'ariadne-sft-' + run_id], check=False, timeout=15)
runtime.unassign()
'''


def build_notebook(drive_bundle, sha256, output):
    relative = PurePosixPath(drive_bundle)
    if (not re.fullmatch('[a-f0-9]{64}', sha256) or relative.is_absolute()
            or '..' in relative.parts or not relative.parts or relative.parts[0] != 'MyDrive'
            or '\\' in drive_bundle):
        raise ValueError('notebook requires a safe MyDrive path and exact SHA256')
    def cell(kind, source):
        result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    description = '''# Gemma 4 E2B · A100 配对 SFT + tmux

在 Colab 选择 **A100**，再按顺序运行。这里执行已获人工批准的 40 对内部数据（32 train / 8 dev），不执行 CPT、不公开数据。

实际流程：硬件/依赖/资源验证 → 两臂准入与模板检查 → Base 250 题 → short / long 各 8 steps → 同一 250 题。
每 2 步以 dev loss 选 best（同分留早者），完整状态单独存 Drive；总步数与 best step 分开报告。
失败保留真实状态；tmux 不能防止 VM 被回收。此 notebook 未运行前不代表训练完成或能力提升。

终端观察：`tmux attach-session -t ariadne-sft-<run_id>`；已在 tmux 内用 `switch-client`。
切换窗口：`Ctrl-b n`，分别为训练、状态、GPU。也可从已连接的 CLI 使用 `colab exec` 调用 capture；CLI 0.7.5 不支持给 ssh 附加 CMD。
当前入口只恢复已成功的工程阶段，不提供经过实测的跨 VM 完整模型训练恢复；保存 best 可能需要回退重算。
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100'},
                             'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
                'cells': [cell('markdown', description),
                          cell('code', '# Drive 挂载需要你完成 Google 授权\nfrom google.colab import drive\ndrive.mount("/content/drive")\n'),
                          cell('code', 'RELEASE = ' + repr({'drive_bundle': drive_bundle, 'sha256': sha256}) + '\n' + BOOTSTRAP),
                          cell('code', MONITOR)]}
    for item in notebook['cells']:
        if item['cell_type'] == 'code':
            compile(''.join(item['source']), '<generated-notebook>', 'exec')
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--drive-bundle', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build_notebook(args.drive_bundle, args.sha256, args.output))


if __name__ == '__main__':
    main()
