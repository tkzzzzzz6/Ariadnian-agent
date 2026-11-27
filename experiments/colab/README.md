# Colab A100 SFT 短程验证

配对40组内部预实验的新入口为 [best + tmux 工作流](../../docs/gemma-sft-best-tmux.md)：
`paired_bundle.py`、`tmux_runner.py` 和 `sft_notebook.py`。以下是原10条工程 smoke 的历史入口，不能当作40组实验完成记录。

使用官方 `googlecolab/google-colab-cli` 的 `colab-operator` 技能；本目录只负责
显式打包、远端准备和结果收集。Colab CLI 负责会话、上传、下载和停止实例。
CPU 本地打包不会启动或计费；`colab new --gpu A100` 会申请运行时并消耗账号额度。

## 传输内容

`bundle.py` 仅收集 `experiments/common` 的共享包、`experiments/ms_swift` 的训练和
评估源代码、固定 4K LoRA smoke 配置、锁定 tokenizer/config，以及明确传入的数据和
审核凭据。不打包 backend、应用用户数据、`.env`、CLI 凭据、本地环境、模型权重或旧计划。
ZIP 和记录级结果放在已有 Git 忽略目录 `experiments/ms_swift/.cache/`。

从仓库根目录：

```powershell
python experiments/colab/bundle.py --repo . `
  --records experiments/ms_swift/.cache/colab_sft_review_20261108_v1/sft_candidates.jsonl `
  --sources experiments/ms_swift/.cache/colab_sft_review_20261108_v1/sources.jsonl `
  --reviews experiments/ms_swift/.cache/colab_sft_review_20261108_v1/answer_reviews.approved.jsonl `
  --source-reviews experiments/ms_swift/.cache/colab_sft_review_20261108_v1/source_reviews.approved.jsonl `
  --output experiments/ms_swift/.cache/colab_20261108/package_new.zip
```

以上 approved 文件来自用户在 2026-11-08 查看全部 10 条后明确批准本次内部试跑；
新数据必须另行审核，不能套用这些凭据。输出的 SHA256 用于远端校验。pending 凭据只能用于 preview，不能触发训练。
人工审核完成后另存新凭据、建立新包；不修改历史 pending 文件或旧计划中的绝对路径。

## 会话与执行

官方 CLI 当前声明支持 Linux/macOS；Windows 应优先使用 WSL。若现有原生 Windows CLI
已经通过 `sessions`、上传和 `exec` 验证，可按具体成功能力使用，不把它称为官方支持。
显式选择认证方式，避免 CLI 版本不同带来的默认值变化。使用独立 `--config` 保存本次状态。

```bash
colab --auth oauth2 --config /path/to/isolated/sessions.json sessions
colab --auth oauth2 --config /path/to/isolated/sessions.json new -s ariadne-sft --gpu A100
colab --auth oauth2 --config /path/to/isolated/sessions.json upload -s ariadne-sft package_new.zip /content/package.zip
colab --auth oauth2 --config /path/to/isolated/sessions.json upload -s ariadne-sft experiments/colab/remote_job.py /content/ariadne_colab_job.py
```

`colab exec -f` 传送脚本到已有 Jupyter kernel。可在本地建立以下启动脚本，使用
`colab ... exec -s ariadne-sft -f setup_driver.py` 执行：

```python
import runpy
import sys

sys.argv = [
    "/content/ariadne_colab_job.py", "--action", "setup", "--run-id", "example_v1",
    "--bundle", "/content/package.zip", "--bundle-sha256", "替换为打包输出的64位SHA256",
]
runpy.run_path(sys.argv[0], run_name="__main__")
```

setup 固定在 `/content/ariadne_sft_example_v1`：拒绝覆盖旧目录；确认分配的是单张 A100；
优先通过已安装的 `uv venv --python python3.12 --seed` 建立独立环境（无 uv 时使用标准 venv），
安装固定 Torch/MS-SWIFT 依赖；执行 `pip check`、真实 BF16
小运算；下载固定 Gemma revision 并核验六个文件的大小和 SHA256。任何一步失败都会保存
状态、日志和已产生的结果包。安装及模型下载有各自超时，训练另有 900 秒上限。

后续将 `sys.argv` 改成下列动作，再调用 `runpy.run_path`：

```python
# 仍然没有人工准入；真实 SWIFT 模板检查也不能授予准入。
sys.argv = ["/content/ariadne_colab_job.py", "--action", "preview", "--run-id", "example_v1"]

# 只有包中是实际人工复核凭据且精确许可通过时才能执行。
sys.argv = ["/content/ariadne_colab_job.py", "--action", "train", "--run-id", "example_v1",
            "--allow-license", "实际通过审核的精确许可标识", "--execute"]
```

训练动作在远端新路径重新 `sft-prepare --mode training` 和 `plan`，复用现有准入校验。
没有 approved train/dev 或来源审核时立即拒绝，不自动把 pending 改为 approved。
20 个 optimizer steps 使用 BF16 + LoRA(rank16)、4K 上限、batch1/累积8；短样本不代表
实际执行了 4K 长文训练。训练后检查保存的 `global_step==20`，重载 adapter 并运行有限值
forward 检查；这只证明保存/重载，不是问答准确率或能力提升。

## 回收结果与停止

每个动作的 `finally` 都生成 `/content/ariadne_sft_example_v1/artifacts.zip`，包含本次显式
上传的数据、计划、模板报告、训练日志、资源采样和 adapter 检查点，不包含基座权重/环境。
归档输出会打印 SHA256。下载后核验哈希再停止自己创建的会话，不操作用户其他会话：

```bash
colab --auth oauth2 --config /path/to/isolated/sessions.json download -s ariadne-sft /content/ariadne_sft_example_v1/artifacts.zip ./artifacts.zip
colab --auth oauth2 --config /path/to/isolated/sessions.json stop -s ariadne-sft
```

若连接中断，先检查自己的会话和远端进程，避免重复训练。`--action archive` 可以重新归档；
setup、preview、train 不应在同一路径重复执行。CLI 不负责本脚本的计费保护，必须在所有
成功/失败分支最终 `colab stop -s <本次会话>`。本脚本不挂载 Drive、不获取 GCP 权限、不推送 Hub。

CPU 传输边界测试：`python -m unittest discover -s experiments/colab/tests`。
