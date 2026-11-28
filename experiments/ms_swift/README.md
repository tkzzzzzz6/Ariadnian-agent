# MS-SWIFT / Gemma 4 E2B

独立的 CPT 准备与训练入口。首模型为 `google/gemma-4-E2B` Base，默认 recipe 为
[configs/gemma4_e2b_pt_lora.json](configs/gemma4_e2b_pt_lora.json)。2026-11-08 已在 Colab
A100-SXM4-40GB 完成首次 20-step SFT，并实际保存、重载 adapter 及验证有限值 forward。
数据是经用户审核的 8 条 train + 2 条 dev 合成短样本；正式 CPT 与模型能力收益尚未验证。
见 [最新训练记录](../results/colab_sft_20261108/README.md) 和 [Colab 操作说明](../colab/README.md)。

现已增加独立 SFT/QLoRA 配置、本机 Gemma 测评、审核包和资源采样入口。原始 100-step
recipe 保留；新增的 4K/8K、20-step smoke 配置用于验证训练流程。本次采用 4K 上限配置，
实际短样本正文仅 84–108 token，不能据此声称已验证 4K/8K 长文显存或训练效果。
完整步骤与状态见[实验操作说明](../../docs/gemma-experiment-workflow.md)。

代码在 `cpt/`，配置在 `configs/`，Node 转发脚本在 `scripts/`，实验测试在 `tests/`。
`ariadne_data` 由相邻 [common/](../common/README.md) 提供，使用本地可安装包复用文档与划分规则。
项目通过 JSONL 与 manifest 接收数据，不依赖后端工作目录或 `sys.path` 注入。

以下命令从本目录执行：

```bash
uv sync --group dev
uv run python -m cpt --help
uv run pytest -q
```

该环境只包含 CPU 准备依赖，不安装 Torch、Transformers 或 MS-SWIFT。
从仓库根目录可以使用 `npm run train:cpt -- ...` 和 `npm run test:experiments`；
参数中的相对路径以本目录为准，`train:cpt` 不接受 `--execute`。

已有本地原生分词资产时可先预览，未有资产时按完整说明的 `fetch-tokenizer` 步骤获取：

```bash
uv run python -m cpt prepare \
  --input ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --model-assets .cache/model_assets/gemma4-e2b \
  --output .cache/preview_v2

uv run python -m cpt plan \
  --prepared .cache/preview_v2 \
  --recipe configs/gemma4_e2b_pt_lora.json \
  --output .cache/plans/ordered_v2 --strategy ordered

uv run python -m cpt run --plan .cache/plans/ordered_v2/launch_plan.json
```

输出目录必须为新目录；已有 `preview_v2` 或计划时选择新名称。默认预览永远不获得训练资格。
审核后的训练导出需要另行提供哈希绑定的复核凭据和精确许可标识；公开基准和 test 文档不进入训练。

真正训练在独立 Linux/CUDA venv 中，从仓库根目录安装本地包和
[requirements-train.txt](requirements-train.txt)，再进入本目录直接运行训练环境的 `python`：

```bash
# 已激活独立训练环境，从仓库根目录安装；CPU uv 环境不执行此步骤。
python -m pip install -e experiments/common -e experiments/ms_swift
python -m pip install -r experiments/ms_swift/requirements-train.txt
```

随后执行 `python -m cpt run --plan <获准的training计划路径> --execute`。
执行前仍会检查固定软件来源、原生 BF16 CUDA 支持、数据 hash 与真实 SWIFT 模板编码；准备命令不保证显存可用。
完整的 CUDA wheel、数据准入和实验协议见[训练说明](../../docs/ms-swift-gemma4.md)。

`.cache/` 保存本机数据与检查点，被 Git 忽略。迁移保留的 `preview_v1` 和 `plans/*_v1`
仍记录旧绝对路径，仅作历史证据；不得修改内部 hash/路径冒充新计划。
汇总见[结果索引](../results/README.md)；迁移或转移服务器后，应在新路径重新导出并生成计划。
