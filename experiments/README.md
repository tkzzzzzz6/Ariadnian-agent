# 研究与训练实验

这里保存训练相关代码、配置、脚本、测试和汇总记录。应用服务位于 `backend/app/`；
训练实验使用独立环境，默认安装不包含 Torch、Transformers 或 MS-SWIFT。

```text
experiments/
  common/          ariadne-data-core：纯标准库文档 schema、读取与来源/家族划分
  colab/           Colab A100 显式打包、远端 SFT 准备、执行与产物收集
  ms_swift/
    cpt/           CPT 准入审计、原生分词、导出、计划及显式执行检查
    gemma_eval/    Gemma Base 原生提示词预算、本机资源探针、公开题预测和 held-out NLL
    configs/       模型与试跑 recipe
    scripts/       根目录 npm 命令的实验转发和测试入口
    tests/         本地合成数据、mock 合同与入口回归测试
    pyproject.toml 独立的 CPU 准备项目
    uv.lock        CPU 准备环境锁；不等于 CUDA 训练环境锁
    requirements-train.txt  独立 Linux/CUDA 训练栈来源与版本
    .cache/        本机资产、审核凭据、导出、计划和检查点；Git 忽略
  results/         可提交的小型 manifest、汇总与说明
```

SFT 已有独立准备与受准入检查的训练入口，QLoRA 使用单独的配置和依赖；这些入口不是已完成训练的证明。已有语料构建和公开基准仍在
`backend/longdoc*`、`backend/longeval` 与 `backend/publicbench`；本轮整理训练链路，未移动这些工具。
训练准备通过文件和 manifest 读取语料产物，使用 [common](common/README.md) 的共享规则，不导入后端代码。

从仓库根目录运行：

```bash
npm run train:cpt -- --help
npm run eval:gemma -- --help
npm run data:review -- --help
npm run train:profile -- --help
npm run train:compare -- --help
npm run test:experiments
```

也可直接使用实验环境：

```bash
cd experiments/ms_swift
uv sync --group dev
uv run python -m cpt --help
uv run pytest -q
```

`npm run train:cpt` 固定使用轻量准备环境，只做审计、导出、计划与空跑复核，不接受 `--execute`。
真实训练需要明确激活 Linux/CUDA 训练 venv，从 `experiments/ms_swift/` 直接调用其 `python`。
步骤与限制见 [MS-SWIFT 项目说明](ms_swift/README.md)及[完整训练说明](../docs/ms-swift-gemma4.md)。

[结果索引](results/README.md)区分准备记录、目录迁移复核与未来模型实验。
原始语料、逐条预测、审核凭据和权重不提交 Git；忽略文件仍可由本机 Agent 按路径读取。
目录迁移不使待审材料自动获得训练资格，也不产生 CPT/SFT 或模型性能收益。

本轮落地、实际数据发现与可执行步骤见[实验操作说明](../docs/gemma-experiment-workflow.md)。
Colab A100 的会话、传输和训练操作见 [Colab SFT 说明](colab/README.md)。
40组配对 SFT 的有限时 tmux、完整 best 和 Drive 保存策略见 [操作说明](../docs/gemma-sft-best-tmux.md)。
