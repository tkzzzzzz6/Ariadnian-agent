# Gemma 实验操作说明

本轮将 Gemma Base 评测、数据审核、SFT/QLoRA 与资源记录放在独立实验目录。
准备统计、程序测试和模型实测分开报告。最新执行状态见
[2026-11-08 Colab SFT 记录](../experiments/results/colab_sft_20261108/README.md)；
此前的本地推理和准备结果保留在 [2026-11-07 记录](../experiments/results/readiness_20261107/README.md)。
Colab A100 上的 20-step SFT 和检查点重载已完成；正式长文 CPT 与能力收益测评仍未执行。

后续的[长短配对预实验实施记录](../experiments/results/paired_pilot_20261108/README.md)和
[可视化报告](../experiments/results/paired_pilot_20261108/report.html)已建立：32 train / 8 dev
配对候选、250道冻结公开题、统一Gemma chat评测及A100分阶段运行器。用户已选择仅完成本地准备，
暂缓Colab登录与训练；40组新数据保持待审核。此前登录返回 `invalid_grant`，
尚未启动本轮资源验证或新SFT。下文31篇审核包和旧42题检查
保留其历史口径，不能代替新实验的来源锁与审核。

## 已有入口

从仓库根目录运行以下轻量命令，不加载模型权重：

```bash
npm run eval:gemma -- --help
npm run data:review -- --help
npm run train:cpt -- --help
npm run train:profile -- --help
npm run train:compare -- --help
npm run test:experiments
```

这些 npm 入口使用独立 CPU 准备环境，不接受 `--execute`。实际 GPU 推理/训练由明确
激活的独立环境直接执行。以下 Python 命令均从 `experiments/ms_swift/` 运行。

## 1. 复核首批数据

打开本机 [31 篇文档审核包](../experiments/ms_swift/.cache/review_20261107_v2/index.html)。
原文、来源、许可候选证据、关联重合候选及审核字段都在本地；远端 clone 没有这些忽略文件，
可按[审核说明](training-data-review.md)重新生成。

本轮选中 24 篇 train、7 篇 dev，覆盖 7 域，27 篇英文、4 篇中文。全部是 pending；
没有自动填写真人身份或批准。当前 data_science 源划分没有 dev，报告保留这一缺口。
另经固定 Gemma tokenizer 核验，未切片正文分别为 226,127 / 45,008 token；不包含真实
SWIFT 模板和重复训练曝光，适合先验证流程，不能据此承诺能力提升。

全库跨 split 检测得到 158 对候选，涉及 97 篇文档；其中有共享代码和公式前导文本。
另有 4 对跨语言同文件名家族建议。原有 split 保留，归并或排除需在新数据版本中明确记录。
首批 31 篇在当前词/CJK shingle 阈值下没有命中，仍需内容/许可复核。
公开检查仅覆盖冻结的 42 题 context：0 命中不证明没有语义重合或基座预训练污染。

完成审核后另存凭据文件，不覆盖 pending 模板。CPT `prepare --mode training` 要求
`--reviews` 及精确的 `--allow-license`，来源文档哈希变化会使旧凭据失效。
复合许可标识也须逐字一致，allowlist 本身不代替许可审核。

## 2. 先核验推理可行性

完整环境、模型固定版本下载及评分命令见[Gemma 本地测评](gemma-local-eval.md)。
已生成本地计划：

| 计划 | 内容 |
|---|---|
| `.cache/gemma_eval_20261107_v1/profile/plan.json` | 精确 2K、4K、8K 输入，各预留 32 输出 token；只测资源 |
| `.cache/gemma_eval_20261107_v1/benchmark_8k/plan.json` | 原公开 42 题，12 题可容纳、30 题超限；没有静默截断 |
| `.cache/gemma_eval_20261107_v1/nll_dev_2k/plan.json` | 41 篇 dev，324,973 个待计分目标，窗口化 NLL |

CPU 检查：

```bash
uv run python -m gemma_eval run --plan .cache/gemma_eval_20261107_v1/profile/plan.json
uv run python -m cpt.profile inspect --output .cache/inventory_new
```

激活推理环境并完成权重缓存后：

```bash
python -m gemma_eval.bounded \
  --plan .cache/gemma_eval_20261107_v1/profile/plan.json \
  --output .cache/profile_run_new --max-seconds 120 --execute
```

默认是 NF4 线性层 + BF16 compute。embedding/PLE 不假定量化为四位；模型参数必须全部在
单张 GPU 的设备标记下，不能用框架 CPU offload 掩盖显存不足；这不能排除 Windows 驱动使用系统内存。
有时限入口将模型加载和推理都计入整次运行预算，超时清理自己启动的进程树并保留原始子运行，
在根目录补齐全部计划 ID 的状态。超时不代表 OOM，未执行题目不计为成功。
首题为冷运行，计时包含同步和 streamer
开销；失败同样保留，不能将加载成功或生成非空文本等同于答案正确。

原公开基准在 8K 下有大量超限。正式质量比较应在推理前冻结相同可容纳题目子集，并扩大
各任务样本；如采用检索/截断，则另建明确改变输入的协议，保留完整原文成绩。
当前原始 Base 使用 completion 协议；若评测 SFT 对话产物，需为所有组统一模板并另存协议。

2026-11-07 真实本机 NF4 探针中，2K 生成成功；4K 运行观测到约 1.57 GiB 共享系统内存，
长时间未完成后结束自有进程，8K 未执行。详见[资源记录](../experiments/results/readiness_20261107/README.md)。
这套配置尚未通过 8GB 纯显存长文验证，不能拿准备计划或 CUDA 参数标记替代结果。

## 3. 训练配置与 SFT 数据

保留既有 100-step `configs/gemma4_e2b_pt_lora.json`。新增 8 个 smoke 配置：

| 文件前缀 | 训练目标 | 更新方式 |
|---|---|---|
| `gemma4_e2b_pt_lora` | CPT 全文本 loss | BF16 基座 + LoRA |
| `gemma4_e2b_pt_qlora` | CPT 全文本 loss | NF4 基座 + LoRA |
| `gemma4_e2b_sft_lora` | 助手答案及必要模板结束符 loss | BF16 基座 + LoRA |
| `gemma4_e2b_sft_qlora` | 助手答案及必要模板结束符 loss | NF4 基座 + LoRA |

每个前缀均有 `_4k_smoke.json` 和 `_8k_smoke.json`，20 个 optimizer steps。
这些配置不承诺目标 GPU 能容纳，也不构成等 token 的正式实验。

QLoRA 在既有 [训练依赖](../experiments/ms_swift/requirements-train.txt)上增加
[固定 bitsandbytes 依赖](../experiments/ms_swift/requirements-train-qlora.txt)。运行前检查
软件来源、原生 BF16、NF4 小型 CUDA 运算和真实模板编码；未获准数据不能执行。

SFT 输入独立于 CPT 原文：每条需要 `id`、`split`、`source_document_ids`、`license`、
`kind` 和 `messages`。支持单轮 user/assistant，加可选 system。样本答案与其来源文档
分别提供哈希绑定的人工复核记录。公开基准、test、evaluation-only 不进入 SFT。
2026-11-08 另建的 10 条原创合成短样本经用户审核后获准用于内部工程试跑，划分为
8 条 train、2 条 dev，已完成真实 SFT。它们不是原有单元测试 fixture，也没有放行上述
31 篇待审文档或公开 benchmark；正式长文 SFT 数据仍需独立准备和审核。

```bash
uv run python -m cpt sft-prepare \
  --input .cache/my_sft_candidates.jsonl \
  --source-corpus ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --model-assets .cache/model_assets/gemma4-e2b \
  --sequence-length 4096 --output .cache/sft_preview_new

uv run python -m cpt plan \
  --prepared .cache/sft_preview_new \
  --recipe configs/gemma4_e2b_sft_qlora_4k_smoke.json \
  --output .cache/plans/sft_preview_new
```

`my_sft_candidates.jsonl` 是待提供的独立数据文件，不能用公开测试题替代。上述默认 preview
始终被禁止训练。正式导出额外传 `--mode training --reviews <样本凭据> --source-reviews
<来源凭据> --allow-license <许可>`。CPT 后继续 SFT 可在 plan 中指定
`--init-adapter <本项目CPT检查点>`；其训练计划、退出记录和权重哈希必须可追溯。

## 4. 云端短试跑与资源记录

已完成的 [Colab A100 短试跑](../experiments/results/colab_sft_20261108/README.md)
使用 BF16 + LoRA、20 个 optimizer steps，训练器报告的训练耗时为 97.32 秒，
随后实际重载 adapter 并通过有限值 forward 检查。4K 是配置上限，10 条样本的正文计数
仅为 84–108 token；这不是 4K/8K 长文训练或质量提升的证据。
独立打包、Colab 环境、结果收集与停止会话步骤见 [Colab 操作说明](../experiments/colab/README.md)。

服务器迁移后在新绝对路径重新 prepare/plan，不能改写旧计划内的路径或哈希。
实际执行使用已激活的 Linux GPU 环境：

```bash
python -m cpt.profile train \
  --plan .cache/plans/my_approved_run/launch_plan.json \
  --output .cache/profiles/my_run \
  --max-seconds 900 --interval 2 --execute
```

`my_approved_run` 必须是实际完成审核后生成的计划。该命令重用原训练启动检查，记录
stdout/stderr、退出状态和整卡采样，超过时间终止自己启动的进程组；不关闭云服务器。
整卡采样包含其他程序，且会漏掉采样间隔中的尖峰，不能充当模型 allocator 峰值。
没有训练器提供真实有效 loss-token 计数时，tokens/s 保持 null，不用样本数冒充。
进程退出 0 也不自动宣称 checkpoint 已成功重载或模型能力提高。

本次云端执行使用用户指定的 Colab A100。AutoDL 账号、租卡和实例操作仍未接入；
本次短样本训练不能直接用于估算完整长文实验成本。

## 5. 对照边界

三组研究设计仍为 Base→SFT、Base→打乱段序CPT→同SFT、Base→保留段序CPT→同SFT。
先做各自 Base/CPT-only 的 NLL，再按相同协议比较问答与部署表现。

```bash
uv run python -m cpt.compare \
  --left-plan .cache/plans/ordered_v2/launch_plan.json \
  --right-plan .cache/plans/shuffled_v2/launch_plan.json \
  --output .cache/comparison_new
```

工具核验两组原文覆盖（含重复次数）、训练配置、验证集与准入状态。可额外提供
`--left-template`/`--right-template` 比较真实 SWIFT 一次编码的 loss token 数与词表直方图。
本次两组覆盖同样 279 篇训练文档、9,506,427 个源字符位置曝光，但尚未准入。

**一次编码相等不等于训练时实际消费量相等。** 当前没有记录每一步真实 loss 标签曝光的
训练器审计，严格等 token 标志仍为 false；`--require-strict` 会返回非零。正式因果结论还需
训练预算匹配、去污染处置、统一评测精度/协议以及足够样本和重复实验。

所有正文、审核凭据、预测、日志、环境和模型在 `.cache/`，Git 忽略；仅代码、说明和
小型聚合记录版本化。后台应用环境没有增加 Torch/Transformers/MS-SWIFT。
