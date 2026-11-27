# MS-SWIFT 与 Gemma 4 E2B：训练准备和实验协议

状态（2026-11-08）：已在 Colab A100-SXM4-40GB 完成首次真实 SFT 工程验证，
包括固定训练环境、模型文件校验、20 个 optimizer steps、adapter 保存及重载后的有限值 forward。
数据为经用户审核的 8 条 train + 2 条 dev 原创合成短样本；尚未执行正式 CPT 或证明模型能力提升。
最新结果见 [Colab SFT 记录](../experiments/results/colab_sft_20261108/README.md)，
操作入口见 [Colab 说明](../experiments/colab/README.md)。
首模型为 `google/gemma-4-E2B` **Base**。训练准备代码、配置、脚本与测试集中在独立的
`experiments/ms_swift/` 项目；CPU 数据准备和 Linux/CUDA 训练使用各自环境。
现有应用后端继续提供语料构建与评测输入，训练通过 JSONL 和 manifest 读取这些产物，不导入应用后端模块。
本说明中的服务器路径是示例，需要换成真实路径；准备文件与打印命令不等于启动训练。

## 原有 assessment 语料不能直接训练

2026-11-05 本地只读检查：`backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl` 有 350 篇保留文档，
其中 train/dev/test 为 281/41/28 篇，**350 篇的 `training_eligible` 均为 false**。
同目录 `training_gate.jsonl` 的 493 条准入记录也全部为 false；该批构建用途为 assessment。
train 的约 260 万 token 使用 Qwen2.5 参考 tokenizer 计数，不能直接当作 Gemma 原生训练预算。

训练版必须另行完成许可与用途审核、内容复核、来源及父文档关联检查、家族划分和测试去污染。
文件存在、内容可下载、清洗分数通过、AI 复核通过都不能自动将 `training_eligible` 改成 true。
没有获准文档时，训练导出应明确失败，不能为了跑通流程批量放行已有 assessment 数据。
现有 AI 生成 QA 仍是候选；未独立复核的答案不标成可靠训练 gold。公开 benchmark 及其参考答案只用于评测。

本地缓存不会随 Git 自动传到服务器。位置和忽略规则见[本地数据访问说明](local-data-access.md)，
整体对照和资源假设见[设备分档计划](cpt-device-tiers.md)。

## 独立实验目录

`experiments/ms_swift/cpt/` 保存 CPT 准备、校验和显式启动逻辑，`configs/` 保存 recipe，
`scripts/` 保存命令转发和测试脚本，`tests/` 保存实验测试。2026-11-07 已增加独立
SFT/QLoRA 准备、配置与受检查的启动入口。2026-11-08 的独立短样本 SFT 已完成实际训练；
原有长文语料的准入状态不因这次工程验证而改变。
新增入口及当前执行状态见[实验操作说明](gemma-experiment-workflow.md)。
共享文档 schema、读取与来源/家族划分来自轻量可安装包 `experiments/common/`（`ariadne_data`），
避免训练代码复制规则或通过 `sys.path` 寻找 `backend/`。

从仓库根目录进入准备环境：

```bash
cd experiments/ms_swift
uv sync --group dev
uv run python -m cpt --help
uv run pytest -q
```

该环境只安装准备依赖与共享数据包，不安装 Torch、Transformers 或 MS-SWIFT。
根目录 `npm run train:cpt -- ...` 会转发至这个独立环境，路径相对于 `experiments/ms_swift/`；
`npm run test:experiments` 单独验证实验入口和 Python 流程，根目录 `npm test` 也会执行该阶段。
`npm run train:cpt` 拒绝 `--execute`；真正 GPU 训练必须按下文激活训练 venv 后直接运行 `python`。

## 固定的软件与模型版本

本项目固定 MS-SWIFT **v4.6.0**，Git commit 为
`ad818a2cdfafaf4305ac40c4acd892a211978bc2`，已用上游 tag 的 `git ls-remote` 核对。
环境定义见 [requirements-train.txt](../experiments/ms_swift/requirements-train.txt)：

| 组件 | 固定版本 |
|---|---|
| Python | 3.12 |
| MS-SWIFT | v4.6.0 对应完整 commit |
| PyTorch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 |
| Transformers | 5.12.1 |
| PEFT / Accelerate | 0.20.0 / 1.12.0 |
| Datasets / TRL | 4.8.4 / 0.29.1 |

这些版本已核对上游发布文件及 Python 包元数据，符合该 SWIFT release 的依赖范围。
2026-11-08 已在独立 Colab Python 3.12 环境完成安装、`pip check`、真实 BF16 运算、
SWIFT 模板检查及短样本 SFT 训练/重载验证；这不替代 4K/8K 长样本或其他 GPU 的测试。
上游支持 `transformers<5.20.0`、`peft<0.21`、`datasets<4.8.5`；不能用旧 Transformers 4.x 代替来加载 Gemma 4。
本文件固定核心训练栈，不声称锁住全部传递依赖。首次安装通过后，另存完整 `pip freeze` 和安装来源。
历史上，2026-11-05 的 Linux/Python 3.12/cu128 元数据诊断使用官方 PyPI 的
`ms-swift==4.6.0`，成功解析 136 个包。当时固定 Git commit 的抓取在本机网络下未完成并已停止，
该次替代诊断没有安装或运行 CUDA。原始证据保留在
[依赖诊断记录](../experiments/results/cpt_20261105_v1/dependency_resolution_report.json)；
后续 Colab 实验已经成功安装上述固定 Git 来源，不能回写旧诊断为成功。
依据：[固定版依赖文件](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/requirements/framework.txt)、
[固定版环境建议](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/README.md#installation)。

训练前还要锁定模型和 tokenizer 的 **HF commit**、配置及文件 SHA-256。
`google/gemma-4-E2B` 与 `google/gemma-4-E2B-it` 是不同起点；各实验臂必须从同一个 Base revision 出发。
不能用可变的 `main` 标记代替运行清单里的最终版本，也不能将其他基座的 adapter 接到本模型上。

在实际 GPU 服务器上安装，以下命令从仓库根目录执行，只作环境建立说明：

```bash
python3.12 -m venv /data/venvs/ariadne-swift
source /data/venvs/ariadne-swift/bin/activate
python -m pip install --upgrade pip
python -m pip install -e experiments/common -e experiments/ms_swift
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/ms_swift/requirements-train.txt
python -m pip check
python -m pip freeze > /data/ariadne-runs/environment.freeze.txt
```

上述 PyTorch 三件套和 cu128 安装入口见 [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/#v280)。
CUDA wheel 选择必须匹配服务器驱动。环境建好后记录 GPU 型号/显存、驱动、CUDA runtime、Python、包版本与源 commit。
FlashAttention、Liger、DeepSpeed、bitsandbytes 和推理引擎未放入默认环境；需要时单独固定版本并重新验证。
8GB 本地量化部署与云端训练是两项实验，不能用量化加载成功推断 8K 训练可行。

## 数据格式与 PT 编码合同

v4.6.0 的本地 JSONL 可传给 `--dataset /absolute/path/train.jsonl`。
推荐导出一行一条纯文本记录：

```json
{"text":"第一章\n文档正文……\n第二章\n后续正文……"}
```

并显式传 `--columns '{"text":"response"}'`。上游也自动将 `text` 识别为 `response` 别名，
其内部标准形式是下面的 **assistant-only messages**；不要人为给 CPT 正文增加用户问题：

```json
{"messages":[{"role":"assistant","content":"第一章\n文档正文……"}]}
```

`swift pt` 使用 `PretrainArguments`，默认 `use_chat_template=false`、`loss_scale=all`。
它与 `swift sft --use_chat_template false --loss_scale all` 对应。
这两种本地输入形式均先经过数据预处理，再由生成式模板编码。
依据：[PT 参数类](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/arguments/pretrain_args.py)、
[本地格式与列映射](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/docs/source_en/Customization/Custom-dataset.md)。

**无聊天模板不等于没有特殊 token。** SWIFT 将模板转为 generation 形式，去掉 user/model turn 包装，
通过目标 tokenizer 提取自动 prefix/suffix；普通未带结束标记的 Gemma 文本通常为 BOS、正文、EOS。
正文已以模板 stop word 结尾等情形会影响是否追加 EOS，所以不能一律将原生正文长度简单加 2。
导出器不能手工重复插入 BOS/EOS，必须按锁定版本的实际模板核对最终 `input_ids`、`labels` 和 loss token 数。
首位置 label 被设为 `-100`；loss token 预算按真正参与 next-token loss 的位置计算。
依据：[generation 模板转换](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/template/template_meta.py)、
[编码与 EOS 处理](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/template/base.py)。

`--cached_dataset` 是 `datasets.load_from_disk` 的缓存入口，**不能视为仅有 input_ids/labels 的免模板直通接口**。
该版本训练管线仍将缓存记录交给 `LazyLLMDataset` 和 `template.encode`，标准输入需要 messages。
本项目首版使用文本接口；若未来实现自定义预编码入口，必须单独验证 collator、labels、位置、长度和 loss 语义。
依据：[缓存读取](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/pipelines/utils.py)、
[训练数据后处理](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/pipelines/train/sft.py)。

## 截断、打包与成对预算

上游 `truncation_strategy` 默认 `delete`，会把超长样本视为异常；`strict` 默认 false，数据处理可能删除坏样本，
惰性读取还可能换取其他样本。首版必须显式 `strict=true`，并在训练前保证每条**模板编码后的总长度**不超过 `max_length`。
CLI 中使用 `--truncation_strategy delete`；它在模板内部映射为 raise，不使用没有公开列入 CLI 枚举的 `raise` 值。
不要以 right/left 截断悄悄丢掉文档末尾，也不要把过滤后的较小数据量当作原预算。

默认 `packing=false`、`padding_free=false`、`streaming=false`，使用已准备的序列单位。
在该版本中开启 packing 会启用 padding_free 并要求 FlashAttention；不能一边使用 SDPA，一边沿用 packing 配置。
`split` 只支持预训练，和 cached_dataset 不兼容；分片首位置的 label 会被 mask，预算随之变化。
依据：[模板参数与截断规则](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/arguments/base_args/template_args.py)、
[packing 检查](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/arguments/sft_args.py)。

ordered/shuffled 两臂使用相同获准源文档、相同原始片段与相同切分规则。重排文本后重新做 BPE 编码，
边界处可能出现不同 token，因此相同字符数、同一批源文档甚至相同 token 总数都不自动证明严格 token 匹配。
准备报告应分别保存每臂的目标 tokenizer 身份、正文/特殊/有效 loss token 数、序列数、长度分布和来源覆盖。
只有实际核验相同 source loss token 身份与预算后才标 strict match；严格检查不通过时应拒绝该对照，
或另存明确标注非严格配对的探索性配置，不能只给较长一臂静默截尾。

## Gemma 4 E2B 的 LoRA 参数

v4.6.0 注册的 `model_type` 是 `gemma4`；E2B/E4B 默认模板名为 **`gemma4_nothinking`**，
不是较大 Gemma 4 的 thinking 模板 `gemma4`。SFT-only 与 CPT+SFT 组必须使用相同的显式模板和损失范围。
纯文本阶段冻结视觉/音频塔与连接器，只在语言部分添加 adapter；冻结不等于从 GPU 自动卸载这些参数。

| 参数 | 首版设置 | 含义 |
|---|---|---|
| `tuner_type` | `lora` | v4.6 参数名；不是旧的 `sft_type` |
| `torch_dtype` | `bfloat16` | 模型加载精度，训练前检查 GPU 支持 |
| `target_modules` | `all-linear` | 由 SWIFT 的多模态架构映射选取语言部分线性层 |
| `freeze_llm` | `false` | 允许语言部分插入 adapter，并非解冻所有基座权重 |
| `freeze_vit` / `freeze_aligner` | `true` / `true` | 保持视觉/音频塔及连接器冻结 |
| `lora_rank` / `lora_alpha` | `16` / `32` | 本项目初始配置，需全部实验臂一致 |
| `gradient_checkpointing` | `true` | 省显存，仍需实测峰值 |
| `attn_impl` | `sdpa` | 首轮兼容性检查；快速 kernel 单独验证 |
| `max_length` / batch | `8192` / `1` | 待服务器冒烟后确认可用，不是显存保证 |
| `seed` / `data_seed` | `17` / `17` | 同时固定模型与数据随机性 |

依据：[Gemma 模型注册](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/model/models/gemma.py)、
[Gemma 模板](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/template/templates/gemma.py)、
[上游 Gemma4 LoRA 示例](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/examples/models/gemma4/train.sh)。
上游示例使用 IT、4096 长度及不同 rank；它证明参数入口存在，不能替代本项目 Base+8K 的验证。

第一轮先检查可训练参数名单、loss 是否有限、checkpoint 能否重载以及真实峰值。
未固定并验证 Liger/分块交叉熵时，不得声称已获得对应省显存收益；Gemma 大词表的完整 logits 本身可能占用数 GiB。
QLoRA 另设配置，不能把 PLE/embedding/LM head 全部默认算成四位，也不能把推理量化和训练量化混为同一产物。

## SFT-only 与 CPT+SFT

推荐三组从同一 Base revision 出发：A 为 Base→SFT，B 为 Base→打乱段序 CPT→SFT，C 为 Base→保留顺序 CPT→SFT。
B/C 正式对照的前置目标是固定 CPT 有效 loss token 预算、序列长度和优化设置；三组还需使用相同获准 SFT 数据、模板、预算及评测题。
C/A 包含额外 CPT 计算，不是等总算力比较；C/B 用于分析组织策略。保留未训练 Base 与 CPT-only 的语言建模评测。

**当前导出尚未完成上述严格配对。** 准备报告明确记录 `comparison.strict_equal_token_budget=false`：
两臂保留相同来源 span，但重排后的 BPE 计数和序列数可能不同；100 个固定训练步仅用于 pilot，
不构成相同 token 暴露量。正式比较前还需实现预算匹配并核验有效 loss token，不能将本轮导出称为已经就绪的等 token 因果实验。

SFT JSONL 使用标准 user/assistant 对话，不能继续将整段问答作为 PT 的全 token 原文：

```json
{"messages":[{"role":"user","content":"根据所给文档回答问题。\n……"},{"role":"assistant","content":"经复核的参考回答。"}]}
```

SFT 明确使用 `--template gemma4_nothinking --use_chat_template true --loss_scale default`，
并固定是否提供 system、引用格式和缺信息拒答规则。检查输出中的 labels，确保用户部分未被当作回答监督。
两种阶段衔接必须选择一种并记录，不能混在同一组结果中：

1. **继续同一个 adapter**：`swift sft --model <同一Base> --adapters <CPT checkpoint> ...`。
   上游会以 `is_trainable=true` 读取 adapter；SFT 显式改回聊天模板与 default loss。
   新阶段不使用 `resume_from_checkpoint` 继承 CPT 的优化器/步数，设置 `--load_args false` 并明确给出模型、模板与数据。
2. **合并后重新做 SFT adapter**：先用 `swift export --adapters <CPT checkpoint> --merge_lora true` 导出完整权重，
   再对合并模型创建新的 SFT LoRA。A 对原 Base 创建同规格新 adapter；B/C 对各自合并权重创建同规格新 adapter。
   保存合并前后校验、来源链及输出 hash，先验证精度一致性；本说明未验证 QLoRA 产物的合并路径。

第二种路线的阶段边界更容易审查，但需要额外存储与合并验证；不在准备阶段自动执行。
依据：[adapter 加载逻辑](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/pipelines/train/tuner.py)、
[官方合并入口](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/examples/export/merge_lora.sh)。

## 运行记录与效果判定

首轮工程验证已使用用户指定的 Colab A100，完成 20 步短样本 SFT；完整长文 CPT/SFT
实验的训练预算与规模仍待确定。其前置工作仍包括数据准入、模型锁定和可审查的 recipe/manifest。
准备命令本身不会租卡、提交远程作业或开始付费训练。
以下项目 CLI 均从 `experiments/ms_swift/` 目录运行。使用准备环境时加 `uv run`；
真正执行训练时必须激活独立训练 venv，直接调用其 `python`，避免 `uv run` 换回轻量准备环境。
命令中的输出目录须为新目录；已有同版本分词资产可直接复用，不覆盖缓存。

```bash
# 只读准入审计，不修改原语料或审核结果。
uv run python -m cpt audit \
  --input ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --output .cache/audit_v2

# 仅下载固定 revision 的惰性 tokenizer/config 元数据，不下载权重。
uv run python -m cpt fetch-tokenizer --output .cache/model_assets/gemma4-e2b

# 默认 preview：生成待审查材料，不获得训练授权。
uv run python -m cpt prepare \
  --input ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --model-assets .cache/model_assets/gemma4-e2b \
  --output .cache/preview_v2

# 生成 ordered 臂计划，不执行；shuffled 臂使用另一个输出目录。
uv run python -m cpt plan \
  --prepared .cache/preview_v2 \
  --recipe configs/gemma4_e2b_pt_lora.json \
  --output .cache/plans/ordered_v2 --strategy ordered

# 默认只复核计划并打印重建的参数，不训练。
uv run python -m cpt run --plan .cache/plans/ordered_v2/launch_plan.json
```

`prepare --mode training --reviews /absolute/path/reviews.jsonl --allow-license <精确许可标识>`
是获准训练导出的显式入口；可重复提供 `--allow-license`。
许可 allowlist 不能代替实际内容复核与来源许可证据，当前 assessment 数据不会因此自动变成已审核训练数据。
preview 计划保留 blockers，`run --execute` 必须拒绝未获准材料。
审核后的 training 导出、计划和输出使用新目录，不能覆盖 preview 或既有实验。

`audit` 在本地生成 `admission_review.jsonl` 待审队列。另存一份填写审核结论：
`document_id` 与 `document_sha256` 必须保持对应原文，`decision` 和 `license_decision`
分别记录内容与许可结论，许可证据填写在 `license_evidence`；获准行还须填写 `reviewer`、`reason`、
`reviewer_type`（`human` 或 `ai_assisted_human`）。AI 独立审查不填成真人复核，待审或缺字段的行不进入训练导出。
软件只能核验凭据格式和原文绑定，不能独立证明审核者身份。显式 benchmark/evaluation-only 记录始终拒绝进入训练，
全文完全重复的来源必须共用同一划分；这些工程检查不替代公开基准上下文的全面去污染。

迁移前的原生分词预览与隔离统计见[历史准备报告](../experiments/results/cpt_20261105_v1/README.md)。
其中 JSON 字节、原始统计与当时的路径声明保持不变；目录迁移与新路径复核另存于
[迁移记录](../experiments/results/cpt_layout_20261105/README.md)。

服务器上完成独立环境验证、准入审核并生成 training 计划后，才显式运行：

```bash
python -m cpt run --plan /absolute/path/training_plan/launch_plan.json --execute
```

PT 执行入口为 `python -m swift.cli.pt`，SFT 为 `python -m swift.cli.sft`；模型 revision 参数名是 `--model_revision`。
项目重建参数而不执行 JSON 中任意给定命令；执行前重新验证数据/recipe/model lock 散列，
并用固定 SWIFT 的 `get_processor(..., download_model=false)`、预处理器及 PT 模板逐条编码 train/dev。
此核验只读取固定元数据，不加载模型权重；超长、丢行、空监督或不符合 full-text PT 的 labels 都会拒绝。
`template_verification.json` 记录实际正文、模板总输入及有效 loss token 数和编码摘要，不能反推训练已经成功。
真正 PT 子进程才会加载/下载锁定来源的模型权重并训练。

计划绑定绝对路径。本次目录迁移保留了本地 `preview_v1` 和 `plans/*_v1` 的原记录，
这些旧记录仍引用迁移前位置，仅作历史证据；新入口使用重新导出的 `preview_v2` 和 `plans/*_v2`。
迁移到服务器后同样需在服务器路径下重新 prepare/plan；
不能移动旧 `launch_plan.json` 后直接执行。中断或已有 checkpoint 的目录不自动续跑，保留 receipt 后另行处理。

每次运行至少保存：输入与模型来源 hash、recipe hash、目标 tokenizer、实际序列及 loss token 统计、软件环境、
可训练参数数/名称、真实训练步数与 token 暴露次数、loss 曲线、显存与速度、失败/中断、checkpoint 与合并/量化 lineage。
不得只记录 `max_steps` 后据此宣称等 token；重复 epoch 和最后 batch 的处理也要核对。

评测区分未训练、CPT-only、SFT-only、CPT+SFT，再区分 BF16 与量化部署。
固定相同留出文本与公开评测题目/指标，报告中文/英文、文档类型、输入长度等分组以及失败分母。
已有 132 次公开 benchmark API 调用只能说明相应推理协议的结果，不能充当 Gemma 训练或本地部署的证据。
小数据单次提升仅是探索结果；若要声称数据组织策略有效，需要扩大独立留出集、重复种子并保留负结果。
