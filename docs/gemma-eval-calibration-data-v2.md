# Gemma 下一阶段：评测校准与配对数据 v2

关联 [Issue #15](https://github.com/tkzzzzzz6/Ariadnian-agent/issues/15)。本轮完成本地数据、CPU 计划和守卫；没有执行新的 Base 校准、SFT 或确认测评。上一阶段的单 seed 结果仍见 [A100 报告](../experiments/results/issue13_sft_20261127/a100_retry/README.md)。

## 本次实际准备

| 项目 | 实际内容 | 状态 |
| --- | --- | --- |
| 配对候选 | 128 train / 32 dev，48 篇来源，32/8 个连接来源族 | 每族最多4题，同问答、来源、split、顺序 |
| 语言 | train：中文68、英文60；dev：中文8、英文24 | 中文共76组 |
| 跨文档 | train 24题/6族；dev 7题/2族 | 意图标签，不是已验证推理准确率 |
| 内容长度 | short 226–1103；long 4394–7463 token | 原生消息内容计数；实际 SWIFT 完整模板待核验 |
| 旧批准 | 原40组、12篇来源逐条两臂和来源哈希相同 | 原内部用途批准继续有效 |
| 新内容 | 120组、36篇新来源 | 待实际人工内容/来源/许可审核；没有写入新批准 |
| 独立评测 | 开发39、示范2、确认候选150 | 确认协议尚未封存、未观察新模型结果 |
| 校准输入 | 24道开发 BoolQ ×4配置，共96次计划调用 | CPU 计划96条均适配16K，原生输入67–674 token；不是推理成功 |

[聚合可视化](../experiments/results/issue15_20261128/report.html)可离线打开；逐条问答、完整输入、来源与 tokenizer 放在忽略目录及私人 Drive，未进入 Git。长短输入 token 和 FLOPs 没有配平；不能把相同更新预算等同于相同计算成本。

## 分区与校准规则

旧250逐字节保持不变，新分区用固定 LongBench/BoolQ 原始版本。先在**完整来源池**中连接 ID、来源族、规范化全文和至少300字符的完整段落/块，排除旧题和训练来源的关联组，最后才按原生长度选题。不合格长度的记录仍参与来源连通，防止切断传递关联。此规则不证明不存在语义重合或模型预训练污染。

LongBench 主任务仍要求输入至少8192，输入加官方输出预算不超过16384；次任务只保留总预算上限，不截断。目标原为开发40、示范2、确认160；NarrativeQA 排除旧题关联来源后只余1个独立族，分给开发，确认缺10题。旧 reserve 的题目数不能解释为独立族数。本次实际确认仅含其余七个 LongBench 任务70题与 BoolQ80题；不支持对独立 NarrativeQA 或新 RULER 的结论。旧 RULER150继续作为回归。

校准只由固定 `google/gemma-4-E2B` Base revision `433813a669639fd7eaa83155828e9000ed357d53`、bf16、无 adapter 选择。四配置固定为零示范8/32/64输出token和2示范32输出token。继续使用原严格整段 Yes/No 解析，不接受解释后提取答案。每配置无失败/空输出、格式有效率至少90%才合格；按上述顺序选择，不按答案准确率选择。若均失败，保持未就绪，不能借确认集选提示词。

所有后续臂和seed共用选中的配置，不另做逐臂挑选；90%是 Base 配置选择门槛，不是删除训练臂失败样本的理由。校准结果必须匹配计划、原始输入、完整请求、模型/adapter/量化契约、原生模板核验、worker 原始预测和外层完整预测；任何混用、加载失败或不完整输出均不能封存。

来源隔离冻结完整350篇候选范围的逐篇哈希。最终48篇来源已经重新核对为该范围的精确子集。新增或修改来源要重做分区与封存；历史文件不改。开发、示范、校准和确认内容也同时绑定，避免封存前替换示范。

## 本地复现入口

在 `experiments/ms_swift/` 中使用独立环境。新 checkout 不自动拥有本机缓存；按 [本地数据说明](local-data-access.md)恢复公开源及原生模型资产，缺失时报告，不假定 Git 带有原始数据。

```powershell
uv run python -m pilot_eval.next_stage `
  --cache <锁定的pilot_eval缓存> `
  --longbench-cache <锁定的publicbench缓存> `
  --regression <原frozen_v1/records.jsonl> `
  --training-scope <完整pipeline_v1/corpus.jsonl> `
  --model-assets <原生gemma4-e2b资产目录> `
  --output .cache/next_stage/new_evaluation `
  --aggregate ../results/<新实验标识>

uv run python -m paired_data `
  --corpus <同一完整corpus.jsonl> --specs <人工可审阅的authored_specs_v2.jsonl> `
  --quarantine <overlap_candidates.jsonl> --assets <原生gemma4-e2b资产目录> `
  --output .cache/next_stage/new_review --aggregate ../results/<新实验标识>
```

候选规格是本次编写的问答，不是公开上游基准。它们在私人备份中与证据、许可证据和审核全文一起保存；纯公开 clone 不能凭来源 URL 自动重建这些人工编写的候选，也不能自动恢复旧批准。

本机本轮忽略目录为 `experiments/ms_swift/.cache/phase2_20261128_v1/`：`review_v2/` 是完整包，`review_delta_v1/index.html` 只列新增120组及36篇新来源，`evaluation_v5/` 是最终分区，`calibration_plan_v4/` 是本机路径绑定的计划。`evaluation_v2/v3/v4` 是未执行模型调用的准备历史，不再作为启动输入。

## 下一次 Colab A100

1. 读取私人 Drive 材料包收据及公开代码的精确 commit。仅 clone 公开 URL，保留 `.git` 供本地回退；不上传 GitHub 用户身份、token 或 `.env`。在新 runtime 重新准备计划，不能编辑本机计划中的路径/哈希。
2. 使用 [Colab 现有环境和tmux流程](gemma-sft-best-tmux.md)；Drive 挂载由用户在终端或 notebook 实际完成。核验真实A100和mount，环境仍用独立Python3.12、固定训练依赖与原生资产。私有材料通过现有 `remote_job.unpack` 的 ZIP/逐文件哈希验证器解包。
3. **先只运行 Base 开发校准**，不调用旧的完整配对训练 supervisor。已准备的96条短输入不能当作8K训练资源验证。以下命令在明确激活的 GPU 环境、`experiments/ms_swift/` 执行：

```bash
python -m gemma_eval prepare --kind chat \
  --input /content/phase2/inputs/evaluation/calibration/records.jsonl \
  --model-assets /content/repo/experiments/ms_swift/.cache/model_assets/gemma4-e2b \
  --context-length 16384 --quantization bf16 --output .cache/phase2/calibration_plan
python -m gemma_eval verify-chat --plan .cache/phase2/calibration_plan/plan.json \
  --output .cache/phase2/template_check
# 在tmux中执行；一次尝试最多1800秒，失败保留，不因分数重试。
python -m gemma_eval.bounded --plan .cache/phase2/calibration_plan/plan.json \
  --output .cache/phase2/calibration_run --max-seconds 1800 --execute
python -m pilot_eval.calibration \
  --records /content/phase2/inputs/evaluation/calibration/records.jsonl \
  --plan .cache/phase2/calibration_plan/plan.json --run .cache/phase2/calibration_run \
  --output .cache/phase2/calibration_assessment.json
```

4. 有合格 Base 配置后，用真实收据与最终来源封存；该命令只准备确认输入，不进行确认推理或训练：

```bash
python -m pilot_eval.seal --prepared /content/phase2/inputs/evaluation \
  --final-sources /content/phase2/inputs/review_v2/sources.jsonl \
  --plan .cache/phase2/calibration_plan/plan.json --run .cache/phase2/calibration_run \
  --model-assets /content/repo/experiments/ms_swift/.cache/model_assets/gemma4-e2b \
  --output .cache/phase2/sealed_confirmation
```

5. 新内容经实际人工批准后，再分别生成两臂 training export、验证实际 SWIFT 输入和监督标签，准备同一seed的有界新预实验。建议待冻结预算为32更新、batch1/累积8，共计划256次序列曝光；这只是两遍128条训练集的预算提案，尚无实际更新或best步。按dev loss选优，每臂/seed稳定保留一份完整best，记录总步数与best前缀曝光；不能直接复用旧40条的8-step runner作为新实验结论。
6. 独立确认使用同一协议，逐题失败保留在原分母。完整best另行验证，轻量归档不代替权重；Drive读回与flush分别留证，再停止本次owned runtime。没有新审核批准时不启动新SFT，也不把候选上传公开HF。

当前尚未测量新数据训练收益、8GB量化质量或跨VM完整训练恢复；CPT继续留待单独算力与方案确定。
