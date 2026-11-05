# 长文数据构建与多方向评测

Ariadne 在研究报告助手之外增加了独立的长文数据实验模块。应用仍用原有的检索块；
长文语料保存完整文档、章节路径、正文顺序和来源，不从带 overlap 的检索块重拼原文。

当前范围为 P0–P2：评测可信度、长文语料管线与多方向效果验证。
**尚未运行 Continued Pretraining，也没有训练数据提升模型能力的结论。**

本次已执行的数据规模、真实结果和失败分析见[2026-11-05 实测报告](long-context-data-results-20261105.md)。

独立公开基准另见[方法与复现](public-benchmarks.md)和[公开基准实测](public-benchmark-results-20261105.md)。它们使用外部官方题目和参考答案，与本项目构造的受控诊断、AI 候选分别统计。

## 模块与输出

| 模块 | 用途 |
|---|---|
| `backend/evals/` | 原应用评测的 v2 口径、失败追踪、候选负例复核、输入/代码/请求版本记录 |
| `backend/longdoc_sources/` | 固定公开来源版本、下载校验、Markdown/RST/JATS 结构化、真实参考 token 统计 |
| `backend/longdoc/` | 可追溯清洗、去重候选、质量分、文档族划分、packing、按 token 配比、数据诊断 |
| `backend/longeval/` | 长文任务校验、三种上下文策略、证据与答案评分、显式开启的模型调用、外部数据导入 |
| `backend/publicbench/` | 固定版本 LongBench v1/v2、官方任务评分、全上下文与预算策略的独立对照 |

所有下载语料与大体积处理中间结果默认放在 `backend/longdoc_sources/.cache/`，
不进入 Git。实际运行报告和固定来源清单另行保存。`sources.lock.json` 允许复取同一批来源；
原文内容改变、哈希不匹配时不会悄悄接受新版本。

公开基准下载在 `backend/publicbench/.cache/`；逐条任务和预测也被忽略，小型汇总与来源清单保留。忽略不影响本机 Agent 按路径读取，远端克隆需要重新获取，见[本地数据访问说明](local-data-access.md)。

## 复现一轮实验

以下在 `backend/` 中执行。`uv run` 使用项目已有依赖；tokenizers 已由锁定依赖安装。

```powershell
# 1. 首次获取公开原文与固定版本 tokenizer；后续步骤不需要联网
uv run python -m longdoc_sources fetch --output longdoc_sources/.cache/public_v1

# 2. 构建完整语料、审计与两种 packing。输出目录必须为空或尚不存在
uv run python -m longdoc build --input longdoc_sources/.cache/public_v1/documents.jsonl --output longdoc_sources/.cache/pipeline_v1 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --sequence-length 8192 --min-quality-score 0.95 --mix-token-budget 1000000

# 3. 构造有明确真值的受控诊断，以及等待复核的自然数据任务
uv run python -m longdoc assess --input longdoc_sources/.cache/pipeline_v1/corpus.jsonl --output longdoc_sources/.cache/assessment_v1 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --sequence-length 8192 --cases-per-type 20
uv run python -m longeval build-diagnostics --documents longdoc_sources/.cache/pipeline_v1/corpus.jsonl --output longdoc_sources/.cache/diagnostics_v1.jsonl --max-tasks 360
uv run python -m longdoc_sources.study --documents longdoc_sources/.cache/pipeline_v1/corpus.jsonl --output longdoc_sources/.cache/study_v1

# 4. 长文上下文选择基线（不调用模型）
uv run python -m longeval run --documents longdoc_sources/.cache/pipeline_v1/corpus.jsonl --tasks longdoc_sources/.cache/diagnostics_v1.jsonl --output-dir evals/results/longdoc_v1/bm25 --strategy bm25 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --budget-tokens 8192
```

将最后一条的 `--strategy` 改为 `structure` 或 `truncate`，并使用不同输出目录，
即可比较章节标题加权检索与按原文顺序保留上下文。`truncate` 是有预算的直接输入基线，
不能把超出窗口的尾部当作已经输入模型。任务与引用原文可先用 `longeval validate` 核验。

根目录也提供 `npm run data:fetch`、`data:build`、`data:assess`、`data:study`、`data:synthesize` 和
`eval:long`。这些入口将参数传给对应 CLI；路径仍相对于 `backend/`。

显式开启模型验证：

```powershell
uv run python -m longeval run --documents longdoc_sources/.cache/pipeline_v1/corpus.jsonl --tasks longdoc_sources/.cache/study_v1/qa_candidates.jsonl --output-dir evals/results/longdoc_v1/model_qa --strategy bm25 --use-configured-model --tier write --limit 6 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --budget-tokens 8192 --max-output-tokens 2048 --timeout 60 --thinking disabled
```

模型调用使用现有设置，但在临时数据库中计量；不修改用户文献库和用量记录。
每题一次有输出上限的请求，失败保留在分母中。提示词包含问题与所选原文，不包含参考答案。
Qwen tokenizer 只用于可比的上下文预算和语料分桶；API 返回的实际 token 用量另外报告。
答案 EM/F1 只适用于对应题型，不能代替摘要/比较任务的语义质量评审。

`--thinking default|disabled|enabled` 只影响本次请求。实际服务商需支持对应参数；
默认 `default` 保留服务商行为。当前 DeepSeek 默认开启思考模式，有限输出预算可能
在最终 JSON 之前耗尽；显式关闭的格式见[官方接口说明](https://api-docs.deepseek.com/guides/thinking_mode/)。
失败和耗尽预算的尝试仍保留，不能当作有效回答。

自然跨章/跨文档 QA 候选另用显式生成入口：

```powershell
uv run python -m longdoc_sources.synthesize --documents longdoc_sources/.cache/pipeline_v1/corpus.jsonl --output longdoc_sources/.cache/generated_qa_v1 --limit-documents 6 --tasks-per-document 2 --use-configured-model --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --max-output-tokens 4096 --timeout 60 --thinking disabled
```

不加 `--use-configured-model` 时只保存计划，不调用模型。开启后每个锚文档至多一次请求，
按语言和领域轮转抽样；可加入同一 split 的相关文档作为伙伴。每道题需要不同章节或来源的
多条证据，自动检查原文引文和出处，拒绝把待找引文直接贴进问题的捷径。生成结果始终是
`ai_generated / unreviewed`，结构校验通过不等于答案语义已经人工核验。

## 六类测评与证据边界

| 方向 | 已实现的测量 | 需要区别的内容 |
|---|---|---|
| 结构保留 | 原块不变、章节路径、代码/表格/公式保护、序列字符跨度完整性 | 序列完整性不是 PDF 解析准确率；自然版面审阅队列为空字段 |
| 清洗与去重 | 精确/近重复、共享模板难负例、受控扰动与误删风险、自然候选复核 | 受控诊断和自然重复候选分别统计；近重复默认只做候选 |
| 质量筛选 | 乱码/控制字符/重复段落维度，噪声注入敏感度与筛选记录 | 规则分不代表语义质量，更不是独立人工核验 |
| 长距离证据 | 单段、跨章、缺失信息，按语料长度/证据距离切片 | 精确引文定位诊断不等于多跳推理 |
| 跨文档与结构化内容 | 多来源证据归因、表格与正文联合定位、证据完整命中 | 自然 QA 候选需复核；抽取式编排不冒充生成式推理数据 |
| 序列与配比 | 顺序/随机段落组织、利用率、超长块审计、来源 token 配比 | Packing 效率不等于模型能力收益，CPT 对照仍待算力 |

训练/开发/测试按文档族与重复连接簇划分，不按段落随机拆分。
跨文档衍生样本的父文档必须同属一个 split。已知公开 benchmark 仅用于评测，
导入器不将其标为人工自建 gold，也不宣称受限子集是官方全量成绩。

## 标注与结果阅读

- `controlled_synthetic`：可由构造过程验证的受控诊断，报告其实际测量范围。
- `ai_generated` / `unreviewed`：AI 编写或自动模板产生、仍需复核的自然任务。
- `human_verified`：仅用于确有人工核验依据的样本；当前构建器不会自动赋予此状态。
- review 队列里的空字段表示未审，不能算作通过。人工、AI 辅助、纯模型复核分别统计。

结果包含逐条预测、汇总 JSON、Markdown 报告、输入哈希、参考 tokenizer、
模型/提示词版本和失败记录。证据指标同时报告 precision、recall、完整证据集命中，
不会把“命中任意一个证据”写成真正的多证据 recall。

原应用评测协议 v2 不再把 unknown 计为负例正确拒绝；自动错配仅列为待复核负例。
历史 JSON 保持原样，旧口径结果不能与 v2 直接作提升对比。详见 `backend/evals/README.md`。

## 后续 CPT

当前只提供 assessment 数据与实验输入，未运行训练。确认算力后，再选择基座和原生窗口，
固定训练 token、优化设置与测试集，比较结构、清洗筛选、序列组织和配比。
在此之前，可报告数据工程与固定模型推理结果，不报告预训练策略的因果收益。
