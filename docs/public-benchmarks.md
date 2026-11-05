# 独立公开长上下文基准：方法与复现

`backend/publicbench/` 使用外部公开题目、原始参考答案、固定版本的官方任务模板和评分规则。
它与自建长文的引文定位诊断分开：不套用 Ariadne 的引用 JSON 提示词，不构造证据跨度真值，
也不把受控题当作公开推理基准。实际运行与分析另见
[2026-11-05 公开基准实测报告](public-benchmark-results-20261105.md)。

原始下载、选中记录和逐条预测保存在本机；路径、读取方式与 Git 忽略规则见
[本地数据访问说明](local-data-access.md)。数据工程与后续 CPT 的范围见
[长文数据构建说明](long-context-data.md)。

## 数据与固定版本

| 来源 | 固定版本 |
|---|---|
| [LongBench v1 数据](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，`data.zip` |
| [LongBench v2 数据](https://huggingface.co/datasets/zai-org/LongBench-v2/tree/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9) | `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`，`data.json` |
| [官方模板、长度上限与评分代码](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e) | `2e00731f8d0bff23dc4325161044d0ed8af94c1e` |
| [参考 tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987) | Qwen2.5-0.5B，`060db6499f32faf8b98477b0a26969ef7d8b9987` |

下载器校验数据包、官方配置与代码文件的 SHA256，并保留来源、许可观察、原始 ID、
数据版本及记录的 canonical JSON 散列。只读取下载的代码文本作为版本依据，不执行远端脚本。
本地评分保留上游 MIT 声明，并按固定实现适配；ROUGE 使用锁定依赖 `rouge==1.0.1`。

默认 v1 子集包括以下六项，各选五题：

| 任务 | 方向 | 原始评分规则 | 官方生成上限 |
|---|---|---|---:|
| Qasper | 英文论文问答 | QA F1，取最佳参考答案 | 128 |
| MultiFieldQA-zh | 中文跨领域问答 | 中文分词后的 QA F1 | 64 |
| HotpotQA | 多文档问答 | QA F1 | 32 |
| GovReport | 长报告摘要 | ROUGE-L F1 | 512 |
| TREC | 问题分类 | 官方分类匹配分；保留首行处理 | 64 |
| Passage Retrieval-en | 给定上下文内的段落检索 | 官方段落编号匹配分 | 32 |

v2 从六个官方领域各选两题：Single-Document QA、Multi-Document QA、Long In-context
Learning、Long-dialogue History Understanding、Code Repository Understanding 和
Long Structured Data Understanding。使用官方 non-CoT 模板，生成上限为 128。
原始 `domain`、`sub_domain`、难度、选项与答案保留在记录中。

## 先筛选资格，再冻结样本

默认复现配方为 seed 13，先用固定 Qwen tokenizer 计算原始 `context` 的参考 token 数，
仅将不超过 65,536 的记录纳入抽样池，再按官方 ID 的带 seed SHA256 排序选取。
v1 为六任务各五题，v2 为六领域各两题；样本及顺序在模型调用前冻结。
这是一组**事前长度筛选的小子集**，不是整个 benchmark 的均匀随机抽样。

`fetch_manifest.json` 保存筛选前数量、合格数量、排除数量、选中 ID、输入文件散列、
参考 tokenizer 与官方资产版本。筛选不看答案内容或模型表现；失败后不换题、不补抽。
参考 token 不是服务商原生 token，也不等于包含问题、指令和聊天模板的完整请求长度。

当前本地 tokenizer 路径为 `backend/longdoc_sources/.cache/public_v1/tokenizer.json`。
新克隆若没有该文件，先按[长文复现说明](long-context-data.md)准备固定版本，或提供同一版本
的本地 tokenizer。不能把文件缺失误报为已有数据。

## 根目录 npm 入口

下面命令在项目根目录执行。入口会转入 `backend/`，因此参数中的相对路径均相对于
`backend/`。示例使用新的缓存和运行目录；已有非空运行目录不会被覆盖。

```powershell
npm run eval:public -- fetch --benchmark both --per-task 5 --v2-per-domain 2 --seed 13 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --max-context-tokens 65536 --output publicbench/.cache/selected_example
npm run eval:public -- run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_full --use-configured-model --limit 42 --thinking disabled --strategy full --context-tokens 65536 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --timeout 60
```

`run` 要求显式开启模型并提供正数 `--limit`。调用前先检查相邻的 fetch manifest、
文件 SHA、记录数和 ID 顺序；来源记录提供的原始 JSON 散列、提示词散列及当前渲染结果也会核验。
缺失来源清单或发现不一致时，在模型设置和请求之前失败。

`--thinking default` 保留服务商默认行为；`disabled` 或 `enabled` 只影响该次请求，
不修改用户持久配置。每题至多一次有官方输出上限的请求，无传输重试、备用模型或取消上限的重试。

## backend 中的 uv 入口

以下命令与上述 npm 入口等价，在 `backend/` 执行：

```powershell
uv run python -m publicbench fetch --benchmark both --per-task 5 --v2-per-domain 2 --seed 13 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --max-context-tokens 65536 --output publicbench/.cache/selected_example
uv run python -m publicbench run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_full --use-configured-model --limit 42 --thinking disabled --strategy full --context-tokens 65536 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json --timeout 60
```

不需要重复执行 npm 和 uv 两组命令；选择其中一种入口即可。
首次 `fetch` 下载固定公开数据与配置，后续复用并校验 `backend/publicbench/.cache/`。
大体积输入留在缓存，结果目录统一使用 `backend/evals/results/publicbench_*/`，小型汇总与逐条数据的
跟踪规则由[本地数据访问说明](local-data-access.md)约定。

## 完整上下文与 8K 预算分开报告

`full` 将完整 context 交给固定的官方模板，不因预算静默裁剪。超过本次参考长度 guard 的题
明确标记为跳过，原始题仍留在分母；服务商窗口错误记为失败，不截断后重试。
完整上下文运行称为 `official_prompt_full_context_subset`，不称官方全量榜单成绩。

请求只有官方模板组成的一条 user 消息，没有附加 system 指令或自建引用 JSON 格式。
API 仍会应用模型本身的聊天包装。本项目固定 `temperature=0`，而该版本 v2 上游 non-CoT
脚本使用 `temperature=0.1`；所以“官方模板、长度上限与指标”不表示完全相同的官方推理配置。
v2 的 `strip` 与连续占位符替换遵照固定上游实现，正文中潜在占位符字面替换另有审计元数据。

8K 对照使用同一固定 v1 样本，分别新建输出目录：

```powershell
npm run eval:public -- run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_bm25_8k --use-configured-model --limit 30 --thinking disabled --strategy bm25 --context-tokens 8192 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json
npm run eval:public -- run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_structure_8k --use-configured-model --limit 30 --thinking disabled --strategy structure --context-tokens 8192 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json
npm run eval:public -- run --input publicbench/.cache/selected_example/selected_records.jsonl --output evals/results/publicbench_example_truncate_8k --use-configured-model --limit 30 --thinking disabled --strategy truncate --context-tokens 8192 --tokenizer longdoc_sources/.cache/public_v1/tokenizer.json
```

上述 `both` 输出先放 v1，再放 v2，因而这个固定配方的前 30 条为相同 v1 题。
backend 对应入口是 `uv run python -m publicbench run`，其余参数不变。

预算策略按空行分段，并将长段落切成不重叠的字符窗口，默认每块最多 2,048 个 Unicode 字符。
BM25 在题目已经给出的 context 内选块；`structure` 额外使用启发式识别的标题词项加权，
没有邻接块扩展；`truncate` 保留开头。选中块恢复原文顺序，最终文本再次核对参考 token 预算。
这既不是外部文献库的全库检索，也不表示识别出的标题已经人工核验。

这些策略更换了输入 context，必须称为 modified budget protocol，与完整原文分表比较。
保存的来源跨度、上下文 SHA、实际长度和请求 SHA 用于检查策略是否真的产生不同输入。
若两种策略选中文本相同，就不能把输出分数波动归因于结构改进。

## 严格评分、覆盖率与恢复记录

v1 保留官方任务特定的规范化、中文 Jieba 分词、最佳参考聚合和适用的首行处理。
v2 只使用官方提取规则：`The correct answer is (A)` 或 `The correct answer is A`；
裸 `A` 不会事后修补为正确格式。参考答案只进入评分，不添加到请求中。
题目原文已有的 few-shot 样例属于官方 context，不是后来拼入的隐藏 gold。

超时、空答案、跳过和评分失败保留在分母中并计零，覆盖率另外报告。
非空但错误的回答仍是一次有效返回，不能把“可评分”写成“答对”。
触及官方生成上限且仍有非空答案时，按官方规则评分，同时保留 `finish_reason=length`。
v1 的 F1、ROUGE-L、分类分和检索分各有含义；只报告任务表及相应宏平均，不把它们解释为统一正确率，
也不将 v1 与 v2 混成一个总分。

每题完成后，已脱敏的 `predictions.partial.jsonl` 会 flush 并同步到磁盘；
`inprogress_manifest.json` 在请求前和题目完成后原子更新。
中断保留已完成输出与已知用量，正在等待的请求明确标为用量未知。
最终产物为逐题预测、`summary.json`、`report.md` 和所执行记录；API 的实际输入、输出、缓存和推理
token 与参考 tokenizer 统计分开。费用没有可核验返回值时保持 `null`。
原文在内存中评分后，才对全部待写字符串使用凭据快照脱敏，避免脱敏先改变答案评分。

离线重评分优先使用带 fetch manifest 的原始输入缓存：

```powershell
npm run eval:public -- rescore --input publicbench/.cache/selected_example/selected_records.jsonl --predictions evals/results/publicbench_example_full/predictions.jsonl --output evals/results/publicbench_example_rescored
uv run python -m publicbench rescore --input publicbench/.cache/selected_example/selected_records.jsonl --predictions evals/results/publicbench_example_bm25_8k/predictions.jsonl --limit 30 --output evals/results/publicbench_example_bm25_rescored
```

第二条在 `backend/` 执行；第一条在根目录执行。`rescore` 不读取模型凭据或发送模型请求，
会重新计算指标并拒绝重复、未知 ID 及不匹配的记录散列。缺失预测计零。
如无匹配的 fetch manifest，只能生成明确标注 `unverified_input_without_fetch_manifest` 的复评分，
不会将其升级为来源已核验的官方模板子集成绩。

## 许可与结论范围

官方代码使用 MIT 许可；v1 的各原始任务没有在本模块中合并成统一训练许可，
v2 数据卡的 Apache-2.0 声明按固定来源保存。代码许可、数据卡声明和第三方内容范围分别记录，
不能由“可公开下载”直接推导出所有训练用途的授权。
v2 在数据服务中名为 `train` 的源 split，在本项目仍固定为 evaluation-only，不进入训练混合。

这些题目用于独立评测固定模型在不同输入策略下的表现，不能当作新增训练语料，
也不能证明清洗、packing、配比或长文合成已经提高模型能力。没有运行 CPT 或权重更新时，
不作预训练数据策略的因果收益、智力上限或稳定总体提升结论。
每任务少量题目用于发现问题和形成下一轮假设；更广泛的结论需要扩大样本并开展相应训练对照。
