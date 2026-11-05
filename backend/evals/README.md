# 测评

衡量 Ariadne 里所有「由模型做判断」的环节到底靠不靠谱。测评直接调用系统里真实在用的函数（`assist.judge_support`、`assist.verify_citation`、`retrieval.search`、`assist._review`、`assist.complete`、`pdfview.locate`、`pdfview.passages`），换模型、改 prompt 之后跑一遍就知道是变好还是变坏。

```bash
npm run eval                                   # 用网页「设置」里配置的模型，跑全部测评
npm run eval -- --limit 30                     # 每项只取 30 条，几分钟内出结果
npm run eval -- --model deepseek/deepseek-chat # 临时换成别的模型做对比（Key 仍用设置里的）
npm run eval -- --tasks judge,fake_cite        # 只跑部分任务
npm run eval -- --offline                      # 不调用模型，只跑离线基线
```

- 测评在临时数据目录里进行，不会改动你的文献库、论文和用量统计。
- 结果保存在 [`results/`](results/)，每次一个 JSON；[`results/README.md`](results/README.md) 是自动生成的对比表。
- 全量一次约 900 次模型调用（以 judge 档为主，每次几百 token），用 DeepSeek 这类模型通常只要几元钱；可以先用 `--limit 30` 试跑。

## v2 评测协议与复现

新运行使用 `ariadne-eval-v2`。现有 `202610*.json` 是历史 v1 结果，保留原样；重新生成的汇总页会把两种协议分开。**不能将 v1/v2 数字直接视为效果提升**，口径有以下变化：

- 判断的 `accuracy` / `total_accuracy` = 正确条数 / 全部样本；`valid_accuracy` = 正确条数 / 有效判定样本。有效输出为 supports / partial / unrelated；unknown 是弃答，非法输出与异常是失败，两者均不算正确。二分类下只有 supports 表示支持，partial / unrelated 表示拒绝支持。同步报告 coverage、abstention_rate、failure_rate 和逐样本错误；被应用层捕获的“判断失败”也计入失败。
- v1 的 `recall@k` 实际是至少命中一项的 Hit@k。v2 同时报告宏平均 `recall@k`（命中证据数 / 该查询全部 gold 证据数）、`hit@k`（至少命中一个）、`complete@k`（命中整个 gold 证据集）与 MRR；无 gold 的查询不可评分，单列 `n_unscorable`。
- 自动随机错配、检索难例和中文错配均为**未人工复核候选**。“不在已知 gold 中”不等于“不支持”。它们不会产生 gold 检出率；中文判断只评分已有支持标签的译文正例，不能与旧中文正负混合准确率相比。
- `fake_cite` 仅将明确人工复核的 `not_supported` 对计入负例检出率。没有人工负例时，检出率为 null，不能写成通过；自动候选的标红比例仅为候选诊断。
- 修复了测评文献占位对象缺少 `id` 的兼容性问题；当前生产判断函数需要该字段。v2 会保存逐样本结果，防止整批 unknown 被汇总准确率掩盖。

每次新结果含代码 commit、运行前后 dirty 状态、源文件散列、数据文件散列、实际抽样输入散列、实际模型请求/系统提示词散列、提示词源代码散列、脱敏后的有效模型配置、抽样种子和协议版本。定位任务还记录实际生成 PDF 的散列。请求审计覆盖 `app.llm.chat` 及 JSON 解析后的修复请求；服务商传输层重试和向量调用不逐次拦截。不保存 API Key；服务地址仅保留 origin 和路径散列，移除用户信息、查询和 fragment。离线模式会关闭向量 API，使用关键词规则和纯 BM25。`locate` 的合成构造/排版仍固定 seed=7/11，其他任务由 `--seed` 控制。

```bash
npm run eval -- --offline --source fixture --seed 13
npm run eval -- --tasks judge,fake_cite,retrieval --seed 13 --results-dir evals/results-v2
npm run eval -- --tasks fake_cite --reviewed-negatives /path/to/reviewed-negatives.jsonl
```

保存结果时，同目录会输出 `*.negative-candidates.jsonl`，含稳定 candidate_id、完整论断与证据、候选类型及空白标注字段。审阅者需阅读完整论断和证据，再填写：

```json
{"candidate_id":"保持导出的ID","claim":"保持原论断","evidence":"保持原证据","label":"not_supported","review_status":"human_reviewed","reviewer":"实际审阅者","review_note":"判断依据","independent_review":false}
```

label 可为 `not_supported` / `supports` / `ambiguous`；未标注保持 null。导入只评分前者，并要求 human_reviewed、非空 reviewer 和内容匹配的 ID；改动论断/证据、与现有正例冲突或同 ID 标签冲突会拒绝导入。`independent_review` 仅在确实完成独立人工审阅时填 true；文件格式通过不证明标注语义正确，AI 辅助复核不得表述成独立人工金标。

## 测什么

| 任务 | 对应功能 | 做法 | 主要指标 |
|---|---|---|---|
| `judge` 引用判断 | 找引用、页边批注 | 给定论断和一段文献原文，判断 supports / partial / unrelated | 准确率、**假支持率**（撑不起的证据被判成“支持”）、漏判支持率 |
| `judge` · 中文论断 | 同上 | 中文论断 + 英文原文；只评分原数据支持标签的译文正例，随机错配导出为未复核候选 | 正例支持率、覆盖率、弃答/失败率 |
| `fake_cite` 假引用检出 | 页边批注的引用核查 | 自动产生随机/检索难例候选；只有导入人工复核负例才计入 gold；走 `verify_citation` 同一条路径 | 人工负例检出率、正确引用误报率、覆盖率、候选诊断 |
| `retrieval` 检索 | 找引用、文献问答 | 在 1,489 段语料里为论断找支撑文献；英文 / 中文 / 中文 + 跨语言关键词扩展；配置了向量检索时再测混合检索 | 真 Recall@1/5/10、Hit@k、完整证据集命中率、MRR@10 |
| `missing` 缺引用检测 | 页边批注「这句话缺少引用」 | 20 段中文研究报告段落逐句标注是否需要引用；过滤规则与 `annotate()` 完全一致（每段最多 2 句、必须原样摘录） | 句级 Precision / Recall / F1 |
| `locate` 原文定位 | 「看原文」高亮、预览里的「最相关的原文」 | 用 SciFact 支持对造出双栏论文 PDF（每段 = 证据 + 3 句与论断用词相近的干扰句，另混入纯干扰段落，排成两栏、跨页），走真实的解析 → 切块 → `locate` / `passages`；用 PDF 里每个词的坐标（另一套方法）判断高亮落点 | **定位准确率**（高亮文字 ≥60% 是证据）、高亮精确率、证据覆盖率、相关段落 Recall@1/3；中文论断同样测一遍 |
| `latency` 延迟与花费 | 行内续写 | 20 次真实续写请求；另外汇总本次测评所有模型调用 | 续写 P50/P95、各档调用次数、错误率、token、花费 |

“标红”指判定为 unrelated（页边卡片红色），“含黄色”指判定不是 supports（红色或黄色卡片都会提醒用户）。

## 数据

| 数据 | 来源 | 规模 | 位置 |
|---|---|---|---|
| SciFact 论断—证据对 | Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020；使用 [COVID-Fact 仓库](https://github.com/asaakyan/covidfact)（Saakyan et al., ACL 2021）整理的 RTE 格式 dev 集 | 209 对（138 支持 / 71 不支持） | 首次运行时自动下载到 `evals/.cache/`（CC BY-NC 2.0，不放进仓库） |
| 检索干扰项 | 同上仓库的 SciFact test 证据 + COVID-Fact 证据 | 去重后共 1,489 段语料 | 同上 |
| 中文论断 | 60 条 SciFact 论断的中文译文 | 60 条 | `data/zh_claims.jsonl` |
| 中文缺引用 | 本项目编写并逐句标注的研究报告段落（计算机、农学等课题） | 20 段 / 72 句，其中 26 句需要引用 | `data/zh_citeworthy.jsonl` |
| 冒烟样例 | 手写 | 6 条 | `data/fixture.jsonl`（`--source fixture`，不联网） |

GitHub raw 连不上时会自动改用 jsDelivr 镜像；都失败的话，按提示手动下载文件放进 `evals/.cache/` 即可。

## 离线基线

本节数字来自 v1 历史结果，其中“难例假引用”为未复核候选，“Recall@5”为旧 Hit@5；新结果请按 v2 口径读取。

`--offline` 不调用任何模型，用离线演示模式里的规则代替：引用判断按关键词重合数（≥3 个算支持，≥1 个算部分支持），缺引用检测标出段落里第一句较长的句子。它是模型结果的下限参照：

- 关键词规则对 SciFact 的**假支持率约 51%**，难例假引用**一条都标不红**——这正是需要模型判断的原因；
- 纯 BM25 用中文论断检索英文文献，Recall@5 只有约 27%（命中的基本是基因名这类英文术语），英文论断则有约 85%——跨语言关键词扩展要把中文拉近英文水平才算有效。

## 原文定位：改进前后

`locate` 测评是为「看原文」功能写的，第一次跑就暴露了三个问题，修复后（离线，109 条论断、15 份共 44 页的双栏 PDF）：

| | 定位准确率 | 高亮精确率 | 证据覆盖率 | 相关段落 R@1 / R@3 |
|---|---|---|---|---|
| 基线：整块高亮 | — | 27.6% | 100% | — |
| 基线：只高亮首句 | 17.4% | — | — | — |
| 改进前 | 48.6% | 53.6% | 34.0% | 54.1% / 54.1% |
| **改进后** | **79.8%** | **82.7%** | **71.7%** | **76.1% / 84.4%** |

1. **切块把句子粘在一起**：超长段落按句切分时吃掉了句间空格（`facts.Purely`），两句被当成一句，挑句子时经常挑到「半句干扰 + 半句证据」。
2. **只挑一句**：证据常常跨两三句。改为先找得分最高的句子，再并入紧挨着、得分不低于一半的句子（最多 3 句）；重合词按在片段中的稀有程度加权，英文词做粗略词干（infected / infection）。
3. **相关段落被截断**：PDF 的文本块常常把几段并成一大段，返回时截到 600 字就丢了证据。改为长段落按句子切成 ≤520 字的小段再打分，结果也更聚焦。

剩下的错误主要是干扰句与论断用词重合更多（测评特意这样构造）——纯词汇匹配的上限；中文论断在离线模式下没有英文关键词扩展，接入模型后用 `--tasks locate` 再测。

## 已知局限

- SciFact 是生物医学领域，和课程报告的领域不同；证据是摘要里的句子，不是全文片段。
- `locate` 用的是合成排版的 PDF（文字真实、版面统一），真实论文里的公式、图表、页眉页脚会让定位更难，数字偏乐观。
- 中文缺引用集规模小，标注带有主观性，适合看趋势和比较模型，不适合当作精确的绝对值。建议按自己的学科补充段落。
- `judge` 的标签只有“支持 / 不支持”，SciFact 里“反驳”和“信息不足”都归入“不支持”。
