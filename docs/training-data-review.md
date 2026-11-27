# 长文训练数据审核包

这个入口把现有语料整理成可逐篇检查的本地页面、待填写凭据和片段重合候选。它不调用模型，不修改原文或划分，也不会批准数据或启动训练。

当前可打开 [2026-11-07 第二版审核索引](../experiments/ms_swift/.cache/review_20261107_v2/index.html)。该链接指向被 Git 忽略的本机文件；新克隆需要用下列命令重新生成。第一版作为历史记录保留。

## 本轮实际结果

完整语料为 350 篇：train 281、dev 41、test 28，覆盖农业、生命科学、机器学习、数值计算、数据科学、编程和数学，英文 281 篇、中文 69 篇。公共测评输入是原先冻结的 LongBench v1/v2 共 42 题。

| 项目 | 实际结果 |
|---|---:|
| 首批待审文档 | 31 篇：train 24、dev 7 |
| 首批语言 | 英文 27、中文 4 |
| 首批领域 | 7 个；data_science 的原划分没有 dev 文档 |
| 首批原参考 tokenizer 计数 | 263,255 token；不是 Gemma 有效训练 token 预算 |
| 完整语料跨划分重合候选 | 158 对，涉及 97 篇文档 |
| 与提供的 42 个公共 context 重合候选 | 当前诊断阈值下 0 对 |
| 首批 31 篇的重合候选 | 当前阈值下均为 0 对 |
| 审核通过数 | 0；所有凭据仍为 pending |

候选优先级是：同一 `split/domain/language` 内，重合候选较少、许可元数据/证据存在、最后按固定 seed 的文档 ID 哈希排序。train 每层最多 3 篇，dev 每层最多 1 篇；test 不进入待训练候选。这个选择便于开展首批审核，不是统计代表性抽样，也不是语义质量排名。

158 对候选中，既有公式渲染前导 LaTeX 的重复，也有不同章节共享代码，以及 D2L 英文/中文同名章节的跨划分候选。4 对具备“跨语言、同文件名、共享词法片段”线索，需检查是否应视为同一来源家族。工具不会把所有命中直接合并或排除；原 split 保持不变。若确认翻译/共享来源关系，应另建带有家族约束的数据版本，保留本轮历史记录。

详细汇总见 [data_review_summary.json](../experiments/results/readiness_20261107/data_review_summary.json)。

## 重新生成

从 `experiments/ms_swift/` 的轻量准备环境执行，选择新的输出目录和汇总文件名：

```bash
uv run python -m cpt.review \
  --input ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --benchmark ../../backend/publicbench/.cache/selected_v1/selected_records.jsonl \
  --output .cache/review_20261107_v2 \
  --summary-output ../results/readiness_20261107/data_review_summary.json
```

本机已有上面的目录，不能原地重复运行；请改用 `v3` 和新的汇总名称。参数 `--per-stratum`、`--dev-per-stratum`、`--seed` 控制首批规模与排序，`--ngram`、`--window` 控制词法诊断。源文件内容在运行前后核对 SHA-256，输出目录和汇总文件均拒绝覆盖已有记录。

## 审核包包含什么

| 本地文件 | 用途 |
|---|---|
| `index.html`、`documents/*.html` | 逐篇完整清洗正文、原始文本差异、来源链接、许可文字、质量诊断、相关重合证据 |
| `selected_reviews.pending.jsonl` | 首批待填写的内容和许可审核凭据，绑定原始文档哈希 |
| `selection_review.jsonl` | 每篇入选文档的 split、领域、语言、重合对数量与选择理由 |
| `overlap_candidates.jsonl` | 所有候选对及原文偏移/片段证据；每对最多展示 3 个片段 |
| `family_review_candidates.jsonl` | 跨划分来源家族关系的待审建议，不自动修改家族或 split |
| `frozen_corpus.jsonl`、`frozen_benchmark.jsonl` | 原始文档/测评记录哈希与固定划分 |
| `admission/` | 完整语料的待审队列与未通过原因 |
| `review_manifest.json` | 输入、实现与全部已生成文件哈希、范围和计数 |

HTML 正文和来源字段都被转义，页面不执行脚本，不嵌入远程资源。正文、审核凭据和逐条候选全部位于被忽略的 `.cache/`；版本化汇总只包含计数、配置、路径和哈希。

## 从待审材料到训练导出

1. 逐篇阅读正文，核对实际来源、第三方内容、许可适用范围、清洗是否损坏结构以及关联的重合候选。存在命中时，在审核理由中记录处置依据。
2. 复制 `selected_reviews.pending.jsonl` 为独立 `reviewed.jsonl`。只有完成真实人工复核，才填写 `reviewer_type` 为 `human` 或 `ai_assisted_human`，以及 `reviewer`、`reason`、`decision`、`license_decision`、`license_evidence`。不批准的材料保持 pending 或填 rejected；不修改原始 hash/split。
3. 把完整冻结语料作为 `--input` 重新运行 `cpt audit`，不要只截取首批文档后重新分组。这样可以保留原有家族和上游来源约束。`--allow-license` 仅列出已核实的精确许可标识，允许多次传递；许可列表本身不代替逐条审核。
4. 审计确认后，以新的目录运行 `cpt prepare --mode training --reviews ... --allow-license ...`。训练导出仍需要原生 tokenizer、长度和模板 loss-token 核验；审核包本身没有训练资格。

```bash
# 下面的 reviewed.jsonl 必须由真实审核产生，不能直接使用 pending 凭据冒充通过。
uv run python -m cpt audit \
  --input ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --reviews .cache/review_20261107_v2/reviewed.jsonl \
  --allow-license CC-BY-4.0 \
  --output .cache/audit_reviewed_new
```

上面的许可值仅示范单一已复核许可；本批包含多个许可标识，未显式允许的文档仍会被排除。软件能核对凭据和原文绑定，不能独立证明审核者身份。

## 诊断边界

默认把汉字逐字、其余 Unicode 字母数字串按词切分，逐 token 做 NFKC 和大小写归一化，忽略标点和空白；这不是模型分词。对 20-token 连续片段做 64 位滚动哈希，在每 12 个相邻片段中保留最右侧最小值，随后对候选再次核对 token 完全一致。满足归一化 token 对齐且无哈希碰撞的条件下，连续 31 token 的相同区域属于该 winnowing 参数的检测保证范围。

同一文档中相同指纹合并为一个代表位置；候选计数是不同选中指纹数，不是出现次数或最长公共片段长度。会扫描所有文档和候选对，只限制展示的证据数。比较范围是 corpus 的跨 split 对，以及 corpus 与所给公共 context 的对；同 split 内重复不在本工具范围。

没有命中不能证明无污染：较短相同片段、释义、翻译和语义重复仍可能漏过，且这里只覆盖提供的 42 道题，没有覆盖整个公开基准或基座模型未知的预训练数据。片段重合诊断、许可元数据和工程测试都不能替代内容审核，也不能证明 CPT 改善了模型。
