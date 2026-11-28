# 配对长文预实验的公开评测

这是独立实验工具，不导入应用 backend，也不安装到 backend 环境。当前协议是
**探索性小样本评测**，不是 LongBench / RULER 官方完整榜单。准备数据不加载模型权重。

## 固定范围

| 数据 | 每个 checkpoint | 用途 |
|---|---:|---|
| LongBench v1 | 60 | 四项 primary 各5题；四项 secondary 各10题 |
| RULER | 150 | 五任务 × 4K/8K/16K ×10题 |
| BoolQ validation | 40 | 短文本能力回退检查 |

primary 是 hotpotqa、multifieldqa_en、narrativeqa、gov_report；其完整 Gemma 对话输入
至少8192token，输入加官方生成预算至多16384。secondary 是 qasper、multifieldqa_zh、
multi_news、2wikimqa，总预算也不超过16384。冻结前不观察模型结果。

RULER 使用锁定官方生成器和词表/文章/SQuAD 快照。Windows 启动器仅把原 shell 命令改为
参数列表，生成器不改；恢复原始 `input + answer_prefix` 后，套用所有模型共同的 Gemma
对话模板。生成预留16token模板开销，冻结时再次实数检查，不做截断。
`qa_1` 默认前10道可回答题均来自 Normans 一篇文章，跨长度按**文章一个簇**统计，
其来源 bootstrap CI 不可估计。其余四项是受控合成诊断，不是人审金标准。

BoolQ 原始 GCS 地址本次返回403，使用 Google 官方 HF parquet（版本写入 manifest）。
答案必须整体为 Yes/No，可带末尾标点；解释、前缀或无法解析的输出计0。
这是自定义生成式控制协议，不能与原论文分类器成绩直接比较。

## 获取、冻结与准备

在 `experiments/ms_swift` 中运行。原生模型资源须先按[训练说明](../README.md)和
[Gemma环境说明](../../../docs/ms-swift-gemma4.md)准备并核验；LongBench 原始文件来自现有 publicbench 获取流程。
新机器没有本地忽略数据时，必须重新获取，不能假设这些路径已存在。

```powershell
# 单独的获取环境；Windows 优先使用已验证的系统 CPython 3.12。
uv venv --python <python312.exe> .cache/public_eval_env
uv pip install --python .cache/public_eval_env/Scripts/python.exe -r requirements-eval.txt
.cache/public_eval_env/Scripts/python.exe -m pilot_eval.acquire acquire --cache .cache/new_public_eval
.cache/public_eval_env/Scripts/python.exe -m pilot_eval.acquire generate `
  --cache .cache/new_public_eval --model-assets .cache/model_assets/gemma4-e2b --generation-tag v1

uv run python -m pilot_eval.freeze `
  --cache .cache/new_public_eval --longbench-cache ../../backend/publicbench/.cache `
  --model-assets .cache/model_assets/gemma4-e2b --training-sources <reviewed-sources.jsonl> `
  --output .cache/new_public_eval/frozen_v1

uv run python -m gemma_eval prepare --kind chat `
  --input .cache/new_public_eval/frozen_v1/records.jsonl `
  --model-assets .cache/model_assets/gemma4-e2b --output .cache/new_public_eval/base_plan `
  --context-length 16384 --quantization bf16
```

`eval_manifest.json` 绑定完整输入、标签、选择顺序、官方来源、分词器和评分代码。
`confirmation_reserve_ids.jsonl` 记录未参与 pilot 的 LongBench/BoolQ 来源组；它还不是正式测试集。
来源去重检查精确规范化全文和至少300字符的完整段落/来源块，保留整个关联组。
这是有界重合检查，不能声称消除了语义近重复或未知预训练污染。

## 评分

真实云端运行前，`gemma_eval verify-chat` 必须在固定 SWIFT 环境中逐题验证模板；
实际运行也强制执行该核验。Base、short、long 从同一个冻结输入生成新计划。
下载模型输出后，在单独评分环境安装项目本身与 `requirements-eval.txt`，然后：

```powershell
python -m pilot_eval.scoring --records <frozen-records.jsonl> `
  --predictions base=<base-predictions.jsonl> `
  --predictions short=<short-predictions.jsonl> `
  --predictions long=<long-predictions.jsonl> `
  --longbench-metrics <official-LongBench-metrics.py> `
  --longbench-metrics-sha256 e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323 `
  --ruler-metrics <official-RULER-scripts/eval/synthetic/constants.py> `
  --ruler-metrics-sha256 6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f `
  --output <new-score-directory>
```

评分拒绝重复ID、未知ID、旧 Base completion 协议和改变的输入。未尝试、失败、空输出和
超预算题保留原分母并计0；失败率单独报告。按任务与长度展示成绩，并计算配对来源簇
bootstrap。LongBench primary 四任务的等权复合指标明确标注混合了3项F1和1项ROUGE，
同时保留全部分项；不生成跨 benchmark 总分。单来源、共享跨任务簇等不能可靠估计的
区间返回 null。预实验不据此宣布正式实验成功。

原文、答案、个体预测和模型文件保持 Git 忽略；只有聚合结果、图表、代码和来源哈希可版本化。
