# 复核本轮报告

本页命令只在CPU上评分和画图，不会创建runtime或启动训练。需要私有Drive的最终轻量归档、原始40对审核包、有效人工批准收据、原生tokenizer资产及锁定的官方评分源码。Git不包含这些逐题输入，不能把空的忽略目录视为数据丢失。

在仓库根目录运行，以下目录参数替换为实际路径；输出目录必须全新。原始计划中的Colab绝对路径保持不变，报告通过最终核验收据的相对路径读取解压后的结果。

```bash
uv run --project experiments/ms_swift \
  --with jieba==0.42.1 --with rouge==1.0.1 \
  --with fuzzywuzzy==0.18.0 --with matplotlib==3.10.7 \
  python experiments/analysis/report_issue13_sft.py \
  --artifact-root FINAL_ARCHIVE_UNPACKED \
  --output NEW_REPORT_DIRECTORY \
  --packet ORIGINAL_REVIEW_V2 \
  --approval-receipt HUMAN_APPROVAL_JSON \
  --model-assets GEMMA_TOKENIZER_ASSETS \
  --metric-root OFFICIAL_METRIC_SOURCE_DIRECTORY \
  --suite-summary ORIGINAL_FROZEN_SUMMARY_JSON
```

脚本首先核验人工准入、审核包文件指纹、冻结250题SHA、三组逐题结果及最终GPU核验记录，再调用原评分器。官方LongBench和RULER文件SHA及本轮评分器SHA保持与冻结清单一致。每个长度桶、任务汇总和失败分母分别核对；`all_lengths`不会重复累加为500题。

`visual/scoring/summary.json`是可版本管理的聚合评分，逐题`record_scores.jsonl`继续忽略。探索性配对来源簇bootstrap固定2000次、seed17，不覆盖训练seed方差或多重比较。BoolQ事后格式诊断仍使用原严格yes/no解析，不改变评分规则。

`visualization_manifest.json`绑定实际报告源码、通用绘图器、官方评分实现、批准收据与最终证据；输出PNG/SVG/HTML使用SHA核验。绘图时间戳、字体与运行环境可能影响图片字节，评分复核应比较固定指标和分母，不能只比较图片SHA。

实际runtime使用不可变v2包加独立哈希恢复驱动。本轮公开输入、生成配置与adapter未因工程重试而变化。后续v5入口复用v4源码包，增加显式60秒Drive flush上限；尚未从头执行这份新版notebook，不能追记成本轮的原始执行版本。
