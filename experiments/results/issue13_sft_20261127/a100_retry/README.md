# Colab A100：真实配对 SFT 与公开测评

状态：**short / long SFT 与 Base / short / long 各250题公开测评全部完成；完整best、Drive读回、flush和本轮A100释放均已验证。**

本轮标识 `sft_20261127_093523_cli`，A100-SXM4-40GB；Python3.12 / Torch2.8.0 / SWIFT4.6.0 / Transformers5.12.1，固定基座revision和全部批准输入指纹。40对、单seed17仅为预实验，不能据此宣称正式、稳定的长文能力提升。

## 结果入口

- [自包含可视化报告](visual/report.html)：配对长度、方向/来源、资源、训练/dev/best、公开任务分桶及探索性来源簇区间；[评分汇总](visual/scoring/summary.json)保留原250题分母。
- [模型说明](MODEL_CARD.md)、[复现命令](REPRODUCE.md)、[方法与限制](method.md)。
- [最终GPU核验](final_validation_summary.json)、[本地归档核验](local_archive_validation.json)、[最终发布与完整云端读回](final_delivery_manifest.json)、[释放记录](lifecycle_summary.json)。
- 私有Drive[本轮目录](https://drive.google.com/drive/folders/1smSbS8yeJm_TYYGrGcuG2kWIadhydyJ3)：输入、完整best、逐题预测/日志、轻量归档与最终报告。[最新v2报告包](https://drive.google.com/file/d/1mqARtbX0YhXzA2snjopU_a0o1EcI3_EA/view)包含自包含HTML、独立PNG/SVG、模型卡及实际报告源码；首版 `reports/final_v1/` 与其[发布记录](report_delivery_manifest.json)保留为历史。

## 实际训练

| 已测项目 | short | long |
|---|---:|---:|
| optimizer updates | 8 | 8 |
| 实际消费序列 | 64 | 64 |
| 实际输入token | 38,092 | 371,796 |
| 实际监督token | 7,084 | 7,084 |
| best step | 8 | 8 |
| best dev loss | 1.33265424 | 1.43616521 |
| optimizer观测累计秒数 | 39.889 | 209.599 |

两臂答案标签及消费顺序哈希相同。输入量约9.76倍，计算量不匹配；时间仅累计optimizer观测，不含完整加载、dev、发布和测评。每臂仅保留一份完整 `checkpoint-8`，adapter/optimizer/scheduler/RNG/trainer state逐文件核验及有限值CUDA重载通过。总完成步数与best步数分开记录。

## 固定公开250题

LongBench60、RULER150、BoolQ40，输入/生成/评分协议冻结，不按公开分数选模。Base244条正常输出+6条空输出，short249+1，long250+0；空输出继续计零。各任务指标独立解释，不计算跨基准总准确率。

| 任务 | 指标 | n | Base | short | long | long−short / pp |
|---|---|---:|---:|---:|---:|---:|
| yes_no | boolq_yes_no | 40 | 0.00 | 0.00 | 0.00 | +0.00 |
| 2wikimqa | qa_f1_score | 10 | 0.71 | 2.35 | 3.48 | +1.13 |
| gov_report | rouge_score | 5 | 30.16 | 30.16 | 30.79 | +0.63 |
| hotpotqa | qa_f1_score | 5 | 0.00 | 0.00 | 0.00 | +0.00 |
| multi_news | rouge_score | 10 | 31.95 | 34.62 | 34.69 | +0.07 |
| multifieldqa_en | qa_f1_score | 5 | 7.23 | 28.17 | 29.25 | +1.08 |
| multifieldqa_zh | qa_f1_zh_score | 10 | 3.52 | 15.70 | 10.94 | -4.76 |
| narrativeqa | qa_f1_score | 5 | 3.06 | 3.06 | 3.06 | +0.00 |
| qasper | qa_f1_score | 10 | 2.89 | 4.47 | 10.75 | +6.28 |
| cwe | ruler_all | 30 | 4.67 | 13.67 | 8.00 | -5.67 |
| niah_multikey_1 | ruler_all | 30 | 0.00 | 90.00 | 100.00 | +10.00 |
| niah_multivalue | ruler_all | 30 | 5.00 | 82.50 | 90.00 | +7.50 |
| qa_1 | ruler_any | 30 | 0.00 | 3.33 | 13.33 | +10.00 |
| vt | ruler_all | 30 | 2.67 | 0.00 | 4.00 | +4.00 |

表中差值使用已显示的任务分数；完整未舍入配对差值和探索性95%来源簇区间见评分汇总。long对多键/多值检索、Qasper等任务有所改善，中文问答和词频任务下降；这些结果有升有降。大多数小样本区间跨零或接触零，未经多重比较校正，也不覆盖训练seed方差。RULER qa_1只有一个独立来源簇，不能估计来源泛化区间。

BoolQ三臂均为0：Base有6条空输出、34条格式不符，short为1+39，long为0+40；没有有效格式的回答。保留原严格整段yes/no协议，不用事后宽松解析抬分，不能把这个0解释为知识全部答错。

## 工程失败、归档与发布

初次缓存故障250条 `model_load_failed` 独立保留。首次short在2700秒时限中保留158条正常输出、1条中断、91条未尝试，清理成功；后续只将每臂工程时限提高至5400秒，原六小时总预算不变，不重复训练。新旧请求指纹一致，先前已完成的158条文本/状态及生成参数哈希全部一致。本页比较明确绑定最终完整运行，不按答案拼接或挑选重试。

最终轻量归档31,325,754字节，SHA `2536070819c8ab2efade6f0fa5e8178cebead4570d2b8cca3e5abc55fac92d54`，模型单独存储。CLI控制连接中断及大文件下载停滞保留为传输故障；15个分块逐块及重组哈希、ZIP CRC、路径、三臂ID/摘要均核验通过。

首版报告及实际源码共26文件，经挂载Drive逐文件读回。flush成功后本次A100已停止，其他CPU会话未操作。提交前统一新报告源码为LF、去掉新生成SVG的无意义行尾空格，再从原证据重算v2报告；三组分任务成绩、分母与配对区间逐项完全一致，原版未改写。

最终v2报告ZIP 2,907,808字节，SHA `9692bcf4540894596f01067df1baf51a66d87b654b1bbb494cbed46e088e7cab`。Drive连接器恢复后已直接上传、确认私有状态并完整下载核验SHA。flush/停止事后收据亦已保存为Drive[最终封存清单](https://drive.google.com/file/d/1fKW2PLoxwXCToYLB9R3ljhTFHkEXatp0/view)，本地和Git保留同一聚合记录。

后续 [v5 notebook](https://colab.research.google.com/drive/1JSl8uq1SQfUZbLUt7SnYHL-dWaswAoOE)位于[私有发布目录](https://drive.google.com/drive/folders/1suFe0Kxi5yWIZqBq4HT5ZrD55v4igSAx)，复用v4源码ZIP；[v5发布清单](release_v5_manifest.json)和[v4源码清单](release_v4_manifest.json)分别保留。本轮实际执行v2包加独立哈希恢复驱动，未追记为v5从头执行。

本机 `npm test`（后端280、实验338 passed/1 skipped/29 subtests、前端74及入口/tsc）、Vite构建、offline fixture、Colab43项和可视化9项测试通过；Windows后端一次既有分页测试失败重跑通过。三臂真实评分与渲染已执行，PNG/SVG和内嵌表实际检查通过。

原文、逐题记录、checkpoint、密钥不进Git。内部SFT批准不包括公开HF数据集发布。本轮没有CPT、消费级8GB量化推理或跨VM完整训练恢复实测。
