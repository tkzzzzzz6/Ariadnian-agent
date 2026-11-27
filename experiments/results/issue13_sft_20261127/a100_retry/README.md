# Colab A100：真实配对 SFT

状态：两臂 SFT 已完成，完整 best 保存与 CUDA 重载通过；short / long 的固定250题公开测评仍在运行。尚无本轮长文能力提升结论。

本轮运行标识 `sft_20261127_093523_cli`，硬件为 A100-SXM4-40GB。用户已实际挂载同账号 Drive。固定 Python3.12、Torch2.8.0、SWIFT4.6.0、Transformers5.12.1；模型和全部批准输入逐文件哈希通过。

| 已测项目 | short | long |
|---|---:|---:|
| optimizer updates | 8 | 8 |
| 实际消费序列 | 64 | 64 |
| 实际输入 token | 38,092 | 371,796 |
| 实际监督 token | 7,084 | 7,084 |
| best step | 8 | 8 |
| best dev loss | 1.33265424 | 1.43616521 |
| optimizer 观测累计秒数 | 39.889 | 209.599 |

两臂消费标签与顺序哈希一致。表内时间仅累计 optimizer 观测，不包括完整加载、dev、发布和测评成本。每臂只保留一份完整 `checkpoint-8`，逐文件读回与有限值 CUDA 重载均已通过。证据见 [training_stage_validation_summary.json](training_stage_validation_summary.json)。

公开 Base 评测已完成250题，244条正常输出、6条空输出；6条继续保留在原分母中。第一次缓存故障导致的250条加载失败另存为工程失败。short 首次在2700秒工程时限处超时，保留158条正常输出、1条中断与91条未尝试；已确认进程组清理和归档完整性。随后在同一A100用每臂5400秒时限重新测评short和long，原六小时总预算不变，不重复训练。新旧请求SHA相同，公开输入、模板、输出长度预算和官方评分代码不变，不按公开分数选模。[方法与限制](method.md)说明计算预算、格式跟随、单seed和来源簇等解释边界。

私有 Drive [本轮实验目录](https://drive.google.com/drive/folders/1smSbS8yeJm_TYYGrGcuG2kWIadhydyJ3)保存输入、完整best、轻量归档和进度。完整模型状态不放GitHub，轻量归档也不重复包含模型。最终还需验证归档并完成Drive flush后释放本轮runtime。

新 [v4 notebook](https://colab.research.google.com/drive/1c7OQLbUpdj0CP3pz93jLF2KEQsN0JIz8)及输入/源码ZIP已放[私有发布目录](https://drive.google.com/drive/folders/1suFe0Kxi5yWIZqBq4HT5ZrD55v4igSAx)。SHA与发布元数据见 [release_v4_manifest.json](release_v4_manifest.json)。v4修复缓存路径、失败样本编排与过短的评测工程时限。本轮实际运行使用保留的v2包加独立恢复驱动，v4尚未从头执行；v3中间发布记录保留。

数据、逐题预测、审核原文和权重保持Git忽略；本轮内部SFT批准不包含公开HF数据集发布。未执行CPT或消费级8GB显卡量化推理测试，尚未实测跨VM完整训练恢复。
