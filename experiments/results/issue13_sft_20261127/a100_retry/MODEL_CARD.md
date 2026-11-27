# Gemma 4 E2B：long / short 内部 SFT 预实验

本卡描述 `sft_20261127_093523_cli` 的两份 LoRA adapter。基座为 `google/gemma-4-E2B`，固定 revision `433813a669639fd7eaa83155828e9000ed357d53`。adapter 需要同一基座，不能作为独立完整模型加载。

| 设置 | 实际值 |
|---|---|
| 环境 | Colab A100-SXM4-40GB，Python3.12，Torch2.8.0，SWIFT4.6.0，Transformers5.12.1 |
| 训练 | BF16 LoRA，rank16 / alpha32，language-model all-linear，冻结视觉模块 |
| 数据 | 每臂32 train / 8 dev，40个长短配对、12份来源；人工批准仅覆盖内部 SFT |
| 优化 | LR 1e-4，batch1，gradient accumulation8，seed17 / data_seed17 |
| 序列 | 上限8192，不packing；全部40对通过实际SWIFT模板与标签核验，无样本丢弃 |
| 预算 | 每臂2 epochs，8次optimizer更新，实际消费64条序列 |
| 选优 | 每2步按dev loss严格改善保存，同分保留早者；公开评测不参与选模 |

| 产物 | short | long |
|---|---:|---:|
| best step | 8 | 8 |
| best dev loss | 1.33265424 | 1.43616521 |
| 实际输入token | 38,092 | 371,796 |
| 实际监督token | 7,084 | 7,084 |
| adapter字节 | 105,415,064 | 105,415,064 |

两臂目标答案和消费顺序完全一致。长文输入量约9.76倍，因此本实验匹配监督与更新预算，不匹配计算量。dev条件输入不同，不能仅比较loss大小判断长文能力。

完整best在私有Drive[本轮目录](https://drive.google.com/drive/folders/1smSbS8yeJm_TYYGrGcuG2kWIadhydyJ3)的 `best/short-seed17/checkpoint-8/` 和 `best/long-seed17/checkpoint-8/`。每臂稳定保留一份，含adapter、optimizer、scheduler、RNG、trainer state及COMPLETE文件指纹。已通过完整状态核验和有限值CUDA加载；这不等于已经实测跨VM恢复完整训练。

统一公开评测沿用冻结250题、原生chat模板和greedy生成。完整分任务结果、长度桶、探索性来源簇区间和失败分母见[实测报告](visual/report.html)及[方法说明](method.md)。单seed、40对样本仅用于预实验，不将指令适配收益全部归因于长文训练，不将多个异质任务混成总准确率。

该产物未做CPT、消费级8GB显卡量化部署或公开HF数据集发布。内部数据准入没有扩大为数据公开再分发许可。
