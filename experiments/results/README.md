# 实验记录

[Issue #13 当前执行状态](issue13_sft_20261127/README.md)：40组已批准内部 SFT；best/tmux/Drive 工程准备与云端分配失败分开记录，尚无本轮真实模型比较。

这里只保存可版本化的汇总、manifest、来源锁和说明。正文、逐条数据、预测、审核凭据和模型权重保持本地，
不因目录整理而加入 Git。训练缓存位于 `experiments/ms_swift/.cache/`。

| 记录 | 含义 |
|---|---|
| [cpt_20261105_v1](cpt_20261105_v1/README.md) | 首轮真实数据审计和 Gemma 原生分词预览；没有模型训练或能力测评 |
| [cpt_layout_20261105](cpt_layout_20261105/README.md) | 从后端迁出训练链路的文件保留、独立环境与新路径复核 |
| [readiness_20261107](readiness_20261107/README.md) | Gemma 评测计划、数据审核与片段重合诊断、SFT/QLoRA 和资源工具验证；模型运行状态另列 |
| [colab_sft_20261108](colab_sft_20261108/README.md) | A100 40GB 上真实完成 Gemma 4 E2B 的 20-step SFT 短样本试跑，adapter 保存、重载及下载校验通过；不是长文能力结论 |
| [long_context_design_20261108](long_context_design_20261108/README.md) | 配对长/短 SFT 与后续 CPT 结构对照方案；含1550条公开题原生预算审计，尚未执行能力实验 |

`cpt_20261105_v1` 原在 `backend/evals/results/`，迁移后所有 JSON 字节、来源 hash 与历史统计保留。
README 仅修正导航和路径说明；其中 393 项测试是迁移前整套后端测试的记录，不是迁移后的测试计数。
旧 prepared manifest 和计划的绝对路径不会原地重写，迁移后导出与计划在新目录单独生成。

工程测试、token 数或文件完整性检查不等于 CPT 有收益。后续模型实验需另存未训练/CPT-only/SFT-only/CPT+SFT
以及不同精度/部署配置的可比较评测，保留失败、中断和实际输入预算。
