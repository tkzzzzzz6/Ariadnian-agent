# Issue #13：配对 SFT 记录索引

2026-11-27 重试已取得 A100，两臂实际 SFT、完整 best 保存与重载均通过。统一公开评测仍在运行，详见[本轮实测记录](a100_retry/README.md)。以下为此前分配失败时的历史准备记录，JSON 与旧发布包保留原样。

## 首次准备与资源阻塞（历史）

状态：**工程准备完成，真实 SFT 等待 A100**。不是训练结果或长文能力提升结论。

- [Issue #13](https://github.com/tkzzzzzz6/Ariadnian-agent/issues/13)：承接既有独立实验源码，补齐 best 保存、tmux、Drive 持久化。
- 固定40组/12来源已获人工批准内部 SFT；两臂重新导出均通过，每臂32train/8dev。没有改写旧审核包；没有公开 HF 数据集。
- native best 保存、有限 dev loss、完整状态核验、同步失败保旧、总/best曝光、tmux重复启动/失败与 notebook 生成有 CPU 测试。实际固定栈保存/重载尚待 GPU，不以 mock 代替。
- 本轮申请 `ariadne-sft-a100-20261127-issue13-v2` 在分配阶段返回 `Service Unavailable`，耗时约74秒；复查服务器0个活跃会话，未开始模型任务。此前两次申请也失败，历史详情留在原记录。
- [execution_status_summary.json](execution_status_summary.json) 记录当前事实与准入指纹；[report.html](report.html) 为可视化状态记录；原始云日志、数据/审核凭据、计划和模型保持 Git 忽略。
- [prerequisite_source_manifest.json](prerequisite_source_manifest.json) 绑定迁入前159文件快照；后续实现相对它的变更通过 Git diff 追踪，不改写该快照。

实际训练用新生成的私有 Drive notebook/ZIP，版本及 SHA 见 `release_manifest.json`。该 notebook 依次做资源验证、真实模板检查、Base250题、两臂SFT、相同250题评测。只在真实资源可用且检查通过时继续；不换GPU、不做CPT。

训练完成后使用独立 `pilot_eval.scoring` 与 `analysis/plot_paired_pilot.py` 生成分任务结果和 PNG/SVG；已增加 best step/选中模型曝光列，并从全程审计日志读取 loss，避免 best 早于最终步时曲线被截短。失败和未尝试题保留在分母中。本轮暂无这些模型结果。

操作与边界见 [best + tmux 说明](../../../docs/gemma-sft-best-tmux.md)。

私有 Drive [A100 SFT notebook](https://colab.research.google.com/drive/1HICY_4YdwTInGocd7Qowd2CKpQS5Lqcs) 与[发布目录](https://drive.google.com/drive/folders/1suFe0Kxi5yWIZqBq4HT5ZrD55v4igSAx)已上传并核对大小/私有状态；尚未在 GPU 上执行。
