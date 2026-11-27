# 配对 SFT：best 保存、tmux 和 Drive

本入口对应 Issue #13。40 组固定数据（32 train / 8 dev、12 份来源）已由维护者实际审阅并批准内部 SFT；以 `16966dc52c4c…` 审核包及 `993b2c8bdddcec…` 批准收据绑定。旧 pending 文件不代表当前准入状态，内部批准不包括公开 HF 数据集发布。

## 训练与选模

Gemma 4 E2B 固定 Base revision，BF16 LoRA、seed17、batch1/accum8、两轮共8次 optimizer updates。两臂问题、答案、来源、顺序相同，输入长度不同；仍需在实际 SWIFT 模板下逐臂验证标签和长度。

每2步评估8条 dev，以有限 `eval_loss` 严格下降为新 best，同分保留较早者。公共250题不参与选模。总完成步数、best step、总曝光和 best 前缀曝光分别记录，不能把“总预算相同”解释为“选中模型曝光相同”。

固定 [Transformers 5.12.1 Trainer](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/trainer.py) 的 `save_strategy=best` 在新 best 时调用完整保存，再触发 `on_save`；固定 [SWIFT mixin](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/trainers/mixin.py) 委托这一过程。[SWIFT SFT pipeline](https://github.com/modelscope/ms-swift/blob/ad818a2cdfafaf4305ac40c4acd892a211978bc2/swift/pipelines/train/sft.py) 最终写训练状态。这里只使用公开 callback，不改安装包或训练张量。

关闭 SWIFT 默认 `add_version`，确保实际路径与 launch plan 一致；关闭异步 flash checkpoint 和 `save_only_model`。原生 `save_total_limit=2` 是短暂安全窗口：新 best 保存后检查 adapter、optimizer、scheduler、RNG、trainer state，再复制 Drive、逐文件读回校验、最后写 COMPLETE 标记。成功后才删除本次旧 best，稳定每臂/seed保留一份。失败会留下旧 best 和本次未完成证据，立即停止继续训练；不清理历史实验或其他 run。

CPU 测试只验证策略与错误分支，实际固定栈保存/重载必须在 GPU 上通过。仅 adapter 不含完整优化器状态；当前入口尚未提供经过实测的跨 VM 完整训练恢复。只有 best 可能使中断后回退重算。

## 启动与观察

用 `paired_bundle.py` 显式打包批准记录、审核凭据、来源、冻结公开题、模型元数据和 allowlist 源码；不包含 `.env`、基座权重、venv 或旧计划。`sft_notebook.py` 生成绑定 ZIP SHA256 的 notebook。选择 A100 后逐格执行，Drive 授权由人完成。Notebook 在本地盘运行训练，不把优化器活跃写入放在 Drive 上。

如已用 CLI 创建实际 A100 并在用户终端挂载 Drive，可安全解包后运行：

```bash
python /content/ariadne_paired_RUN/experiments/colab/tmux_runner.py \
  --action start --run-id RUN \
  --drive-run /content/drive/MyDrive/Ariadne/experiments/RUN \
  --allow-license PSF-2.0 --allow-license CC-BY-4.0 \
  --allow-license 'CC-BY-SA-4.0 AND MIT-0' --allow-license BSD-3-Clause

tmux attach-session -t ariadne-sft-RUN
# 已在 tmux 中时：
tmux switch-client -t ariadne-sft-RUN
# 非交互抓取：
tmux capture-pane -p -t ariadne-sft-RUN:train -S -100
```

RUN 换成实际生成的小写 run_id；只允许全新目录，重复启动被拒绝。三个窗口分别显示阶段输出、已完成更新/loss/dev/best、GPU 状态。`Ctrl-b n` 切换窗口；详细 stdout 位于本地 `evidence/*.log`。CLI 0.7.5 的 ssh 不支持附加 CMD；Agent 用 `colab exec -s SESSION -f local_capture.py`，脚本中通过 `subprocess.run([...])` 调用上述命令。用户可通过自己的交互终端进入会话。

supervisor 设置6小时总墙钟上限；每臂完整250题评测最多90分钟，外层额外60秒用于清理，仍受原六小时总预算截断。真实adapter解码慢于Base，原45分钟不足；新入口只提高工程时限，题目与逐题生成预算不变。终止信号进入清理流程，只结束自己启动的进程组。tmux 抵御终端断开，不能防止 Colab runtime 被回收；无人工保活承诺。

## 持久化与结束

Drive `MyDrive/Ariadne/experiments/RUN/` 下保存：

- `inputs/`、`bundle_manifest.json`：本次输入快照与逐文件指纹；源码还保存在 Git PR 及私有 Drive 的发布 ZIP。
- `best/{short,long}-seed17/`：完整 best、OWNER 和 COMPLETE 文件哈希；不覆盖其他 run。
- `progress/{short,long}/`：每次 optimizer step 同步的轻量审计、计划、指标与选中曝光。进程退出后的 launch receipt 收在阶段归档中。
- `artifacts/<sha256>.zip`：每阶段不可变轻量归档，包含失败、分母、日志、计划和评测结果，排除 checkpoint 目录，避免重复保存模型。
- `latest_archive.json`、`supervisor_exit.json`：最新完整归档哈希与最终退出记录。观察器等待后者，再核验归档并 flush Drive。

FUSE 的读回哈希是完整性检查，不保证云端已经持久落盘。结束时 `drive.flush_and_unmount()` 成功后才停止本轮 runtime；同步失败保留 VM 取回本地文件。CLI 操作也须先下载/核验再 `colab stop -s`，只操作本轮明确创建的 session。不会向 Colab 传递 GitHub 写入密钥，不自动公开 HF 数据。

## 统一评测与图表

所有 arm 从完全相同的250条冻结公开题生成计划：LongBench60、RULER150、BoolQ40。保留 timeout/失败/未尝试的题及原分母。按 `pilot_eval/README.md` 的固定官方评分代码分别评分，不生成跨任务“总准确率”。

评测 CLI 对完整执行但有空输出或失败题的运行也会返回1。配对运行器保留这个非零进程收据，另行检查模型确已加载、250条完整顺序、逐题文件和子进程摘要哈希及失败分母；这些检查通过，才允许继续后续阶段。空输出和失败题仍按原协议计分，不重跑或删除。模型加载失败、超时和不完整输出不会因此获准继续。显式 Hub 缓存目录使用 `HF_HOME/hub`，与模型下载位置一致。

`analysis/plot_paired_pilot.py` 接收资源目录、两臂 plan、三组评测 summary 与绑定评分 summary，生成自包含 HTML、PNG/SVG、输入哈希和完整/选中曝光表。缺少真实 GPU 或评分结果时标记未运行，不画零准确率。此规模仅为 seed17 预实验；单来源簇、32K缺少质量题等限制沿用冻结套件说明。

官方评分导出同时包含互斥长度分桶与 `all_lengths` 任务汇总。报告只累加长度分桶计算总分母，另验汇总与分桶的条数、失败状态一致；不会把同一250题重复统计为500题。
