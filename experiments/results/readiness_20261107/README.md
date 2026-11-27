# Gemma 实验准备与验证：2026-11-07

代码与数据准备记录，不能作为 CPT/SFT 收益或正式模型排行榜成绩。
操作步骤见[实验操作说明](../../../docs/gemma-experiment-workflow.md)。

| 产物 | 实际内容 |
|---|---|
| [数据审核汇总](data_review_summary.json) | 350 篇审计、31 篇多领域待审包、158 对跨 split 片段候选；0 approved |
| [审核包核验](data_review_validation.json) | 43 个文件哈希、66 条页面链接、pending 凭据与原始 split 核验 |
| [首批原生 token 统计](selected_native_budget_summary.json) | 24 篇 train 为 226,127、7 篇 dev 为 45,008 个 Gemma 正文 token；不是实际训练 loss-token 预算 |
| [评测准备](evaluation_preparation_summary.json) | 精确 2K/4K/8K 资源探针；公开 42 题在 8K 下 12 ready / 30 overflow；41 篇窗口 NLL 计划 |
| [训练准备](training_preparation_summary.json) | 真实语料 4K 导出及 LoRA/QLoRA 的 4K/8K 四份计划，均保持未准入、未执行 |
| [对照准备检查](comparison_preflight_report.json) | 两组 279 篇、源字符覆盖及重复次数一致；实际训练 loss-token 暴露量尚未验证 |
| [权重校验](model_weights.lock.json) | 固定 Base revision 的 10,246,621,918 字节文件，SHA-256 与官方 LFS 一致 |
| [硬件/环境清单](resource_inventory_summary.json) | 本机 GPU、驱动、准备环境；清单不代表真实模型运行成功 |
| [首轮代码与回归验证](implementation_validation.json) | 全项目测试通过；该轮实验测试 211 项，配置打包及文档链接核验通过 |
| [最终代码与回归验证](implementation_final_validation.json) | 收尾实验 Python 229 项、入口 13 项通过；最终 wheel 包含权重锁及有时限入口 |
| [有时限运行清理验证](bounded_cleanup_validation.json) | 真实 Windows CPU 子进程树清理通过；该检查没有加载模型或 GPU |
| [真实预览训练拒绝检查](training_guard_validation.json) | 对真实 QLoRA 预览计划请求 execute，未启动子进程，正确保留未审核阻断 |
| [真实本机资源探针](local_gpu_probe_report.json) | RTX 4070 Laptop 8GB：2K 成功；4K 出现共享系统内存并长时间未完成后中断；8K 未执行 |

审核材料可在本机打开[审核索引](../../ms_swift/.cache/review_20261107_v2/index.html)。
逐篇正文、凭据、预测、权重及隔离运行环境位于被 Git 忽略的 `.cache/`，新克隆需重新获取。

## 实际 GPU 结果

隔离环境 Python 3.12.11 / Torch 2.8.0+cu128 / Transformers 5.12.1 /
bitsandbytes 0.50.0，固定 Gemma E2B Base，NF4 double quant + BF16 compute。
依赖检查和 CUDA BF16 矩阵运算通过。使用本机 RTX 4070 Laptop（8188 MiB），不是 RTX 4060 实测。

| 输入 | 结果 | 测量 |
|---|---|---|
| 2,048 token | 32-token 生成成功 | 4.04 秒；首 token 2.34 秒；之后解码 18.15 token/s；Torch 分配峰值 6.87 GiB |
| 4,096 token | 资源审查后中断 | 上一条完成后约 362 秒仍未完成；运行中 Windows 计数器观测到 dedicated 6.96 GiB / shared 1.57 GiB |
| 8,192 token | 未执行 | 中断后保留为未尝试，未伪造 OOM 或分数 |

只有一次合成输入资源探针，不能作为稳定吞吐或质量成绩。4K 不是已测得的完整延迟，
也不是 CUDA OOM。框架没有设置 CPU offload，但 WDDM 可使用系统内存；参数标记 CUDA
不能证明纯显存驻留。Windows 计数是运行中的单次观测，不是峰值，也不单独证明变慢原因。
共享内存的含义见 [Microsoft 官方说明](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)。

本轮已结束该自有进程并确认 GPU 占用释放，保留逐条输出、原进度、系统内存观测及中断凭据。
Gemma 公开基准和 NLL 尚未实际运行；相应文件仅为准备计划。

后续推荐使用 `python -m gemma_eval.bounded --plan <计划> --output <新目录>
--max-seconds 120 --execute` 限制资源试跑时间。预算包含加载和整次推理；超时保留已完成行、
预先确定的超限状态及其余未完成 ID，不把超时当作 OOM 或完整延迟。子运行原始文件保留。

当前没有获准的 CPT/SFT 数据，也没有真实训练结果。不将 mock 测试、准备成功、合成探针
或进程退出当作模型能力提升。该 NF4 路径尚未通过 8GB 纯显存长文推理验证；它不等价于
对 embedding 也进行量化的其他部署格式，后者需要独立导出、来源核验和测评。
