# Gemma 4 E2B / MS-SWIFT 准备记录（2026-11-05）

本目录记录真实执行的数据审计和原生分词预览，**没有 CPT 训练或模型能力测评结果**。
目录整理后本报告位于 `experiments/results/`。所有 JSON 文件保留迁移前字节、统计及来源声明；
下列 393 项测试也是迁移前整套后端测试的历史记录。新目录复核另见[迁移记录](../cpt_layout_20261105/README.md)。
未下载模型权重、安装 SWIFT 训练栈或启动付费服务器；没有将待审材料批量改成已审核。

## 已执行的准备

- 固定 Base：`google/gemma-4-E2B`，revision `433813a669639fd7eaa83155828e9000ed357d53`。
- 下载并核验 5 个惰性 tokenizer/config 文件，合计约 32 MB。原生窗口为 131,072 token，本轮序列上限为 8,192。
- 审计 350 篇语料：train/dev/test 为 281/41/28；获准训练文档为 **0**。
- 按原划分导出 ordered/shuffled 两臂预览，28 篇 test 全部不进入训练或验证导出。
- 两篇含过长原子块的 train 文档整篇隔离，两臂使用同一隔离集合；原始文件保持不变。

| 预览 | 源文档 | 序列数 | Gemma 原生正文 token | 最长正文 |
|---|---:|---:|---:|---:|
| train ordered | 279 | 441 | 2,664,268 | 8,190 |
| train shuffled | 279 | 441 | 2,664,268 | 8,189 |
| dev ordered | 41 | 56 | 324,917 | 8,182 |
| dev shuffled | 41 | 56 | 324,917 | 8,187 |

正文 token 不含模板特殊 token；导出保守预留 2 个位置，真正执行前仍必须逐条核验固定 SWIFT 的 PT 模板、labels 和实际 loss token。
本轮两臂正文总数恰好相同，尚未证明 loss token 身份或训练暴露量匹配，`strict_equal_token_budget=false`。
100 步 LoRA recipe 只是可编辑的试跑默认值，不能据此声称严格公平或 CPT 有收益。

独立复核逐条从原始来源 span 重建全部预览文本，重新计算每条原生 token 数，检查片段完整、无重叠、两臂来源相同及测试隔离。
两份计划均已默认空跑复核，`executed=false`、`training_ready=false`；没有训练子进程。

最终工程验证：`cd backend && uv run pytest -q`，**393 passed**；根目录 CLI 帮助和 Git 忽略规则验证通过。
其中 SWIFT 模板合同测试使用 mock，不代表真实 SWIFT/GPU 集成测试已通过。

## 文件与复现

[preparation_summary.json](preparation_summary.json) 保存维度统计、序列计数、源/产物 hash 和复核结论；
[audit_manifest.json](audit_manifest.json) 保存准入概览；[native_model.lock.json](native_model.lock.json) 保存固定模型元数据来源与文件 hash。
本目录不包含正文、逐文审核队列、逐序列 lineage 或权重。

[dependency_resolution_report.json](dependency_resolution_report.json) 记录补充依赖诊断：
官方 PyPI 的同版本替代来源可解析 136 包，固定 Git 来源抓取未完成。没有生成冒充固定 Git 环境的正式 lock，
此诊断不代表依赖已安装或 GPU 训练已验证。

这些本地材料已从 `backend/cpt/.cache/` 移至 `experiments/ms_swift/.cache/`，继续被 Git 忽略，
仍可由本机 agent 按路径读取。旧 JSON 中的路径记录不修改，新的 `preview_v2` 和计划单独生成。
入口为根目录 `npm run train:cpt -- --help`；详细步骤见[MS-SWIFT 使用说明](../../../docs/ms-swift-gemma4.md)。

后续需完成来源许可和内容审核、公开基准上下文去污染，在选定 GPU 上验证训练环境与短程训练，
再分别测未训练/CPT-only/SFT-only/CPT+SFT，以及 8GB/16GB 本地量化部署。
已有 DeepSeek API 的公开基准结果不充当本模型训练前基线。
