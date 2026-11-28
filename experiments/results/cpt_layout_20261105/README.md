# 训练目录迁移记录（2026-11-05）

CPT 代码、配置、脚本与测试已归入 `experiments/ms_swift/`，使用独立的 Python 准备环境。
配置位于 `configs/`；实验命令和测试入口位于 `scripts/`；原生分词器、导出、审核队列和计划位于被 Git 忽略的 `.cache/`。
SFT 尚未实现，没有创建假训练入口。

## 共享边界

文档 schema、读取、来源关联与划分规则由 `experiments/common/` 的 `ariadne-data-core` 提供。
它只依赖 Python 标准库。CPT 直接导入 `ariadne_data`，不依赖后端代码、应用虚拟环境或注入后端路径。
`backend/longdoc/pipeline.py` 保持原接口的重导出，其他数据与评测工具继续使用原入口。
未来数据构建的实现快照会纳入共享模块源码 hash；历史报告不重新计算。

## 数据与历史

迁移前的 21 个本地缓存文件逐个核验 SHA-256，迁移后全部一致；原始 350 篇语料保持原位置与内容。
原准备报告移动至 [cpt_20261105_v1](../cpt_20261105_v1/README.md)，4 个历史 JSON 保持原字节；README 仅更新导航与历史说明。

旧 `.cache/preview_v1` 和 `.cache/plans/*_v1` 保留其原始绝对路径，作为历史记录。
不会直接修改 manifest 或计划中的路径去绕过 hash 检查。当前路径下重新导出 `.cache/preview_v2`，再生成 `plans/*_v2`。
新预览继续为待审核材料，默认计划不执行训练。

新位置的预览已完成重新导出，8 个导出与审计文件和 V1 **逐字节一致**，包括两臂正文、来源片段、隔离与准入记录。
ordered/shuffled 的 V2 计划已重新生成并默认复核，均为 `executed=false`、`training_ready=false`。
旧计划在迁移后因原绑定路径缺失而被拒绝，没有被自动改写。

## 入口

从仓库根目录：

```bash
npm run train:cpt -- --help
npm run test:experiments
npm test
```

`train:cpt` 只使用实验项目的轻量环境，拒绝执行训练；真实 GPU 训练需在独立 Linux 训练环境中直接运行 Python。
根 `npm test` 包含后端、实验工具与前端验证，CI 也增加独立实验作业。

## 验证结果

`npm test` 全部通过：后端 **280 项**、独立实验 Python **129 项**、实验脚本 **11 项**、前端 **74 项**；前端类型检查通过。
实验套件包括从仓库外隔离导入、共享包纯标准库运行及原有训练准入与模板合同测试。
共享包 wheel 构建通过，Markdown 链接、Git 忽略规则和 CI 配置解析通过。
详细计数、文件 hash、环境与计划状态见 [layout_validation.json](layout_validation.json)。

Dockerfile 已补充轻量共享包的复制路径，Docker 忽略规则排除实验环境、缓存和报告。
本机 Docker 服务未运行，本轮未启动它或验证镜像构建；GitHub CI 配置也尚未远程执行。

详见 [MS-SWIFT 使用说明](../../../docs/ms-swift-gemma4.md)和[实验目录说明](../../README.md)。
本次是工程目录与运行边界整理，没有加载模型权重、进行 CPT/SFT 训练或产生模型能力结果。
