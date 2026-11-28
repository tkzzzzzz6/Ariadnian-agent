# Gemma SFT best-only 与 tmux Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline; one fresh whole-branch review at the end.

**Goal:** 在 Issue #13 独立分支承接既有实验工具，实现可观察、有限时、Drive 持久化的配对 SFT；有 A100 时实际执行，无 A100 时明确保留阻塞证据和可运行 notebook。

**Architecture:** 使用固定 HF/SWIFT 原生 best 保存，项目 callback 仅检查有限 dev loss、核验完整保存、发布 Drive 并清理本次旧 best。独立 stdlib tmux supervisor 调用既有准入、资源、训练和评测入口，冻结 250 公共输入不参与选模。

**Tech Stack:** Python stdlib/pytest，独立 uv 实验环境；Linux tmux；固定 Python 3.12 / Torch 2.8 / SWIFT 4.6.0 / Transformers 5.12.1。

**Spec:** [Issue #13](https://github.com/tkzzzzzz6/Ariadnian-agent/issues/13)，外部 handoff.json、issue.md。

## Global Constraints

- 只改本 worktree；不改维护者 dirty main、旧准入/计划/数据。内部训练批准已获，无需重复确认；数据不公开。
- 不绕过 launch/template/数据审核校验；训练依赖留在 experiments。
- dev 每 2 步，有限 eval_loss 严格改善才保存，同分留早者；稳定每 arm/seed 一份完整 best。临时保留旧、新两代直到校验发布成功。
- 总训练 8 步与 best step/选中模型曝光分别报告；公共 250 不选模。
- tmux 只抵御终端断开。Drive 读回哈希不代表云端完全落盘，结束时 flush；不宣称硬回收零丢失或完整恢复已实测。
- A100 有界重试，不换 GPU；交互 Drive 挂载由用户终端完成。无资源时完成代码、测试、PR 和 notebook，保留未执行状态。

## Review Focus

准入链完整、同步失败保旧、清理范围、原生 best 行为与 mock 的边界、终止/超时不孤留训练、密钥不进入产物、未测结果不伪报。

## Task 1: 承接和验证源码

已核验外部 ZIP 159 文件并在 bd76d49 上应用 integration.patch。跑独立实验与 colab CPU 基线，检查版本清单无密钥/缓存。保存基线来源记录。

## Task 2: best 策略与完整持久化（RED → GREEN）

1. 在 `experiments/colab/tests/test_best.py` 添加改善/同分/非有限/缺文件/复制失败/损坏/仅清理所属历史代测试；在现有配对测试加入 recipe 与选中曝光断言，先运行观察失败。
2. 扩展 `cpt/schema.py` 的受限原生参数；新增 `experiments/colab/best_checkpoint.py`，集成 `training_audit.py` 和 `paired_job.py`。固定原生 best、min eval_loss、load_best、临时 limit2，callback 发布后仅留1。
3. 每代 manifest 最后写，读回逐文件哈希；未完成代不当 best，发布失败保留旧代。训练终结验证原生 best step 与发布一致；分别记录总与选中前缀曝光。
4. 针对完整 suite 运行 GREEN；实际固定栈行为待 A100 后验证，不用 CPU mock 冒充。

## Task 3: tmux 与新运行包（RED → GREEN）

1. 新增 `test_tmux.py`，测试重复启动、argv 引用、退出码、状态、超时以及所管会话范围，先 RED。
2. 新增 stdlib `tmux_runner.py`：命名 owned session，训练日志/状态/GPU 观察窗口；有限 supervisor，现有 command 进程组超时。提供非交互 capture 与用户 attach。
3. 更新 `paired_bundle.py` allowlist，新增可执行 SFT notebook 生成器：Drive 挂载、输入快照、资源门槛、两臂、冻结公共测评、每阶段归档，失败保留。
4. 新 bundle、fresh plans、hash receipt；私有 Drive 上传且 metadata readback。更新 docs/README，保留旧 notebook。

## Task 4: 实际执行、交付与审阅

1. CPU 实验 suite、npm test、vite build、离线 fixture（如涉及 eval）通过。
2. 有界 A100 申请；成功后核验硬件、请求真实挂载，运行 setup/probe/base/两臂SFT/250评测，保存实际失败分母、可视化、完整 best，校验归档并释放仅本轮 runtime。分配失败则记录真实 HTTP 状态，不声称训练完成。
3. 按现有报告工具记录阶段、训练曲线、总/best曝光、评测比较；无实际模型结果则只呈现状态。
4. 一次 fresh 全分支审阅，修复重要发现并重跑相关验证；中文 Conventional Commit + Agent 署名；创建 PR 并立即链接，维护者合并。

## Completion Contract

实现与工程验证可独立完成；实际 GPU 训练/选优/重载/测评只有真实运行通过才算完成。外部资源或用户挂载阻塞必须明确记录，不以计划替代结果。
