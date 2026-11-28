# Eval Calibration and Paired Data v2 Implementation Plan

> **For agentic workers:** Use executing-plans inline for this authorized work. Steps use checkbox syntax.

**Goal:** 准备独立开发/确认数据、可解释格式校准及128/32配对候选审核包。
**Architecture:** 独立CPU producer复用现有来源锁、原生tokenizer和评分解析；历史协议不变。新模块负责分区和开发配置，paired_data负责候选与审核。
**Tech Stack:** Python3.12、标准库、tokenizers、pytest；真实训练环境仍独立。
**Spec:** ../specs/2026-11-28-eval-calibration-data-v2-design.md

## Global Constraints

- Issue #15，最新origin/main分支，不编辑维护者原工作区。
- 128train/32dev目标；每连接族最多4题，不复制补缺。
- 开发40、示范2、确认160的目标按实际来源容量报告。
- 原严格yes/no与旧250不变；格式门槛90%；新训练待人工准入。
- 新文件LF，旧计划/预测/审核包不可改；Torch不进backend。

## Review Focus

- 相同来源换ID不能绕过隔离；共享段落和连接族同时检查。
- 未执行/失败预测不能被判作格式校准通过。
- 来源不足须报告短缺，不能重复取样。
- 审核页面不能把新160条仍描述为旧40条或已批准。
- 旧记录批准不能因新包生成而自动扩张。

### Task 1: 独立分区与来源重建

Files (relative to `experiments/ms_swift/`): `pilot_eval/partition.py`, `pilot_eval/next_stage.py`, `tests/test_eval_partition.py`。
Consumes: 锁定原始LB/BoolQ、旧records/manifest、完整350篇训练候选来源范围及逐篇哈希、模型资产。
Produces: `partition_records(pool, regression, training_sources, quotas, seed)`及开发/示范/确认候选包、来源/分区审计。

- [x] 写来源换ID、共享段落、训练重合、回答无关选择、容量不足的失败测试。
- [x] 实现确定性分区与CLI；核验输入锁；主任务输入>=8192且输入+官方输出<=16384，不截断；最终v2来源必须匹配冻结范围。
- [x] 运行针对性测试和真实本地来源准备；记录缺额。

### Task 2: 开发校准契约与报告

Files (relative to `experiments/ms_swift/`): `pilot_eval/calibration.py`, `tests/test_eval_calibration.py`。
Consumes: 开发/示范记录，完整原生评测预测及其summary。
Produces: 仅固定Base bf16无adapter选择、各臂统一沿用；逐profile绑定真实计划/模型/量化/原始输入/收据哈希。四配置开发记录/manifest、按配置格式诊断和确定性选择结果；确认封存门。

- [x] 写空输出、失败、ID缺失、未知profile、示范泄漏、90%边界及优先级测试。
- [x] 实现 `build_calibration_records` 与 `assess_calibration`，继承原yes/no解析，不改旧scoring。
- [x] 运行CPU测试；实际GPU评测若尚未执行则保持not_run。

### Task 3: 配对候选与审核升级

Files: `paired_data/build.py`, `tests/test_paired_data.py`、本地忽略规格/候选包。

- [x] 写精确重复QA拒绝、动态审核统计和历史小包兼容测试。
- [x] 实现动态审核文案与统计，保持来源/两臂准入守卫。
- [x] 从真实来源准备128/32候选、配对/长度/来源审计及完整人工审核HTML；不足处明确报告。
- [x] 准备后续训练/校准计划，人工批准保持独立前提。

### Task 4: 交付

- [x] 聚合可视化、来源锁与复现说明，README同步入口。
- [x] 独立测试、npm test、Vite、offline fixture通过。
- [ ] 提交PR，区分已验证、需人审、未执行GPU和后续研究步骤。

## Execution evidence

- CPU本地准备：128/32候选、48来源；新增120/36待审，原40/12哈希相同。开发39+示范2+确认150，NarrativeQA来源缺额保留。
- 审阅发现的完整池传递关联与worker原文一致性问题均以RED→GREEN修复；34项针对性测试通过。
- `npm test`：backend280、experiments365（1skip、29subtests）、前端回归74通过；Vite与offline fixture通过。
- 真实GPU校准、实际SWIFT新输入、SFT及确认仍not_run；本计划的本地交付不等于实验结果完成。
