# Repository guidance

## Local assessment data

Downloaded corpora, generated tasks and individual predictions are intentionally
excluded from Git. An ignored file remains readable by local agents and scripts;
an empty `git status` or ordinary `rg --files` result does not establish that the
data are absent. Use the known data path directly, or restrict `rg --no-ignore`
to the relevant data directory.

Common local paths:

- `backend/longdoc_sources/.cache/public_v1/`: original public documents, raw
  sources and the reference tokenizer.
- `backend/longdoc_sources/.cache/pipeline_v1/`: processed corpus and data audits.
- Other `backend/longdoc_sources/.cache/` subdirectories: diagnostics, generated
  QA, AI review candidates and source-linked compilations.
- `backend/publicbench/.cache/`: downloaded public benchmarks and local imports.
- `backend/evals/results/longdoc_*/` and `publicbench_*/`: run summaries plus local
  record-level data and predictions. Their small manifests/summaries/reports and
  source locks are retained as versionable provenance.
- `backend/evals/results/*.negative-candidates.jsonl` and new timestamped P0 v2
  case JSONs: local evidence and review candidates.

For example, `rg --files --no-ignore backend/longdoc_sources/.cache -g '*.jsonl'`
finds local corpora without changing the ignore rules. Read a specific JSONL file
with a script or `Get-Content -LiteralPath <path> -TotalCount 1`. Limit inspection
to task-relevant dataset paths; do not automatically scan `.env`, credentials or
the application user's library/settings in `backend/data/`.

Do not remove local data to make Git clean, and do not force-add ignored data as
part of routine code work. Existing small offline fixtures in
`backend/evals/data/` remain versioned. A fresh clone or remote agent does not
receive ignored data: recreate it using the documented acquisition commands,
pinned source locks and hashes; report unavailable sources rather than assuming
that local cached data are present remotely. See `docs/local-data-access.md`.

## Evidence and labels

Keep controlled quote-location diagnostics, AI-generated/reviewed candidates and
independently human-verified gold distinct. Missing or blank review fields are
not passed checks. Quote/schema validity and engineering tests do not establish
reasoning accuracy or a pretraining-data improvement. Preserve failed calls and
their denominators when reporting model evaluations.

## 协作流程（Issue → PR）

本仓库所有改动都经过 issue 和 PR：先在 issue 里写清楚要做什么，再在独立分支上实现，提交 PR 由维护者审阅、合并。Agent 不直接修改维护者本机的工作区，也不直接提交到 `main`。

### 开始之前

- 每项工作对应一个 issue。没有 issue 时，先按 `.github/ISSUE_TEMPLATE/` 的格式起草，写明背景、工作范围、验收标准和不做的内容，由维护者确认后再动手。
- 从最新的 `origin/main` 拉分支，一个 issue 一个分支、一个 PR。分支命名：`<agent>/<issue 编号>-<简短英文描述>`，例如 `claude/11-collaboration-workflow`、`codex/12-domain-map`。人工分支可用 `feat/...`、`fix/...`。
- 动手前先读 issue 里列出的参考入口，以及相关的 `docs/` 说明（例如 `docs/agent-observability.md`、`docs/rich-content.md`）。
- 你只能看到已经推送到 GitHub 的内容。维护者本机未推送的改动、被 Git 忽略的数据（见上文「Local assessment data」）和 `.env` 都不在你的工作副本里；需要时在 issue 或 PR 里说明，不要猜测它们的内容。

### 实现

- 只做 issue 范围内的事。顺手发现的其他问题记下来，在 PR 里列出或另开 issue，不要混进同一个 PR。
- 保持现有代码风格：中文注释和界面文案，说明「为什么」而不是复述代码。
- 不要改写与任务无关的文件，不要批量格式化，不要改动换行符（仓库内文本文件统一为 LF，见 `.gitattributes`）。
- 改了用户可见的行为，同步更新 `README.md` 的功能表、路线图或对应的 `docs/` 说明。
- 新功能和缺陷修复都要有测试：后端在 `backend/tests/`，前端回归测试在 `frontend/tests/`（`node --test`）。

### 开 PR 之前

在仓库根目录运行，全部通过才能提交 PR：

```bash
npm test                     # 后端 pytest + 前端 tsc -b + 前端回归测试
(cd frontend && npx vite build)
(cd backend && uv run python -m evals --offline --source fixture --no-save)   # 改了测评代码时
```

这些检查与 CI（`.github/workflows/ci.yml`）一致。改了界面时，用离线演示模式（`npm run demo`，或后端设置 `LLM_FAKE=1`）实际操作一遍，截图放进 PR。

### 提交与 PR

- 提交信息用中文，`feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:` 开头，标题一行说清楚做了什么，正文分条列出要点。由 Agent 创建的提交，在末尾附上该 Agent 要求的署名行。
- PR 按 `.github/pull_request_template.md` 填写：
  - 关联 issue：`Closes #编号`（全部完成）或 `Refs #编号`（部分完成）；
  - 改了什么、为什么这样改；
  - 验证：分成「已验证」（跑过哪些测试、在什么环境下手动操作过）和「需要在本地确认」（桌面版打包、Windows 表现、依赖本机数据或真实模型的部分）；
  - 边界：没有做什么、已知限制。
- 给 PR 打 label，Release 说明按 label 分类（见 `.github/release.yml`）：新功能 `enhancement`，缺陷 `bug`，文档 `documentation`，破坏性变更 `breaking-change`。
- 不要合并自己的 PR，合并由维护者决定。

### 审阅之后

- 审阅意见在同一个分支上继续提交修改并推送，不要另开 PR。逐条回复处理结果；不同意的意见说明理由。
- `main` 有新提交、与 PR 冲突时，由提交 PR 的一方变基到最新的 `origin/main` 并解决冲突，再请维护者审阅。只在自己的 PR 分支上强制推送（`--force-with-lease`），绝不强制推送 `main`。
