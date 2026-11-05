# 参与贡献

欢迎提 issue 和 PR。这份文档写给人看：怎么搭开发环境、怎么提需求、怎么提交改动。Agent（Codex、Claude Code 等）需要遵守的细则写在 [`AGENTS.md`](AGENTS.md)，流程与这里一致。

## 工作方式：Issue → PR

1. **先有 issue**：新功能用「功能需求」模板，问题用「缺陷报告」模板。写清楚背景、要做什么、验收标准和不做的内容。验收标准写得越具体，实现和审阅时越不容易跑偏。
2. **在分支上实现**：从最新的 `main` 拉分支，一个 issue 一个分支、一个 PR。命名如 `feat/12-domain-map`、`fix/15-export-crash`；Agent 的分支用 `codex/...`、`claude/...` 前缀。
3. **提交 PR**：按模板填写关联 issue、改动、验证和边界，界面改动附截图。CI 会自动跑测试和构建。
4. **审阅与修改**：审阅意见在同一个分支上继续提交；和 `main` 冲突时，由提交者变基解决。
5. **合并**：维护者合并后，在本地 `git pull` 即可运行。

不要直接提交到 `main`，也不要强制推送 `main`。

## 开发环境

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Node.js ≥ 20，命令都在仓库根目录执行：

```bash
npm run setup     # 第一次：安装前后端依赖，并生成 backend/.env
npm run dev       # 前后端一起启动，热更新
npm run demo      # 离线演示模式：不调用模型、不需要 API key，适合调界面和截图
```

更多命令（生产模式、测评、桌面版打包）见 [README 的「快速开始」](README.md#快速开始)。

## 提交前的检查

```bash
npm test                     # 后端 pytest + 前端 tsc -b + 前端回归测试
cd frontend && npx vite build
```

改了测评代码时，再跑一次离线测评冒烟：`cd backend && uv run python -m evals --offline --source fixture --no-save`。这些检查与 CI（`.github/workflows/ci.yml`）一致，CI 会在 Ubuntu 和 Windows 上跑后端测试。

新功能和缺陷修复都要带测试：后端测试在 `backend/tests/`，前端回归测试在 `frontend/tests/`。

## 代码与提交约定

- 注释和界面文案用中文，注释说明「为什么」，不复述代码。
- 只改和这次任务相关的文件，不做无关的批量格式化。仓库内文本文件统一用 LF 换行（见 `.gitattributes`）。
- 用户可见的行为变了，同步更新 `README.md`（功能表、路线图）或 `docs/` 下的说明。
- 提交信息用中文，`feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:` 开头，标题一行，正文分条列出要点。

## Label 与 Release 说明

Release 说明由 GitHub 按 PR 的 label 自动分类（规则见 `.github/release.yml`），合并前请给 PR 打上 label：

| label | 用途 |
|---|---|
| `enhancement` / `feature` | 新功能 |
| `bug` | 缺陷修复 |
| `documentation` | 文档 |
| `breaking-change` | 破坏性变更 |
| `ci` | CI / 构建 |

## 本地数据

测评用到的原文、生成数据和逐条预测只保存在本机，被 Git 忽略；仓库里只保留来源锁定文件、代码、小型样例和汇总报告。重新获取或构建数据的方法见 [`docs/local-data-access.md`](docs/local-data-access.md)。不要把这些数据或 `backend/data/`（你自己的文献库和设置）、`.env` 提交进仓库。
