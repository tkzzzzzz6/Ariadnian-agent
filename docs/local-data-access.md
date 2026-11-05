# 本地评测数据与 Git 忽略规则

原始语料、处理后语料、任务和逐条预测保留在本机，不进入 Git 的普通提交候选。
`.gitignore` 只影响 Git 跟踪与部分工具的默认搜索；不会删除文件，也不会阻止本机
Agent 或 Python 通过明确路径读取。项目代码、小型离线 fixture、来源锁文件及
小型 manifest / summary / report 继续保留。

| 使用环境 | 能否直接读取本机被忽略的数据 |
|---|---|
| 本机运行、有该目录读取工具与权限的 Codex / Agent | 可以；本次已直接读取并验证 |
| 通过同一本机文件工具连接的 ChatGPT / 其他 Agent | 取决于连接提供的目录与权限，Git 忽略本身不构成限制 |
| 仅连接 GitHub 仓库的 ChatGPT / 远端 Agent | 看不到未提交的本地缓存，需要按清单下载或构建 |
| 普通云端对话，没有本地文件连接 | 不能仅凭 `D:` 路径读取，需要提供文件或可访问的数据来源 |

因此，本地保留与远端可复现应同时做好：本次加入根目录 `AGENTS.md` 指出缓存路径，
并保留来源版本、下载命令、散列和汇总；不会把 Git 忽略当作文件访问权限控制。

| 数据 | 本机位置 |
|---|---|
| 公开长文与原始下载 | `backend/longdoc_sources/.cache/public_v1/` |
| 处理后完整语料与清洗审计 | `backend/longdoc_sources/.cache/pipeline_v1/` |
| QA 候选、AI 抽审、受控题、长文编排 | `backend/longdoc_sources/.cache/` 对应子目录 |
| 公开 benchmark 下载与本地导入 | `backend/publicbench/.cache/` |
| 长文运行的逐题数据和预测 | `backend/evals/results/longdoc_*/` |
| 公开 benchmark 的逐题数据和预测 | `backend/evals/results/publicbench_*/` |
| P0 待复核负例与含逐条证据的新结果 JSON | `backend/evals/results/` 中的 `.negative-candidates.jsonl` 和带秒/微秒时间戳的 v2 JSON |

长文和公开 benchmark 结果目录中的 `.jsonl`、`.jsonl.gz`、`.parquet`、`.arrow`、
`.zip` 及数据型 `.json` 被忽略；名称明确为 manifest、summary、report、validation
或来源锁的小型 JSON 放行。缓存目录整体保持本地，其 manifest 若需要版本化，
应使用结果目录中已有的小型元数据快照。Markdown 报告与脚本不受这些数据规则影响。

从项目根目录可以直接查找和读取已有数据：

```powershell
rg --files --no-ignore backend/longdoc_sources/.cache -g '*.jsonl'
rg --files --no-ignore backend/evals/results/longdoc_20261105_v1 -g '*predictions.jsonl'
git check-ignore -v backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl

$record = Get-Content -LiteralPath backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl -TotalCount 1 | ConvertFrom-Json
$record.id
$record.blocks.Count
```

如需查新 benchmark，针对 `backend/publicbench/.cache/` 使用相同方法。不要为了
寻找评测数据而全盘绕过忽略规则扫描 `.env`、密钥文件或 `backend/data/` 用户文献库。
不要通过删除本地文件或强制加入 Git 来处理正常的忽略状态。

新克隆、GitHub 和远端 Agent 不会自动获得这些本地文件。需要根据
[长文复现说明](long-context-data.md)、对应模块 README、来源锁定清单及散列重新下载
或构建；公开 benchmark 采用其模块记录的来源、版本和许可信息。只有本机有缓存时，
本机 Agent 才能直接复用；缺失时应明确指出缺失的文件与重建步骤。

这次调整没有删除文件、取消已有文件的跟踪或上传数据。已有
`backend/evals/data/fixture.jsonl`、`zh_claims.jsonl`、`zh_citeworthy.jsonl` 和历史汇总
JSON 保持原有跟踪状态。忽略规则也不替代数据许可审核或独立人工核验。
