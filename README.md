# Ariadne · 研究报告助手

<p align="center"><img src="./frontend/public/ariadne-512.png" width="128" alt="Ariadne"/></p>

<p align="center">
  <a href="https://github.com/tkzzzzzz6/Ariadnian-Agent/actions/workflows/ci.yml"><img src="https://github.com/tkzzzzzz6/Ariadnian-Agent/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/tkzzzzzz6/Ariadnian-Agent/releases"><img src="https://img.shields.io/github/v/release/tkzzzzzz6/Ariadnian-Agent?include_prereleases&label=%E4%B8%8B%E8%BD%BD" alt="Release"/></a>
  <img src="https://img.shields.io/github/last-commit/tkzzzzzz6/Ariadnian-Agent?style=flat&label=%E6%9C%80%E8%BF%91%E6%8F%90%E4%BA%A4" alt="Last commit"/>
  <img src="https://img.shields.io/badge/LangGraph-1f9c9c?style=flat&logo=langgraph&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
</p>

**在线体验：[agent.tanke.asia](https://agent.tanke.asia/)** · [通过 Cloudflare Tunnel 开放公网访问](docs/cloudflare-tunnel.md)

Ariadne 是一个帮你**快速走进一个陌生研究领域**的研究报告写作助手。它不替你写，而是把「读懂一个领域」拆成三步，让理解留在你自己脑子里：

1. **AI 检索，搭出骨架**：输入感兴趣的主题，综述 Agent 检索论文、提炼要点，给出一份带出处的大纲。
2. **你来审阅，形成理解**：Agent 在「选哪些文献」「大纲怎么定」两处停下来等你确认。逐篇取舍、改写大纲的过程，就是你理清这个领域脉络的过程。
3. **边写边纠错，加深记忆**：动手写作时有续写、改写和 `@` 插入引用帮忙；页边批注会核查每一处引用是否真的被原文支撑、指出缺少依据的论断。写完一稿，还可以让审稿 Agent 挑逻辑和论据上的毛病，改完再审一轮。被指出过的错误，往往记得最牢。

> 名字来自阿里阿德涅之线 —— 在文献的迷宫里牵一根线，每一处引用都能追溯到原文。

![主界面：左侧目录、正文、右侧页边批注](docs/screenshots/editor.png)

## 功能

| 功能 | 怎么用 | 实现要点 |
|---|---|---|
| **行内续写建议** | 停顿约 0.7 秒出现灰色建议；`Tab` 接受，`Ctrl+→` 接受一个词，`Esc` 忽略 | 防抖 + AbortController 取消过期请求；走快模型，不经过 Agent |
| **选中改写** | 选中文字 → 润色 / 学术化 / 改写 / 扩写 / 精简 / 译英 / 译中 / 自定义 | 流式输出，修订模式 diff，逐次接受或放弃；改写时保留引用标记 |
| **插入引用** | 正文输入 `@` 搜索文献库；或文献库条目上的 ❞ 按钮 | 引用是独立的 citation 节点，绑定文献库 id |
| **阅读记录与使用情况** | 文献库点「阅读记录」；原文面板可「记录读至第 N 页」或修改记录 | 手动确认未读 / 阅读中 / 已读，保存读至页码和总页数；打开 PDF 不自动视为已读。显示库内划线/笔记数、当前文档每篇文献的引用节点次数（含草稿），引用至少 2 次且未确认已读时提醒。未知状态与明确未读分开显示；详见 [首期范围与验收](docs/issue-7-reading-progress.md) |
| **为一句话找引用** | 选中文字，在浮动菜单里点「找引用」 | 混合检索 + LLM 判断「支持 / 部分支持 / 不相关」，并给出原文片段 |
| **页边批注** | 默认开启（顶栏 ✎ 图标切换）；写完一段、光标离开后自动检查。平时每条只占一行，点一下或光标进入那句话才展开；`Alt+↓ / Alt+↑` 逐条查看；底部状态条按类型筛选（审稿 / 引用问题 / 缺引用 / 精简） | 逐段调用 `/api/assist/annotate`：核查已有引用、找出缺引用的论断并给出候选文献、至多一条精简建议；结果按段落内容缓存。呈现参考 Google Docs 评论栏：同一句话的批注合并成一组、按重要程度排序，展开的那组与原句对齐、其余上下让开，离原句太远的收进「还有 N 条」 |
| **批量处理引用** | 底部状态条点「批量处理引用」→ 预览候选、调整勾选 → 批量插入或忽略 | 只处理尚无引用的完整句子；每句默认选一个判为「支持」的候选，「部分支持」需手动勾选，已有引用问题单独审阅。推荐与复核使用同一证据选择路径，相同模型配置、论断与证据复用判断；插入前核对原句与文献，跳过已改动原句、已有引用和重复建议，整批插入可一次撤销 |
| **审阅进度与精简确认** | 底部显示检查覆盖情况，批量面板标明检查是否完成 | 仅修改引用不会重新挑选精简建议；接受精简后结束该段本轮自动精简，主动「检查全文」或后续文字编辑可开启新一轮。忽略记录绑定具体建议，补入引用不会让同一建议重新出现；精简操作保护已有引用节点 |
| **原文阅读** | 页边批注、找引用结果、正文引用的悬浮提示、文献库条目上点「看原文」；检索结果和 Agent 候选文献上点「预览 PDF」 | pdf.js 渲染（文本层可选中复制）；后端用 PyMuPDF 在原文里定位依据句（整句找不到就按 6 词一段分段匹配），返回归一化坐标，前端跳到对应页并高亮；未入库的论文只临时缓存预览，加入文献库时直接复用 |
| **分屏对照** | 顶栏「写作 ⇄ 分屏对照」 | 左边正文、右边原文，多篇论文用标签页切换（拖动分隔条调整比例，双击恢复 1:1）；原文带目录（PDF 书签，没有就按页），选中原文可以引用这句、问 AI、记笔记；正文里点引用的「看原文」直接在右边打开；页边放不下时，批注浮在句子下方 |
| **划线与笔记** | 在「原文」里选中一段文字：引用这句 / 插入原句 / 划线 / 记笔记；正文输入 `@` 能搜到自己的笔记 | 选区坐标归一化后存进 SQLite（`notes` 表，随文献删除）；引用节点带页码，显示为 `[1, p. 3]` / `(Lewis et al., 2020, p. 3)`，导出为 `\cite[p.~3]{key}` 和 pandoc 的 `[@key, p. 3]` |
| **笔记纳入综述** | 综述输入框上方可开关；确认大纲时查看「本轮提供的笔记」 | 只筛选选中文献的相关条目，摘录与用户观点分开输入；保存带来源和页码的快照。来源修改/删除时提示，可刷新后重新归纳，或明确沿用旧快照；删除原笔记不会清除已有任务副本。详见 [范围与验收](docs/issue-5-review-notes.md) |
| **整篇引用核查** | 助手抽屉「找引用」→ 核查全文引用 | 逐条检查被引文献原文能否支撑那句话，能找出凭空挂上的引用 |
| **参考文献自动生成** | 无需操作 | 顺序编码（GB/T 7714 简化）/ 作者-年份（APA）一键切换 |
| **文献库** | 在线检索 Semantic Scholar / arXiv / OpenAlex、上传 PDF、下载开放获取全文 | PyMuPDF 解析，按章节切块（记录章节和页码），遇到参考文献就停止 |
| **文献问答** | 助手抽屉「文献问答」；打开时根据你的论文和文献库推荐几个可以问的问题（可「换一批」） | 回答带 [n] 来源，可一键插入正文并自动转换成引用；推荐问题由快模型结合论文题目、章节、文献库和你的笔记生成候选，再逐个试检索，优先保留文献库里找得到依据的，并标出能回答它的文献；结果按输入缓存 |
| **综述 Agent** | 助手抽屉「Agent」→ 综述，填主题 → 勾选文献 → 修改/确认大纲 → 写入文档 | LangGraph + `interrupt()` 人工确认 + SQLite checkpointer（刷新页面、重启服务都能恢复） |
| **分节写作 Agent** | 大纲确认后点「逐节写作」，或在「Agent」首页选分节写作；勾选要写的章节，选「逐节确认 / 连续写完 / 并行写」，可选「总编统稿」 | 草稿流式写进正文对应标题下（紫色草稿标记 + 光标），你可以直接在正文里改，最后保留或删除；并行模式用 LangGraph `Send` 给每节派一个子 Agent 同时写，总编 Agent 通读后只改衔接、重复和前后不一致，每处修改在正文里逐处呈现；每节单独检索（指定文献优先 + 你的笔记），引用只能来自本节证据，写完逐段核查引用、能改的先改，剩下的成为页边批注 |
| **审稿 Agent** | 「Agent」首页选审稿 → 意见挂在正文旁边 → 逐条标「已修改 / 不改（附理由）」→ 再审一轮 | 审稿人只挑逻辑和论据问题（推理跳跃、论据不足、结论过度、结构），每条意见摘录原句定位到正文；第 2 轮起先逐条复查上一轮（解决 / 部分 / 未解决），再只补充新的主要问题，最多 N 轮保证收敛；论据类问题附上文献库里可能用得上的文献 |
| **Agent 交互** | 三个 Agent 共用一个启动页；顶部「历史」可回到这篇文档以前的任何任务 | 运行中可以「停止」，之后从 LangGraph 检查点「继续」（停在请求半路时会把你最后一次的选择补发出去）；Agent 停下来等你、而你没在看面板时，图标栏和页签出红点并弹出提示 |
| **图片、表格与公式** | 顶部「插入」或段落开头输入 `/`；选中内容后出现编辑操作 | PNG/JPEG 上传、粘贴截图、图题与宽度；规则表格与 Excel/TSV 粘贴；LaTeX 行内/独立公式和实时预览；保存、撤销与重做；Word 保留图片、原生表格和可编辑公式，LaTeX/Markdown 导出包含图片与参考文献的 ZIP。[使用与验收说明](docs/rich-content.md) |
| **导出** | 顶栏「导出」 | **Word**（pandoc + citeproc，参考文献按 GB/T 7714-2015 顺序编码 / 著者-出版年或 APA 排版，中文版式模板：宋体 / Times New Roman 小四、1.5 倍行距、首行缩进）；LaTeX + refs.bib（可直接传 Overleaf）、Markdown（pandoc 引用语法）、BibTeX |
| **Agent 运行统计** | Agent 面板底部的状态条：步数 · 调用次数 · 输出速度（tok/s），总 token · 缓存命中率，上下文占用（圆环）；悬停看明细 | 每次模型调用按任务归属（contextvar，子 Agent 并行写作的调用也算在同一个任务里），随 SSE 实时推送；运行中按收到的片段估算实时速度，结束后用服务返回的 token 数校正；上下文占用 = 单次调用最大输入 / 模型窗口（LiteLLM 模型信息）；停下、继续都接着累计，存进任务记录，历史列表也显示每个任务的 token 总量 |
| **Agent 运行详情** | 底部状态条或任务历史中的「运行详情」 | 按次回看节点、工具和模型调用，区分并行、等待与停止恢复；展开查看耗时、用量和异常，可复制脱敏诊断摘要。详细输入输出及模型返回的推理信息默认不采集，可在记录设置中开启；[使用与验收说明](docs/agent-observability.md) |
| **用量统计** | 设置 →「用量统计」 | 每次调用记录耗时、首 token、token 数和缓存命中；按 DeepSeek / Kimi 官网价格折算人民币（含 DeepSeek 高峰分时价和缓存价），其他模型可在页面里填单价；展示每日趋势、各档 P50/P95 延迟和各模型费用 |

**防止编造引用**是贯穿各个功能的原则：大纲里的 bibkey 会和文献库核对，库里没有的直接丢弃；改写结果里的 `[@key]` 只有在文献库中存在才会变回引用；引用核查会对照被引文献的原文逐条检查。

## 界面

> 以下截图来自系统实际运行画面，在**离线演示模式**（`npm run demo`，`LLM_FAKE=1`）下拍摄：界面和操作流程是真实的，文献元数据是真实论文，但 AI 给出的判断、改写和大纲是演示数据（卡片上标有「离线演示」）。接入模型后内容由模型生成。

**页边批注**（见页首截图）：停笔后逐段检查，缺引用的论断、可能撑不起论断的引用、可以精简的句子直接贴在段落旁边，一键插入候选文献或接受改写。

<table>
<tr>
<td width="50%"><b>@ 插入引用</b>：正文里输入 <code>@</code> 搜索文献库<br/><img src="docs/screenshots/cite-at.png" alt="@ 插入引用"/></td>
<td width="50%"><b>选中改写</b>：修订模式显示增删，接受或放弃<br/><img src="docs/screenshots/rewrite.png" alt="选中改写"/></td>
</tr>
</table>

**综述 Agent**：输入主题后检索文献，在「勾选文献」和「确认大纲」两处停下来等你确认；打开抽屉时左侧目录缩成短横线，鼠标移上去即可预览章节。

<table>
<tr>
<td width="50%"><img src="docs/screenshots/agent-papers.png" alt="综述 Agent：勾选文献"/></td>
<td width="50%"><img src="docs/screenshots/agent-outline.png" alt="综述 Agent：确认大纲，短横线目录悬停预览"/></td>
</tr>
<tr>
<td><b>找引用</b>：为一句论断检索文献，给出「支持 / 部分支持」判断和原文片段<br/><img src="docs/screenshots/find-citations.png" alt="找引用"/></td>
<td><b>设置</b>：在网页里选模型服务商、填 API Key、测试连接，无需改 .env<br/><img src="docs/screenshots/settings.png" alt="设置"/></td>
</tr>
</table>

**原文阅读**：点页边批注上的「看原文」，右侧抽屉打开被引论文的 PDF，跳到依据所在页，并把与正文那句话最相关的原文标黄；在找引用结果、Agent 候选文献上点「预览 PDF」，可以先在弹窗里看原文和最相关的几段，再决定是否引用。（截图中的 PDF 为本地生成的演示文件）

<table>
<tr>
<td width="50%"><b>抽屉阅读</b>：从页边批注跳到原文依据<br/><img src="docs/screenshots/reader.png" alt="原文阅读：抽屉里打开 PDF 并高亮依据句"/></td>
<td width="50%"><b>检索预览</b>：先看原文再决定引不引用<br/><img src="docs/screenshots/pdf-preview.png" alt="检索预览：在线看 PDF 和最相关的原文段落"/></td>
</tr>
</table>

**划线与笔记**：在原文里选中一段文字，可以直接「引用这句」（自动带页码，如 `[1, p. 3]`）、把原句加引号插入正文，或者划线、写下自己的理解。笔记会标在原文上，点一下就能修改或引用；写作时输入 `@` 也能搜到它们，插入的引用自动带页码。读的时候记下的想法，写的时候随手就能用上。

![划线与笔记](docs/screenshots/pdf-notes.png)

**分屏对照**：写「相关工作」这类要边读边写的章节时，顶栏切到「分屏对照」：左边写，右边读，多篇论文用标签页切换。正文里点引用的「看原文」会直接在右边打开并跳到依据；在原文里选中一句话，可以引用到光标处（带页码）、问 AI 或记笔记。正文变窄后页边批注收成正文里的标记，点一下，批注浮在句子下方。

![分屏对照：左边正文，右边原文；页边放不下时批注浮在句子下方](docs/screenshots/split.png)

**分节写作 Agent**：先勾选要写哪些章节（已有正文的默认不选），再选写法：「逐节确认」一节一节写，每节写完停下来由你保留、按意见重写或删掉；「连续写完」一口气写完；「并行写」给每节派一个子 Agent 同时写，最快。Agent 边写边把草稿直接写进正文对应的标题下（紫色标记，末尾有光标，可开关「跟随」自动滚到正在写的位置），每节先检索证据（文献库片段 + 你在原文上记的笔记），引用只能从这些证据里挑；写完自动核查每条引用，撑不起论断的先自己改，自查后的定稿替换初稿，剩下的问题变成页边批注。勾选「总编统稿」后，总编 Agent 会通读新写的各节，修改衔接、重复和前后不一致的地方，每处修改都在正文里重新「打」一遍并高亮。草稿可以直接在正文里改，满意就保留，不要的删掉。

![分节写作：三个子 Agent 并行写完，总编 Agent 正在正文里修改第 2 节的开头](docs/screenshots/writer.png)

**审稿 Agent**：模拟审稿人通读全文，只挑逻辑和论据上的问题，每条意见都挂在正文对应的句子旁边（主要问题红线、次要问题黄线，整节问题挂在标题上），右侧面板给出总评、三项打分和意见列表。你在正文里修改，逐条标「已修改」，或者「不改」并写下理由；点「再审一轮」后，审稿人先复查上一轮的每条意见（理由成立也算解决），再看有没有新的主要问题。默认最多 3 轮。

![审稿：意见挂在正文旁边，右侧是总评和逐条处理状态](docs/screenshots/critique.png)

**用量统计**：每次模型调用都会记账。按官网价格折算成人民币，并展示每日费用、各档延迟（P50 一般情况 / P95 最慢 5%）和各模型的花费；没有内置价格的模型，可以直接在页面上填写单价。（截图中的调用数据为模拟数据）

![用量统计](docs/screenshots/usage.png)

正文始终在视野中心；目录悬浮在左侧页边，打开抽屉或窗口变窄时缩成一列短横线（1/2/3 级标题长短不同，当前章节蓝色高亮，鼠标移上去立即预览标题和正文开头，点击跳转）；AI 助手平时收起，左侧图标栏或 `Ctrl+J` 打开。

## 架构

<p align="center"><img src="docs/architecture.svg" width="100%" alt="Ariadne 系统架构（动态）"/></p>

<details>
<summary>文字版（模块与接口）</summary>

```
frontend/  React 19 + Vite + MUI 7 + TipTap 3
  ├─ 编辑器：Citation 节点 / @ 引用建议 / GhostText 续写 / 选区浮动菜单 + diff / 悬浮目录
  ├─ 页边批注：段落 hash 缓存 → /api/assist/annotate → 正文装饰（ReviewMarks）+ 对齐的卡片
  └─ 助手抽屉：Agent（综述 / 分节写作 / 审稿）· 找引用 · 文献问答 · 文献库 · 我的论文
        │   /api（开发时由 Vite 代理）
        ▼
backend/   FastAPI
  ├─ LLM 接入层 llm.py：LiteLLM，四档模型 complete / write / judge / agent，各自的超时、重试、备用模型，调用日志
  ├─ 快速路径（直接调用 LLM，低延迟）：/api/assist/complete · edit(SSE) · cite · verify · annotate · ask(SSE)
  ├─ Agent 路径：/api/agent/start · /{id}/resume（SSE） → LangGraph
  │     plan_queries → search_papers → select_papers⏸ → ingest → synthesize → make_outline → review_outline⏸ → finish
  │                                                                              ▲──────── regenerate ────────┘
  └─ 共享：SQLite（文献、切块、文档）· BM25(jieba) + 可选向量的 RRF 混合检索 · 跨语言查询扩展
```

</details>

为什么把两条路径分开：续写、改写这些操作要求几百毫秒内响应，直接调用模型；综述这类长任务要中途暂停、等用户确认、断点续跑，这部分交给 LangGraph。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/)（Windows 可用 `winget install astral-sh.uv`）和 Node.js ≥ 20。所有命令都在仓库根目录执行：

```bash
npm run setup     # 第一次：安装前后端依赖，并生成 backend/.env
npm run dev       # 开发：后端 :8000 + 前端 :5173 一起启动，热更新，自动打开浏览器，Ctrl+C 一起退出
```

| 命令 | 作用 |
|---|---|
| `npm run setup` | `uv sync` + `npm install`，没有 `backend/.env` 时从 `.env.example` 生成 |
| `npm run dev` | 开发模式，前后端同时启动 |
| `npm run demo` | 同上，但设置 `LLM_FAKE=1`：不调用模型、不需要 API key，适合演示和调界面 |
| `npm start` | 生产模式：前端有改动才重新构建，由后端单端口提供服务 http://localhost:8000 |
| `npm test` | 后端离线测试 + 前端类型检查与回归测试（引用批处理、阅读统计等） |
| `npm run eval` | 测评引用判断、假引用检出、检索、缺引用检测、原文定位和续写延迟，见下文「测评」 |
| `npm run desktop` | 打包 Electron 桌面版（见下文「桌面版」）；加 `-- --dev` 则不打包、直接用桌面窗口运行 |

依赖没装时 `dev` / `demo` / `start` 会自动先执行一次 `setup`。设置环境变量 `NO_OPEN=1` 可以不自动打开浏览器，`PORT` 可以修改 `npm start` 的端口。

## 桌面版

不想开命令行和浏览器的话，可以打包成桌面程序：双击打开就是一个独立窗口，关掉窗口就完全退出。技术路线和 Claude、Codex 的桌面端一样，用 **Electron 外壳 + 独立的后端进程**。

![Windows 桌面版](docs/screenshots/desktop.png)

<p align="center"><sub>Windows 桌面版实际运行：综述 Agent 生成的大纲在右侧，页边批注标出了一处撑不起论断的引用和一处缺引用的句子</sub></p>

```bash
npm run desktop          # 构建前端 → PyInstaller 打包 Python 后端 → electron-builder 生成安装包
npm run desktop -- --dev # 不打包，直接用 Electron 窗口运行当前源码（调试用）
```

- **下载**：每次发布版本时，CI 会自动打包并上传到 [Releases](https://github.com/tkzzzzzz6/Ariadnian-Agent/releases)，直接下载安装即可。
- **产物**（在 `desktop/dist/`）：Windows 为 `Ariadne-Setup-<版本>.exe` 安装程序和免安装 zip；macOS 为 dmg；Linux 为 AppImage。只能打包当前系统的版本，跨平台由 CI 分别打包。
- **数据位置**：文献库、论文、设置保存在 `%APPDATA%\Ariadne`（macOS 为 `~/Library/Application Support/Ariadne`），卸载或升级都不会删除。日志在其中的 `logs/`（`electron.log`、`server.log`）。它和 `npm run dev` 用的 `backend/data` 是分开的，想沿用已有的文献库，把 `backend/data` 里的文件复制过去即可。
- **工作方式**：Electron 主进程（`desktop/electron/main.js`）先显示启动页，分配一个空闲端口，启动打包好的后端 `ariadne-server`（`desktop/server.py`，只监听 `127.0.0.1`），等健康检查通过后加载页面。同一时间只运行一个实例；关闭窗口时结束后端，后端也会在 Electron 意外退出后自行退出。外部链接用系统浏览器打开，导出文件弹出「另存为」。
- **为什么是 Electron**：编辑器重度依赖可编辑区域、选区和输入法，Electron 在所有平台上都是同一个 Chromium 内核，表现一致；Tauri 和 pywebview 用的是系统 WebView，在 macOS（WebKit）和 Linux（WebKitGTK）上需要分别适配。
- **体积**：安装包约 250 MB，其中 Python 后端（LiteLLM、PyMuPDF、numpy 等）约占四分之三。
- **下载加速**：本地打包时 Electron 相关的二进制默认从 npmmirror 镜像下载；设置 `CI=1` 时改用官方源。
- **首次运行提示**：安装包没有代码签名，Windows 可能提示「Windows 已保护你的电脑」，点「更多信息 → 仍要运行」即可。

## Docker 部署（网页版）

想放到服务器上给同学、课题组用，可以用 Docker 部署。镜像里是构建好的前端加 FastAPI 后端，一个容器、一个端口（8000）；文献库、论文 PDF 和设置都存在数据卷里，升级、重建容器不会丢。

```bash
cp deploy/ariadne.env.example deploy/ariadne.env   # 填访问密码 ARIA_PASSWORD 和模型 Key
docker compose up -d --build                         # 从源码构建并启动，访问 http://服务器IP:8000
```

有域名的话推荐开 HTTPS：在 `deploy/ariadne.env` 里填 `DOMAIN`，域名解析到服务器并放行 80 / 443 端口，然后

```bash
ARIA_BIND=127.0.0.1 docker compose --profile https up -d --build   # 加一个 Caddy，自动申请并续期证书；8000 端口只对本机开放
```

不想在服务器上构建，也可以直接用 CI 发布的镜像：`docker compose pull && docker compose up -d`（镜像在 `ghcr.io/tkzzzzzz6/ariadnian-agent`，打过版本标签后才有 `latest`；ghcr.io 上新发布的镜像默认私有，要在 GitHub 的 Packages 页面把它改为 Public 才能匿名拉取）。国内网络连不上 Docker Hub 时见下面的「国内网络」。

**国内网络**：构建要从 Docker Hub 拉 `node`、`python` 基础镜像，再装 npm 和 PyPI 依赖，国内经常超时（典型报错是 `failed to fetch anonymous token ... auth.docker.io`）。在仓库根目录新建 `.env`（已被 git 忽略），写入镜像源后照常 `docker compose up -d --build`：

```bash
# 基础镜像走镜像站（下面以 DaoCloud 为例，也可以换成其他可用的镜像站）
NODE_IMAGE=m.daocloud.io/docker.io/library/node:22-slim
PYTHON_IMAGE=m.daocloud.io/docker.io/library/python:3.12-slim
CADDY_IMAGE=m.daocloud.io/docker.io/library/caddy:2
# npm / PyPI 依赖走国内源
NPM_REGISTRY=https://registry.npmmirror.com
PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

也可以在 Docker Desktop → Settings → Docker Engine 里配置 `registry-mirrors`，对所有镜像生效。公共镜像站的可用性会变化，拉取失败时换一个试试。

**更好的 PDF 解析（可选）**：默认的 PyMuPDF 不用额外部署；遇到复杂排版、公式多、中文或扫描版的 PDF，可以加一个解析服务，在 `deploy/ariadne.env` 里打开对应的 `PDF_PARSER`（服务没启动时会自动退回 PyMuPDF）：

```bash
docker compose --profile grobid up -d --build   # GROBID：英文论文结构解析，CPU 即可，约 4GB 内存；PDF_PARSER=grobid、GROBID_URL=http://grobid:8070
docker compose --profile mineru up -d --build   # MinerU：中文 / 扫描件 OCR / 复杂版式，需要 NVIDIA GPU；PDF_PARSER=mineru、MINERU_URL=http://mineru:8000
```

MinerU 官方没有发布现成镜像，需要先按 [MinerU 的说明](https://github.com/opendatalab/MinerU) 构建 `mineru:latest`；国内拉 GROBID 镜像可以用 `GROBID_IMAGE=m.daocloud.io/docker.io/grobid/grobid:0.9.1-crf`。换了解析器后，已入库的文献在文献库里点「用当前解析器重新解析」即可。

**部署给别人用时的安全设置**（都在 `deploy/ariadne.env` 里，本地和桌面版默认不开启，行为不变）：

| 设置 | 作用 |
|---|---|
| `ARIA_PASSWORD` | 访问密码。打开网页时浏览器弹出登录框，用户名随意（`ARIA_USERNAME` 可限定）。**镜像默认要求设置**，没设就拒绝启动，避免忘了设密码直接暴露在公网 |
| `ARIA_LOCK_SETTINGS=1` | 锁定「设置」：其他人只能查看，不能修改模型和密钥，也不能拿服务器上的 Key 去测试别的地址。模型配置写在 `deploy/ariadne.env` 里 |
| `ARIA_MAX_UPLOAD_MB` | 上传文件大小上限，默认 50 MB；下载论文 PDF 也受这个限制 |
| `ARIA_ALLOW_PRIVATE_URLS=1` | 允许从内网 / 本机地址下载 PDF。默认禁止（含重定向），防止有人通过修改文献的 PDF 链接探测服务器内网 |

其他细节：容器以普通用户（uid 10001）运行；默认限制 2 GB 内存、2 个 CPU；带健康检查（`/api/health`，不需要密码）。数据卷是 Docker 的命名卷 `ariadne_ariadne-data`，备份可以用 `docker run --rm -v ariadne_ariadne-data:/data -v $PWD:/backup busybox tar czf /backup/ariadne-data.tgz /data`。

> 目前所有人共用**同一个文献库和论文列表**，适合自己或信任的小团队使用；也就是说，有密码的人都能看到、修改彼此的论文。多人账号和数据隔离在路线图里。Basic 认证的密码在 HTTP 下是明文传输的，公网部署请务必开 HTTPS。

## 公网访问（Cloudflare Tunnel）

当前体验站使用名为 `agent` 的 Cloudflare Tunnel，将 **`agent.tanke.asia` → `http://127.0.0.1:8000`**。应用仍在运行 Ariadne 的电脑上，访问者通过固定的 HTTPS 域名进入，无需处于同一局域网。

已有隧道及配置时，日常演示只需打开两个 PowerShell 终端：

```powershell
# 终端 1：启动应用
cd D:\code\Ariadnian-agent
npm start
```

```powershell
# 终端 2：启动现有 agent 隧道
cd D:\code\Ariadnian-agent
.\tools\cloudflared.exe tunnel run agent
```

首次设置见 [Cloudflare Tunnel 公网访问指南](docs/cloudflare-tunnel.md)：包括下载工具、授权和创建隧道、绑定 DNS、`config.yml` 示例、访问密码、验收及排错。其他人自行部署时，需要使用自己的域名、隧道 UUID 和凭据路径。

本机启动方式在 `backend/.env` 中配置 `ARIA_PASSWORD` 和 `ARIA_LOCK_SETTINGS=1`，修改后重启应用；Docker 部署则使用 `deploy/ariadne.env`。电脑、应用和隧道都要保持运行，关机、休眠或停止任一服务都会中断访问。访问入口提供 HTTPS，不代表每位用户有独立的数据空间，也不保证中国大陆所有网络的访问质量。

Agent、改写等功能依赖 SSE 流式输出，因此使用正式的命名隧道；[Cloudflare 官方说明](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations)中的临时 Quick Tunnel 不支持 SSE，不适合作为完整体验入口。

## CI/CD

GitHub Actions 配置在 `.github/workflows/`：

| 工作流 | 触发 | 做什么 |
|---|---|---|
| `ci.yml` | 每次 push / PR | 后端单元测试（Ubuntu + Windows）、测评冒烟（离线）、前端类型检查与构建 |
| `release.yml` | 推送 `v*` 标签 | 在 Windows、macOS、Linux 上分别打包桌面版，自动创建 Release 并上传安装包；也可以手动运行，只打包不发布 |
| `docker.yml` | push 到 main、推送 `v*` 标签、改动镜像相关文件的 PR | 构建网页版 Docker 镜像，发布到 ghcr.io（标签版本号 + `latest`，main 分支为 `edge`）；PR 只构建并启动容器做冒烟测试 |
| `eval.yml` | 手动 | 用真实模型跑测评，结果写进运行摘要并保存为 Artifact；需要在仓库 Secrets 里添加 `LLM_API_KEY` |

发布新版本：

```bash
git tag v0.2.0
git push origin v0.2.0
```

几分钟后 Releases 页面会出现 `Ariadne-Setup-0.2.0.exe`、`Ariadne-0.2.0-mac-arm64.dmg`、`Ariadne-0.2.0-linux-x86_64.AppImage`。CI 上安装 Python 依赖时用官方 PyPI（仓库默认的清华镜像只在本地生效）。

Release 说明由 GitHub 自动生成，分类规则写在 `.github/release.yml`。它按 **PR label** 分章节，不解析 commit 前缀 —— 想让更新说明分出「新功能 / 缺陷修复 / 文档 / 构建」，给 PR 打上 `feature`、`bug`、`documentation`、`ci` 等 label 即可，没打 label 的会落到「其他变更」。

## 配置要点

**推荐在网页里配置**：右上角 ⚙ 打开「设置」，可以配置模型（选服务商预设、填 API Key、主模型和快模型，高级里还能按四档分别配置）、文献检索（Semantic Scholar / OpenAlex 的 key、默认来源、缓存天数）、向量检索，并查看各档模型的用量统计。每一项都有「测试」按钮，保存后立即生效，不用重启。

- 网页里保存的设置写在本机 `backend/data/settings.json`，**优先于 `.env`**；网页里没设置的项继续读 `.env`，所以原来的 `.env` 配置照样可用。
- API Key 只写不读：接口只返回是否已设置和末 4 位，测试失败时报错信息里的 key 也会被遮住。
- 第一次打开、还没配置模型时，会自动弹出设置页。

也可以继续用 `.env`（适合部署或脚本化），字段说明见 `backend/.env.example`：

- **模型（LiteLLM）**：模型名写成 `provider/model`，例如 `deepseek/deepseek-chat`、`openai/qwen-plus`（OpenAI 兼容服务，需同时填 `LLM_API_BASE`）、`ollama_chat/qwen2.5:7b`，Claude、Gemini 等原生接口也能用。按任务分四档：

  | 档位 | 环境变量 | 用途 | 默认超时 |
  |---|---|---|---|
  | complete | `MODEL_COMPLETE` | 行内续写，延迟最敏感 | 8 s，不重试 |
  | write | `MODEL_WRITE` | 选中改写、文献问答 | 90 s |
  | judge | `MODEL_JUDGE` | 引用判断、中文→英文检索关键词 | 30 s |
  | agent | `MODEL_AGENT` | 综述与大纲（JSON 输出） | 180 s |

  没单独配置的档位使用 `MODEL_ID` / `FAST_MODEL_ID`。每档还可以配置 `_API_KEY`、`_API_BASE`、`_TIMEOUT`、`_FALLBACKS`（备用模型）。旧版的 `MODEL_PROVIDER` + `OPENAI_API_KEY` + `OPENAI_BASE_URL` 配置仍然有效。每次调用的耗时、首 token 延迟、token 数和费用都会记录下来，在右上角「模型设置」里查看（接口为 `/api/llm/stats`）。
- **中文写作 + 英文文献**：系统会把中文句子转换成英文关键词，再和原句一起检索。条件允许的话，建议再开启多语言向量模型（例如 `EMBEDDING_PROVIDER=ollama`、`EMBEDDING_MODEL=bge-m3`）。
- **学术检索与限流**：Semantic Scholar 不配 key 时与所有匿名用户共享配额，高峰期经常返回 429，强烈建议免费申请 `SEMANTIC_SCHOLAR_API_KEY`。检索层会自动处理：同一来源的请求限速（S2 每秒 1 次、arXiv 3 秒 1 次），429/5xx 按 Retry-After 或指数退避重试 3 次，仍失败就冷却 60 秒并改用 OpenAlex 补位（可选 `OPENALEX_API_KEY`），结果缓存 7 天（`SEARCH_CACHE_DAYS`）。

## 测评

`npm run eval` 用网页里配置的模型跑一遍测评，结果写进 [`backend/evals/results/`](backend/evals/results/README.md)，换模型或改 prompt 之后可以直接对比。测的都是系统里真实在用的函数：

| 测评 | 数据 | 关键指标 |
|---|---|---|
| 引用判断 | SciFact 209 对论断—证据（英文），以及中文论断 + 英文原文 | 准确率、**假支持率**（撑不起的证据被判成“支持”） |
| 假引用检出 | 把正确引用换成随机文献 / 主题相近的难例文献 | 检出率、正确引用误报率 |
| 检索 | 1,489 段语料，英文 / 中文 / 中文 + 跨语言关键词扩展 | Recall@5、MRR |
| 缺引用检测 | 20 段中文研究报告、72 句逐句标注 | Precision / Recall / F1 |
| 原文定位 | 用 SciFact 证据造出 15 份双栏论文 PDF（证据混在相似的干扰句里） | **定位准确率**（高亮的是不是证据）、相关段落 Recall@3 |
| 延迟与花费 | 20 次行内续写 + 本次全部模型调用 | P50 / P95、token、花费 |

当前结果（全量数据；离线基线不调用模型，只用关键词规则 + 纯 BM25）：

| 运行 | 引用判断准确率 | 假支持率 ↓ | 假引用检出（随机 / 难例） | 正确引用误报 ↓ | 检索 R@5（英文 / 中文 / 中文+扩展） | 缺引用 F1 |
|---|---|---|---|---|---|---|
| 离线基线 | 56.9% | 50.7% | 85.5% / 0.0% | 4.3% | 84.7% / 26.7% / — | 34.8% |
| deepseek-flash（推理模型） | 67.0% | **0.0%** | 100% / **91.3%** | 8.7% | 84.7% / 26.7% / 53.3% | **83.0%** |
| deepseek-chat | **75.1%** | 5.6% | 100% / **91.3%** | 10.9% | 84.7% / 26.7% / **86.7%** | 73.3% |

- **模型判断远好于规则**：关键词规则会把一半撑不起论断的证据判成“支持”、主题相近的难例假引用一条都标不出；两个模型的难例检出都是 91%。
- **跨语言扩展效果明显**：中文论断直接检索英文文献 Recall@5 只有 26.7%，用 deepseek-chat 生成英文关键词后达到 86.7%，与英文论断（84.7%）持平。
- **两个模型各有取舍**：deepseek-chat 判断更准（中文论断 83.8%）、续写 P50 0.78 秒，但 71 条撑不起的证据里有 4 条被判成“支持”；deepseek-flash 一条都不放过，但更保守（正确引用一半出黄色提示），而且作为推理模型，续写慢一倍、关键词扩展有一半因思考耗尽额度而为空。默认推荐续写和引用判断两档用 deepseek-chat。

- **原文定位靠测评找出了问题**：「看原文」第一版只有 48.6% 的高亮落在证据上；测评暴露出切块粘句、只挑一句、段落截断三个问题，修复后定位准确率 **79.8%**（只高亮首句 17.4%），高亮精确率 82.7%（整块高亮只有 27.6%），相关段落 Recall@3 从 54.1% 提到 **84.4%**。细节见 [backend/evals/README.md](backend/evals/README.md#原文定位改进前后)。

完整对比表见 [backend/evals/results](backend/evals/results/README.md)。

各项定义、数据来源和局限见 [backend/evals/README.md](backend/evals/README.md)。

## 目录

```
backend/
  app/
    main.py            FastAPI 入口
    security.py        部署用的安全设置：访问密码、设置锁定、上传上限、PDF 下载地址检查
    pdfview.py         原文阅读：提供 PDF、在原文中定位依据句（返回高亮坐标）、找出与论断最相关的段落、预览缓存
    routers/settings.py  网页设置：读取 / 保存（settings.json 覆盖 .env）、测试模型连接与检索来源
    config.py          .env 配置（兼容旧版字段）
    llm.py             LiteLLM 接入层：四档模型、JSON 模式、备用模型、调用统计、LLM_FAKE
    db.py              SQLite 表结构
    library.py         文献元数据、去重、bibkey、BibTeX
    ingest.py          PDF 解析（双栏按先左后右的阅读顺序）与按章节切块
    retrieval.py       BM25 + 向量 RRF 混合检索
    embeddings.py      可选嵌入（Ollama / OpenAI）
    scholar.py         Semantic Scholar / arXiv / OpenAlex 检索：限速、429 重试、自动补位、结果缓存；PDF 下载
    assist.py          续写、改写、找引用、核查、问答
    agent/graph.py     LangGraph 综述→大纲工作流
    agent/writer.py    LangGraph 分节写作：选节 → 逐节 / Send 并行子 Agent 流式写 → 自查引用 → 总编统稿
    agent/critic.py    LangGraph 审稿：挑问题并定位原句 → 作者修改 / 回复 → 复查 + 再审，最多 N 轮
    routers/           API 路由（notes.py：PDF 笔记的增删改查）
  evals/             测评：数据、任务、指标，结果存在 evals/results/
  tests/
package.json         一键命令（setup / dev / demo / start / test / eval / desktop）
scripts/             启动脚本，只用 Node 内置模块，Windows / macOS / Linux 通用
Dockerfile           网页版镜像（前端构建 + FastAPI），docker-compose.yml 一键部署
deploy/              部署配置：ariadne.env.example、Caddyfile（HTTPS）
desktop/             桌面版：Electron 外壳（electron/）、后端入口 server.py、PyInstaller 配置
.github/workflows/   CI（测试、构建）、Release（三平台打包发布）、Docker 镜像、模型测评
docs/                架构图（gen_architecture.py 生成）、界面截图、布局草图（layouts/）
frontend/
  src/
    editor/            TipTap 编辑器与扩展（引用节点、@ 引用、灰色续写、选区改写、页边批注、悬浮目录）
    panels/            右侧抽屉：Agent（综述 / 分节写作 / 审稿）/ 找引用 / 文献问答 / 文献库 / 原文 / 我的论文
    components/        时间线卡片、输入框、设置页（模型 / 文献检索 / 向量检索 / 用量）
    pdf/               PDF 阅读器（pdf.js，按需渲染 + 文本层 + 高亮 + 选区菜单）、检索预览弹窗、划线与笔记
    lib/               API、引用格式化、导出、页边批注的段落缓存与定位
```

## 路线图

- [x] 分节写作 Agent：按大纲逐节生成草稿，每节单独检索，写完自动跑一遍引用核查
- [x] 审稿 Agent：模拟审稿人挑逻辑和论据问题，带有限轮次的修改循环
- [x] 测评：引用判断、假引用检出、检索、缺引用检测、延迟与花费
- [ ] 测评扩展：更多学科的中文段落、全文片段级的引用核查、端到端写作任务
- [x] GROBID / MinerU 解析（可选服务），提升复杂排版、公式和中文 PDF 的解析质量
- [x] Word 导出（pandoc，GB/T 7714 / APA 参考文献）
- [x] 图表与公式节点：图片、规则表格、行内/独立公式，Word 与 LaTeX/Markdown 资源包导出（见 [说明](docs/rich-content.md)）
- [ ] 网页版多人账号：每人独立的文献库和论文，按人统计用量
- [x] 原文阅读：抽屉阅读、检索预览、依据句定位（附测评）、划线引用与笔记
- [x] 分屏对照模式（左写右读，多篇论文标签页，原文目录，选中原文引用 / 问 AI / 记笔记）
- [x] 笔记纳入综述 Agent：有界筛选、来源快照、用户观点标注、来源变化确认及刷新重生成（[范围与验收](docs/issue-5-review-notes.md)）
- [x] Agent 运行统计：步数、调用次数、tok/s、token 总量、缓存命中率、上下文占用
- [ ] 领域地图：根据文献库的引用关系和主题聚类画一张图，标出奠基论文、主要流派和最新进展
- [x] 阅读记录首期：手动阅读状态与读至页码、划线/笔记数、当前文档引用次数；对多次引用且未确认已读的文献提示，并区分未知与未读（[范围与验收](docs/issue-7-reading-progress.md)）
- [ ] 成本控制：Agent 开始前估算 token 和费用并请你确认；检索改写、自查等粗活交给小模型
- [ ] 可观测性：每次 Agent 运行的节点耗时、token、失败节点和输入输出做成 trace 页面，出问题时能复现（调用已按任务记录 thread_id）
