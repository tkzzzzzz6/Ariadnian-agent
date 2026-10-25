# Issue #3 与 #9 整合验收

验收日期：2026-10-24。基线为 `1881a2e`，整合 `codex/feat/9-agent-observability`（`924db71`）与 `codex/feat/3-rich-content`（`78a6666`）。保留两个功能分支的原始提交历史。

## 合并处理

- `frontend/package.json` 同时保留引用批处理、富内容和运行观察测试入口；`scripts/test.mjs` 的测试说明同步更新。
- 联合浏览器验收发现：无表头表格以分隔行开头导出 Markdown 时，Pandoc 将其解析为普通文本。现在输出空表头行；Pandoc 会省略空表头，因此 Word 不增加虚假的表头或额外数据行。新增前端序列化及真实 Word 表格结构回归测试，覆盖中间空行和首行数据保留。
- 数据库保留图片资源表和模型调用的运行关联字段；运行追踪继续独立存储。升级沿用现有初始化补表、补列机制。

## 已执行的验证

- `npm test`：127 个后端测试、TypeScript 检查、41 个前端测试通过。
- `npm --prefix frontend run build`：通过；仍有大于 500 kB 的已有分包体积提示。
- `npm run eval -- --offline --source fixture --no-save` 对应的后端离线命令：完成。只代表离线流程可执行，不代表真实模型效果提高。
- Edge 浏览器在独立离线数据目录中：创建含图片、无表头表格和公式的文档，启动综述并等待文献确认，打开运行详情并固定到侧栏，保持观察时插入公式和增加表格行，保存后刷新，确认内容仍在。
- 真实点击 Word 下载：文件内有图片、原生表格及 OMML 数学公式；浏览器无页面异常。

本次未调用付费模型，未发布、部署或修改远程仓库；GitHub CI 尚未触发。Word/WPS 视觉版式与各模型提供商推理字段仍按对应功能文档人工验收。

使用说明：[图片、表格与公式](rich-content.md)、[Agent 运行详情](agent-observability.md)。
