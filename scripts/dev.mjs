// npm run dev —— 同时启动后端（8000，自动重载）和前端（5173，热更新），Ctrl+C 一起退出
// npm run demo —— 同上，但设置 LLM_FAKE=1，不调用任何模型
import { BACKEND, FRONTEND, ensureInstalled, killTree, log, openBrowser, paint, requireTools, serve, waitFor } from './lib.mjs'

const demo = process.argv.includes('--demo')
const apiPort = Number(process.env.ARIA_API_PORT || 8000)
const webPort = Number(process.env.ARIA_WEB_PORT || 5173)
if (![apiPort, webPort].every(p => Number.isInteger(p) && p > 0 && p <= 65535) || apiPort === webPort) {
  throw new Error('ARIA_API_PORT / ARIA_WEB_PORT 必须是不同的有效端口')
}
requireTools()
ensureInstalled()

// 离线演示也模拟一点生成速度，才能看到流式写作的效果
const env = demo ? { LLM_FAKE: '1', LLM_FAKE_STREAM_DELAY: process.env.LLM_FAKE_STREAM_DELAY ?? '0.05' } : {}
if (demo) log(paint('warn', '离线演示模式：不调用模型，学术检索返回示例数据'))

const api = serve('api', 'uv', ['run', 'uvicorn', 'app.main:app', '--reload', '--port', String(apiPort)], { cwd: BACKEND, env })
const web = serve('web', 'npm', ['run', 'dev', '--', '--strictPort', '--port', String(webPort)], { cwd: FRONTEND })

const stop = () => {
  killTree(api)
  killTree(web)
  process.exit(0)
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
for (const [name, child] of [['后端', api], ['前端', web]]) {
  child.on('exit', (code) => {
    if (code) console.error(paint('err', `${name}进程退出（code ${code}），全部停止`))
    stop()
  })
}

const ok = (await waitFor(`http://127.0.0.1:${apiPort}/api/health`)) && (await waitFor(`http://127.0.0.1:${webPort}`))
if (ok) {
  log(`已就绪：${paint('web', `http://localhost:${webPort}`)}   （Ctrl+C 退出）`)
  if (!process.env.NO_OPEN) openBrowser(`http://localhost:${webPort}`)
}
