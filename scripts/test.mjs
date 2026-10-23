// npm test —— 后端离线测试 + 前端类型检查与编辑器测试
import { BACKEND, FRONTEND, ensureInstalled, log, requireTools, run } from './lib.mjs'

requireTools()
ensureInstalled()
log('后端测试')
run('uv', ['run', 'pytest', '-q'], { cwd: BACKEND })
log('前端类型检查')
run('npx', ['tsc', '-b'], { cwd: FRONTEND })
log('前端编辑器测试')
run('npm', ['test'], { cwd: FRONTEND })
log('全部通过')
