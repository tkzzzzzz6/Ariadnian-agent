// npm test —— 后端、独立实验工具与前端验证
import { fileURLToPath } from 'node:url'
import { BACKEND, FRONTEND, ensureInstalled, log, requireTools, run } from './lib.mjs'

requireTools()
ensureInstalled()
log('后端测试')
run('uv', ['run', 'pytest', '-q'], { cwd: BACKEND })
log('独立实验工具测试')
run(process.execPath, [fileURLToPath(new URL('../experiments/ms_swift/scripts/test.mjs', import.meta.url))])
log('前端类型检查')
run('npx', ['tsc', '-b'], { cwd: FRONTEND })
log('前端回归测试')
run('npm', ['test'], { cwd: FRONTEND })
log('全部通过')
