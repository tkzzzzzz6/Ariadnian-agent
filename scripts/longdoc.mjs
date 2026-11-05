// Paths passed to the Python CLIs are relative to backend/, as with npm run eval.
import { spawnSync } from 'node:child_process'
import { BACKEND } from './lib.mjs'

const commands = {
  fetch: ['longdoc_sources', 'fetch'],
  build: ['longdoc', 'build'],
  assess: ['longdoc', 'assess'],
  study: ['longdoc_sources.study'],
  synthesize: ['longdoc_sources.synthesize'],
  eval: ['longeval'],
  public: ['publicbench'],
}
const [name, ...args] = process.argv.slice(2)
if (!commands[name]) {
  console.error('Choose fetch, build, assess, study, synthesize, eval, or public')
  process.exit(2)
}
const [moduleName, ...fixed] = commands[name]
const result = spawnSync('uv', ['run', 'python', '-m', moduleName, ...fixed, ...args], {
  cwd: BACKEND,
  stdio: 'inherit',
  shell: false,
  env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
})
if (result.error) console.error(result.error.message)
process.exit(result.status ?? 1)
