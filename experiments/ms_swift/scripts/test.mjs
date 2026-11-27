// The experimental Python project stays independent of backend/.venv.
import { spawnSync } from 'node:child_process'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { EXPERIMENT_ROOT, projectEnvironment } from './cpt.mjs'

export function testInvocations(args = [], environment = process.env) {
  const options = { cwd: EXPERIMENT_ROOT, stdio: 'inherit', shell: false, env: projectEnvironment(environment) }
  return [
    { command: process.execPath, args: ['--test', join(EXPERIMENT_ROOT, 'tests', 'entrypoints.test.mjs')], options },
    { command: 'uv', args: ['run', '--project', EXPERIMENT_ROOT, 'python', '-m', 'pytest', '-q', ...args], options },
  ]
}

export function main(args = process.argv.slice(2), spawn = spawnSync, output = console) {
  for (const invocation of testInvocations(args)) {
    const result = spawn(invocation.command, invocation.args, invocation.options)
    if (result.error) output.error(result.error.message)
    if (result.status !== 0) return result.status ?? 1
  }
  return 0
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exit(main())
