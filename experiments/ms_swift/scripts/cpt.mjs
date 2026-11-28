// Lightweight preparation only. Real training uses the dedicated Linux venv.
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const EXPERIMENT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')

export function projectEnvironment(environment = process.env) {
  const env = { ...environment }
  // An activated application venv or shell PYTHONPATH must not redirect this
  // command to backend/.venv or import the old training package accidentally.
  delete env.VIRTUAL_ENV
  delete env.PYTHONPATH
  delete env.PYTHONHOME
  env.UV_PROJECT_ENVIRONMENT = join(EXPERIMENT_ROOT, '.venv')
  env.PYTHONIOENCODING = 'utf-8'
  env.PYTHONUTF8 = '1'
  return env
}

export function cptInvocation(args, environment = process.env) {
  // argparse accepts long-option abbreviations, so reject --exec and --e too.
  if (args.some((arg) => {
    const option = arg.split('=', 1)[0]
    return option.startsWith('--') && option.length > 2 && '--execute'.startsWith(option)
  })) {
    throw new Error('此入口仅用于数据准备和计划复核。真实训练请在 Linux 独立训练环境中，从 experiments/ms_swift/ 直接运行 python -m cpt run --plan <计划路径> --execute。')
  }
  return {
    command: 'uv',
    args: ['run', '--project', EXPERIMENT_ROOT, 'python', '-m', 'cpt', ...args],
    options: { cwd: EXPERIMENT_ROOT, stdio: 'inherit', shell: false, env: projectEnvironment(environment) },
  }
}

export function main(args = process.argv.slice(2), spawn = spawnSync, environment = process.env, output = console) {
  let invocation
  try {
    invocation = cptInvocation(args, environment)
  } catch (error) {
    output.error(error.message)
    return 2
  }
  const result = spawn(invocation.command, invocation.args, invocation.options)
  if (result.error) output.error(result.error.message)
  return result.status ?? 1
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exit(main())
