// CPU preparation entry points; GPU execution uses the activated GPU Python.
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { cptInvocation } from './cpt.mjs'

export function experimentInvocation(module, args, environment = process.env) {
  if (!['gemma_eval', 'cpt.review', 'cpt.profile', 'cpt.compare'].includes(module)) {
    throw new Error('未知的实验准备模块。')
  }
  if (args.some((arg) => arg === '--execute' || arg.startsWith('--execute='))) {
    throw new Error('此入口使用轻量准备环境。实际 GPU 运行请激活独立环境后直接调用 python -m ' + module + '。')
  }
  const invocation = cptInvocation(args, environment)
  invocation.args[5] = module
  return invocation
}

export function main(args = process.argv.slice(2), spawn = spawnSync, output = console) {
  try {
    const invocation = experimentInvocation(args[0], args.slice(1))
    const result = spawn(invocation.command, invocation.args, invocation.options)
    if (result.error) output.error(result.error.message)
    return result.status ?? 1
  } catch (error) {
    output.error(error.message)
    return 2
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exit(main())
