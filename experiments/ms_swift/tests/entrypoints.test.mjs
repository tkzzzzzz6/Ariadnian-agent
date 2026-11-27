import assert from 'node:assert/strict'
import { test } from 'node:test'
import { join } from 'node:path'
import { cptInvocation, EXPERIMENT_ROOT, main, projectEnvironment } from '../scripts/cpt.mjs'
import { main as runTests, testInvocations } from '../scripts/test.mjs'
import { experimentInvocation } from '../scripts/experiment.mjs'

test('preparation keeps argument boundaries and runs the isolated experiment project', () => {
  const args = ['prepare', '--input', 'a corpus/with spaces.jsonl', '--output', 'literal;output']
  const invocation = cptInvocation(args, {})
  assert.equal(invocation.command, 'uv')
  assert.deepEqual(invocation.args, ['run', '--project', EXPERIMENT_ROOT, 'python', '-m', 'cpt', ...args])
  assert.equal(invocation.options.cwd, EXPERIMENT_ROOT)
  assert.equal(invocation.options.shell, false)
})

test('active backend environment cannot redirect imports or the experiment venv', () => {
  const inherited = { VIRTUAL_ENV: '/backend/.venv', PYTHONPATH: '/backend', PYTHONHOME: '/other/python',
    UV_PROJECT_ENVIRONMENT: '/backend/.venv', KEEP: 'yes' }
  const env = projectEnvironment(inherited)
  assert.equal(env.UV_PROJECT_ENVIRONMENT, join(EXPERIMENT_ROOT, '.venv'))
  for (const name of ['VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME']) assert.equal(env[name], undefined)
  assert.equal(env.KEEP, 'yes')
  assert.equal(inherited.PYTHONPATH, '/backend')
})

test('plan inspection invokes preparation without adding an execution flag', () => {
  const calls = []
  const status = main(['run', '--plan', 'local plan.json'], (...args) => { calls.push(args); return { status: 0 } }, {})
  assert.equal(status, 0)
  assert.equal(calls.length, 1)
  assert.deepEqual(calls[0][1].slice(-3), ['run', '--plan', 'local plan.json'])
  assert.equal(calls[0][1].includes('--execute'), false)
})

for (const executionFlag of ['--execute', '--exec', '--ex', '--e', '--execute=true']) {
  test(`${executionFlag} is refused before spawning a preparation process`, () => {
    const messages = []
    const status = main(['run', '--plan', 'plan.json', executionFlag], () => {
      assert.fail('must not invoke uv or training')
    }, {}, { error: (message) => messages.push(message) })
    assert.equal(status, 2)
    assert.match(messages[0], /Linux/)
    assert.match(messages[0], /python -m cpt run/)
  })
}

test('spawn errors and nonzero exits are returned to the caller', () => {
  assert.equal(main(['--help'], () => ({ status: 7 }), {}), 7)
  const messages = []
  assert.equal(main(['--help'], () => ({ error: new Error('uv missing') }), {}, { error: (m) => messages.push(m) }), 1)
  assert.deepEqual(messages, ['uv missing'])
})

test('experiment tests use their own Python project and preserve pytest arguments', () => {
  const calls = testInvocations(['-k', 'isolation'], {})
  assert.equal(calls.length, 2)
  assert.equal(calls[0].command, process.execPath)
  assert.equal(calls[1].command, 'uv')
  assert.deepEqual(calls[1].args, ['run', '--project', EXPERIMENT_ROOT, 'python', '-m', 'pytest', '-q', '-k', 'isolation'])
  for (const call of calls) assert.equal(call.options.cwd, EXPERIMENT_ROOT)
})

test('experiment test failure stops before running the next phase', () => {
  let count = 0
  assert.equal(runTests([], () => { count += 1; return { status: 3 } }), 3)
  assert.equal(count, 1)
})

test('additional experiment modules preserve environment isolation and argument boundaries', () => {
  for (const module of ['gemma_eval', 'cpt.review', 'cpt.profile', 'cpt.compare']) {
    const invocation = experimentInvocation(module, ['--output', 'with spaces'], { PYTHONPATH: '/backend' })
    assert.deepEqual(invocation.args.slice(4), ['-m', module, '--output', 'with spaces'])
    assert.equal(invocation.options.env.PYTHONPATH, undefined)
    assert.equal(invocation.options.shell, false)
  }
})

test('generic preparation entry cannot become an arbitrary module or execute GPU work', () => {
  assert.throws(() => experimentInvocation('arbitrary', []))
  for (const flag of ['--execute', '--execute=true', '--exec', '--e']) {
    assert.throws(() => experimentInvocation('gemma_eval', ['run', flag]))
  }
})
