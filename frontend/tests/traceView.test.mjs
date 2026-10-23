import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import ts from 'typescript'

const cache = new URL('../node_modules/.cache/ariadne-tests/', import.meta.url)
mkdirSync(cache, { recursive: true })
const compiled = new URL(`trace-view-${process.pid}.mjs`, cache)
writeFileSync(compiled, ts.transpileModule(readFileSync(new URL('../src/lib/traceView.ts', import.meta.url), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText)
const { expandedByDefault, groupCalls, groupAttempts, currentMessage, timelineRange } = await import(compiled.href)
after(() => unlinkSync(compiled))

test('completed stages collapse, current and problematic stages stay observable', () => {
  assert.equal(expandedByDefault({ status: 'done' }, false), false)
  assert.equal(expandedByDefault({ status: 'running' }, false), true)
  assert.equal(expandedByDefault({ status: 'waiting' }, false), true)
  assert.equal(expandedByDefault({ status: 'done' }, true), true)
})
test('tool/model hierarchy survives page boundaries without losing orphan calls', () => {
  const calls = [{ id: 'tool', parent_id: 'stage' }, { id: 'model', parent_id: 'tool' }, { id: 'orphan', parent_id: 'earlier-page' }]
  const tree = groupCalls(calls)
  assert.deepEqual(tree.get('').map(s => s.id), ['tool', 'orphan'])
  assert.deepEqual(tree.get('tool').map(s => s.id), ['model'])
})
test('only attempts with an explicit shared operation are grouped', () => {
  const groups = groupAttempts([
    { id: 'a', kind: 'llm', data: { operation: 'request1' } },
    { id: 'b', kind: 'llm', data: { operation: 'request2' } },
    { id: 'c', kind: 'llm', data: { operation: 'request1' } },
    { id: 'legacy', kind: 'llm', data: {} },
  ])
  assert.deepEqual(groups.map(g => g.map(s => s.id)), [['a', 'c'], ['b'], ['legacy']])
})
test('parallel durations use one wall-clock axis instead of their sum', () => {
  const spans = [{ started: 10, ended: 20 }, { started: 10, ended: 20 }]
  assert.equal(timelineRange(spans, 10, 20, 20000), 10)
})
test('a user confirmation is stated as an action, without guessed model activity', () => {
  assert.equal(currentMessage({ run: { status: 'waiting' }, spans: [] }), '需要你确认后继续')
  assert.equal(currentMessage({ run: { status: 'running' }, spans: [{ status: 'running', label: '撰写引言' }] }), '撰写引言')
})
