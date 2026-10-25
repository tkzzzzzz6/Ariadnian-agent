import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import ts from 'typescript'

// Inject only the transport; exercise the actual store's refresh/save ordering.
const cache = new URL('../node_modules/.cache/ariadne-tests/', import.meta.url)
mkdirSync(cache, { recursive: true })
const compiled = new URL(`reading-store-${process.pid}.mjs`, cache)
const rules = new URL(`reading-store-rules-${process.pid}.mjs`, cache)
const options = { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }
writeFileSync(rules, ts.transpileModule(readFileSync(new URL('../src/lib/reading.ts', import.meta.url), 'utf8'), options).outputText)
let read, save
const injection = `__aria_reading_test_${process.pid}`
globalThis[injection] = { reading: () => read(), saveReading: (...args) => save(...args) }
const source = readFileSync(new URL('../src/lib/readingStore.ts', import.meta.url), 'utf8')
  .replace("import { api } from './api'", `const api = globalThis[${JSON.stringify(injection)}]`)
writeFileSync(compiled, ts.transpileModule(source, options).outputText.replace(/(['"])\.\/reading\1/g, JSON.stringify(rules.href)))
const { readingStore } = await import(compiled.href)
delete globalThis[injection]
after(() => { unlinkSync(compiled); unlinkSync(rules) })

test('late refresh cannot undo a saved record; failures retain data and deleted papers disappear on next refresh', async () => {
  const record = (id, status) => ({ paper_id: id, status, read_through_page: null, total_pages: null, updated_at: null })
  let finish
  read = () => new Promise(resolve => { finish = resolve })
  save = async (id, patch) => ({ ...record(id, 'unknown'), ...patch })
  const refresh = readingStore.refresh()
  await readingStore.save('a', { status: 'read' })
  finish([record('a', 'unknown'), record('b', 'unread')])
  await refresh
  assert.equal(readingStore.get().loaded, true)
  assert.equal(readingStore.get().records.a.status, 'read')
  assert.equal(readingStore.get().records.b.status, 'unread')
  read = async () => { throw new Error('offline') }
  await readingStore.refresh()
  assert.equal(readingStore.get().error, 'offline')
  assert.equal(readingStore.get().records.a.status, 'read')
  save = async () => { throw new Error('write failed') }
  await assert.rejects(readingStore.save('a', { status: 'unread' }), /write failed/)
  assert.equal(readingStore.get().records.a.status, 'read')
  read = async () => [record('b', 'read')]
  await readingStore.refresh()
  assert.equal(readingStore.get().records.a, undefined)
  assert.equal(readingStore.get().error, null)
})


test('refreshes requested during a library change wait for one fresh list', async () => {
  const record = id => ({ paper_id: id, status: 'unknown', read_through_page: null, total_pages: null, updated_at: null })
  const releases = []
  read = () => new Promise(resolve => releases.push(resolve))
  const initial = readingStore.refresh()
  // A paper was added (or deleted) after the server captured this first list.
  const afterChange = readingStore.refresh()
  const duplicate = readingStore.refresh()
  assert.equal(afterChange, duplicate)
  assert.notEqual(initial, afterChange)
  assert.equal(releases.length, 1)
  let settled = false
  afterChange.then(() => { settled = true })
  releases[0]([record('old-paper')])
  await initial
  assert.equal(settled, false)
  assert.equal(releases.length, 2)
  releases[1]([record('new-paper')])
  await afterChange
  assert.equal(releases.length, 2)
  assert.equal(readingStore.get().records['old-paper'], undefined)
  assert.equal(readingStore.get().records['new-paper'].status, 'unknown')
})

test('a queued refresh can recover after the preceding request failed', async () => {
  let fail, finish, calls = 0
  read = () => ++calls === 1 ? new Promise((_resolve, reject) => { fail = reject }) : new Promise(resolve => { finish = resolve })
  const first = readingStore.refresh()
  const retry = readingStore.refresh()
  fail(new Error('temporary outage'))
  await first
  assert.equal(calls, 2)
  assert.equal(readingStore.get().error, 'temporary outage')
  finish([])
  await retry
  assert.equal(readingStore.get().error, null)
  assert.deepEqual(readingStore.get().records, {})
})
