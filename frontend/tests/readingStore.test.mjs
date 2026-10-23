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
