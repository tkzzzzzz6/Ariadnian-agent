import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import ts from 'typescript'
import { Schema } from '@tiptap/pm/model'
import { EditorState } from '@tiptap/pm/state'
import { history, undo, redo } from '@tiptap/pm/history'

const cache = new URL('../node_modules/.cache/ariadne-tests/', import.meta.url)
mkdirSync(cache, { recursive: true })
const compiled = new URL(`reading-${process.pid}.mjs`, cache)
writeFileSync(compiled, ts.transpileModule(readFileSync(new URL('../src/lib/reading.ts', import.meta.url), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText)
const { collectCitationCounts, readingReminder, pageProgressLabel } = await import(compiled.href)
after(() => unlinkSync(compiled))
const schema = new Schema({ nodes: {
  doc: { content: 'paragraph+' }, paragraph: { content: 'inline*', group: 'block', attrs: { draft: { default: null } } },
  text: { group: 'inline' }, citation: { inline: true, atom: true, group: 'inline', attrs: { ids: { default: [] } } },
} })
const cite = (ids) => schema.node('citation', { ids })
const doc = (...nodes) => schema.node('doc', null, schema.node('paragraph', null, nodes))
const counts = (d) => ({ ...collectCitationCounts(d) })

test('counts nodes per paper, deduplicates each group and ignores plain-text citations', () => {
  assert.deepEqual(counts(doc(cite(['a', 'b', 'a']), schema.text('(Author, 2025)'), cite(['a']))), { a: 2, b: 1 })
  assert.deepEqual(counts(doc(cite([null, 1, '', 'a']), cite(null))), { a: 1 })
})
test('citation insertion, undo, redo, removal and undo update counts from actual transactions', () => {
  let state = EditorState.create({ schema, doc: doc(schema.text('正文')), plugins: [history()] })
  const dispatch = (tr) => { state = state.apply(tr) }
  dispatch(state.tr.insert(1, cite(['a', 'b'])))
  assert.deepEqual(counts(state.doc), { a: 1, b: 1 })
  assert.equal(undo(state, dispatch), true)
  assert.deepEqual(counts(state.doc), {})
  assert.equal(redo(state, dispatch), true)
  assert.deepEqual(counts(state.doc), { a: 1, b: 1 })
  dispatch(state.tr.delete(1, 2))
  assert.deepEqual(counts(state.doc), {})
  assert.equal(undo(state, dispatch), true)
  assert.deepEqual(counts(state.doc), { a: 1, b: 1 })
})
test('separate document counts do not leak and Agent drafts count as document content', () => {
  assert.deepEqual(counts(doc(cite(['a']), cite(['a']))), { a: 2 })
  assert.deepEqual(counts(doc(cite(['b']))), { b: 1 })
  const draft = schema.node('doc', null, schema.node('paragraph', { draft: 'task-1' }, cite(['a'])))
  assert.deepEqual(counts(draft), { a: 1 })
})
test('reminders use the threshold and distinguish unknown, confirmed unread and unfinished', () => {
  for (const status of ['unknown', 'unread', 'reading', 'read']) assert.equal(readingReminder(status, 1), null)
  assert.match(readingReminder('unknown', 2), /尚未记录阅读情况/)
  assert.match(readingReminder('unread', 2), /你已标为未读/)
  assert.match(readingReminder('reading', 2), /尚未确认读完/)
  assert.equal(readingReminder('read', 99), null)
})
test('unknown pages are not presented as zero percent or completed reading', () => {
  assert.equal(pageProgressLabel({ read_through_page: null, total_pages: null }), '读至页码未记录')
  assert.match(pageProgressLabel({ read_through_page: 2, total_pages: null }), /总页数未知/)
  assert.equal(pageProgressLabel({ read_through_page: 2, total_pages: 10 }), '读至第 2 页 / 共 10 页')
})
