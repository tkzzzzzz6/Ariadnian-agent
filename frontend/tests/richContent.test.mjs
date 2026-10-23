import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs'
import ts from 'typescript'
import { Schema } from '@tiptap/pm/model'
import { EditorState, TextSelection } from '@tiptap/pm/state'
import { history, undo, redo, closeHistory } from '@tiptap/pm/history'

const cache = new URL('../node_modules/.cache/ariadne-tests/', import.meta.url)
mkdirSync(cache, { recursive: true })
const names = ['citations', 'richContent', 'export', 'editorText', 'writer', 'reviewPolicy']
const urls = Object.fromEntries(names.map(name => [name, new URL(`rich-${name}-${process.pid}.mjs`, cache)]))
for (const name of names) {
  let code = ts.transpileModule(readFileSync(new URL(`../src/lib/${name}.ts`, import.meta.url), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText
  for (const dependency of names) code = code.replaceAll(`'./${dependency}'`, JSON.stringify(urls[dependency].href)).replaceAll(`"./${dependency}"`, JSON.stringify(urls[dependency].href))
  writeFileSync(urls[name], code)
}
after(() => names.forEach(n => unlinkSync(urls[n])))
const { tableJSON, parseTSV, pasteTable, containsRich, richStructureError } = await import(urls.richContent.href)
const { toMarkdown, toLatex, imageIds } = await import(urls.export.href)
const { setDraftText, removeDraft, keepDraft } = await import(urls.writer.href)
const { textBetween } = await import(urls.editorText.href)
const { rewriteTransaction } = await import(urls.reviewPolicy.href)

const schema = new Schema({ nodes: {
  doc: { content: 'block+' }, text: { group: 'inline' },
  paragraph: { group: 'block', content: 'inline*', attrs: { draft: { default: null }, live: { default: null } } },
  heading: { group: 'block', content: 'inline*', attrs: { level: { default: 1 } } },
  inlineMath: { inline: true, atom: true, group: 'inline', attrs: { latex: {} } },
  blockMath: { atom: true, group: 'block', attrs: { latex: {} } },
  image: { atom: true, group: 'block', attrs: { assetId: {}, extension: {}, caption: {}, alt: { default: '' }, widthPercent: {} } },
  citation: { inline: true, atom: true, group: 'inline', attrs: { ids: { default: [] }, pages: { default: {} } } },
  table: { group: 'block', content: 'tableRow+' }, tableRow: { content: '(tableCell|tableHeader)+' },
  tableCell: { content: 'paragraph+', attrs: { colspan: { default: 1 }, rowspan: { default: 1 } } },
  tableHeader: { content: 'paragraph+', attrs: { colspan: { default: 1 }, rowspan: { default: 1 } } },
} })
const aid = 'a'.repeat(32)
const image = { type: 'image', attrs: { assetId: aid, extension: 'png', caption: 'Figure A', widthPercent: 50 } }
const math = { type: 'inlineMath', attrs: { latex: '\\frac{a}{b}' } }
const p = (text, attrs) => ({ type: 'paragraph', attrs, content: text ? [{ type: 'text', text }] : [] })
function editorFor(content) {
  const e = { state: EditorState.create({ schema, doc: schema.nodeFromJSON({ type: 'doc', content }), plugins: [history()] }) }
  e.view = { dispatch: tr => { e.state = e.state.apply(tr) } }
  return e
}

test('clipboard tables preserve empty cells, reject ragged and oversized data', () => {
  assert.deepEqual(parseTSV('A\tB\r\n1\t\r\n'), [['A', 'B'], ['1', '']])
  assert.equal(parseTSV('ordinary text'), null)
  assert.throws(() => parseTSV('a\tb\nc'))
  assert.throws(() => tableJSON(51, 2))
  assert.throws(() => tableJSON(2, 11))
  assert.equal(schema.nodeFromJSON(tableJSON(3, 2)).childCount, 3)
})

test('Markdown/LaTeX keep figure, native table, inline/block math in order', () => {
  const doc = { type: 'doc', content: [p('Before'), image, tableJSON(2, 2, true, [['Method', 'Score'], ['A | B', '0.8']]),
    { type: 'paragraph', content: [math] }, { type: 'blockMath', attrs: { latex: 'x^2+y^2' } }, p('After')] }
  const md = toMarkdown(doc, 'Title "quoted"\nnewline')
  assert.match(md, /title: "Title \\"quoted\\"\\nnewline"/)
  assert.ok(md.includes(`images/${aid}.png`))
  assert.ok(md.includes('A \\| B'))
  assert.ok(md.includes('$\\frac{a}{b}$'))
  const tex = toLatex(doc, 'Report')
  for (const token of ['\\includegraphics', '\\begin{longtable}', '\\(\\frac{a}{b}\\)', '\\[\nx^2+y^2']) assert.ok(tex.includes(token), token)
  assert.ok(tex.indexOf('Before') < tex.indexOf('includegraphics') && tex.indexOf('includegraphics') < tex.indexOf('longtable}') && tex.indexOf('longtable}') < tex.indexOf('After'))
  assert.deepEqual(imageIds(doc), [aid])
  assert.ok(!tex.includes('\\bibliography{refs}'))
})

test('plain dollar and markup text are not reinterpreted as formulas or images', () => {
  const md = toMarkdown({ type: 'doc', content: [p('$5 and [x] *text* \\input{file}')] }, 'plain')
  assert.ok(md.includes('\\$5'))
  assert.ok(md.includes('\\[x\\]'))
  assert.ok(md.includes('\\*text\\*'))
})

test('Agent context includes formula source, figure description and table rows', () => {
  const e = editorFor([{ type: 'paragraph', content: [math] }, image, tableJSON(2, 2, false, [['a', 'b'], ['c', 'd']])])
  const text = textBetween(e, 0, e.state.doc.content.size)
  assert.ok(text.includes('$\\frac{a}{b}$'))
  assert.ok(text.includes('[图片：Figure A]'))
  assert.ok(text.includes('a\tb\nc\td'))
  assert.equal(containsRich(e.state.doc, 1, 2), true)
  assert.equal(richStructureError(e.state.doc), null)
})

test('streaming draft updates and removal preserve user blocks between drafts', () => {
  const e = editorFor([p('old1', { draft: 'agent', live: 'agent' }), image, tableJSON(1, 1), { type: 'blockMath', attrs: { latex: 'x' } }, p('old2', { draft: 'agent' })])
  setDraftText(e, 'agent', 'new draft', false)
  assert.deepEqual(e.state.doc.content.content.map(n => n.type.name), ['paragraph', 'image', 'table', 'blockMath'])
  removeDraft(e, 'agent')
  assert.deepEqual(e.state.doc.content.content.map(n => n.type.name), ['image', 'table', 'blockMath'])
})

test('keeping a draft does not delete an otherwise empty formula paragraph', () => {
  const e = editorFor([{ type: 'paragraph', content: [math] }, p('draft', { draft: 'agent' })])
  keepDraft(e, 'agent')
  assert.equal(e.state.doc.firstChild.firstChild.type.name, 'inlineMath')
  assert.equal(e.state.doc.lastChild.attrs.draft, null)
})

test('one undo/redo round trip restores a mixed rich-content insertion', () => {
  const e = editorFor([p('existing')])
  const before = e.state.doc.toJSON()
  const inserted = schema.nodeFromJSON(image)
  e.view.dispatch(closeHistory(e.state.tr).insert(e.state.doc.content.size, inserted))
  e.view.dispatch(closeHistory(e.state.tr).setMeta('addToHistory', false))
  assert.equal(undo(e.state, e.view.dispatch), true)
  assert.deepEqual(e.state.doc.toJSON(), before)
  assert.equal(redo(e.state, e.view.dispatch), true)
  assert.equal(e.state.doc.lastChild.type.name, 'image')
})

test('text-only rewrite must not remove a formula hidden inside the anchor', () => {
  const e = editorFor([{ type: 'paragraph', content: [{ type: 'text', text: 'A' }, math, { type: 'text', text: 'B' }] }])
  const note = { kind: 'rewrite', anchor: 'AB', from: 1, to: 4, revised: 'new' }
  assert.equal(rewriteTransaction(e.state, note), null)
})

test('TSV cell paste grows a table, preserves surrounding blocks and undoes once', () => {
  const e = editorFor([p('Before'), tableJSON(2, 2, false, [['old', 'keep'], ['last', 'stay']]), image])
  const before = e.state.doc.toJSON()
  const tablePos = e.state.doc.firstChild.nodeSize
  e.view.dispatch(e.state.tr.setSelection(TextSelection.near(e.state.doc.resolve(tablePos + 4))))
  e.view.dispatch(pasteTable(e.state, [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]))
  assert.equal(e.state.doc.child(1).childCount, 3)
  assert.equal(e.state.doc.child(1).firstChild.childCount, 3)
  assert.equal(e.state.doc.child(1).lastChild.lastChild.textContent, 'I')
  assert.equal(e.state.doc.lastChild.type.name, 'image')
  assert.equal(undo(e.state, e.view.dispatch), true)
  assert.deepEqual(e.state.doc.toJSON(), before)
})
