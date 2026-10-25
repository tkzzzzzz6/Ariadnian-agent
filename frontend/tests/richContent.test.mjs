import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs'
import ts from 'typescript'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { Schema } from '@tiptap/pm/model'
import { EditorState, TextSelection } from '@tiptap/pm/state'
import { history, undo, redo, closeHistory } from '@tiptap/pm/history'

const cache = new URL('../node_modules/.cache/ariadne-tests/', import.meta.url)
mkdirSync(cache, { recursive: true })
const names = ['citations', 'richContent', 'export', 'editorText', 'writer', 'reviewPolicy', 'mathShortcuts', 'formatting', 'markdown', 'codeLanguages', 'codeHighlight']
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
const { delimitedMath, mathPasteTransaction, mathInputRules, mathInputMatch, syntaxShortcutTransaction } = await import(urls.mathShortcuts.href)
const { safeLink, formatAllowed } = await import(urls.formatting.href)
const { markdownToContent, citationParts, sourceLinks } = await import(urls.markdown.href)
const { codeLanguage, codeLanguageLabel } = await import(urls.codeLanguages.href)
const { highlightCode } = await import(urls.codeHighlight.href)
const { textBetween } = await import(urls.editorText.href)
const previewURL = new URL(`code-preview-${process.pid}.mjs`,cache)
let previewCode=ts.transpileModule(readFileSync(new URL('../src/components/CodePreview.tsx',import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX}}).outputText
for(const dep of ['codeHighlight','codeLanguages']) previewCode=previewCode.replaceAll(`'../lib/${dep}'`,JSON.stringify(urls[dep].href)).replaceAll(`"../lib/${dep}"`,JSON.stringify(urls[dep].href))
writeFileSync(previewURL,previewCode)
after(()=>unlinkSync(previewURL))
const {MarkdownCodePreview}=await import(previewURL.href)

const { rewriteTransaction } = await import(urls.reviewPolicy.href)

const schema = new Schema({ nodes: {
  doc: { content: 'block+' }, text: { group: 'inline' },
  paragraph: { group: 'block', content: 'inline*', attrs: { draft: { default: null }, live: { default: null } } },
  heading: { group: 'block', content: 'inline*', attrs: { level: { default: 1 } } },
  inlineMath: { inline: true, atom: true, group: 'inline', attrs: { latex: {} } },
  blockMath: { atom: true, group: 'block', attrs: { latex: {} } },
  codeBlock: {group:'block',content:'text*',marks:'',code:true,attrs:{language:{default:null},fencePending:{default:false},wrap:{default:true}}},
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

test('headerless Markdown tables include an empty header without consuming the first data row', () => {
  const md = toMarkdown({ type: 'doc', content: [tableJSON(3, 2, false, [['Method', 'Score'], ['', ''], ['Baseline', '0.8']])] }, 'Report')
  assert.ok(md.includes('|  |  |\n| --- | --- |\n| Method | Score |\n|  |  |\n| Baseline | 0.8 |'))
})


test('dollar shortcuts distinguish inline/display, escaped currency, code and invalid formulas', () => {
  const source = '正文 $x^2$，$$\\frac{a}{b}$$ 和 $y+1$'
  assert.deepEqual(delimitedMath(source).map(m => [m.latex, m.block]), [['x^2', false], ['\\frac{a}{b}', true], ['y+1', false]])
  for (const text of ['价格 $5 和 $10', '\\$x$', '`$x$`', '```\n$$x$$\n```', '$ x $', '$$', '$$$$', '$\\input{file}$', '$\\invalidcommand{x}$']) {
    assert.deepEqual(delimitedMath(text), [], text)
  }
  assert.equal(delimitedMath('$$\n\\sum_{i=1}^n x_i\n$$')[0].latex, '\\sum_{i=1}^n x_i')
})

test('a mixed formula paste preserves surrounding prose and is one undo/redo step', () => {
  const e = editorFor([p('Before after')])
  e.view.dispatch(e.state.tr.setSelection(TextSelection.create(e.state.doc, 8)))
  const before = e.state.doc.toJSON()
  e.view.dispatch(mathPasteTransaction(e.state, '行内 $x^2$\n$$\n\\frac{a}{b}\n$$\n结尾'))
  assert.ok(JSON.stringify(e.state.doc.toJSON()).includes('blockMath'))
  assert.ok(JSON.stringify(e.state.doc.toJSON()).includes('inlineMath'))
  assert.ok(e.state.doc.textContent.startsWith('Before '))
  assert.ok(e.state.doc.textContent.endsWith('after'))
  assert.equal(undo(e.state, e.view.dispatch), true)
  assert.deepEqual(e.state.doc.toJSON(), before)
  assert.equal(redo(e.state, e.view.dispatch), true)
})

test('formula shortcuts keep active drafts literal and never place display math in a table', () => {
  const e = editorFor([p('draft', { draft: 'agent' })])
  e.view.dispatch(e.state.tr.setSelection(TextSelection.create(e.state.doc, 2)))
  assert.equal(mathPasteTransaction(e.state, '$x$'), null)
  const table = editorFor([tableJSON(1, 1, false, [['cell']])])
  table.view.dispatch(table.state.tr.setSelection(TextSelection.create(table.state.doc, 4)))
  assert.equal(mathPasteTransaction(table.state, '$$x$$'), null)
  table.view.dispatch(mathPasteTransaction(table.state, '$x$'))
  assert.ok(JSON.stringify(table.state.doc.toJSON()).includes('inlineMath'))
})

test('loading old dollar text does not migrate it and input rules require a closing delimiter', () => {
  const e = editorFor([p('old $x$ and $$y$$')])
  assert.equal(e.state.doc.firstChild.textContent, 'old $x$ and $$y$$')
  const rules = mathInputRules(schema.nodes.inlineMath, schema.nodes.blockMath)
  const match = text => rules.map(r => r.find(text)).filter(Boolean)
  assert.equal(match('$x').length, 0)
  assert.equal(match('$$x$').length, 0)
  assert.equal(match('$x$')[0].data.block, false)
  assert.equal(match('$$x$$')[0].data.block, true)
})


test('Markdown answers become editable headings, lists, tables, math and real citations', () => {
  const source = '# Result\n\n**Strong** and [1,2], unknown [8].\n\n- First\n- Second\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n$x^2$\n\n$$\nx+y\n$$\n\n```python\nprint("[1]")\n```'
  const nodes=markdownToContent(source,[{n:1,paper_id:'p1'},{n:2,paper_id:'p2'}])
  assert.equal(nodes[0].type,'heading')
  assert.equal(markdownToContent('$$x+y$$')[0].type,'blockMath')
  const encoded=JSON.stringify(nodes)
  for(const kind of ['bold','citation','bulletList','table','inlineMath','blockMath','codeBlock']) assert.ok(encoded.includes(kind),kind)
  assert.ok(encoded.includes('[8]'))
  const cite=nodes[1].content.find(n=>n.type==='citation')
  assert.deepEqual(cite.attrs.ids,['p1','p2'])
  assert.equal(nodes.at(-1).content[0].text,'print("[1]")')
  assert.deepEqual(citationParts('[1,9]',[{n:1,paper_id:'p1'}]),[{type:'text',text:'[1,9]'}])
})

test('Markdown and editor links never admit executable protocols, images or fabricated sources', () => {
  for(const url of ['javascript:alert(1)','data:text/html,test','file:///secret','https://a.test/\nx']) assert.equal(safeLink(url),null,url)
  assert.equal(safeLink('https://example.com'),'https://example.com/')
  const content=markdownToContent('[x](javascript:alert%281%29) ![remote](https://example.com/a.png) `$x$ [1]` <script>alert(1)</script>')
  const encoded=JSON.stringify(content)
  assert.ok(!encoded.includes('"type":"image"'))
  assert.ok(!encoded.includes('"type":"link"'))
  assert.ok(!encoded.includes('"type":"citation"'))
  assert.ok(encoded.includes('$x$ [1]'))
})

test('format exports retain links, underline, strike, super/subscripts and approved highlight', () => {
  const nodes=['underline','strike','superscript','subscript','highlight','link'].map(type=>({type:'text',text:type,marks:[{type,attrs:type==='link'?{href:'https://example.com?q=x&n=2'}:{color:'#c8e6c9'}}]}))
  const doc={type:'doc',content:[{type:'paragraph',content:nodes}]}
  const md=toMarkdown(doc,'Report'), tex=toLatex(doc,'Report')
  for(const token of ['.underline','~~strike~~','^superscript^','~subscript~','.aria-highlight','https://example.com']) assert.ok(md.includes(token),token)
  for(const token of ['\\underline','\\sout','\\textsuperscript','\\textsubscript','\\colorbox','\\href','{ulem}']) assert.ok(tex.includes(token),token)
  const unsafe=toMarkdown({type:'doc',content:[{type:'paragraph',content:[{type:'text',text:'safe',marks:[{type:'link',attrs:{href:'javascript:alert(1)'}}]}]}]},'Report')
  assert.ok(!unsafe.includes('javascript'))
})

test('source preview links are injected only into normal text, not code, math or explicit links', () => {
  const tree={type:'root',children:[{type:'paragraph',children:[{type:'text',value:'[1] [9]'}, {type:'inlineCode',value:'[1]'}, {type:'inlineMath',value:'[1]'}, {type:'link',url:'https://a.test',children:[{type:'text',value:'[1]'}]}]}]}
  sourceLinks([{n:1,paper_id:'p1'}])()(tree)
  const children=tree.children[0].children
  assert.equal(children[0].url,'#aria-source-1')
  assert.equal(children[2].type,'inlineCode')
  assert.equal(children[3].type,'inlineMath')
  assert.equal(children[4].url,'https://a.test')
  assert.ok(children[1].value.includes('[9]'))
})


test('separate source numbers for chunks of the same paper stay distinct in preview', () => {
  const tree={type:'root',children:[{type:'paragraph',children:[{type:'text',value:'[1,2]'}]}]}
  sourceLinks([{n:1,paper_id:'p1'},{n:2,paper_id:'p1'}])()(tree)
  assert.deepEqual(tree.children[0].children.map(n=>n.url),['#aria-source-1','#aria-source-2'])
  assert.deepEqual(citationParts('[1,2]',[{n:1,paper_id:'p1'},{n:2,paper_id:'p1'}])[0].attrs.ids,['p1'])
})


test('unsupported native footnotes retain their marker and note text on insertion', () => {
  const content=markdownToContent('正文[^note].\n\n[^note]: 备注内容')
  const encoded=JSON.stringify(content)
  assert.ok(encoded.includes('[注 note]'))
  assert.ok(encoded.includes('备注内容'))
})


test('invalid math in Markdown remains literal instead of creating an unsavable formula', () => {
  const encoded=JSON.stringify(markdownToContent('$\\invalidcommand{x}$\n\n$$\n\\input{secret}\n$$'))
  assert.ok(!encoded.includes('"type":"inlineMath"'))
  assert.ok(!encoded.includes('"type":"blockMath"'))
  assert.ok(encoded.includes('invalidcommand'))
  assert.ok(encoded.includes('secret'))
})


test('newly closed formula suffix is recognized after an unrelated currency dollar', () => {
  assert.equal(mathInputMatch('价格 $5，公式 $x$').latex,'x')
  assert.equal(mathInputMatch('正文 $$x+y$$').block,true)
  assert.equal(mathInputMatch('`code $x$'),null)
  assert.equal(mathInputMatch('$$x$'),null)
  assert.equal(mathInputMatch('$old$foo$'),null)
})

test('Space/Enter converts restored math with trailing whitespace and preserves adjacent text', () => {
  const e=editorFor([p('正文 $$x+y$$ 后文')])
  const source=e.state.doc.firstChild.textContent
  const end=1+source.indexOf(' 后文')
  e.view.dispatch(e.state.tr.setSelection(TextSelection.create(e.state.doc,end)))
  const before=e.state.doc.toJSON()
  const result=syntaxShortcutTransaction(e.state,'Enter')
  assert.equal(result.consume,true)
  e.view.dispatch(result.tr)
  assert.deepEqual(e.state.doc.content.content.map(n=>n.type.name),['paragraph','blockMath','paragraph'])
  assert.equal(e.state.doc.firstChild.textContent,'正文 ')
  assert.equal(e.state.doc.lastChild.textContent,' 后文')
  assert.equal(undo(e.state,e.view.dispatch),true)
  assert.deepEqual(e.state.doc.toJSON(),before)
  const trailing=editorFor([p('$$x$$ ')])
  trailing.view.dispatch(trailing.state.tr.setSelection(TextSelection.atEnd(trailing.state.doc)))
  trailing.view.dispatch(syntaxShortcutTransaction(trailing.state,'Enter').tr)
  assert.deepEqual(trailing.state.doc.content.content.map(n=>n.type.name),['blockMath','paragraph'])
})

test('inline formula completion can recover on Space without reinterpreting code or drafts', () => {
  const e=editorFor([p('结果 $x$ ')])
  e.view.dispatch(e.state.tr.setSelection(TextSelection.atEnd(e.state.doc)))
  const result=syntaxShortcutTransaction(e.state,' ')
  assert.equal(result.consume,false)
  e.view.dispatch(result.tr)
  assert.equal(e.state.doc.firstChild.child(1).type.name,'inlineMath')
  const draft=editorFor([p('$x$',{draft:'agent'})])
  draft.view.dispatch(draft.state.tr.setSelection(TextSelection.atEnd(draft.state.doc)))
  assert.equal(syntaxShortcutTransaction(draft.state,'Enter'),null)
})

test('three typed backticks create an editable code block immediately', () => {
  const e=editorFor([p('``` ')])
  const rules=mathInputRules(schema.nodes.inlineMath,schema.nodes.blockMath,schema.nodes.codeBlock)
  const rule=rules[0]
  const match=rule.find.exec('```')
  assert.ok(match)
  const tr=e.state.tr
  rule.handler({state:{doc:e.state.doc,tr},range:{from:1,to:4},match})
  e.view.dispatch(tr)
  assert.equal(e.state.doc.firstChild.type.name,'codeBlock')
  assert.equal(e.state.selection.$from.parent.type.name,'codeBlock')
})

test('restored language fences enter code on Enter and a closing fence exits to prose', () => {
  const e=editorFor([p('```python')])
  e.view.dispatch(e.state.tr.setSelection(TextSelection.atEnd(e.state.doc)))
  e.view.dispatch(syntaxShortcutTransaction(e.state,'Enter').tr)
  assert.equal(e.state.doc.firstChild.attrs.language,'python')
  e.view.dispatch(e.state.tr.insertText('print(1)\n```'))
  const result=syntaxShortcutTransaction(e.state,'Enter')
  assert.equal(result.consume,true)
  e.view.dispatch(result.tr)
  assert.equal(e.state.doc.firstChild.textContent,'print(1)')
  assert.equal(e.state.doc.lastChild.type.name,'paragraph')
  assert.equal(e.state.selection.$from.parent.type.name,'paragraph')
})


test('code language aliases normalize safely and stay labeled in Markdown/LaTeX exports', () => {
  assert.equal(codeLanguage('py'),'python')
  assert.equal(codeLanguage('C++'),'cpp')
  assert.equal(codeLanguageLabel('ts'),'TypeScript')
  assert.equal(codeLanguage('python\nmalicious'),null)
  const doc={type:'doc',content:[{type:'codeBlock',attrs:{language:'python'},content:[{type:'text',text:'def hello():\n    return "world"'}]}]}
  assert.ok(toMarkdown(doc,'Code').includes('```python'))
  assert.ok(toLatex(doc,'Code').includes('Python'))
  assert.equal(markdownToContent('```python\ndef hello():\n    return 1\n```')[0].attrs.language,'python')
})

test('code highlighting decorates keywords while preserving every source character', () => {
  const source='def hello(name):\n    # comment\n    return "Hi " + name + str(42)'
  const tree=highlightCode(source,'python')
  const text=n=>n.type==='text'?n.value:(n.children||[]).map(text).join('')
  assert.equal(text(tree),source)
  assert.ok(JSON.stringify(tree).includes('hljs-keyword'))
  assert.ok(JSON.stringify(tree).includes('hljs-string'))
  assert.ok(JSON.stringify(tree).includes('hljs-comment'))
  const plain=highlightCode(source,'plaintext')
  assert.equal(text(plain),source)
  assert.ok(!JSON.stringify(plain).includes('hljs-keyword'))
  assert.equal(text(highlightCode(source,'unknown-language')),source)
  assert.equal(text(highlightCode('<script>alert(1)</script>','html')),'<script>alert(1)</script>')
})

test('large code blocks retain literal source when automatic highlighting is bounded', () => {
  const source='x'.repeat(13000)
  const result=highlightCode(source)
  assert.equal(result.data.language,'plaintext')
  assert.equal(result.children[0].value,source)
})

test('typing a language immediately after a shortcut fence sets language without losing real code', () => {
  const e=editorFor([{type:'codeBlock',attrs:{fencePending:true},content:[{type:'text',text:'python'}]}])
  e.view.dispatch(e.state.tr.setSelection(TextSelection.atEnd(e.state.doc)))
  const result=syntaxShortcutTransaction(e.state,'Enter')
  assert.equal(result.consume,true)
  e.view.dispatch(result.tr)
  assert.equal(e.state.doc.firstChild.attrs.language,'python')
  assert.equal(e.state.doc.firstChild.textContent,'')
  const literal=editorFor([{type:'codeBlock',attrs:{fencePending:true},content:[{type:'text',text:'print(1)'}]}])
  literal.view.dispatch(literal.state.tr.setSelection(TextSelection.atEnd(literal.state.doc)))
  const keep=syntaxShortcutTransaction(literal.state,'Enter')
  assert.equal(keep.consume,false)
  literal.view.dispatch(keep.tr)
  assert.equal(literal.state.doc.firstChild.textContent,'print(1)')
})


test('Markdown answer code renders its language and highlighted tokens without executing source HTML', () => {
  const html=renderToStaticMarkup(createElement(MarkdownCodePreview,{children:createElement('code',{className:'language-python'},'def greet():\n    return "hello"')}))
  assert.ok(html.includes('Python'))
  assert.ok(html.includes('hljs-keyword'))
  assert.ok(html.includes('hljs-string'))
  const literal=renderToStaticMarkup(createElement(MarkdownCodePreview,{children:createElement('code',{className:'language-html'},'<script>alert(1)</script>')}))
  assert.ok(!literal.includes('<script>'))
  assert.ok(literal.includes('&lt;'))
})
