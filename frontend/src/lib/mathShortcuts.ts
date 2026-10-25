import type { JSONContent } from '@tiptap/core'
import { InputRule } from '@tiptap/core'
import { Fragment, Slice, type NodeType, type ResolvedPos } from '@tiptap/pm/model'
import { closeHistory } from '@tiptap/pm/history'
import { TextSelection, type EditorState, type Transaction } from '@tiptap/pm/state'
import katex from 'katex'
import { knownCodeLanguage } from './codeLanguages'
import { unsafeTex } from './richContent'

interface Formula { from: number; to: number; latex: string; block: boolean }
const escaped = (text: string, at: number) => {
  let count = 0
  while (at > 0 && text[--at] === '\\') count++
  return count % 2 === 1
}
export function validMathSource(latex: string, block: boolean) {
  if (!latex || latex.length > 8000 || unsafeTex.test(latex)) return false
  try { katex.renderToString(latex, { displayMode: block, throwOnError: true, trust: false, maxExpand: 1000 }); return true }
  catch { return false }
}

/** Parse only newly typed/pasted text; never migrate saved documents or Agent output. */
export function delimitedMath(text: string): Formula[] {
  const matches: Formula[] = []
  for (let i = 0; i < text.length;) {
    // Markdown code spans/fences are literal, even in a plain-text paste.
    if (text[i] === '`' && !escaped(text, i)) {
      let end = i + 1
      while (text[end] === '`') end++
      const close = text.indexOf(text.slice(i, end), end)
      i = close < 0 ? end : close + end - i
      continue
    }
    if (text[i] !== '$' || escaped(text, i)) { i++; continue }
    const start = i
    while (text[i] === '$') i++
    const width = i - start
    if (width > 2) continue
    const contentStart = i
    let end = i
    for (; end < text.length; end++) {
      if (width === 1 && text[end] === '\n') break
      if (text[end] !== '$' || escaped(text, end)) continue
      let after = end + 1
      while (text[after] === '$') after++
      if (after - end !== width) { end = after - 1; continue }
      const raw = text.slice(contentStart, end)
      const latex = raw.trim()
      if ((width === 2 || raw === latex) && validMathSource(latex, width === 2)) {
        matches.push({ from: start, to: after, latex, block: width === 2 })
      }
      i = after
      break
    }
    if (end >= text.length || (width === 1 && text[end] === '\n')) i = end
  }
  return matches
}

function allowed(pos: ResolvedPos, block: boolean) {
  for (let depth = pos.depth; depth > 0; depth--) {
    const n = pos.node(depth)
    if (n.type.spec.code || n.attrs.live || n.attrs.draft) return false
    if (block && ['tableCell', 'tableHeader'].includes(n.type.name)) return false
  }
  return !pos.marks().some(m => m.type.spec.code)
}

/** Prefer the newly closed suffix, even when earlier prose contains a lone currency $. */
export function mathInputMatch(text: string): Formula | null {
  if (!text.endsWith('$')) return null
  const end=text.length
  let close=end-1
  while(close>0&&text[close-1]==='$') close--
  const width=end-close
  if(width>2||escaped(text,close)) return null
  for(let at=close-1;at>=0;at--) {
    if(text[at]!=='$'||escaped(text,at)) continue
    let start=at
    while(start>0&&text[start-1]==='$') start--
    const run=at-start+1
    at=start
    if(run!==width) continue
    const raw=text.slice(start+width,close),latex=raw.trim()
    if(width===1&&(raw!==latex||raw.includes('\n'))) return null
    // A closing delimiter of a prior complete formula cannot open a new one.
    if(delimitedMath(text.slice(0,start+width)).some(m=>m.to===start+width)) return null
    const prefix=text.slice(0,start)
    let code=false
    for(let i=0;i<prefix.length;i++) if(prefix[i]==='`'&&!escaped(prefix,i)) {
      let j=i+1;while(prefix[j]==='`')j++
      const fence=prefix.slice(i,j),next=prefix.indexOf(fence,j)
      if(next<0){code=true;break}
      i=next+fence.length-1
    }
    if(code||!validMathSource(latex,width===2)) return null
    return {from:start,to:end,latex,block:width===2}
  }
  return null
}

function replaceMath(tr: Transaction, from: number, to: number, m: Formula) {
  const a=tr.doc.resolve(from),b=tr.doc.resolve(to)
  if(!a.sameParent(b)||!allowed(a,m.block)) return false
  let code=false
  tr.doc.nodesBetween(from,to,n=>{if(n.marks.some(m=>m.type.spec.code))code=true})
  if(code) return false
  const math=tr.doc.type.schema.nodes[m.block?'blockMath':'inlineMath'].create({latex:m.latex})
  if(!m.block) {
    tr.replaceWith(from,to,math)
    tr.setSelection(TextSelection.near(tr.doc.resolve(from+1)))
    return true
  }
  if(a.parent.type.name!=='paragraph') return false
  const paragraph=a.parent.type
  const before=a.parent.content.cut(0,a.parentOffset),after=a.parent.content.cut(b.parentOffset)
  const meaningful=(fragment:Fragment)=>fragment.content.some(n=>!n.isText||n.text!.trim())
  const left=meaningful(before)?paragraph.create(a.parent.attrs,before):null
  const right=paragraph.create(a.parent.attrs,meaningful(after)?after:null)
  const nodes=left?[left,math,right]:[math,right]
  const parent=a.node(-1),index=a.index(-1)
  if(!parent.canReplace(index,a.indexAfter(-1),Fragment.fromArray(nodes))) nodes.unshift(paragraph.create())
  if(!parent.canReplace(index,a.indexAfter(-1),Fragment.fromArray(nodes))) return false
  const start=a.before()
  tr.replaceWith(start,b.after(),nodes)
  const cursor=start+nodes.slice(0,-1).reduce((size,n)=>size+n.nodeSize,0)+1
  tr.setSelection(TextSelection.create(tr.doc,cursor))
  return true
}

/** Immediate typed fences and formula delimiters share Tiptap's undoable input rules. */
export function mathInputRules(_inline: NodeType, _block: NodeType, code?: NodeType) {
  const rules=[true,false].map(display=>new InputRule({
    find:text=>{
      const m=mathInputMatch(text)
      return m&&m.block===display?{index:m.from,text:text.slice(m.from),data:m}:null
    },
    handler:({state,range,match})=>{
      if(!replaceMath(state.tr,range.from,range.to,match.data as Formula)) return null
    },
  }))
  if(code) rules.unshift(new InputRule({find:/^\s*```$/,
    handler:({state,range})=>{
      const pos=state.doc.resolve(range.from)
      if(!allowed(pos,true)||pos.parent.type.name!=='paragraph'||state.doc.textBetween(range.to,pos.end()).trim())return null
      const parent=pos.node(-1),index=pos.index(-1)
      if(!parent.canReplaceWith(index,pos.indexAfter(-1),code))return null
      const start=pos.before()
      state.tr.replaceWith(start,pos.after(),code.create({fencePending:true}))
      state.tr.setSelection(TextSelection.create(state.tr.doc,start+1))
    },
  }))
  return rules
}

/** Explicit Space/Enter also converts source inserted by IME, rich paste or restored text. */
export function syntaxShortcutTransaction(state: EditorState, key: string) {
  if(!state.selection.empty||![' ','Enter'].includes(key))return null
  const pos=state.selection.$from
  if(pos.parent.type.name==='codeBlock') {
    if(key!=='Enter'||pos.parentOffset!==pos.parent.content.size)return null
    if(pos.parent.attrs.fencePending&&!pos.parent.textContent.includes('\n')) {
      const language=knownCodeLanguage(pos.parent.textContent.trim())
      const tr=closeHistory(state.tr).setNodeMarkup(pos.before(),undefined,{...pos.parent.attrs,language:language||pos.parent.attrs.language,fencePending:false})
      if(language)tr.delete(pos.start(),pos.end()).setSelection(TextSelection.create(tr.doc,pos.start()))
      return {tr,consume:!!language}
    }
    const lines=pos.parent.textContent.split('\n')
    if(lines.at(-1)!=='```')return null
    lines.pop()
    const value=lines.join('\n'),schema=state.schema
    const code=pos.parent.type.create(pos.parent.attrs,value?schema.text(value):null)
    const tr=closeHistory(state.tr).replaceWith(pos.before(),pos.after(),[code,schema.nodes.paragraph.create()])
    return {tr:tr.setSelection(TextSelection.create(tr.doc,pos.before()+code.nodeSize+1)),consume:true}
  }
  if(!allowed(pos,false)||pos.parent.type.name!=='paragraph')return null
  const text=pos.parent.textBetween(0,pos.parentOffset,'\n','\ufffc')
  if(/^\s*```(?:[a-zA-Z0-9_+-]+)?\s*$/.test(text)&&!state.doc.textBetween(pos.pos,pos.end()).trim()) {
    if(!allowed(pos,true))return null
    const code=state.schema.nodes.codeBlock
    const parent=pos.node(-1),index=pos.index(-1)
    if(!code||!parent.canReplaceWith(index,pos.indexAfter(-1),code))return null
    const language=text.trim().slice(3)||null
    const tr=closeHistory(state.tr).replaceWith(pos.before(),pos.after(),code.create({language}))
    return {tr:tr.setSelection(TextSelection.create(tr.doc,pos.before()+1)),consume:true}
  }
  const trimmed=text.trimEnd(),m=mathInputMatch(trimmed)
  if(!m)return null
  const tr=closeHistory(state.tr)
  if(!replaceMath(tr,pos.start()+m.from,pos.start()+m.to,m))return null
  return {tr,consume:m.block&&key==='Enter'}
}

/** A single paste transaction preserves surrounding content and undoes as one edit. */
export function mathPasteTransaction(state: EditorState, text: string) {
  const matches = delimitedMath(text)
  if (!matches.length) return null
  const { $from, $to } = state.selection
  const display = matches.some(m => m.block)
  if (!$from.sameParent($to) || !allowed($from, display) || !allowed($to, display)) return null
  let protectedText = false
  state.doc.nodesBetween(state.selection.from, state.selection.to, n => {
    if (n.marks.some(m => m.type.spec.code)) protectedText = true
  })
  if (protectedText) return null
  const blocks: JSONContent[] = []
  let inline: JSONContent[] = []
  const flush = () => { blocks.push({ type: 'paragraph', content: inline }); inline = [] }
  const append = (value: string) => {
    value.replace(/\r\n?/g, '\n').split('\n').forEach((line, i) => {
      if (i) flush()
      if (line) inline.push({ type: 'text', text: line })
    })
  }
  let cursor = 0
  for (const m of matches) {
    append(text.slice(cursor, m.from))
    if (m.block) {
      if (inline.length) flush()
      blocks.push({ type: 'blockMath', attrs: { latex: m.latex } })
    } else inline.push({ type: 'inlineMath', attrs: { latex: m.latex } })
    cursor = m.to
  }
  append(text.slice(cursor))
  if (inline.length) flush()
  const nodes = blocks.map(n => state.schema.nodeFromJSON(n))
  // Inline pastes stay inside their paragraph, preserving its heading/format.
  const fragment = !display && blocks.length === 1 ? nodes[0].content : Fragment.fromArray(nodes)
  return closeHistory(state.tr).replaceSelection(new Slice(fragment, 0, 0)).scrollIntoView()
}
