import type { Editor, JSONContent } from '@tiptap/core'
import { CITE_RE, citeStore, citeToken, parseCiteBody, type CitePages } from './citations'

const leaf = (node: any) => {
  if (node.type.name === 'citation') return citeToken(node.attrs.ids ?? [], node.attrs.pages)
  if (node.type.name === 'hardBreak') return '\n'
  if (node.type.name === 'inlineMath') return `$${node.attrs.latex ?? ''}$`
  if (node.type.name === 'blockMath') return `$$${node.attrs.latex ?? ''}$$`
  if (node.type.name === 'image') return `[图片：${node.attrs.caption || node.attrs.alt || '未填写说明'}]`
  return ''
}

/** 把文档区间序列化成纯文本，引用写成 [@bibkey] */
export function textBetween(editor: Editor, from: number, to: number): string {
  const doc = editor.state.doc
  const parts: string[] = []
  let cursor = from
  doc.forEach((node, pos) => {
    if (node.type.name !== 'table' || pos < from || pos + node.nodeSize > to) return
    if (pos > cursor) parts.push(doc.textBetween(cursor, pos, '\n\n', leaf))
    const rows: string[] = []
    node.forEach(row => {
      const cells: string[] = []
      row.forEach(cell => cells.push(cell.textBetween(0, cell.content.size, ' ', leaf)))
      rows.push(cells.join('\t'))
    })
    parts.push('[表格]\n' + rows.join('\n'))
    cursor = pos + node.nodeSize
  })
  parts.push(doc.textBetween(cursor, to, '\n\n', leaf))
  return parts.filter(Boolean).join('\n\n')
}

/** 把含 [@key] 标记的文本转换回 TipTap 内容（引用变回 citation 节点）。 */
export function inlineFromText(text: string): JSONContent[] {
  const byKey = new Map(Object.values(citeStore.get().papers).map((p) => [p.bibkey, p.id]))
  const out: JSONContent[] = []
  let last = 0
  for (const m of text.matchAll(CITE_RE)) {
    const ids: string[] = []
    const pages: CitePages = {}
    for (const { key, page } of parseCiteBody(m[1])) {
      const id = byKey.get(key)
      if (!id) continue
      ids.push(id)
      if (page) pages[id] = page
    }
    if (!ids.length) continue // 文献库里没有的 key 原样保留为文本，绝不凭空生成引用
    if (m.index! > last) out.push({ type: 'text', text: text.slice(last, m.index) })
    out.push({ type: 'citation', attrs: { ids, pages } })
    last = m.index! + m[0].length
  }
  if (last < text.length) out.push({ type: 'text', text: text.slice(last) })
  return out.filter((n) => n.type !== 'text' || n.text)
}

export function contentFromText(text: string): JSONContent[] | JSONContent {
  const paras = text.replace(/\r/g, '').split(/\n{2,}/).filter((p) => p.trim())
  if (paras.length <= 1) return inlineFromText(text.trim())
  return paras.map((p) => ({ type: 'paragraph', content: inlineFromText(p.trim()) }))
}

/** 光标/选区所在的句子（用于"为这句话找引用"） */
export function sentenceAround(editor: Editor): { text: string; end: number } {
  const { state } = editor
  const { from, to, empty } = state.selection
  if (!empty) return { text: textBetween(editor, from, to), end: to }
  const $from = state.selection.$from
  const start = $from.start()
  const paraText = textBetween(editor, start, $from.end())
  const offset = from - start
  const before = paraText.slice(0, offset)
  const after = paraText.slice(offset)
  const sStart = Math.max(before.search(/[^。！？.!?]*$/), 0)
  const endRel = after.search(/[。！？.!?]/)
  const sentence = before.slice(sStart) + (endRel === -1 ? after : after.slice(0, endRel + 1))
  return { text: sentence.trim(), end: from }
}

/** 光标之前最近的标题 */
export function sectionAt(editor: Editor, pos: number): string {
  let title = ''
  editor.state.doc.nodesBetween(0, pos, (node) => {
    if (node.type.name === 'heading') title = node.textContent
    return node.isBlock && node.type.name !== 'heading'
  })
  return title
}

/** 文档里每处引用及其所在的句子（用于整篇引用核查） */
export function collectCitationContexts(editor: Editor): { pos: number; ids: string[]; sentence: string }[] {
  const out: { pos: number; ids: string[]; sentence: string }[] = []
  editor.state.doc.descendants((node, pos, parent) => {
    if (node.type.name !== 'citation' || !parent) return
    const $pos = editor.state.doc.resolve(pos)
    const start = $pos.start()
    const para = textBetween(editor, start, $pos.end())
    const off = textBetween(editor, start, pos).length
    const before = para.slice(0, off)
    const sStart = Math.max(before.search(/[^。！？.!?]*$/), 0)
    let sentence = before.slice(sStart).trim()
    if (sentence.length < 12) {
      // 引用在句末标点之后的情况："……。[1]" → 取前一句
      const prev = before.slice(0, sStart).replace(/[。！？.!?]\s*$/, '')
      sentence = (prev.slice(Math.max(prev.search(/[^。！？.!?]*$/), 0)) + sentence).trim()
    }
    sentence = sentence.replace(CITE_RE, '').trim()
    out.push({ pos, ids: node.attrs.ids, sentence })
  })
  return out
}
