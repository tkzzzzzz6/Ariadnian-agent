/**
 * 分节写作 Agent 的前端辅助：从文档里读大纲、把写好的一节插到对应标题下、把后端算好的批注直接交给页边批注。
 */
import type { Editor, JSONContent } from '@tiptap/core'
import type { Node as PMNode } from '@tiptap/pm/model'
import type { Annotation } from './review'
import { CITE_RE, citeStore } from './citations'
import { inlineFromText, textBetween } from './editorText'
import type { OutlineSection } from './types'

export interface SectionDraft {
  title: string
  status: 'pending' | 'drafted' | 'accepted' | 'skipped' | 'excluded' | 'error'
  paragraphs?: string[]
  annotations?: Annotation[][]
  fixed?: number
  issues?: number
  evidence?: { bibkey: string; page?: number | null; source: string }[]
}

const norm = (s: string) => s.replace(/\s+/g, '').toLowerCase()

/** 文档里的大纲：一级标题 + 紧跟的「写作要点：…」段落（insertOutline 写入的格式） */
export function outlineFromDoc(editor: Editor): OutlineSection[] {
  const papers = citeStore.get().papers
  const out: OutlineSection[] = []
  let cur: OutlineSection | null = null
  editor.state.doc.forEach((node) => {
    if (node.type.name === 'heading') {
      cur = node.attrs.level === 1 && node.textContent.trim() ? { title: node.textContent.trim(), goal: '', bibkeys: [] } : null
      if (cur) out.push(cur)
      return
    }
    if (!cur || node.type.name !== 'paragraph') return
    const text = node.textContent.trim()
    if (!text.startsWith('写作要点')) return
    cur.goal = text.replace(/^写作要点[:：]\s*/, '').replace(/\s*可引用[:：]?\s*$/, '').trim()
    node.descendants((c) => {
      if (c.type.name === 'citation') for (const id of c.attrs.ids ?? []) if (papers[id]) cur!.bibkeys.push(papers[id].bibkey)
      return true
    })
  })
  return out
}

const isPlaceholder = (n: PMNode) =>
  n.type.name === 'paragraph' && !n.attrs.draft && !n.content.content.some(c => c.isLeaf && !c.isText && c.type.name !== 'hardBreak')
    && (!n.textContent.trim() || n.textContent.trim().startsWith('写作要点'))

/** 章节标题位置，以及这一节结束的位置（下一个同级或更高级标题之前） */
function sectionRange(doc: PMNode, title: string): { end: number; headEnd: number } | null {
  let head: { level: number; end: number } | null = null
  let end = doc.content.size
  doc.forEach((node, offset) => {
    if (node.type.name !== 'heading') return
    if (head && end === doc.content.size && node.attrs.level <= head.level) end = offset
    if (!head && norm(node.textContent) === norm(title)) head = { level: node.attrs.level, end: offset + node.nodeSize }
  })
  return head ? { end, headEnd: (head as { end: number }).end } : null
}

/** 这一节在文档里是否已经有正文（不算「写作要点」、空段落和 Agent 草稿） */
export function sectionHasBody(editor: Editor, title: string): boolean {
  const r = sectionRange(editor.state.doc, title)
  if (!r) return false
  let body = false
  editor.state.doc.nodesBetween(r.headEnd, r.end, (node, pos) => {
    if (pos >= r.headEnd && node.isBlock && node.type.name !== 'heading' && !isPlaceholder(node) && !node.attrs?.draft) body = true
    return false
  })
  return body
}

export const draftKey = (threadId: string, i: number) => `${threadId.slice(0, 8)}-${i}`

/** 文档里属于某个草稿的段落（顶层） */
export function findDraft(editor: Editor, key: string): { pos: number; node: PMNode }[] {
  const out: { pos: number; node: PMNode }[] = []
  editor.state.doc.forEach((node, offset) => {
    if (node.type.name === 'paragraph' && node.attrs.draft === key) out.push({ pos: offset, node })
  })
  return out
}

/** 文档里所有草稿的 key（可按任务前缀过滤） */
export function listDrafts(editor: Editor, prefix = ''): string[] {
  const keys: string[] = []
  editor.state.doc.forEach((node) => {
    const k = node.attrs?.draft
    if (k && k.startsWith(prefix) && !keys.includes(k)) keys.push(k)
  })
  return keys
}

function paragraphsJSON(text: string, key: string, live: boolean): JSONContent[] {
  const parts = text.split(/\n\s*\n|\n/).map((p) => p.trim()).filter(Boolean)
  if (!parts.length) parts.push('')
  return parts.map((p, i) => ({
    type: 'paragraph',
    attrs: { draft: key, live: live && i === parts.length - 1 ? key : null },
    ...(p ? { content: inlineFromText(p) } : {}),
  }))
}

/**
 * 在章节标题下准备草稿的位置（已存在就不动）：在这一节末尾放一个空的草稿段落（「写作要点」先留着，保留草稿时再删）；
 * 找不到标题就在文末新建一级标题。
 */
export function ensureDraft(editor: Editor, key: string, title: string): void {
  if (findDraft(editor, key).length) return
  const { state } = editor
  const doc = state.doc
  const para = state.schema.nodeFromJSON(paragraphsJSON('', key, true)[0])
  const r = sectionRange(doc, title)
  let tr = state.tr
  if (!r) {
    const head = state.schema.nodeFromJSON({ type: 'heading', attrs: { level: 1 }, content: [{ type: 'text', text: title }] })
    const last = doc.lastChild
    const at = last && isPlaceholder(last) && !last.textContent ? doc.content.size - last.nodeSize : doc.content.size
    tr = tr.insert(at, [head, para])
  } else {
    // 节末若是空段落，草稿放在它前面
    let at = r.end
    const before = doc.resolve(r.end).nodeBefore
    if (before && r.end > r.headEnd && isPlaceholder(before) && !before.textContent) at = r.end - before.nodeSize
    tr = tr.insert(at, para)
  }
  editor.view.dispatch(tr.setMeta('addToHistory', false))
}

/** 用新文本替换草稿的全部段落（流式写作时反复调用） */
export function setDraftText(editor: Editor, key: string, text: string, live: boolean): void {
  const nodes = findDraft(editor, key)
  if (!nodes.length) return
  const from = nodes[0].pos
  const json = paragraphsJSON(text, key, live)
  // Replace only this Agent's draft paragraphs, preserving any user blocks
  // inserted between them (images, tables, formulas or regular paragraphs).
  let safe = editor.state.tr
  for (const { pos, node } of [...nodes].reverse()) safe = safe.delete(pos, pos + node.nodeSize)
  safe = safe.insert(from, json.map(j => editor.state.schema.nodeFromJSON(j)))
  editor.view.dispatch(safe.setMeta('addToHistory', false))
}

/** 保留草稿：去掉草稿标记，变成普通正文；同时删掉这一节的「写作要点」和空段落 */
export function keepDraft(editor: Editor, key: string): void {
  const nodes = findDraft(editor, key)
  if (!nodes.length) return
  const doc = editor.state.doc
  // 草稿所在章节的范围：前一个标题之后，到下一个标题之前
  const first = nodes[0].pos
  let start = 0
  let end = doc.content.size
  doc.forEach((node, offset) => {
    if (node.type.name !== 'heading') return
    if (offset < first) start = offset + node.nodeSize
    else if (end === doc.content.size) end = offset
  })
  let tr = editor.state.tr
  for (const { pos } of nodes) tr = tr.setNodeAttribute(pos, 'draft', null).setNodeAttribute(pos, 'live', null)
  const drop: [number, number][] = []
  doc.forEach((node, offset) => {
    if (offset >= start && offset < end && isPlaceholder(node)) drop.push([offset, offset + node.nodeSize])
  })
  for (const [from, to] of drop.reverse()) tr = tr.delete(from, to)
  editor.view.dispatch(tr)
}

/** 删除草稿 */
export function removeDraft(editor: Editor, key: string): void {
  const nodes = findDraft(editor, key)
  if (!nodes.length) return
  let tr = editor.state.tr
  for (const { pos, node } of [...nodes].reverse()) tr = tr.delete(pos, pos + node.nodeSize)
  editor.view.dispatch(tr)
}

/** 刷新页面后：之前没写完的流式段落去掉「正在写」标记 */
export function clearLive(editor: Editor): void {
  let tr = editor.state.tr
  let changed = false
  editor.state.doc.forEach((node, offset) => {
    if (node.type.name === 'paragraph' && node.attrs.live) {
      tr = tr.setNodeAttribute(offset, 'live', null)
      changed = true
    }
  })
  if (changed) editor.view.dispatch(tr.setMeta('addToHistory', false))
}

/** 滚动到草稿 / 段落 */
export function scrollToPos(editor: Editor, pos: number): void {
  const dom = editor.view.nodeDOM(pos) as HTMLElement | null
  // 只滚动编辑区（scrollIntoView 会连带滚动外层容器，把整个界面推出视口）
  const box = dom?.closest('.editor-scroll') as HTMLElement | null
  if (!dom || !box) return
  const top = dom.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - box.clientHeight / 3
  box.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
}

/** 按内容找段落（忽略引用和空白） */
function findParagraphByText(editor: Editor, text: string): { pos: number; node: PMNode } | null {
  const want = paraKey(text)
  let hit: { pos: number; node: PMNode } | null = null
  editor.state.doc.forEach((node, offset) => {
    if (hit || node.type.name !== 'paragraph') return
    if (paraKey(textBetween(editor, offset, offset + node.nodeSize)) === want) hit = { pos: offset, node }
  })
  return hit
}

export function locateParagraph(editor: Editor, text: string): number | null {
  return findParagraphByText(editor, text)?.pos ?? null
}

/**
 * 总编的一处修改：在文档里找到原段落，把它「重新打」成修改后的内容（逐步显示，能看到改了哪里），并短暂高亮。
 * 原段落已经被用户改过时返回 false。
 */
export async function animateEdit(editor: Editor, before: string, after: string, follow: boolean): Promise<boolean> {
  const hit = findParagraphByText(editor, before)
  if (!hit) return false
  const attrs = hit.node.attrs
  const tag = `edit-${Math.random().toString(36).slice(2, 8)}`
  const find = () => {
    let p: { pos: number; node: PMNode } | null = null
    editor.state.doc.forEach((node, offset) => {
      if (!p && node.attrs?.live === tag) p = { pos: offset, node }
    })
    return p as { pos: number; node: PMNode } | null
  }
  const put = (text: string, live: boolean) => {
    const target = find() ?? hit
    const node = editor.state.schema.nodeFromJSON({
      type: 'paragraph',
      attrs: { ...attrs, live: live ? tag : null },
      ...(text ? { content: inlineFromText(text) } : {}),
    })
    editor.view.dispatch(editor.state.tr.replaceWith(target.pos, target.pos + target.node.nodeSize, node).setMeta('addToHistory', false))
  }
  if (follow) scrollToPos(editor, hit.pos)
  // 先保留与原文相同的开头，再逐步打出后面的部分
  let common = 0
  while (common < before.length && common < after.length && before[common] === after[common]) common++
  put(after.slice(0, common), true)
  const steps = Math.min(24, Math.max(6, Math.ceil((after.length - common) / 6)))
  for (let k = 1; k <= steps; k++) {
    await new Promise((r) => setTimeout(r, 35))
    if (editor.isDestroyed || !find()) return false
    put(after.slice(0, common + Math.ceil(((after.length - common) * k) / steps)), k < steps)
  }
  const done = findParagraphByText(editor, after)
  if (done) {
    const dom = editor.view.nodeDOM(done.pos) as HTMLElement | null
    dom?.classList.add('ai-edit-flash')
    setTimeout(() => dom?.classList.remove('ai-edit-flash'), 2500)
  }
  return true
}

/** 段落去掉引用和空白后的文本，用于把后端批注对应到文档段落 */
export function paraKey(text: string): string {
  return norm(text.replace(CITE_RE, ''))
}

export const SEED_EVENT = 'aria:seed-review'

/** 写入文档后，把后端已经算好的批注交给页边批注（不再重复请求） */
export function seedAnnotations(draft: SectionDraft): void {
  const items = (draft.paragraphs ?? []).map((p, i) => ({ key: paraKey(p), annotations: draft.annotations?.[i] ?? [] }))
  window.dispatchEvent(new CustomEvent(SEED_EVENT, { detail: items }))
}

/** 把 [@key, p. 3] 换成可读的引用标签，用于预览 */
export function previewSegments(text: string): { text: string; cite?: { ids: string[]; pages: Record<string, string> } }[] {
  const nodes = inlineFromText(text)
  return nodes.map((n) => (n.type === 'citation' ? { text: '', cite: { ids: n.attrs!.ids, pages: n.attrs!.pages ?? {} } } : { text: n.text ?? '' }))
}
