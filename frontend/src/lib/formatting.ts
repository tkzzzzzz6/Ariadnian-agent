import type { Editor, Range } from '@tiptap/core'
import { closeHistory } from '@tiptap/pm/history'
import type { EditorState } from '@tiptap/pm/state'

export const HIGHLIGHTS = ['#fff59d', '#c8e6c9', '#bbdefb'] as const
export function safeLink(value: string): string | null {
  const url = value.trim()
  if (!url || url.length > 2048 || /[\u0000-\u0020\u007f]/.test(url)) return null
  if (/^#[\w-]+$/.test(url)) return url
  try { const parsed = new URL(url); return ['http:', 'https:', 'mailto:'].includes(parsed.protocol) ? parsed.href : null }
  catch { return null }
}
export function formatAllowed(state: EditorState): boolean {
  const { $from, $to, from, to } = state.selection
  for (const pos of [$from, $to]) for (let d = pos.depth; d > 0; d--) {
    if (pos.node(d).attrs.live || pos.node(d).attrs.draft) return false
  }
  let protectedNode = false
  state.doc.nodesBetween(from, to, n => { if (n.attrs.live || n.attrs.draft) protectedNode = true })
  return !protectedNode
}
export type BlockKind = 'paragraph' | 'heading1' | 'heading2' | 'heading3' | 'bulletList' | 'orderedList' | 'blockquote' | 'codeBlock' | 'horizontalRule'
export const BLOCK_ITEMS: {kind: BlockKind; label: string; keywords: string; description: string}[] = [
  {kind:'paragraph',label:'正文',keywords:'text paragraph zhengwen',description:'普通正文段落'},
  ...[1,2,3].map(n => ({kind:`heading${n}` as BlockKind,label:`${n} 级标题`,keywords:`h${n} heading biaoti`,description:'设置文章层级'})),
  {kind:'bulletList',label:'无序列表',keywords:'bullet list liebiao',description:'按要点组织内容'},
  {kind:'orderedList',label:'有序列表',keywords:'number list liebiao',description:'按步骤组织内容'},
  {kind:'blockquote',label:'引用块',keywords:'quote yinyong',description:'区分摘录与正文'},
  {kind:'codeBlock',label:'代码块',keywords:'code daima',description:'保留代码缩进与换行'},
  {kind:'horizontalRule',label:'分割线',keywords:'divider line fengexian',description:'分隔正文内容'},
]
export function insertBlock(editor: Editor, kind: BlockKind, range?: Range) {
  if (!formatAllowed(editor.state)) return false
  const from = editor.state.selection.$from
  for (let d = from.depth; d > 0; d--) if (['tableCell','tableHeader'].includes(from.node(d).type.name) && kind !== 'paragraph') return false
  let chain = editor.chain().focus().command(({tr}) => { closeHistory(tr); return true })
  if (range) chain = chain.deleteRange(range)
  if (kind.startsWith('heading')) chain = chain.setHeading({level:Number(kind.slice(-1)) as 1|2|3})
  else if (kind === 'paragraph') chain = chain.setParagraph()
  else if (kind === 'bulletList') chain = chain.toggleBulletList()
  else if (kind === 'orderedList') chain = chain.toggleOrderedList()
  else if (kind === 'blockquote') chain = chain.toggleBlockquote()
  else if (kind === 'codeBlock') chain = chain.toggleCodeBlock()
  else chain = chain.setHorizontalRule()
  const ok = chain.run()
  editor.view.dispatch(closeHistory(editor.state.tr).setMeta('addToHistory',false))
  return ok
}
