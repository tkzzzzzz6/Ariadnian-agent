import { closeHistory } from '@tiptap/pm/history'
import type { EditorState } from '@tiptap/pm/state'
import type { Annotation, PlacedAnnotation } from './review'

type Rewrite = Extract<Annotation, { kind: 'rewrite' }>
export type ProseCache = Record<string, Rewrite | null>

export function textFingerprint(text: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(36)
}

/** Ignore the same advice until its claim or cited papers/verdicts change. */
export function reviewDismissKey(annotation: Annotation): string {
  const sources = annotation.kind === 'cite_check'
    ? annotation.items.map((item) => `${item.paper_id ?? item.bibkey}:${item.verdict}`).sort()
    : []
  return `advice:${textFingerprint(JSON.stringify([annotation.kind, annotation.anchor, sources]))}`
}

export function rememberProse(cache: ProseCache, plain: string, rewrite: Rewrite | null): ProseCache {
  return Object.fromEntries(Object.entries({ ...cache, [textFingerprint(plain)]: rewrite }).slice(-512))
}

/** Citation-only edits must not start another style-review round. */
export function stabilizeProse(cache: ProseCache, plain: string, annotations: Annotation[]) {
  const key = textFingerprint(plain)
  const rewrite = key in cache ? cache[key] : annotations.find((a): a is Rewrite => a.kind === 'rewrite') ?? null
  return {
    prose: rememberProse(cache, plain, rewrite),
    annotations: stableReviewAnnotations(cache, plain, annotations),
  }
}

export function stableReviewAnnotations(cache: ProseCache, plain: string, annotations: Annotation[]): Annotation[] {
  const key = textFingerprint(plain)
  const rewrite = key in cache ? cache[key] : annotations.find((a): a is Rewrite => a.kind === 'rewrite') ?? null
  return [...annotations.filter((a) => a.kind !== 'rewrite'), ...(rewrite && plain.includes(rewrite.anchor) ? [rewrite] : [])]
}

/** Apply plain text only, and never erase citation nodes as part of a rewrite. */
export function rewriteTransaction(state: EditorState, annotation: PlacedAnnotation) {
  if (annotation.kind !== 'rewrite' || !annotation.revised.trim()) return null
  const { from, to } = annotation
  if (from < 0 || to > state.doc.content.size || from >= to) return null
  const start = state.doc.resolve(from)
  if (!start.sameParent(state.doc.resolve(to)) || start.parent.type.name !== 'paragraph' || start.parent.attrs.live) return null
  if (state.doc.textBetween(from, to, '', '') !== annotation.anchor) return null
  let containsCitation = false
  state.doc.nodesBetween(from, to, (node) => { if (node.isLeaf && !node.isText && node.type.name !== 'hardBreak') containsCitation = true })
  if (containsCitation) return null
  const transaction = closeHistory(state.tr).insertText(annotation.revised, from, to)
  const paragraph = transaction.doc.nodeAt(start.before())!
  const plainAfter = paragraph.textBetween(0, paragraph.content.size, '', (leaf) => leaf.type.name === 'hardBreak' ? '\n' : '')
  return { transaction, plainAfter }
}
