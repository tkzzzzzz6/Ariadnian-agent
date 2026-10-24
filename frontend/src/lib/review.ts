import type { Editor } from '@tiptap/core'
import { closeHistory } from '@tiptap/pm/history'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { citeStore } from './citations'
import { paraKey, SEED_EVENT } from './writer'
import { containingSentence, sentenceHasCitation } from './sentenceSpans'
import { rememberProse, reviewDismissKey, rewriteTransaction, stableReviewAnnotations, stabilizeProse, textFingerprint, type ProseCache } from './reviewPolicy'

/** 后端 /api/assist/annotate 返回的批注 */
export interface CiteItem {
  paper_id: string | null
  bibkey: string
  title: string
  verdict: string
  reason?: string
  snippet?: string
  page?: number | null
}
export interface Candidate {
  paper_id: string
  bibkey: string
  title: string
  year?: number | null
  verdict: string
  reason?: string
  snippet?: string
  page?: number | null
}
export type Annotation =
  | { id: string; kind: 'cite_check'; anchor: string; items: CiteItem[] }
  | { id: string; kind: 'missing_citation'; anchor: string; reason: string; candidates: Candidate[] }
  | { id: string; kind: 'rewrite'; anchor: string; revised: string; reason: string }
  /** 审稿 Agent 的意见（不来自 annotate，由 critique 状态生成） */
  | { id: string; kind: 'reviewer'; anchor: string; issue: import('./critique').CritiqueIssue }

/** 定位到当前文档后的批注 */
export type PlacedAnnotation = Annotation & {
  key: string // 段落 hash + 批注 id，忽略列表按它记
  paraPos: number
  from: number
  to: number
  /** 句末标点的位置（插入引用时放在它前面） */
  insertAt: number
  /** cite_check：句子里对应的引用节点位置 */
  citePos: { pos: number; verdict: string }[]
  /** 是否需要在页边显示卡片（全部「支持」的引用核查只标记，不出卡片） */
  card: boolean
  dismissKey?: string
}

interface Para {
  pos: number
  marked: string // 引用写成 [@bibkey]
  plain: string // 去掉引用
  map: number[] // plain 下标 → 文档位置
  cites: { pos: number; ids: string[]; plainIdx: number }[]
  hash: string
}

/** 把文档里的段落序列化，同时记下纯文本每个字符对应的文档位置 */
export function collectParagraphs(editor: Editor, minLen = 30): Para[] {
  const papers = citeStore.get().papers
  const out: Para[] = []
  editor.state.doc.descendants((node, pos) => {
    if (['table', 'image', 'blockMath'].includes(node.type.name)) return false
    if (node.type.name !== 'paragraph') return true
    if (node.attrs.live) return false // Agent 正在流式写入的段落：写完再检查
    let hasMath = false
    node.descendants(child => { if (child.type.name === 'inlineMath') hasMath = true })
    if (hasMath) return false // Do not turn formula-bearing paragraphs into plain-text rewrites.
    let marked = ''
    let plain = ''
    const map: number[] = []
    const cites: Para['cites'] = []
    node.descendants((child, offset) => {
      const at = pos + 1 + offset
      if (child.isText) {
        const t = child.text ?? ''
        marked += t
        for (let i = 0; i < t.length; i++) map.push(at + i)
        plain += t
      } else if (child.type.name === 'citation') {
        const ids: string[] = child.attrs.ids ?? []
        marked += `[${ids.map((id) => '@' + (papers[id]?.bibkey ?? id)).join('; ')}]`
        cites.push({ pos: at, ids, plainIdx: plain.length })
      } else if (child.type.name === 'hardBreak') {
        marked += '\n'
        plain += '\n'
        map.push(at)
      }
      return false
    })
    const t = plain.trim()
    if (t.length >= minLen && !t.startsWith('写作要点')) out.push({ pos, marked, plain, map, cites, hash: textFingerprint(marked) })
    return false
  })
  return out
}

function place(p: Para, a: Annotation): PlacedAnnotation | null {
  const idx = p.plain.indexOf(a.anchor)
  if (idx < 0 || !a.anchor) return null
  const end = idx + a.anchor.length
  if (a.kind === 'missing_citation') {
    const sentence = containingSentence(p.plain, idx, end)
    if (!sentence || sentence.start !== idx || sentence.end !== end || sentenceHasCitation(sentence, p.cites)) return null
  }
  if (a.kind === 'rewrite') {
    const sentence = containingSentence(p.plain, idx, end)
    if (!sentence || sentenceHasCitation(sentence, p.cites)) return null
  }
  const from = p.map[idx]
  const to = p.map[end - 1] + 1
  const lastChar = p.plain[end - 1]
  const insertAt = /[。．.!?！？；;,，]/.test(lastChar) ? p.map[end - 1] : to
  const citePos: PlacedAnnotation['citePos'] = []
  let card = true
  if (a.kind === 'cite_check') {
    // 句子范围内（含句末标点后紧跟的）引用节点
    const near = p.cites.filter((c) => c.plainIdx >= idx && c.plainIdx <= end + 1)
    for (const it of a.items) {
      const c = near.find((x) => it.paper_id && x.ids.includes(it.paper_id)) ?? near.find((x) => x.ids.some((id) => citeStore.get().papers[id]?.bibkey === it.bibkey))
      if (c) citePos.push({ pos: c.pos, verdict: it.verdict })
    }
    card = a.items.some((it) => it.verdict !== 'supports')
  }
  return { ...a, key: `${p.hash}:${a.id}`, dismissKey: reviewDismissKey(a), paraPos: p.pos, from, to, insertAt, citePos, card }
}

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? (JSON.parse(v) as T) : fallback
  } catch {
    return fallback
  }
}
function save(key: string, v: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* 超出配额等情况忽略 */
  }
}

export interface ReviewState {
  placed: PlacedAnnotation[]
  pending: number
  total: number
  running: boolean
  reviewAll: () => void
  dismiss: (key: string) => void
  dismissMany: (keys: string[]) => void
  acceptRewrite: (annotation: PlacedAnnotation) => boolean
  checked: number
  paragraphCount: number
}

/**
 * 页边批注：编辑停顿后逐段调用 /api/assist/annotate。
 * - 结果按「段落内容 hash」缓存在 localStorage，段落不变就不会重复请求，刷新页面也能立即显示；
 * - 光标所在段落等用户离开后再检查，避免写到一半就被打扰；
 * - 段落一改，它的旧批注立即消失，等再次停顿后重新检查。
 */
export function useMarginReview(editor: Editor | null, docId: string, title: string, enabled: boolean): ReviewState {
  // Old results could label cited sentences/subclauses as missing citations.
  const cacheKey = `aria.review.v2.${docId}`
  const dismissKey = `aria.dismiss.${docId}`
  const proseKey = `aria.prose.${docId}`
  const initialProse = useMemo(() => load<ProseCache>(proseKey, {}), [proseKey])
  const prose = useRef(initialProse)
  const proseHydrated = useRef(false)
  const [cache, setCache] = useState<Record<string, Annotation[]>>(() => load(cacheKey, {}))
  const [dismissed, setDismissed] = useState<string[]>(() => load(dismissKey, []))
  const [paras, setParas] = useState<Para[]>([])
  const [queue, setQueue] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [total, setTotal] = useState(0)
  const inflight = useRef(new Set<string>())
  const timer = useRef<number | undefined>(undefined)
  const titleRef = useRef(title)
  titleRef.current = title

  // Preserve the current round when loading old caches, and migrate ignores
  // away from paragraph hashes so inserting a citation cannot undo an ignore.
  useEffect(() => {
    const migrated: string[] = []
    let remembered = false
    for (const paragraph of paras) {
      const annotations = cache[paragraph.hash]
      if (!annotations) continue
      if (!proseHydrated.current && !(textFingerprint(paragraph.plain) in prose.current)) {
        prose.current = stabilizeProse(prose.current, paragraph.plain, annotations).prose
        remembered = true
      }
      for (const annotation of annotations) {
        const stable = reviewDismissKey(annotation)
        if (dismissed.includes(`${paragraph.hash}:${annotation.id}`) && !dismissed.includes(stable)) migrated.push(stable)
      }
    }
    if (paras.length) proseHydrated.current = true
    if (remembered) save(proseKey, prose.current)
    if (migrated.length) setDismissed((current) => {
      const next = [...new Set([...current, ...migrated])].slice(-500)
      save(dismissKey, next)
      return next
    })
  }, [paras, cache, dismissed, proseKey, dismissKey])

  // 段落快照：文档变化后稍等再算（打字时不必每个字符都算）
  useEffect(() => {
    if (!editor) return
    let raf = 0
    const refresh = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => setParas(collectParagraphs(editor)))
    }
    refresh()
    editor.on('update', refresh)
    const unsub = citeStore.subscribe(refresh)
    return () => {
      editor.off('update', refresh)
      unsub()
      cancelAnimationFrame(raf)
    }
  }, [editor])

  // 分节写作写入的段落：后端已经核查过，直接用它的批注
  const seeds = useRef(new Map<string, Annotation[]>())
  useEffect(() => {
    const on = (e: Event) => {
      for (const it of (e as CustomEvent<{ key: string; annotations: Annotation[] }[]>).detail ?? []) seeds.current.set(it.key, it.annotations)
      if (editor) setParas(collectParagraphs(editor))
    }
    window.addEventListener(SEED_EVENT, on)
    return () => window.removeEventListener(SEED_EVENT, on)
  }, [editor])
  useEffect(() => {
    if (!seeds.current.size) return
    const add: Record<string, Annotation[]> = {}
    for (const p of paras) {
      const k = paraKey(p.plain)
      const a = seeds.current.get(k)
      if (a && !(p.hash in cache)) {
        const stable = stabilizeProse(prose.current, p.plain, a)
        prose.current = stable.prose
        save(proseKey, prose.current)
        add[p.hash] = stable.annotations
      }
      if (a) seeds.current.delete(k)
    }
    if (Object.keys(add).length)
      setCache((c) => {
        const next = { ...c, ...add }
        save(cacheKey, next)
        return next
      })
  }, [paras, cache, cacheKey, proseKey])

  // 停顿后把还没检查过的段落放进队列（跳过光标所在段落）
  const schedule = useCallback(
    (force = false) => {
      if (!editor || (!enabled && !force)) return
      const sel = editor.state.selection.from
      const current = collectParagraphs(editor)
      const todo = current
        .filter((p) => (force || !(p.hash in cache)) && !inflight.current.has(p.hash))
        .filter((p) => force || !editor.isFocused || sel < p.pos || sel > p.pos + p.marked.length + 2)
        .map((p) => p.hash)
      if (todo.length) {
        setQueue((q) => [...q, ...todo.filter((h) => !q.includes(h))])
        setTotal((t) => t + todo.length)
      }
    },
    [editor, enabled, cache],
  )

  useEffect(() => {
    if (!editor || !enabled) return
    const kick = () => {
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => schedule(false), 2500)
    }
    kick()
    editor.on('update', kick)
    editor.on('selectionUpdate', kick)
    editor.on('blur', kick)
    return () => {
      editor.off('update', kick)
      editor.off('selectionUpdate', kick)
      editor.off('blur', kick)
      window.clearTimeout(timer.current)
    }
  }, [editor, enabled, schedule])

  // 逐个处理队列（同时最多 2 个请求）
  useEffect(() => {
    if (!editor || !queue.length || inflight.current.size >= 2) return
    const hash = queue[0]
    setQueue((q) => q.slice(1))
    const p = collectParagraphs(editor).find((x) => x.hash === hash)
    if (!p) return
    inflight.current.add(hash)
    setRunning(true)
    api
      .annotate(p.marked, titleRef.current)
      .then((r) =>
        setCache((c) => {
          const stable = stabilizeProse(prose.current, p.plain, r.annotations)
          prose.current = stable.prose
          save(proseKey, prose.current)
          const next = { ...c, [hash]: stable.annotations }
          // 只保留当前文档里还存在的段落，防止缓存无限增长
          const live = new Set(collectParagraphs(editor).map((x) => x.hash))
          for (const k of Object.keys(next)) if (!live.has(k) && k !== hash) delete next[k]
          save(cacheKey, next)
          return next
        }),
      )
      .catch(() => undefined)
      .finally(() => {
        inflight.current.delete(hash)
        if (!inflight.current.size) setRunning(false)
        setQueue((q) => [...q]) // 触发下一个
      })
  }, [queue, editor, cacheKey, proseKey])

  useEffect(() => {
    if (!queue.length && !running) setTotal(0)
  }, [queue.length, running])

  const placed = useMemo(() => {
    const out: PlacedAnnotation[] = []
    for (const p of paras) {
      for (const a of stableReviewAnnotations(prose.current, p.plain, cache[p.hash] ?? [])) {
        const pa = place(p, a)
        if (pa && !dismissed.includes(pa.key) && !dismissed.includes(pa.dismissKey!)) out.push(pa)
      }
    }
    return out.sort((a, b) => a.from - b.from)
  }, [paras, cache, dismissed])

  const dismissMany = useCallback(
    (keys: string[]) =>
      setDismissed((d) => {
        const next = [...new Set([...d, ...keys])].slice(-500)
        save(dismissKey, next)
        return next
      }),
    [dismissKey],
  )

  const acceptRewrite = useCallback((annotation: PlacedAnnotation) => {
    if (!editor || editor.isDestroyed) return false
    const result = rewriteTransaction(editor.state, annotation)
    if (!result) return false
    // Accepting a suggestion completes this paragraph's automatic style round.
    // Another style pass requires an explicit full review or a later text edit.
    prose.current = rememberProse(prose.current, result.plainAfter, null)
    save(proseKey, prose.current)
    editor.view.dispatch(result.transaction)
    editor.view.dispatch(closeHistory(editor.state.tr))
    editor.commands.focus(undefined, { scrollIntoView: false })
    return true
  }, [editor, proseKey])

  return {
    placed,
    pending: queue.length + inflight.current.size,
    total,
    running,
    reviewAll: () => {
      // Only an explicit review starts another style round for unchanged text.
      for (const paragraph of editor ? collectParagraphs(editor) : []) delete prose.current[textFingerprint(paragraph.plain)]
      save(proseKey, prose.current)
      schedule(true)
    },
    dismiss: (key: string) => dismissMany([key]),
    dismissMany,
    acceptRewrite,
    checked: paras.filter((paragraph) => paragraph.hash in cache && !queue.includes(paragraph.hash) && !inflight.current.has(paragraph.hash)).length,
    paragraphCount: paras.length,
  }
}
