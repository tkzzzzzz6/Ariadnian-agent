import type { Editor } from '@tiptap/core'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { Box, Button, ButtonBase, IconButton, Tooltip, Typography } from '@mui/material'
import UnfoldMoreIcon from '@mui/icons-material/UnfoldMore'
import CloseIcon from '@mui/icons-material/Close'
import PictureAsPdfOutlinedIcon from '@mui/icons-material/PictureAsPdfOutlined'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import FormatQuoteIcon from '@mui/icons-material/FormatQuote'
import ReportProblemOutlinedIcon from '@mui/icons-material/ReportProblemOutlined'
import { useApp } from '../lib/appContext'
import { authorYearLabel, citeStore } from '../lib/citations'
import type { PlacedAnnotation } from '../lib/review'
import { VerdictChip } from '../panels/CitePanel'
import { ReadSourceButton } from '../panels/ReaderPanel'
import { brand, card, shadow } from '../theme'
import { DiffView } from './DiffView'
import { IssueBody, IssueHeader, severityColor } from '../panels/CritiqueIssue'
import { CATEGORY_LABEL } from '../lib/critique'
import { groupNotes, summaryOf, type NoteGroup } from '../lib/marginView'

const GAP = 8

function Quote({ text }: { text: string }) {
  return (
    <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
      “{text}”
    </Typography>
  )
}

function Actions({ children }: { children: React.ReactNode }) {
  return <Box sx={{ display: 'flex', gap: 0.5, mt: 1, alignItems: 'center', '& .MuiButton-root': { py: 0.1, px: 1, fontSize: 12 } }}>{children}</Box>
}

function NoteCard({
  a, editor, first, onClose, onHover, onDismiss, onAcceptRewrite,
}: {
  a: PlacedAnnotation
  editor: Editor
  /** 组内第一条：右上角放「收起」 */
  first: boolean
  onClose?: () => void
  onHover: (k: string | null) => void
  onDismiss: () => void
  onAcceptRewrite: (annotation: PlacedAnnotation) => void
}) {
  const { requestCitations, notify, openReader } = useApp()
  const papers = citeStore.get().papers
  const insertCite = (paperId: string) => {
    editor.chain().focus().insertContentAt(a.insertAt, { type: 'citation', attrs: { ids: [paperId] } }).run()
    notify('已插入引用', 'success')
  }
  const kind = a.kind === 'missing_citation' ? card.agent : a.kind === 'reviewer' ? { bgcolor: '#fff', border: `1px solid ${brand.line}` } : card.plain
  const accent =
    a.kind === 'reviewer' ? severityColor(a.issue.severity) : a.kind === 'missing_citation' ? brand.purple : a.kind === 'rewrite' ? brand.blue : a.kind === 'cite_check' && a.items.some((i) => i.verdict === 'unrelated' || i.verdict === 'missing') ? brand.red : brand.amber

  return (
    <Box
      onMouseEnter={() => onHover(a.key)}
      onMouseLeave={() => onHover(null)}
      sx={{
        ...kind, border: 'none', borderLeft: `3px solid ${accent}`, borderRadius: first ? '8px 8px 0 0' : 0, px: 1.25, pr: first ? 3.5 : 1.25, py: 1, fontSize: 12.5,
        bgcolor: kind.bgcolor, position: 'relative',
      }}
    >
      {first && onClose && (
        <Tooltip title="收起">
          <IconButton aria-label="收起" onClick={onClose} sx={{ position: 'absolute', top: 4, right: 4, p: 0.25, color: brand.faint, zIndex: 1 }}>
            <CloseIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Tooltip>
      )}
      {a.kind === 'missing_citation' && (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <FormatQuoteIcon sx={{ fontSize: 15, color: brand.purple }} />
            <Typography sx={{ fontSize: 12.5, fontWeight: 500, color: brand.purple }}>这句话缺少引用</Typography>
          </Box>
          <Quote text={a.anchor} />
          {a.candidates.length ? (
            a.candidates.map((c) => (
              <Box key={c.paper_id} sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mt: 0.75 }}>
                <VerdictChip v={c.verdict} />
                <Typography noWrap sx={{ fontSize: 12, flex: 1, minWidth: 0 }} title={c.title}>
                  {papers[c.paper_id] ? authorYearLabel(papers[c.paper_id]) : c.title}
                </Typography>
                <Tooltip title={c.page ? `看原文 p.${c.page}` : '看原文'}>
                  <IconButton
                    aria-label="看原文"
                    onClick={() =>
                      openReader({
                        paperId: c.paper_id, claim: a.anchor, insertAt: a.insertAt,
                        evidences: [{ page: c.page, snippet: c.snippet, verdict: c.verdict, reason: c.reason }],
                      })
                    }
                    sx={{ p: 0.25, color: brand.blue }}
                  >
                    <PictureAsPdfOutlinedIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
                <Button size="small" variant="contained" onClick={() => insertCite(c.paper_id)} sx={{ py: 0, minHeight: 22, fontSize: 11.5 }}>
                  插入
                </Button>
              </Box>
            ))
          ) : (
            <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.5 }}>文献库里没有找到能支撑它的文献</Typography>
          )}
          <Actions>
            {!a.candidates.length && <Button onClick={() => requestCitations(a.anchor, a.insertAt)}>去检索</Button>}
            <Button onClick={onDismiss} sx={{ color: brand.muted }}>
              忽略
            </Button>
          </Actions>
        </>
      )}

      {a.kind === 'cite_check' && (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <ReportProblemOutlinedIcon sx={{ fontSize: 15, color: accent }} />
            <Typography sx={{ fontSize: 12.5, fontWeight: 500, color: accent }}>引用可能撑不起这句话</Typography>
          </Box>
          <Quote text={a.anchor} />
          {a.items
            .filter((i) => i.verdict !== 'supports')
            .map((i) => (
              <Box key={i.bibkey} sx={{ mt: 0.75 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                  <VerdictChip v={i.verdict} />
                  <Typography noWrap sx={{ fontSize: 12, fontWeight: 500 }} title={i.title}>
                    {i.paper_id && papers[i.paper_id] ? authorYearLabel(papers[i.paper_id]) : i.bibkey}
                  </Typography>
                </Box>
                {i.reason && <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25 }}>{i.reason}</Typography>}
                {i.paper_id && (i.snippet || i.page) && (
                  <Box sx={{ mt: 0.5 }}>
                    <ReadSourceButton
                      paperId={i.paper_id}
                      page={i.page}
                      onClick={() =>
                        openReader({
                          paperId: i.paper_id!, claim: a.anchor,
                          evidences: [{ page: i.page, snippet: i.snippet, verdict: i.verdict, reason: i.reason }],
                        })
                      }
                    />
                  </Box>
                )}
              </Box>
            ))}
          <Actions>
            <Button onClick={() => requestCitations(a.anchor, a.insertAt)}>换一篇</Button>
            <Button onClick={onDismiss} sx={{ color: brand.muted }}>
              忽略
            </Button>
          </Actions>
        </>
      )}

      {a.kind === 'reviewer' && (
        <>
          <IssueHeader issue={a.issue} />
          {a.anchor ? <Quote text={a.anchor} /> : <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25 }}>针对《{a.issue.section}》整节</Typography>}
          <IssueBody issue={a.issue} insertAt={a.anchor ? a.insertAt : undefined} />
        </>
      )}

      {a.kind === 'rewrite' && (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <AutoFixHighIcon sx={{ fontSize: 15, color: brand.blue }} />
            <Typography sx={{ fontSize: 12.5, fontWeight: 500 }}>可以更简洁</Typography>
            {a.reason && <Typography sx={{ fontSize: 11.5, color: brand.muted, ml: 0.5 }}>· {a.reason}</Typography>}
          </Box>
          <Box sx={{ mt: 0.5, '& > div': { fontSize: '12.5px !important', lineHeight: '1.65 !important' } }}>
            <DiffView before={a.anchor} after={a.revised} />
          </Box>
          <Actions>
            <Button
              variant="contained"
              onClick={() => onAcceptRewrite(a)}
            >
              接受
            </Button>
            <Button onClick={onDismiss} sx={{ color: brand.muted }}>
              忽略
            </Button>
          </Actions>
        </>
      )}
    </Box>
  )
}

/** 展开的一组批注（页边栏里，或页边空间不够时浮在句子下方） */
export function NoteGroupCard({
  editor, notes, onClose, onHover, onDismiss, onAcceptRewrite, sx,
}: {
  editor: Editor
  notes: PlacedAnnotation[]
  onClose?: () => void
  onHover: (k: string | null) => void
  onDismiss: (k: string) => void
  onAcceptRewrite: (annotation: PlacedAnnotation) => void
  sx?: object
}) {
  return (
    <Box sx={{ borderRadius: '8px', boxShadow: shadow.lg, bgcolor: '#fff', transition: 'transform .15s', overflow: 'hidden', ...sx }}>
      {notes.map((n, i) => (
        <Box key={n.key} sx={{ borderTop: i ? `1px solid ${brand.lineSoft}` : 'none' }}>
          <NoteCard a={n} editor={editor} first={i === 0} onClose={onClose} onHover={onHover} onDismiss={() => onDismiss(n.dismissKey ?? n.key)} onAcceptRewrite={onAcceptRewrite} />
        </Box>
      ))}
    </Box>
  )
}

const COMPACT_H = 34
/** 卡片离原句超过这个距离，就收进「还有 N 条」里，免得卡片漂到别的段落旁边 */
const MAX_DRIFT = 220

/** 收起状态：一行，标出类型、条数和原句开头 */
function CompactCard({ g, handled, onOpen, onHover }: { g: NoteGroup; handled: boolean; onOpen: () => void; onHover: (k: string | null) => void }) {
  const top = g.notes[0]
  const { label, color } = summaryOf(top, (c) => CATEGORY_LABEL[c as keyof typeof CATEGORY_LABEL] ?? c)
  const quote = top.anchor || (top.kind === 'reviewer' ? `《${top.issue.section}》整节` : '')
  const kinds = [...new Set(g.notes.map((n) => summaryOf(n, (c) => c).color))]
  return (
    <Box
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onOpen()}
      onMouseEnter={() => onHover(top.key)}
      onMouseLeave={() => onHover(null)}
      sx={{
        height: COMPACT_H - 4, display: 'flex', alignItems: 'center', gap: 0.75, px: 1, borderRadius: '6px', cursor: 'pointer',
        bgcolor: '#fff', border: `1px solid ${brand.lineSoft}`, borderLeft: `3px solid ${color}`, opacity: handled ? 0.55 : 1,
        transition: 'background .12s, border-color .12s', '&:hover': { bgcolor: '#fafafa', borderColor: brand.line, borderLeftColor: color },
      }}
    >
      <Box sx={{ display: 'flex', gap: '3px', flexShrink: 0 }}>
        {kinds.map((c) => (
          <Box key={c} sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: c }} />
        ))}
      </Box>
      <Typography sx={{ fontSize: 12, fontWeight: 500, color, flexShrink: 0, whiteSpace: 'nowrap' }}>
        {handled ? '已处理' : label}
      </Typography>
      <Typography noWrap sx={{ fontSize: 11.5, color: brand.faint, flex: 1, minWidth: 0 }}>
        {quote}
      </Typography>
      {g.notes.length > 1 && (
        <Box sx={{ fontSize: 10.5, color: brand.muted, bgcolor: 'rgba(0,0,0,0.06)', borderRadius: '8px', px: 0.6, lineHeight: '16px', flexShrink: 0 }}>{g.notes.length}</Box>
      )}
    </Box>
  )
}

interface Layout {
  tops: Record<string, number>
  /** 离原句太远、收起来的若干组：在第一组的位置显示「还有 N 条」 */
  clusters: { key: string; top: number; groups: string[] }[]
}

/**
 * 页边批注列（参考 Google Docs 的评论栏）：
 * - 平时每组只占一行，点一下（或光标进入那句话）才展开，同时只展开一组；
 * - 展开的那组与原句顶端对齐，其余的从它向上、向下依次让开；
 * - 离原句太远的收进「还有 N 条」，点开再看。
 */
export function MarginNotes({
  editor, notes, active, setActive, onHover, onDismiss, onAcceptRewrite, columnRef, isHandled, expandAll = false, relayoutKey,
}: {
  /** 外观设置「批注默认全部展开」：每组都显示完整卡片，按 Google Docs 评论栏的方式依次让开 */
  expandAll?: boolean
  /** 正文字号、行距等变化时重新对齐（这些变化不会触发编辑器 update） */
  relayoutKey?: string
  editor: Editor
  notes: PlacedAnnotation[]
  active: string | null
  setActive: (k: string | null) => void
  onHover: (k: string | null) => void
  onDismiss: (k: string) => void
  onAcceptRewrite: (annotation: PlacedAnnotation) => void
  columnRef: RefObject<HTMLDivElement | null>
  isHandled: (a: PlacedAnnotation) => boolean
}) {
  const refs = useRef(new Map<string, HTMLDivElement>())
  const [layout, setLayout] = useState<Layout>({ tops: {}, clusters: [] })
  const [tick, setTick] = useState(0)
  const groups = useMemo(() => groupNotes(notes, isHandled), [notes, isHandled])
  const activeGroup = groups.find((g) => g.notes.some((n) => n.key === active))?.key ?? null
  const isOpen = (key: string) => expandAll || key === activeGroup

  useEffect(() => {
    // 字体变化后等浏览器重新排版完再量
    const t = window.setTimeout(() => setTick((x) => x + 1), 60)
    return () => window.clearTimeout(t)
  }, [relayoutKey, expandAll])

  useLayoutEffect(() => {
    const bump = () => setTick((t) => t + 1)
    editor.on('update', bump)
    const ro = new ResizeObserver(bump)
    if (columnRef.current) ro.observe(columnRef.current)
    return () => {
      editor.off('update', bump)
      ro.disconnect()
    }
  }, [editor, columnRef])

  useLayoutEffect(() => {
    const col = columnRef.current
    if (!col) return
    const base = col.getBoundingClientRect().top
    const items: { key: string; want: number; h: number }[] = []
    for (const g of groups) {
      let want = 0
      try {
        want = editor.view.coordsAtPos(Math.min(g.from, editor.state.doc.content.size)).top - base - 4
      } catch {
        continue
      }
      const measured = refs.current.get(g.key)?.offsetHeight
      items.push({ key: g.key, want, h: isOpen(g.key) ? measured ?? 220 : COMPACT_H })
    }
    const place = (list: { key: string; want: number; h: number }[]): number[] => {
      const tops = list.map((x) => x.want)
      const a = list.findIndex((x) => x.key === activeGroup)
      const start = a >= 0 ? a : 0
      // 从展开的那组（没有就从第一组）向下依次让开
      for (let i = start + 1; i < list.length; i++) tops[i] = Math.max(list[i].want, tops[i - 1] + list[i - 1].h + GAP)
      if (a < 0) for (let i = 1; i < list.length; i++) tops[i] = Math.max(list[i].want, tops[i - 1] + list[i - 1].h + GAP)
      // 向上让开
      for (let i = start - 1; i >= 0; i--) tops[i] = Math.min(list[i].want, tops[i + 1] - list[i].h - GAP)
      const min = Math.min(0, ...tops)
      return min < 0 ? tops.map((t) => t - min) : tops
    }
    // 第一遍：找出漂得太远的组；第二遍：把相邻的这些组收成一个「还有 N 条」
    const first = place(items)
    const ai = items.findIndex((x) => x.key === activeGroup)
    const merged: { key: string; want: number; h: number; members: string[] }[] = []
    items.forEach((x, i) => {
      // 展开的那组把下面的卡片推开是正常的，这部分距离不算「漂远」
      const allowance = ai >= 0 && i > ai ? items[ai].h : 0
      // 全部展开时不收起：卡片依次往下排（和 Google Docs 一样），离原句远一些也保留
      const far = !expandAll && x.key !== activeGroup && first[i] - x.want > MAX_DRIFT + allowance
      const last = merged[merged.length - 1]
      if (far && last && last.members.length && last.key.startsWith('cluster:')) last.members.push(x.key)
      else if (far) merged.push({ key: `cluster:${x.key}`, want: x.want, h: 28, members: [x.key] })
      else merged.push({ ...x, members: [] })
    })
    const second = place(merged)
    const next: Layout = { tops: {}, clusters: [] }
    merged.forEach((m, i) => {
      if (m.key.startsWith('cluster:')) next.clusters.push({ key: m.key, top: second[i], groups: m.members })
      else next.tops[m.key] = second[i]
    })
    const same =
      Object.keys(next.tops).length === Object.keys(layout.tops).length &&
      Object.entries(next.tops).every(([k, v]) => Math.abs((layout.tops[k] ?? -1) - v) < 0.5) &&
      next.clusters.length === layout.clusters.length &&
      next.clusters.every((c, i) => c.key === layout.clusters[i].key && Math.abs(c.top - layout.clusters[i].top) < 0.5 && c.groups.length === layout.clusters[i].groups.length)
    if (!same) setLayout(next)
  }, [groups, activeGroup, tick, editor, columnRef, layout, expandAll]) // eslint-disable-line react-hooks/exhaustive-deps

  const hidden = new Set(layout.clusters.flatMap((c) => c.groups))
  return (
    <Box ref={columnRef} sx={{ position: 'relative', minHeight: '100%' }}>
      {groups.map((g) => {
        if (hidden.has(g.key)) return null
        const open = isOpen(g.key)
        return (
          <Box
            key={g.key}
            ref={(el: HTMLDivElement | null) => {
              if (el) refs.current.set(g.key, el)
              else refs.current.delete(g.key)
            }}
            sx={{
              position: 'absolute', left: 0, right: 0, top: layout.tops[g.key] ?? 0, opacity: g.key in layout.tops ? 1 : 0, zIndex: open ? 2 : 1,
              transition: 'top .2s ease, opacity .2s',
            }}
          >
            {open ? (
              <NoteGroupCard
                editor={editor} notes={g.notes} onClose={expandAll ? undefined : () => setActive(null)} onHover={onHover} onDismiss={onDismiss} onAcceptRewrite={onAcceptRewrite}
                sx={g.key === activeGroup ? { transform: 'translateX(-4px)' } : expandAll ? { boxShadow: 'none', border: `1px solid ${brand.lineSoft}` } : undefined}
              />
            ) : (
              <CompactCard g={g} handled={g.notes.every(isHandled)} onOpen={() => setActive(g.notes[0].key)} onHover={onHover} />
            )}
          </Box>
        )
      })}
      {layout.clusters.map((c) => (
        <Box key={c.key} sx={{ position: 'absolute', left: 0, right: 0, top: c.top, transition: 'top .2s ease' }}>
          <ButtonBase
            onClick={() => {
              const g = groups.find((x) => x.key === c.groups[0])
              if (g) setActive(g.notes[0].key)
            }}
            sx={{ width: '100%', height: 24, justifyContent: 'flex-start', gap: 0.5, px: 1, borderRadius: '12px', fontSize: 11.5, color: brand.muted, border: `1px dashed ${brand.line}`, bgcolor: '#fff', '&:hover': { color: brand.ink, borderColor: brand.faint } }}
          >
            <UnfoldMoreIcon sx={{ fontSize: 14 }} />
            还有 {c.groups.reduce((a, k) => a + (groups.find((g) => g.key === k)?.notes.length ?? 0), 0)} 条建议
          </ButtonBase>
        </Box>
      ))}
    </Box>
  )
}
