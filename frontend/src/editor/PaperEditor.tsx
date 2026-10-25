import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { EditorContent, useEditor, useEditorState } from '@tiptap/react'
import Highlight from '@tiptap/extension-highlight'
import Superscript from '@tiptap/extension-superscript'
import Subscript from '@tiptap/extension-subscript'
import { safeLink } from '../lib/formatting'
import { LinkEditor } from './FormatActions'
import StarterKit from '@tiptap/starter-kit'
import { CharacterCount, Placeholder } from '@tiptap/extensions'
import { Box, Button, CircularProgress, InputBase, Popper, Tooltip, Typography } from '@mui/material'
import CheckIcon from '@mui/icons-material/Check'
import CloudOffOutlinedIcon from '@mui/icons-material/CloudOffOutlined'
import { api } from '../lib/api'
import { useApp } from '../lib/appContext'
import { citeStore, collectCitationOrder } from '../lib/citations'
import { useMarginReview } from '../lib/review'
import { critiqueStore, useCritique, useCritiqueNotes } from '../lib/critique'
import { groupNotes, kindOf, loadFilter, MARGIN_KINDS, saveFilter, type MarginFilter } from '../lib/marginView'
import type { PlacedAnnotation } from '../lib/review'
import { brand } from '../theme'
import { Citation } from './extensions/Citation'
import { CitationSuggest } from './extensions/CitationSuggest'
import { GhostText } from './extensions/GhostText'
import { ReviewMarks } from './extensions/ReviewMarks'
import { DraftParagraph } from './extensions/DraftParagraph'
import { EditorToolbar } from './EditorToolbar'
import { FloatingToc } from './FloatingToc'
import { MarginNotes, NoteGroupCard } from './MarginNotes'
import { BatchCitationsDialog } from './BatchCitationsDialog'
import { References } from './References'
import { SelectionAssist } from './SelectionAssist'
import { useGhostCompletion } from './useGhostCompletion'
import { RichCodeBlock } from './extensions/RichCodeBlock'
import { richExtensions } from './extensions/RichNodes'
import { RichContentControls, type RichContentHandle } from './RichContentControls'

export type SaveState = 'saved' | 'saving' | 'dirty' | 'error'

function SaveBadge({ state }: { state: SaveState }) {
  const tip = { saved: '已保存', saving: '保存中…', dirty: '有未保存的修改', error: '保存失败，请检查后端' }[state]
  return (
    <Tooltip title={tip} placement="left">
      <Box
        sx={{
          width: 24, height: 24, borderRadius: '50%', display: 'grid', placeItems: 'center', flexShrink: 0,
          border: `2px solid ${state === 'error' ? brand.red : state === 'saved' ? brand.green : '#bdbdbd'}`,
          color: state === 'error' ? brand.red : brand.green, bgcolor: '#fff',
        }}
      >
        {state === 'saved' ? (
          <CheckIcon sx={{ fontSize: 14 }} />
        ) : state === 'error' ? (
          <CloudOffOutlinedIcon sx={{ fontSize: 13 }} />
        ) : (
          <CircularProgress size={12} thickness={5} sx={{ color: '#9e9e9e' }} />
        )}
      </Box>
    </Tooltip>
  )
}

function Counts({ editor }: { editor: any }) {
  const s = useEditorState({
    editor,
    selector: ({ editor: e }) => {
      let cites = 0
      e.state.doc.descendants((n: any) => {
        if (n.type.name === 'citation') cites++
      })
      return { chars: e.storage.characterCount?.characters?.() ?? 0, cites }
    },
  })
  return (
    <Typography sx={{ fontSize: 11.5, color: brand.faint, whiteSpace: 'nowrap' }}>
      {s.chars.toLocaleString()} 字 · {s.cites} 处引用
    </Typography>
  )
}

/** 根据可用宽度决定：目录（完整 / 短横线 / 隐藏）、页边批注列宽度、正文宽度 */
function useColumns(el: HTMLElement | null) {
  const [w, setW] = useState(1200)
  useEffect(() => {
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [el])
  // 宽屏显示完整目录；打开抽屉或窗口较窄时变成一列短横线（悬停预览标题和正文）
  const toc: 'full' | 'mini' = w >= 1240 ? 'full' : 'mini'
  const tocW = toc === 'full' ? 150 : 32
  const marginW = w >= 960 ? (w >= 1240 ? 300 : 260) : 0
  const gap = w >= 1240 ? 48 : w >= 760 ? 32 : 16
  // 分屏对照时正文区可能只有 500px 左右：收窄边距，正文列最少 300px
  const paperW = Math.max(300, Math.min(760, w - tocW - marginW - gap * 2 - (w >= 760 ? 48 : 20)))
  return { toc, tocW, marginW, paperW, gap }
}

export function PaperEditor({ docId }: { docId: string }) {
  const { setEditor, docTitle, setDocTitle, notify, copilot, review, bumpDocs, drawerOpen, setDrawerOpen } = useApp()
  const [loaded, setLoaded] = useState(false)
  const rich = useRef<RichContentHandle>(null)
  const [batchOpen, setBatchOpen] = useState(false)
  const [saveState, setSaveState] = useState<SaveState>('saved')
  const [active, setActive] = useState<string | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const cardsRef = useRef<PlacedAnnotation[]>([])
  const [filter, setFilterState] = useState<MarginFilter>(loadFilter)
  const setFilter = (f: MarginFilter) => {
    setFilterState(f)
    saveFilter(f)
  }
  const { responses } = useCritique()
  const isHandled = useCallback((a: PlacedAnnotation) => a.kind === 'reviewer' && !!responses[a.id], [responses])
  const lastFocus = useRef(0)
  const [scrollEl, setScrollEl] = useState<HTMLDivElement | null>(null)
  const marginRef = useRef<HTMLDivElement | null>(null)
  const saveTimer = useRef<number | undefined>(undefined)
  const titleTimer = useRef<number | undefined>(undefined)
  const loadingDoc = useRef(false)
  const cols = useColumns(scrollEl)

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ codeBlock: false, heading: { levels: [1, 2, 3] }, link: { openOnClick: false, isAllowedUri: url => !!safeLink(url) } }),
      Placeholder.configure({
        placeholder: ({ node }) =>
          node.type.name === 'heading' ? '章节标题' : node.type.name === 'paragraph' ? (node.attrs.live ? 'Agent 正在写作…' : '开始写作…  输入 @ 插入引用，选中文字让 AI 润色，停顿片刻会出现灰色续写建议（Tab 接受）') : '',
      }),
      RichCodeBlock, CharacterCount, Highlight.configure({multicolor:true}), Superscript, Subscript,
      Citation,
      CitationSuggest,
      GhostText,
      ReviewMarks,
      DraftParagraph,
      ...richExtensions({
        insert: (kind, range) => rich.current?.insert(kind, range), upload: (file, pos) => rich.current?.upload(file, pos),
        editImage: pos => rich.current?.editImage(pos), editMath: pos => rich.current?.editMath(pos),
        warn: message => notify(message, 'warning'),
      }),
    ],
    editorProps: { attributes: { class: 'paper-body', spellcheck: 'false' } },
    onUpdate: ({ editor: e, transaction }) => {
      citeStore.setOrder(collectCitationOrder(e.state.doc))
      if (loadingDoc.current || !transaction.docChanged) return
      setSaveState('dirty')
      window.clearTimeout(saveTimer.current)
      saveTimer.current = window.setTimeout(async () => {
        setSaveState('saving')
        try {
          await api.saveDoc(docId, { content: e.getJSON() })
          setSaveState('saved')
          bumpDocs()
        } catch {
          setSaveState('error')
        }
      }, 800)
    },
  })

  useEffect(() => {
    if (!editor) return
    let cancelled = false
    setLoaded(false)
    api
      .getDoc(docId)
      .then((d) => {
        if (cancelled) return
        loadingDoc.current = true
        editor.commands.setContent(d.content, { emitUpdate: true })
        loadingDoc.current = false
        citeStore.setOrder(collectCitationOrder(editor.state.doc))
        citeStore.setStyle(d.cite_style ?? 'numeric')
        setDocTitle(d.title)
        setSaveState('saved')
        setLoaded(true)
      })
      .catch((e) => notify(`加载文档失败：${e.message}`, 'error'))
    return () => {
      cancelled = true
      window.clearTimeout(saveTimer.current)
    }
  }, [editor, docId]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setEditor(editor)
    return () => setEditor(null)
  }, [editor, setEditor])

  const { loading } = useGhostCompletion(editor, copilot, docTitle)
  const rv = useMarginReview(editor, docId, docTitle, review && loaded)
  const crit = useCritiqueNotes(editor)
  // 审稿意见不受「页边建议」开关影响：它是你主动发起的
  const allNotes = useMemo(() => [...(review ? rv.placed : []), ...crit].sort((a, b) => a.from - b.from), [review, rv.placed, crit])
  // 按类型筛选（底部状态条上的开关），正文标记和页边卡片一起筛
  const notes = useMemo(() => allNotes.filter((n) => filter[kindOf(n)]), [allNotes, filter])
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const n of allNotes) if (n.card) c[kindOf(n)] = (c[kindOf(n)] ?? 0) + 1
    return c
  }, [allNotes])

  // Agent 面板里点了某条审稿意见：高亮并滚动到原文
  const critRef = useRef(crit)
  critRef.current = crit
  useEffect(
    () =>
      critiqueStore.subscribe(() => {
        const f = critiqueStore.get().focus
        if (!f || !editor || !scrollEl || f.nonce === lastFocus.current) return
        lastFocus.current = f.nonce
        const n = critRef.current.find((x) => x.id === f.id)
        if (!n) return
        setActive(n.key)
        try {
          const top = editor.view.coordsAtPos(n.from).top - scrollEl.getBoundingClientRect().top + scrollEl.scrollTop - 140
          scrollEl.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
        } catch {
          /* 位置失效时忽略 */
        }
      }),
    [editor, scrollEl],
  )
  const cards = useMemo(() => notes.filter((n) => n.card), [notes])
  cardsRef.current = cards
  const activeGroup = useMemo(() => groupNotes(cards).find((g) => g.notes.some((n) => n.key === active)) ?? null, [cards, active])
  const activeRef = useRef(active)
  activeRef.current = active

  // 正文里的标记与页边卡片同步
  useEffect(() => {
    if (!editor) return
    // 放到下一帧再派发：在 React 提交阶段直接改编辑器会让引用节点的 NodeView 同步重渲染（flushSync 警告）
    const id = requestAnimationFrame(() => {
      if (editor.isDestroyed) return
      editor.commands.setReviewMarks(
        notes.map((n) => ({
          key: n.key, kind: n.kind, from: n.from, to: n.to, citePos: n.citePos,
          cls: n.kind === 'reviewer' ? `rv-critic-${n.issue.severity}` : undefined,
        })),
        hover ?? active,
      )
    })
    return () => cancelAnimationFrame(id)
  }, [editor, notes, active, hover])

  // 光标进入某句有批注的话：展开它的批注（像 Google Docs 的评论）
  useEffect(() => {
    if (!editor) return
    const onSel = () => {
      const { from, empty } = editor.state.selection
      if (!empty) return
      const hit = cardsRef.current.find((n) => from >= n.from && from <= n.to)
      // 光标离开批注的句子就收起（与 Google Docs 一致）；还在同一句里就保持当前展开的那条
      setActive((a) => (hit ? (a && cardsRef.current.find((n) => n.key === a && from >= n.from && from <= n.to) ? a : hit.key) : null))
    }
    editor.on('selectionUpdate', onSel)
    return () => {
      editor.off('selectionUpdate', onSel)
    }
  }, [editor])

  // Alt+↓ / Alt+↑：跳到下一条 / 上一条批注
  useEffect(() => {
    if (!editor || !scrollEl) return
    const onKey = (e: KeyboardEvent) => {
      if (!e.altKey || (e.key !== 'ArrowDown' && e.key !== 'ArrowUp')) return
      const groups = groupNotes(cardsRef.current)
      if (!groups.length) return
      e.preventDefault()
      const cur = groups.findIndex((g) => g.notes.some((n) => n.key === activeRef.current))
      const sel = editor.state.selection.from
      let next: number
      if (cur >= 0) next = e.key === 'ArrowDown' ? Math.min(groups.length - 1, cur + 1) : Math.max(0, cur - 1)
      else if (e.key === 'ArrowDown') next = Math.max(0, groups.findIndex((g) => g.from > sel))
      else next = Math.max(0, groups.map((g) => g.from < sel).lastIndexOf(true))
      const g = groups[next]
      setActive(g.notes[0].key)
      try {
        const top = editor.view.coordsAtPos(g.from).top - scrollEl.getBoundingClientRect().top + scrollEl.scrollTop - scrollEl.clientHeight / 3
        scrollEl.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
      } catch {
        /* 位置失效时忽略 */
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [editor, scrollEl])

  // 点击正文里的波浪线 → 高亮对应卡片
  useEffect(() => {
    if (!editor) return
    const dom = editor.view.dom
    const onClick = (e: MouseEvent) => {
      const el = (e.target as HTMLElement).closest('[data-ann]') as HTMLElement | null
      if (el?.dataset.ann) setActive(el.dataset.ann) // 没点在标记上时交给光标位置决定
    }
    dom.addEventListener('click', onClick)
    return () => dom.removeEventListener('click', onClick)
  }, [editor])

  const changeTitle = (t: string) => {
    setDocTitle(t)
    setSaveState('dirty')
    window.clearTimeout(titleTimer.current)
    titleTimer.current = window.setTimeout(async () => {
      try {
        await api.saveDoc(docId, { title: t.trim() || '未命名论文' })
        setSaveState('saved')
        bumpDocs()
      } catch {
        setSaveState('error')
      }
    }, 600)
  }

  if (!editor) return null

  return (
    <Box
      sx={{
        height: '100%', border: `1px solid ${brand.line}`, borderRadius: '12px', bgcolor: '#fff', position: 'relative',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}
    >
      <LinkEditor editor={editor} />
      <EditorToolbar editor={editor} onInsert={kind => rich.current?.insert(kind)} />
      <RichContentControls key={docId} ref={rich} editor={editor} docId={docId} />

      <Box ref={setScrollEl} sx={{ flex: 1, overflow: 'auto' }} className="editor-scroll">
        <Box
          sx={{
            display: 'grid', justifyContent: 'center', columnGap: `${cols.gap}px`, px: cols.gap >= 32 ? 3 : 1,
            gridTemplateColumns: [cols.tocW ? `${cols.tocW}px` : null, `${cols.paperW}px`, cols.marginW ? `${cols.marginW}px` : null]
              .filter(Boolean)
              .join(' '),
          }}
        >
          <Box>
            <Box sx={{ position: 'sticky', top: 0, pt: 12, zIndex: 5 }}>
              <FloatingToc editor={editor} mode={cols.toc} scrollEl={scrollEl} />
            </Box>
          </Box>
          <Box className="paper-col">
            <InputBase
              value={docTitle}
              onChange={(e) => changeTitle(e.target.value)}
              placeholder="论文标题"
              multiline
              fullWidth
              sx={{ fontSize: 30, fontWeight: 700, lineHeight: 1.3, color: brand.ink, mb: 1.5, p: 0, letterSpacing: -0.2 }}
            />
            {!loaded && <CircularProgress size={18} sx={{ my: 3 }} />}
            <EditorContent editor={editor} />
            <References />
          </Box>
          {cols.marginW > 0 && (
            <Box sx={{ pt: '28px', pb: 12 }}>
              <MarginNotes
                editor={editor}
                notes={cards}
                active={active}
                setActive={setActive}
                onHover={setHover}
                onDismiss={rv.dismiss}
                onAcceptRewrite={(annotation) => {
                  if (rv.acceptRewrite(annotation)) notify('已接受精简；本段不会自动接着生成新的精简建议，需要时可主动检查全文', 'success')
                  else notify('原句已变化或包含引用，请重新审阅', 'warning')
                }}
                columnRef={marginRef}
                isHandled={isHandled}
              />
            </Box>
          )}
        </Box>
      </Box>

      {/* 底部状态条 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, px: 2, height: 38, borderTop: `1px solid ${brand.lineSoft}`, flexShrink: 0 }}>
        <Counts editor={editor} />
        {review && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, overflowX: 'auto' }}>
            {rv.pending > 0 ? (
              <Typography sx={{ fontSize: 11.5, color: brand.faint, display: 'flex', alignItems: 'center', gap: 0.75, whiteSpace: 'nowrap' }}>
                <CircularProgress size={10} sx={{ color: brand.faint }} />
                检查中 · 已覆盖 {rv.checked}/{rv.paragraphCount} 段，待完成 {rv.pending} 段
              </Typography>
            ) : (
              <Tooltip title={cols.marginW ? 'Alt+↓ / Alt+↑ 逐条查看' : drawerOpen ? '收起右侧抽屉以显示页边建议' : ''}>
                <Typography
                  onClick={() => !cols.marginW && drawerOpen && setDrawerOpen(false)}
                  sx={{ fontSize: 11.5, color: cards.length ? brand.ink : brand.faint, cursor: !cols.marginW && drawerOpen ? 'pointer' : 'default', whiteSpace: 'nowrap' }}
                >
                  {rv.checked < rv.paragraphCount ? `已检查 ${rv.checked}/${rv.paragraphCount} 段 · 当前建议 ${cards.length}` : allNotes.some((n) => n.card) ? `页边建议 ${cards.length}` : '页边暂无建议'}
                  {!cols.marginW && cards.length ? '（页边放不下：点正文里的标记查看）' : ''}
                </Typography>
              </Tooltip>
            )}
            {MARGIN_KINDS.filter((k) => counts[k.key]).map((k) => (
              <Tooltip key={k.key} title={filter[k.key] ? `隐藏「${k.label}」` : `显示「${k.label}」`}>
                <Box
                  role="button"
                  aria-pressed={filter[k.key]}
                  onClick={() => setFilter({ ...filter, [k.key]: !filter[k.key] })}
                  sx={{
                    display: 'inline-flex', alignItems: 'center', gap: 0.5, px: 0.75, height: 20, borderRadius: '10px', cursor: 'pointer', fontSize: 11, whiteSpace: 'nowrap',
                    color: filter[k.key] ? brand.ink : brand.faint, bgcolor: filter[k.key] ? 'rgba(0,0,0,0.05)' : 'transparent',
                    border: `1px solid ${filter[k.key] ? 'transparent' : brand.line}`, '&:hover': { bgcolor: 'rgba(0,0,0,0.08)' },
                  }}
                >
                  <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: filter[k.key] ? k.color : brand.faint }} />
                  {k.label} {counts[k.key]}
                </Box>
              </Tooltip>
            ))}
            <Button size="small" title="重新检查全文，并主动开启新一轮精简审阅" onClick={rv.reviewAll} disabled={rv.pending > 0} sx={{ fontSize: 11.5, py: 0, minHeight: 22, whiteSpace: 'nowrap', flexShrink: 0 }}>
              检查全文
            </Button>
            <Button size="small" onClick={() => setBatchOpen(true)} disabled={!rv.placed.some((n) => n.kind === 'missing_citation' || (n.kind === 'cite_check' && n.card))}
              sx={{ fontSize: 11.5, py: 0, minHeight: 22, whiteSpace: 'nowrap', flexShrink: 0 }}>
              批量处理引用
            </Button>
          </Box>
        )}
        <Box sx={{ flex: 1 }} />
        {loading && (
          <Typography sx={{ fontSize: 11.5, color: brand.faint, display: 'flex', alignItems: 'center', gap: 0.75, whiteSpace: 'nowrap' }}>
            <CircularProgress size={10} sx={{ color: brand.faint }} /> 正在生成续写建议
          </Typography>
        )}
        <SaveBadge state={saveState} />
      </Box>
      <SelectionAssist editor={editor} />
      {batchOpen && <BatchCitationsDialog editor={editor} notes={rv.placed.filter((n) => n.card)} coverage={{ checked: rv.checked, total: rv.paragraphCount, pending: rv.pending }} onClose={() => setBatchOpen(false)} onDismissMany={rv.dismissMany} />}
      {/* 页边放不下批注时（比如分屏对照）：正文里只留标记，点标记或光标进入那句话，批注浮在句子下方 */}
      {!cols.marginW && activeGroup && (
        <Popper
          open
          placement="bottom-start"
          anchorEl={{
            // contextElement：让 Popper 跟着正文的滚动容器一起滚
            contextElement: editor.view.dom,
            getBoundingClientRect: () => {
              try {
                const a = editor.view.coordsAtPos(activeGroup.from)
                return new DOMRect(a.left, a.top, 1, a.bottom - a.top)
              } catch {
                return new DOMRect(0, 0, 0, 0)
              }
            },
          }}
          modifiers={[{ name: 'offset', options: { offset: [0, 6] } }, { name: 'preventOverflow', options: { padding: 12 } }]}
          sx={{ zIndex: 20, width: 340 }}
        >
          <NoteGroupCard editor={editor} notes={activeGroup.notes} onClose={() => setActive(null)} onHover={setHover} onDismiss={rv.dismiss}
            onAcceptRewrite={(annotation) => {
              if (rv.acceptRewrite(annotation)) notify('已接受精简；本段不会自动接着生成新的精简建议，需要时可主动检查全文', 'success')
              else notify('原句已变化或包含引用，请重新审阅', 'warning')
            }} />
        </Popper>
      )}
    </Box>
  )
}
