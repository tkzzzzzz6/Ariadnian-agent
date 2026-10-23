/**
 * 抽屉「原文」：在写作界面旁边读文献库里的论文 PDF。
 * 从页边批注、找引用或文献库打开时，会跳到依据所在的页，并高亮与正文那句话最相关的原文。
 */
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Box, Button, CircularProgress, IconButton, Tooltip, Typography } from '@mui/material'
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import FormatQuoteIcon from '@mui/icons-material/FormatQuote'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import PictureAsPdfOutlinedIcon from '@mui/icons-material/PictureAsPdfOutlined'
import { api, pdfFileUrl } from '../lib/api'
import { useApp, type Evidence, type ReaderTarget } from '../lib/appContext'
import { authorYearLabel, citeStore, formatInline } from '../lib/citations'
import type { Paper, PdfHighlight } from '../lib/types'
import { NOTE_COLORS, notesStore, useNotes, type Note } from '../lib/notes'
import { NotePopover, NotesList, SelectionMenu } from '../pdf/NotesUI'
// pdf.js 体积较大，用到时才加载
const PdfViewer = lazy(() => import('../pdf/PdfViewer').then((m) => ({ default: m.PdfViewer })))
import { Card } from '../components/Thread'
import { brand } from '../theme'
import { VerdictChip } from './CitePanel'
import { ReadingDialog } from '../components/ReadingDialog'
import { READING_LABELS, pageProgressLabel } from '../lib/reading'
import { useReading, readingStore } from '../lib/readingStore'

export function ReaderPanel({ target, variant = 'drawer' }: { target?: ReaderTarget | null; variant?: 'drawer' | 'split' } = {}) {
  const ctx = useApp()
  const { papers, refreshPapers, editor, notify, openReader, askAI } = ctx
  // 抽屉里读的是全局的 reader；分屏里每个标签页传入自己的 target
  const reader = variant === 'split' ? target ?? null : ctx.reader
  const paper: Paper | undefined = useMemo(() => papers.find((p) => p.id === reader?.paperId), [papers, reader?.paperId])
  const given: Evidence[] = reader?.evidences?.filter((e) => e.snippet || e.page) ?? []
  // 只给了那句话、没有依据（比如从正文里的引用打开）：在原文里找最相关的几段当作依据
  const [auto, setAuto] = useState<Evidence[] | null>(null)
  const evidences = given.length ? given : auto ?? []
  const [idx, setIdx] = useState(0)
  const [hl, setHl] = useState<PdfHighlight | null>(null)
  const [locating, setLocating] = useState(false)
  const [fetching, setFetching] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const ev = evidences[idx]
  const notes = useNotes(paper?.id)
  const [activeNote, setActiveNote] = useState<string | null>(null)
  const [popover, setPopover] = useState<{ note: Note; at: { x: number; y: number } } | null>(null)
  const [focus, setFocus] = useState<{ page: number; y?: number; key: number } | null>(null)
  const readingState = useReading()
  const reading = paper ? readingState.records[paper.id] : undefined
  const [readingOpen, setReadingOpen] = useState(false)
  const [recording, setRecording] = useState(false)
  const [position, setPosition] = useState<{ paperId: string; page: number; pages: number } | null>(null)
  const paperId = paper?.id
  const onPageChange = useCallback((page: number, pages: number) => {
    if (paperId) setPosition({ paperId, page, pages })
  }, [paperId])
  const currentPosition = position?.paperId === paperId ? position : null
  useEffect(() => { setReadingOpen(false); setPosition(null) }, [paperId])

  useEffect(() => {
    setIdx(0)
    setHl(null)
    setAuto(null)
    setFetchError(null)
  }, [reader?.nonce])

  useEffect(() => {
    if (!paper?.has_pdf || given.length || !reader?.claim) return
    let cancelled = false
    api
      .pdfPassages({ paper_id: paper.id }, reader.claim, 3)
      .then((r) => !cancelled && setAuto(r.passages.map((p) => ({ page: p.page, snippet: p.text, section: p.section }))))
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [reader?.nonce, paper?.id, paper?.has_pdf]) // eslint-disable-line react-hooks/exhaustive-deps

  // 没有本地 PDF 但有开放链接：自动下载并解析全文（之后检索和核查也能用上全文）
  useEffect(() => {
    if (!paper || paper.has_pdf || !paper.pdf_url || fetching || fetchError) return
    setFetching(true)
    api
      .fetchFulltext(paper.id)
      .then(() => refreshPapers())
      .catch((e: Error) => setFetchError(e.message))
      .finally(() => setFetching(false))
  }, [paper?.id, paper?.has_pdf]) // eslint-disable-line react-hooks/exhaustive-deps

  // 定位依据句
  useEffect(() => {
    if (!paper?.has_pdf || !ev?.snippet) {
      setHl(ev?.page ? { page: ev.page, rects: [] } : null)
      return
    }
    let cancelled = false
    setLocating(true)
    api
      .pdfLocate({ paper_id: paper.id }, ev.snippet, ev.page, reader?.claim ?? '')
      .then((r) => !cancelled && setHl(r))
      .catch(() => !cancelled && setHl(ev.page ? { page: ev.page, rects: [] } : null))
      .finally(() => !cancelled && setLocating(false))
    return () => {
      cancelled = true
    }
  }, [paper?.id, paper?.has_pdf, ev?.snippet, ev?.page, reader?.claim, reader?.nonce]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!reader || !paper) return <ReaderEmpty papers={papers} onOpen={(id) => openReader({ paperId: id })} />

  /** 插入引用（可带页码）；quote 不为空时连同原句一起插入：“原句”[1, p. 3] */
  const insertCite = (page?: number | null, quote?: string) => {
    if (!editor) return
    let pos = Math.min(reader.insertAt ?? editor.state.selection.to, editor.state.doc.content.size)
    const prev = editor.state.doc.textBetween(Math.max(0, pos - 1), pos)
    if (reader.insertAt == null && !quote && /[。．.!?！？]/.test(prev)) pos -= 1
    const pages = page ? { [paper.id]: String(page) } : {}
    const cite = { type: 'citation', attrs: { ids: [paper.id], pages } }
    editor.chain().focus().insertContentAt(pos, quote ? [{ type: 'text', text: `“${quote}”` }, cite] : cite).run()
    notify(`已插入引用 ${formatInline([paper.id], citeStore.get(), pages)}`, 'success')
  }

  const saveNote = async (sel: { page: number; rects: number[][]; text: string }, color: Note['color'], comment: string) => {
    try {
      const n = await api.createNote({ paper_id: paper.id, page: sel.page, quote: sel.text, rects: sel.rects, color, comment })
      notesStore.upsert(n)
      notify(comment ? '已保存笔记，写作时输入 @ 可以引用它' : '已划线', 'success')
    } catch (e: any) {
      notify(e.message, 'error')
      throw e
    }
  }

  const openNote = (n: Note) => {
    setActiveNote(n.id)
    setFocus({ page: n.page, y: Math.min(...n.rects.map((r) => r[1]), 0.5), key: Date.now() })
  }

  const upload = async (f: File) => {
    try {
      await api.uploadPaper(f, paper.id)
      await refreshPapers()
      setFetchError(null)
      notify('已上传原文', 'success')
    } catch (e: any) {
      notify(e.message, 'error')
    }
  }

  return (
    <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      {/* 论文信息 + 这处引用的判断 */}
      <Box sx={{ px: 2, pb: 1.25 }}>
        <Card sx={{ cursor: 'default', bgcolor: brand.subtle }}>
          <Typography sx={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.4 }}>{paper.title}</Typography>
          <Typography noWrap sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25 }}>
            {authorYearLabel(paper)}
            {paper.venue ? ` · ${paper.venue}` : ''}
            {paper.has_fulltext ? ` · 全文 ${paper.chunk_count} 段` : ' · 仅摘要'}
          </Typography>
          {ev && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mt: 0.75 }}>
              {ev.verdict && <VerdictChip v={ev.verdict} />}
              <Typography noWrap sx={{ fontSize: 11.5, color: brand.muted, flex: 1, minWidth: 0 }}>
                {given.length ? '依据' : '最相关的原文'}：{hl?.page ?? ev.page ? `第 ${hl?.page ?? ev.page} 页` : '页码未知'}
                {ev.section ? ` · ${ev.section}` : ''}
                {locating ? ' · 定位中…' : hl && hl.found === false && ev.snippet ? ' · 没能精确定位，已跳到所在页' : ''}
              </Typography>
              {evidences.length > 1 && (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <IconButton aria-label="上一处依据" onClick={() => setIdx((idx - 1 + evidences.length) % evidences.length)} sx={{ p: 0.25 }}>
                    <NavigateBeforeIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                  <Typography sx={{ fontSize: 11.5, color: brand.blue }}>
                    {idx + 1} / {evidences.length}
                  </Typography>
                  <IconButton aria-label="下一处依据" onClick={() => setIdx((idx + 1) % evidences.length)} sx={{ p: 0.25 }}>
                    <NavigateNextIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Box>
              )}
            </Box>
          )}
          {ev?.reason && <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25 }}>{ev.reason}</Typography>}
          <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.75 }}>
            {reading ? `${READING_LABELS[reading.status]} · ${pageProgressLabel(reading)}` : readingState.error ? '阅读记录加载失败' : '阅读记录加载中…'}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button disabled={!reading} onClick={() => setReadingOpen(true)}>修改阅读记录</Button>
            {readingState.error && <Button onClick={() => void readingStore.refresh()}>重试加载</Button>}
            {currentPosition && <Tooltip title="由你确认已读至当前页；不会自动标为全文已读。更换 PDF 后请核对页码。">
              <span><Button disabled={!reading || recording} onClick={async () => {
                if (!reading) return
                setRecording(true)
                try {
                  await readingStore.save(paper.id, { status: reading.status === 'read' ? 'read' : 'reading', read_through_page: currentPosition.page, total_pages: currentPosition.pages })
                  notify(`已记录读至第 ${currentPosition.page} 页`, 'success')
                } catch (e: any) { notify(e.message, 'error') }
                finally { setRecording(false) }
              }}>记录读至第 {currentPosition.page} 页</Button></span>
            </Tooltip>}
          </Box>
        </Card>
      </Box>
      {readingOpen && reading && <ReadingDialog key={paper.id} record={reading} title={paper.title} onClose={() => setReadingOpen(false)} />}

      {paper.has_pdf && <NotesList notes={notes} active={activeNote} onOpen={openNote} onCite={(n) => insertCite(n.page)} />}

      {/* 原文 */}
      {paper.has_pdf ? (
        <Suspense fallback={<Box sx={{ flex: 1, bgcolor: '#ececec' }} />}>
        <PdfViewer
          key={paper.id}
          src={pdfFileUrl({ paper_id: paper.id })}
          toc={variant === 'split'}
          page={ev?.page ?? null}
          highlight={hl}
          jumpKey={`${reader.nonce}-${idx}`}
          onError={(m) => notify(m, 'error')}
          onPageChange={onPageChange}
          focus={focus}
          marks={notes.map((n) => ({ id: n.id, page: n.page, rects: n.rects, fill: NOTE_COLORS[n.color].fill }))}
          activeMark={activeNote}
          onMarkClick={(id, at) => {
            const n = notes.find((x) => x.id === id)
            if (n) {
              setActiveNote(id)
              setPopover({ note: n, at })
            }
          }}
          renderSelectionMenu={(sel, close) => (
            <SelectionMenu
              sel={sel}
              close={close}
              canCite={!!editor}
              onCite={(s2, withQuote) => insertCite(s2.page, withQuote ? s2.text : undefined)}
              onSave={saveNote}
              onAsk={(s2) => askAI(`请解释《${paper.title}》第 ${s2.page} 页的这段原文，并说明它和我正在写的内容有什么关系：「${s2.text.slice(0, 400)}」`)}
            />
          )}
        />
        </Suspense>
      ) : (
        <Box sx={{ flex: 1, overflow: 'auto', px: 2, borderTop: `1px solid ${brand.lineSoft}`, pt: 2 }}>
          {fetching ? (
            <Typography sx={{ fontSize: 13, color: brand.muted, display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={14} /> 正在下载并解析全文…
            </Typography>
          ) : (
            <>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: brand.muted, mb: 1 }}>
                <PictureAsPdfOutlinedIcon fontSize="small" />
                <Typography sx={{ fontSize: 13 }}>
                  {fetchError ? `没能下载原文：${fetchError}` : '这篇文献没有开放获取的 PDF。'}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Button variant="outlined" startIcon={<UploadFileOutlinedIcon />} onClick={() => fileRef.current?.click()}>
                  上传 PDF
                </Button>
                {paper.url && (
                  <Button component="a" href={paper.url} target="_blank" rel="noreferrer">
                    出版商页面 ↗
                  </Button>
                )}
              </Box>
              {ev?.snippet && (
                <>
                  <Typography sx={{ fontSize: 12, color: brand.faint, mb: 0.5 }}>依据片段</Typography>
                  <Typography sx={{ fontSize: 12.5, lineHeight: 1.7, pl: 1, borderLeft: `3px solid #f5b400`, mb: 2 }}>{ev.snippet}</Typography>
                </>
              )}
              <Typography sx={{ fontSize: 12, color: brand.faint, mb: 0.5 }}>摘要</Typography>
              <Typography sx={{ fontSize: 12.5, lineHeight: 1.7 }}>{paper.abstract || '（无摘要）'}</Typography>
            </>
          )}
          <input ref={fileRef} type="file" accept=".pdf" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0]).then(() => (e.target.value = ''))} />
        </Box>
      )}

      <NotePopover
        note={popover ? notes.find((n) => n.id === popover.note.id) ?? null : null}
        at={popover?.at ?? null}
        onClose={() => setPopover(null)}
        onCite={(n) => {
          insertCite(n.page)
          setPopover(null)
        }}
        onUpdate={(n, fields) =>
          api
            .updateNote(n.id, fields)
            .then((x) => notesStore.upsert(x))
            .catch((e) => notify(e.message, 'error'))
        }
        onDelete={(n) =>
          api
            .deleteNote(n.id)
            .then(() => {
              notesStore.remove(n.id)
              setPopover(null)
              notify('已删除笔记', 'success')
            })
            .catch((e) => notify(e.message, 'error'))
        }
      />

      {/* 底部操作 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 2, py: 1.25, borderTop: `1px solid ${brand.line}`, bgcolor: '#fff' }}>
        <Button variant="contained" startIcon={<FormatQuoteIcon />} disabled={!editor} onClick={() => insertCite(hl?.page ?? ev?.page ?? null)}>
          插入引用 {formatInline([paper.id], citeStore.get(), hl?.page ?? ev?.page ? { [paper.id]: String(hl?.page ?? ev?.page) } : undefined)}
        </Button>
        {evidences.length > 1 && (
          <Button variant="outlined" onClick={() => setIdx((idx + 1) % evidences.length)}>
            换一处依据
          </Button>
        )}
        <Box sx={{ flex: 1 }} />
        {reader.insertAt != null && (
          <Typography sx={{ fontSize: 11.5, color: brand.faint }}>插入到批注对应的句子</Typography>
        )}
      </Box>
    </Box>
  )
}

function ReaderEmpty({ papers, onOpen }: { papers: Paper[]; onOpen: (id: string) => void }) {
  const withPdf = papers.filter((p) => p.has_pdf)
  return (
    <Box sx={{ flex: 1, overflow: 'auto', px: 2, pt: 1 }}>
      <Typography sx={{ fontSize: 13, color: brand.muted, lineHeight: 1.7, mb: 2 }}>
        在页边批注上点「看原文」、在找引用结果或文献库里点「阅读原文」，这里会打开那篇论文，并跳到依据所在的位置。
      </Typography>
      {withPdf.length > 0 && (
        <>
          <Typography sx={{ fontSize: 12, color: brand.faint, mb: 0.75 }}>有原文的文献（{withPdf.length}）</Typography>
          {withPdf.map((p) => (
            <Card key={p.id} onClick={() => onOpen(p.id)} sx={{ mb: 0.75, display: 'flex', alignItems: 'center', gap: 1 }}>
              <PictureAsPdfOutlinedIcon sx={{ fontSize: 18, color: brand.red }} />
              <Box sx={{ minWidth: 0 }}>
                <Typography noWrap sx={{ fontSize: 12.5, fontWeight: 500 }}>
                  {p.title}
                </Typography>
                <Typography noWrap sx={{ fontSize: 11.5, color: brand.muted }}>
                  {authorYearLabel(p)}
                </Typography>
              </Box>
            </Card>
          ))}
        </>
      )}
    </Box>
  )
}

/** 打开「原文」的小按钮，用在页边批注、找引用结果等处 */
export function ReadSourceButton({
  paperId, page, onClick, label,
}: {
  paperId: string | null | undefined
  page?: number | null
  onClick: () => void
  label?: string
}) {
  if (!paperId) return null
  return (
    <Tooltip title="在侧栏打开原文，跳到依据所在位置">
      <Button
        size="small"
        onClick={(e) => {
          e.stopPropagation()
          onClick()
        }}
        startIcon={<PictureAsPdfOutlinedIcon sx={{ fontSize: '14px !important' }} />}
        sx={{ py: 0, px: 0.75, minHeight: 22, fontSize: 11.5, color: brand.blue, bgcolor: 'rgba(0,120,212,0.07)', '&:hover': { bgcolor: 'rgba(0,120,212,0.13)' } }}
      >
        {label ?? (page ? `看原文 p.${page}` : '看原文')}
      </Button>
    </Tooltip>
  )
}
