import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Box, Button, Checkbox, Chip, CircularProgress, Collapse, Dialog, DialogActions, DialogContent, DialogTitle,
  Divider, FormControlLabel, IconButton, InputBase, Link, ListItemIcon, Menu, MenuItem, TextField, Tooltip, Typography,
} from '@mui/material'
import ArticleOutlinedIcon from '@mui/icons-material/ArticleOutlined'
import { Card, SectionLabel, Thread, ThreadItem } from '../components/Thread'
import SearchIcon from '@mui/icons-material/Search'
import TravelExploreIcon from '@mui/icons-material/TravelExplore'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import MoreVertIcon from '@mui/icons-material/MoreVert'
import FormatQuoteIcon from '@mui/icons-material/FormatQuote'
import CloudDownloadOutlinedIcon from '@mui/icons-material/CloudDownloadOutlined'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddIcon from '@mui/icons-material/Add'
import CheckIcon from '@mui/icons-material/Check'
import PictureAsPdfOutlinedIcon from '@mui/icons-material/PictureAsPdfOutlined'
import { api } from '../lib/api'
import { useApp } from '../lib/appContext'
import { normalizeTitle } from '../lib/util'
import type { Paper, PaperIn } from '../lib/types'
import { brand } from '../theme'
import AutorenewIcon from '@mui/icons-material/Autorenew'
import { ReadingDialog } from '../components/ReadingDialog'
import { useNotes, useNotesLoaded, notesStore } from '../lib/notes'
import { pageProgressLabel, READING_LABELS, readingReminder, type ReadingRecord } from '../lib/reading'
import { readingStore, useCitationCounts, useReading } from '../lib/readingStore'

const PARSER_LABEL: Record<string, string> = { grobid: 'GROBID', mineru: 'MinerU', pymupdf: 'PyMuPDF' }

function meta(p: PaperIn) {
  const a = p.authors ?? []
  return [a.length ? a[0] + (a.length > 1 ? ' 等' : '') : '佚名', p.year, p.venue].filter(Boolean).join(' · ')
}

function SearchDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { papers, refreshPapers, notify, openPreview } = useApp()
  const [q, setQ] = useState('')
  const [sources, setSources] = useState({ semantic_scholar: true, arxiv: true, openalex: false })
  useEffect(() => {
    if (!open) return
    api
      .settings()
      .then((v) => {
        const on = (v.values.SEARCH_SOURCES?.value || 'semantic_scholar,arxiv').split(',')
        setSources({ semantic_scholar: on.includes('semantic_scholar'), arxiv: on.includes('arxiv'), openalex: on.includes('openalex') })
      })
      .catch(() => undefined)
  }, [open])
  const [results, setResults] = useState<PaperIn[]>([])
  const [errors, setErrors] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [adding, setAdding] = useState<string | null>(null)
  const inLib = useMemo(() => new Set(papers.map((p) => normalizeTitle(p.title))), [papers])

  const search = async () => {
    if (q.trim().length < 2) return
    setLoading(true)
    try {
      const src = Object.entries(sources).filter(([, v]) => v).map(([k]) => k).join(',')
      const r = await api.searchExternal(q.trim(), src)
      setResults(r.results)
      setErrors(r.errors)
    } catch (e: any) {
      notify(e.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  const add = async (p: PaperIn) => {
    setAdding(p.title)
    try {
      await api.addPaper(p)
      await refreshPapers()
    } catch (e: any) {
      notify(e.message, 'error')
    } finally {
      setAdding(null)
    }
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ pb: 1 }}>在线检索文献</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
          <TextField
            autoFocus
            fullWidth
            placeholder="英文关键词效果最好，例如：retrieval augmented generation citation"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && search()}
          />
          <Button variant="contained" onClick={search} disabled={loading} startIcon={loading ? <CircularProgress size={14} /> : <SearchIcon />}>
            检索
          </Button>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {(['semantic_scholar', 'arxiv', 'openalex'] as const).map((s) => (
            <FormControlLabel
              key={s}
              control={<Checkbox size="small" checked={sources[s]} onChange={(e) => setSources({ ...sources, [s]: e.target.checked })} />}
              label={<Typography variant="body2">{s === 'arxiv' ? 'arXiv' : s === 'openalex' ? 'OpenAlex' : 'Semantic Scholar'}</Typography>}
            />
          ))}
        </Box>
        {errors.map((e) => (
          <Typography key={e} variant="caption" component="div" sx={{ color: e.startsWith('提示') ? 'text.secondary' : 'warning.main' }}>
            {e.startsWith('提示') ? '💡 ' : '⚠ '}
            {e}
          </Typography>
        ))}
        <Box sx={{ mt: 1 }}>
          {results.map((p) => {
            const have = inLib.has(normalizeTitle(p.title))
            return (
              <Box key={p.title} sx={{ py: 1.25, borderBottom: `1px solid ${brand.line}`, display: 'flex', gap: 1.5 }}>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {p.url ? (
                      <Link href={p.url} target="_blank" rel="noreferrer" underline="hover" color="inherit">
                        {p.title}
                      </Link>
                    ) : (
                      p.title
                    )}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" component="div">
                    {meta(p)}
                    {p.citation_count != null && ` · 被引 ${p.citation_count}`}
                    {p.pdf_url && ' · 有开放 PDF'}
                  </Typography>
                  {p.abstract && (
                    <Typography variant="caption" component="div" sx={{ mt: 0.5, color: brand.ink, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {p.abstract}
                    </Typography>
                  )}
                </Box>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75, alignItems: 'stretch' }}>
                  {have ? (
                    <Chip size="small" icon={<CheckIcon />} label="已在库中" />
                  ) : (
                    <Button variant="outlined" startIcon={adding === p.title ? <CircularProgress size={12} /> : <AddIcon />} onClick={() => add(p)} disabled={!!adding}>
                      加入
                    </Button>
                  )}
                  {p.pdf_url && (
                    <Button
                      startIcon={<PictureAsPdfOutlinedIcon />}
                      onClick={() =>
                        openPreview({
                          paper: p,
                          actions: have ? [] : [{ label: '加入文献库', primary: true, onClick: () => add(p) }],
                        })
                      }
                    >
                      预览
                    </Button>
                  )}
                </Box>
              </Box>
            )
          })}
          {!loading && !results.length && (
            <Typography variant="body2" color="text.secondary" sx={{ py: 3, textAlign: 'center' }}>
              输入关键词后回车检索。加入文献库的论文可以在正文中用 @ 引用。
            </Typography>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>完成</Button>
      </DialogActions>
    </Dialog>
  )
}

function PaperItem({ p, reading, noteCount, citeCount }: { p: Paper; reading?: ReadingRecord; noteCount: number | null; citeCount: number | null }) {
  const { editor, refreshPapers, notify, openReader } = useApp()
  const [open, setOpen] = useState(false)
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [busy, setBusy] = useState(false)
  const [readingOpen, setReadingOpen] = useState(false)
  const reminder = reading && citeCount != null ? readingReminder(reading.status, citeCount) : null
  const fileRef = useRef<HTMLInputElement>(null)

  const act = async (fn: () => Promise<any>, ok?: string) => {
    setAnchor(null)
    setBusy(true)
    try {
      const r = await fn()
      await refreshPapers()
      // 外部解析器不可用、退回了 PyMuPDF：告诉用户原因
      if (r?.note) notify(`${ok ?? '完成'}（${r.note}，已改用 PyMuPDF）`, 'warning')
      else if (ok) notify(r?.parser && r.parser !== 'pymupdf' ? `${ok}（${PARSER_LABEL[r.parser] ?? r.parser}）` : ok, 'success')
    } catch (e: any) {
      notify(e.message, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <ThreadItem icon={<ArticleOutlinedIcon />} dense>
      <Card variant={open ? 'data' : 'plain'} onClick={() => setOpen(!open)} sx={{ pr: 0.5, '&:hover .ops': { opacity: 1 } }}>
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'flex-start' }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography sx={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4 }}>{p.title}</Typography>
            <Typography noWrap sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25 }}>
              {meta(p)}
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.75, mt: 0.5, alignItems: 'center' }}>
              <Box
                sx={{
                  fontSize: 10.5, px: 0.6, borderRadius: '3px', lineHeight: '17px',
                  color: p.has_fulltext ? brand.green : brand.muted, bgcolor: p.has_fulltext ? 'rgba(46,125,50,0.10)' : 'rgba(0,0,0,0.05)',
                }}
              >
                {p.has_fulltext ? `全文 · ${p.chunk_count} 段` : '仅摘要'}
                {p.has_pdf ? ' · PDF' : ''}
                {p.has_fulltext && p.parser && p.parser !== 'pymupdf' ? ` · ${PARSER_LABEL[p.parser] ?? p.parser}` : ''}
              </Box>
              <Typography sx={{ fontFamily: 'Roboto Mono', fontSize: 10.5, color: brand.faint }}>{p.bibkey}</Typography>
              {busy && <CircularProgress size={10} />}
            </Box>
          </Box>
          <Box className="ops" sx={{ display: 'flex', opacity: open ? 1 : 0, transition: 'opacity .15s' }} onClick={(e) => e.stopPropagation()}>
            <Tooltip title="在光标处引用">
              <span>
                <IconButton aria-label="在光标处引用" disabled={!editor} onClick={() => editor?.chain().focus().insertCitation([p.id]).run()}>
                  <FormatQuoteIcon sx={{ fontSize: 16, color: brand.blue }} />
                </IconButton>
              </span>
            </Tooltip>
            {(p.has_pdf || p.pdf_url) && (
              <Tooltip title={p.has_pdf ? '阅读原文' : '下载并阅读原文'}>
                <IconButton aria-label="阅读原文" onClick={() => openReader({ paperId: p.id })}>
                  <PictureAsPdfOutlinedIcon sx={{ fontSize: 16, color: brand.blue }} />
                </IconButton>
              </Tooltip>
            )}
            <IconButton onClick={(e) => setAnchor(e.currentTarget)} aria-label="更多">
              <MoreVertIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Box>
        </Box>
        <Box sx={{ mt: 0.75 }} onClick={(e) => e.stopPropagation()}>
          <Typography sx={{ fontSize: 11.5, color: brand.muted }}>
            {reading ? READING_LABELS[reading.status] : '阅读记录待加载'} · 划线/笔记 {noteCount ?? '未知'} 条 · 当前文档引用 {citeCount ?? '未知'} 次
          </Typography>
          {reading && <Typography sx={{ fontSize: 11.5, color: brand.muted }}>{pageProgressLabel(reading)}</Typography>}
          {reminder && <Typography sx={{ fontSize: 11.5, color: 'warning.main' }}>{reminder}</Typography>}
          <Button size="small" disabled={!reading} onClick={() => setReadingOpen(true)}>阅读记录</Button>
          {reading?.read_through_page != null && (p.has_pdf || p.pdf_url) && (
            <Button size="small" onClick={() => openReader({ paperId: p.id, evidences: [{ page: reading.read_through_page }] })}>继续阅读 p.{reading.read_through_page}</Button>
          )}
        </Box>
        <Collapse in={open}>
          <Typography sx={{ mt: 1, fontSize: 12, color: brand.ink, lineHeight: 1.65 }}>{p.abstract || '（无摘要）'}</Typography>
          {p.url && (
            <Link href={p.url} target="_blank" rel="noreferrer" sx={{ fontSize: 12 }} onClick={(e) => e.stopPropagation()}>
              打开原文页面
            </Link>
          )}
        </Collapse>
      </Card>
      {readingOpen && reading && <ReadingDialog record={reading} title={p.title} onClose={() => setReadingOpen(false)} />}
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.txt,.md"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) act(() => api.uploadPaper(f, p.id), '全文已解析')
          e.target.value = ''
        }}
      />
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        <MenuItem disabled={!p.pdf_url} onClick={() => act(() => api.fetchFulltext(p.id), '全文已下载并解析')}>
          <ListItemIcon>
            <CloudDownloadOutlinedIcon fontSize="small" />
          </ListItemIcon>
          下载开放获取全文
        </MenuItem>
        <MenuItem disabled={!p.has_pdf} onClick={() => act(() => api.reparse(p.id), '已重新解析')}>
          <ListItemIcon>
            <AutorenewIcon fontSize="small" />
          </ListItemIcon>
          用当前解析器重新解析
        </MenuItem>
        <MenuItem onClick={() => { setAnchor(null); fileRef.current?.click() }}>
          <ListItemIcon>
            <UploadFileIcon fontSize="small" />
          </ListItemIcon>
          上传本地 PDF 作为全文
        </MenuItem>
        <MenuItem
          onClick={async () => {
            setAnchor(null)
            await navigator.clipboard.writeText(await api.bibtex([p.id]))
            notify('BibTeX 已复制', 'success')
          }}
        >
          <ListItemIcon>
            <ContentCopyIcon fontSize="small" />
          </ListItemIcon>
          复制 BibTeX
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => act(() => api.deletePaper(p.id), '已从文献库删除')} sx={{ color: 'error.main' }}>
          <ListItemIcon>
            <DeleteOutlineIcon fontSize="small" color="error" />
          </ListItemIcon>
          从文献库删除
        </MenuItem>
      </Menu>
    </ThreadItem>
  )
}

export function LibraryPanel() {
  const { papers, refreshPapers, notify, editor, docId } = useApp()
  const reading = useReading()
  const counts = useCitationCounts(editor)
  const notes = useNotes()
  const notesLoaded = useNotesLoaded()
  const noteCounts = useMemo(() => {
    const result: Record<string, number> = Object.create(null)
    for (const n of notes) result[n.paper_id] = (result[n.paper_id] ?? 0) + 1
    return result
  }, [notes])
  const [filter, setFilter] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase()
    if (!f) return papers
    return papers.filter((p) => `${p.title} ${p.authors.join(' ')} ${p.year} ${p.bibkey}`.toLowerCase().includes(f))
  }, [papers, filter])

  const upload = async (files: FileList) => {
    setUploading(true)
    for (const f of Array.from(files)) {
      try {
        const r = await api.uploadPaper(f)
        notify(`《${r.paper.title.slice(0, 30)}》已解析为 ${r.chunks} 个片段`, 'success')
      } catch (e: any) {
        notify(`${f.name}：${e.message}`, 'error')
      }
    }
    await refreshPapers()
    setUploading(false)
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ px: 2, pt: 0.5, pb: 1 }}>
        <SectionLabel>文献库 · {papers.length} 篇</SectionLabel>
        <Typography sx={{ fontSize: 11.5, color: brand.muted, mb: 0.75 }}>引用次数仅统计当前文档（含草稿）；同一引用节点每篇计一次。阅读状态由你确认。</Typography>
        {(reading.error || !notesLoaded) && <Box sx={{ mb: 0.75 }}>
          <Typography sx={{ fontSize: 11.5, color: 'warning.main' }}>{reading.error ? `阅读记录加载失败：${reading.error}` : '笔记数量待加载，暂显示未知'}</Typography>
          <Button size="small" onClick={() => { void readingStore.refresh(); notesStore.refresh().catch((e) => notify(e.message, 'error')) }}>重试加载</Button>
        </Box>}
        <Box
          sx={{
            display: 'flex', alignItems: 'center', gap: 0.75, border: `1px solid ${brand.line}`, borderRadius: '6px', px: 1, py: 0.25,
            '&:focus-within': { borderColor: 'rgba(0,120,212,0.5)' },
          }}
        >
          <SearchIcon sx={{ fontSize: 17, color: brand.faint }} />
          <InputBase fullWidth placeholder="筛选标题、作者、年份" value={filter} onChange={(e) => setFilter(e.target.value)} sx={{ fontSize: 13 }} />
        </Box>
      </Box>
      <Box sx={{ flex: 1, overflow: 'auto', px: 2, pb: 2 }}>
        <Thread>
          {shown.map((p) => (
            <PaperItem key={p.id} p={p} reading={reading.records[p.id]} noteCount={notesLoaded ? noteCounts[p.id] ?? 0 : null} citeCount={editor && docId ? counts[p.id] ?? 0 : null} />
          ))}
          <ThreadItem dense>
            <Box sx={{ display: 'flex', gap: 0.75 }}>
              <Card variant="dashed" onClick={() => setSearchOpen(true)} sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 0.75, color: brand.muted }}>
                <TravelExploreIcon sx={{ fontSize: 16 }} /> 在线检索文献
              </Card>
              <Card
                variant="dashed"
                onClick={uploading ? undefined : () => fileRef.current?.click()}
                sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 0.75, color: brand.muted }}
              >
                {uploading ? <CircularProgress size={14} /> : <UploadFileIcon sx={{ fontSize: 16 }} />} 上传 PDF
              </Card>
            </Box>
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md" multiple hidden onChange={(e) => e.target.files && upload(e.target.files).then(() => (e.target.value = ''))} />
          </ThreadItem>
        </Thread>
        {!papers.length && (
          <Typography sx={{ fontSize: 12.5, color: brand.muted, mt: 1, lineHeight: 1.7 }}>
            文献库还是空的。可以在线检索、上传 PDF，或者让「AI 助手」里的综述 Agent 帮你找文献。
          </Typography>
        )}
      </Box>
      <SearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} />
    </Box>
  )
}
