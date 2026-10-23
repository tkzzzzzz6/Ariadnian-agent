import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Box, Button, Checkbox, Chip, CircularProgress, Collapse, IconButton, InputBase, Menu, MenuItem, Tooltip, Typography,
} from '@mui/material'
import PersonOutlineIcon from '@mui/icons-material/PersonOutline'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import PauseCircleFilledIcon from '@mui/icons-material/PauseCircleFilled'
import ErrorIcon from '@mui/icons-material/Error'
import AccountTreeOutlinedIcon from '@mui/icons-material/AccountTreeOutlined'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddIcon from '@mui/icons-material/Add'
import AutorenewIcon from '@mui/icons-material/Autorenew'
import NoteAddOutlinedIcon from '@mui/icons-material/NoteAddOutlined'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { api } from '../lib/api'
import { useAgentStream } from '../lib/agentStream'
import { runStats } from '../lib/runStats'
import { RunStatsBar } from './RunStatsBar'
import { collectSections, critiqueStore } from '../lib/critique'
import { CritiqueRun } from './CritiqueRun'
import { AgentHistory, type ThreadInfo } from './AgentHistory'
import RateReviewOutlinedIcon from '@mui/icons-material/RateReviewOutlined'
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined'
import { useApp } from '../lib/appContext'
import { contentFromText } from '../lib/editorText'
import { outlineFromDoc } from '../lib/writer'
import EditNoteIcon from '@mui/icons-material/EditNote'
import { WriterRun, type WriterStart } from './WriterRun'
import PictureAsPdfOutlinedIcon from '@mui/icons-material/PictureAsPdfOutlined'
import type { OutlineSection, PaperIn } from '../lib/types'
import { Card, Composer, Thread, ThreadItem } from '../components/Thread'
import { brand } from '../theme'

type Status = 'idle' | 'running' | 'waiting' | 'done' | 'stopped' | 'error'

interface Candidate extends PaperIn {
  cid: number
}
interface Result {
  topic?: string
  outline?: OutlineSection[]
  synthesis?: string
  notes?: { bibkey: string; contribution: string; limitation?: string }[]
  paper_ids?: string[]
  errors?: string[]
}
interface Step {
  node: string
  label: string
  ms: number
}

const fmt = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.max(ms, 1)} ms`)

const EXAMPLES = [
  '面向学术写作的检索增强智能体与引用可信度验证',
  '大语言模型幻觉的检测与缓解方法综述',
  '基于图神经网络的推荐系统冷启动问题研究',
]

// ------------------------------------------------------------------ 小下拉
function MiniSelect<T extends string | number>({ value, options, onChange, label }: { value: T; options: [T, string][]; onChange: (v: T) => void; label: string }) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  return (
    <>
      <Tooltip title={label}>
        <Box
          role="button"
          onClick={(e) => setAnchor(e.currentTarget)}
          sx={{
            display: 'inline-flex', alignItems: 'center', fontSize: 12, color: brand.muted, px: 0.75, py: 0.25, borderRadius: '6px',
            cursor: 'pointer', whiteSpace: 'nowrap', '&:hover': { bgcolor: 'rgba(0,0,0,0.05)', color: brand.ink },
          }}
        >
          {options.find(([v]) => v === value)?.[1]}
          <ExpandMoreIcon sx={{ fontSize: 14 }} />
        </Box>
      </Tooltip>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        {options.map(([v, l]) => (
          <MenuItem
            key={String(v)}
            selected={v === value}
            dense
            onClick={() => {
              onChange(v)
              setAnchor(null)
            }}
          >
            {l}
          </MenuItem>
        ))}
      </Menu>
    </>
  )
}

// ------------------------------------------------------------------ 勾选文献
function SelectPapersCard({ value, onSubmit }: { value: any; onSubmit: (v: any) => void }) {
  const { openPreview } = useApp()
  const cands: Candidate[] = value.candidates ?? []
  const [sel, setSel] = useState<number[]>(() => cands.slice(0, value.max_papers ?? 8).map((c) => c.cid))
  const [open, setOpen] = useState<number | null>(null)
  return (
    <Card variant="data" sx={{ p: 0, overflow: 'hidden' }}>
      <Box sx={{ px: 1.25, py: 1, display: 'flex', alignItems: 'center' }}>
        <Typography sx={{ fontSize: 13, fontWeight: 500, flex: 1 }}>勾选要纳入文献库的论文</Typography>
        <Typography sx={{ fontSize: 12, color: brand.muted }}>
          {sel.length}/{cands.length}
        </Typography>
      </Box>
      <Box sx={{ maxHeight: 340, overflow: 'auto', bgcolor: '#fff', borderTop: `1px solid rgba(0,120,212,0.18)` }}>
        {cands.map((c) => (
          <Box key={c.cid} sx={{ display: 'flex', gap: 0.5, px: 0.75, py: 0.75, borderBottom: `1px solid ${brand.lineSoft}` }}>
            <Checkbox
              size="small"
              sx={{ p: 0.5, alignSelf: 'flex-start' }}
              checked={sel.includes(c.cid)}
              onChange={(e) => setSel(e.target.checked ? [...sel, c.cid] : sel.filter((x) => x !== c.cid))}
            />
            <Box sx={{ minWidth: 0, cursor: 'pointer', pt: 0.25 }} onClick={() => setOpen(open === c.cid ? null : c.cid)}>
              <Typography sx={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4 }}>{c.title}</Typography>
              <Typography sx={{ fontSize: 11.5, color: brand.muted }}>
                {c.authors?.[0] ?? '佚名'}
                {(c.authors?.length ?? 0) > 1 ? ' 等' : ''} · {c.year ?? 'n.d.'}
                {c.venue ? ` · ${c.venue}` : ''}
                {c.citation_count != null ? ` · 被引 ${c.citation_count}` : ''}
              </Typography>
              <Collapse in={open === c.cid}>
                <Typography sx={{ mt: 0.5, fontSize: 12, lineHeight: 1.6, color: brand.ink }}>{c.abstract || '（无摘要）'}</Typography>
              </Collapse>
              {c.pdf_url && (
                <Box
                  component="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    const picked = sel.includes(c.cid)
                    openPreview({
                      paper: c,
                      claim: value.topic ?? '',
                      actions: [
                        {
                          label: picked ? '不纳入这篇' : '勾选这篇',
                          primary: !picked,
                          onClick: () => setSel(picked ? sel.filter((x) => x !== c.cid) : [...sel, c.cid]),
                        },
                      ],
                    })
                  }}
                  sx={{ all: 'unset', cursor: 'pointer', mt: 0.5, display: 'inline-flex', alignItems: 'center', gap: 0.5, fontSize: 11.5, color: brand.blue, '&:hover': { textDecoration: 'underline' } }}
                >
                  <PictureAsPdfOutlinedIcon sx={{ fontSize: 14 }} /> 预览 PDF
                </Box>
              )}
            </Box>
          </Box>
        ))}
      </Box>
      <Box sx={{ display: 'flex', gap: 1, px: 1.25, py: 1, justifyContent: 'flex-end', bgcolor: '#fff', borderTop: `1px solid ${brand.lineSoft}` }}>
        <Button onClick={() => onSubmit({ selected: [] })} sx={{ color: brand.muted }}>
          都不需要
        </Button>
        <Button variant="contained" disabled={!sel.length} onClick={() => onSubmit({ selected: sel })}>
          确认并继续（{sel.length} 篇）
        </Button>
      </Box>
    </Card>
  )
}

// ------------------------------------------------------------------ 确认大纲
function ReviewOutlineCard({ value, onSubmit }: { value: any; onSubmit: (v: any) => void }) {
  const [outline, setOutline] = useState<OutlineSection[]>(value.outline ?? [])
  const [feedback, setFeedback] = useState('')
  const [showSyn, setShowSyn] = useState(false)
  useEffect(() => setOutline(value.outline ?? []), [value])
  const upd = (i: number, f: Partial<OutlineSection>) => setOutline(outline.map((s, j) => (j === i ? { ...s, ...f } : s)))
  const move = (i: number, d: number) => {
    const j = i + d
    if (j < 0 || j >= outline.length) return
    const next = [...outline]
    ;[next[i], next[j]] = [next[j], next[i]]
    setOutline(next)
  }
  return (
    <Card variant="data" sx={{ p: 0, overflow: 'hidden' }}>
      <Box sx={{ px: 1.25, py: 1, display: 'flex', alignItems: 'center' }}>
        <Typography sx={{ fontSize: 13, fontWeight: 500, flex: 1 }}>确认或修改大纲</Typography>
        {value.synthesis && (
          <Button size="small" onClick={() => setShowSyn(!showSyn)} sx={{ py: 0, fontSize: 12 }}>
            {showSyn ? '收起' : '研究现状'}
          </Button>
        )}
      </Box>
      <Collapse in={showSyn}>
        <Typography sx={{ px: 1.25, pb: 1, fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-wrap', color: brand.ink }}>{value.synthesis}</Typography>
      </Collapse>
      <Box sx={{ bgcolor: '#fff', borderTop: `1px solid rgba(0,120,212,0.18)`, p: 1 }}>
        {outline.map((s, i) => (
          <Box key={i} sx={{ display: 'flex', gap: 1, py: 0.75, borderBottom: `1px solid ${brand.lineSoft}`, '&:hover .ops': { opacity: 1 } }}>
            <Typography sx={{ fontSize: 12, color: brand.blue, fontWeight: 600, width: 16, pt: '3px', textAlign: 'right' }}>{i + 1}</Typography>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <InputBase fullWidth value={s.title} onChange={(e) => upd(i, { title: e.target.value })} sx={{ fontSize: 13.5, fontWeight: 600, p: 0 }} />
              <InputBase
                fullWidth
                multiline
                value={s.goal}
                placeholder="本节要论证什么"
                onChange={(e) => upd(i, { goal: e.target.value })}
                sx={{ fontSize: 12, color: brand.muted, p: 0 }}
              />
              {s.bibkeys?.length > 0 && (
                <Box sx={{ mt: 0.5, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {s.bibkeys.map((k) => (
                    <Chip key={k} size="small" label={k} sx={{ height: 18, fontSize: 10, fontFamily: 'Roboto Mono', bgcolor: 'rgba(0,120,212,0.08)', color: brand.blue }} />
                  ))}
                </Box>
              )}
            </Box>
            <Box className="ops" sx={{ display: 'flex', alignItems: 'flex-start', opacity: 0, transition: 'opacity .15s', '& svg': { fontSize: 15 } }}>
              <IconButton onClick={() => move(i, -1)} disabled={i === 0} aria-label="上移">
                <ArrowUpwardIcon />
              </IconButton>
              <IconButton onClick={() => move(i, 1)} disabled={i === outline.length - 1} aria-label="下移">
                <ArrowDownwardIcon />
              </IconButton>
              <IconButton onClick={() => setOutline(outline.filter((_, j) => j !== i))} aria-label="删除">
                <DeleteOutlineIcon />
              </IconButton>
            </Box>
          </Box>
        ))}
        <Button size="small" startIcon={<AddIcon />} onClick={() => setOutline([...outline, { title: '新章节', goal: '', bibkeys: [] }])} sx={{ mt: 0.5 }}>
          添加章节
        </Button>
        <Box sx={{ display: 'flex', gap: 1, mt: 1, alignItems: 'center', border: `1px solid ${brand.line}`, borderRadius: '6px', px: 1, py: 0.25 }}>
          <InputBase fullWidth multiline maxRows={3} placeholder="不满意？写下意见让 AI 重写，例如：方法拆成两节" value={feedback} onChange={(e) => setFeedback(e.target.value)} sx={{ fontSize: 12.5 }} />
          <Button size="small" startIcon={<AutorenewIcon />} disabled={!feedback.trim()} onClick={() => onSubmit({ action: 'regenerate', feedback })}>
            重写
          </Button>
        </Box>
      </Box>
      <Box sx={{ px: 1.25, py: 1, bgcolor: '#fff', borderTop: `1px solid ${brand.lineSoft}` }}>
        <Button fullWidth variant="contained" disabled={!outline.length} onClick={() => onSubmit({ action: 'approve', outline })}>
          确认大纲
        </Button>
      </Box>
    </Card>
  )
}

// ------------------------------------------------------------------ 首页的 Agent 卡片
function Launcher({
  icon, title, desc, hint, disabled, onClick, children,
}: {
  icon: React.ReactNode
  title: string
  desc: string
  hint?: string
  disabled?: boolean
  onClick?: () => void
  children?: React.ReactNode
}) {
  return (
    <Box
      role={onClick ? 'button' : undefined}
      aria-disabled={disabled || undefined}
      onClick={disabled ? undefined : onClick}
      sx={{
        display: 'flex', gap: 1.25, p: 1.25, mb: 1, borderRadius: '8px', border: `1px solid ${brand.line}`, bgcolor: '#fff',
        cursor: onClick && !disabled ? 'pointer' : 'default', opacity: disabled ? 0.6 : 1, transition: 'border-color .15s, box-shadow .15s',
        '&:hover': onClick && !disabled ? { borderColor: brand.purple, boxShadow: '0 2px 8px rgba(135,100,184,0.15)' } : {},
      }}
    >
      <Box sx={{ color: brand.purple, '& svg': { fontSize: 20 }, pt: '1px' }}>{icon}</Box>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography sx={{ fontSize: 13.5, fontWeight: 600 }}>{title}</Typography>
        <Typography sx={{ fontSize: 12, color: brand.muted, lineHeight: 1.6, mt: 0.25 }}>{desc}</Typography>
        {hint && (
          <Typography sx={{ fontSize: 11.5, color: disabled ? brand.amber : brand.faint, mt: 0.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
            {hint}
          </Typography>
        )}
        {children}
      </Box>
    </Box>
  )
}

// ------------------------------------------------------------------ 主面板
export function AgentPanel() {
  const { docId, editor, papers, refreshPapers, notify, alertAgent } = useApp()
  const { run, stop, running, stopping } = useAgentStream()
  const [input, setInput] = useState('')
  const [topic, setTopic] = useState('')
  const [language, setLanguage] = useState<'zh' | 'en'>('zh')
  const [paperType, setPaperType] = useState('学术论文')
  const [maxPapers, setMaxPapers] = useState(8)
  const [fetchPdf, setFetchPdf] = useState<'yes' | 'no'>('yes')

  const [threadId, setThreadId] = useState<string | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [steps, setSteps] = useState<Step[]>([])
  const [log, setLog] = useState<string[]>([])
  const [pending, setPending] = useState<any>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sub, setSub] = useState<
    { kind: 'write'; start?: WriterStart; threadId?: string } | { kind: 'critique'; threadId?: string } | null
  >(null)
  const [docOutline, setDocOutline] = useState<OutlineSection[]>([])
  const [docChars, setDocChars] = useState(0)
  const lastT = useRef(0)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 文档里已有的一级标题大纲：可以直接按它逐节写作
  useEffect(() => {
    if (!editor) return
    let t = 0
    const calc = () => {
      setDocOutline(outlineFromDoc(editor))
      setDocChars(collectSections(editor).reduce((a, s) => a + s.text.length, 0))
    }
    const upd = () => {
      window.clearTimeout(t)
      t = window.setTimeout(calc, 400)
    }
    calc()
    editor.on('update', upd)
    return () => {
      editor.off('update', upd)
      window.clearTimeout(t)
    }
  }, [editor])

  const reset = () => {
    runStats.focus(null)
    setSub(null)
    setThreadId(null)
    setStatus('idle')
    setSteps([])
    setLog([])
    setPending(null)
    setResult(null)
    setError(null)
  }

  // 打开历史里的某次任务
  const openThread = async (t: ThreadInfo) => {
    reset()
    runStats.focus(t.thread_id)
    if (t.kind === 'write') return setSub({ kind: 'write', threadId: t.thread_id })
    if (t.kind === 'critique') return setSub({ kind: 'critique', threadId: t.thread_id })
    const st = await api.agentState(t.thread_id)
    setThreadId(t.thread_id)
    setTopic(t.topic)
    if (st.interrupt) {
      setStatus('waiting')
      setPending(st.interrupt)
      setLog(['已从上次中断处恢复'])
    } else if (st.status === 'done') {
      setStatus('done')
      setResult(st.result)
    } else {
      setStatus('stopped')
    }
  }

  // 开始新的分节写作 / 审稿（还没选历史任务）：先清掉上一个任务的统计，新任务开始后自动显示
  useEffect(() => {
    if (sub && !sub.threadId) runStats.focus(null)
  }, [sub])

  // 切换文档：恢复这篇文档最近一次还没结束的任务（等你确认 / 被停止），否则回到首页
  useEffect(() => {
    reset()
    critiqueStore.clear()
    if (!docId) return
    api
      .agentThreads(docId)
      .then((threads) => {
        const t = threads[0]
        // 审稿结束后意见仍留在页边，所以最近一次审稿也恢复
        if (t && (['waiting', 'stopped', 'running', 'stopping', 'interrupted'].includes(t.status) || (t.kind === 'critique' && t.status === 'done' && !critiqueStore.isDismissed(t.thread_id)))) openThread(t)
        // 面板打开的是别的任务时，最近一次审稿的意见照样挂回页边
        const c = threads.find((x) => x.kind === 'critique' && x.status !== 'error')
        if (c && c !== t && !critiqueStore.isDismissed(c.thread_id))
          api.agentState(c.thread_id).then((st) => {
            const v = st.interrupt?.kind === 'revise' ? st.interrupt : st.result
            if (v?.issues?.length && !critiqueStore.get().threadId) critiqueStore.load(c.thread_id, v, !st.interrupt)
          })
      })
      .catch(() => undefined)
  }, [docId]) // eslint-disable-line react-hooks/exhaustive-deps

  const scroll = () => setTimeout(() => scrollRef.current?.scrollTo({ top: 1e6, behavior: 'smooth' }), 30)

  const handle = useCallback(
    (e: any) => {
      const now = performance.now()
      if (e.type === 'status') {
        if (e.thread_id) setThreadId(e.thread_id)
        setStatus(e.status)
      } else if (e.type === 'progress') {
        setLog((l) => [...l, e.message].slice(-4))
        if (e.paper_id) refreshPapers()
      } else if (e.type === 'node') {
        setSteps((s) => [...s, { node: e.node, label: e.label, ms: Math.round(now - lastT.current) }])
        lastT.current = now
        setLog([])
      } else if (e.type === 'interrupt') {
        setPending(e.value)
        alertAgent(e.value.kind === 'select_papers' ? '综述 Agent：检索完成，等你勾选文献' : '综述 Agent：大纲生成好了，等你确认')
      } else if (e.type === 'done') {
        setStatus('done')
        setResult(e.result)
        setPending(null)
        refreshPapers()
        alertAgent('综述 Agent 已完成')
      } else if (e.type === 'error') {
        setStatus('error')
        setError(e.message)
      }
      scroll()
    },
    [refreshPapers, alertAgent],
  )

  const start = async (t: string) => {
    reset()
    setTopic(t)
    setInput('')
    setStatus('running')
    lastT.current = performance.now()
    try {
      const r = await run(
        '/api/agent/start',
        { topic: t, doc_id: docId, language, paper_type: paperType, max_papers: maxPapers, fetch_pdf: fetchPdf === 'yes' },
        handle,
      )
      if (r === 'stopped') setStatus('stopped')
    } catch (e: any) {
      setStatus('error')
      setError(e.message)
    }
  }

  // 停在请求半路时，「继续」把最近一次的选择补发出去
  const lastSent = useRef<{ kind: string; value: any } | null>(null)
  const cont = async () => {
    if (!threadId) return
    const st = await api.agentState(threadId).catch(() => null)
    if (st?.interrupt) {
      if (lastSent.current?.kind === st.interrupt.kind) return resume(lastSent.current!.value)
      setPending(st.interrupt)
      return setStatus('waiting')
    }
    if (st?.status === 'done') {
      setResult(st.result)
      return setStatus('done')
    }
    resume({})
  }

  const resume = async (value: any) => {
    if (!threadId) return
    if (pending) lastSent.current = { kind: pending.kind, value }
    setPending(null)
    setStatus('running')
    lastT.current = performance.now()
    scroll()
    try {
      const r = await run(`/api/agent/${threadId}/resume`, { value }, handle)
      if (r === 'stopped') setStatus('stopped')
    } catch (e: any) {
      setStatus('error')
      setError(e.message)
    }
  }

  const insertOutline = () => {
    if (!editor || !result?.outline) return
    const byKey = new Map(papers.map((p) => [p.bibkey, p.id]))
    const nodes: any[] = []
    for (const s of result.outline) {
      nodes.push({ type: 'heading', attrs: { level: 1 }, content: [{ type: 'text', text: s.title }] })
      const ids = (s.bibkeys ?? []).map((k) => byKey.get(k)).filter(Boolean)
      const para: any[] = [{ type: 'text', text: `写作要点：${s.goal}`, marks: [{ type: 'italic' }] }]
      if (ids.length) para.push({ type: 'text', text: ' 可引用：', marks: [{ type: 'italic' }] }, { type: 'citation', attrs: { ids } })
      nodes.push({ type: 'paragraph', content: para }, { type: 'paragraph' })
    }
    const isEmpty = editor.state.doc.textContent.trim().length < 10
    if (isEmpty) editor.chain().focus().setContent({ type: 'doc', content: nodes }).run()
    else editor.chain().focus('end').insertContent(nodes).run()
    notify(isEmpty ? '大纲已写入文档' : '大纲已追加到文档末尾', 'success')
  }

  const insertSynthesis = () => {
    if (!editor || !result?.synthesis) return
    const c = contentFromText(result.synthesis)
    editor.chain().focus().insertContent(Array.isArray(c) ? { type: 'paragraph', content: c } : c).run()
  }

  const total = steps.filter((s) => !['select_papers', 'review_outline'].includes(s.node)).reduce((a, s) => a + s.ms, 0)

  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pb: 0.5, gap: 1 }}>
        <Typography sx={{ fontSize: 12, color: brand.faint, flex: 1 }}>
          {sub?.kind === 'write' ? '分节写作' : sub?.kind === 'critique' ? '审稿' : status === 'idle' ? '选择一个 Agent' : '综述'}
        </Typography>
        <AgentHistory docId={docId} onOpen={openThread} disabled={running} />
      </Box>
      <Box ref={scrollRef} sx={{ flex: 1, overflow: 'auto', px: 2, pt: 0.5 }}>
        {sub?.kind === 'write' ? (
          <WriterRun key={sub.threadId ?? 'new'} start={sub.start} threadId={sub.threadId} language={language} paperType={paperType} onExit={reset} />
        ) : sub?.kind === 'critique' ? (
          <CritiqueRun key={sub.threadId ?? 'new'} threadId={sub.threadId} language={language} paperType={paperType} onExit={reset} />
        ) : status === 'idle' ? (
          <Box>
            <Launcher
              icon={<AccountTreeOutlinedIcon />}
              title="综述"
              desc="从研究主题出发：检索文献 → 你勾选 → 归纳研究现状 → 生成大纲交给你确认"
              hint="在下方输入框写主题，或点一个例子"
            >
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
                {EXAMPLES.map((ex) => (
                  <Chip
                    key={ex}
                    size="small"
                    label={ex}
                    onClick={() => setInput(ex)}
                    sx={{ height: 'auto', py: 0.25, fontSize: 11.5, color: brand.muted, bgcolor: '#fff', border: `1px dashed ${brand.line}`, '& .MuiChip-label': { whiteSpace: 'normal' } }}
                  />
                ))}
              </Box>
            </Launcher>
            <Launcher
              icon={<EditNoteIcon />}
              title="分节写作"
              desc="按文档里的一级标题大纲逐节写草稿：每节单独检索，引用只来自文献库，写完自查引用"
              disabled={!docOutline.length}
              hint={docOutline.length ? `${docOutline.length} 节：${docOutline.map((x) => x.title).join(' · ')}` : '文档里还没有一级标题：先用「综述」生成大纲，或自己写几个一级标题'}
              onClick={() => setSub({ kind: 'write', start: { outline: docOutline, topic: docOutline.map((x) => x.title).join('；'), from: 'doc' } })}
            />
            <Launcher
              icon={<RateReviewOutlinedIcon />}
              title="审稿"
              desc="模拟审稿人挑逻辑和论据上的问题，意见挂在正文旁边；你修改后再审一轮，最多几轮"
              disabled={docChars < 50}
              hint={docChars >= 50 ? `当前正文约 ${docChars} 字` : '正文还太少，写一些内容再来审稿'}
              onClick={() => setSub({ kind: 'critique' })}
            />
            <Typography sx={{ fontSize: 11.5, color: brand.faint, mt: 1.5, lineHeight: 1.6 }}>
              每次暂停都会保存，刷新页面或关掉程序后也能接着做；右上角「历史」可以回到以前的任务。
            </Typography>
          </Box>
        ) : (
          <Thread>
            <ThreadItem icon={<PersonOutlineIcon />}>
              <Card variant="user">
                {topic}
                <Box sx={{ fontSize: 11.5, color: brand.muted, mt: 0.25 }}>
                  {language === 'zh' ? '中文' : 'English'} · {paperType} · {maxPapers} 篇
                </Box>
              </Card>
            </ThreadItem>

            <ThreadItem icon={<AccountTreeOutlinedIcon sx={{ color: `${brand.purple} !important` }} />}>
              <Card variant="agent" sx={{ py: 0.75 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: steps.length || status === 'running' ? 0.5 : 0 }}>
                  <Typography sx={{ fontSize: 13, fontWeight: 500, color: brand.purple, flex: 1 }}>综述 Agent</Typography>
                  {total > 0 && <Typography sx={{ fontSize: 11.5, color: brand.muted }}>{fmt(total)}</Typography>}
                  {status === 'running' ? (
                    <Button size="small" disabled={stopping} startIcon={<StopCircleOutlinedIcon />} onClick={() => stop(threadId)} sx={{ ml: 0.5, color: brand.red, py: 0, minHeight: 24, fontSize: 12 }}>
                      {stopping ? '正在保存并停止…' : '停止'}
                    </Button>
                  ) : (
                    <Tooltip title="返回首页">
                      <IconButton onClick={reset} sx={{ ml: 0.5, p: 0.25 }} aria-label="返回">
                        <RestartAltIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  )}
                </Box>
                {steps.map((s, i) => (
                    <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 0.75, py: 0.2 }}>
                      <CheckCircleIcon sx={{ fontSize: 14, color: brand.green }} />
                      <Typography sx={{ fontSize: 12.5, flex: 1 }}>{s.label}</Typography>
                      <Typography sx={{ fontSize: 11.5, color: brand.faint }}>
                        {['select_papers', 'review_outline'].includes(s.node) ? '你已确认' : `完成 · ${fmt(s.ms)}`}
                      </Typography>
                    </Box>
                  ))}
                {!steps.length && status !== 'running' && (
                  <Typography sx={{ fontSize: 12, color: brand.muted }}>
                    {status === 'waiting' ? '已从上次中断处恢复，等你确认后继续' : status === 'done' ? '已完成（从保存的状态恢复）' : status === 'stopped' ? '已停止' : ''}
                  </Typography>
                )}
                {status === 'running' && (
                  <Box sx={{ display: 'flex', gap: 0.75, py: 0.2, alignItems: 'flex-start' }}>
                    <CircularProgress size={12} sx={{ mt: '3px', mx: '1px', color: brand.purple }} />
                    <Box sx={{ minWidth: 0 }}>
                      {(log.length ? log : ['处理中…']).map((l, i) => (
                        <Typography key={i} noWrap sx={{ fontSize: 12, color: i === log.length - 1 ? brand.ink : brand.faint }}>
                          {l}
                        </Typography>
                      ))}
                    </Box>
                  </Box>
                )}
              </Card>
            </ThreadItem>

            {status === 'stopped' && !pending && (
              <ThreadItem icon={<PauseCircleFilledIcon sx={{ color: `${brand.faint} !important` }} />}>
                <Card sx={{ fontSize: 12.5, color: brand.muted, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ flex: 1 }}>已停止。已经完成的步骤都保存了，可以从停下的地方继续。</Box>
                  <Button size="small" variant="outlined" onClick={cont}>
                    继续
                  </Button>
                </Card>
              </ThreadItem>
            )}

            {pending && (
              <ThreadItem icon={<PauseCircleFilledIcon sx={{ color: `${brand.orange} !important` }} />}>
                {pending.kind === 'select_papers' && <SelectPapersCard value={pending} onSubmit={resume} />}
                {pending.kind === 'review_outline' && <ReviewOutlineCard value={pending} onSubmit={resume} />}
              </ThreadItem>
            )}

            {error && (
              <ThreadItem icon={<ErrorIcon sx={{ color: `${brand.red} !important` }} />}>
                <Card sx={{ color: brand.red, fontSize: 12.5 }}>{error}</Card>
              </ThreadItem>
            )}

            {status === 'done' && result && (
              <ThreadItem icon={<CheckCircleIcon sx={{ color: `${brand.green} !important` }} />}>
                {result.outline?.length ? (
                  <Card variant="data">
                    <Typography sx={{ fontSize: 13, fontWeight: 500, mb: 0.75 }}>大纲 · {result.outline.length} 节</Typography>
                    {result.outline.map((s, i) => (
                      <Box key={i} sx={{ mb: 0.75 }}>
                        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>
                          {i + 1}. {s.title}
                        </Typography>
                        <Typography sx={{ fontSize: 12, color: brand.muted }}>{s.goal}</Typography>
                      </Box>
                    ))}
                    <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
                      <Button variant="contained" startIcon={<NoteAddOutlinedIcon />} onClick={insertOutline} disabled={!editor}>
                        写入文档
                      </Button>
                      <Button
                        variant="outlined"
                        startIcon={<EditNoteIcon />}
                        disabled={!editor}
                        onClick={() => setSub({ kind: 'write', start: { outline: result.outline!, topic: result.topic ?? topic, from: 'agent' } })}
                        sx={{ bgcolor: '#fff' }}
                      >
                        逐节写作
                      </Button>
                      {result.synthesis && (
                        <Button variant="outlined" onClick={insertSynthesis} disabled={!editor} sx={{ bgcolor: '#fff' }}>
                          插入研究现状
                        </Button>
                      )}
                    </Box>
                  </Card>
                ) : (
                  <Card sx={{ color: brand.muted }}>任务已结束。</Card>
                )}
                {!!result.errors?.length && (
                  <Box sx={{ mt: 0.75 }}>
                    {result.errors.map((e, i) => (
                      <Typography key={i} sx={{ fontSize: 11.5, color: e.startsWith('提示') ? brand.muted : brand.amber }}>
                        {e.startsWith('提示') ? '💡 ' : '⚠ '}
                        {e}
                      </Typography>
                    ))}
                  </Box>
                )}
              </ThreadItem>
            )}
          </Thread>
        )}
      </Box>
      <RunStatsBar />
      <Composer
        value={input}
        onChange={setInput}
        onSubmit={() => start(input.trim())}
        busy={running}
        disabled={!docId || !!sub}
        placeholder={status === 'idle' ? '输入研究主题或论文题目…' : '输入新主题，开始新的综述任务…'}
        left={
          <>
            <MiniSelect label="写作语言" value={language} onChange={setLanguage} options={[['zh', '中文'], ['en', 'English']]} />
            <MiniSelect
              label="论文类型"
              value={paperType}
              onChange={setPaperType}
              options={['学术论文', '毕业论文', '综述', '会议论文', '课程报告'].map((t) => [t, t] as [string, string])}
            />
            <MiniSelect label="目标文献数" value={maxPapers} onChange={setMaxPapers} options={[4, 6, 8, 12, 16].map((n) => [n, `${n} 篇`] as [number, string])} />
            <MiniSelect label="是否自动下载开放获取全文" value={fetchPdf} onChange={setFetchPdf} options={[['yes', '下载全文'], ['no', '仅摘要']]} />
          </>
        }
      />
    </>
  )
}
