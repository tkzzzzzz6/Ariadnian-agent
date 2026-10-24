import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Alert, Box, Button, CircularProgress, Link, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { api } from '../lib/api'
import { useApp } from '../lib/appContext'
import { traceDuration, traceStatus, type TraceIssue, type TraceProjection, type TraceSpan } from '../lib/trace'
import { groupAttempts, groupCalls, modelPhase } from '../lib/traceView'
import { brand } from '../theme'

const verdict: Record<string, string> = { supports: '支持', partial: '部分支持', unrelated: '不支持', unknown: '待核实', missing: '文献缺失' }
const num = (n?: number | null) => n == null ? '未返回' : n.toLocaleString()

function FollowText({ text, label }: { text: string; label: string }) {
  const [follow, setFollow] = useState(true)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight }, [text, follow])
  return <Box sx={{ mt: 1 }}>
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><Typography sx={{ fontSize: 12, fontWeight: 500, flex: 1 }}>{label}</Typography>
      <Button size="small" onClick={() => setFollow(!follow)}>{follow ? '暂停跟随' : '回到最新'}</Button></Box>
    <Box ref={ref} onScroll={e => { const x = e.currentTarget; if (x.scrollHeight - x.clientHeight - x.scrollTop > 30) setFollow(false) }}
      component="pre" sx={{ m: 0, p: 1, maxHeight: 220, overflowY: 'auto', bgcolor: brand.subtle, fontSize: 12,
        fontFamily: 'inherit', lineHeight: 1.7, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{text}</Box>
  </Box>
}

function Projection({ value, onNavigate }: { value: TraceProjection; onNavigate: () => void }) {
  const { papers, openPreview, editor, notify } = useApp()
  const locate = (anchor: string) => {
    if (!editor) return notify('当前没有可定位的正文', 'info')
    const text = anchor.replace(/\[@[^\]]+\]/g, '').trim()
    const matches: number[] = []
    editor.state.doc.descendants((node, pos) => {
      if (node.isTextblock && text && node.textContent.includes(text)) matches.push(pos + 1)
    })
    if (matches.length !== 1) return notify(matches.length ? '正文有多处匹配，请结合章节查看' : '正文已修改或尚未写入，暂时无法定位', 'info')
    onNavigate()
    requestAnimationFrame(() => editor.chain().focus().setTextSelection(matches[0]).scrollIntoView().run())
  }
  return <Box sx={{ mt: 1 }}>
    {value.query && <Typography sx={{ fontSize: 12, mb: 0.5 }}>检索词：{value.query}</Typography>}
    {value.anchor && <Button onClick={() => locate(value.anchor!)}>定位对应正文</Button>}
    {value.items.map((item, i) => {
      const paper = papers.find(p => p.id === item.paper_id || (item.bibkey && p.bibkey === item.bibkey))
      const url = item.pdf_url && /^https?:\/\//i.test(item.pdf_url) ? item.pdf_url : undefined
      const readable = !!paper || !!url
      return <Box key={i} sx={{ py: 0.75, borderBottom: `1px solid ${brand.lineSoft}` }}>
        <Typography sx={{ fontSize: 12, fontWeight: 500 }}>{item.title || item.anchor || item.bibkey || `结果 ${i + 1}`}</Typography>
        <Typography sx={{ fontSize: 11.5, color: brand.muted }}>{item.source === 'note' ? '用户笔记' : item.source ?? ''}{item.page != null ? ` · 第 ${item.page} 页` : ''}
          {item.verdict ? ` · ${verdict[item.verdict] ?? item.verdict}` : ''}</Typography>
        {(item.text || item.reason) && <Typography sx={{ fontSize: 12, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{item.text || item.reason}</Typography>}
        {item.checks?.map((check, j) => <Typography key={j} sx={{ fontSize: 12 }}>{verdict[check.verdict ?? ''] ?? '待核实'} · {check.title} · {check.reason}</Typography>)}
        {readable && <Button size="small" onClick={() => {
          onNavigate()
          openPreview({ paper: paper ?? { title: item.title || '论文', authors: [], year: item.year, pdf_url: url }, claim: item.text })
        }}>查看论文{item.page != null ? ` · 依据页 ${item.page}` : ''}</Button>}
        {item.url && /^https?:\/\//i.test(item.url) && <Link href={item.url} target="_blank" rel="noopener noreferrer" sx={{ fontSize: 12, ml: 1 }}>来源网页</Link>}
        {item.anchor && <Button size="small" onClick={() => locate(item.anchor!)}>定位正文</Button>}
      </Box>
    })}
    {value.sampled && <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.5 }}>共 {value.total} 项，仅保留 {value.items.length} 项结果摘要。</Typography>}
  </Box>
}

function CallRow({ span: s, tid, rid, now, capture, focus, reveal, navigation, issue, children, onNavigate }: {
  span: TraceSpan; tid: string; rid: string; now: number; capture: boolean; focus: boolean; reveal: boolean; navigation: number; issue?: TraceIssue; children?: ReactNode; onNavigate: () => void
}) {
  const [manual, setManual] = useState<boolean | null>(null)
  const [detail, setDetail] = useState<Record<string, string> | null>(null)
  const [error, setError] = useState('')
  const row = useRef<HTMLDivElement>(null)
  const open = manual ?? (s.status === 'running' || (s.status === 'error' && issue?.state !== 'recovered'))
  useEffect(() => { if (reveal) setManual(true); if (focus) row.current?.scrollIntoView({ block: 'nearest' }) }, [focus, reveal, navigation])
  useEffect(() => {
    if (!open || !s.has_detail) return
    let cancelled = false
    api.agentTraceDetail(tid, rid, s.id).then(r => { if (!cancelled) { setDetail(r.detail); setError('') } })
      .catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [tid, rid, s.id, open, s.has_detail, s.status, s.data.content_revision])
  let projection: TraceProjection | null = null
  try { if (detail?.structured) projection = JSON.parse(detail.structured) } catch { /* old/truncated content remains under technical details */ }
  const label = issue?.state === 'recovered' ? '失败后已恢复' : s.status === 'running' && s.kind === 'llm'
    ? modelPhase[s.data.phase ?? 'waiting'] : traceStatus[s.status] ?? s.status
  const duration = s.duration_ms ?? (s.status === 'running' ? Math.max(0, now - s.started * 1000) : null)
  return <Box ref={row} sx={{ borderLeft: `2px solid ${focus ? brand.purple : brand.lineSoft}`, pl: 1, my: 0.5 }}>
    <Button fullWidth aria-expanded={open} onClick={() => setManual(!open)} sx={{ p: 0.5, gap: 0.5, color: brand.ink, justifyContent: 'flex-start', textAlign: 'left' }}>
      <ExpandMoreIcon sx={{ fontSize: 16, flexShrink: 0, transform: open ? undefined : 'rotate(-90deg)' }} />
      <Typography component="span" sx={{ fontSize: 12, flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>{s.label}</Typography>
      <Typography component="span" sx={{ fontSize: 11.5, color: issue?.state === 'error' ? brand.red : brand.muted, maxWidth: '42%', textAlign: 'right' }}>{label}{duration != null ? ` · ${traceDuration(duration)}` : ''}</Typography>
    </Button>
    {open && <Box sx={{ px: 1, pb: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
      {s.kind === 'llm' ? <>
        <Typography sx={{ fontSize: 12 }}>{s.data.model ?? '模型信息尚未返回'} · 输入 {num(s.data.prompt_tokens)} / 输出 {num(s.data.completion_tokens)} Token{s.data.estimated ? '（离线估算）' : ''}</Typography>
        <Typography sx={{ fontSize: 11.5, color: brand.muted }}>首次正文 {s.data.first_token_ms == null ? '未记录' : traceDuration(s.data.first_token_ms)}
          {s.data.first_reasoning_ms != null ? ` · 首次推理 ${traceDuration(s.data.first_reasoning_ms)}` : ''}</Typography>
        {s.data.reasoning_tokens != null && <Typography sx={{ fontSize: 11.5, color: brand.muted }}>推理 Token {s.data.reasoning_tokens}，可能已含在输出用量中</Typography>}
        {detail?.reasoning ? <FollowText text={detail.reasoning} label="模型返回的推理内容 / 摘要" /> : <Typography sx={{ fontSize: 12, my: 0.5, color: brand.muted }}>{s.data.reasoning_available
          ? capture ? '正在接收或详细内容已清理' : '模型返回了推理信息；内容采集未开启'
          : s.status === 'running' ? '尚未收到可展示的推理信息' : '此调用未提供可展示的推理信息'}</Typography>}
        {detail?.output && <FollowText text={detail.output} label="模型正文输出" />}
      </> : <>
        {s.data.result_count != null && <Typography sx={{ fontSize: 12 }}>返回 {s.data.result_count} 项{s.data.notes_count != null ? `，其中用户笔记 ${s.data.notes_count} 条` : ''}</Typography>}
        {s.data.checked_count != null && <Typography sx={{ fontSize: 12 }}>检查 {s.data.checked_count} 处引用 · 发现 {s.data.issue_count ?? 0} 个问题</Typography>}
        {s.data.source && <Typography sx={{ fontSize: 12 }}>{s.data.source} · {s.data.cache_hit ? '命中本地缓存' : '查询外部来源'}</Typography>}
        {projection && <Projection value={projection} onNavigate={onNavigate} />}
      </>}
      {(s.data.error || s.data.handled_error) && <Alert severity={issue?.state === 'recovered' ? 'info' : 'error'} sx={{ my: 1 }}>{issue?.state === 'recovered' ? '该次尝试失败，后续尝试已成功。' : ''}{s.data.error || s.data.handled_error}</Alert>}
      {s.data.warnings?.map((w, i) => <Alert key={i} severity="warning" sx={{ my: 0.5 }}>{w}</Alert>)}
      {!capture && <Typography sx={{ fontSize: 11.5, color: brand.muted, my: 0.5 }}>详细内容未采集；可在记录设置中为后续运行开启。</Typography>}
      {error && <Alert severity="warning">{error}<Button onClick={() => { setManual(false); setTimeout(() => setManual(true), 0) }}>重试读取</Button></Alert>}
      {capture && s.has_detail && !detail && !error && <CircularProgress size={14} />}
      <Box component="details" sx={{ mt: 1, fontSize: 11.5, color: brand.muted }}>
        <summary>输入输出与技术详情</summary>
        <Typography sx={{ fontSize: 11.5 }}>内部标识 {s.name} · {s.id}</Typography>
        {detail?.input && <Box component="pre" sx={{ fontFamily: 'inherit', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>输入（不含系统提示）：{detail.input}</Box>}
        {s.kind !== 'llm' && detail?.output && <Box component="pre" sx={{ fontFamily: 'inherit', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>原始输出：{detail.output}</Box>}
      </Box>
      {children}
    </Box>}
  </Box>
}

export function TraceCallDetails({ calls, tid, rid, now, capture, focusId, navigation, issues, onNavigate }: {
  calls: TraceSpan[]; tid: string; rid: string; now: number; capture: boolean; focusId: string; navigation: number; issues: TraceIssue[]; onNavigate: () => void
}) {
  const tree = groupCalls(calls)
  const focusPath = new Set<string>()
  let target = calls.find(c => c.id === focusId)
  while (target && !focusPath.has(target.id)) { focusPath.add(target.id); target = calls.find(c => c.id === target!.parent_id) }
  const render = (parent: string, depth = 0): ReactNode => {
    if (depth > 8) return null
    return groupAttempts(tree.get(parent) ?? []).map(group => {
      const elements = group.map(s => <CallRow key={s.id} span={s} tid={tid} rid={rid} now={now} capture={capture}
        focus={s.id === focusId} reveal={focusPath.has(s.id)} navigation={navigation} issue={issues.find(i => i.id === s.id)} onNavigate={onNavigate}>{render(s.id, depth + 1)}</CallRow>)
      if (group.length === 1) return elements
      const recovered = group.some(s => s.status === 'error') && group.some(s => s.status === 'done')
      return <Box key={group[0].id} sx={{ my: 1, p: 0.5, bgcolor: brand.subtle }}>
        <Typography sx={{ fontSize: 12, fontWeight: 500 }}>同一模型请求 · {group.length} 次尝试{recovered ? ' · 已恢复' : ''}</Typography>{elements}
      </Box>
    })
  }
  return <>{render('')}</>
}
