import { useEffect, useRef, useState } from 'react'
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { api } from '../lib/api'
import { traceDuration, traceStatus, type TraceIssue, type TraceOverview, type TraceSpan } from '../lib/trace'
import { expandedByDefault, modelPhase, timelineRange } from '../lib/traceView'
import { brand } from '../theme'
import { TraceCallDetails } from './TraceCallDetails'

export function TraceStage({ node: n, data, tid, now, active, revision, focus, onNavigate }: {
  node: TraceSpan; data: TraceOverview; tid: string; now: number; active: boolean; revision: number; focus: TraceIssue | null; onNavigate: () => void
}) {
  const [manual, setManual] = useState<boolean | null>(null)
  const [calls, setCalls] = useState<TraceSpan[] | null>(null)
  const [offset, setOffset] = useState(0)
  const [more, setMore] = useState(false)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const row = useRef<HTMLDivElement>(null)
  const focusedCall = useRef('')
  const issues = data.issues.filter(i => i.node_id === n.id)
  const unresolved = issues.some(i => i.state !== 'recovered')
  const open = manual ?? expandedByDefault(n, unresolved)
  const focused = focus?.node_id === n.id
  const pollRevision = n.status === 'running' ? revision : 0
  useEffect(() => {
    focusedCall.current = ''
    if (focused) { setManual(true); row.current?.scrollIntoView({ block: 'nearest' }) }
  }, [focused, focus?.id, focus?.navigation])
  useEffect(() => {
    if (!open || !active) return
    let cancelled = false
    const target = focused && focus?.id !== focusedCall.current ? focus?.id : ''
    api.agentTraceNode(tid, data.run.id, n.id, offset, target).then(r => {
      if (!cancelled) { focusedCall.current = focus?.id ?? ''; setCalls(r.spans); setMore(r.has_more); setError(''); if (r.offset !== offset) setOffset(r.offset) }
    }).catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [open, active, tid, data.run.id, n.id, n.status, pollRevision, offset, retry, focus?.id, focus?.navigation])
  const liveCall = data.active_calls.filter(c => c.data.node_id === n.id).find(c => c.kind === 'llm')
  const duration = n.duration_ms ?? (n.status === 'running' ? Math.max(0, now - n.started * 1000) : null)
  let status = traceStatus[n.status] ?? n.status
  if (n.status === 'done' && issues.length) status = unresolved ? '已完成，含异常' : '已完成，失败后已恢复'
  if (n.data.outcome === 'error') status = '步骤执行失败'
  return <Box ref={row} data-trace-stage={n.id} sx={{ my: 1, border: `1px solid ${focused ? brand.purple : brand.line}`, borderRadius: 1, bgcolor: '#fff' }}>
    <Button fullWidth aria-expanded={open} onClick={() => setManual(!open)} sx={{ p: 1, gap: 0.75, color: brand.ink, textAlign: 'left', alignItems: 'flex-start' }}>
      {n.status === 'done' && !issues.length ? <CheckCircleOutlineIcon sx={{ fontSize: 18, color: brand.green, mt: 0.2 }} />
        : n.status === 'running' ? <CircularProgress size={15} sx={{ mt: 0.5, flexShrink: 0, color: brand.purple }} />
        : <Box component="span" aria-hidden="true" sx={{ color: unresolved || n.status === 'error' ? brand.red : brand.orange, fontSize: 15 }}>●</Box>}
      <Box component="span" sx={{ flex: 1, minWidth: 0 }}>
        <Typography component="span" sx={{ display: 'block', fontSize: 13, fontWeight: 500, overflowWrap: 'anywhere' }}>{n.label}</Typography>
        <Typography component="span" sx={{ fontSize: 11.5, display: 'block', color: brand.muted }}>{n.data.tool_count ?? 0} 次工具 · {n.data.call_count ?? 0} 次模型调用
          {n.data.result_count != null ? ` · 返回 ${n.data.result_count} 项` : ''}</Typography>
        {liveCall && <Typography component="span" sx={{ display: 'block', fontSize: 12, color: brand.purple }}>{modelPhase[liveCall.data.phase ?? 'waiting']}</Typography>}
      </Box>
      <Typography component="span" sx={{ fontSize: 11.5, maxWidth: '34%', textAlign: 'right', color: unresolved ? brand.orange : brand.muted }}>{status}
        <Box component="span" sx={{ display: 'block' }}>{duration == null ? '耗时未知' : traceDuration(duration)}</Box></Typography>
      <ExpandMoreIcon sx={{ fontSize: 16, flexShrink: 0, transform: open ? undefined : 'rotate(-90deg)' }} />
    </Button>
    {open && <Box sx={{ px: 1.25, pb: 1 }}>
      {n.data.error && <Alert severity="error">{n.data.error}</Alert>}
      {n.data.handled_error && <Alert severity="warning">{n.data.handled_error}</Alert>}
      {error && <Alert severity="warning">{error}<Button onClick={() => setRetry(v => v + 1)}>重试读取</Button></Alert>}
      {!calls && !error && <CircularProgress size={16} aria-label="加载步骤调用" />}
      {calls?.length === 0 && <Typography sx={{ fontSize: 12, color: brand.muted }}>{n.status === 'waiting' ? '请在任务卡片确认后继续。' : '暂无工具或模型调用记录。'}</Typography>}
      {calls && <TraceCallDetails calls={calls} tid={tid} rid={data.run.id} now={now} capture={!!data.run.capture} focusId={focused ? focus?.id ?? '' : ''} navigation={focus?.navigation ?? 0} issues={issues} onNavigate={onNavigate} />}
      {(offset > 0 || more) && <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between' }}>
        <Button disabled={!offset} onClick={() => { setOffset(Math.max(0, offset - 50)) }}>上一页调用</Button>
        <Typography sx={{ fontSize: 11.5, alignSelf: 'center' }}>{offset + 1}—{offset + (calls?.length ?? 0)}</Typography>
        <Button disabled={!more} onClick={() => setOffset(offset + 50)}>下一页调用</Button>
      </Box>}
      <Box component="details" sx={{ fontSize: 11.5, color: brand.muted, mt: 1 }}><summary>步骤技术详情</summary>
        {n.name} · {n.id}<br />开始于 +{traceDuration((n.started - data.run.started) * 1000)}
      </Box>
    </Box>}
  </Box>
}

export function TraceTimeline({ data, now, onSelect }: { data: TraceOverview; now: number; onSelect: (node: TraceSpan) => void }) {
  const end = data.run.ended ?? (['running', 'stopping'].includes(data.run.status) ? null : data.last_activity)
  const range = timelineRange(data.spans, data.run.started, end, now)
  return <Box sx={{ mt: 1 }} aria-label="步骤耗时时间轴">
    <Typography sx={{ fontSize: 12, color: brand.muted, mb: 1 }}>同一时间轴显示重叠执行。点击步骤查看调用；嵌套步骤耗时不相加。{data.run.status === 'interrupted' ? '轴到最后已记录事件，不代表中断结束时间。' : ''}</Typography>
    <Box sx={{ ml: '32%', display: 'flex', justifyContent: 'space-between', fontSize: 11, color: brand.muted }}><span>0 秒</span><span>{traceDuration(range * 500)}</span><span>{traceDuration(range * 1000)}</span></Box>
    {data.spans.map(n => {
      const start = Math.max(0, n.started - data.run.started)
      const end = n.ended ?? (n.status === 'running' ? now / 1000 : n.started)
      const left = Math.min(99.5, start / range * 100)
      const width = Math.min(100 - left, Math.max(0.5, (end - n.started) / range * 100))
      const color = n.status === 'error' || n.data.outcome === 'error' ? brand.red : n.status === 'running' ? brand.purple : n.status === 'done' ? brand.green : brand.faint
      return <Button key={n.id} fullWidth onClick={() => onSelect(n)} aria-label={`查看步骤 ${n.label}`} sx={{ display: 'grid', gridTemplateColumns: '30% 1fr', gap: 1, py: 0.75, px: 0, color: brand.ink, textAlign: 'left' }}>
        <Typography component="span" sx={{ fontSize: 12, overflowWrap: 'anywhere' }}>{n.label}</Typography>
        <Box component="span" sx={{ minWidth: 0 }}>
          <Box component="span" sx={{ display: 'block', position: 'relative', height: 9, bgcolor: brand.pill, borderRadius: 1 }}>
            <Box component="span" sx={{ position: 'absolute', left: `${left}%`, width: `${width}%`, height: '100%', bgcolor: color, borderRadius: 1 }} />
          </Box>
          <Typography component="span" sx={{ display: 'block', fontSize: 11, mt: 0.25, color: brand.muted }}>{n.data.outcome === 'error' ? '步骤执行失败' : traceStatus[n.status]} · {n.duration_ms == null ? n.status === 'running' ? traceDuration((end - n.started) * 1000) : '结束时间未知' : traceDuration(n.duration_ms)}</Typography>
        </Box>
      </Button>
    })}
  </Box>
}
