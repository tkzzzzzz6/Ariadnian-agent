import { useEffect, useRef, useState } from 'react'
import { Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle,
  FormControlLabel, IconButton, MenuItem, Switch, TextField, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { api } from '../lib/api'
import { traceDuration, traceStatus, type TraceList, type TracePage, type TraceSpan } from '../lib/trace'
import { brand } from '../theme'

const phase: Record<string, string> = { waiting: '等待模型响应', reasoning: '正在接收推理内容', output: '正在生成正文', finished: '调用结束' }
const kind = { node: '步骤', tool: '工具', llm: '模型' }
const num = (n?: number | null) => n == null ? '未返回' : n.toLocaleString()

function SpanRow({ span: s, tid, rid, depth, now, capture, runStarted }: { span: TraceSpan; tid: string; rid: string; depth: number; now: number; capture: boolean; runStarted: number }) {
  const [open, setOpen] = useState(s.status === 'error' || !!s.data.handled_error)
  const [detail, setDetail] = useState<Record<string, string> | null>(null)
  const [error, setError] = useState('')
  const [loaded, setLoaded] = useState(false)
  useEffect(() => {
    if (!open || !s.has_detail) return
    let cancelled = false
    api.agentTraceDetail(tid, rid, s.id).then(r => { if (!cancelled) { setDetail(r.detail); setLoaded(true); setError('') } })
      .catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [open, tid, rid, s.id, s.has_detail, s.status])
  const duration = s.duration_ms ?? (s.status === 'running' ? Math.max(0, now / 1000 - s.started) * 1000 : null)
  const problem = s.status === 'error' || !!s.data.handled_error
  return <Box sx={{ ml: Math.min(depth, 2) * 1.5, borderBottom: `1px solid ${brand.lineSoft}`, minWidth: 0 }}>
    <Button fullWidth onClick={() => setOpen(!open)} aria-expanded={open} sx={{ py: 1, px: 0.5, gap: 0.75, justifyContent: 'flex-start', color: brand.ink, textAlign: 'left' }}>
      <ExpandMoreIcon sx={{ fontSize: 16, transform: open ? undefined : 'rotate(-90deg)', flexShrink: 0 }} />
      <Typography component="span" sx={{ fontSize: 11, color: brand.muted, flexShrink: 0 }}>{kind[s.kind]}</Typography>
      <Box component="span" sx={{ flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
        <Typography component="span" sx={{ fontSize: 13 }}>{s.label}</Typography>
        <Typography component="span" sx={{ display: 'block', fontSize: 11, color: brand.muted }}>开始于 +{traceDuration((s.started - runStarted) * 1000)}</Typography>
      </Box>
      <Box component="span" sx={{ fontSize: 11.5, textAlign: 'right', flexShrink: 0, maxWidth: { xs: '38%', sm: '45%' }, whiteSpace: 'normal', color: problem ? brand.red : brand.muted }}>
        {s.status === 'running' && s.kind === 'llm' ? phase[s.data.phase ?? 'waiting'] : traceStatus[s.status] ?? s.status}
        {s.data.warnings?.length || s.data.handled_error ? ' · 有异常' : ''}
        {s.kind === 'node' && s.data.call_count ? ` · ${s.data.call_count} 次模型调用` : ''}
        <Box component="span" sx={{ display: 'block', fontVariantNumeric: 'tabular-nums' }}>{duration == null ? '耗时未知' : traceDuration(duration)}</Box>
      </Box>
    </Button>
    {open && <Box sx={{ pl: 3, pb: 1.5, fontSize: 12, overflowWrap: 'anywhere', '& p': { my: 0.5 } }}>
      <Typography sx={{ fontSize: 11.5, color: brand.muted }}>内部标识：{s.name} · {s.id.slice(0, 8)}</Typography>
      {s.data.node_label && <Typography sx={{ fontSize: 11.5, color: brand.muted }}>所属步骤：{s.data.node_label}</Typography>}
      {s.kind === 'llm' && <>
        <Typography component="p" sx={{ fontSize: 12 }}>{s.data.model ?? '等待模型信息'}{s.data.estimated ? ' · 离线演示，Token 为估算' : ''}</Typography>
        <Typography component="p" sx={{ fontSize: 12 }}>输入 {num(s.data.prompt_tokens)} / 输出 {num(s.data.completion_tokens)} Token</Typography>
        <Typography component="p" sx={{ fontSize: 12 }}>首次正文：{s.data.first_token_ms == null ? '未记录' : traceDuration(s.data.first_token_ms)}
          {s.data.first_reasoning_ms != null && ` · 首次推理内容：${traceDuration(s.data.first_reasoning_ms)}`}</Typography>
        {s.data.reasoning_tokens != null && <Typography component="p" sx={{ fontSize: 12 }}>推理 Token：{num(s.data.reasoning_tokens)}（可能包含在输出用量中，不重复相加）</Typography>}
        <Typography component="p" sx={{ fontSize: 12, color: brand.muted }}>{s.status === 'running' && !s.data.reasoning_available ? '尚未收到推理信息'
          : !s.data.reasoning_available ? '此调用未提供可展示的推理信息' : !capture ? '模型返回了推理信息，内容采集未开启' : '推理内容由模型服务返回'}</Typography>
      </>}
      {s.data.result_count != null && <Typography component="p" sx={{ fontSize: 12 }}>返回 {s.data.result_count} 项</Typography>}
      {s.data.source && <Typography component="p" sx={{ fontSize: 12 }}>来源：{s.data.source}{s.data.cache_hit ? ' · 命中本地缓存' : ' · 查询外部来源'}</Typography>}
      {(s.data.error || s.data.handled_error) && <Alert severity="error" sx={{ my: 1 }}>{s.data.error || s.data.handled_error}</Alert>}
      {s.data.warnings?.map((w, i) => <Alert severity="warning" key={i} sx={{ my: 0.5 }}>{w}</Alert>)}
      {s.kind !== 'node' && !s.has_detail && <Typography sx={{ fontSize: 12, color: brand.muted }}>{capture ? '暂无详细内容或已按保留设置清理' : '详细输入输出未采集'}</Typography>}
      {!!s.has_detail && !loaded && !error && <CircularProgress size={14} />}
      {error && <Alert severity="warning">{error}</Alert>}
      {detail && Object.entries(detail).map(([key, value]) => <Box key={key} sx={{ mt: 1 }}>
        <Typography sx={{ fontSize: 12, fontWeight: 500 }}>{({ input: '输入（不含系统提示）', output: '输出', reasoning: '模型返回的推理内容 / 摘要' } as Record<string, string>)[key] ?? key}</Typography>
        <Box component="pre" sx={{ m: 0, mt: 0.5, p: 1, bgcolor: brand.subtle, fontSize: 12, fontFamily: 'inherit', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{value}</Box>
      </Box>)}
    </Box>}
  </Box>
}

export function RunTraceDialog({ threadId: tid, onClose }: { threadId: string; onClose: () => void }) {
  const [list, setList] = useState<TraceList | null>(null)
  const [rid, setRid] = useState('')
  const [page, setPage] = useState<TracePage | null>(null)
  const [offset, setOffset] = useState(0)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [now, setNow] = useState(Date.now())
  const [settings, setSettings] = useState(false)
  const [locked, setLocked] = useState(true)
  const [capture, setCapture] = useState(false)
  const [days, setDays] = useState('30')
  const [detailDays, setDetailDays] = useState('7')
  const [busy, setBusy] = useState(false)
  const [confirmClear, setConfirmClear] = useState(false)
  const [diagnostic, setDiagnostic] = useState('')
  const selected = useRef('')
  selected.current = rid

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    async function refresh() {
      if (document.visibilityState === 'hidden') { timer = setTimeout(refresh, 2000); return }
      try {
        const result = await api.agentTraces(tid)
        if (!alive) return
        setList(result); setError('')
        const id = selected.current || result.runs[0]?.id
        if (!selected.current && id) setRid(id)
      } catch (e: any) { if (alive) setError(e.message) }
      if (alive) timer = setTimeout(refresh, 2000)
    }
    void refresh()
    api.settings().then(s => { if (alive) setLocked(!!s.locked) }).catch(() => undefined)
    return () => { alive = false; clearTimeout(timer) }
  }, [tid])

  useEffect(() => {
    if (!rid) return
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    setPage(null); setDiagnostic('')
    async function refresh() {
      if (document.visibilityState === 'hidden') { timer = setTimeout(refresh, 1500); return }
      try {
        const result = await api.agentTrace(tid, rid, offset)
        if (!alive) return
        setPage(result); setError(''); setNow(Date.now())
        if (['running', 'stopping'].includes(result.run.status)) timer = setTimeout(refresh, 1500)
      } catch (e: any) { if (alive) setError(e.message) }
    }
    void refresh()
    return () => { alive = false; clearTimeout(timer) }
  }, [tid, rid, offset])

  const run = page?.run
  const latest = list?.runs[0]?.id === rid
  useEffect(() => {
    if (!run || !['running', 'stopping', 'waiting'].includes(run.status) || !latest) return
    const timer = setInterval(() => { if (document.visibilityState !== 'hidden') setNow(Date.now()) }, 1000)
    return () => clearInterval(timer)
  }, [run?.status, latest])
  const nextStart = run ? list?.runs.filter(r => r.started > run.started).at(-1)?.started : undefined
  const elapsed = run?.ended != null ? (run.ended - run.started) * 1000
    : run && ['running', 'stopping'].includes(run.status) ? now - run.started * 1000 : null
  const running = list?.runs.some(r => ['running', 'stopping'].includes(r.status))
  const latestEvent = page?.events.at(-1)
  const parentMap = new Map(page?.spans.map(s => [s.id, s]))
  const ordered: TraceSpan[] = []
  const append = (item: TraceSpan) => {
    ordered.push(item)
    page?.spans.filter(s => s.parent_id === item.id).forEach(append)
  }
  page?.spans.filter(s => !s.parent_id || !parentMap.has(s.parent_id)).forEach(append)
  const depth = (s: TraceSpan) => { let d = 0; let p = s.parent_id; while (p && d < 3) { d++; p = parentMap.get(p)?.parent_id ?? null } return d }
  const perform = async (fn: () => Promise<void>) => { setBusy(true); setNotice(''); try { await fn() } catch (e: any) { setNotice(e.message) } finally { setBusy(false) } }

  return <Dialog open onClose={onClose} fullWidth maxWidth="md" aria-labelledby="run-trace-title">
    <DialogTitle id="run-trace-title" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>运行详情
      <Box sx={{ flex: 1 }} /><Button onClick={() => { setSettings(!settings); if (list) { setCapture(list.policy.capture); setDays(String(list.policy.retention_days)); setDetailDays(String(list.policy.detail_days)) } }}>记录设置</Button>
      <IconButton onClick={onClose} aria-label="关闭运行详情"><CloseIcon /></IconButton>
    </DialogTitle>
    <DialogContent dividers sx={{ minHeight: 180 }}>
      {settings && <Box sx={{ mb: 2, p: 1.5, bgcolor: brand.subtle }}>
        <FormControlLabel control={<Switch checked={capture} disabled={locked || busy} onChange={e => setCapture(e.target.checked)} />} label="采集详细输入、输出及模型返回的推理内容" />
        <Typography sx={{ fontSize: 12, color: brand.muted, mb: 1 }}>仅影响之后开始或恢复的运行；脱敏并截断内容，不采集系统提示。查看记录不会触发模型调用。</Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField label="基础记录保留天数" type="number" size="small" value={days} disabled={locked || busy} onChange={e => setDays(e.target.value)} sx={{ width: 165 }} />
          <TextField label="详细内容保留天数" type="number" size="small" value={detailDays} disabled={locked || busy} onChange={e => setDetailDays(e.target.value)} sx={{ width: 165 }} />
          <Button disabled={locked || busy} onClick={() => perform(async () => {
            await api.saveSettings({ TRACE_CAPTURE: capture ? '1' : '0', TRACE_RETENTION_DAYS: days, TRACE_DETAIL_DAYS: detailDays })
            setList(await api.agentTraces(tid)); setNotice('设置已保存，作用于后续运行'); setSettings(false)
          })}>保存设置</Button>
        </Box>
        {locked && <Typography sx={{ fontSize: 12 }}>管理员已锁定设置</Typography>}
      </Box>}
      {error && <Alert severity="warning" sx={{ mb: 1 }}>{error}</Alert>}
      {notice && <Alert severity="info" sx={{ mb: 1 }}>{notice}</Alert>}
      {!list && !error && <CircularProgress size={20} aria-label="加载运行记录" />}
      {list && !list.runs.length && <Typography sx={{ color: brand.muted }}>此任务没有可用的执行记录；可能创建于追踪功能启用之前，或记录已被清理。</Typography>}
      {!!list?.runs.length && <TextField select fullWidth size="small" label="本次运行" value={rid || list.runs[0].id} onChange={e => { setRid(e.target.value); setOffset(0) }} sx={{ mb: 2 }}>
        {list.runs.map((r, i) => <MenuItem key={r.id} value={r.id}>第 {list.runs.length - i} 次运行 · {new Date(r.started * 1000).toLocaleString()} · {traceStatus[r.status]}</MenuItem>)}
      </TextField>}
      {run && <>
        <Box sx={{ p: 1.5, mb: 1.5, bgcolor: 'rgba(135,100,184,0.07)', borderRadius: 1 }}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}><Chip size="small" label={traceStatus[run.status] ?? run.status} color={run.status === 'error' ? 'error' : 'default'} />
            <Typography sx={{ fontSize: 12 }}>本次执行 {elapsed == null ? '时长记录不完整' : traceDuration(elapsed)}</Typography>
            {run.status === 'waiting' && run.ended && <Typography sx={{ fontSize: 12 }}>{latest ? '等待确认' : '随后等待'} {traceDuration(Math.max(0, (nextStart ?? now / 1000) - run.ended) * 1000)}</Typography>}
          </Box>
          {latestEvent && <Typography sx={{ fontSize: 12, mt: 0.75, overflowWrap: 'anywhere' }}>最近进展：{latestEvent.message}</Typography>}
          {page && <Typography sx={{ fontSize: 12, mt: 0.75 }}>{page.summary.steps} 步 · {page.summary.tools} 次工具调用 · {page.summary.calls} 次模型调用
            {page.summary.calls > 0 && page.summary.unknown_calls === page.summary.calls ? ' · Token 未返回' : ` · 已知 Token ${num(page.summary.known_tokens ?? 0)}`}{page.summary.estimated ? '（含离线估算）' : ''}
            {page.summary.unknown_calls ? ` · ${page.summary.unknown_calls} 次用量未返回` : ''}</Typography>}
          <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.5 }}>并行步骤耗时分别计算，不相加为总耗时。{run.status === 'waiting' ? '请关闭详情，回到任务卡片确认后继续。' : ''}</Typography>
        </Box>
        {!!run.incomplete && <Alert severity="warning" sx={{ mb: 1 }}>部分记录可能缺失；耗时与调用量可能不完整。</Alert>}
        {run.error && <Alert severity="error" sx={{ mb: 1 }}>{run.error}</Alert>}
        {!page?.spans.length && <Typography sx={{ fontSize: 12, color: brand.muted }}>暂无步骤记录；执行刚开始或记录写入失败时可能出现此状态。</Typography>}
        {ordered.map(s => <SpanRow key={`${rid}-${s.id}`} span={s} tid={tid} rid={rid} depth={depth(s)} now={now} capture={!!run.capture} runStarted={run.started} />)}
        <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between' }}>
          <Button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 50))}>上一页</Button>
          <Typography sx={{ fontSize: 12, alignSelf: 'center' }}>第 {offset + 1}—{offset + (page?.spans.length ?? 0)} 条</Typography>
          <Button disabled={!page?.has_more} onClick={() => setOffset(offset + 50)}>下一页</Button>
        </Box>
        <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 1 }}>详细内容{run.capture ? '已开启采集；展开调用查看' : '未采集'}。模型内部重试不一定对应用可见；历史回看不等于重新执行。</Typography>
        {diagnostic && <TextField fullWidth multiline minRows={3} label="诊断摘要（可手动复制）" value={diagnostic} slotProps={{ input: { readOnly: true } }} sx={{ mt: 2 }} />}
      </>}
    </DialogContent>
    <DialogActions sx={{ flexWrap: 'wrap', gap: 0.5 }}>
      <Button disabled={busy || !run} onClick={() => perform(async () => {
        const text = JSON.stringify(await api.agentDiagnostic(tid, rid), null, 2)
        setDiagnostic(text)
        try { await navigator.clipboard.writeText(text); setNotice('诊断摘要已复制，不包含完整输入输出与推理内容') }
        catch { setNotice('请在下方诊断摘要中手动复制') }
      })}>复制诊断摘要</Button>
      <Button color="error" disabled={busy || !!running || locked || !list?.runs.length} onClick={() => setConfirmClear(true)}>清理本任务记录</Button>
      <Button onClick={onClose}>返回任务</Button>
    </DialogActions>
    <Dialog open={confirmClear} onClose={() => setConfirmClear(false)}><DialogTitle>清理本任务的运行记录？</DialogTitle>
      <DialogContent>清理后无法回看这些记录，论文、文档和任务检查点会保留。</DialogContent>
      <DialogActions><Button onClick={() => setConfirmClear(false)}>取消</Button><Button color="error" disabled={busy} onClick={() => perform(async () => {
        await api.agentClearTraces(tid); setConfirmClear(false); setRid(''); setPage(null); setOffset(0); setList(await api.agentTraces(tid)); setNotice('运行记录已清理')
      })}>确认清理</Button></DialogActions>
    </Dialog>
  </Dialog>
}
