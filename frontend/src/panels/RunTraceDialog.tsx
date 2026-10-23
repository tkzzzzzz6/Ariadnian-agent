import { useEffect, useState } from 'react'
import { Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle,
  FormControlLabel, IconButton, MenuItem, Switch, TextField, Tooltip, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import PushPinOutlinedIcon from '@mui/icons-material/PushPinOutlined'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import { api } from '../lib/api'
import { useApp } from '../lib/appContext'
import { useRunStats } from '../lib/runStats'
import { currentMessage, modelPhase } from '../lib/traceView'
import { useTrace } from '../lib/useTrace'
import { traceDuration, traceStatus, type TraceIssue } from '../lib/trace'
import { brand } from '../theme'
import { TraceStage, TraceTimeline } from './TraceStages'

export function RunTraceDialog({ threadId, onClose }: { threadId: string; onClose: () => void }) {
  return <TraceInspector threadId={threadId} onClose={onClose} active mode="dialog" />
}

export function RunTracePanel({ threadId, active }: { threadId: string; active: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const { pinTrace } = useApp()
  return <>
    <TraceInspector threadId={threadId} active={active && !expanded} mode="sidebar" onClose={() => pinTrace(null)} onExpand={() => setExpanded(true)} />
    {expanded && <RunTraceDialog threadId={threadId} onClose={() => setExpanded(false)} />}
  </>
}

function TraceInspector({ threadId: tid, onClose, active, mode, onExpand }: {
  threadId: string; onClose: () => void; active: boolean; mode: 'dialog' | 'sidebar'; onExpand?: () => void
}) {
  const { list, page, rid, setRid, error, now, revision, reload } = useTrace(tid, active)
  const { pinTrace, openAgentTask, docId, notify } = useApp()
  const current = useRunStats()
  const [view, setView] = useState<'steps' | 'timeline'>('steps')
  const [onlyIssues, setOnlyIssues] = useState(false)
  const [focus, setFocus] = useState<TraceIssue | null>(null)
  const [settings, setSettings] = useState(false)
  const [locked, setLocked] = useState(true)
  const [capture, setCapture] = useState(false)
  const [days, setDays] = useState('30')
  const [detailDays, setDetailDays] = useState('7')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [confirmClear, setConfirmClear] = useState(false)
  const [diagnostic, setDiagnostic] = useState('')
  useEffect(() => {
    if (!active) return
    let cancelled = false
    api.settings().then(s => { if (!cancelled) setLocked(!!s.locked) }).catch(() => undefined)
    return () => { cancelled = true }
  }, [active, tid])
  useEffect(() => { setFocus(null); setDiagnostic(''); setOnlyIssues(false) }, [rid])
  const run = page?.run
  const activeRun = run && ['running', 'stopping'].includes(run.status)
  const issues = page?.issues ?? []
  const recovered = issues.filter(i => i.state === 'recovered').length
  const nodes = page?.spans ?? []
  const activeNodes = nodes.filter(n => n.status === 'running')
  const planned = nodes.find(n => n.name === 'prepare')?.data.planned_sections
  const chapters = nodes.filter(n => ['write_one', 'write_section'].includes(n.name))
  const quiet = !!activeRun && now / 1000 - (page?.last_activity ?? now / 1000) >= 30
  const elapsed = run?.ended != null ? traceDuration((run.ended - run.started) * 1000)
    : activeRun ? traceDuration(now - run!.started * 1000) : '时长记录不完整'
  const nextStart = run ? list?.runs.filter(r => r.started > run.started).at(-1)?.started : undefined
  const perform = async (fn: () => Promise<void>) => {
    setBusy(true); setNotice('')
    try { await fn() } catch (e: any) { setNotice(e.message) } finally { setBusy(false) }
  }
  const returnTask = () => {
    if (!page?.task) return notify('任务信息暂时不可用，请从历史列表打开', 'info')
    if (page.task.doc_id !== docId) return notify('请切回该任务对应的文档', 'info')
    if (current.running && current.threadId !== tid) return notify('另一个任务正在运行，请等待保存后再切换', 'info')
    openAgentTask(page.task, current.threadId !== tid)
    if (mode === 'dialog') onClose()
  }
  const jump = (issue?: TraceIssue) => {
    const item = issue ?? issues[(issues.findIndex(i => i.id === focus?.id) + 1) % issues.length]
    if (!item) return
    setFocus(previous => ({ ...item, navigation: (previous?.navigation ?? 0) + 1 })); setView('steps')
  }
  const header = <Box sx={{ p: mode === 'dialog' ? 2 : 1.25, pb: 1, display: 'flex', gap: 0.5, alignItems: 'center', flexWrap: 'wrap' }}>
    <Typography component="h2" sx={{ fontSize: mode === 'dialog' ? 18 : 14, fontWeight: 500, m: 0, flex: 1 }}>{mode === 'dialog' ? '运行详情' : '固定观察'}</Typography>
    <Button onClick={() => { setSettings(!settings); if (list) { setCapture(list.policy.capture); setDays(String(list.policy.retention_days)); setDetailDays(String(list.policy.detail_days)) } }}>记录设置</Button>
    {mode === 'dialog' ? <Tooltip title="固定到右侧，可同时审阅正文"><IconButton aria-label="固定到观察侧栏" onClick={() => { pinTrace(tid); onClose() }}><PushPinOutlinedIcon fontSize="small" /></IconButton></Tooltip>
      : <IconButton aria-label="放大运行详情" onClick={onExpand}><OpenInFullIcon fontSize="small" /></IconButton>}
    <IconButton onClick={onClose} aria-label={mode === 'dialog' ? '关闭运行详情' : '取消固定观察'}><CloseIcon fontSize="small" /></IconButton>
  </Box>
  const body = <Box sx={{ p: mode === 'dialog' ? 2 : 1.25, pt: 0, overflowY: 'auto', minHeight: 100, flex: 1 }}>
    {settings && <Box sx={{ p: 1, my: 1, bgcolor: brand.subtle }}>
      <FormControlLabel control={<Switch checked={capture} disabled={locked || busy} onChange={e => setCapture(e.target.checked)} />} label="采集详细输入、输出及模型返回的推理内容" />
      <Typography sx={{ fontSize: 12, color: brand.muted, mb: 1 }}>仅影响之后开始或恢复的运行；脱敏、截断，不采集系统提示。</Typography>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <TextField label="基础记录保留天数" size="small" type="number" value={days} onChange={e => setDays(e.target.value)} disabled={locked || busy} sx={{ width: 155 }} />
        <TextField label="详细内容保留天数" size="small" type="number" value={detailDays} onChange={e => setDetailDays(e.target.value)} disabled={locked || busy} sx={{ width: 155 }} />
        <Button disabled={locked || busy} onClick={() => perform(async () => {
          await api.saveSettings({ TRACE_CAPTURE: capture ? '1' : '0', TRACE_RETENTION_DAYS: days, TRACE_DETAIL_DAYS: detailDays })
          setSettings(false); setNotice('设置已保存，作用于后续运行'); reload()
        })}>保存设置</Button>
      </Box>
      {locked && <Typography sx={{ fontSize: 12 }}>管理员已锁定设置</Typography>}
    </Box>}
    {error && <Alert severity="warning" sx={{ my: 1 }}>{error}<Button onClick={reload}>重试读取</Button></Alert>}
    {notice && <Alert severity="info" sx={{ my: 1 }}>{notice}</Alert>}
    {!list && !error && <CircularProgress size={20} aria-label="加载运行记录" />}
    {list && !list.runs.length && <Typography sx={{ color: brand.muted, py: 2 }}>没有可用的执行记录；任务可能创建于追踪功能启用之前，或记录已清理。</Typography>}
    {!!list?.runs.length && <TextField select fullWidth size="small" label="查看哪次运行" value={rid || list.runs[0].id} onChange={e => setRid(e.target.value)} sx={{ mt: 1, mb: 1.5 }}>
      {list.runs.map((r, i) => <MenuItem key={r.id} value={r.id}>第 {list.runs.length - i} 次 · {new Date(r.started * 1000).toLocaleString()} · {traceStatus[r.status]}</MenuItem>)}
    </TextField>}
    {page?.task && <Typography sx={{ fontSize: 12.5, color: brand.muted, mb: 1, overflowWrap: 'anywhere' }}>任务：{page.task.topic}</Typography>}
    {current.running && current.threadId && current.threadId !== tid && <Alert severity="info" sx={{ mb: 1 }}>你正在回看另一个任务，当前还有任务在运行。
      <Button onClick={() => { pinTrace(current.threadId!); if (mode === 'dialog') onClose() }}>观察当前任务</Button>
    </Alert>}
    {page && run && <>
      <Box sx={{ position: 'sticky', top: 0, zIndex: 1, bgcolor: '#f5f1fa', p: 1.5, borderRadius: 1, mb: 1.5 }} aria-label="当前运行状态">
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <Chip label={traceStatus[run.status]} size="small" color={run.status === 'error' ? 'error' : 'default'} />
          <Typography sx={{ fontSize: 12 }}>本次执行 {elapsed}</Typography>
        </Box>
        <Typography sx={{ fontSize: 15, fontWeight: 500, mt: 1, overflowWrap: 'anywhere' }}>{nextStart && run.status === 'waiting' ? '这次运行停在确认点，后续已继续' : currentMessage(page)}</Typography>
        {run.status === 'waiting' && !nextStart && nodes.find(n => n.status === 'waiting') && <Typography sx={{ fontSize: 12, mt: 0.5 }}>当前待处理：{nodes.find(n => n.status === 'waiting')!.label}</Typography>}
        {activeNodes.length > 1 && activeNodes.slice(0, 3).map(n => <Typography key={n.id} sx={{ fontSize: 12 }}>{n.label}</Typography>)}
        {page.active_calls.filter(c => c.kind === 'llm').slice(0, 3).map(c => <Typography key={c.id} sx={{ fontSize: 12, mt: 0.5 }}>{c.data.node_label ? `${c.data.node_label} · ` : ''}{modelPhase[c.data.phase ?? 'waiting']}</Typography>)}
        {planned != null && <Typography sx={{ fontSize: 12, mt: 0.5 }}>计划 {planned} 节 · 已完成 {chapters.filter(n => n.status === 'done' && n.data.outcome !== 'error').length} 节 · 正在处理 {chapters.filter(n => n.status === 'running').length} 节
          {chapters.some(n => n.data.outcome === 'error') ? ` · 失败 ${chapters.filter(n => n.data.outcome === 'error').length} 节` : ''}</Typography>}
        {run.status === 'waiting' && <Typography sx={{ fontSize: 12, mt: 0.5 }}>{nextStart ? '当时等待确认' : '等待确认'} {traceDuration(Math.max(0, (nextStart ?? now / 1000) - (run.ended ?? now / 1000)) * 1000)}{nextStart ? '' : '，确认期间不会继续执行。'}</Typography>}
        {['waiting', 'stopped', 'error', 'interrupted'].includes(run.status) && <Button variant="contained" size="small" onClick={returnTask} sx={{ mt: 1 }}>{nextStart ? '查看当前任务' : run.status === 'waiting' ? '前往任务确认' : '返回任务查看可用操作'}</Button>}
        <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 1 }}>{page.summary.steps} 步 · {page.summary.tools} 次工具 · {page.summary.calls} 次模型调用
          {page.summary.calls > 0 && page.summary.unknown_calls === page.summary.calls ? ' · Token 未返回' : ` · 已知 Token ${(page.summary.known_tokens ?? 0).toLocaleString()}`}
          {page.summary.estimated ? '（含离线估算）' : ''}</Typography>
      </Box>
      {quiet && <Alert severity="info" sx={{ mb: 1 }}>已有 {traceDuration(now - page.last_activity * 1000)} 未收到新执行事件；当前状态仍为{page.active_calls.some(c => c.data.phase === 'waiting') ? '等待模型响应' : '运行中'}，这不表示任务失败。</Alert>}
      {!!run.incomplete && <Alert severity="warning" sx={{ mb: 1 }}>部分记录缺失，耗时与调用量可能不完整。</Alert>}
      {run.error && <Alert severity="error" sx={{ mb: 1 }}>{run.error}</Alert>}
      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center', mb: 1 }}>
        <Button aria-pressed={view === 'steps'} onClick={() => setView('steps')}>步骤</Button><Button aria-pressed={view === 'timeline'} onClick={() => setView('timeline')}>耗时</Button>
        <Box sx={{ flex: 1 }} /><Button disabled={!issues.length} onClick={() => jump()}>跳到异常{issues.length ? `（${issues.length}）` : ''}</Button>
        <Button aria-pressed={onlyIssues} onClick={() => { setOnlyIssues(!onlyIssues); setView('steps') }}>{onlyIssues ? '显示全部步骤' : '仅看异常'}</Button>
      </Box>
      {!!issues.length && <Typography sx={{ fontSize: 12, color: brand.muted, mb: 1 }}>异常记录 {issues.length} 项，其中 {recovered} 项后续尝试已成功。{issues.some(i => i.state === 'error') && run.status === 'done' ? '任务已完成，仍请检查有异常的步骤是否影响结果。' : ''}</Typography>}
      {onlyIssues && issues.map(i => <Button key={i.id} fullWidth onClick={() => jump(i)} sx={{ justifyContent: 'flex-start', textAlign: 'left', mb: 0.5, color: i.state === 'error' ? brand.red : brand.muted }}>{i.state === 'recovered' ? '已恢复' : i.state === 'warning' ? '部分成功 / 提示' : '异常'} · {i.label}</Button>)}
      {view === 'timeline' ? <TraceTimeline data={page} now={now} onSelect={n => { setView('steps'); setOnlyIssues(false); setFocus(previous => ({ id: n.id, node_id: n.id, label: n.label, state: 'warning', message: '', navigation: (previous?.navigation ?? 0) + 1 })) }} />
        : nodes.filter(n => !onlyIssues || issues.some(i => i.node_id === n.id)).map(n => <TraceStage key={`${rid}-${n.id}`} node={n} data={page} tid={tid} now={now} active={active} revision={revision} focus={focus} onNavigate={() => { if (mode === 'dialog') onClose() }} />)}
      {view === 'steps' && !nodes.length && <Typography sx={{ fontSize: 12, color: brand.muted }}>暂无步骤记录，执行刚开始或记录写入失败时可能出现此状态。</Typography>}
      {onlyIssues && !issues.length && <Typography sx={{ fontSize: 12, color: brand.muted }}>没有记录到异常。</Typography>}
      {(page.nodes_truncated || page.issues_truncated) && <Alert severity="info">为控制页面开销，概览最多展示 500 个步骤和 100 条异常。</Alert>}
      <Typography sx={{ fontSize: 11.5, color: brand.muted, my: 1 }}>{page.events.at(-1)?.message ? `最近进展：${page.events.at(-1)!.message}` : ''}</Typography>
      <Typography sx={{ fontSize: 11.5, color: brand.muted }}>详细内容{run.capture ? '已开启采集' : '未采集'}。回看记录不会调用模型，耗时条不表示完成百分比。</Typography>
      {diagnostic && <TextField fullWidth multiline minRows={3} label="诊断摘要（可手动复制）" value={diagnostic} slotProps={{ input: { readOnly: true } }} sx={{ mt: 2 }} />}
    </>}
  </Box>
  const actions = <Box sx={{ p: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap', borderTop: `1px solid ${brand.lineSoft}` }}>
    <Button disabled={busy || !run} onClick={() => perform(async () => {
      const text = JSON.stringify(await api.agentDiagnostic(tid, rid), null, 2); setDiagnostic(text)
      try { await navigator.clipboard.writeText(text); setNotice('诊断摘要已复制，不包含完整输入输出与推理内容') }
      catch { setNotice('请在诊断摘要文本框中手动复制') }
    })}>复制诊断摘要</Button>
    <Button color="error" disabled={busy || list?.runs.some(r => ['running', 'stopping'].includes(r.status)) || locked || !list?.runs.length} onClick={() => setConfirmClear(true)}>清理记录</Button>
    <Button disabled={!page?.task} onClick={returnTask}>返回任务</Button>
  </Box>
  const confirmation = <Dialog open={confirmClear} onClose={() => setConfirmClear(false)}><DialogTitle>清理本任务的运行记录？</DialogTitle>
    <DialogContent>清理后无法回看这些记录，论文、文档和任务检查点会保留。</DialogContent>
    <DialogActions><Button onClick={() => setConfirmClear(false)}>取消</Button><Button color="error" disabled={busy} onClick={() => perform(async () => {
      await api.agentClearTraces(tid); setConfirmClear(false); setNotice('运行记录已清理'); reload()
    })}>确认清理</Button></DialogActions>
  </Dialog>
  if (mode === 'sidebar') return <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>{header}{body}{actions}{confirmation}</Box>
  return <Dialog open onClose={onClose} fullWidth maxWidth="md" aria-label="运行详情"><Box sx={{ display: 'flex', flexDirection: 'column', minHeight: 0, maxHeight: 'calc(100dvh - 64px)' }}>{header}{body}{actions}{confirmation}</Box></Dialog>
}
