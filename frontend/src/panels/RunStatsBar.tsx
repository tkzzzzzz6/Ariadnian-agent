/**
 * Agent 面板底部的运行统计条：步数 / 调用次数 / 输出速度 · 总 token / 缓存命中 · 上下文占用。
 * 运行中实时刷新（耗时在本地继续走），历史任务显示保存下来的统计。悬停每一项看明细。
 */
import { useEffect, useState, type ReactNode } from 'react'
import { Box, Button, CircularProgress, Tooltip, Typography } from '@mui/material'
import SpeedOutlinedIcon from '@mui/icons-material/SpeedOutlined'
import StorageOutlinedIcon from '@mui/icons-material/StorageOutlined'
import { fmtTokens, useRunStats, type ContextUse } from '../lib/runStats'
import { brand } from '../theme'
import { RunTraceDialog } from './RunTraceDialog'
import { traceStatus } from '../lib/trace'

function fmtDur(ms: number): string {
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} 秒`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m} 分 ${s % 60} 秒` : `${Math.floor(m / 60)} 小时 ${m % 60} 分`
}

const pct = (c: ContextUse | null) => (c?.max ? Math.min(100, (c.used / c.max) * 100) : null)

function Tip({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <Box component="table" sx={{ borderSpacing: 0, fontSize: 12, '& td': { py: 0.15 }, '& td:last-of-type': { pl: 2, textAlign: 'right', fontVariantNumeric: 'tabular-nums' } }}>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td>{k}</td>
            <td>{v}</td>
          </tr>
        ))}
      </tbody>
    </Box>
  )
}

function Item({ icon, children, tip }: { icon?: ReactNode; children: ReactNode; tip: ReactNode }) {
  return (
    <Tooltip title={tip} placement="top" arrow>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0, cursor: 'default', '& svg': { fontSize: 15, color: brand.faint } }}>
        {icon}
        <Typography noWrap component="span" sx={{ fontSize: 11.5, color: brand.muted, fontVariantNumeric: 'tabular-nums' }}>
          {children}
        </Typography>
      </Box>
    </Tooltip>
  )
}

export function RunStatsBar() {
  const { threadId, stats: s, running, at, step, status, warning } = useRunStats()
  const [details, setDetails] = useState(false)
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [running])
  if (!s) return null

  const elapsed = s.elapsed_ms + (running ? Math.max(0, now - at) : 0)
  const speed = running && s.live_tok_s ? s.live_tok_s : s.tok_s
  const ctx = s.peak_context
  const ctxPct = pct(ctx)
  const color = ctxPct == null ? brand.faint : ctxPct >= 85 ? brand.red : ctxPct >= 60 ? brand.orange : brand.purple
  const models = Object.entries(s.models ?? {})

  return (
    <>
    <Box sx={{ px: 2, pt: 0.75, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 0.75, borderTop: `1px solid ${brand.lineSoft}` }}>
      <Typography sx={{ fontSize: 12, color: status === 'error' ? brand.red : brand.muted, flex: 1, minWidth: 120, overflowWrap: 'anywhere' }}>
        {status ? traceStatus[status] ?? status : '已保存的任务'} · 累计执行 {fmtDur(elapsed)}
        {running && step ? ` · ${step}` : ''}{warning ? ` · ${warning}` : ''}
      </Typography>
      <Button disabled={!threadId} onClick={() => setDetails(true)} sx={{ color: brand.purple, flexShrink: 0 }}>运行详情</Button>
    </Box>
    <Box
      aria-label="Agent 运行统计"
      sx={{
        display: 'flex', alignItems: 'center', gap: 1.5, px: 2, py: 0.5, flexShrink: 0, overflow: 'hidden',
        bgcolor: 'rgba(0,0,0,0.015)', flexWrap: 'wrap',
      }}
    >
      <Item
        icon={<SpeedOutlinedIcon />}
        tip={
          <Tip
            rows={[
              ['图节点步数', s.steps],
              ['模型调用', `${s.calls} 次${s.failed ? `（失败 ${s.failed}）` : ''}`],
              ['平均输出速度', s.tok_s != null ? `${s.tok_s} tok/s` : '—'],
              ...(running && s.live_tok_s ? ([['当前输出速度', `${s.live_tok_s} tok/s${(s.streaming ?? 0) > 1 ? `（${s.streaming} 路并行）` : ''}`]] as [string, ReactNode][]) : []),
              ['运行时间', fmtDur(elapsed)],
            ]}
          />
        }
      >
        {s.steps} 步 · {s.calls} 次调用{speed != null && ` · ${Math.round(speed)} tok/s`}
      </Item>
      <Item
        icon={<StorageOutlinedIcon />}
        tip={
          <Tip
            rows={[
              ['输入', fmtTokens(s.prompt_tokens)],
              ['　其中命中缓存', s.cache_hit != null ? `${fmtTokens(s.cached_tokens)}（${Math.round(s.cache_hit * 100)}%）` : '服务未返回'],
              ['输出', fmtTokens(s.completion_tokens)],
              ['费用', s.cost_usd ? `$${s.cost_usd < 0.01 ? s.cost_usd.toFixed(4) : s.cost_usd.toFixed(2)}` : '—'],
              ...models.map(([m, n]) => [m, `${n} 次`] as [string, ReactNode]),
            ]}
          />
        }
      >
        {fmtTokens(s.total_tokens)} tok{s.cache_hit != null && ` · 缓存命中 ${Math.round(s.cache_hit * 100)}%`}
      </Item>
      <Box sx={{ flex: 1 }} />
      {ctx && (
        <Item
          tip={
            <Tip
              rows={[
                ['单次调用最大上下文', ctx.max ? `${fmtTokens(ctx.used)} / ${fmtTokens(ctx.max)}` : `${fmtTokens(ctx.used)}（窗口大小未知）`],
                ...(s.context && s.context !== ctx && s.context.used !== ctx.used
                  ? ([['最近一次', s.context.max ? `${fmtTokens(s.context.used)} / ${fmtTokens(s.context.max)}` : fmtTokens(s.context.used)]] as [string, ReactNode][])
                  : []),
                ['模型', ctx.model],
              ]}
            />
          }
        >
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
            {ctxPct != null && (
              <Box component="span" sx={{ position: 'relative', display: 'inline-flex', width: 14, height: 14 }}>
                <CircularProgress variant="determinate" value={100} size={14} thickness={6} sx={{ color: 'rgba(0,0,0,0.08)', position: 'absolute' }} />
                <CircularProgress variant="determinate" value={Math.max(ctxPct, 2)} size={14} thickness={6} sx={{ color }} />
              </Box>
            )}
            <Box component="span" sx={{ color: ctxPct != null && ctxPct >= 60 ? color : undefined }}>
              {ctxPct != null ? `${ctxPct < 1 ? '<1' : Math.round(ctxPct)}%` : `上下文 ${fmtTokens(ctx.used)}`}
            </Box>
          </Box>
        </Item>
      )}
    </Box>
    {details && threadId && <RunTraceDialog key={threadId} threadId={threadId} onClose={() => setDetails(false)} />}
    </>
  )
}
