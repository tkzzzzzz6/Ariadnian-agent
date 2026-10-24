/** Agent 历史任务：按文档列出综述 / 分节写作 / 审稿，点一下回到那次任务 */
import { useState } from 'react'
import { Box, Button, ListItemIcon, ListItemText, Menu, MenuItem, Typography } from '@mui/material'
import HistoryIcon from '@mui/icons-material/History'
import AccountTreeOutlinedIcon from '@mui/icons-material/AccountTreeOutlined'
import EditNoteIcon from '@mui/icons-material/EditNote'
import RateReviewOutlinedIcon from '@mui/icons-material/RateReviewOutlined'
import { api } from '../lib/api'
import { fmtTokens } from '../lib/runStats'
import { brand } from '../theme'
import { RunTraceDialog } from './RunTraceDialog'

export interface ThreadInfo {
  thread_id: string
  topic: string
  status: string
  kind?: 'review' | 'write' | 'critique'
  created_at: string
  updated_at?: string
  tokens?: number | null
}

const STATUS: Record<string, [string, string]> = {
  waiting: ['等你确认', brand.orange],
  running: ['运行中', brand.purple],
  stopping: ['正在停止', brand.orange],
  interrupted: ['意外中断', brand.orange],
  stopped: ['已停止', brand.faint],
  done: ['已完成', brand.green],
  error: ['出错', brand.red],
}

/** 后端存的是 UTC 时间（SQLite datetime('now')） */
function ago(ts?: string): string {
  if (!ts) return ''
  const t = new Date(ts.replace(' ', 'T') + (ts.endsWith('Z') ? '' : 'Z')).getTime()
  const m = Math.round((Date.now() - t) / 60000)
  if (Number.isNaN(m)) return ''
  if (m < 1) return '刚刚'
  if (m < 60) return `${m} 分钟前`
  if (m < 60 * 24) return `${Math.round(m / 60)} 小时前`
  return `${Math.round(m / 1440)} 天前`
}

const ICON = { review: AccountTreeOutlinedIcon, write: EditNoteIcon, critique: RateReviewOutlinedIcon }

export function AgentHistory({ docId, onOpen, disabled }: { docId: string | null; onOpen: (t: ThreadInfo) => void; disabled?: boolean }) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [items, setItems] = useState<ThreadInfo[] | null>(null)
  const [traceId, setTraceId] = useState<string | null>(null)
  const [error, setError] = useState('')
  return (
    <>
      <Button
        size="small"
        startIcon={<HistoryIcon />}
        disabled={!docId || disabled}
        onClick={(e) => {
          setAnchor(e.currentTarget)
          setItems(null)
          setError('')
          if (docId) api.agentThreads(docId).then(setItems).catch(e => { setItems([]); setError(e.message) })
        }}
        sx={{ color: brand.muted, fontSize: 12, py: 0, minHeight: 24, '& svg': { fontSize: '16px !important' } }}
      >
        历史
      </Button>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)} slotProps={{ paper: { sx: { width: 340, maxHeight: 420 } } }}>
        {items === null && (
          <MenuItem disabled dense>
            加载中…
          </MenuItem>
        )}
        {items?.length === 0 && (
          <MenuItem disabled dense>
            {error || '这篇文档还没有 Agent 任务'}
          </MenuItem>
        )}
        {items?.map((t) => {
          const Icon = ICON[t.kind ?? 'review']
          const [label, color] = STATUS[t.status] ?? [t.status, brand.faint]
          return (
            <MenuItem
              key={t.thread_id}
              dense
              onClick={() => {
                setAnchor(null)
                onOpen(t)
              }}
            >
              <ListItemIcon sx={{ minWidth: 30, color: brand.purple }}>
                <Icon sx={{ fontSize: 18 }} />
              </ListItemIcon>
              <ListItemText
                primary={t.topic}
                secondary={
                  <Box component="span" sx={{ display: 'flex', gap: 1 }}>
                    <Typography component="span" sx={{ fontSize: 11.5, color }}>
                      {label}
                    </Typography>
                    <Typography component="span" sx={{ fontSize: 11.5, color: brand.faint }}>
                      {ago(t.updated_at || t.created_at)}
                    </Typography>
                    {!!t.tokens && (
                      <Typography component="span" sx={{ fontSize: 11.5, color: brand.faint }}>
                        {fmtTokens(t.tokens)} tok
                      </Typography>
                    )}
                  </Box>
                }
                slotProps={{ primary: { noWrap: true, sx: { fontSize: 13 } } }}
              />
              <Button size="small" onClick={e => { e.stopPropagation(); setAnchor(null); setTraceId(t.thread_id) }} sx={{ ml: 1, flexShrink: 0, fontSize: 11.5 }}>运行详情</Button>
            </MenuItem>
          )
        })}
      </Menu>
      {traceId && <RunTraceDialog threadId={traceId} onClose={() => setTraceId(null)} />}
    </>
  )
}
