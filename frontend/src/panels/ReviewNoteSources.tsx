import { useState } from 'react'
import { Alert, Box, Button, Collapse, Typography } from '@mui/material'
import { useApp } from '../lib/appContext'
import type { ReviewNoteContext } from '../lib/reviewNotes'

export function ReviewNoteSources({ context }: { context?: ReviewNoteContext | null }) {
  const { papers, openReader } = useApp()
  const [open, setOpen] = useState(false)
  if (!context) return <Typography sx={{ fontSize: 11.5, color: 'text.secondary', px: 1.25, py: 1 }}>旧任务未记录笔记来源。可刷新笔记并重新生成。</Typography>
  if (!context.enabled) return <Typography sx={{ fontSize: 11.5, color: 'text.secondary', px: 1.25, py: 1 }}>本轮已关闭用户笔记输入。</Typography>
  return <Box sx={{ px: 1.25, py: 1 }}>
    <Button size="small" onClick={() => setOpen(!open)} aria-expanded={open}>本轮提供的笔记 · {context.items.length} 条{open ? '（收起）' : '（查看来源）'}</Button>
    <Typography sx={{ fontSize: 11.5, color: 'text.secondary' }}>摘录与评论分开提供；评论属于用户观点，不能替代论文证据。新增笔记需刷新后纳入；删除原笔记不会清除任务中已保存的历史副本。</Typography>
    {context.changed && <Alert severity="warning" sx={{ mt: 0.75 }}>来源已修改或删除。当前结果仍基于历史快照，请刷新后重新生成，或核对后明确沿用旧版本。</Alert>}
    <Collapse in={open}>
      <Typography sx={{ fontSize: 11.5, color: 'text.secondary', mt: 0.75 }}>快照：{new Date(context.captured_at).toLocaleString()} · 词项重合筛选 · 扫描 {context.scanned} 条 · 匹配 {context.relevant} 条 · 因限额省略 {context.omitted} 条{context.scan_limited ? '；仅扫描最新 500 条' : ''}</Typography>
      {!context.items.length && <Typography sx={{ fontSize: 12, mt: 0.75 }}>本轮未筛选到相关笔记，综述仍依据文献内容生成。</Typography>}
      {context.items.map(n => <Box key={n.id} sx={{ mt: 1, pl: 1, borderLeft: '2px solid', borderColor: 'divider' }}>
        <Typography sx={{ fontSize: 12, fontWeight: 600 }}>{n.title} · p.{n.page}</Typography>
        <Typography sx={{ fontSize: 11.5, color: n.source_status === 'unchanged' ? 'text.secondary' : 'warning.main' }}>
          {n.bibkey} · 笔记 {n.id} · {{ unchanged: '来源未变化', changed: '原笔记已修改', deleted: '原笔记已删除', paper_deleted: '所属文献已删除' }[n.source_status]}
        </Typography>
        <Typography sx={{ fontSize: 12, whiteSpace: 'pre-wrap', mt: 0.5 }}>用户保存的摘录（未重新核验）：{n.quote}</Typography>
        {n.comment && <Typography sx={{ fontSize: 12, whiteSpace: 'pre-wrap', mt: 0.5 }}>用户观点（非论文结论）：{n.comment}</Typography>}
        {n.truncated && <Typography sx={{ fontSize: 11.5, color: 'text.secondary' }}>本轮输入已截断；请回原文/笔记核对完整内容。</Typography>}
        <Button size="small" disabled={!papers.some(p => p.id === n.paper_id)} onClick={() => openReader({ paperId: n.paper_id, evidences: [{ page: n.page, snippet: n.quote }] })}>核对原文 p.{n.page}</Button>
      </Box>)}
    </Collapse>
  </Box>
}
