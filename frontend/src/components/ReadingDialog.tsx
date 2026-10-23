import { useState } from 'react'
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Stack, TextField, Typography } from '@mui/material'
import { useApp } from '../lib/appContext'
import { READING_LABELS, type ReadingRecord, type ReadingStatus } from '../lib/reading'
import { readingStore } from '../lib/readingStore'

/** Mounted afresh when opened; unsaved form edits never change the shared record. */
export function ReadingDialog({ record, title, onClose }: { record: ReadingRecord; title: string; onClose: () => void }) {
  const { notify } = useApp()
  const [status, setStatus] = useState(record.status)
  const [page, setPage] = useState(record.read_through_page?.toString() ?? '')
  const [total, setTotal] = useState(record.total_pages?.toString() ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const canHavePage = status === 'reading' || status === 'read'
  const save = async () => {
    const valid = (s: string) => !s || (/^[1-9]\d*$/.test(s) && Number(s) <= 100000)
    if (!valid(total) || (canHavePage && !valid(page))) { setError('页码需为 1–100000 的整数，或留空表示未知'); return }
    if (canHavePage && page && total && Number(page) > Number(total)) { setError('读至页码不能超过总页数'); return }
    setBusy(true)
    setError('')
    try {
      await readingStore.save(record.paper_id, {
        status, read_through_page: canHavePage && page ? Number(page) : null, total_pages: total ? Number(total) : null,
      })
      notify('阅读记录已保存', 'success')
      onClose()
    } catch (e: any) { setError(e.message) }
    finally { setBusy(false) }
  }
  return (
    <Dialog open onClose={busy ? undefined : onClose} maxWidth="xs" fullWidth>
      <DialogTitle>阅读记录</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ pt: 1 }}>
          <Typography variant="body2">{title}</Typography>
          <Typography variant="caption" color="text.secondary">记录用于整个文献库。页码为 PDF 页序号；打开或滚动文件不会自动标为已读。没有本地 PDF 也可以手动确认阅读状态。</Typography>
          <TextField select label="阅读状态" value={status} onChange={(e) => setStatus(e.target.value as ReadingStatus)} disabled={busy}>
            {Object.entries(READING_LABELS).map(([value, label]) => <MenuItem key={value} value={value}>{label}</MenuItem>)}
          </TextField>
          <TextField label="读至第几页" value={canHavePage ? page : ''} disabled={busy || !canHavePage} onChange={(e) => setPage(e.target.value)} helperText="可回退或清空；未知时留空。标为未读或未记录会清空此项。" />
          <TextField label="总页数（可选）" value={total} disabled={busy} onChange={(e) => setTotal(e.target.value)} />
          {error && <Alert severity="error">{error}</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions><Button disabled={busy} onClick={onClose}>取消</Button><Button variant="contained" disabled={busy} onClick={save}>{busy ? '保存中…' : '保存'}</Button></DialogActions>
    </Dialog>
  )
}
