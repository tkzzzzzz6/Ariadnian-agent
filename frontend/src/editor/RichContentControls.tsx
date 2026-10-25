import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import type { Editor, JSONContent, Range } from '@tiptap/core'
import { useEditorState } from '@tiptap/react'
import { TextSelection, type Transaction } from '@tiptap/pm/state'
import { closeHistory } from '@tiptap/pm/history'
import { Alert, Box, Button, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle,
  FormControlLabel, TextField, Typography } from '@mui/material'
import katex from 'katex'
import { api } from '../lib/api'
import { tableJSON, unsafeTex, type InsertKind } from '../lib/richContent'
import { useApp } from '../lib/appContext'
import { BLOCK_ITEMS, insertBlock, type BlockKind } from '../lib/formatting'
import { brand } from '../theme'

export interface RichContentHandle {
  insert: (kind: InsertKind, range?: Range) => void
  editImage: (pos: number) => void
  editMath: (pos: number) => void
  upload: (file: File, pos?: number) => void
}
interface Target { from: number; to: number; kind: InsertKind; edit: boolean; original?: string; stale?: boolean }
const templates = [['分数', '\\frac{a}{b}'], ['求和', '\\sum_{i=1}^{n} x_i'], ['积分', '\\int_a^b f(x)\\,dx'], ['矩阵', '\\begin{bmatrix}a&b\\\\c&d\\end{bmatrix}']]

export const RichContentControls = forwardRef<RichContentHandle, { editor: Editor; docId: string }>(function RichContentControls({ editor, docId }, ref) {
  const { notify } = useApp()
  const [mode, setMode] = useState<InsertKind | null>(null)
  const target = useRef<Target | null>(null)
  const mounted = useRef(true)
  const pending = useRef(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [caption, setCaption] = useState('')
  const [alt, setAlt] = useState('')
  const [width, setWidth] = useState('100')
  const [source, setSource] = useState('')
  const [rows, setRows] = useState('3')
  const [cols, setCols] = useState('3')
  const [header, setHeader] = useState(true)
  const selected = useEditorState({ editor, selector: ({ editor: e }) => ({
    image: e.isActive('image'), table: e.isActive('table'), inlineMath: e.isActive('inlineMath'), blockMath: e.isActive('blockMath'),
    cell: e.isActive('tableCell') || e.isActive('tableHeader'),
  }) })

  useEffect(() => {
    mounted.current = true
    const map = ({ transaction: tr }: { transaction: Transaction }) => {
      const t = target.current
      if (!t || !tr.docChanged) return
      const a = tr.mapping.mapResult(t.from, 1)
      const b = tr.mapping.mapResult(t.to, -1)
      t.from = a.pos; t.to = Math.max(a.pos, b.pos)
      if (a.deletedAcross || b.deletedAcross) t.stale = true
    }
    editor.on('transaction', map)
    return () => { mounted.current = false; editor.off('transaction', map) }
  }, [editor])

  const allowed = (kind: InsertKind) => {
    const p = editor.state.selection.$from
    for (let i = p.depth; i > 0; i--) {
      const n = p.node(i)
      if (n.attrs.live || n.attrs.draft) { notify('请先保留当前 Agent 草稿，再插入图表或公式', 'warning'); return false }
      if (['tableCell', 'tableHeader'].includes(n.type.name) && kind !== 'inlineMath') { notify('单元格中仅支持文字、引用和行内公式', 'warning'); return false }
    }
    return true
  }
  const resetFields = () => { setError(''); setFile(null); setCaption(''); setAlt(''); setWidth('100'); setSource(''); setRows('3'); setCols('3'); setHeader(true) }
  const insert = (kind: InsertKind, range?: Range) => {
    if (pending.current) return
    if (BLOCK_ITEMS.some(i=>i.kind===kind)) {
      if (!insertBlock(editor,kind as BlockKind,range)) notify('当前位置无法转换此内容，请选择普通正文或先保留草稿','info')
      return
    }
    if (!allowed(kind)) return
    resetFields()
    target.current = { kind, from: range?.from ?? editor.state.selection.from, to: range?.to ?? editor.state.selection.to, edit: false }
    setMode(kind)
  }
  const edit = (pos: number) => {
    if (pending.current) return
    const n = editor.state.doc.nodeAt(pos)
    if (!n || !['image', 'inlineMath', 'blockMath'].includes(n.type.name)) return
    resetFields()
    target.current = { kind: n.type.name as InsertKind, from: pos, to: pos + n.nodeSize, edit: true, original: JSON.stringify(n.toJSON()) }
    setCaption(n.attrs.caption ?? ''); setAlt(n.attrs.alt ?? ''); setWidth(String(n.attrs.widthPercent ?? 100)); setSource(n.attrs.latex ?? '')
    setMode(n.type.name as InsertKind)
  }
  const commit = (node: JSONContent) => {
    const t = target.current
    if (!mounted.current || editor.isDestroyed || !t || t.stale || t.from > editor.state.doc.content.size
      || (t.edit && JSON.stringify(editor.state.doc.nodeAt(t.from)?.toJSON()) !== t.original)) throw new Error('插入位置或内容已变化，请重新选择位置')
    const ok = editor.chain().focus().command(({ tr }) => { closeHistory(tr); return true })
      .insertContentAt({ from: t.from, to: t.to }, node).run()
    if (!ok) throw new Error('当前位置无法插入此内容，请选择普通正文段落')
    editor.view.dispatch(closeHistory(editor.state.tr).setMeta('addToHistory', false))
    target.current = null; setMode(null); setFile(null)
  }
  const checkedFile = (value: File) => {
    if (value.size > 10 * 1024 * 1024) throw new Error('单张图片不能超过 10 MB')
    if (!['image/png', 'image/jpeg'].includes(value.type)) throw new Error('目前支持 PNG、JPEG 图片')
  }
  const upload = async (value: File, pos?: number) => {
    if (pending.current) return
    try {
      checkedFile(value)
      if (pos != null) editor.commands.setTextSelection(TextSelection.near(editor.state.doc.resolve(pos)).from)
      if (!allowed('image')) return
      target.current = { kind: 'image', from: editor.state.selection.from, to: editor.state.selection.to, edit: false }
      pending.current = true; setBusy(true)
      const a = await api.uploadImage(docId, value)
      commit({ type: 'image', attrs: { assetId: a.id, extension: a.extension, src: a.src, alt: '', caption: '', widthPercent: 100 } })
      notify('图片已插入', 'success')
    } catch (e: any) { if (mounted.current) notify(e.message, 'error') }
    finally { pending.current = false; if (mounted.current) setBusy(false) }
  }
  useImperativeHandle(ref, () => ({ insert, upload, editImage: edit, editMath: edit }))

  const preview = useMemo(() => {
    if (!source.trim()) return { html: '', error: '' }
    if (source.length > 8000 || unsafeTex.test(source)) return { html: '', error: '公式过长或包含不支持的命令' }
    try { return { html: katex.renderToString(source, { displayMode: mode === 'blockMath', throwOnError: true, trust: false, maxExpand: 1000 }), error: '' } }
    catch (e: any) { return { html: '', error: e.message } }
  }, [source, mode])
  const previewUrl = useMemo(() => file ? URL.createObjectURL(file) : '', [file])
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  const apply = async () => {
    if (pending.current || !mode) return
    setError('')
    try {
      if (mode === 'table') commit(tableJSON(Number(rows), Number(cols), header))
      else if (mode === 'inlineMath' || mode === 'blockMath') {
        if (!source.trim() || preview.error) throw new Error(preview.error || '请输入公式')
        commit({ type: mode, attrs: { latex: source.trim() } })
      } else {
        const percent = Number(width)
        if (!Number.isFinite(percent) || percent < 10 || percent > 100) throw new Error('图片宽度需要在 10% 到 100% 之间')
        const old = target.current?.edit ? editor.state.doc.nodeAt(target.current.from)?.attrs : null
        if (!file && !old) throw new Error('请选择一张图片')
        pending.current = true; setBusy(true)
        const a = file ? await api.uploadImage(docId, file) : null
        commit({ type: 'image', attrs: { ...(old ?? {}), ...(a ? { assetId: a.id, src: a.src, extension: a.extension } : {}), caption, alt, widthPercent: percent } })
      }
    } catch (e: any) { if (mounted.current) setError(e.message) }
    finally { pending.current = false; if (mounted.current) setBusy(false) }
  }
  const nodePos = () => editor.state.selection.from
  const tableCommand = (command: 'addRowBefore' | 'addRowAfter' | 'deleteRow' | 'addColumnBefore' | 'addColumnAfter' | 'deleteColumn' | 'toggleHeaderRow' | 'deleteTable') => {
    editor.chain().focus().command(({ tr }) => { closeHistory(tr); return true })[command]().run()
    editor.view.dispatch(closeHistory(editor.state.tr).setMeta('addToHistory', false))
  }
  return <>
    {busy && !mode && <Typography sx={{ px: 2, fontSize: 12 }} role="status">正在上传图片…</Typography>}
    {(selected.image || selected.table || selected.inlineMath || selected.blockMath) && <Box aria-label="内容编辑操作" sx={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 0.5, px: 1, pb: 0.5, flexShrink: 0 }}>
      {selected.image && <><Button onClick={() => edit(nodePos())}>图片：替换、宽度与图题</Button><Button color="error" onClick={() => editor.chain().focus().deleteSelection().run()}>删除图片</Button></>}
      {(selected.inlineMath || selected.blockMath) && <><Button onClick={() => edit(nodePos())}>编辑公式</Button><Button color="error" onClick={() => editor.chain().focus().deleteSelection().run()}>删除公式</Button></>}
      {selected.table && <>{([['addRowBefore', '上方加行'], ['addRowAfter', '下方加行'], ['deleteRow', '删除行'], ['addColumnBefore', '左侧加列'], ['addColumnAfter', '右侧加列'], ['deleteColumn', '删除列'], ['toggleHeaderRow', '切换表头'], ['deleteTable', '删除表格']] as const)
        .map(([cmd, label]) => <Button key={cmd} color={cmd === 'deleteTable' ? 'error' : 'primary'} onMouseDown={e => e.preventDefault()} onClick={() => tableCommand(cmd)}>{label}</Button>)}</>}
    </Box>}
    <Dialog open={!!mode} onClose={() => { if (!busy) { setMode(null); setFile(null); target.current = null } }} fullWidth maxWidth="sm" aria-labelledby="rich-insert-title">
      <DialogTitle id="rich-insert-title">{mode === 'image' ? '图片' : mode === 'table' ? '插入表格' : mode === 'blockMath' ? '独立公式' : '行内公式'}</DialogTitle>
      <DialogContent dividers onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); void apply() } }}>
        {error && <Alert severity="error" sx={{ mb: 1.5 }}>{error}</Alert>}
        {mode === 'image' && <>
          <Button component="label" disabled={busy} variant="outlined">{target.current?.edit ? '选择替换图片' : '选择图片'}<input type="file" accept="image/png,image/jpeg" hidden onChange={e => {
            const f = e.target.files?.[0]; if (!f) return
            try { checkedFile(f); setFile(f); setError('') } catch (err: any) { setError(err.message) }
            e.target.value = ''
          }} /></Button>
          <Typography sx={{ fontSize: 12, my: 1, color: brand.muted }}>{file?.name ?? 'PNG、JPEG；单张最多 10 MB、2500 万像素'}</Typography>
          {previewUrl && <Box component="img" src={previewUrl} alt="待插入图片预览" sx={{ maxWidth: '100%', maxHeight: 220, objectFit: 'contain', display: 'block', mb: 1 }} />}
          <TextField fullWidth label="图题（可选）" value={caption} disabled={busy} onChange={e => setCaption(e.target.value)} sx={{ my: 1 }} slotProps={{ htmlInput: { maxLength: 500 } }} />
          <TextField fullWidth label="替代说明（可选）" value={alt} disabled={busy} onChange={e => setAlt(e.target.value)} sx={{ my: 1 }} slotProps={{ htmlInput: { maxLength: 500 } }} />
          <TextField label="宽度，占正文 %" type="number" value={width} disabled={busy} onChange={e => setWidth(e.target.value)} sx={{ my: 1, width: 180 }} />
          <Box sx={{ display: 'flex', gap: 1 }}>{[25, 50, 75, 100].map(n => <Button key={n} disabled={busy} onClick={() => setWidth(String(n))}>{n}%</Button>)}</Box>
        </>}
        {mode === 'table' && <>
          <Box sx={{ display: 'flex', gap: 2, mt: 1 }}><TextField autoFocus label="行数（1—50）" type="number" value={rows} onChange={e => setRows(e.target.value)} /><TextField label="列数（1—10）" type="number" value={cols} onChange={e => setCols(e.target.value)} /></Box>
          <FormControlLabel control={<Checkbox checked={header} onChange={e => setHeader(e.target.checked)} />} label="第一行为表头" />
          <Typography sx={{ fontSize: 12, color: brand.muted }}>也可以直接粘贴 Excel 中的规则行列数据。Tab 切换单元格；末格按方向键离开表格。</Typography>
        </>}
        {(mode === 'inlineMath' || mode === 'blockMath') && <>
          <TextField autoFocus fullWidth multiline minRows={3} label="LaTeX 公式源码" value={source} onChange={e => setSource(e.target.value)} placeholder="例如：\frac{a}{b}" sx={{ mt: 1 }} slotProps={{ htmlInput: { maxLength: 8000 } }} />
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 1 }}>{templates.map(([label, code]) => <Button key={label} onClick={() => setSource(code)}>{label}</Button>)}</Box>
          <Typography sx={{ mt: 1, fontSize: 12, color: brand.muted }}>正文快捷输入：$公式$ 为行内公式；独立段落中的 $$公式$$ 为独立公式。</Typography>
          <Typography sx={{ mt: 1, fontSize: 12, color: brand.muted }}>实时预览</Typography>
          {preview.error ? <Alert severity="warning" sx={{ overflowWrap: 'anywhere' }}>{preview.error}</Alert> : <Box sx={{ p: 2, bgcolor: brand.subtle, overflowX: 'auto', minHeight: 70 }} dangerouslySetInnerHTML={{ __html: preview.html }} />}
        </>}
      </DialogContent>
      <DialogActions><Button disabled={busy} onClick={() => { setMode(null); target.current = null; setFile(null) }}>取消</Button><Button variant="contained" disabled={busy} onClick={() => void apply()}>{busy ? '上传中…' : target.current?.edit ? '保存修改' : '插入'}</Button></DialogActions>
    </Dialog>
  </>
})
