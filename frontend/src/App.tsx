import { useEffect, useRef, useState } from 'react'
import { Allotment, type AllotmentHandle } from 'allotment'
import 'allotment/dist/style.css'
import {
  Badge, Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, Divider, IconButton, ListItemIcon, ListItemText, Menu, MenuItem,
  Tooltip, Typography,
} from '@mui/material'
import RateReviewOutlinedIcon from '@mui/icons-material/RateReviewOutlined'
import RateReviewIcon from '@mui/icons-material/RateReview'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddCircleIcon from '@mui/icons-material/AddCircle'
import AutoAwesomeOutlinedIcon from '@mui/icons-material/AutoAwesomeOutlined'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import StorageOutlinedIcon from '@mui/icons-material/StorageOutlined'
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined'
import SettingsOutlinedIcon from '@mui/icons-material/SettingsOutlined'
import FileDownloadOutlinedIcon from '@mui/icons-material/FileDownloadOutlined'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined'
import FunctionsIcon from '@mui/icons-material/Functions'
import MenuBookOutlinedIcon from '@mui/icons-material/MenuBookOutlined'
import CheckIcon from '@mui/icons-material/Check'
import AddIcon from '@mui/icons-material/Add'
import avatar from './assets/ariadne-avatar.png'
import { PillToggle } from './components/Thread'
import { SettingsDialog } from './components/SettingsDialog'
import { PaperEditor } from './editor/PaperEditor'
import { SplitReader } from './panels/SplitReader'
import { api } from './lib/api'
import { useApp, type RightTab } from './lib/appContext'
import { citeStore, useCiteState } from './lib/citations'
import { createDoc, relTime, useBootstrapDoc, useDocList } from './lib/docs'
import { citedIds, docxStyleFor, download, exportDocx, exportBundle, type DocxStyle } from './lib/export'
import ArticleOutlinedIcon from '@mui/icons-material/ArticleOutlined'
import type { CiteStyle } from './lib/types'
import { DrawerPanel } from './panels/DrawerPanel'
import { PdfPreviewDialog } from './pdf/PdfPreviewDialog'
import { brand } from './theme'

// ------------------------------------------------------------------ 顶栏
function DocSwitcher() {
  const { docId, setDocId, docTitle, bumpDocs } = useApp()
  const docs = useDocList()
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [confirm, setConfirm] = useState(false)
  const remove = async () => {
    setConfirm(false)
    if (!docId) return
    await api.deleteDoc(docId)
    const list = await api.docs()
    if (list.length) setDocId(list[0].id)
    else setDocId((await createDoc()).id)
    bumpDocs()
  }
  return (
    <>
      <Box
        role="button"
        onClick={(e) => setAnchor(e.currentTarget)}
        sx={{
          display: 'flex', alignItems: 'center', gap: 0.25, px: 1.25, py: 0.5, borderRadius: '6px', cursor: 'pointer',
          maxWidth: 420, '&:hover': { bgcolor: 'rgba(0,0,0,0.04)' },
        }}
      >
        <Typography noWrap sx={{ fontSize: 15, color: brand.ink }}>
          {docTitle || '未命名论文'}
        </Typography>
        <ExpandMoreIcon sx={{ fontSize: 18, color: brand.muted }} />
      </Box>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)} slotProps={{ paper: { sx: { minWidth: 300, maxHeight: 420 } } }}>
        {docs.map((d) => (
          <MenuItem
            key={d.id}
            selected={d.id === docId}
            onClick={() => {
              setDocId(d.id)
              setAnchor(null)
            }}
          >
            <ListItemIcon>{d.id === docId ? <CheckIcon fontSize="small" color="primary" /> : <DescriptionOutlinedIcon fontSize="small" />}</ListItemIcon>
            <ListItemText primary={d.title} secondary={relTime(d.updated_at)} slotProps={{ primary: { noWrap: true, fontSize: 13 }, secondary: { fontSize: 11 } }} />
          </MenuItem>
        ))}
        <Divider />
        <MenuItem
          onClick={async () => {
            const d = await createDoc()
            bumpDocs()
            setDocId(d.id)
            setAnchor(null)
          }}
        >
          <ListItemIcon>
            <AddIcon fontSize="small" />
          </ListItemIcon>
          新建论文
        </MenuItem>
        <MenuItem
          onClick={() => {
            setAnchor(null)
            setConfirm(true)
          }}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon>
            <DeleteOutlineIcon fontSize="small" color="error" />
          </ListItemIcon>
          删除当前论文
        </MenuItem>
      </Menu>
      <Dialog open={confirm} onClose={() => setConfirm(false)}>
        <DialogTitle>删除这篇论文？</DialogTitle>
        <DialogContent>「{docTitle || '未命名论文'}」将被永久删除，文献库不受影响。</DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirm(false)}>取消</Button>
          <Button color="error" variant="contained" onClick={remove}>
            删除
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )
}

function ExportButton() {
  const { editor, docId, docTitle, notify } = useApp()
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const st = useCiteState()
  const [busy, setBusy] = useState(false)
  const exportAs = async (kind: 'md' | 'tex' | 'bib' | DocxStyle) => {
    setAnchor(null)
    if (!editor) return
    const json = editor.getJSON()
    const name = (docTitle || 'paper').replace(/[\\/:*?"<>|]/g, '_')
    const ids = citedIds(json)
    if (kind !== 'md' && kind !== 'tex' && kind !== 'bib') {
      setBusy(true)
      try {
        download(`${name}.docx`, await exportDocx(json, docTitle || '论文', kind, docId ?? undefined))
        notify('已导出 Word 文件', 'success')
      } catch (e: any) {
        notify(e.message, 'error')
      } finally {
        setBusy(false)
      }
      return
    }
    try {
      setBusy(true)
      if (kind === 'md' || kind === 'tex') {
        if (!docId) throw new Error('请先打开文档')
        download(`${name}-${kind}.zip`, await exportBundle(json, docTitle, kind, docId))
        notify('已导出包含正文、图片和参考文献的资源包', 'success')
      } else download('refs.bib', ids.length ? await api.bibtex(ids) : '', 'application/x-bibtex')
    } catch (e: any) {
      notify(e.message, 'error')
    } finally { setBusy(false) }
  }
  return (
    <>
      <Tooltip title="导出">
        <IconButton aria-label="导出" onClick={(e) => setAnchor(e.currentTarget)} disabled={busy} sx={{ color: brand.muted }}>
          {busy ? <CircularProgress size={16} /> : <FileDownloadOutlinedIcon fontSize="small" />}
        </IconButton>
      </Tooltip>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)} anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }} transformOrigin={{ vertical: 'top', horizontal: 'right' }}>
        <MenuItem onClick={() => editor && exportAs(docxStyleFor(st.style, editor.getJSON()))}>
          <ListItemIcon><ArticleOutlinedIcon fontSize="small" sx={{ color: '#2b579a' }} /></ListItemIcon>
          <ListItemText
            primary="Word（.docx）"
            secondary={st.style === 'numeric' ? '参考文献按 GB/T 7714—2015 顺序编码' : '参考文献按著者-出版年（中文 GB/T 7714，英文 APA）'}
          />
        </MenuItem>
        <MenuItem onClick={() => exportAs('tex')}>
          <ListItemIcon><FunctionsIcon fontSize="small" /></ListItemIcon>
          <ListItemText primary="LaTeX 资源包（ZIP）" secondary="含图片与 refs.bib，可上传 Overleaf" />
        </MenuItem>
        <MenuItem onClick={() => exportAs('md')}>
          <ListItemIcon><DescriptionOutlinedIcon fontSize="small" /></ListItemIcon>
          <ListItemText primary="Markdown 资源包（ZIP）" secondary="含图片与 refs.bib，Pandoc 引用语法" />
        </MenuItem>
        <MenuItem onClick={() => exportAs('bib')}>
          <ListItemIcon><MenuBookOutlinedIcon fontSize="small" /></ListItemIcon>
          <ListItemText primary="仅 BibTeX" secondary="正文引用到的文献" />
        </MenuItem>
      </Menu>
    </>
  )
}

function TopBar() {
  const { docId, health, copilot, setCopilot, review, setReview, openSettings, split, setSplit } = useApp()
  const st = useCiteState()
  const changeStyle = (s: CiteStyle) => {
    citeStore.setStyle(s)
    if (docId) api.saveDoc(docId, { cite_style: s })
  }
  return (
    <Box sx={{ height: 52, display: 'flex', alignItems: 'center', px: 2, gap: 2, flexShrink: 0, bgcolor: '#fff' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: 300 }}>
        <Box component="img" src={avatar} alt="Ariadne" sx={{ width: 30, height: 30, borderRadius: '7px', display: 'block' }} />
        <Typography sx={{ fontSize: 18, fontWeight: 500, color: brand.ink, letterSpacing: 0.2 }}>Ariadne</Typography>
        <Typography sx={{ fontSize: 12, color: brand.faint, ml: 0.5 }}>研究报告助手</Typography>
      </Box>
      <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', minWidth: 0 }}>
        <DocSwitcher />
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
        {health && (
          <Tooltip title={health.fake ? '离线演示模式：不调用模型。点击打开设置' : health.llm_ready ? '点击修改模型设置' : '还没有配置模型，点击设置'}>
            <Typography
              onClick={() => openSettings('model')}
              sx={{ fontSize: 13, color: health.llm_ready ? brand.muted : brand.red, cursor: 'pointer', maxWidth: 200 }}
              noWrap
            >
              {health.fake ? '离线演示' : health.llm_ready ? health.model.split('/').slice(1).join('/') || health.model : '未配置模型'}
            </Typography>
          </Tooltip>
        )}
        <Box sx={{ width: '1px', height: 22, bgcolor: brand.line }} />
        <Tooltip title="分屏对照：左边写，右边读原文（可同时开多篇）">
          <span>
            <PillToggle<'write' | 'split'>
              value={split ? 'split' : 'write'}
              onChange={(v) => setSplit(v === 'split')}
              options={[
                ['write', '写作'],
                ['split', '分屏对照'],
              ]}
            />
          </span>
        </Tooltip>
        <PillToggle<CiteStyle>
          value={st.style}
          onChange={changeStyle}
          options={[
            ['numeric', '[1]'],
            ['author-year', '作者-年'],
          ]}
        />
        <Tooltip title={copilot ? '续写建议：开（停顿时出现灰色建议，Tab 接受）' : '续写建议：关'}>
          <IconButton aria-label="续写建议" onClick={() => setCopilot(!copilot)} sx={{ color: copilot ? brand.blue : brand.muted }}>
            {copilot ? <AutoAwesomeIcon fontSize="small" /> : <AutoAwesomeOutlinedIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
        <Tooltip title={review ? '页边建议：开（空闲时检查引用和表达，结果显示在段落旁）' : '页边建议：关'}>
          <IconButton aria-label="页边建议" onClick={() => setReview(!review)} sx={{ color: review ? brand.blue : brand.muted }}>
            {review ? <RateReviewIcon fontSize="small" /> : <RateReviewOutlinedIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
        <ExportButton />
        <Tooltip title="设置：模型、文献检索、向量检索">
          <IconButton aria-label="设置" onClick={() => openSettings('model')} sx={{ color: brand.muted }}>
            <SettingsOutlinedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  )
}

// ------------------------------------------------------------------ 左侧图标栏（控制右侧抽屉）
const RAIL: [RightTab, string, typeof StorageOutlinedIcon][] = [
  ['agent', 'AI 助手（Ctrl+J）', AutoAwesomeOutlinedIcon],
  ['library', '文献库', StorageOutlinedIcon],
  ['docs', '我的论文', FolderOutlinedIcon],
]

function Rail() {
  const { drawerOpen, rightTab, toggleDrawer, setDocId, bumpDocs, agentAlert } = useApp()
  const group = (t: RightTab) => (t === 'agent' || t === 'cite' || t === 'ask' ? 'agent' : t === 'reader' ? 'library' : t)
  return (
    <Box sx={{ width: 52, flexShrink: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.75, pt: 1.5, borderRight: `1px solid ${brand.lineSoft}` }}>
      <Tooltip title="新建论文" placement="right">
        <IconButton
          aria-label="新建论文"
          onClick={async () => {
            const d = await createDoc()
            bumpDocs()
            setDocId(d.id)
          }}
          sx={{ color: brand.blue, mb: 0.5 }}
        >
          <AddCircleIcon />
        </IconButton>
      </Tooltip>
      {RAIL.map(([key, label, Icon]) => {
        const active = drawerOpen && group(rightTab) === key
        return (
          <Tooltip key={key} title={key === 'agent' && agentAlert ? agentAlert : label} placement="right">
            <IconButton
              aria-label={label.replace(/（.*）/, '')}
              onClick={() => toggleDrawer(key)}
              sx={{
                borderRadius: '8px', color: active ? brand.blue : '#5f6368', bgcolor: active ? 'rgba(0,120,212,0.09)' : 'transparent',
                '&:hover': { bgcolor: active ? 'rgba(0,120,212,0.13)' : 'rgba(0,0,0,0.05)' },
              }}
            >
              <Badge variant="dot" color="error" invisible={key !== 'agent' || !agentAlert} overlap="circular">
                <Icon sx={{ fontSize: 21 }} />
              </Badge>
            </IconButton>
          </Tooltip>
        )
      })}
    </Box>
  )
}

// ------------------------------------------------------------------ 主布局：图标栏 | 正文（目录 + 页边批注）| 助手抽屉
export default function App() {
  const { docId, drawerOpen, setDrawerOpen, reader, rightTab, split } = useApp()
  useBootstrapDoc()
  const splitRef = useRef<AllotmentHandle>(null)
  const wrapRef = useRef<HTMLDivElement>(null)

  // 打开「原文」时把抽屉拉宽一些，PDF 才看得清；正文尽量留够 1000px，页边批注才不会被挤掉
  useEffect(() => {
    if (!reader || rightTab !== 'reader') return
    const t = window.setTimeout(() => {
      const total = wrapRef.current?.clientWidth ?? 0
      const drawer = Math.max(420, Math.min(620, total - 1000))
      if (total && total - drawer >= 560) splitRef.current?.resize([total - drawer, drawer])
    }, 30)
    return () => window.clearTimeout(t)
  }, [reader?.nonce]) // eslint-disable-line react-hooks/exhaustive-deps

  // 进入 / 退出分屏、开关抽屉时：正文和原文各占一半（抽屉保持原宽度）
  const layoutSplit = () => {
    const total = wrapRef.current?.clientWidth ?? 0
    if (!total) return
    const drawer = drawerOpen ? 400 : 0
    if (split) {
      // 1:1；打开抽屉后空间不够时优先保证正文至少 600px，原文区再窄也有 360px
      const avail = total - drawer
      const editor = Math.round(Math.max(avail / 2, Math.min(600, avail - 360)))
      splitRef.current?.resize([editor, avail - editor, drawer])
    } else splitRef.current?.resize([total - drawer, 0, drawer])
  }
  const wasSplit = useRef(split)
  useEffect(() => {
    // 非分屏时开关抽屉交给 Allotment 自己处理（打开原文时还会把抽屉拉宽），只在刚退出分屏时重排一次
    if (!split && !wasSplit.current) return
    wasSplit.current = split
    const t = window.setTimeout(layoutSplit, 30)
    return () => window.clearTimeout(t)
  }, [split, drawerOpen]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault()
        setDrawerOpen(!drawerOpen)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [drawerOpen, setDrawerOpen])

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#fff' }}>
      <TopBar />
      <SettingsDialog />
      <PdfPreviewDialog />
      <Box sx={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <Rail />
        <Box ref={wrapRef} sx={{ flex: 1, minWidth: 0 }}>
          <Allotment ref={splitRef} proportionalLayout={false} separator={false} onReset={layoutSplit}>
            <Allotment.Pane minSize={split ? 420 : 560}>
              <Box sx={{ height: '100%', p: 1.5, pr: drawerOpen || split ? 0.75 : 1.5, pt: 0.5, boxSizing: 'border-box' }}>
                {docId ? <PaperEditor key={docId} docId={docId} /> : <Box sx={{ p: 4, color: 'text.secondary' }}>正在加载…</Box>}
              </Box>
            </Allotment.Pane>
            {/* 分屏对照：拖动中间的分隔条调整比例，双击恢复 1:1 */}
            <Allotment.Pane minSize={360} visible={split} snap={false}>
              <SplitReader />
            </Allotment.Pane>
            <Allotment.Pane preferredSize={400} minSize={320} maxSize={760} visible={drawerOpen}>
              <DrawerPanel />
            </Allotment.Pane>
          </Allotment>
        </Box>
      </Box>
    </Box>
  )
}
