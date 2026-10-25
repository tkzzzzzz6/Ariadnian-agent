import type { Editor } from '@tiptap/core'
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Box, Button, ButtonGroup, CircularProgress, Divider, IconButton, Paper, TextField, Tooltip, Typography,
} from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import FormatQuoteIcon from '@mui/icons-material/FormatQuote'
import CloseIcon from '@mui/icons-material/Close'
import CheckIcon from '@mui/icons-material/Check'
import ReplayIcon from '@mui/icons-material/Replay'
import EditNoteIcon from '@mui/icons-material/EditNote'
import { streamSSE } from '../lib/api'
import { useApp } from '../lib/appContext'
import { contentFromText, textBetween } from '../lib/editorText'
import { brand } from '../theme'
import { DiffView } from './DiffView'
import { FormatActions } from './FormatActions'
import { containsRich } from '../lib/richContent'

const ACTIONS: { key: string; label: string; tip: string }[] = [
  { key: 'polish', label: '润色', tip: '修正语法用词，更流畅' },
  { key: 'academic', label: '学术化', tip: '改成正式的学术语体' },
  { key: 'rewrite', label: '改写', tip: '换一种表达方式' },
  { key: 'expand', label: '扩写', tip: '补充论证与过渡（不编造数据）' },
  { key: 'shorten', label: '精简', tip: '删去冗余' },
  { key: 'translate_en', label: '译英', tip: '翻译为学术英文' },
  { key: 'translate_zh', label: '译中', tip: '翻译为学术中文' },
]
const LABEL: Record<string, string> = Object.fromEntries([...ACTIONS.map((a) => [a.key, a.label]), ['custom', '自定义']])

interface EditSession {
  from: number
  to: number
  original: string
  action: string
  instruction: string
  result: string
  status: 'input' | 'streaming' | 'done' | 'error'
  error?: string
}

function rectOf(editor: Editor, from: number, to: number) {
  const a = editor.view.coordsAtPos(from)
  const b = editor.view.coordsAtPos(to)
  return { top: Math.min(a.top, b.top), bottom: Math.max(a.bottom, b.bottom), left: Math.min(a.left, b.left) }
}

export function SelectionAssist({ editor }: { editor: Editor }) {
  const { requestCitations, notify } = useApp()
  const [menu, setMenu] = useState<{ top: number; left: number } | null>(null)
  const [session, setSession] = useState<EditSession | null>(null)
  const [panelPos, setPanelPos] = useState<{ top: number; left: number } | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const sessionRef = useRef(session)
  sessionRef.current = session

  // 选区变化 → 更新浮动菜单位置
  const updateMenu = useCallback(() => {
    const { from, to, empty } = editor.state.selection
    if (empty || sessionRef.current || !editor.isEditable || !editor.isFocused) return setMenu(null)
    const text = textBetween(editor, from, to)
    if (text.trim().length < 1) return setMenu(null)
    const r = rectOf(editor, from, to)
    setMenu({ top: Math.max(8, r.top - 84), left: Math.max(8, Math.min(r.left, window.innerWidth - 600)) })
  }, [editor])

  useEffect(() => {
    const onSel = () => updateMenu()
    const onBlur = ({ event }: { event: FocusEvent }) => {
      const rel = event.relatedTarget as HTMLElement | null
      if (rel?.closest('[data-selection-assist]')) return
      setMenu(null)
    }
    editor.on('selectionUpdate', onSel)
    editor.on('focus', onSel)
    editor.on('blur', onBlur)
    window.addEventListener('scroll', onSel, true)
    window.addEventListener('resize', onSel)
    return () => {
      editor.off('selectionUpdate', onSel)
      editor.off('focus', onSel)
      editor.off('blur', onBlur)
      window.removeEventListener('scroll', onSel, true)
      window.removeEventListener('resize', onSel)
    }
  }, [editor, updateMenu])

  const placePanel = useCallback(
    (from: number, to: number) => {
      const r = rectOf(editor, from, to)
      const height = 320
      const top = r.bottom + 8 + height < window.innerHeight ? r.bottom + 8 : Math.max(8, r.top - height - 8)
      setPanelPos({ top, left: Math.max(8, Math.min(r.left, window.innerWidth - 620)) })
    },
    [editor],
  )

  const run = useCallback(
    async (s: EditSession) => {
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      const ctx = textBetween(editor, Math.max(0, s.from - 800), s.from)
      setSession({ ...s, result: '', status: 'streaming', error: undefined })
      let acc = ''
      try {
        await streamSSE(
          '/api/assist/edit',
          { text: s.original, action: s.action, instruction: s.instruction, context: ctx },
          (e) => {
            if (e.type === 'token') {
              acc += e.text
              setSession((cur) => (cur ? { ...cur, result: acc } : cur))
            } else if (e.type === 'error') {
              throw new Error(e.message)
            }
          },
          ctrl.signal,
        )
        setSession((cur) => (cur ? { ...cur, result: acc.trim(), status: 'done' } : cur))
      } catch (err: any) {
        if (ctrl.signal.aborted) return
        setSession((cur) => (cur ? { ...cur, status: 'error', error: err.message } : cur))
      }
    },
    [editor],
  )

  const start = (action: string) => {
    const { from, to } = editor.state.selection
    if (containsRich(editor.state.doc, from, to)) {
      notify('此选区包含图表或公式，请选择普通正文文字进行改写', 'warning')
      return setMenu(null)
    }
    const original = textBetween(editor, from, to)
    setMenu(null)
    placePanel(from, to)
    const s: EditSession = { from, to, original, action, instruction: '', result: '', status: action === 'custom' ? 'input' : 'streaming' }
    setSession(s)
    if (action !== 'custom') run(s)
  }

  const close = () => {
    abortRef.current?.abort()
    setSession(null)
    editor.commands.focus()
  }

  const accept = () => {
    if (!session) return
    if (containsRich(editor.state.doc, session.from, session.to)) {
      notify('选区出现了图表或公式，已取消整体替换，请重新选择文字', 'warning')
      return close()
    }
    if (textBetween(editor, session.from, session.to) !== session.original) {
      notify('原文在改写期间被修改了，请重新选中后再试', 'warning')
      return close()
    }
    editor
      .chain()
      .focus()
      .insertContentAt({ from: session.from, to: session.to }, contentFromText(session.result))
      .run()
    setSession(null)
  }

  const findCites = () => {
    const { from, to } = editor.state.selection
    setMenu(null)
    requestCitations(textBetween(editor, from, to), to)
  }

  // 快捷键：Ctrl+Enter 接受，Esc 放弃
  useEffect(() => {
    if (!session) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        close()
      } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && session.status === 'done') {
        e.preventDefault()
        accept()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return createPortal(
    <>
      {menu && !session && (
        <Paper
          data-selection-assist
          elevation={4}
          onMouseDown={(e) => e.preventDefault()}
          sx={{
            position: 'fixed', top: menu.top, left: menu.left, zIndex: 1400, display: 'flex', alignItems: 'center',
            flexWrap: 'wrap', width: 580, maxWidth: 'calc(100vw - 16px)', gap: 0.25, px: 0.5, py: 0.25, border: `1px solid ${brand.line}`, borderRadius: 1.5,
          }}
        >
          <Box sx={{width:'100%',display:'flex',alignItems:'center',borderBottom:`1px solid ${brand.lineSoft}`,pb:0.25}}><FormatActions editor={editor}/><Typography sx={{fontSize:11,color:brand.faint,ml:1}}>文字格式</Typography></Box>
          <AutoFixHighIcon sx={{ fontSize: 16, color: brand.blue, mx: 0.75 }} />
          <ButtonGroup variant="text" size="small" sx={{ overflowX: 'auto', maxWidth: 'calc(100% - 40px)', '& .MuiButton-root': { flexShrink: 0, whiteSpace: 'nowrap', px: 1, color: brand.ink, borderColor: 'transparent !important' } }}>
            {ACTIONS.map((a) => (
              <Tooltip key={a.key} title={a.tip} describeChild>
                <Button onClick={() => start(a.key)}>{a.label}</Button>
              </Tooltip>
            ))}
            <Tooltip title="按你的要求改写" describeChild>
              <Button onClick={() => start('custom')} startIcon={<EditNoteIcon sx={{ fontSize: 16 }} />}>
                自定义
              </Button>
            </Tooltip>
          </ButtonGroup>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />
          <Tooltip title="在文献库中查找能支撑这段话的文献" describeChild>
            <Button size="small" onClick={findCites} startIcon={<FormatQuoteIcon sx={{ fontSize: 16 }} />} sx={{ color: brand.navy, fontWeight: 600 }}>
              找引用
            </Button>
          </Tooltip>
        </Paper>
      )}

      {session && panelPos && (
        <Paper
          data-selection-assist
          elevation={8}
          sx={{
            position: 'fixed', top: panelPos.top, left: panelPos.left, zIndex: 1400, width: 600, maxWidth: 'calc(100vw - 16px)', maxHeight: 360,
            display: 'flex', flexDirection: 'column', border: `1px solid ${brand.line}`, borderTop: `3px solid ${brand.blue}`,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.75, gap: 1 }}>
            <AutoFixHighIcon sx={{ fontSize: 16, color: brand.blue }} />
            <Typography variant="subtitle2" sx={{ flex: 1 }}>
              AI {LABEL[session.action]}
              {session.status === 'streaming' && <CircularProgress size={12} sx={{ ml: 1 }} />}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Ctrl+Enter 接受 · Esc 放弃
            </Typography>
            <IconButton onClick={close}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>
          <Divider />
          {session.status === 'input' ? (
            <Box sx={{ p: 1.5, display: 'flex', gap: 1 }}>
              <TextField
                autoFocus
                fullWidth
                placeholder="例如：改成被动语态；突出与已有方法的区别；控制在 80 字以内"
                value={session.instruction}
                onChange={(e) => setSession({ ...session, instruction: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && session.instruction.trim()) run(session)
                }}
              />
              <Button variant="contained" disabled={!session.instruction.trim()} onClick={() => run(session)}>
                生成
              </Button>
            </Box>
          ) : (
            <Box sx={{ p: 1.5, overflow: 'auto', flex: 1 }}>
              {session.status === 'error' ? (
                <Typography color="error" variant="body2">
                  {session.error}
                </Typography>
              ) : session.status === 'streaming' ? (
                <Typography sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.8, fontSize: 14, fontFamily: 'var(--paper-font)' }}>
                  {session.result || '…'}
                </Typography>
              ) : (
                <DiffView before={session.original} after={session.result} />
              )}
            </Box>
          )}
          {session.status !== 'input' && (
            <>
              <Divider />
              <Box sx={{ display: 'flex', gap: 1, px: 1.5, py: 1, justifyContent: 'flex-end' }}>
                <Button startIcon={<ReplayIcon />} onClick={() => run(session)} disabled={session.status === 'streaming'}>
                  重新生成
                </Button>
                <Button onClick={close}>放弃</Button>
                <Button variant="contained" color="primary" startIcon={<CheckIcon />} onClick={accept} disabled={session.status !== 'done' || !session.result}>
                  接受
                </Button>
              </Box>
            </>
          )}
        </Paper>
      )}
    </>,
    document.body,
  )
}
