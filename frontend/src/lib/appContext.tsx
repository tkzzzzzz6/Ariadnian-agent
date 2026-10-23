import type { Editor } from '@tiptap/core'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Alert, Button, Snackbar } from '@mui/material'
import { api } from './api'
import { citeStore } from './citations'
import { notesStore } from './notes'
import { readingStore } from './readingStore'
import type { Health, Paper, PaperIn } from './types'

/** 右侧抽屉里的视图 */
export type RightTab = 'agent' | 'cite' | 'ask' | 'library' | 'reader' | 'docs'

export interface CiteRequest {
  sentence: string
  insertAt: number
  nonce: number
}

/** 一处引用依据：页码 + 原文片段，在「原文」里高亮 */
export interface Evidence {
  page?: number | null
  snippet?: string
  section?: string | null
  verdict?: string
  reason?: string
}

/** 在抽屉「原文」里打开一篇文献库论文 */
export interface ReaderTarget {
  paperId: string
  /** 正文里的那句话：用来从依据片段里挑出最相关的句子高亮 */
  claim?: string
  evidences?: Evidence[]
  /** 插入引用的位置（来自页边批注）；没有就插到光标处 */
  insertAt?: number | null
  nonce: number
}

export interface PreviewAction {
  label: string
  primary?: boolean
  onClick: () => void | Promise<void>
}

/** 检索预览弹窗：可以是文献库论文（有 id），也可以是还没入库的检索结果 */
export interface PreviewTarget {
  paper: PaperIn & { id?: string; has_pdf?: boolean }
  claim?: string
  verdict?: string
  reason?: string
  actions?: PreviewAction[]
}

interface AppCtx {
  editor: Editor | null
  setEditor: (e: Editor | null) => void
  papers: Paper[]
  refreshPapers: () => Promise<Paper[]>
  docId: string | null
  setDocId: (id: string | null) => void
  docTitle: string
  setDocTitle: (t: string) => void
  rightTab: RightTab
  setRightTab: (t: RightTab) => void
  citeRequest: CiteRequest | null
  requestCitations: (sentence: string, insertAt: number) => void
  notify: (msg: string, severity?: 'success' | 'info' | 'warning' | 'error') => void
  health: Health | null
  drawerOpen: boolean
  setDrawerOpen: (v: boolean) => void
  /** 打开抽屉并切到某个视图；已经打开在同一视图时再点一次就收起 */
  toggleDrawer: (tab: RightTab) => void
  copilot: boolean
  setCopilot: (v: boolean) => void
  review: boolean
  setReview: (v: boolean) => void
  docsVersion: number
  bumpDocs: () => void
  /** 打开设置页，可指定页签 */
  settingsTab: SettingsTab | null
  openSettings: (tab?: SettingsTab) => void
  closeSettings: () => void
  refreshHealth: () => Promise<void>
  reader: ReaderTarget | null
  openReader: (t: Omit<ReaderTarget, 'nonce'>) => void
  preview: PreviewTarget | null
  openPreview: (t: PreviewTarget) => void
  closePreview: () => void
  /** Agent 停下来等你 / 跑完了，而你此刻没在看 Agent 面板：图标栏和页签上出红点 */
  agentAlert: string | null
  alertAgent: (msg: string) => void
  /** 分屏对照：左边正文、右边原文（可同时开多篇，标签页切换） */
  split: boolean
  setSplit: (v: boolean) => void
  splitTabs: ReaderTarget[]
  splitActive: string | null
  setSplitActive: (paperId: string) => void
  closeSplitTab: (paperId: string) => void
  /** 在「文献问答」里提问（来自原文选中的句子等） */
  askRequest: { question: string; nonce: number } | null
  askAI: (question: string) => void
}

export type SettingsTab = 'model' | 'search' | 'embedding' | 'parse' | 'usage'

const Ctx = createContext<AppCtx | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [editor, setEditor] = useState<Editor | null>(null)
  const [papers, setPapers] = useState<Paper[]>([])
  const [docId, setDocId] = useState<string | null>(null)
  const [docTitle, setDocTitle] = useState('')
  const [rightTab, setRightTab] = useState<RightTab>('agent')
  const [citeRequest, setCiteRequest] = useState<CiteRequest | null>(null)
  const [toast, setToast] = useState<{ msg: string; severity: 'success' | 'info' | 'warning' | 'error'; action?: { label: string; onClick: () => void } } | null>(null)
  const [agentAlert, setAgentAlert] = useState<string | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [docsVersion, setDocsVersion] = useState(0)
  const bumpDocs = useCallback(() => setDocsVersion((v) => v + 1), [])
  const [copilot, setCopilotState] = useState(() => {
    try {
      return localStorage.getItem('aria.copilot') !== '0'
    } catch {
      return true
    }
  })
  const setCopilot = useCallback((v: boolean) => {
    setCopilotState(v)
    try {
      localStorage.setItem('aria.copilot', v ? '1' : '0')
    } catch {
      /* ignore */
    }
  }, [])
  const [review, setReviewState] = useState(() => {
    try {
      return localStorage.getItem('aria.review') !== '0'
    } catch {
      return true
    }
  })
  const setReview = useCallback((v: boolean) => {
    setReviewState(v)
    try {
      localStorage.setItem('aria.review', v ? '1' : '0')
    } catch {
      /* ignore */
    }
  }, [])

  const [settingsTab, setSettingsTab] = useState<SettingsTab | null>(null)
  const openSettings = useCallback((tab: SettingsTab = 'model') => setSettingsTab(tab), [])
  const closeSettings = useCallback(() => setSettingsTab(null), [])
  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health())
    } catch {
      setHealth(null)
    }
  }, [])

  const [reader, setReader] = useState<ReaderTarget | null>(null)
  const [split, setSplitState] = useState(() => {
    try {
      return localStorage.getItem('aria.split') === '1'
    } catch {
      return false
    }
  })
  const [splitTabs, setSplitTabs] = useState<ReaderTarget[]>(() => {
    try {
      return (JSON.parse(localStorage.getItem('aria.splitTabs') || '[]') as string[]).map((paperId, i) => ({ paperId, nonce: i }))
    } catch {
      return []
    }
  })
  const [splitActive, setSplitActive] = useState<string | null>(() => splitTabs[0]?.paperId ?? null)
  useEffect(() => {
    try {
      localStorage.setItem('aria.split', split ? '1' : '0')
      localStorage.setItem('aria.splitTabs', JSON.stringify(splitTabs.map((t) => t.paperId)))
    } catch {
      /* ignore */
    }
  }, [split, splitTabs])
  const splitRef = useRef(split)
  splitRef.current = split
  const readerRef = useRef(reader)
  readerRef.current = reader
  const setSplit = useCallback((v: boolean) => {
    setSplitState(v)
    // 进入分屏时，把抽屉里正在读的那篇带过去
    const r = readerRef.current
    if (v && r) {
      setSplitTabs((tabs) => (tabs.some((t) => t.paperId === r.paperId) ? tabs : [...tabs, r]))
      setSplitActive(r.paperId)
    }
    if (v) setRightTab((t) => (t === 'reader' ? 'agent' : t))
  }, [])
  const openReader = useCallback((t: Omit<ReaderTarget, 'nonce'>) => {
    const target = { ...t, nonce: Date.now() }
    if (splitRef.current) {
      // 分屏模式：在右边的原文区打开（已开过的标签就切过去并跳到新的依据）
      setSplitTabs((tabs) => (tabs.some((x) => x.paperId === t.paperId) ? tabs.map((x) => (x.paperId === t.paperId ? target : x)) : [...tabs, target]))
      setSplitActive(t.paperId)
      return
    }
    setReader(target)
    setRightTab('reader')
    setDrawerOpen(true)
  }, [])
  const closeSplitTab = useCallback((paperId: string) => {
    setSplitTabs((tabs) => {
      const i = tabs.findIndex((t) => t.paperId === paperId)
      const next = tabs.filter((t) => t.paperId !== paperId)
      setSplitActive((a) => (a === paperId ? next[Math.max(0, i - 1)]?.paperId ?? null : a))
      return next
    })
  }, [])
  const [askRequest, setAskRequest] = useState<{ question: string; nonce: number } | null>(null)
  const askAI = useCallback((question: string) => {
    setAskRequest({ question, nonce: Date.now() })
    setRightTab('ask')
    setDrawerOpen(true)
  }, [])
  const [preview, setPreview] = useState<PreviewTarget | null>(null)
  const openPreview = useCallback((t: PreviewTarget) => setPreview(t), [])
  const closePreview = useCallback(() => setPreview(null), [])

  const notify = useCallback<AppCtx['notify']>((msg, severity = 'info') => setToast({ msg, severity }), [])

  const refreshPapers = useCallback(async () => {
    const list = await api.papers()
    setPapers(list)
    citeStore.setPapers(list)
    notesStore.refresh().catch(() => undefined) // 删除文献时笔记也会被删掉
    void readingStore.refresh()
    return list
  }, [])

  useEffect(() => {
    refreshPapers().catch((e) => notify(`无法连接后端：${e.message}`, 'error'))
    api
      .health()
      .then((h) => {
        setHealth(h)
        // 第一次打开、还没配置模型：直接带用户去设置页
        let seen = false
        try {
          seen = localStorage.getItem('aria.onboarded') === '1'
          localStorage.setItem('aria.onboarded', '1')
        } catch {
          /* ignore */
        }
        if (!h.llm_ready && !seen) setSettingsTab('model')
      })
      .catch(() => setHealth(null))
  }, [refreshPapers, notify])

  const toggleDrawer = useCallback(
    (tab: RightTab) => {
      // 综述 Agent / 找引用 / 文献问答 同属「AI 助手」，图标栏上是同一个按钮
      const group = (t: RightTab) => (t === 'agent' || t === 'cite' || t === 'ask' ? 'assistant' : t)
      if (drawerOpen && group(tab) === group(rightTab)) {
        setDrawerOpen(false)
        return
      }
      if (group(tab) !== group(rightTab)) setRightTab(tab)
      setDrawerOpen(true)
    },
    [drawerOpen, rightTab],
  )

  // 打开了 Agent 页签，提醒就算看过了
  useEffect(() => {
    if (drawerOpen && rightTab === 'agent') setAgentAlert(null)
  }, [drawerOpen, rightTab])

  const watching = useRef({ drawerOpen, rightTab })
  watching.current = { drawerOpen, rightTab }
  const alertAgent = useCallback((msg: string) => {
    const w = watching.current
    if (w.drawerOpen && w.rightTab === 'agent' && !document.hidden) return
    setAgentAlert(msg)
    setToast({
      msg, severity: 'info',
      action: {
        label: '去看看',
        onClick: () => {
          setRightTab('agent')
          setDrawerOpen(true)
          setToast(null)
        },
      },
    })
  }, [])

  const requestCitations = useCallback((sentence: string, insertAt: number) => {
    setDrawerOpen(true)
    setRightTab('cite')
    setCiteRequest({ sentence, insertAt, nonce: Date.now() })
  }, [])

  const value = useMemo<AppCtx>(
    () => ({
      editor, setEditor, papers, refreshPapers, docId, setDocId, docTitle, setDocTitle,
      rightTab, setRightTab, citeRequest, requestCitations, notify, health,
      drawerOpen, setDrawerOpen, toggleDrawer, copilot, setCopilot, review, setReview, docsVersion, bumpDocs,
      settingsTab, openSettings, closeSettings, refreshHealth, reader, openReader, preview, openPreview, closePreview,
      agentAlert, alertAgent, split, setSplit, splitTabs, splitActive, setSplitActive, closeSplitTab, askRequest, askAI,
    }),
    [editor, papers, refreshPapers, docId, docTitle, rightTab, citeRequest, requestCitations, notify, health,
      drawerOpen, toggleDrawer, copilot, setCopilot, review, setReview, docsVersion, bumpDocs,
      settingsTab, openSettings, closeSettings, refreshHealth, reader, openReader, preview, openPreview, closePreview,
      agentAlert, alertAgent, split, setSplit, splitTabs, splitActive, closeSplitTab, askRequest, askAI],
  )

  return (
    <Ctx.Provider value={value}>
      {children}
      <Snackbar
        open={!!toast}
        autoHideDuration={4000}
        onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {toast ? (
          <Alert
            severity={toast.severity}
            variant="filled"
            onClose={() => setToast(null)}
            action={
              toast.action ? (
                <Button color="inherit" size="small" onClick={toast.action.onClick} sx={{ fontWeight: 600 }}>
                  {toast.action.label}
                </Button>
              ) : undefined
            }
            sx={{ maxWidth: 560, alignItems: 'center' }}
          >
            {toast.msg}
          </Alert>
        ) : undefined}
      </Snackbar>
    </Ctx.Provider>
  )
}

export function useApp(): AppCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp must be used inside AppProvider')
  return v
}
