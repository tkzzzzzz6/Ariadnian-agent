import TextFieldsIcon from '@mui/icons-material/TextFields'
import { AppearancePanel } from './AppearancePanel'
import { resetAppearance } from '../lib/appearance'
import { useCallback, useEffect, useMemo, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'
import {
  Autocomplete, Box, Button, Checkbox, Chip, CircularProgress, Collapse, Dialog, FormControlLabel, IconButton, InputAdornment,
  Link, Radio, RadioGroup, Switch, TextField, Tooltip, Typography,
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined'
import TravelExploreIcon from '@mui/icons-material/TravelExplore'
import HubOutlinedIcon from '@mui/icons-material/HubOutlined'
import QueryStatsIcon from '@mui/icons-material/QueryStats'
import PictureAsPdfOutlinedIcon from '@mui/icons-material/PictureAsPdfOutlined'
import VisibilityOutlinedIcon from '@mui/icons-material/VisibilityOutlined'
import VisibilityOffOutlinedIcon from '@mui/icons-material/VisibilityOffOutlined'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ErrorIcon from '@mui/icons-material/Error'
import { api } from '../lib/api'
import { useApp, type SettingsTab } from '../lib/appContext'
import type { SettingsView, TestResult } from '../lib/types'
import { UsagePanel } from './UsagePanel'
import { brand, card } from '../theme'

/** 网页里「明确留空」，不回退到 .env 的旧值 */
const EMPTY = '__none__'
const EMPTY_AS_NONE = new Set(['LLM_API_BASE', 'EMBEDDING_BASE_URL', 'MODEL_COMPLETE_API_BASE', 'MODEL_WRITE_API_BASE', 'MODEL_JUDGE_API_BASE', 'MODEL_AGENT_API_BASE'])

interface Preset {
  id: string
  name: string
  base: string
  main: string
  fast: string
  models: string[]
  site?: string
  noKey?: boolean
  note?: string
}

// 模型名只是示例，各家会更新型号，以服务商控制台为准
const PRESETS: Preset[] = [
  { id: 'deepseek', name: 'DeepSeek', base: '', main: 'deepseek/deepseek-chat', fast: 'deepseek/deepseek-chat', models: ['deepseek/deepseek-chat', 'deepseek/deepseek-reasoner'], site: 'https://platform.deepseek.com/' },
  { id: 'qwen', name: '通义千问', base: 'https://dashscope.aliyuncs.com/compatible-mode/v1', main: 'openai/qwen-plus', fast: 'openai/qwen-turbo', models: ['openai/qwen-plus', 'openai/qwen-turbo', 'openai/qwen-max'], site: 'https://bailian.console.aliyun.com/' },
  { id: 'kimi', name: 'Kimi', base: 'https://api.moonshot.cn/v1', main: 'openai/moonshot-v1-32k', fast: 'openai/moonshot-v1-8k', models: ['openai/moonshot-v1-32k', 'openai/moonshot-v1-8k'], site: 'https://platform.moonshot.cn/' },
  { id: 'zhipu', name: '智谱 GLM', base: 'https://open.bigmodel.cn/api/paas/v4', main: 'openai/glm-4-plus', fast: 'openai/glm-4-flash', models: ['openai/glm-4-plus', 'openai/glm-4-flash'], site: 'https://open.bigmodel.cn/' },
  { id: 'siliconflow', name: '硅基流动', base: 'https://api.siliconflow.cn/v1', main: 'openai/deepseek-ai/DeepSeek-V3', fast: 'openai/Qwen/Qwen2.5-7B-Instruct', models: ['openai/deepseek-ai/DeepSeek-V3', 'openai/Qwen/Qwen2.5-7B-Instruct'], site: 'https://cloud.siliconflow.cn/' },
  { id: 'openai', name: 'OpenAI', base: '', main: 'openai/gpt-4o-mini', fast: 'openai/gpt-4o-mini', models: ['openai/gpt-4o', 'openai/gpt-4o-mini'], site: 'https://platform.openai.com/' },
  { id: 'ollama', name: 'Ollama 本地', base: 'http://localhost:11434', main: 'ollama_chat/qwen2.5:7b', fast: 'ollama_chat/qwen2.5:7b', models: ['ollama_chat/qwen2.5:7b', 'ollama_chat/qwen2.5:14b', 'ollama_chat/llama3.1:8b'], noKey: true, note: '在本机运行，不需要 key，先用 ollama pull 下载模型' },
  { id: 'custom', name: '自定义', base: '', main: '', fast: '', models: [], note: '任何 OpenAI 兼容接口：填接口地址，模型写成 openai/模型名' },
]

const TIERS: { key: 'complete' | 'write' | 'judge' | 'agent'; label: string; desc: string; inherit: 'MODEL_ID' | 'FAST_MODEL_ID' }[] = [
  { key: 'complete', label: '行内续写', desc: '延迟最敏感，建议非推理的快模型', inherit: 'FAST_MODEL_ID' },
  { key: 'write', label: '改写 / 问答', desc: '质量优先，流式输出', inherit: 'MODEL_ID' },
  { key: 'judge', label: '引用判断', desc: '页边批注、引用核查，调用量最大，建议便宜的模型', inherit: 'FAST_MODEL_ID' },
  { key: 'agent', label: '综述 / 大纲', desc: '长上下文，要能稳定输出 JSON', inherit: 'MODEL_ID' },
]

const SOURCE_TEXT: Record<string, string> = { ui: '网页设置', env: '.env', default: '默认值' }

// ------------------------------------------------------------------ 小部件
function Section({ title, desc, children, right }: { title: string; desc?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', mb: 1.25 }}>
        <Box sx={{ flex: 1 }}>
          <Typography sx={{ fontSize: 14, fontWeight: 600 }}>{title}</Typography>
          {desc && <Typography sx={{ fontSize: 12, color: brand.muted, mt: 0.25 }}>{desc}</Typography>}
        </Box>
        {right}
      </Box>
      {children}
    </Box>
  )
}

function SourceTag({ source }: { source?: string }) {
  if (!source || source === 'default') return null
  return (
    <Chip
      size="small"
      label={SOURCE_TEXT[source]}
      sx={{ height: 18, fontSize: 10.5, ml: 1, bgcolor: source === 'ui' ? 'rgba(0,120,212,0.08)' : 'rgba(0,0,0,0.05)', color: source === 'ui' ? brand.blue : brand.muted }}
    />
  )
}

function TestBadge({ r, busy }: { r?: TestResult | null; busy?: boolean }) {
  if (busy)
    return (
      <Typography sx={{ fontSize: 12, color: brand.muted, display: 'flex', alignItems: 'center', gap: 0.75 }}>
        <CircularProgress size={12} /> 正在测试…
      </Typography>
    )
  if (!r) return null
  if (r.ok)
    return (
      <Typography sx={{ fontSize: 12, color: brand.green, display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <CheckCircleIcon sx={{ fontSize: 15 }} />
        {r.fake ? r.reply ?? r.sample : `连接成功 · ${r.latency_ms} ms${r.version ? ` · 版本 ${r.version}` : ''}${r.reply ? ` · 回复「${r.reply}」` : ''}${r.sample ? ` · 示例：${r.sample.slice(0, 40)}` : ''}`}
      </Typography>
    )
  return (
    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.5 }}>
      <ErrorIcon sx={{ fontSize: 15, mt: '1px', flexShrink: 0, color: brand.red }} />
      <Box sx={{ minWidth: 0 }}>
        <Typography sx={{ fontSize: 12.5, color: brand.red, fontWeight: 500 }}>{r.hint ?? '测试失败'}</Typography>
        <Typography sx={{ fontSize: 11, color: brand.faint, wordBreak: 'break-all' }}>{r.error}</Typography>
      </Box>
    </Box>
  )
}

function SecretField({
  label, name, view, draft, set, helper,
}: {
  label: string
  name: string
  view: SettingsView
  draft: Record<string, string | null>
  set: (k: string, v: string | null) => void
  helper?: ReactNode
}) {
  const [show, setShow] = useState(false)
  const info = view.values[name]
  const cleared = draft[name] === EMPTY
  const typed = name in draft && !cleared ? draft[name] ?? '' : ''
  const placeholder = cleared ? '将清除' : info?.set ? `已设置 ${info.hint ?? ''}，留空则保持不变` : '未设置'
  return (
    <TextField
      fullWidth
      label={
        <span>
          {label}
          {!(name in draft) && <SourceTag source={info?.set ? info.source : undefined} />}
        </span>
      }
      type={show ? 'text' : 'password'}
      value={typed}
      placeholder={placeholder}
      autoComplete="off"
      onChange={(e) => set(name, e.target.value)}
      helperText={helper}
      slotProps={{
        inputLabel: { shrink: true },
        input: {
          endAdornment: (
            <InputAdornment position="end">
              {info?.set && !cleared && (
                <Button size="small" onClick={() => set(name, EMPTY)} sx={{ color: brand.muted, minWidth: 0, fontSize: 12 }}>
                  清除
                </Button>
              )}
              <IconButton onClick={() => setShow(!show)} aria-label={show ? '隐藏' : '显示'}>
                {show ? <VisibilityOffOutlinedIcon fontSize="small" /> : <VisibilityOutlinedIcon fontSize="small" />}
              </IconButton>
            </InputAdornment>
          ),
        },
      }}
    />
  )
}

// ------------------------------------------------------------------ 可拖拽调整大小
const SIZE_KEY = 'aria.settings.size'
const MIN_W = 680
const MIN_H = 460

function clampSize(w: number, h: number) {
  const maxW = Math.max(MIN_W, window.innerWidth - 32)
  const maxH = Math.max(MIN_H, window.innerHeight - 32)
  return { w: Math.round(Math.min(maxW, Math.max(MIN_W, w))), h: Math.round(Math.min(maxH, Math.max(MIN_H, h))) }
}

/** 设置窗口的宽高：拖右边、下边或右下角调整，记在浏览器里，下次打开保持 */
function useResizableSize() {
  const [size, setSize] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(SIZE_KEY) || 'null')
      if (saved && typeof saved.w === 'number' && typeof saved.h === 'number') return clampSize(saved.w, saved.h)
    } catch {
      /* 读不到就用默认大小 */
    }
    return clampSize(900, 760)
  })

  useEffect(() => {
    const onResize = () => setSize((s) => clampSize(s.w, s.h))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const start = useCallback(
    (dir: 'x' | 'y' | 'xy') => (e: ReactPointerEvent) => {
      e.preventDefault()
      const x0 = e.clientX
      const y0 = e.clientY
      const { w: w0, h: h0 } = size
      let last = size
      // 窗口居中显示，边缘移动 1px 需要尺寸变化 2px，拖动才跟手
      const move = (ev: PointerEvent) => {
        last = clampSize(dir === 'y' ? w0 : w0 + (ev.clientX - x0) * 2, dir === 'x' ? h0 : h0 + (ev.clientY - y0) * 2)
        setSize(last)
      }
      const up = () => {
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        try {
          localStorage.setItem(SIZE_KEY, JSON.stringify(last))
        } catch {
          /* 存不了就算了 */
        }
      }
      document.body.style.cursor = dir === 'x' ? 'ew-resize' : dir === 'y' ? 'ns-resize' : 'nwse-resize'
      document.body.style.userSelect = 'none'
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
    },
    [size],
  )
  return { size, start }
}

function ResizeHandles({ start }: { start: (dir: 'x' | 'y' | 'xy') => (e: ReactPointerEvent) => void }) {
  return (
    <>
      <Box onPointerDown={start('x')} sx={{ position: 'absolute', top: 0, right: 0, width: 6, height: '100%', cursor: 'ew-resize', zIndex: 2 }} />
      <Box onPointerDown={start('y')} sx={{ position: 'absolute', left: 0, bottom: 0, height: 6, width: '100%', cursor: 'ns-resize', zIndex: 2 }} />
      <Box
        onPointerDown={start('xy')}
        aria-label="拖动调整窗口大小"
        sx={{
          position: 'absolute', right: 2, bottom: 2, width: 16, height: 16, cursor: 'nwse-resize', zIndex: 3,
          // 右下角三条斜线的拖拽提示
          background: `linear-gradient(135deg, transparent 0 55%, ${brand.faint} 55% 60%, transparent 60% 70%, ${brand.faint} 70% 75%, transparent 75% 85%, ${brand.faint} 85% 90%, transparent 90%)`,
          opacity: 0.7, '&:hover': { opacity: 1 },
        }}
      />
    </>
  )
}

// ------------------------------------------------------------------ 主组件
export function SettingsDialog() {
  const { settingsTab, openSettings, closeSettings, refreshHealth, notify } = useApp()
  const open = settingsTab !== null
  const [view, setView] = useState<SettingsView | null>(null)
  const [draft, setDraft] = useState<Record<string, string | null>>({})
  const [saving, setSaving] = useState(false)
  const [tests, setTests] = useState<Record<string, TestResult | null>>({})
  const [busy, setBusy] = useState<Record<string, boolean>>({})
  const [advanced, setAdvanced] = useState(false)
  const { size, start } = useResizableSize()

  useEffect(() => {
    if (!open) return
    setDraft({})
    setTests({})
    api.settings().then(setView).catch((e) => notify(`读取设置失败：${e.message}`, 'error'))
  }, [open, notify])

  const set = (k: string, v: string | null) => setDraft((d) => ({ ...d, [k]: v }))
  /** 普通字段当前显示值：草稿优先，其次已生效的值 */
  const val = (k: string): string => {
    if (k in draft) return draft[k] === EMPTY ? '' : draft[k] ?? ''
    const v = view?.values[k]?.value
    return v && v !== EMPTY ? v : ''
  }
  const dirty = Object.keys(draft).length > 0

  const currentPreset = useMemo(() => {
    const base = val('LLM_API_BASE')
    const main = val('MODEL_ID')
    if (main.startsWith('ollama')) return 'ollama'
    if (main.startsWith('deepseek/') && !base) return 'deepseek'
    const hit = PRESETS.find((p) => p.base && base && base.replace(/\/$/, '') === p.base)
    if (hit) return hit.id
    if (!base && main.startsWith('openai/gpt')) return 'openai'
    return 'custom'
  }, [draft, view]) // eslint-disable-line react-hooks/exhaustive-deps

  const applyPreset = (p: Preset) => {
    setDraft((d) => {
      const next: Record<string, string | null> = { ...d, LLM_API_BASE: p.base || EMPTY }
      if (p.main) next.MODEL_ID = p.main
      if (p.fast) next.FAST_MODEL_ID = p.fast
      return next
    })
  }

  const runTest = async (id: string, fn: () => Promise<TestResult>) => {
    setBusy((b) => ({ ...b, [id]: true }))
    try {
      const r = await fn()
      setTests((t) => ({ ...t, [id]: r }))
    } catch (e: any) {
      setTests((t) => ({ ...t, [id]: { ok: false, latency_ms: 0, error: e.message } }))
    } finally {
      setBusy((b) => ({ ...b, [id]: false }))
    }
  }

  /** 测试用到的值：草稿里有就用草稿（还没保存也能测） */
  const draftOrUndef = (k: string): string | undefined => (k in draft && draft[k] !== EMPTY ? draft[k] ?? undefined : undefined)
  const testMain = (which: 'MODEL_ID' | 'FAST_MODEL_ID') =>
    runTest(which, () =>
      api.testLLM({
        tier: which === 'MODEL_ID' ? 'write' : 'complete',
        model: val(which) || undefined,
        api_base: 'LLM_API_BASE' in draft ? (draft.LLM_API_BASE === EMPTY ? '' : draft.LLM_API_BASE ?? '') : undefined,
        api_key: draftOrUndef('LLM_API_KEY'),
      }),
    )

  const save = async () => {
    if (!dirty) return closeSettings()
    const payload: Record<string, string | null> = {}
    for (const [k, v] of Object.entries(draft)) {
      if (v === EMPTY) payload[k] = EMPTY
      else if (v === null || v.trim() === '') payload[k] = EMPTY_AS_NONE.has(k) ? EMPTY : null
      else payload[k] = v.trim()
    }
    setSaving(true)
    try {
      setView(await api.saveSettings(payload))
      setDraft({})
      await refreshHealth()
      notify('设置已保存，立即生效', 'success')
    } catch (e: any) {
      notify(`保存失败：${e.message}`, 'error')
    } finally {
      setSaving(false)
    }
  }

  const NAV: [SettingsTab, string, ReactNode][] = [
    ['appearance', '外观', <TextFieldsIcon fontSize="small" />],
    ['model', '模型', <SmartToyOutlinedIcon fontSize="small" />],
    ['search', '文献检索', <TravelExploreIcon fontSize="small" />],
    ['embedding', '向量检索', <HubOutlinedIcon fontSize="small" />],
    ['parse', 'PDF 解析', <PictureAsPdfOutlinedIcon fontSize="small" />],
    ['usage', '用量统计', <QueryStatsIcon fontSize="small" />],
  ]

  const sources = (val('SEARCH_SOURCES') || 'semantic_scholar,arxiv').split(',').filter(Boolean)
  const toggleSource = (s: string, on: boolean) => {
    const next = on ? [...new Set([...sources, s])] : sources.filter((x) => x !== s)
    if (next.length) set('SEARCH_SOURCES', next.join(','))
  }

  return (
    <Dialog
      open={open}
      onClose={closeSettings}
      maxWidth={false}
      slotProps={{
        // 外观页：背景几乎不遮挡，调的时候能直接看到背后正文的变化
        backdrop: settingsTab === 'appearance' ? { sx: { bgcolor: 'rgba(0,0,0,0.06)' } } : undefined,
        paper: {
          sx: {
            width: size.w, height: size.h, maxWidth: 'none', maxHeight: 'none', m: 2, overflow: 'hidden', position: 'relative',
            // 输入框上方的标签用蓝色，和边框区分开
            '& .MuiInputLabel-root': { color: brand.blue, fontWeight: 500 },
            '& .MuiInputLabel-root.Mui-disabled': { color: brand.faint },
          },
        },
      }}
    >
      <ResizeHandles start={start} />
      <Box sx={{ display: 'flex', height: '100%' }}>
        {/* 左侧导航 */}
        <Box sx={{ width: 168, flexShrink: 0, bgcolor: '#fafafa', borderRight: `1px solid ${brand.lineSoft}`, py: 2 }}>
          <Typography sx={{ px: 2.5, fontSize: 16, fontWeight: 600, mb: 2 }}>设置</Typography>
          {NAV.map(([k, label, icon]) => (
            <Box
              key={k}
              role="button"
              onClick={() => openSettings(k)}
              sx={{
                mx: 1, px: 1.5, py: 0.9, borderRadius: '6px', display: 'flex', alignItems: 'center', gap: 1, cursor: 'pointer', fontSize: 13.5,
                color: settingsTab === k ? brand.blue : brand.ink, bgcolor: settingsTab === k ? 'rgba(0,120,212,0.08)' : 'transparent',
                '&:hover': { bgcolor: settingsTab === k ? 'rgba(0,120,212,0.1)' : 'rgba(0,0,0,0.04)' },
              }}
            >
              {icon}
              {label}
            </Box>
          ))}
        </Box>

        {/* 右侧内容 */}
        <Box sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', px: 1, pt: 1 }}>
            <IconButton onClick={closeSettings} aria-label="关闭设置">
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>
          <Box sx={{ flex: 1, overflow: 'auto', px: 3.5, pb: 2 }}>
            {settingsTab === 'appearance' ? (
              <AppearancePanel />
            ) : !view ? (
              <CircularProgress size={20} />
            ) : settingsTab === 'model' ? (
              <>
                <Section
                  title="离线演示模式"
                  desc="打开后不调用任何模型和检索服务，返回演示内容，适合先熟悉界面。"
                  right={<Switch checked={val('LLM_FAKE') === '1' || (!('LLM_FAKE' in draft) && view.fake)} onChange={(e) => set('LLM_FAKE', e.target.checked ? '1' : '0')} />}
                >
                  {null}
                </Section>

                <Section title="模型服务商" desc="选一个服务商会自动填好接口地址和示例模型，再填上你的 API Key 即可。">
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1 }}>
                    {PRESETS.map((p) => {
                      const on = currentPreset === p.id
                      return (
                        <Box
                          key={p.id}
                          role="button"
                          aria-pressed={on}
                          onClick={() => applyPreset(p)}
                          sx={{
                            ...(on ? card.data : card.plain), borderRadius: '8px', px: 1.25, py: 1, cursor: 'pointer', fontSize: 13,
                            fontWeight: on ? 600 : 400, color: on ? brand.blue : brand.ink, textAlign: 'center',
                            '&:hover': { borderColor: 'rgba(0,120,212,0.55)' },
                          }}
                        >
                          {p.name}
                        </Box>
                      )
                    })}
                  </Box>
                  {(() => {
                    const p = PRESETS.find((x) => x.id === currentPreset)
                    return p?.note || p?.site ? (
                      <Typography sx={{ fontSize: 12, color: brand.muted, mt: 1 }}>
                        {p.note}
                        {p.site && (
                          <>
                            {p.note ? '　' : ''}在{' '}
                            <Link href={p.site} target="_blank" rel="noreferrer">
                              {p.name} 控制台
                            </Link>{' '}
                            创建 API Key。模型名仅为示例，以控制台为准。
                          </>
                        )}
                      </Typography>
                    ) : null
                  })()}
                </Section>

                <Section title="连接">
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                    {currentPreset !== 'ollama' && <SecretField label="API Key" name="LLM_API_KEY" view={view} draft={draft} set={set} />}
                    <TextField
                      fullWidth
                      label={
                        <span>
                          接口地址
                          {!('LLM_API_BASE' in draft) && <SourceTag source={view.values.LLM_API_BASE?.set ? view.values.LLM_API_BASE.source : undefined} />}
                        </span>
                      }
                      placeholder="留空表示使用服务商默认地址（DeepSeek、OpenAI 不需要填）"
                      value={val('LLM_API_BASE')}
                      onChange={(e) => set('LLM_API_BASE', e.target.value)}
                      slotProps={{ inputLabel: { shrink: true } }}
                    />
                    {(['MODEL_ID', 'FAST_MODEL_ID'] as const).map((k) => (
                      <Box key={k}>
                        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
                          <Autocomplete
                            freeSolo
                            fullWidth
                            options={PRESETS.find((p) => p.id === currentPreset)?.models ?? []}
                            inputValue={val(k)}
                            onInputChange={(_, v, reason) => reason !== 'reset' && set(k, v)}
                            onChange={(_, v) => typeof v === 'string' && set(k, v)}
                            renderInput={(params) => (
                              <TextField
                                {...params}
                                label={
                                  <span>
                                    {k === 'MODEL_ID' ? '主模型（改写、问答、综述）' : '快模型（续写、引用判断）'}
                                    {!(k in draft) && <SourceTag source={view.values[k]?.source} />}
                                  </span>
                                }
                                placeholder={k === 'FAST_MODEL_ID' ? '留空则与主模型相同' : 'deepseek/deepseek-chat'}
                                helperText={k === 'MODEL_ID' ? '格式：服务商/模型名。OpenAI 兼容接口写 openai/模型名' : undefined}
                                slotProps={{ inputLabel: { shrink: true } }}
                              />
                            )}
                          />
                          <Button variant="outlined" sx={{ mt: '2px', height: 37, flexShrink: 0 }} onClick={() => testMain(k)} disabled={busy[k]}>
                            测试
                          </Button>
                        </Box>
                        <Box sx={{ mt: 0.5, minHeight: 4 }}>
                          <TestBadge r={tests[k]} busy={busy[k]} />
                        </Box>
                      </Box>
                    ))}
                  </Box>
                </Section>

                <Box
                  role="button"
                  onClick={() => setAdvanced(!advanced)}
                  sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'pointer', color: brand.blue, fontSize: 13, mb: 1 }}
                >
                  <ExpandMoreIcon sx={{ fontSize: 18, transform: advanced ? 'none' : 'rotate(-90deg)', transition: 'transform .15s' }} />
                  按任务分档（高级）
                </Box>
                <Collapse in={advanced}>
                  <Typography sx={{ fontSize: 12, color: brand.muted, mb: 1.5 }}>
                    每个任务可以单独指定模型、地址和 key；留空就继承上面的主模型或快模型。
                  </Typography>
                  {TIERS.map((t) => {
                    const up = t.key.toUpperCase()
                    const eff = view.tiers[t.key]
                    return (
                      <Box key={t.key} sx={{ ...card.plain, borderRadius: '8px', p: 2, mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'baseline', mb: 1.75 }}>
                          <Typography sx={{ fontSize: 13.5, fontWeight: 600 }}>{t.label}</Typography>
                          <Typography noWrap title={t.desc} sx={{ fontSize: 12, color: brand.muted, ml: 1, flex: 1, minWidth: 0 }}>{t.desc}</Typography>
                          <Typography sx={{ fontSize: 11.5, color: brand.faint, fontFamily: 'Roboto Mono', ml: 1.5, flexShrink: 0, whiteSpace: 'nowrap' }}>当前：{eff.model}</Typography>
                        </Box>
                        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 110px', columnGap: 1.25, rowGap: 2 }}>
                          <TextField label="模型" placeholder={`继承${t.inherit === 'MODEL_ID' ? '主模型' : '快模型'}`} value={val(`MODEL_${up}`)} onChange={(e) => set(`MODEL_${up}`, e.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
                          <TextField label="超时（秒）" placeholder={String(eff.timeout)} value={val(`MODEL_${up}_TIMEOUT`)} onChange={(e) => set(`MODEL_${up}_TIMEOUT`, e.target.value.replace(/\D/g, ''))} slotProps={{ inputLabel: { shrink: true } }} />
                          <TextField label="接口地址" placeholder="继承上面的接口地址" value={val(`MODEL_${up}_API_BASE`)} onChange={(e) => set(`MODEL_${up}_API_BASE`, e.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
                          <Box />
                          <SecretField label="API Key（可选）" name={`MODEL_${up}_API_KEY`} view={view} draft={draft} set={set} />
                          <Box />
                          <TextField
                            label="备用模型"
                            placeholder="主模型失败时依次尝试，逗号分隔"
                            value={val(`MODEL_${up}_FALLBACKS`)}
                            onChange={(e) => set(`MODEL_${up}_FALLBACKS`, e.target.value)}
                            slotProps={{ inputLabel: { shrink: true } }}
                          />
                          <Button
                            variant="outlined"
                            onClick={() =>
                              runTest(t.key, () =>
                                api.testLLM({
                                  tier: t.key,
                                  model: val(`MODEL_${up}`) || val(t.inherit) || undefined,
                                  api_base: `MODEL_${up}_API_BASE` in draft ? val(`MODEL_${up}_API_BASE`) : undefined,
                                  api_key: draftOrUndef(`MODEL_${up}_API_KEY`) ?? draftOrUndef('LLM_API_KEY'),
                                }),
                              )
                            }
                            disabled={busy[t.key]}
                            sx={{ height: 37 }}
                          >
                            测试
                          </Button>
                        </Box>
                        <Box sx={{ mt: 0.75 }}>
                          <TestBadge r={tests[t.key]} busy={busy[t.key]} />
                        </Box>
                      </Box>
                    )
                  })}
                </Collapse>
              </>
            ) : settingsTab === 'search' ? (
              <>
                <Section title="默认检索来源" desc="综述 Agent 和「在线检索」默认使用的来源。Semantic Scholar 被限流时会自动改用 OpenAlex 补位。">
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    {[
                      ['semantic_scholar', 'Semantic Scholar'],
                      ['arxiv', 'arXiv'],
                      ['openalex', 'OpenAlex'],
                    ].map(([k, l]) => (
                      <FormControlLabel key={k} control={<Checkbox size="small" checked={sources.includes(k)} onChange={(e) => toggleSource(k, e.target.checked)} />} label={<Typography sx={{ fontSize: 13 }}>{l}</Typography>} />
                    ))}
                  </Box>
                </Section>
                {(
                  [
                    ['semantic_scholar', 'Semantic Scholar', 'SEMANTIC_SCHOLAR_API_KEY', '不填 key 时与所有匿名用户共用配额，高峰期经常被限流（429）。', 'https://www.semanticscholar.org/product/api'],
                    ['openalex', 'OpenAlex', 'OPENALEX_API_KEY', '不填 key 每天约 100 次检索，免费 key 约 1000 次。', 'https://developers.openalex.org/api-reference/authentication'],
                    ['arxiv', 'arXiv', null, '不需要 key，官方建议每 3 秒最多请求一次，程序已自动限速。', null],
                  ] as const
                ).map(([src, name, keyName, note, link]) => (
                  <Section
                    key={src}
                    title={name}
                    desc={
                      <>
                        {note}
                        {link && (
                          <>
                            {' '}
                            <Link href={link} target="_blank" rel="noreferrer">
                              申请 key
                            </Link>
                          </>
                        )}
                      </>
                    }
                    right={
                      <Tooltip title={dirty ? '测试使用已保存的设置，请先保存' : ''}>
                        <span>
                          <Button variant="outlined" onClick={() => runTest(src, () => api.testSearch(src))} disabled={busy[src] || dirty}>
                            测试
                          </Button>
                        </span>
                      </Tooltip>
                    }
                  >
                    {keyName && <SecretField label="API Key" name={keyName} view={view} draft={draft} set={set} />}
                    <Box sx={{ mt: 0.75 }}>
                      <TestBadge r={tests[src]} busy={busy[src]} />
                      {tests[src]?.ok && keyName && !tests[src]?.fake && (
                        <Typography sx={{ fontSize: 11.5, color: brand.muted }}>{tests[src]?.authenticated ? '已使用你的 key' : '未使用 key（匿名额度）'}</Typography>
                      )}
                    </Box>
                  </Section>
                ))}
                <Section title="检索结果缓存" desc="同一个检索式在这段时间内直接用本地结果，不重复请求。0 表示不缓存。">
                  <TextField
                    sx={{ width: 160 }}
                    label="缓存天数"
                    value={val('SEARCH_CACHE_DAYS')}
                    placeholder="7"
                    onChange={(e) => set('SEARCH_CACHE_DAYS', e.target.value.replace(/\D/g, ''))}
                    slotProps={{ inputLabel: { shrink: true } }}
                  />
                </Section>
              </>
            ) : settingsTab === 'parse' ? (
              <>
                <Section
                  title="全文解析器"
                  desc="上传 PDF 或下载开放获取全文时，用哪个工具把 PDF 解析成段落（带章节和页码）。外部服务不可用时会自动退回 PyMuPDF，上传不会失败；更换后，已入库的文献在文献库里「重新解析」即可生效。"
                >
                  <RadioGroup value={val('PDF_PARSER') || 'pymupdf'} onChange={(e) => set('PDF_PARSER', e.target.value)}>
                    {(
                      [
                        ['pymupdf', 'PyMuPDF（默认）', '零依赖、速度快，普通的英文论文足够；双栏会先读左栏再读右栏。'],
                        ['grobid', 'GROBID', '专门解析英文学术论文：章节、段落、公式结构更准，不会把页眉、脚注、图表说明混进正文。CPU 即可，约 4GB 内存。'],
                        ['mineru', 'MinerU', '擅长中文论文、扫描件（OCR）、复杂版式、公式和表格。需要 NVIDIA GPU（显存 4GB 以上）。'],
                      ] as const
                    ).map(([v, l, d]) => (
                      <FormControlLabel
                        key={v}
                        value={v}
                        control={<Radio size="small" sx={{ alignSelf: 'flex-start', mt: '-2px' }} />}
                        sx={{ alignItems: 'flex-start', mb: 1 }}
                        label={
                          <Box>
                            <Typography sx={{ fontSize: 13, fontWeight: 500 }}>{l}</Typography>
                            <Typography sx={{ fontSize: 12, color: brand.muted, lineHeight: 1.6 }}>{d}</Typography>
                          </Box>
                        }
                      />
                    ))}
                  </RadioGroup>
                </Section>
                {(val('PDF_PARSER') === 'grobid' || val('PDF_PARSER') === 'mineru') && (
                  <Section
                    title={val('PDF_PARSER') === 'grobid' ? 'GROBID 服务' : 'MinerU 服务'}
                    desc={
                      val('PDF_PARSER') === 'grobid' ? (
                        <>
                          本机启动：<code>docker run --rm --init -p 8070:8070 grobid/grobid:0.9.1-crf</code>；Docker 部署用{' '}
                          <code>docker compose --profile grobid up -d</code>，地址填 <code>http://grobid:8070</code>。
                        </>
                      ) : (
                        <>
                          本机启动：<code>pip install "mineru[core]"</code> 后运行 <code>mineru-api --host 0.0.0.0 --port 8001</code>；Docker 部署用{' '}
                          <code>docker compose --profile mineru up -d</code>，地址填 <code>http://mineru:8000</code>。兼容 MinerU 2.x 与 4.x 的接口。
                        </>
                      )
                    }
                    right={
                      <Tooltip title={dirty ? '测试使用已保存的设置，请先保存' : ''}>
                        <span>
                          <Button
                            variant="outlined"
                            disabled={busy.parser || dirty}
                            onClick={() => runTest('parser', () => api.testParser(val('PDF_PARSER') as 'grobid' | 'mineru'))}
                          >
                            测试
                          </Button>
                        </span>
                      </Tooltip>
                    }
                  >
                    {val('PDF_PARSER') === 'grobid' ? (
                      <TextField fullWidth label="服务地址" placeholder="http://localhost:8070" value={val('GROBID_URL')} onChange={(e) => set('GROBID_URL', e.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
                    ) : (
                      <Box sx={{ display: 'flex', gap: 1.5 }}>
                        <TextField sx={{ flex: 1 }} label="服务地址" placeholder="http://localhost:8001" value={val('MINERU_URL')} onChange={(e) => set('MINERU_URL', e.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
                        <TextField
                          select
                          sx={{ width: 190 }}
                          label="解析后端（2.x）"
                          value={val('MINERU_BACKEND') || 'pipeline'}
                          onChange={(e) => set('MINERU_BACKEND', e.target.value)}
                          slotProps={{ inputLabel: { shrink: true }, select: { native: true } }}
                        >
                          <option value="pipeline">pipeline（最稳）</option>
                          <option value="hybrid-auto-engine">hybrid（更准）</option>
                          <option value="vlm-auto-engine">vlm（显存 8GB+）</option>
                        </TextField>
                      </Box>
                    )}
                    <Box sx={{ mt: 0.75 }}>
                      <TestBadge r={tests.parser} busy={busy.parser} />
                    </Box>
                  </Section>
                )}
              </>
            ) : settingsTab === 'embedding' ? (
              <>
                <Section
                  title="向量检索（可选）"
                  desc="默认只用关键词检索（BM25），零配置就能用。中文写作、英文文献时，开启多语言向量模型（如 bge-m3）能明显提高找引用的准确率。"
                >
                  <RadioGroup row value={val('EMBEDDING_PROVIDER') || 'none'} onChange={(e) => set('EMBEDDING_PROVIDER', e.target.value)}>
                    <FormControlLabel value="none" control={<Radio size="small" />} label={<Typography sx={{ fontSize: 13 }}>不启用</Typography>} />
                    <FormControlLabel value="ollama" control={<Radio size="small" />} label={<Typography sx={{ fontSize: 13 }}>Ollama 本地</Typography>} />
                    <FormControlLabel value="openai" control={<Radio size="small" />} label={<Typography sx={{ fontSize: 13 }}>OpenAI 兼容接口</Typography>} />
                  </RadioGroup>
                </Section>
                {(val('EMBEDDING_PROVIDER') || 'none') !== 'none' && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                    <TextField label="向量模型" placeholder="bge-m3" value={val('EMBEDDING_MODEL')} onChange={(e) => set('EMBEDDING_MODEL', e.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
                    <TextField
                      label="接口地址"
                      placeholder={val('EMBEDDING_PROVIDER') === 'ollama' ? 'http://localhost:11434' : '留空则使用模型的接口地址'}
                      value={val('EMBEDDING_BASE_URL')}
                      onChange={(e) => set('EMBEDDING_BASE_URL', e.target.value)}
                      slotProps={{ inputLabel: { shrink: true } }}
                    />
                    {val('EMBEDDING_PROVIDER') === 'openai' && <SecretField label="API Key（留空则使用模型的 key）" name="EMBEDDING_API_KEY" view={view} draft={draft} set={set} />}
                    <Typography sx={{ fontSize: 12, color: brand.amber }}>
                      注意：启用或更换向量模型后，之前入库的文献需要在文献库里重新上传或下载全文，才会生成新的向量。
                    </Typography>
                  </Box>
                )}
              </>
            ) : (
              <>
                <UsagePanel tiers={TIERS} priceRaw={val('MODEL_PRICES')} onPriceChange={(v) => set('MODEL_PRICES', v)} reloadKey={view} />
              </>
            )}
          </Box>

          {/* 底部 */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 3.5, py: 1.5, borderTop: `1px solid ${brand.lineSoft}` }}>
            {settingsTab === 'appearance' ? (
              <>
                <Typography noWrap sx={{ fontSize: 11.5, color: brand.faint, flex: 1, minWidth: 0 }}>
                  外观保存在这台设备上（每个浏览器各自一套），改动立即生效
                </Typography>
                <Button onClick={resetAppearance}>恢复默认</Button>
                <Button variant="contained" onClick={closeSettings}>完成</Button>
              </>
            ) : (<>
            {view?.locked ? (
              <Typography noWrap sx={{ fontSize: 11.5, color: brand.faint, flex: 1, minWidth: 0 }}>
                管理员已锁定设置：这里只能查看，修改请联系部署者
              </Typography>
            ) : (
              <Typography noWrap title={view?.settings_file} sx={{ fontSize: 11.5, color: brand.faint, flex: 1, minWidth: 0 }}>
                保存在本机 {view?.settings_file ?? 'settings.json'}，优先于 .env，保存后立即生效
              </Typography>
            )}
            <Button onClick={closeSettings}>{dirty ? '放弃修改' : '关闭'}</Button>
            <Button variant="contained" onClick={save} disabled={!dirty || saving || !!view?.locked} startIcon={saving ? <CircularProgress size={14} color="inherit" /> : undefined}>
              保存
            </Button>
            </>)}
          </Box>
        </Box>
      </Box>
    </Dialog>
  )
}
