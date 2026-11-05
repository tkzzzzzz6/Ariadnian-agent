/**
 * 外观设置：正文字号 / 行距 / 字体 / 版心宽度 / 段首缩进、写作辅助的显示方式、原文阅读的默认值。
 * 界面缩放不在这里：桌面版用 Electron 的页面缩放（lib/desktopZoom.ts），网页版用浏览器自带的 Ctrl + = / −。
 * （试过 CSS zoom：MUI 的菜单、提示、弹出卡片在缩放后的页面上定位会偏，所以不用。）
 *
 * 属于个人习惯，存在当前浏览器（桌面版是本机）的 localStorage，不写进后端设置；改了立即生效。
 * 正文样式通过 <html> 上的 CSS 变量和 class 生效（index.css），只影响屏幕显示，不影响 Word / LaTeX 导出。
 */
import { useSyncExternalStore } from 'react'

export type LineHeight = 'compact' | 'normal' | 'loose'
export type PaperFont = 'sans' | 'serif'
export type PaperWidth = 'narrow' | 'normal' | 'wide'
export type MarginMode = 'compact' | 'expanded'

export interface Appearance {
  fontSize: number
  lineHeight: LineHeight
  fontFamily: PaperFont
  width: PaperWidth
  indent: boolean
  ghostDelay: number
  marginMode: MarginMode
  focusMode: boolean
  runStats: boolean
  pdfZoom: 'fit' | 1
  splitRatio: 'even' | 'editor'
}

export const DEFAULT_APPEARANCE: Appearance = {
  fontSize: 16,
  lineHeight: 'normal',
  fontFamily: 'sans',
  width: 'normal',
  indent: false,
  ghostDelay: 700,
  marginMode: 'compact',
  focusMode: false,
  runStats: true,
  pdfZoom: 'fit',
  splitRatio: 'even',
}

export const FONT_SIZE = { min: 14, max: 20 }
export const GHOST_DELAYS = [500, 700, 1200]
export const LINE_HEIGHT: Record<LineHeight, { label: string; value: number }> = {
  compact: { label: '紧凑', value: 1.6 },
  normal: { label: '标准', value: 1.9 },
  loose: { label: '宽松', value: 2.2 },
}
export const PAPER_WIDTH: Record<PaperWidth, { label: string; px: number }> = {
  narrow: { label: '窄', px: 640 },
  normal: { label: '标准', px: 760 },
  wide: { label: '宽', px: 900 },
}
export const PAPER_FONT: Record<PaperFont, { label: string; desc: string; css: string }> = {
  sans: {
    label: '无衬线',
    desc: '黑体风格，屏幕上清晰',
    css: "Roboto, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', Arial, sans-serif",
  },
  serif: {
    label: '衬线',
    desc: '宋体 + Times，接近打印稿',
    css: "'Times New Roman', Times, 'Songti SC', STSong, SimSun, 'Noto Serif CJK SC', 'Source Han Serif SC', serif",
  },
}

const KEY = 'aria.appearance'

function clamp(a: Partial<Appearance>): Appearance {
  const d = DEFAULT_APPEARANCE
  const pick = <K extends keyof Appearance>(k: K, ok: (v: unknown) => boolean): Appearance[K] => (ok(a[k]) ? (a[k] as Appearance[K]) : d[k])
  return {
    fontSize: typeof a.fontSize === 'number' ? Math.round(Math.min(FONT_SIZE.max, Math.max(FONT_SIZE.min, a.fontSize))) : d.fontSize,
    lineHeight: pick('lineHeight', (v) => typeof v === 'string' && v in LINE_HEIGHT),
    fontFamily: pick('fontFamily', (v) => typeof v === 'string' && v in PAPER_FONT),
    width: pick('width', (v) => typeof v === 'string' && v in PAPER_WIDTH),
    indent: pick('indent', (v) => typeof v === 'boolean'),
    ghostDelay: pick('ghostDelay', (v) => GHOST_DELAYS.includes(v as number)),
    marginMode: pick('marginMode', (v) => v === 'compact' || v === 'expanded'),
    focusMode: pick('focusMode', (v) => typeof v === 'boolean'),
    runStats: pick('runStats', (v) => typeof v === 'boolean'),
    pdfZoom: pick('pdfZoom', (v) => v === 'fit' || v === 1),
    splitRatio: pick('splitRatio', (v) => v === 'even' || v === 'editor'),
  }
}

function load(): Appearance {
  try {
    return clamp(JSON.parse(localStorage.getItem(KEY) || '{}'))
  } catch {
    return DEFAULT_APPEARANCE
  }
}

let state: Appearance = load()
const listeners = new Set<() => void>()

/** 把设置写到 <html>：CSS 变量控制正文样式，class 控制专注模式 */
export function applyAppearance(a: Appearance = state) {
  const root = document.documentElement
  root.style.setProperty('--paper-size', `${a.fontSize}px`)
  root.style.setProperty('--paper-line', String(LINE_HEIGHT[a.lineHeight].value))
  root.style.setProperty('--paper-font', PAPER_FONT[a.fontFamily].css)
  root.style.setProperty('--paper-indent', a.indent ? '2em' : '0')
  root.classList.toggle('focus-mode', a.focusMode)
}

export function setAppearance(patch: Partial<Appearance>) {
  state = clamp({ ...state, ...patch })
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    /* 隐私模式等存不了时只在本次生效 */
  }
  applyAppearance(state)
  listeners.forEach((l) => l())
}

export const resetAppearance = () => setAppearance(DEFAULT_APPEARANCE)
export const getAppearance = () => state

export function useAppearance(): Appearance {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    () => state,
  )
}

/** Ctrl + ] / [ 调正文字号（和 Word 一样），0 表示恢复默认；返回新字号，已经到头时返回 null */
export function stepFontSize(dir: 1 | -1 | 0): number | null {
  const next = dir === 0 ? DEFAULT_APPEARANCE.fontSize : state.fontSize + dir
  if (next < FONT_SIZE.min || next > FONT_SIZE.max) return null
  setAppearance({ fontSize: next })
  return next
}

// 其他窗口 / 标签页改了外观，这里跟着变
if (typeof window !== 'undefined')
  window.addEventListener('storage', (e) => {
    if (e.key !== KEY) return
    state = load()
    applyAppearance(state)
    listeners.forEach((l) => l())
  })
