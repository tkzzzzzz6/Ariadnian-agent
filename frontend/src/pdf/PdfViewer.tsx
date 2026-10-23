/**
 * PDF 阅读器（pdf.js）：逐页画到 canvas，上面叠透明文本层（可选中复制）和高亮层。
 * 只渲染滚动到附近的页面，长论文也不卡；缩放默认「适应宽度」，随抽屉宽度变化。
 */
import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { Box, CircularProgress, IconButton, InputBase, Tooltip, Typography } from '@mui/material'
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import TocIcon from '@mui/icons-material/Toc'
import RemoveIcon from '@mui/icons-material/Remove'
import AddIcon from '@mui/icons-material/Add'
import FitScreenOutlinedIcon from '@mui/icons-material/FitScreenOutlined'
import FileDownloadOutlinedIcon from '@mui/icons-material/FileDownloadOutlined'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
// 用 legacy 构建：带 polyfill，较旧的浏览器（以及 Electron）也能用；正式构建依赖很新的 JS 特性
import { GlobalWorkerOptions, TextLayer, getDocument, type PDFDocumentProxy, type RenderTask } from 'pdfjs-dist/legacy/build/pdf.mjs'
import workerSrc from 'pdfjs-dist/legacy/build/pdf.worker.min.mjs?url'
import type { PdfHighlight } from '../lib/types'
import { brand } from '../theme'
import './pdf.css'

GlobalWorkerOptions.workerSrc = workerSrc

const GAP = 12 // 页与页之间的间距
const PAD = 16 // 左右留白
const HL = 'rgba(255, 213, 0, 0.42)'

interface Size {
  w: number
  h: number
}

/** 一条划线 / 笔记在页面上的位置 */
export interface PdfMark {
  id: string
  page: number
  rects: number[][]
  /** 填充色（rgba） */
  fill: string
}

/** 用户在原文上选中的文字 */
export interface PdfSelection {
  page: number
  /** 归一化矩形（同一行的合并成一个） */
  rects: number[][]
  text: string
}

export interface PdfViewerProps {
  src: string
  /** 打开后跳到这一页（从 1 开始） */
  page?: number | null
  highlight?: PdfHighlight | null
  /** 变化时重新滚动到 page / highlight（同一处依据再点一次也能跳回去） */
  jumpKey?: number | string
  /** 工具栏左侧的额外内容，比如「在线预览」标记 */
  badge?: ReactNode
  /** 滚动到某页的某个位置（y 为 0~1）；key 变化时生效。用于从笔记列表跳转 */
  focus?: { page: number; y?: number; key: string | number } | null
  /** 划线 / 笔记 */
  marks?: PdfMark[]
  activeMark?: string | null
  onMarkClick?: (id: string, at: { x: number; y: number }) => void
  /** 选中文字后显示的浮动菜单 */
  renderSelectionMenu?: (sel: PdfSelection, close: () => void) => ReactNode
  onError?: (msg: string) => void
  onLoaded?: (pages: number) => void
  /** 仅报告浏览位置；调用方不得据此自动认定已读。 */
  onPageChange?: (page: number, pages: number) => void
  /** 显示「目录」按钮（PDF 书签；没有书签时按页列出），宽屏时默认展开 */
  toc?: boolean
}

interface TocItem {
  title: string
  page: number | null
  depth: number
}

/** 读取 PDF 书签（目录），解析出每项对应的页码 */
async function loadOutline(doc: PDFDocumentProxy): Promise<TocItem[]> {
  const out: TocItem[] = []
  const walk = async (items: any[] | null, depth: number) => {
    for (const it of items ?? []) {
      let page: number | null = null
      try {
        const dest = typeof it.dest === 'string' ? await doc.getDestination(it.dest) : it.dest
        if (Array.isArray(dest) && dest[0]) page = (typeof dest[0] === 'number' ? dest[0] : await doc.getPageIndex(dest[0])) + 1
      } catch {
        /* 书签指向的目标解析不了：只显示标题 */
      }
      out.push({ title: String(it.title ?? '').trim(), page, depth })
      if (depth < 2) await walk(it.items, depth + 1)
    }
  }
  try {
    await walk(await doc.getOutline(), 0)
  } catch {
    /* 没有目录 */
  }
  return out.filter((x) => x.title)
}

export function PdfViewer({
  src, page, highlight, jumpKey, badge, onError, onLoaded, onPageChange, focus, marks, activeMark, onMarkClick, renderSelectionMenu, toc,
}: PdfViewerProps) {
  const [showToc, setShowToc] = useState(false)
  const [outline, setOutline] = useState<TocItem[] | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const pageEls = useRef(new Map<number, HTMLDivElement>())
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null)
  const [sizes, setSizes] = useState<Size[]>([])
  const [error, setError] = useState<string | null>(null)
  const [width, setWidth] = useState(0)
  const [zoom, setZoom] = useState<'fit' | number>('fit')
  const [current, setCurrent] = useState(1)
  const [pageInput, setPageInput] = useState('1')

  // ---------------------------------------------------------------- 加载文档
  useEffect(() => {
    let cancelled = false
    setDoc(null)
    setCurrent(1)
    setPageInput('1')
    setSizes([])
    setError(null)
    // 先自己请求文件：后端出错时（下载失败、不是 PDF、地址被拦截）能拿到具体原因
    let task: ReturnType<typeof getDocument> | null = null
    ;(async () => {
      try {
        const res = await fetch(src, { credentials: 'same-origin' })
        if (!res.ok) {
          let detail = `${res.status} ${res.statusText}`
          try {
            detail = (await res.json()).detail ?? detail
          } catch {
            /* 不是 JSON */
          }
          throw new Error(detail)
        }
        const data = new Uint8Array(await res.arrayBuffer())
        if (cancelled) return
        task = getDocument({ data })
        const d = await task.promise
        // 先拿到每一页的尺寸（不渲染），用来排版和计算滚动位置
        const ss: Size[] = []
        for (let i = 1; i <= d.numPages; i++) {
          const vp = (await d.getPage(i)).getViewport({ scale: 1 })
          ss.push({ w: vp.width, h: vp.height })
        }
        if (cancelled) return
        setSizes(ss)
        setDoc(d)
        onLoaded?.(d.numPages)
        if (toc) loadOutline(d).then((o) => !cancelled && setOutline(o))
      } catch (e: any) {
        if (cancelled) return
        const msg = String(e?.message ?? e)
        setError(msg)
        onError?.(msg)
      }
    })()
    return () => {
      cancelled = true
      task?.destroy()
    }
  }, [src]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (doc && sizes.length) onPageChange?.(current, sizes.length)
  }, [doc, current, sizes.length, onPageChange])

  // ---------------------------------------------------------------- 宽度与缩放
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  const maxW = sizes.reduce((m, s) => Math.max(m, s.w), 0) || 612
  const fitScale = width ? Math.max(0.3, (width - PAD * 2) / maxW) : 1
  const scale = zoom === 'fit' ? fitScale : zoom
  const setScale = (s: number) => setZoom(Math.min(4, Math.max(0.3, Math.round(s * 100) / 100)))

  // ---------------------------------------------------------------- 跳转
  const scrollToPage = useCallback(
    (n: number, yFrac = 0, smooth = true) => {
      const el = pageEls.current.get(n)
      const box = scrollRef.current
      if (!el || !box) return
      box.scrollTo({ top: Math.max(0, el.offsetTop - GAP + yFrac * el.offsetHeight - (yFrac ? 90 : 0)), behavior: smooth ? 'smooth' : 'auto' })
    },
    [],
  )

  useEffect(() => {
    if (!doc || !sizes.length) return
    const target = highlight?.page ?? page
    if (!target) return
    const y = highlight?.rects?.length ? Math.min(...highlight.rects.map((r) => r[1])) : 0
    // 等布局（缩放后的高度）生效再滚
    const t = window.setTimeout(() => scrollToPage(Math.min(Math.max(1, target), sizes.length), y), 60)
    return () => window.clearTimeout(t)
  }, [doc, sizes.length, page, highlight, jumpKey, scrollToPage]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!doc || !sizes.length || !focus) return
    const t = window.setTimeout(() => scrollToPage(Math.min(Math.max(1, focus.page), sizes.length), focus.y ?? 0), 60)
    return () => window.clearTimeout(t)
  }, [doc, sizes.length, focus?.key]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------- 选中文字 / 点击划线
  const [sel, setSel] = useState<(PdfSelection & { x: number; y: number }) | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const closeSel = useCallback(() => {
    setSel(null)
    window.getSelection()?.removeAllRanges()
  }, [])
  useEffect(() => setSel(null), [src, scale]) // eslint-disable-line react-hooks/exhaustive-deps

  const onMouseUp = (e: React.MouseEvent) => {
    if (menuRef.current?.contains(e.target as Node)) return
    const box = scrollRef.current
    if (!box) return
    const s = window.getSelection()
    const text = s?.toString().replace(/\s+/g, ' ').replace(/(\w)- (\w)/g, '$1$2').trim() ?? ''
    const pageEl = (e.target as HTMLElement).closest?.('.aria-pdf-page') as HTMLElement | null
    if (!s || s.isCollapsed || text.length < 2) {
      setSel(null)
      // 没有选中文字：看看是不是点在了某条划线上
      if (pageEl && marks?.length && onMarkClick) {
        const pr = pageEl.getBoundingClientRect()
        const n = Number(pageEl.dataset.page)
        const x = (e.clientX - pr.left) / pr.width
        const y = (e.clientY - pr.top) / pr.height
        const hit = marks.find((m) => m.page === n && m.rects.some((r) => x >= r[0] - 0.003 && x <= r[2] + 0.003 && y >= r[1] - 0.003 && y <= r[3] + 0.003))
        if (hit) onMarkClick(hit.id, { x: e.clientX, y: e.clientY })
      }
      return
    }
    if (!renderSelectionMenu) return
    const range = s.getRangeAt(0)
    const startPage = ((range.startContainer as HTMLElement).parentElement ?? (range.startContainer as HTMLElement))?.closest?.('.aria-pdf-page') as HTMLElement | null
    if (!startPage) return
    const pr = startPage.getBoundingClientRect()
    // 只取起始页上的矩形（跨页选择时其余部分不画线，但文字保留）
    const raw = Array.from(range.getClientRects()).filter(
      (r) => r.width > 1 && r.height > 1 && r.top >= pr.top - 2 && r.bottom <= pr.bottom + 2 && r.left >= pr.left - 2 && r.right <= pr.right + 2,
    )
    if (!raw.length) return
    const lines: { top: number; bottom: number; left: number; right: number }[] = []
    for (const r of raw.sort((a, b) => a.top - b.top || a.left - b.left)) {
      const line = lines.find((l) => Math.abs(l.top - r.top) < r.height * 0.5)
      if (line) {
        line.left = Math.min(line.left, r.left)
        line.right = Math.max(line.right, r.right)
        line.bottom = Math.max(line.bottom, r.bottom)
      } else lines.push({ top: r.top, bottom: r.bottom, left: r.left, right: r.right })
    }
    const rects = lines.map((l) => [
      +((l.left - pr.left) / pr.width).toFixed(4), +((l.top - pr.top) / pr.height).toFixed(4),
      +((l.right - pr.left) / pr.width).toFixed(4), +((l.bottom - pr.top) / pr.height).toFixed(4),
    ])
    const last = lines[lines.length - 1]
    const br = box.getBoundingClientRect()
    setSel({
      page: Number(startPage.dataset.page), rects, text,
      // 菜单约 400px 宽：尽量从选区左端开始，放不下就贴着可视区域右边
      x: Math.max(box.scrollLeft + 8, Math.min(last.left - br.left + box.scrollLeft, box.scrollLeft + box.clientWidth - 410)),
      y: last.bottom - br.top + box.scrollTop + 8,
    })
  }

  // 缩放时保持当前页在视野里
  const prevScale = useRef(scale)
  useLayoutEffect(() => {
    const box = scrollRef.current
    if (box && prevScale.current !== scale && prevScale.current > 0) {
      box.scrollTop = box.scrollTop * (scale / prevScale.current)
    }
    prevScale.current = scale
  }, [scale])

  const onScroll = () => {
    const box = scrollRef.current
    if (!box) return
    const mid = box.scrollTop + box.clientHeight / 3
    let n = 1
    for (const [k, el] of pageEls.current) if (el.offsetTop <= mid && k > n) n = k
    if (n !== current) {
      setCurrent(n)
      setPageInput(String(n))
    }
  }

  const go = (n: number) => {
    const t = Math.min(Math.max(1, n), sizes.length || 1)
    setCurrent(t)
    setPageInput(String(t))
    scrollToPage(t)
  }

  return (
    <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', bgcolor: '#ececec' }}>
      {/* 工具栏 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25, px: 0.75, height: 38, flexShrink: 0, bgcolor: '#fafafa', borderBottom: `1px solid ${brand.line}` }}>
        {toc && (
          <Tooltip title={showToc ? '收起目录' : '目录'}>
            <IconButton aria-label="目录" onClick={() => setShowToc(!showToc)} sx={{ color: showToc ? brand.blue : undefined }}>
              <TocIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
        <IconButton aria-label="上一页" onClick={() => go(current - 1)} disabled={current <= 1}>
          <NavigateBeforeIcon fontSize="small" />
        </IconButton>
        <Box sx={{ display: 'flex', alignItems: 'center', border: `1px solid ${brand.line}`, borderRadius: '4px', bgcolor: '#fff', px: 0.75, height: 24 }}>
          <InputBase
            value={pageInput}
            onChange={(e) => setPageInput(e.target.value.replace(/\D/g, ''))}
            onKeyDown={(e) => e.key === 'Enter' && go(Number(pageInput) || 1)}
            onBlur={() => setPageInput(String(current))}
            inputProps={{ 'aria-label': '页码', style: { width: `${Math.max(1, pageInput.length)}ch`, textAlign: 'right', padding: 0, fontSize: 12 } }}
          />
          <Typography sx={{ fontSize: 12, color: brand.muted, ml: 0.5 }}>/ {sizes.length || '–'}</Typography>
        </Box>
        <IconButton aria-label="下一页" onClick={() => go(current + 1)} disabled={!sizes.length || current >= sizes.length}>
          <NavigateNextIcon fontSize="small" />
        </IconButton>
        <Box sx={{ ml: 1, minWidth: 0, overflow: 'hidden' }}>{badge}</Box>
        <Box sx={{ flex: 1 }} />
        <IconButton aria-label="缩小" onClick={() => setScale(scale / 1.15)}>
          <RemoveIcon fontSize="small" />
        </IconButton>
        <Typography sx={{ fontSize: 12, width: 40, textAlign: 'center', color: brand.ink }}>{Math.round(scale * 100)}%</Typography>
        <IconButton aria-label="放大" onClick={() => setScale(scale * 1.15)}>
          <AddIcon fontSize="small" />
        </IconButton>
        <Tooltip title="适应宽度">
          <IconButton aria-label="适应宽度" onClick={() => setZoom('fit')} sx={{ color: zoom === 'fit' ? brand.blue : undefined }}>
            <FitScreenOutlinedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="下载 PDF">
          <IconButton aria-label="下载 PDF" component="a" href={src} download>
            <FileDownloadOutlinedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="在新标签页打开">
          <IconButton aria-label="在新标签页打开" component="a" href={`${src}#page=${current}`} target="_blank" rel="noreferrer">
            <OpenInNewIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </Tooltip>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, display: 'flex' }}>
      {/* 目录：PDF 书签，没有书签时按页列出 */}
      {toc && showToc && (
        <Box sx={{ width: 190, flexShrink: 0, overflow: 'auto', bgcolor: '#fafafa', borderRight: `1px solid ${brand.line}`, py: 0.75 }}>
          {(outline?.length ? outline : sizes.map((_, i) => ({ title: `第 ${i + 1} 页`, page: i + 1, depth: 0 }))).map((it, i) => (
            <Box
              key={i}
              role="button"
              onClick={() => it.page && go(it.page)}
              sx={{
                pl: 1.25 + it.depth * 1.25, pr: 1, py: 0.4, fontSize: it.depth ? 11.5 : 12, cursor: it.page ? 'pointer' : 'default', lineHeight: 1.45,
                color: it.page && it.page <= current && (outline?.[i + 1]?.page ?? Infinity) > current ? brand.blue : it.depth ? brand.muted : brand.ink,
                fontWeight: it.depth ? 400 : 500, '&:hover': { bgcolor: 'rgba(0,0,0,0.05)' },
              }}
            >
              {it.title}
            </Box>
          ))}
        </Box>
      )}
      {/* 页面 */}
      <Box
        ref={scrollRef}
        onScroll={onScroll}
        onMouseUp={onMouseUp}
        onMouseDown={(e) => {
          if (sel && !menuRef.current?.contains(e.target as Node)) setSel(null)
        }}
        sx={{ flex: 1, minHeight: 0, overflow: 'auto', position: 'relative' }}
      >
        {error ? (
          <Typography sx={{ p: 3, fontSize: 13, color: brand.muted }}>{error}</Typography>
        ) : !doc ? (
          <Box sx={{ p: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, color: brand.muted, fontSize: 13 }}>
            <CircularProgress size={16} /> 正在打开 PDF…
          </Box>
        ) : (
          <Box sx={{ py: `${GAP}px`, px: `${PAD}px`, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: `${GAP}px`, minWidth: 'fit-content' }}>
            {sizes.map((s, i) => (
              <PageView
                key={i}
                doc={doc}
                num={i + 1}
                size={s}
                scale={scale}
                root={scrollRef}
                rects={highlight && highlight.page === i + 1 ? highlight.rects : null}
                flashKey={jumpKey}
                marks={marks?.filter((m) => m.page === i + 1)}
                activeMark={activeMark}
                register={(el) => (el ? pageEls.current.set(i + 1, el) : pageEls.current.delete(i + 1))}
              />
            ))}
          </Box>
        )}
        {sel && renderSelectionMenu && (
          <Box ref={menuRef} onMouseDown={(e) => !(e.target as HTMLElement).closest('input, textarea') && e.preventDefault()} sx={{ position: 'absolute', left: sel.x, top: sel.y, zIndex: 5 }}>
            {renderSelectionMenu(sel, closeSel)}
          </Box>
        )}
      </Box>
      </Box>
    </Box>
  )
}

// ------------------------------------------------------------------ 单页
const PageView = memo(function PageView({
  doc, num, size, scale, root, rects, flashKey, register, marks, activeMark,
}: {
  marks?: PdfMark[]
  activeMark?: string | null
  doc: PDFDocumentProxy
  num: number
  size: Size
  scale: number
  root: React.RefObject<HTMLDivElement | null>
  rects: number[][] | null
  flashKey?: number | string
  register: (el: HTMLDivElement | null) => void
}) {
  const boxRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const textRef = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [rendered, setRendered] = useState(false)

  // 滚动到附近（上下 1.5 屏）才渲染
  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { root: root.current, rootMargin: '150% 0px' })
    io.observe(el)
    return () => io.disconnect()
  }, [root])

  useEffect(() => {
    if (!visible) return
    let task: RenderTask | null = null
    let text: TextLayer | null = null
    let cancelled = false
    ;(async () => {
      const page = await doc.getPage(num)
      if (cancelled) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2.5)
      const vp = page.getViewport({ scale })
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = Math.floor(vp.width * dpr)
      canvas.height = Math.floor(vp.height * dpr)
      task = page.render({ canvas, viewport: vp, transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined })
      try {
        await task.promise
      } catch {
        return // 被取消（缩放、滚走）
      }
      if (cancelled) return
      setRendered(true)
      const tl = textRef.current
      if (!tl) return
      tl.replaceChildren()
      text = new TextLayer({ textContentSource: page.streamTextContent(), container: tl, viewport: vp })
      await text.render().catch(() => undefined)
    })()
    return () => {
      cancelled = true
      task?.cancel()
      text?.cancel()
    }
  }, [visible, doc, num, scale])

  const w = Math.floor(size.w * scale)
  const h = Math.floor(size.h * scale)
  return (
    <Box
      ref={(el: HTMLDivElement | null) => {
        boxRef.current = el
        register(el)
      }}
      className="aria-pdf-page"
      data-page={num}
      style={{ ['--total-scale-factor' as any]: scale }}
      sx={{ position: 'relative', width: w, height: h, bgcolor: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.18)', flexShrink: 0 }}
    >
      <canvas ref={canvasRef} style={{ width: w, height: h, display: 'block' }} />
      {!rendered && (
        <Typography sx={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: brand.faint }}>
          第 {num} 页
        </Typography>
      )}
      {marks?.flatMap((m) =>
        m.rects.map((r, i) => (
          <Box
            key={`${m.id}-${i}`}
            sx={{
              position: 'absolute', zIndex: 1, pointerEvents: 'none', borderRadius: '2px', bgcolor: m.fill, mixBlendMode: 'multiply',
              left: `${r[0] * 100}%`, top: `${r[1] * 100}%`, width: `${(r[2] - r[0]) * 100}%`, height: `${(r[3] - r[1]) * 100}%`,
              outline: activeMark === m.id ? '2px solid rgba(0,120,212,0.6)' : 'none', outlineOffset: 1,
            }}
          />
        )),
      )}
      {rects?.map((r, i) => (
        <Box
          key={`${flashKey}-${i}`}
          sx={{
            position: 'absolute', zIndex: 1, pointerEvents: 'none', borderRadius: '2px', bgcolor: HL, mixBlendMode: 'multiply',
            left: `${r[0] * 100}%`, top: `${r[1] * 100}%`, width: `${(r[2] - r[0]) * 100}%`, height: `${(r[3] - r[1]) * 100}%`,
            animation: 'aria-pdf-flash 1.2s ease-out',
          }}
        />
      ))}
      {rects && rects.length > 0 && (
        <Box
          sx={{
            position: 'absolute', left: -9, width: 4, borderRadius: 2, bgcolor: '#f5b400', zIndex: 1,
            top: `${Math.min(...rects.map((r) => r[1])) * 100}%`,
            height: `${(Math.max(...rects.map((r) => r[3])) - Math.min(...rects.map((r) => r[1]))) * 100}%`,
          }}
        />
      )}
      <div ref={textRef} className="textLayer" />
    </Box>
  )
})
