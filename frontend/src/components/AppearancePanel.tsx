/**
 * 设置 → 外观：正文、界面、写作辅助、原文阅读四组。改动立即生效，顶部的预览和背后的编辑器同步变化。
 */
import type { ReactNode } from 'react'
import { Box, Slider, Switch, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { useApp } from '../lib/appContext'
import {
  FONT_SIZE, GHOST_DELAYS, LINE_HEIGHT, PAPER_FONT, PAPER_WIDTH, setAppearance, useAppearance,
  type LineHeight, type MarginMode, type PaperFont, type PaperWidth,
} from '../lib/appearance'
import { DESKTOP_ZOOMS, useDesktopZoom } from '../lib/desktopZoom'
import { brand } from '../theme'

const MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Box sx={{ mb: 3 }}>
      <Typography sx={{ fontSize: 14, fontWeight: 600, mb: 0.5 }}>{title}</Typography>
      <Box sx={{ borderTop: `1px solid ${brand.lineSoft}` }}>{children}</Box>
    </Box>
  )
}

function Row({ label, desc, children, disabled }: { label: string; desc?: ReactNode; children: ReactNode; disabled?: boolean }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 1.25, borderBottom: `1px solid ${brand.lineSoft}`, opacity: disabled ? 0.5 : 1 }}>
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography sx={{ fontSize: 13.5 }}>{label}</Typography>
        {desc && <Typography sx={{ fontSize: 12, color: brand.muted, mt: 0.25 }}>{desc}</Typography>}
      </Box>
      <Box sx={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>{children}</Box>
    </Box>
  )
}

function Choice<T extends string | number>({ value, options, onChange, disabled, label }: {
  value: T
  options: [T, string][]
  onChange: (v: T) => void
  disabled?: boolean
  label: string
}) {
  return (
    <ToggleButtonGroup
      size="small"
      exclusive
      aria-label={label}
      value={value}
      disabled={disabled}
      onChange={(_, v) => v !== null && onChange(v as T)}
      sx={{ '& .MuiToggleButton-root': { px: 1.5, py: 0.4, fontSize: 12.5, textTransform: 'none', lineHeight: 1.6 } }}
    >
      {options.map(([v, l]) => (
        <ToggleButton key={String(v)} value={v}>
          {l}
        </ToggleButton>
      ))}
    </ToggleButtonGroup>
  )
}

/** 一小段示例正文：用和编辑器相同的 class，字号、行距、字体、缩进的变化在这里立刻看得到 */
function Preview() {
  const look = useAppearance()
  return (
    <Box sx={{ display: 'flex', gap: 2, p: 2, mb: 3, borderRadius: '8px', border: `1px solid ${brand.line}`, bgcolor: '#fff', overflow: 'hidden' }}>
      <Box sx={{ flex: 1, minWidth: 0, maxHeight: 230, overflow: 'hidden', maskImage: 'linear-gradient(#000 80%, transparent)' }}>
        <div className="paper-body" style={{ minHeight: 0 }}>
          <h2>2 相关工作</h2>
          <p>
            检索增强生成（Retrieval-Augmented Generation）在生成前先从外部语料中检索相关段落，使模型的回答能够追溯到具体来源
            <span className="cite-chip">[1]</span>。后续工作进一步让模型自己判断何时需要检索，并对检索结果进行反思
            <span className="cite-chip">[2]</span>，<span className="rv-missing">但这些方法在长文档写作中的效果仍缺少系统评估。</span>
          </p>
          <p>本文关注研究报告写作场景：作者需要在几十篇文献之间来回查证，每一处论断都应当有可核对的依据。</p>
        </div>
      </Box>
      <Box sx={{ width: 150, flexShrink: 0, pt: 6, display: { xs: 'none', sm: 'block' } }}>
        <Box sx={{ px: 1, py: 0.75, borderRadius: '6px', border: `1px solid ${brand.lineSoft}`, borderLeft: '3px solid #8764b8', bgcolor: '#fff' }}>
          <Typography sx={{ fontSize: 12, fontWeight: 500, color: '#8764b8' }}>缺少引用</Typography>
          {look.marginMode === 'expanded' && (
            <Typography sx={{ fontSize: 11.5, color: brand.muted, mt: 0.5, lineHeight: 1.5 }}>这句论断需要文献支持，文献库里有 2 篇可能相关。</Typography>
          )}
        </Box>
      </Box>
    </Box>
  )
}

export function AppearancePanel() {
  const look = useAppearance()
  const { copilot, setCopilot, review, setReview } = useApp()
  const zoom = useDesktopZoom()

  return (
    <Box sx={{ maxWidth: 720 }}>
      <Typography sx={{ fontSize: 12, color: brand.muted, mb: 1.5 }}>
        改动立即生效，只影响屏幕上的显示；导出 Word / LaTeX 仍按各自的版式模板排版。
      </Typography>
      <Preview />

      <Group title="正文">
        <Row label="字号" desc={`标题按比例跟着变 · 快捷键 ${MAC ? '⌘' : 'Ctrl'} + ] 放大、${MAC ? '⌘' : 'Ctrl'} + [ 缩小（和 Word 一样）`}>
          <Box sx={{ width: 220, display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Slider
              size="small"
              aria-label="正文字号"
              min={FONT_SIZE.min}
              max={FONT_SIZE.max}
              step={1}
              marks
              value={look.fontSize}
              onChange={(_, v) => setAppearance({ fontSize: v as number })}
            />
            <Typography sx={{ fontSize: 13, width: 36, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{look.fontSize}px</Typography>
          </Box>
        </Row>
        <Row label="行距">
          <Choice<LineHeight>
            label="行距"
            value={look.lineHeight}
            options={(Object.keys(LINE_HEIGHT) as LineHeight[]).map((k) => [k, `${LINE_HEIGHT[k].label} ${LINE_HEIGHT[k].value}`])}
            onChange={(v) => setAppearance({ lineHeight: v })}
          />
        </Row>
        <Row label="字体" desc={PAPER_FONT[look.fontFamily].desc}>
          <Choice<PaperFont>
            label="字体"
            value={look.fontFamily}
            options={(Object.keys(PAPER_FONT) as PaperFont[]).map((k) => [k, PAPER_FONT[k].label])}
            onChange={(v) => setAppearance({ fontFamily: v })}
          />
        </Row>
        <Row label="版心宽度" desc="正文一行的最大宽度；窗口不够宽时会自动收窄">
          <Choice<PaperWidth>
            label="版心宽度"
            value={look.width}
            options={(Object.keys(PAPER_WIDTH) as PaperWidth[]).map((k) => [k, `${PAPER_WIDTH[k].label} ${PAPER_WIDTH[k].px}`])}
            onChange={(v) => setAppearance({ width: v })}
          />
        </Row>
        <Row label="段首缩进两字" desc="中文论文的排版习惯；列表、引用块和表格不缩进">
          <Switch checked={look.indent} onChange={(e) => setAppearance({ indent: e.target.checked })} slotProps={{ input: { 'aria-label': '段首缩进两字' } }} />
        </Row>
      </Group>

      <Group title="界面">
        {zoom.available ? (
          <Row label="界面缩放" desc="整个界面一起放大或缩小（正文也会跟着变）· 快捷键 Ctrl + = / −，Ctrl + 0 恢复 100%">
            <Choice<number>
              label="界面缩放"
              value={DESKTOP_ZOOMS.includes(zoom.zoom) ? zoom.zoom : 1}
              options={DESKTOP_ZOOMS.map((z) => [z, `${Math.round(z * 100)}%`])}
              onChange={zoom.setZoom}
            />
          </Row>
        ) : (
          <Row
            label="界面缩放"
            desc={
              <>
                网页版用浏览器自带的缩放：<b>{MAC ? '⌘' : 'Ctrl'} + =</b> 放大，<b>{MAC ? '⌘' : 'Ctrl'} + −</b> 缩小，<b>{MAC ? '⌘' : 'Ctrl'} + 0</b> 恢复。
                整个界面一起缩放，浏览器会按网址记住这个比例。桌面版可以直接在这里选。
              </>
            }
          >
            {null}
          </Row>
        )}
      </Group>

      <Group title="写作辅助">
        <Row label="行内续写建议" desc="停顿后出现灰色建议，Tab 接受">
          <Switch checked={copilot} onChange={(e) => setCopilot(e.target.checked)} slotProps={{ input: { 'aria-label': '行内续写建议' } }} />
        </Row>
        <Row label="续写建议出现时机" desc="停止输入多久后开始生成" disabled={!copilot}>
          <Choice<number>
            label="续写建议出现时机"
            disabled={!copilot}
            value={look.ghostDelay}
            options={GHOST_DELAYS.map((d) => [d, `${d / 1000} 秒`])}
            onChange={(v) => setAppearance({ ghostDelay: v })}
          />
        </Row>
        <Row label="页边批注" desc="写完一段后自动检查引用和论断">
          <Switch checked={review} onChange={(e) => setReview(e.target.checked)} slotProps={{ input: { 'aria-label': '页边批注' } }} />
        </Row>
        <Row label="批注显示方式" desc={look.marginMode === 'expanded' ? '每条都显示完整内容，依次往下排' : '平时每条一行，点一下或光标进入那句话才展开'} disabled={!review}>
          <Choice<MarginMode>
            label="批注显示方式"
            disabled={!review}
            value={look.marginMode}
            options={[['compact', '一行摘要'], ['expanded', '全部展开']]}
            onChange={(v) => setAppearance({ marginMode: v })}
          />
        </Row>
        <Row label="专注模式" desc="编辑时淡化光标所在段落以外的内容">
          <Switch checked={look.focusMode} onChange={(e) => setAppearance({ focusMode: e.target.checked })} slotProps={{ input: { 'aria-label': '专注模式' } }} />
        </Row>
        <Row label="Agent 运行统计条" desc="Agent 面板底部的步数、速度、token 和上下文占用">
          <Switch checked={look.runStats} onChange={(e) => setAppearance({ runStats: e.target.checked })} slotProps={{ input: { 'aria-label': 'Agent 运行统计条' } }} />
        </Row>
      </Group>

      <Group title="原文阅读">
        <Row label="默认缩放" desc="打开 PDF 时的缩放，之后仍可在阅读器里调">
          <Choice<'fit' | 1>
            label="默认缩放"
            value={look.pdfZoom}
            options={[['fit', '适应宽度'], [1, '100%']]}
            onChange={(v) => setAppearance({ pdfZoom: v })}
          />
        </Row>
        <Row label="分屏默认比例" desc="进入分屏对照、双击分隔条时恢复的比例">
          <Choice<'even' | 'editor'>
            label="分屏默认比例"
            value={look.splitRatio}
            options={[['even', '1 : 1'], ['editor', '写作区更宽 6 : 4']]}
            onChange={(v) => setAppearance({ splitRatio: v })}
          />
        </Row>
      </Group>
    </Box>
  )
}
