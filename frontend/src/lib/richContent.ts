import type { JSONContent } from '@tiptap/core'
import type { Node as PMNode } from '@tiptap/pm/model'
import { BLOCK_ITEMS, type BlockKind } from './formatting'
import { TextSelection, type EditorState } from '@tiptap/pm/state'
import { closeHistory } from '@tiptap/pm/history'

export type InsertKind = 'image' | 'table' | 'inlineMath' | 'blockMath' | BlockKind
export const INSERT_ITEMS: { kind: InsertKind; label: string; keywords: string; description: string }[] = [
  ...BLOCK_ITEMS,
  { kind: 'image', label: '图片', keywords: 'image picture tupian', description: '上传图片或粘贴截图' },
  { kind: 'table', label: '表格', keywords: 'table biaoge', description: '插入规则行列表格' },
  { kind: 'inlineMath', label: '行内公式', keywords: 'math formula gongshi', description: '公式与文字放在同一行' },
  { kind: 'blockMath', label: '独立公式', keywords: 'math equation formula gongshi', description: '单独一行的公式' },
]
export const RICH_TYPES = new Set(['image', 'table', 'tableCell', 'tableHeader', 'inlineMath', 'blockMath'])
export const unsafeTex = /\\(?:input|include|includegraphics|write\d*|openout|openin|read|catcode|csname|endcsname|def|gdef|edef|xdef|newcommand|renewcommand|usepackage|documentclass|href|url|html\w*|special|directlua|loop|repeat)\b/i

export function imageFilename(attrs: Record<string, any>): string {
  if (!/^[0-9a-f]{32}$/.test(attrs.assetId ?? '') || !['png', 'jpg'].includes(attrs.extension)) throw new Error('图片资源无效，请重新上传')
  return `images/${attrs.assetId}.${attrs.extension}`
}

export function tableJSON(rows: number, cols: number, header = true, values?: string[][]): JSONContent {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || rows > 50 || cols < 1 || cols > 10) throw new Error('表格最多 50 行、10 列')
  return { type: 'table', content: Array.from({ length: rows }, (_, r) => ({ type: 'tableRow', content: Array.from({ length: cols }, (_, c) => ({
    type: header && r === 0 ? 'tableHeader' : 'tableCell', content: [{ type: 'paragraph', content: values?.[r]?.[c] ? [{ type: 'text', text: values[r][c] }] : [] }],
  })) })) }
}

/** Clipboard TSV: reject non-rectangular or oversized grids rather than drop cells. */
export function parseTSV(text: string): string[][] | null {
  if (!text.includes('\t')) return null
  const rows = text.replace(/\r\n?/g, '\n').replace(/\n$/, '').split('\n').map(r => r.split('\t'))
  const cols = rows[0]?.length ?? 0
  if (rows.length > 50 || cols > 10 || rows.some(r => r.length !== cols)) throw new Error('请粘贴不超过 50 行、10 列的规则表格')
  return rows
}

/** Paste a rectangle at the current cell, growing the table within its limits. */
export function pasteTable(state: EditorState, values: string[][]) {
  const { $from } = state.selection
  let depth = $from.depth
  while (depth > 0 && $from.node(depth).type.name !== 'table') depth--
  if (!depth) return null
  const table = $from.node(depth)
  const rowIndex = $from.index(depth)
  const colIndex = $from.index(depth + 1)
  const rows = Math.max(table.childCount, rowIndex + values.length)
  const cols = Math.max(table.firstChild!.childCount, colIndex + values[0].length)
  if (rows > 50 || cols > 10) throw new Error('粘贴后表格将超过 50 行或 10 列，请缩小粘贴范围')
  const cells = []
  for (let r = 0; r < rows; r++) {
    const oldRow = r < table.childCount ? table.child(r) : null
    const row = []
    for (let c = 0; c < cols; c++) {
      const oldCell = oldRow && c < oldRow.childCount ? oldRow.child(c) : null
      const value = values[r - rowIndex]?.[c - colIndex]
      if (value === undefined && oldCell) row.push(oldCell)
      else {
        const cellType = oldCell?.type ?? state.schema.nodes[r === 0 && table.firstChild?.firstChild?.type.name === 'tableHeader' ? 'tableHeader' : 'tableCell']
        row.push(cellType.create(oldCell?.attrs, state.schema.nodes.paragraph.create(null, value ? state.schema.text(value) : null)))
      }
    }
    cells.push(state.schema.nodes.tableRow.create(oldRow?.attrs, row))
  }
  const pos = $from.before(depth)
  const updated = table.type.create(table.attrs, cells)
  const tr = closeHistory(state.tr).replaceWith(pos, pos + table.nodeSize, updated)
  const cellPos = pos + 1 + cells.slice(0, rowIndex).reduce((n, row) => n + row.nodeSize, 0) + 1
    + cells[rowIndex].content.content.slice(0, colIndex).reduce((n, cell) => n + cell.nodeSize, 0) + 2
  return tr.setSelection(TextSelection.near(tr.doc.resolve(cellPos)))
}

export function containsRich(doc: PMNode, from: number, to: number): boolean {
  let found = false
  doc.nodesBetween(from, to, n => { if (RICH_TYPES.has(n.type.name)) found = true; return !found })
  const start = doc.resolve(from)
  for (let depth = start.depth; depth > 0; depth--) if (RICH_TYPES.has(start.node(depth).type.name)) found = true
  return found
}

export function richStructureError(doc: PMNode): string | null {
  let error: string | null = null
  doc.descendants(n => {
    if (n.type.name === 'table') {
      const cols = n.firstChild?.childCount ?? 0
      if (n.childCount > 50 || cols < 1 || cols > 10) error = '表格最多 50 行、10 列'
      n.forEach(row => { if (row.childCount !== cols) error = '暂不支持非规则表格' })
    }
    if (['tableCell', 'tableHeader'].includes(n.type.name) && (n.attrs.colspan !== 1 || n.attrs.rowspan !== 1)) error = '暂不支持合并单元格'
    return !error
  })
  return error
}
