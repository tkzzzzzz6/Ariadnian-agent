import type { JSONContent } from '@tiptap/core'
import { citeStore, citeToken, collectCitationOrder, pageLabel } from './citations'
import { codeLanguage, codeLanguageLabel } from './codeLanguages'
import { HIGHLIGHTS, safeLink } from './formatting'
import { imageFilename } from './richContent'

const mdEsc = (text: string) => text.replace(/[\\`*_{}\[\]<>#$|]/g, '\\$&')

function keysOf(ids: string[]): string[] {
  const papers = citeStore.get().papers
  return ids.map((id) => papers[id]?.bibkey).filter(Boolean) as string[]
}

// ------------------------------------------------------------------ Markdown（pandoc 引用语法）
function mdInline(nodes: JSONContent[] = [], inTable = false): string {
  return nodes
    .map((n) => {
      if (n.type === 'citation') return citeToken((n.attrs?.ids ?? []).filter((id: string) => citeStore.get().papers[id]), n.attrs?.pages)
      if (n.type === 'hardBreak') return '  \n'
      if (n.type === 'inlineMath') {
        let latex = n.attrs?.latex ?? ''
        if (inTable) {
          if (/\\begin\{array\}/.test(latex) && latex.includes('|')) throw new Error('表格内 array 公式的竖线列布局暂不支持 Word 导出，请使用矩阵模板')
          latex = latex.replace(/\\\|/g, '\\Vert ').replace(/\|/g, '\\vert ')
        }
        return `$${latex}$`
      }
      if (n.marks?.some(m => m.type === 'code')) {
        const raw = n.text ?? ''
        const fence = '`'.repeat(Math.max(1, ...(raw.match(/`+/g) ?? []).map(m => m.length + 1)))
        return `${fence} ${inTable ? raw.replace(/\|/g, '\\|') : raw} ${fence}`
      }
      let t = mdEsc(n.text ?? '')
      for (const m of n.marks ?? []) {
        if (m.type === 'bold') t = `**${t}**`
        else if (m.type === 'italic') t = `*${t}*`
        else if (m.type === 'code') t = `\`${t}\``
        else if (m.type === 'strike') t = `~~${t}~~`
        else if (m.type === 'underline') t = `[${t}]{.underline}`
        else if (m.type === 'superscript') t = `^${t.replace(/ /g, '\\ ')}^`
        else if (m.type === 'subscript') t = `~${t.replace(/ /g, '\\ ')}~`
        else if (m.type === 'highlight') {
          const color=HIGHLIGHTS.find(c=>c===m.attrs?.color)||HIGHLIGHTS[0]
          t = `[${t}]{.aria-highlight data-color="${color}"}`
        } else if (m.type === 'link') {
          const href=safeLink(m.attrs?.href||'')
          if(href) t=`[${t}](${href.replace(/[()]/g,c=>c==='('?'%28':'%29')})`
        }
      }
      return t
    })
    .join('')
}

function mdBlock(n: JSONContent, indent = ''): string {
  switch (n.type) {
    case 'heading':
      return `${'#'.repeat(n.attrs?.level ?? 1)} ${mdInline(n.content)}`
    case 'paragraph':
      return indent + mdInline(n.content)
    case 'blockquote':
      return (n.content ?? []).map((c) => '> ' + mdBlock(c)).join('\n>\n')
    case 'bulletList':
    case 'orderedList':
      return (n.content ?? [])
        .map((li, i) => {
          const bullet = n.type === 'bulletList' ? '-' : `${i + 1}.`
          const inner = (li.content ?? []).map((c) => mdBlock(c, indent + '   ')).join('\n')
          return `${indent}${bullet} ${inner.trimStart()}`
        })
        .join('\n')
    case 'codeBlock':
      return (() => {
        const raw = (n.content ?? []).map(c => c.text ?? '').join('')
        const fence = '`'.repeat(Math.max(3, ...(raw.match(/`+/g) ?? []).map(m => m.length + 1)))
        return fence + (codeLanguage(n.attrs?.language) ?? '') + '\n' + raw + '\n' + fence
      })()
    case 'horizontalRule':
      return '---'
    case 'image':
      return `![${mdEsc(n.attrs?.caption || n.attrs?.alt || '')}](${imageFilename(n.attrs ?? {})}){width=${n.attrs?.widthPercent ?? 100}%}`
    case 'blockMath':
      return `$$\n${n.attrs?.latex ?? ''}\n$$`
    case 'table': {
      const rows = n.content ?? []
      const line = (r: JSONContent) => '| ' + (r.content ?? []).map(c => (c.content ?? []).map(p => mdInline(p.content, true)).join(' ')).join(' | ') + ' |'
      const separator = '| ' + (rows[0]?.content ?? []).map(() => '---').join(' | ') + ' |'
      const header = rows[0]?.content?.every(c => c.type === 'tableHeader')
      // Pandoc needs an empty header row to parse a headerless pipe table;
      // its AST omits that empty header, so Word receives only the actual rows.
      const emptyHeader = '| ' + (rows[0]?.content ?? []).map(() => '').join(' | ') + ' |'
      return header ? [line(rows[0]), separator, ...rows.slice(1).map(line)].join('\n') : [emptyHeader, separator, ...rows.map(line)].join('\n')
    }
    default:
      return mdInline(n.content)
  }
}

export function toMarkdown(doc: JSONContent, title: string): string {
  return `---\ntitle: ${JSON.stringify(title)}\nbibliography: refs.bib\n---\n\n` + (doc.content ?? []).map((b) => mdBlock(b)).join('\n\n') + '\n'
}

// ------------------------------------------------------------------ LaTeX
const TEX_ESC: Record<string, string> = {
  '\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', $: '\\$', '#': '\\#', _: '\\_', '{': '\\{', '}': '\\}',
  '~': '\\textasciitilde{}', '^': '\\textasciicircum{}',
}
const esc = (s: string) => s.replace(/[\\&%$#_{}~^]/g, (c) => TEX_ESC[c])

/** \\cite{a,b}；带页码的文献单独写成 \\cite[p.~3]{a} */
function texCite(ids: string[], pages?: Record<string, string>): string {
  const papers = citeStore.get().papers
  const plain = ids.filter((id) => papers[id] && !pages?.[id])
  const located = ids.filter((id) => papers[id] && pages?.[id])
  const parts = located.map((id) => `\\cite[${pageLabel(pages![id]).replace(' ', '~')}]{${papers[id].bibkey}}`)
  if (plain.length) parts.unshift(`\\cite{${keysOf(plain).join(',')}}`)
  return parts.join('')
}

function texInline(nodes: JSONContent[] = []): string {
  return nodes
    .map((n) => {
      if (n.type === 'citation') return texCite(n.attrs?.ids ?? [], n.attrs?.pages)
      if (n.type === 'hardBreak') return '\\\\\n'
      if (n.type === 'inlineMath') return `\\(${n.attrs?.latex ?? ''}\\)`
      let t = esc(n.text ?? '')
      for (const m of n.marks ?? []) {
        if (m.type === 'bold') t = `\\textbf{${t}}`
        else if (m.type === 'italic') t = `\\emph{${t}}`
        else if (m.type === 'code') t = `\\texttt{${t}}`
        else if (m.type === 'underline') t = `\\underline{${t}}`
        else if (m.type === 'strike') t = `\\sout{${t}}`
        else if (m.type === 'superscript') t = `\\textsuperscript{${t}}`
        else if (m.type === 'subscript') t = `\\textsubscript{${t}}`
        else if (m.type === 'highlight') t = `\\colorbox[HTML]{${(HIGHLIGHTS.find(c=>c===m.attrs?.color)||HIGHLIGHTS[0]).slice(1).toUpperCase()}}{${t}}`
        else if (m.type === 'link') { const href=safeLink(m.attrs?.href||''); if(href) t=`\\href{${esc(href)}}{${t}}` }
      }
      return t
    })
    .join('')
}

function texBlock(n: JSONContent): string {
  switch (n.type) {
    case 'heading': {
      const cmd = ['section', 'section', 'subsection', 'subsubsection'][n.attrs?.level ?? 1] ?? 'paragraph'
      return `\\${cmd}{${texInline(n.content)}}`
    }
    case 'paragraph':
      return texInline(n.content)
    case 'blockquote':
      return `\\begin{quote}\n${(n.content ?? []).map(texBlock).join('\n\n')}\n\\end{quote}`
    case 'bulletList':
    case 'orderedList': {
      const env = n.type === 'bulletList' ? 'itemize' : 'enumerate'
      const items = (n.content ?? []).map((li) => '  \\item ' + (li.content ?? []).map(texBlock).join('\n')).join('\n')
      return `\\begin{${env}}\n${items}\n\\end{${env}}`
    }
    case 'codeBlock':
      return `${n.attrs?.language ? `{\\small\\texttt{${esc(codeLanguageLabel(n.attrs.language))}}}\n` : ''}\\begin{verbatim}\n${(n.content ?? []).map((c) => c.text).join('')}\n\\end{verbatim}`
    case 'horizontalRule':
      return '\\medskip\\hrule\\medskip'
    case 'image': {
      const caption = n.attrs?.caption ? `\n\\par{\\small ${esc(n.attrs.caption)}}` : ''
      return `\\begin{center}\n\\includegraphics[width=${(n.attrs?.widthPercent ?? 100) / 100}\\linewidth]{${imageFilename(n.attrs ?? {})}}${caption}\n\\end{center}`
    }
    case 'blockMath':
      return `\\[\n${n.attrs?.latex ?? ''}\n\\]`
    case 'table': {
      const cols = n.content?.[0]?.content?.length ?? 1
      const spec = Array.from({ length: cols }, () => `p{\\dimexpr(\\linewidth-${cols + 1}\\arrayrulewidth)/${cols}-2\\tabcolsep\\relax}`).join('|')
      const rows = (n.content ?? []).map(row => (row.content ?? []).map(cell => {
        const text = (cell.content ?? []).map(p => texInline(p.content)).join('\\par ')
        return cell.type === 'tableHeader' ? `\\textbf{${text}}` : text
      }).join(' & ') + ' \\\\ \\hline').join('\n')
      return `\\begin{longtable}{|${spec}|}\n\\hline\n${rows}\n\\end{longtable}`
    }
    default:
      return texInline(n.content)
  }
}

export function toLatex(doc: JSONContent, title: string): string {
  const body = (doc.content ?? []).map(texBlock).filter(Boolean).join('\n\n')
  const zh = /[一-鿿]/.test(body + title)
  return `\\documentclass[11pt]{${zh ? 'ctexart' : 'article'}}
\\usepackage[margin=2.5cm]{geometry}
\\usepackage{hyperref,xcolor}
\\usepackage[normalem]{ulem}
\\usepackage{graphicx,amsmath,amssymb,longtable,array}
\\usepackage[numbers,sort&compress]{natbib}

\\title{${esc(title)}}
\\author{}
\\date{\\today}

\\begin{document}
\\maketitle

${body}

${citedIds(doc).length ? `\\bibliographystyle{${zh ? 'unsrtnat' : 'plainnat'}}\n\\bibliography{refs}` : ''}
\\end{document}
`
}

export function citedIds(doc: JSONContent): string[] {
  const nodes: JSONContent[] = []
  const walk = (n: JSONContent) => {
    nodes.push(n)
    n.content?.forEach(walk)
  }
  walk(doc)
  return collectCitationOrder({
    descendants: (f: (n: any) => void) => nodes.forEach((n) => f({ type: { name: n.type }, attrs: n.attrs ?? {} })),
  })
}

export function download(filename: string, text: string | Blob, mime = 'text/plain') {
  const blob = typeof text === 'string' ? new Blob([text], { type: `${mime};charset=utf-8` }) : text
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

// ------------------------------------------------------------------ Word（后端 pandoc + citeproc）
export type DocxStyle = 'gbt-numeric' | 'gbt-author-date' | 'apa'

/** 编辑器里的引用样式 → Word 的参考文献格式：顺序编码用 GB/T 7714；作者-年份中文用 GB/T 7714 著者-出版年，英文用 APA */
export function docxStyleFor(citeStyle: string, doc: JSONContent): DocxStyle {
  if (citeStyle === 'numeric') return 'gbt-numeric'
  const text = JSON.stringify(doc)
  const cjk = (text.match(/[\u4e00-\u9fff]/g) ?? []).length
  return cjk > 50 ? 'gbt-author-date' : 'apa'
}

export function imageIds(doc: JSONContent): string[] {
  const ids = new Set<string>()
  const walk = (n: JSONContent) => { if (n.type === 'image') ids.add(n.attrs?.assetId); n.content?.forEach(walk) }
  walk(doc)
  return [...ids]
}

export async function exportBundle(doc: JSONContent, title: string, format: 'md' | 'tex', docId: string): Promise<Blob> {
  return exportRequest('/api/export/bundle', { source: format === 'tex' ? toLatex(doc, title) : toMarkdown(doc, title),
    title, format, doc_id: docId, asset_ids: imageIds(doc), paper_ids: citedIds(doc) })
}

async function exportRequest(path: string, body: unknown): Promise<Blob> {
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Ariadne-Export': 'blob' }, body: JSON.stringify(body) })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try { msg = (await res.json()).detail ?? msg } catch { /* response may not be JSON */ }
    throw new Error(`导出失败：${msg}`)
  }
  const blob = await res.blob()
  if (!blob.size) throw new Error('导出未收到文件，请检查浏览器的下载拦截后重试')
  return blob
}

export async function exportDocx(doc: JSONContent, title: string, style: DocxStyle, docId?: string): Promise<Blob> {
  const res = await fetch('/api/export/docx', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Ariadne-Export': 'blob' },
    body: JSON.stringify({ markdown: toMarkdown(doc, title), title, paper_ids: citedIds(doc), style, doc_id: docId, asset_ids: imageIds(doc) }),
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      msg = (await res.json()).detail ?? msg
    } catch {
      /* ignore */
    }
    throw new Error(`Word 导出失败：${msg}`)
  }
  const blob = await res.blob()
  if (!blob.size) throw new Error('导出未收到文件，请检查浏览器的下载拦截后重试')
  return blob
}
