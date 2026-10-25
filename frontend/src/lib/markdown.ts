import type { JSONContent } from '@tiptap/core'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { safeLink } from './formatting'
import { validMathSource } from './mathShortcuts'

const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath)
export interface AnswerSource { n: number; paper_id: string }
export function citationParts(text: string, sources: AnswerSource[] = []): JSONContent[] {
  const out: JSONContent[] = []
  let last = 0
  for (const m of text.matchAll(/\[(\d+(?:\s*[,，]\s*\d+)*)\]/g)) {
    const numbers = m[1].split(/[,，]/).map(Number)
    // Mixed known/unknown markers stay literal; never hide an unsupported reference.
    if (!numbers.every(n => sources.some(s => s.n === n && s.paper_id))) continue
    if (m.index! > last) out.push({type:'text',text:text.slice(last,m.index)})
    out.push({type:'citation',attrs:{ids:[...new Set(numbers.map(n => sources.find(s => s.n === n)!.paper_id))]}})
    last = m.index! + m[0].length
  }
  if (last < text.length) out.push({type:'text',text:text.slice(last)})
  return out
}
export function markdownToContent(text: string, sources: AnswerSource[] = []): JSONContent[] {
  if (text.length > 200000) throw new Error('内容过长，请分段插入')
  const root: any = parser.parse(text)
  const definitions = new Map<string,string>()
  for (const n of root.children) if (n.type === 'definition') definitions.set(n.identifier, n.url)
  const inline = (n: any, marks: any[] = []): JSONContent[] => {
    if (n.type === 'text') return citationParts(n.value, sources).map(x => x.type === 'text' && marks.length ? {...x,marks} : x)
    if (n.type === 'inlineCode') return [{type:'text',text:n.value || ' ',marks:[{type:'code'}]}]
    if (n.type === 'inlineMath') return validMathSource(n.value, false)
      ? [{type:'inlineMath',attrs:{latex:n.value}}] : [{type:'text',text:`$${n.value}$`}]
    if (n.type === 'footnoteReference') return [{type:'text',text:`[注 ${n.label || n.identifier}]`}]
    if (n.type === 'break') return [{type:'hardBreak'}]
    if (n.type === 'image' || n.type === 'imageReference') return [{type:'text',text:`[图片：${n.alt || '请上传本地图片'}]`}]
    const mark = ({strong:'bold',emphasis:'italic',delete:'strike'} as Record<string,string>)[n.type]
    let next = mark ? [...marks,{type:mark}] : marks
    if (n.type === 'link' || n.type === 'linkReference') {
      const href = safeLink(n.url || definitions.get(n.identifier) || '')
      if (href) next = [...marks,{type:'link',attrs:{href,target:'_blank',rel:'noopener noreferrer'}}]
      // Link labels and code are not bibliographic citation markers.
      return (n.children ?? []).flatMap((c: any) => c.type === 'text' ? [{type:'text',text:c.value,marks:next}] : inline(c,next))
    }
    if (n.children) return n.children.flatMap((c: any) => inline(c,next))
    return n.value ? [{type:'text',text:n.value, ...(marks.length ? {marks} : {})}] : []
  }
  const block = (n: any): JSONContent[] => {
    switch(n.type) {
      case 'definition': return []
      case 'footnoteDefinition': return [{type:'paragraph',content:[{type:'text',text:`[注 ${n.label || n.identifier}]`}]},...n.children.flatMap(block)]
      case 'paragraph': {
        const math=n.children.length===1&&n.children[0].type==='inlineMath'?n.children[0]:null
        const raw=math?.position?text.slice(math.position.start.offset,math.position.end.offset):''
        if(math&&raw.startsWith('$$')&&raw.endsWith('$$')&&validMathSource(math.value,true)) return [{type:'blockMath',attrs:{latex:math.value}}]
        return [{type:'paragraph',content:n.children.flatMap((c: any) => inline(c))}]
      }
      case 'heading': return [{type:'heading',attrs:{level:Math.min(3,n.depth)},content:n.children.flatMap((c: any) => inline(c))}]
      case 'blockquote': return [{type:'blockquote',content:n.children.flatMap(block)}]
      case 'list': return [{type:n.ordered?'orderedList':'bulletList',attrs:n.ordered?{start:n.start || 1}:undefined,content:n.children.map((item: any) => {
        const content = item.children.flatMap(block)
        if (item.checked != null && content[0]?.type === 'paragraph') content[0].content.unshift({type:'text',text:item.checked?'☑ ':'☐ '})
        return {type:'listItem',content:content.length?content:[{type:'paragraph'}]}
      })}]
      case 'code': return [{type:'codeBlock',attrs:{language:n.lang || null},content:n.value?[{type:'text',text:n.value}]:[]}]
      case 'math': return validMathSource(n.value, true)
        ? [{type:'blockMath',attrs:{latex:n.value}}] : [{type:'paragraph',content:[{type:'text',text:`$$\n${n.value}\n$$`}]}]
      case 'thematicBreak': return [{type:'horizontalRule'}]
      case 'table': {
        if (n.children.length > 50 || n.children[0]?.children.length > 10) throw new Error('表格最多 50 行、10 列，请分段插入')
        return [{type:'table',content:n.children.map((row: any,i: number) => ({type:'tableRow',content:row.children.map((cell: any) => ({
          type:i===0?'tableHeader':'tableCell',content:[{type:'paragraph',content:cell.children.flatMap((c: any) => inline(c))}]
        }))}))}]
      }
      default: return n.children ? n.children.flatMap(block) : n.value ? [{type:'paragraph',content:[{type:'text',text:n.value}]}] : []
    }
  }
  const result = root.children.flatMap(block)
  return result.length ? result : [{type:'paragraph'}]
}
export function hasMarkdown(text: string): boolean {
  return /(^|\n)\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|```|~~~|\|.*\||\$\$)|\*\*[^*]+\*\*|~~[^~]+~~|`[^`]+`|\[[^\]]+\]\([^\s)]+\)/.test(text)
}
/** Source links are added only to text nodes, never code, math, HTML or real links. */
export function sourceLinks(sources: AnswerSource[]) {
  return () => (tree: any) => {
    const visit = (parent: any) => {
      if (!parent.children || ['link','code','inlineCode','math','inlineMath','html'].includes(parent.type)) return
      parent.children = parent.children.flatMap((n: any) => {
        if (n.type !== 'text') { visit(n); return [n] }
        const parts: any[]=[]
        let last=0
        for(const m of n.value.matchAll(/\[(\d+(?:\s*[,，]\s*\d+)*)\]/g)) {
          const numbers=m[1].split(/[,，]/).map(Number)
          if(!numbers.every((number:number)=>sources.some(s=>s.n===number&&s.paper_id))) continue
          if(m.index>last) parts.push({type:'text',value:n.value.slice(last,m.index)})
          for(const number of [...new Set(numbers)]) parts.push({type:'link',url:`#aria-source-${number}`,children:[{type:'text',value:`[${number}]`}]})
          last=m.index+m[0].length
        }
        if(last<n.value.length) parts.push({type:'text',value:n.value.slice(last)})
        return parts
      })
    }
    visit(tree)
  }
}
