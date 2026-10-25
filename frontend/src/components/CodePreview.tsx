import { Children, isValidElement, memo, useMemo, type ReactNode } from 'react'
import { highlightCode } from '../lib/codeHighlight'
import { codeLanguageLabel } from '../lib/codeLanguages'

const highlighted=(nodes:any[]):ReactNode=>nodes.map((n,i)=>n.type==='text'?n.value:<span key={i} className={(n.properties?.className||[]).join(' ')}>{highlighted(n.children||[])}</span>)
export const CodePreview=memo(function CodePreview({text,language}:{text:string;language?:string}) {
  const tree=useMemo(()=>highlightCode(text,language),[text,language])
  return <div className="answer-code"><div className="code-block-toolbar"><span>{codeLanguageLabel(language)}</span></div><pre><code>{highlighted(tree.children)}</code></pre></div>
})
export function MarkdownCodePreview({children}:{children:ReactNode}) {
  const child=Children.toArray(children).find(c=>isValidElement(c))
  if(!isValidElement<{children?:ReactNode;className?:string}>(child))return <pre>{children}</pre>
  const value=child.props.children
  const text=typeof value==='string'?value:Array.isArray(value)&&value.every(v=>typeof v==='string')?value.join(''):null
  if(text==null)return <pre>{children}</pre>
  const language=/\blanguage-([^\s]+)/.exec(child.props.className||'')?.[1]
  return <CodePreview text={text} language={language}/>
}
