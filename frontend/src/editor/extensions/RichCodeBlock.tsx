import { useEffect, useState } from 'react'
import CodeBlockLowlight from '@tiptap/extension-code-block-lowlight'
import { NodeViewContent, NodeViewWrapper, ReactNodeViewRenderer, type ReactNodeViewProps } from '@tiptap/react'
import { closeHistory } from '@tiptap/pm/history'
import { CODE_LANGUAGES, codeLanguage } from '../../lib/codeLanguages'
import { codeHighlighter } from '../../lib/codeHighlight'

function CodeView({node,updateAttributes,editor}:ReactNodeViewProps) {
  const [copied,setCopied]=useState(false)
  const [copyError,setCopyError]=useState(false)
  useEffect(()=>{setCopied(false)},[node.textContent])
  const language=codeLanguage(node.attrs.language)||''
  const known=!language||CODE_LANGUAGES.some(([id])=>id===language)
  const wrap=node.attrs.wrap!==false
  const change=(attrs:Record<string,unknown>)=>{
    editor.view.dispatch(closeHistory(editor.state.tr))
    updateAttributes(attrs)
    editor.view.dispatch(closeHistory(editor.state.tr).setMeta('addToHistory',false))
  }
  return <NodeViewWrapper className="report-code" data-code-language={language||'auto'}>
    <div className="code-block-toolbar" contentEditable={false}>
      <label>语言 <select aria-label="代码语言" value={language} disabled={!editor.isEditable} onChange={e=>change({language:e.target.value||null,fencePending:false})}>
        <option value="">自动识别</option>
        {CODE_LANGUAGES.map(([id,label])=><option key={id} value={id}>{label}</option>)}
        {!known&&<option value={language}>{language}（未注册，纯文本显示）</option>}
      </select></label>
      <div className="code-block-actions">
        <button type="button" aria-pressed={wrap} disabled={!editor.isEditable} onClick={()=>change({wrap:!wrap})}>自动换行</button>
        <button type="button" onClick={async()=>{
          try{await navigator.clipboard.writeText(node.textContent);setCopied(true);setCopyError(false)}catch{setCopyError(true)}
        }} onBlur={()=>setCopied(false)}>{copied?'已复制':'复制代码'}</button>
      </div>
      {node.textContent.length>(language?40000:12000)&&<span>长代码已暂停高亮，内容仍可编辑</span>}
      {copyError&&<span role="status">复制失败，请选中代码手动复制</span>}
    </div>
    <pre><NodeViewContent<'code'> as="code" data-code-placeholder={node.content.size === 0 ? '输入代码…' : undefined} className={language?`language-${language}`:undefined} style={{whiteSpace:wrap?'pre-wrap':'pre',overflowWrap:wrap?'anywhere':'normal'}}/></pre>
  </NodeViewWrapper>
}

export const RichCodeBlock=CodeBlockLowlight.extend({
  addAttributes(){return {...this.parent?.(),
    wrap:{default:true,parseHTML:()=>true,renderHTML:()=>({})},
    fencePending:{default:false,parseHTML:()=>false,renderHTML:()=>({})},
  }},
  addNodeView(){return ReactNodeViewRenderer(CodeView)},
}).configure({lowlight:codeHighlighter,defaultLanguage:null})
