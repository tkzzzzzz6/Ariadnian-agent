import { createLowlight } from 'lowlight'
import plaintext from 'highlight.js/lib/languages/plaintext'
import python from 'highlight.js/lib/languages/python'
import javascript from 'highlight.js/lib/languages/javascript'
import typescript from 'highlight.js/lib/languages/typescript'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import java from 'highlight.js/lib/languages/java'
import go from 'highlight.js/lib/languages/go'
import rust from 'highlight.js/lib/languages/rust'
import bash from 'highlight.js/lib/languages/bash'
import sql from 'highlight.js/lib/languages/sql'
import json from 'highlight.js/lib/languages/json'
import yaml from 'highlight.js/lib/languages/yaml'
import xml from 'highlight.js/lib/languages/xml'
import css from 'highlight.js/lib/languages/css'
import markdown from 'highlight.js/lib/languages/markdown'
import { codeLanguage } from './codeLanguages'

const lowlight=createLowlight({plaintext,python,javascript,typescript,c,cpp,java,go,rust,bash,sql,json,yaml,xml,css,markdown})
const plain=(value:string)=>({type:'root' as const,children:[{type:'text' as const,value}],data:{language:'plaintext',relevance:0}})
export function highlightCode(value:string,language?:string|null) {
  if(value.length>40000)return plain(value)
  const name=codeLanguage(language)
  if(name) return lowlight.registered(name)?lowlight.highlight(name,value):plain(value)
  // Automatic detection is bounded to keep large pasted source responsive.
  return value.length>12000?plain(value):lowlight.highlightAuto(value,{subset:lowlight.listLanguages().filter(n=>n!=='plaintext')})
}
export const codeHighlighter={
  listLanguages:()=>lowlight.listLanguages(),
  // Unknown labels are explicitly handled as plain text, preserving their source.
  registered:()=>true,
  highlight:(language:string,value:string)=>highlightCode(value,language),
  highlightAuto:(value:string)=>highlightCode(value),
}
