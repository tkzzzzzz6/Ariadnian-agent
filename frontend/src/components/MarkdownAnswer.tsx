import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { Box, Button } from '@mui/material'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { MarkdownCodePreview } from './CodePreview'
import { safeLink } from '../lib/formatting'
import { sourceLinks, type AnswerSource } from '../lib/markdown'
import 'katex/dist/katex.min.css'

/** Throttle Markdown parsing during streaming; final text always flushes immediately. */
export const MarkdownAnswer = memo(function MarkdownAnswer({text,streaming,sources,prefix}:{text:string;streaming:boolean;sources:AnswerSource[];prefix:string}) {
  const [raw,setRaw] = useState(false)
  const [rendered,setRendered] = useState(text)
  const latest = useRef(text)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    latest.current = text
    if (!streaming) { if (timer.current) clearTimeout(timer.current); timer.current=null; setRendered(text) }
    else if (!timer.current) timer.current=setTimeout(() => { timer.current=null; setRendered(latest.current) },80)
  },[text,streaming])
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) },[])
  const plugins = useMemo(() => [remarkGfm,remarkMath,sourceLinks(sources)],[sources])
  return <Box sx={{minWidth:0}}>
    <Box sx={{display:'flex',justifyContent:'flex-end',gap:0.5,mb:0.5}}>
      <Button size="small" aria-pressed={!raw} onClick={()=>setRaw(false)}>预览</Button>
      <Button size="small" aria-pressed={raw} onClick={()=>setRaw(true)}>Markdown 源码</Button>
    </Box>
    {raw ? <Box component="pre" className="answer-raw">{text}</Box> : <Box className="markdown-answer">
      <Markdown skipHtml remarkPlugins={plugins} rehypePlugins={[[rehypeKatex,{trust:false,strict:'ignore',maxExpand:1000}]]}
        urlTransform={url => safeLink(url) || ''}
        components={{
          pre: ({children}) => <MarkdownCodePreview>{children}</MarkdownCodePreview>,
          a: ({href,children}) => /^#aria-source-\d+$/.test(href || '')
            ? <button type="button" className="answer-citation" aria-label={`查看来源 ${href!.split('-').at(-1)}`} onClick={()=>document.getElementById(`${prefix}-${href!.split('-').at(-1)}`)?.scrollIntoView({block:'nearest'})}>{children}</button>
            : href ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}</span>,
          img: ({alt}) => <span>[图片：{alt || '未加载外部图片'}]</span>,
          table: ({children}) => <div className="answer-table"><table>{children}</table></div>,
        }}>{rendered}</Markdown>
    </Box>}
  </Box>
})
