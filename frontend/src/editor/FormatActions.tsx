import type { Editor } from '@tiptap/core'
import { useEditorState } from '@tiptap/react'
import { useEffect, useRef, useState } from 'react'
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Menu, MenuItem, TextField, Tooltip } from '@mui/material'
import FormatBoldIcon from '@mui/icons-material/FormatBold'
import FormatItalicIcon from '@mui/icons-material/FormatItalic'
import FormatUnderlinedIcon from '@mui/icons-material/FormatUnderlined'
import LinkIcon from '@mui/icons-material/Link'
import MoreHorizIcon from '@mui/icons-material/MoreHoriz'
import type { SelectionBookmark } from '@tiptap/pm/state'
import { closeHistory } from '@tiptap/pm/history'
import { formatAllowed, HIGHLIGHTS, safeLink } from '../lib/formatting'
import { useApp } from '../lib/appContext'

export function FormatActions({editor}:{editor:Editor}) {
  const [anchor,setAnchor]=useState<HTMLElement|null>(null)
  const s=useEditorState({editor,selector:({editor:e})=>({bold:e.isActive('bold'),italic:e.isActive('italic'),underline:e.isActive('underline'),allowed:formatAllowed(e.state)})})
  const run=(f:()=>void)=>{ if (!formatAllowed(editor.state)) return; editor.view.dispatch(closeHistory(editor.state.tr)); f(); editor.view.dispatch(closeHistory(editor.state.tr).setMeta('addToHistory',false)) }
  const icon=(label:string,active:boolean,onClick:()=>void,child:React.ReactNode)=><Tooltip title={label}><span><IconButton size="small" aria-label={label} aria-pressed={active} disabled={!s.allowed} onMouseDown={e=>e.preventDefault()} onClick={()=>run(onClick)} sx={{borderRadius:1,color:active?'#0078d4':'#5f6368',bgcolor:active?'#0078d418':undefined,'& svg':{fontSize:18}}}>{child}</IconButton></span></Tooltip>
  return <Box sx={{display:'inline-flex',alignItems:'center',flexShrink:0}}>
    {icon('加粗 (Ctrl+B)',s.bold,()=>editor.chain().focus().toggleBold().run(),<FormatBoldIcon/>)}
    {icon('斜体 (Ctrl+I)',s.italic,()=>editor.chain().focus().toggleItalic().run(),<FormatItalicIcon/>)}
    {icon('下划线 (Ctrl+U)',s.underline,()=>editor.chain().focus().toggleUnderline().run(),<FormatUnderlinedIcon/>)}
    {icon('设置链接',editor.isActive('link'),()=>editor.view.dom.dispatchEvent(new CustomEvent('aria:edit-link')),<LinkIcon/>)}
    <Tooltip title="更多文字格式"><span><IconButton aria-label="更多文字格式" disabled={!s.allowed} size="small" onMouseDown={e=>e.preventDefault()} onClick={e=>setAnchor(e.currentTarget)}><MoreHorizIcon fontSize="small"/></IconButton></span></Tooltip>
    <Menu data-selection-assist anchorEl={anchor} open={!!anchor} onClose={()=>setAnchor(null)}>
      {([
        ['删除线',()=>editor.chain().focus().toggleStrike().run()],
        ['行内代码',()=>editor.chain().focus().toggleCode().run()],
        ['上标',()=>editor.chain().focus().unsetSubscript().toggleSuperscript().run()],
        ['下标',()=>editor.chain().focus().unsetSuperscript().toggleSubscript().run()],
        ['清除文字格式',()=>editor.chain().focus().unsetAllMarks().run()],
      ] as [string,()=>void][]).map(([name,action])=><MenuItem key={name} onClick={()=>{setAnchor(null);run(action)}}>{name}</MenuItem>)}
      <Box sx={{px:2,pt:1,fontSize:12,color:'#666'}}>文字高亮</Box>
      <Box sx={{display:'flex',gap:1,p:1,px:2}}>{HIGHLIGHTS.map((color,i)=><Button key={color} aria-label={['黄色高亮','绿色高亮','蓝色高亮'][i]} onMouseDown={e=>e.preventDefault()} onClick={()=>{setAnchor(null);run(()=>editor.chain().focus().toggleHighlight({color}).run())}} sx={{minWidth:28,width:28,height:26,bgcolor:color,border:editor.isActive('highlight',{color})?'2px solid #555':'1px solid #ccc','&:hover':{bgcolor:color}}}/>)}</Box>
      <MenuItem onClick={()=>{setAnchor(null);run(()=>editor.chain().focus().unsetHighlight().run())}}>取消高亮</MenuItem>
    </Menu>
  </Box>
}

export function LinkEditor({editor}:{editor:Editor}) {
  const {notify}=useApp()
  const [open,setOpen]=useState(false)
  const [url,setUrl]=useState('')
  const [text,setText]=useState('')
  const [error,setError]=useState('')
  const target=useRef<{bookmark:SelectionBookmark;text:string;empty:boolean}|null>(null)
  useEffect(()=>{
    const begin=()=>{
      if (!formatAllowed(editor.state)) return
      if (editor.isActive('link')) editor.commands.extendMarkRange('link')
      const {from,to,empty}=editor.state.selection
      let atom=false
      editor.state.doc.nodesBetween(from,to,n=>{if(n.isAtom&&!n.isText) atom=true})
      if(atom) return notify('请选择普通文字设置链接，引用与公式保持独立','info')
      const value=editor.state.doc.textBetween(from,to)
      target.current={bookmark:editor.state.selection.getBookmark(),text:value,empty}
      setText(value);setUrl(editor.getAttributes('link').href||'');setError('');setOpen(true)
    }
    const map=({transaction:tr}:any)=>{if(target.current&&tr.docChanged) target.current.bookmark=target.current.bookmark.map(tr.mapping)}
    editor.view.dom.addEventListener('aria:edit-link',begin)
    const key=(event:KeyboardEvent)=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k'){event.preventDefault();begin()}}
    editor.view.dom.addEventListener('keydown',key)
    editor.on('transaction',map)
    return()=>{editor.view.dom.removeEventListener('aria:edit-link',begin);editor.view.dom.removeEventListener('keydown',key);editor.off('transaction',map)}
  },[editor,notify])
  const apply=(remove=false)=>{
    try {
      const t=target.current;if(!t) return
      const selection=t.bookmark.resolve(editor.state.doc)
      if(editor.state.doc.textBetween(selection.from,selection.to)!==t.text) throw new Error('原文字已变化，请重新选择')
      const href=remove?null:safeLink(url)
      if(!remove&&!href) throw new Error('请输入有效的 http、https 或 mailto 链接')
      editor.view.dispatch(editor.state.tr.setSelection(selection))
      if(!formatAllowed(editor.state)) throw new Error('请先保留当前 Agent 草稿')
      const tr=closeHistory(editor.state.tr)
      const type=editor.schema.marks.link
      if(remove) tr.removeMark(selection.from,selection.to,type)
      else if(t.empty) tr.insertText(text.trim()||href!,selection.from).addMark(selection.from,selection.from+(text.trim()||href!).length,type.create({href,target:'_blank',rel:'noopener noreferrer'}))
      else tr.addMark(selection.from,selection.to,type.create({href,target:'_blank',rel:'noopener noreferrer'}))
      editor.view.dispatch(tr);editor.view.dispatch(closeHistory(editor.state.tr).setMeta('addToHistory',false));setOpen(false);target.current=null;editor.commands.focus()
    }catch(e:any){setError(e.message)}
  }
  return <Dialog open={open} onClose={()=>{setOpen(false);target.current=null}} fullWidth maxWidth="sm" aria-labelledby="link-title">
    <DialogTitle id="link-title">设置链接</DialogTitle><DialogContent>
      <TextField autoFocus fullWidth label="链接地址" placeholder="https://example.com" value={url} onChange={e=>setUrl(e.target.value)} error={!!error} helperText={error||'支持网页和邮件链接'} sx={{mt:1}} onKeyDown={e=>{if(e.key==='Enter')apply()}}/>
      {target.current?.empty&&<TextField fullWidth label="显示文字（可选）" value={text} onChange={e=>setText(e.target.value)} sx={{mt:2}}/>}
    </DialogContent><DialogActions><Button color="error" onClick={()=>apply(true)} disabled={!url}>移除链接</Button><Button onClick={()=>{setOpen(false);target.current=null}}>取消</Button><Button variant="contained" onClick={()=>apply()}>保存链接</Button></DialogActions>
  </Dialog>
}
