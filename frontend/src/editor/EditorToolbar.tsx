import type { Editor } from '@tiptap/core'
import { useState } from 'react'
import { useEditorState } from '@tiptap/react'
import { Box, Button, IconButton, Menu, MenuItem, Tooltip, Typography } from '@mui/material'
import FormatBoldIcon from '@mui/icons-material/FormatBold'
import FormatItalicIcon from '@mui/icons-material/FormatItalic'
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted'
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered'
import FormatQuoteIcon from '@mui/icons-material/FormatQuote'
import UndoIcon from '@mui/icons-material/Undo'
import RedoIcon from '@mui/icons-material/Redo'
import AlternateEmailIcon from '@mui/icons-material/AlternateEmail'
import { brand, shadow } from '../theme'
import { INSERT_ITEMS, type InsertKind } from '../lib/richContent'

function TB({ title, active, onClick, children, disabled }: { title: string; active?: boolean; onClick: () => void; children: React.ReactNode; disabled?: boolean }) {
  return (
    <Tooltip title={title}>
      <span>
        <IconButton
          aria-label={title}
          disabled={disabled}
          onMouseDown={(e) => e.preventDefault()}
          onClick={onClick}
          sx={{
            borderRadius: '6px', width: 30, height: 30, color: active ? brand.blue : '#5f6368',
            bgcolor: active ? 'rgba(0,120,212,0.10)' : 'transparent', '& svg': { fontSize: 18 },
          }}
        >
          {children}
        </IconButton>
      </span>
    </Tooltip>
  )
}

const Sep = () => <Box sx={{ width: '1px', height: 18, bgcolor: brand.line, mx: 0.5 }} />

/** 顶部居中的悬浮药丸工具栏 */
export function EditorToolbar({ editor, onInsert }: { editor: Editor; onInsert: (kind: InsertKind) => void }) {
  const [insertAnchor, setInsertAnchor] = useState<HTMLElement | null>(null)
  const s = useEditorState({
    editor,
    selector: ({ editor: e }) => ({
      bold: e.isActive('bold'),
      italic: e.isActive('italic'),
      h1: e.isActive('heading', { level: 1 }),
      h2: e.isActive('heading', { level: 2 }),
      h3: e.isActive('heading', { level: 3 }),
      bullet: e.isActive('bulletList'),
      ordered: e.isActive('orderedList'),
      quote: e.isActive('blockquote'),
      canUndo: e.can().undo(),
      canRedo: e.can().redo(),
    }),
  })
  const H = ({ level }: { level: 1 | 2 | 3 }) => (
    <TB title={`${level} 级标题`} active={s[`h${level}`]} onClick={() => editor.chain().focus().toggleHeading({ level }).run()}>
      <Typography sx={{ fontWeight: 600, fontSize: 12.5 }}>H{level}</Typography>
    </TB>
  )
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', pt: 1.25, pb: 0.5, flexShrink: 0 }}>
      <Box
        sx={{
          display: 'flex', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center', gap: 0.25, px: 0.75, py: 0.25, borderRadius: '10px',
          border: `1px solid ${brand.line}`, boxShadow: shadow.sm, bgcolor: '#fff',
        }}
      >
        <TB title="撤销" onClick={() => editor.chain().focus().undo().run()} disabled={!s.canUndo}>
          <UndoIcon />
        </TB>
        <TB title="重做" onClick={() => editor.chain().focus().redo().run()} disabled={!s.canRedo}>
          <RedoIcon />
        </TB>
        <Sep />
        <H level={1} />
        <H level={2} />
        <H level={3} />
        <Sep />
        <TB title="加粗 (Ctrl+B)" active={s.bold} onClick={() => editor.chain().focus().toggleBold().run()}>
          <FormatBoldIcon />
        </TB>
        <TB title="斜体 (Ctrl+I)" active={s.italic} onClick={() => editor.chain().focus().toggleItalic().run()}>
          <FormatItalicIcon />
        </TB>
        <TB title="无序列表" active={s.bullet} onClick={() => editor.chain().focus().toggleBulletList().run()}>
          <FormatListBulletedIcon />
        </TB>
        <TB title="有序列表" active={s.ordered} onClick={() => editor.chain().focus().toggleOrderedList().run()}>
          <FormatListNumberedIcon />
        </TB>
        <TB title="引用块" active={s.quote} onClick={() => editor.chain().focus().toggleBlockquote().run()}>
          <FormatQuoteIcon />
        </TB>
        <Sep />
        <Button aria-label="插入内容" aria-haspopup="menu" aria-expanded={!!insertAnchor} onMouseDown={e => e.preventDefault()} onClick={e => setInsertAnchor(e.currentTarget)} sx={{ minWidth: 52 }}>插入</Button>
        <Menu anchorEl={insertAnchor} open={!!insertAnchor} onClose={() => setInsertAnchor(null)}>
          {INSERT_ITEMS.map(item => <MenuItem key={item.kind} onClick={() => { setInsertAnchor(null); onInsert(item.kind) }}>{item.label}</MenuItem>)}
        </Menu>
        <TB title="插入引用（也可以直接输入 @）" onClick={() => editor.chain().focus().insertContent('@').run()}>
          <AlternateEmailIcon />
        </TB>
      </Box>
    </Box>
  )
}
