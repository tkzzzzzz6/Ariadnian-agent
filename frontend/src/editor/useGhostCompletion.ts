import type { Editor } from '@tiptap/core'
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { sectionAt, textBetween } from '../lib/editorText'
import { getGhost } from './extensions/GhostText'

/**
 * 停止输入一小段时间后请求补全；继续输入则取消上一个请求（AbortController）。
 * 只在光标位于段落末尾或空白前、且光标前已有一定内容时触发，避免打扰。
 */
export function useGhostCompletion(editor: Editor | null, enabled: boolean, title: string, delay = 700) {
  const timer = useRef<number | undefined>(undefined)
  const ctrl = useRef<AbortController | null>(null)
  const [loading, setLoading] = useState(false)
  const titleRef = useRef(title)
  titleRef.current = title

  useEffect(() => {
    if (!editor) return
    const cancel = () => {
      window.clearTimeout(timer.current)
      ctrl.current?.abort()
      ctrl.current = null
      setLoading(false)
    }
    if (!enabled) {
      cancel()
      editor.commands.setGhost(null)
      return
    }

    const onUpdate = ({ transaction }: { transaction: any }) => {
      if (transaction.getMeta('ghostAccept') || !transaction.docChanged) return
      cancel()
      const { state } = editor
      const { from, empty, $from } = state.selection
      if (!empty || !$from.parent.isTextblock || $from.parent.type.name === 'codeBlock') return
      if (Array.from({ length: $from.depth }, (_, i) => $from.node(i + 1).type.name).some(n => ['tableCell', 'tableHeader'].includes(n))) return
      let hasMath = false
      $from.parent.descendants(n => { if (n.type.name === 'inlineMath') hasMath = true })
      if (hasMath) return
      const after = $from.parent.textBetween($from.parentOffset, $from.parent.content.size)
      if (after.trim().length > 0 && !/^\s/.test(after)) return // 光标在词中间
      const before = textBetween(editor, Math.max(0, from - 2000), from)
      const lastPara = before.split('\n\n').pop() ?? ''
      if (lastPara.trim().length < 6) return

      timer.current = window.setTimeout(async () => {
        const docAtRequest = editor.state.doc
        const c = new AbortController()
        ctrl.current = c
        setLoading(true)
        try {
          const { text } = await api.complete(
            {
              before,
              after: textBetween(editor, from, Math.min(editor.state.doc.content.size, from + 300)),
              title: titleRef.current,
              section: sectionAt(editor, from),
            },
            c.signal,
          )
          // 请求期间文档或光标变了 → 丢弃
          if (c.signal.aborted || editor.state.doc !== docAtRequest || editor.state.selection.from !== from) return
          if (text && text.trim() && editor.isFocused) editor.commands.setGhost({ text, pos: from })
        } catch {
          /* 取消或失败都静默 */
        } finally {
          if (ctrl.current === c) {
            ctrl.current = null
            setLoading(false)
          }
        }
      }, delay)
    }

    const onBlur = () => {
      cancel()
      if (getGhost(editor.state)) editor.commands.setGhost(null)
    }
    editor.on('update', onUpdate)
    editor.on('blur', onBlur)
    return () => {
      editor.off('update', onUpdate)
      editor.off('blur', onBlur)
      cancel()
    }
  }, [editor, enabled, delay])

  return { loading }
}
