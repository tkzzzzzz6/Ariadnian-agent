import { Extension, type Range } from '@tiptap/core'
import { useEffect, useState } from 'react'
import Image from '@tiptap/extension-image'
import { BlockMath, InlineMath } from '@tiptap/extension-mathematics'
import { Table, TableCell, TableHeader, TableRow } from '@tiptap/extension-table'
import { NodeViewWrapper, ReactNodeViewRenderer, type ReactNodeViewProps } from '@tiptap/react'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { closeHistory } from '@tiptap/pm/history'
import Suggestion, { exitSuggestion } from '@tiptap/suggestion'
import { INSERT_ITEMS, parseTSV, pasteTable, richStructureError, tableJSON, type InsertKind } from '../../lib/richContent'
import { hasMarkdown, markdownToContent } from '../../lib/markdown'
import { formatAllowed } from '../../lib/formatting'
import { brand } from '../../theme'
import { mathInputRules, mathPasteTransaction, syntaxShortcutTransaction } from '../../lib/mathShortcuts'
import 'katex/dist/katex.min.css'

export interface RichActions {
  insert: (kind: InsertKind, range?: Range) => void
  upload: (file: File, pos?: number) => void
  editImage: (pos: number) => void
  editMath: (pos: number) => void
  warn: (message: string) => void
}

export function richExtensions(actions: RichActions) {
  let plainPasteUntil = 0
  function ImageView({ node, selected, getPos, editor }: ReactNodeViewProps) {
    const [failed, setFailed] = useState(false)
    useEffect(() => setFailed(false), [node.attrs.src])
    const safe = /^\/api\/documents\/[a-zA-Z0-9_-]+\/assets\/[0-9a-f]{32}$/.test(node.attrs.src ?? '')
    const edit = () => { const pos = getPos(); if (typeof pos === 'number') actions.editImage(pos) }
    return <NodeViewWrapper as="figure" className={`report-image${selected ? ' is-selected' : ''}`} style={{ width: `${node.attrs.widthPercent}%` }} data-asset-id={node.attrs.assetId}>
      {safe && !failed ? <img src={node.attrs.src} alt={node.attrs.alt || '报告图片'} draggable={false} onError={() => setFailed(true)} onClick={() => {
        const pos = getPos(); if (typeof pos === 'number') editor.commands.setNodeSelection(pos)
      }} onDoubleClick={edit} /> : <span>图片加载失败，请重新上传</span>}
      {node.attrs.caption && <figcaption contentEditable={false}>{node.attrs.caption}</figcaption>}
      {selected && <button type="button" className="rich-image-edit" onClick={edit}>编辑图片</button>}
    </NodeViewWrapper>
  }
  const ReportImage = Image.extend({
    addAttributes() { return { ...this.parent?.(), assetId: { default: null }, extension: { default: 'png' },
      caption: { default: '' }, widthPercent: { default: 100 } } },
    addNodeView() { return ReactNodeViewRenderer(ImageView) },
  }).configure({ allowBase64: false })

  const Slash = Extension.create({
    name: 'richSlash',
    addProseMirrorPlugins() {
      const key = new PluginKey('rich-slash')
      return [Suggestion({ editor: this.editor, pluginKey: key, char: '/', startOfLine: true,
        allowSpaces: false,
        allow: ({ state, range }) => {
          const pos = state.doc.resolve(range.from)
          return pos.parent.type.name === 'paragraph' && !pos.parent.attrs.live && !pos.parent.attrs.draft
            && !Array.from({ length: pos.depth }, (_, i) => pos.node(i + 1).type.name).some(x => ['table', 'tableCell', 'tableHeader'].includes(x))
        },
        items: ({ query }) => INSERT_ITEMS.filter(i => `${i.label} ${i.keywords}`.toLowerCase().includes(query.toLowerCase())),
        command: ({ range, props, editor }) => { actions.insert(props.kind, range); exitSuggestion(editor.view, key) },
        render: () => {
          let menu: HTMLDivElement | null = null
          let current: any
          let active = 0
          const draw = (props: any) => {
            current = props
            if (!menu) return
            menu.replaceChildren()
            props.items.forEach((item: typeof INSERT_ITEMS[number], index: number) => {
              const button = document.createElement('button')
              button.type = 'button'; button.setAttribute('role', 'option'); button.setAttribute('aria-selected', String(index === active))
              button.textContent = `${item.label} · ${item.description}`
              button.style.cssText = `display:block;width:100%;text-align:left;border:0;padding:10px 12px;background:${index === active ? '#edf5fb' : '#fff'};color:#242424;font-size:13px;cursor:pointer`
              button.addEventListener('mousedown', e => { e.preventDefault(); props.command(item) })
              menu!.appendChild(button)
            })
            if (!props.items.length) menu.textContent = '没有匹配的插入操作'
            const rect = props.clientRect?.()
            if (rect) { menu.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 328))}px`; menu.style.top = `${Math.max(8, Math.min(rect.bottom + 5, window.innerHeight - Math.min(menu.scrollHeight + 8, 420)))}px` }
            menu.querySelector('[aria-selected="true"]')?.scrollIntoView({block:'nearest'})
          }
          return {
            onStart: props => {
              active = 0; menu = document.createElement('div'); menu.setAttribute('role', 'listbox'); menu.setAttribute('aria-label', '插入内容')
              menu.style.cssText = `position:fixed;z-index:1500;width:min(320px,calc(100vw - 16px));padding:4px;max-height:min(420px,calc(100vh - 16px));overflow-y:auto;background:white;border:1px solid ${brand.line};border-radius:8px;box-shadow:0 4px 20px #0002`
              document.body.appendChild(menu); draw(props)
            },
            onUpdate: props => { active = 0; draw(props) },
            onKeyDown: ({ event }) => {
              if (event.key === 'Escape') { exitSuggestion(current.editor.view, key); return true }
              if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { active = (active + (event.key === 'ArrowDown' ? 1 : current.items.length - 1)) % Math.max(1, current.items.length); draw(current); return true }
              if (event.key === 'Enter' && current.items[active]) { current.command(current.items[active]); return true }
              return false
            },
            onExit: () => { menu?.remove(); menu = null },
          }
        },
      })]
    },
  })

  const Clipboard = Extension.create({
    name: 'richClipboard',
    priority: 1100,
    addProseMirrorPlugins() { return [new Plugin({
      filterTransaction: tr => {
        if (!tr.docChanged) return true
        const error = richStructureError(tr.doc)
        if (error) { queueMicrotask(() => actions.warn(error)); return false }
        return true
      },
      props: {
        handleKeyDown: (_view,event) => { if((event.ctrlKey||event.metaKey)&&event.shiftKey&&event.key.toLowerCase()==='v') plainPasteUntil=Date.now()+1500; return false },
        handlePaste: (view, event) => {
          if (plainPasteUntil > Date.now()) return false
          const files = [...(event.clipboardData?.files ?? [])]
          if (files.length) { if (files.length > 1) actions.warn('请一次插入一张图片'); else actions.upload(files[0]); return true }
          if (/<img\b/i.test(event.clipboardData?.getData('text/html') ?? '')) { actions.warn('请粘贴截图或上传本地图片，暂不支持外链图片'); return true }
          try {
            if (!formatAllowed(view.state)) return false
            const rows = parseTSV(event.clipboardData?.getData('text/plain') ?? '')
            if (!rows) return false
            const inside = Array.from({ length: view.state.selection.$from.depth }, (_, i) => view.state.selection.$from.node(i + 1).type.name).includes('table')
            if (inside) {
              const tr = pasteTable(view.state, rows)
              if (tr) view.dispatch(tr)
            } else this.editor.chain().focus().command(({ tr }) => { closeHistory(tr); return true }).insertContent(tableJSON(rows.length, rows[0].length, false, rows)).run()
            view.dispatch(closeHistory(view.state.tr).setMeta('addToHistory', false))
            return true
          } catch (e: any) { actions.warn(e.message); return true }
        },
        handleDrop: (view, event, _slice, moved) => {
          if (moved || !event.dataTransfer?.files.length) return false
          const files = [...event.dataTransfer.files]
          const point = view.posAtCoords({ left: event.clientX, top: event.clientY })
          if (files.length > 1) actions.warn('请一次插入一张图片')
          else actions.upload(files[0], point?.pos)
          return true
        },
      },
    })] },
  })
  const MathShortcuts = Extension.create({
    name: 'mathShortcuts',
    addInputRules() { return mathInputRules(this.editor.schema.nodes.inlineMath, this.editor.schema.nodes.blockMath, this.editor.schema.nodes.codeBlock) },
    addProseMirrorPlugins() { return [new Plugin({ props: {
      handleKeyDown:(view,event)=>{
        if(event.isComposing||event.ctrlKey||event.metaKey||event.altKey)return false
        const result=syntaxShortcutTransaction(view.state,event.key)
        if(!result)return false
        view.dispatch(result.tr)
        view.dispatch(closeHistory(view.state.tr).setMeta('addToHistory',false))
        if(result.consume)event.preventDefault()
        return result.consume
      },
      handlePaste: (view, event) => {
        if (plainPasteUntil > Date.now()) { plainPasteUntil=0; return false }
        if (!formatAllowed(view.state) || view.state.selection.$from.parent.type.spec.code || view.state.selection.$from.marks().some(m=>m.type.spec.code)) return false
        // Let the normal HTML pipeline preserve existing rich formatting/nodes.
        if (event.clipboardData?.getData('text/html')) return false
        const text=event.clipboardData?.getData('text/plain') ?? ''
        if (hasMarkdown(text)) {
          const inCell=Array.from({length:view.state.selection.$from.depth},(_,i)=>view.state.selection.$from.node(i+1).type.name).some(n=>['tableCell','tableHeader'].includes(n))
          if (inCell) return false
          try {
            const nodes=markdownToContent(text)
            this.editor.chain().focus().command(({tr})=>{closeHistory(tr);return true}).insertContent(nodes).run()
            view.dispatch(closeHistory(view.state.tr).setMeta('addToHistory',false))
            return true
          } catch(e:any) { actions.warn(e.message); return false }
        }
        const tr = mathPasteTransaction(view.state, text)
        if (!tr) return false
        view.dispatch(tr)
        view.dispatch(closeHistory(view.state.tr).setMeta('addToHistory', false))
        return true
      },
    } })] },
  })
  return [ReportImage,
    Table.configure({ resizable: false, allowTableNodeSelection: false }), TableRow,
    TableCell.extend({ content: 'paragraph+' }), TableHeader.extend({ content: 'paragraph+' }),
    InlineMath.extend({ addInputRules: () => [], addPasteRules: () => [] }).configure({
      onClick: (_node, pos) => actions.editMath(pos), katexOptions: { trust: false, throwOnError: false, maxExpand: 1000 },
    }),
    BlockMath.extend({ addInputRules: () => [], addPasteRules: () => [] }).configure({
      onClick: (_node, pos) => actions.editMath(pos), katexOptions: { trust: false, throwOnError: false, maxExpand: 1000 },
    }), Slash, Clipboard, MathShortcuts]
}
