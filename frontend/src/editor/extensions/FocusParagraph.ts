/**
 * 专注模式用：给光标所在的顶层块（段落、标题、图表……）加上 has-focus class。
 * 是否淡化其他内容由 CSS 决定（<html class="focus-mode">，见 index.css），这里只负责标出当前块。
 */
import { Extension } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'

export const FocusParagraph = Extension.create({
  name: 'focusParagraph',
  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: new PluginKey('focusParagraph'),
        props: {
          decorations: ({ doc, selection }) => {
            const decos: Decoration[] = []
            // 选区跨了几段时，这几段都保持清晰
            doc.forEach((node, offset) => {
              const end = offset + node.nodeSize
              if (end > selection.from && offset < selection.to + (selection.empty ? 1 : 0))
                decos.push(Decoration.node(offset, end, { class: 'has-focus' }))
            })
            return DecorationSet.create(doc, decos)
          },
        },
      }),
    ]
  },
})
