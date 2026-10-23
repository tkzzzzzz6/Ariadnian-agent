import { useEffect, useState, useSyncExternalStore } from 'react'
import type { Editor } from '@tiptap/core'
import { api } from './api'
import { collectCitationCounts, type ReadingPatch, type ReadingRecord } from './reading'

let snapshot: { records: Record<string, ReadingRecord>; loaded: boolean; error: string | null } = {
  records: {}, loaded: false, error: null,
}
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())
let pending: Promise<void> | null = null
let revision = 0
const editedAt = new Map<string, number>()

export const readingStore = {
  get: () => snapshot,
  subscribe(l: () => void) { listeners.add(l); return () => { listeners.delete(l) } },
  refresh(): Promise<void> {
    if (pending) return pending
    const started = revision
    pending = api.reading().then((records) => {
      // A refresh started before a successful edit must not roll it back.
      const next = Object.fromEntries(records.map((r) => [r.paper_id, r]))
      for (const [id, at] of editedAt) {
        if (at > started && snapshot.records[id]) next[id] = snapshot.records[id]
      }
      snapshot = { records: next, loaded: true, error: null }
    }).catch((e) => {
      snapshot = { ...snapshot, error: String(e.message ?? e) }
    }).finally(() => { pending = null; emit() })
    return pending
  },
  async save(id: string, patch: ReadingPatch) {
    const record = await api.saveReading(id, patch)
    revision++
    editedAt.set(id, revision)
    snapshot = { ...snapshot, records: { ...snapshot.records, [id]: record } }
    emit()
    return record
  },
}

export function useReading() {
  const state = useSyncExternalStore(readingStore.subscribe, readingStore.get)
  useEffect(() => { void readingStore.refresh() }, [])
  return state
}

/** Subscribe to every document-changing transaction, including history and Agent edits. */
export function useCitationCounts(editor: Editor | null) {
  const [state, setState] = useState<{ editor: Editor | null; counts: Record<string, number> }>({ editor: null, counts: {} })
  useEffect(() => {
    if (!editor) return
    const update = () => setState({ editor, counts: collectCitationCounts(editor.state.doc) })
    const changed = ({ transaction }: { transaction: { docChanged: boolean } }) => { if (transaction.docChanged) update() }
    update()
    editor.on('transaction', changed)
    return () => { editor.off('transaction', changed) }
  }, [editor])
  return state.editor === editor ? state.counts : {}
}
