/**
 * PDF 笔记：在原文上划线、写批注。全局保存一份列表（@ 引用菜单在编辑器扩展里，拿不到 React 上下文），
 * 组件用 useNotes() 订阅。
 */
import { useEffect, useSyncExternalStore } from 'react'
import { api } from './api'

export type NoteColor = 'yellow' | 'green' | 'blue' | 'pink'

export interface Note {
  id: string
  paper_id: string
  page: number
  quote: string
  comment: string
  /** 归一化矩形 [x0, y0, x1, y1] */
  rects: number[][]
  color: NoteColor
  created_at: string
  updated_at: string
}

export const NOTE_COLORS: Record<NoteColor, { fill: string; dot: string; label: string }> = {
  yellow: { fill: 'rgba(255, 213, 0, 0.38)', dot: '#f5b400', label: '黄' },
  green: { fill: 'rgba(76, 175, 80, 0.30)', dot: '#43a047', label: '绿' },
  blue: { fill: 'rgba(0, 120, 212, 0.24)', dot: '#0078d4', label: '蓝' },
  pink: { fill: 'rgba(233, 30, 99, 0.22)', dot: '#e91e63', label: '粉' },
}

let notes: Note[] = []
let loaded = false
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())

export const notesStore = {
  get: () => notes,
  isLoaded: () => loaded,
  subscribe(l: () => void) {
    listeners.add(l)
    return () => listeners.delete(l)
  },
  async refresh() {
    try {
      notes = await api.notes()
      loaded = true
      emit()
      return notes
    } catch (e) {
      loaded = false
      emit()
      throw e
    }
  },
  upsert(n: Note) {
    notes = [n, ...notes.filter((x) => x.id !== n.id)]
    emit()
  },
  remove(id: string) {
    notes = notes.filter((x) => x.id !== id)
    emit()
  },
}

/** Failed or unfinished loads must not be displayed as zero notes. */
export function useNotesLoaded(): boolean {
  return useSyncExternalStore(notesStore.subscribe, notesStore.isLoaded)
}

/** 全部笔记；传 paperId 时只要这篇论文的，按页码排序 */
export function useNotes(paperId?: string | null): Note[] {
  const all = useSyncExternalStore(notesStore.subscribe, notesStore.get)
  useEffect(() => {
    if (!loaded) notesStore.refresh().catch(() => undefined)
  }, [])
  if (!paperId) return all
  return all.filter((n) => n.paper_id === paperId).sort((a, b) => a.page - b.page || a.created_at.localeCompare(b.created_at))
}
