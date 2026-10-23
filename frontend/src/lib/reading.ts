/** 阅读记录与引用统计的纯规则；不从打开文件、滚动或引用行为推断已读。 */
export type ReadingStatus = 'unknown' | 'unread' | 'reading' | 'read'
export interface ReadingRecord {
  paper_id: string
  status: ReadingStatus
  read_through_page: number | null
  total_pages: number | null
  updated_at: string | null
}
export type ReadingPatch = Partial<Pick<ReadingRecord, 'status' | 'read_through_page' | 'total_pages'>>
export const READING_LABELS: Record<ReadingStatus, string> = {
  unknown: '未记录阅读', unread: '未读（手动确认）', reading: '阅读中', read: '已读（手动确认）',
}
export const UNREAD_CITATION_THRESHOLD = 2

export function readingReminder(status: ReadingStatus, count: number): string | null {
  if (count < UNREAD_CITATION_THRESHOLD || status === 'read') return null
  if (status === 'unknown') return `当前文档引用 ${count} 次，尚未记录阅读情况`
  if (status === 'unread') return `当前文档引用 ${count} 次，你已标为未读`
  return `当前文档引用 ${count} 次，尚未确认读完`
}

export function collectCitationCounts(doc: { descendants: (f: (n: any) => void) => void }): Record<string, number> {
  const counts: Record<string, number> = Object.create(null)
  doc.descendants((node: any) => {
    if (node.type.name !== 'citation' || !Array.isArray(node.attrs.ids)) return
    for (const id of new Set<unknown>(node.attrs.ids)) {
      if (typeof id === 'string' && id) counts[id] = (counts[id] ?? 0) + 1
    }
  })
  return counts
}

export function pageProgressLabel(record: ReadingRecord): string {
  const page = record.read_through_page
  return page == null ? '读至页码未记录' : `读至第 ${page} 页${record.total_pages == null ? '（总页数未知）' : ` / 共 ${record.total_pages} 页`}`
}
