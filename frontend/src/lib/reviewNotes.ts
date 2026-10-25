export interface ReviewNoteContext {
  enabled: boolean
  captured_at: string
  scanned: number
  relevant: number
  omitted: number
  scan_limited: boolean
  changed: boolean
  items: {
    id: string; paper_id: string; bibkey: string; title: string; page: number
    quote: string; comment: string; truncated: boolean
    source_status: 'unchanged' | 'changed' | 'deleted' | 'paper_deleted'
  }[]
}
