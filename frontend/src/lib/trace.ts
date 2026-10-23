export interface TraceRun {
  id: string
  thread_id: string
  started: number
  ended: number | null
  status: string
  capture: boolean
  incomplete: boolean
  error: string | null
  version: string
}
export interface TraceSpan {
  id: string
  parent_id: string | null
  kind: 'node' | 'tool' | 'llm'
  name: string
  label: string
  status: string
  started: number
  ended: number | null
  duration_ms: number | null
  data: {
    model?: string; tier?: string; error?: string; handled_error?: string; warnings?: string[]
    result_count?: number; prompt_tokens?: number | null; completion_tokens?: number | null
    reasoning_tokens?: number | null; first_token_ms?: number | null; first_reasoning_ms?: number | null
    estimated?: boolean; reasoning_available?: boolean; phase?: string
    source?: string; cache_hit?: boolean
    call_count?: number
    node_label?: string
  }
  has_detail: boolean
}
export interface TraceList {
  runs: TraceRun[]
  policy: { capture: boolean; retention_days: number; detail_days: number }
}
export interface TracePage {
  run: TraceRun
  spans: TraceSpan[]
  has_more: boolean
  summary: { calls: number; tools: number; steps: number; known_tokens: number | null; unknown_calls: number | null; estimated: boolean }
  events: { id: number; parent_id: string | null; at: number; message: string; phase?: string }[]
}
export const traceStatus: Record<string, string> = {
  running: '运行中', waiting: '等待你确认', stopping: '正在保存并停止', stopped: '已停止',
  done: '已完成', error: '失败', cancelled: '已取消', interrupted: '意外中断', disconnected: '连接中断，状态待确认',
}
export const traceDuration = (ms: number) => ms < 1000 ? `${Math.max(0, Math.round(ms))} 毫秒`
  : ms < 60000 ? `${(ms / 1000).toFixed(1)} 秒` : `${Math.floor(ms / 60000)} 分 ${Math.floor(ms % 60000 / 1000)} 秒`
