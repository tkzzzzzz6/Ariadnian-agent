/**
 * Agent 运行统计（token、速度、缓存命中、上下文占用）的共享状态。
 * 三种 Agent 各自用 useAgentStream 跑任务，统计事件都汇总到这里，由 Agent 面板底部的状态条统一显示。
 */
import { useSyncExternalStore } from 'react'
import { api } from './api'

export interface ContextUse {
  used: number
  max: number | null
  model: string
}

export interface RunStats {
  calls: number
  failed: number
  steps: number
  prompt_tokens: number
  completion_tokens: number
  cached_tokens: number
  total_tokens: number
  cache_hit: number | null
  tok_s: number | null
  live_tok_s?: number | null
  streaming?: number
  cost_usd: number | null
  elapsed_ms: number
  context: ContextUse | null
  peak_context: ContextUse | null
  models: Record<string, number>
}

interface State {
  threadId: string | null
  stats: RunStats | null
  running: boolean
  /** 收到最近一条统计的时间：运行中据此让耗时继续走 */
  at: number
  status?: string
  step?: string
  warning?: string
}

let state: State = { threadId: null, stats: null, running: false, at: 0 }
const listeners = new Set<() => void>()
const set = (patch: Partial<State>) => {
  state = { ...state, ...patch }
  listeners.forEach((l) => l())
}

export const runStats = {
  /** 流里的事件：status 带 thread_id 时切到这个任务；stats 更新数字 */
  onEvent(e: any) {
    if (e?.type === 'status') {
      if (e.thread_id && e.thread_id !== state.threadId) set({ threadId: e.thread_id, stats: null, step: '', warning: '' })
      set({ status: e.status })
    }
    else if (e?.type === 'stats') set({ stats: e.stats, at: Date.now() })
    else if (e?.type === 'progress' && e.message) set({ step: e.message })
    else if (e?.type === 'trace' && e.kind === 'node' && e.status === 'running') set({ step: e.label })
    else if (e?.type === 'trace_warning') set({ warning: e.message })
    else if (e?.type === 'done') set({ status: 'done' })
    else if (e?.type === 'error') set({ status: 'error' })
  },
  setRunning(running: boolean) {
    if (running !== state.running) set({ running, at: Date.now() })
    if (!running && ['running', 'stopping'].includes(state.status ?? '')) set({ status: 'disconnected' })
  },
  setStopping() { set({ status: 'stopping' }) },
  /** 打开历史任务 / 回到首页：读取保存的统计，null 表示清空 */
  focus(threadId: string | null) {
    if (threadId === state.threadId && (state.stats || !threadId)) return
    set({ threadId, stats: null, step: '', warning: '', status: undefined })
    if (threadId)
      api
        .agentStats(threadId)
        .then((s) => state.threadId === threadId && !state.running && set({ stats: s, at: Date.now() }))
        .catch(() => undefined)
  },
}

export function useRunStats(): State {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    () => state,
  )
}

export function fmtTokens(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`
  return String(n)
}
