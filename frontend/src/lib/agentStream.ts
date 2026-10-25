import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamSSE } from './api'
import { runStats } from './runStats'

/**
 * Agent 的 SSE 流：可以中途停止。
 * 停止时保留连接，等后端保存当前步骤并释放执行权，再允许从检查点继续。
 */
export function useAgentStream() {
  const ctrl = useRef<AbortController | null>(null)
  const [running, setRunning] = useState(false)
  const [stopping, setStopping] = useState(false)
  useEffect(() => () => ctrl.current?.abort(), [])

  const run = useCallback(async (path: string, body: unknown, onEvent: (e: any) => void): Promise<'ok' | 'stopped'> => {
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    setRunning(true)
    setStopping(false)
    runStats.setRunning(true)
    try {
      let stopped = false
      await streamSSE(path, body, (event) => {
        runStats.onEvent(event)
        if (event.type === 'status' && event.status === 'stopped') stopped = true
        if (!['stats', 'trace', 'trace_warning'].includes(event?.type)) onEvent(event)
      }, c.signal)
      return stopped ? 'stopped' : 'ok'
    } catch (e: any) {
      if (c.signal.aborted) return 'stopped'
      throw e
    } finally {
      if (ctrl.current === c) {
        ctrl.current = null
        setRunning(false)
        setStopping(false)
        runStats.setRunning(false)
      }
    }
  }, [])

  const stop = useCallback((threadId: string | null) => {
    if (!threadId) return ctrl.current?.abort()
    setStopping(true)
    runStats.setStopping()
    // If the stop request itself fails, disconnecting still asks the worker to stop.
    const current = ctrl.current
    api.agentStop(threadId).catch(() => current?.abort())
  }, [])

  return { run, stop, running, stopping }
}
