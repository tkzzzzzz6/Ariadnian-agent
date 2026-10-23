import type { TraceSpan, TraceOverview } from './trace'

export const modelPhase: Record<string, string> = { waiting: '等待模型响应', reasoning: '正在接收推理内容', output: '正在生成正文', finished: '调用结束' }
export function expandedByDefault(node: TraceSpan, hasIssue: boolean) {
  return node.status === 'running' || node.status === 'waiting' || node.status === 'error' || hasIssue
}
export function currentMessage(data: TraceOverview): string {
  if (data.run.status === 'waiting') return '需要你确认后继续'
  if (data.run.status === 'stopping') return '正在保存当前步骤并停止'
  if (data.run.status === 'error') return '任务执行失败'
  if (data.run.status === 'interrupted') return '运行意外中断，结束时间未知'
  if (data.run.status === 'done') return '本次运行已完成'
  if (data.run.status === 'stopped') return '已保存并停止，可返回任务继续'
  const nodes = data.spans.filter(n => n.status === 'running')
  return nodes.length > 1 ? `${nodes.length} 个步骤并行执行` : nodes[0]?.label ?? '正在准备执行'
}
export function timelineRange(nodes: TraceSpan[], started: number, ended: number | null, now: number) {
  return Math.max(0.001, (ended ?? now / 1000) - started,
    ...nodes.map(n => Math.max(0, (n.ended ?? n.started) - started)))
}
export function groupCalls(calls: TraceSpan[]) {
  const ids = new Set(calls.map(c => c.id))
  const byParent = new Map<string, TraceSpan[]>()
  for (const c of calls) {
    const key = c.parent_id && ids.has(c.parent_id) ? c.parent_id : ''
    const list = byParent.get(key) ?? []; list.push(c); byParent.set(key, list)
  }
  return byParent
}
export function groupAttempts(calls: TraceSpan[]): TraceSpan[][] {
  const result: TraceSpan[][] = []
  const positions = new Map<string, number>()
  for (const call of calls) {
    const key = call.kind === 'llm' ? call.data.operation : undefined
    if (!key || !positions.has(key)) {
      if (key) positions.set(key, result.length)
      result.push([call])
    } else result[positions.get(key)!].push(call)
  }
  return result
}
