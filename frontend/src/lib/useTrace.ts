import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import type { TraceList, TraceOverview } from './trace'
import { useRunStats } from './runStats'

/** One request at a time, no polling in hidden/inactive inspector surfaces. */
export function useTrace(tid: string, active: boolean) {
  const [list, setList] = useState<TraceList | null>(null)
  const [rid, setRid] = useState('')
  const [followLatest, setFollowLatest] = useState(true)
  const live = useRunStats()
  const [page, setPage] = useState<TraceOverview | null>(null)
  const [error, setError] = useState('')
  const [now, setNow] = useState(Date.now())
  const [refresh, setRefresh] = useState(0)
  const [revision, setRevision] = useState(0)
  const selected = useRef(rid)
  selected.current = rid
  useEffect(() => { setList(null); setRid(''); setPage(null) }, [tid])
  useEffect(() => {
    if (!active) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    async function load() {
      if (document.hidden) { timer = setTimeout(load, 1500); return }
      try {
        const result = await api.agentTraces(tid)
        if (cancelled) return
        setList(result)
        const current = selected.current
        const id = !followLatest && result.runs.some(r => r.id === current) ? current : result.runs[0]?.id ?? ''
        if (id !== current) { setRid(id); return }
        if (id) {
          const data = await api.agentTraceOverview(tid, id)
          if (cancelled || (selected.current && selected.current !== id)) return
          setPage(data); setRevision(v => v + 1)
        } else setPage(null)
        setError(''); setNow(Date.now())
        const running = result.runs.some(r => ['running', 'stopping'].includes(r.status))
        if (running || (live.threadId === tid && live.running)) timer = setTimeout(load, 1500)
      } catch (e: any) {
        if (!cancelled) { setError(e.message); timer = setTimeout(load, 4000) }
      }
    }
    const wake = () => { if (!document.hidden) { clearTimeout(timer); setRefresh(v => v + 1) } }
    void load()
    document.addEventListener('visibilitychange', wake)
    return () => { cancelled = true; clearTimeout(timer); document.removeEventListener('visibilitychange', wake) }
  }, [tid, active, rid, refresh, followLatest, live.runId, live.status, live.running])
  useEffect(() => {
    if (!active || !page || !['running', 'stopping', 'waiting'].includes(page.run.status)) return
    const timer = setInterval(() => { if (!document.hidden) setNow(Date.now()) }, 1000)
    return () => clearInterval(timer)
  }, [active, page?.run.status])
  return { list, page: page?.run.id === rid ? page : null, rid, setRid: (id: string) => { setFollowLatest(id === list?.runs[0]?.id); setRid(id) }, error, now, revision,
    reload: () => setRefresh(v => v + 1) }
}
