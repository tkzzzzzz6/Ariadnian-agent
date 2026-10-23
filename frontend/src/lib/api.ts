import type {
  CiteCandidate, DocFull, DocMeta, Health, Paper, PaperIn, PdfHighlight, PdfPassage, PdfSource, SettingsView, TestResult, UsageStats,
} from './types'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      msg = j.detail ?? msg
    } catch {
      /* ignore */
    }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  const ct = res.headers.get('content-type') ?? ''
  return (ct.includes('application/json') ? res.json() : res.text()) as Promise<T>
}

/** 读取 text/event-stream（POST 版 SSE）。每个 data: 行解析为一个 JSON 事件。 */
/** 解析全文的结果：parser 是实际用的解析器，note 是外部解析器不可用、退回 PyMuPDF 的原因 */
export interface ParseResult {
  paper: Paper
  chunks: number
  parser?: string
  note?: string | null
}

export async function streamSSE(
  path: string,
  body: unknown,
  onEvent: (e: any) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    let msg = `${res.status} ${res.statusText}`
    try {
      msg = (await res.json()).detail ?? msg
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const raw = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      for (const line of raw.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)))
      }
    }
  }
}

export const api = {
  uploadImage: (docId: string, file: File) => {
    const data = new FormData()
    data.append('file', file)
    return req<{ id: string; src: string; extension: string; width: number; height: number; size: number }>(`/api/documents/${docId}/assets`, { method: 'POST', body: data })
  },
  health: () => req<Health>('/api/health'),
  settings: () => req<SettingsView>('/api/settings'),
  saveSettings: (values: Record<string, string | null>) =>
    req<SettingsView>('/api/settings', { method: 'PUT', body: JSON.stringify({ values }) }),
  testLLM: (body: { tier?: string; model?: string; api_base?: string | null; api_key?: string }) =>
    req<TestResult>('/api/settings/test-llm', { method: 'POST', body: JSON.stringify(body) }),
  testParser: (parser: 'grobid' | 'mineru') => req<TestResult>('/api/settings/test-parser', { method: 'POST', body: JSON.stringify({ parser }) }),
  testSearch: (source: string) => req<TestResult>('/api/settings/test-search', { method: 'POST', body: JSON.stringify({ source }) }),
  llmStats: (days = 30) => req<UsageStats>(`/api/llm/stats?days=${days}`),

  // 文献库
  papers: () => req<Paper[]>('/api/papers'),
  addPaper: (p: PaperIn) => req<{ paper: Paper; created: boolean }>('/api/papers', { method: 'POST', body: JSON.stringify(p) }),
  updatePaper: (id: string, fields: Partial<Paper>) =>
    req<Paper>(`/api/papers/${id}`, { method: 'PATCH', body: JSON.stringify(fields) }),
  deletePaper: (id: string) => req(`/api/papers/${id}`, { method: 'DELETE' }),
  searchExternal: (q: string, sources = 'semantic_scholar,arxiv') =>
    req<{ results: PaperIn[]; errors: string[] }>(
      `/api/papers/external?q=${encodeURIComponent(q)}&sources=${encodeURIComponent(sources)}`,
    ),
  uploadPaper: (file: File, paperId?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (paperId) fd.append('paper_id', paperId)
    return req<ParseResult>('/api/papers/upload', { method: 'POST', body: fd })
  },
  fetchFulltext: (id: string) => req<ParseResult>(`/api/papers/${id}/fulltext`, { method: 'POST' }),
  reparse: (id: string) => req<ParseResult>(`/api/papers/${id}/reparse`, { method: 'POST' }),
  chunks: (id: string) => req<{ id: number; section: string; page: number; text: string }[]>(`/api/papers/${id}/chunks`),
  bibtex: (ids?: string[]) => req<string>(`/api/papers/bibtex${ids ? `?ids=${ids.join(',')}` : ''}`),

  // 原文 PDF
  pdfLocate: (src: PdfSource, text: string, page?: number | null, claim = '') =>
    req<PdfHighlight & { pages: number }>('/api/pdf/locate', { method: 'POST', body: JSON.stringify({ ...src, text, page, claim }) }),
  pdfPassages: (src: PdfSource, claim: string, k = 3) =>
    req<{ passages: PdfPassage[] }>('/api/pdf/passages', { method: 'POST', body: JSON.stringify({ ...src, claim, k }) }),

  // PDF 笔记
  notes: (paperId?: string) => req<import('./notes').Note[]>(`/api/notes${paperId ? `?paper_id=${encodeURIComponent(paperId)}` : ''}`),
  createNote: (n: { paper_id: string; page: number; quote: string; comment?: string; rects: number[][]; color?: string }) =>
    req<import('./notes').Note>('/api/notes', { method: 'POST', body: JSON.stringify(n) }),
  updateNote: (id: string, fields: { comment?: string; color?: string }) =>
    req<import('./notes').Note>(`/api/notes/${id}`, { method: 'PATCH', body: JSON.stringify(fields) }),
  deleteNote: (id: string) => req(`/api/notes/${id}`, { method: 'DELETE' }),

  // 文档
  docs: () => req<DocMeta[]>('/api/documents'),
  createDoc: (title?: string, content?: any) =>
    req<DocFull>('/api/documents', { method: 'POST', body: JSON.stringify({ title, content }) }),
  getDoc: (id: string) => req<DocFull>(`/api/documents/${id}`),
  saveDoc: (id: string, fields: Partial<Pick<DocFull, 'title' | 'content' | 'cite_style'>>) =>
    req(`/api/documents/${id}`, { method: 'PUT', body: JSON.stringify(fields) }),
  deleteDoc: (id: string) => req(`/api/documents/${id}`, { method: 'DELETE' }),

  // 写作辅助
  complete: (body: { before: string; after: string; title: string; section: string }, signal?: AbortSignal) =>
    req<{ text: string }>('/api/assist/complete', { method: 'POST', body: JSON.stringify(body), signal }),
  cite: (sentence: string, k = 5) =>
    req<{ candidates: CiteCandidate[] }>('/api/assist/cite', { method: 'POST', body: JSON.stringify({ sentence, k }) }),
  verify: (sentence: string, paper_ids: string[]) =>
    req<{ results: { paper_id: string; verdict: string; reason: string; snippet?: string; page?: number }[] }>(
      '/api/assist/verify',
      { method: 'POST', body: JSON.stringify({ sentence, paper_ids }) },
    ),

  annotate: (paragraph: string, title = '') =>
    req<{ annotations: import('./review').Annotation[] }>('/api/assist/annotate', {
      method: 'POST',
      body: JSON.stringify({ paragraph, title }),
    }),

  agentThreads: (docId?: string) =>
    req<{ thread_id: string; topic: string; status: string; kind?: 'review' | 'write' | 'critique'; created_at: string; updated_at?: string; tokens?: number | null }[]>(
      `/api/agent${docId ? `?doc_id=${docId}` : ''}`,
    ),
  questions: (title: string, headings: string[], refresh = false) =>
    req<{ questions: { q: string; papers: { id: string; bibkey: string; short: string; year?: number | null }[]; score: number }[] }>(
      '/api/assist/questions',
      { method: 'POST', body: JSON.stringify({ title, headings, refresh }) },
    ),
  agentStats: (tid: string) => req<import('./runStats').RunStats | null>(`/api/agent/${tid}/stats`),
  agentStop: (tid: string) => req<{ ok: boolean }>(`/api/agent/${tid}/stop`, { method: 'POST' }),
  agentState: (tid: string) => req<{ thread_id: string; status: string; interrupt: any; result: any }>(`/api/agent/${tid}`),
}

/** 原文 PDF 的地址：文献库论文 / 临时预览 */
export function pdfFileUrl(src: PdfSource): string {
  if (src.paper_id) return `/api/pdf/file?paper_id=${encodeURIComponent(src.paper_id)}`
  return `/api/pdf/file?url=${encodeURIComponent(src.url ?? '')}`
}
