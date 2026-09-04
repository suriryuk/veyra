export type Session = { id: string; status: string; workspace: string; updated_at: string; recent_task: string }
export type AgentEnvelope = { id: number; type: string; occurred_at: string; session_id: string; task_id?: string; payload: Record<string, unknown> }

const headers = (json = false): HeadersInit => {
  const token = sessionStorage.getItem('veyra.token')
  return { ...(json ? { 'content-type': 'application/json' } : {}), ...(token ? { authorization: `Bearer ${token}` } : {}) }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/v1${path}`, { ...init, headers: { ...headers(init.body instanceof FormData ? false : Boolean(init.body)), ...init.headers } })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ message: response.statusText }))
    throw new Error(body.message ?? `HTTP ${response.status}`)
  }
  return response.status === 204 ? undefined as T : response.json()
}

export async function streamEvents(sessionId: string, after: number, signal: AbortSignal, onEvent: (event: AgentEnvelope) => void) {
  const response = await fetch(`/api/v1/sessions/${sessionId}/events?after=${after}`, { headers: headers(), signal })
  if (!response.ok || !response.body) throw new Error(`이벤트 연결 실패: HTTP ${response.status}`)
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const blocks = buffer.split('\n\n'); buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      const data = block.split('\n').find((line) => line.startsWith('data:'))?.slice(5).trim()
      if (data) onEvent(JSON.parse(data) as AgentEnvelope)
    }
  }
}

export function reduceEvents(events: AgentEnvelope[], incoming: AgentEnvelope): AgentEnvelope[] {
  if (events.some((event) => event.id === incoming.id)) return events
  return [...events, incoming].sort((a, b) => a.id - b.id)
}
