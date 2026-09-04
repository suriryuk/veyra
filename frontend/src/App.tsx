import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertTriangle, Bot, Braces, Check, ChevronRight, CircleGauge, FileCode2, FileSearch, FolderGit2, Gauge, Plus, Search, Send, ShieldCheck, X } from 'lucide-react'
import { api, reduceEvents, streamEvents, type AgentEnvelope, type Session } from './api'
import { ApprovalDialog, type Approval } from './ApprovalDialog'

type Tab = 'activity' | 'workspace' | 'documents' | 'research' | 'audit'
type TreeEntry = { name: string; directory: boolean }

export function App() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [selected, setSelected] = useState<string>()
  const [events, setEvents] = useState<AgentEnvelope[]>([])
  const [message, setMessage] = useState('')
  const [tab, setTab] = useState<Tab>('activity')
  const [detail, setDetail] = useState<Record<string, unknown>>({})
  const [model, setModel] = useState<Record<string, unknown>>({ available: false })
  const [tree, setTree] = useState<TreeEntry[]>([])
  const [preview, setPreview] = useState('파일을 선택하면 내용이 여기에 표시됩니다.')
  const [error, setError] = useState('')
  const [token, setToken] = useState(() => sessionStorage.getItem('veyra.token') ?? '')
  const [connection, setConnection] = useState<'connected' | 'reconnecting' | 'idle'>('idle')
  const composer = useRef<HTMLInputElement>(null)

  const showError = useCallback((value: unknown) => setError(value instanceof Error ? value.message : String(value)), [])
  const refreshSessions = useCallback(() => api<{ items: Session[] }>('/sessions').then((v) => { setSessions(v.items); setSelected((current) => current ?? v.items[0]?.id) }).catch(showError), [showError])

  useEffect(() => { refreshSessions(); api<Record<string, unknown>>('/models/status').then(setModel).catch(() => setModel({ available: false })) }, [refreshSessions])
  useEffect(() => {
    if (!selected) return
    api<Record<string, unknown>>(`/sessions/${selected}`).then(setDetail).catch(showError)
    api<{ items: TreeEntry[] }>(`/workspace/tree?session_id=${selected}`).then((v) => setTree(v.items)).catch(showError)
    const controller = new AbortController(); let last = 0
    const connect = async () => { while (!controller.signal.aborted) { try { await Promise.resolve(); setConnection('connected'); await streamEvents(selected, last, controller.signal, (event) => { last = Math.max(last, event.id); setEvents((all) => reduceEvents(all, event)) }) } catch (reason) { if (!controller.signal.aborted) { setConnection('reconnecting'); showError(reason); await new Promise((resolve) => window.setTimeout(resolve, 1200)) } } } }
    void connect(); return () => controller.abort()
  }, [selected, showError])

  const approval = useMemo(() => {
    const requested = [...events].reverse().find((event) => event.type === 'approval_requested')
    if (!requested) return undefined
    const value = requested.payload.request as Approval
    const resolved = events.some((event) => event.type === 'approval_resolved' && event.payload.approval_id === value.approval_id)
    return resolved ? undefined : value
  }, [events])
  const plan = [...events].reverse().find((event) => event.type === 'plan_created')?.payload.steps as Array<{ id: string; description: string; status: string }> | undefined
  const context = [...events].reverse().find((event) => event.type === 'context_usage_observed')?.payload
  const answer = events.filter((event) => event.type === 'token_delta').map((event) => String(event.payload.text ?? '')).join('')

  async function createSession() { try { const value = await api<{ id: string }>('/sessions', { method: 'POST', body: JSON.stringify({ workspace: '.' }) }); await refreshSessions(); setSelected(value.id); composer.current?.focus() } catch (reason) { showError(reason) } }
  async function submit(event: FormEvent) { event.preventDefault(); if (!selected || !message.trim()) return; const value = message; setMessage(''); try { await api(`/sessions/${selected}/messages`, { method: 'POST', body: JSON.stringify({ message: value }) }) } catch (reason) { setMessage(value); showError(reason) } }
  async function decide(allow: boolean) { if (!approval) return; try { await api(`/approvals/${approval.approval_id}/${allow ? 'allow' : 'deny'}`, { method: 'POST' }) } catch (reason) { showError(reason) } }
  async function openFile(path: string) { if (!selected) return; setTab('workspace'); try { const value = await api<{ content: string }>(`/workspace/file?session_id=${selected}&path=${encodeURIComponent(path)}`); setPreview(value.content) } catch (reason) { showError(reason) } }
  async function loadGit(kind: 'status' | 'diff') { if (!selected) return; try { const value = await api<{ output: string }>(`/workspace/git/${kind}?session_id=${selected}`); setPreview(value.output || `${kind}: 변경 사항 없음`) } catch (reason) { showError(reason) } }
  async function upload(file?: File) { if (!file || !selected) return; const body = new FormData(); body.append('session_id', selected); body.append('file', file); try { await api('/documents', { method: 'POST', body }); setTab('documents') } catch (reason) { showError(reason) } }
  function saveToken(value: string) { setToken(value); if (value) sessionStorage.setItem('veyra.token', value); else sessionStorage.removeItem('veyra.token') }

  return <main className="shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Braces size={19}/></span><strong>VEYRA</strong><span className="version">0.9</span></div>
      <div className="runtime"><label className="token-field">TOKEN<input type="password" value={token} onChange={(event) => saveToken(event.target.value)} onBlur={refreshSessions} aria-label="원격 API 토큰"/></label><span className={`pulse ${model.available ? 'online' : ''}`}/><span>{model.available ? 'MODEL READY' : 'MODEL OFFLINE'}</span><span className="divider"/><Gauge size={15}/><span>{String(context?.estimated_prompt_tokens ?? 0)} TOKENS</span><span className={`connection ${connection}`}>{connection}</span></div>
    </header>
    <section className="workspace-grid">
      <aside className="rail">
        <div className="section-title"><span>SESSIONS</span><button aria-label="새 세션" onClick={createSession}><Plus size={17}/></button></div>
        <nav aria-label="세션 목록">{sessions.map((session) => <button className={`session ${selected === session.id ? 'active' : ''}`} key={session.id} onClick={() => setSelected(session.id)}><span className="session-icon"><Bot size={17}/></span><span><strong>{session.recent_task || '새 세션'}</strong><small>{session.status} · {shortId(session.id)}</small></span><ChevronRight size={15}/></button>)}</nav>
        <div className="section-title workspace-title"><span>WORKSPACE</span><FolderGit2 size={15}/></div>
        <div className="tree">{tree.map((entry) => <button key={entry.name} onClick={() => !entry.directory && openFile(entry.name)}><span>{entry.directory ? '▸' : '·'}</span>{entry.name}</button>)}</div>
      </aside>
      <section className="conversation">
        <div className="conversation-head"><div><small>ACTIVE SESSION</small><strong>{selected ? shortId(selected) : '세션 없음'}</strong></div><span className="status-chip">{String((detail.session as Record<string, unknown>)?.status ?? 'ready')}</span></div>
        <div className="messages" aria-live="polite">
          {!selected && <Empty icon={<Bot/>} title="작업을 시작할 세션을 만드세요"/>}
          {selected && !answer && <Empty icon={<Activity/>} title="메시지를 보내면 실행 과정이 여기에 표시됩니다"/>}
          {answer && <article className="assistant-message"><div className="avatar"><Bot size={18}/></div><div><label>VEYRA</label><p>{answer}</p></div></article>}
        </div>
        <form className="composer" onSubmit={submit}><input ref={composer} value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Agent에게 작업을 요청하세요…" aria-label="Agent 메시지" disabled={!selected}/><button aria-label="메시지 보내기" disabled={!selected || !message.trim()}><Send size={18}/></button></form>
      </section>
      <aside className="inspector">
        <PanelTitle icon={<CircleGauge size={16}/>} title="PLAN"/>
        <div className="plan">{plan?.map((step, index) => <div className="plan-step" key={step.id}><span className={step.status}>{step.status === 'completed' ? <Check size={13}/> : index + 1}</span><p>{step.description}</p></div>) ?? <p className="muted">아직 생성된 계획이 없습니다.</p>}</div>
        <PanelTitle icon={<Gauge size={16}/>} title="CONTEXT"/>
        <div className="meter"><div><span>사용량</span><strong>{String(context?.estimated_prompt_tokens ?? 0)} / 32K</strong></div><progress max="32768" value={Number(context?.estimated_prompt_tokens ?? 0)}/><small>{String(context?.profile ?? 'default')} profile</small></div>
        <PanelTitle icon={<ShieldCheck size={16}/>} title="RUNTIME"/>
        <dl className="facts"><div><dt>API</dt><dd>{connection}</dd></div><div><dt>Model</dt><dd>{model.available ? 'available' : 'offline'}</dd></div><div><dt>Workspace</dt><dd>{String((detail.session as Record<string, unknown>)?.workspace ?? '—')}</dd></div></dl>
      </aside>
    </section>
    <section className="dock">
      <div className="dock-tabs" role="tablist">{(['activity','workspace','documents','research','audit'] as Tab[]).map((item) => <button role="tab" aria-selected={tab === item} key={item} onClick={() => setTab(item)}>{tabIcon(item)}{item}</button>)}<label className="upload"><FileSearch size={15}/>문서 업로드<input type="file" onChange={(e) => upload(e.target.files?.[0])}/></label></div>
      <div className="dock-body">{tab === 'activity' && <EventLog events={events}/>} {tab === 'workspace' && <><div className="workspace-actions"><button onClick={() => loadGit('status')}>Git status</button><button onClick={() => loadGit('diff')}>Diff 보기</button></div><pre>{preview}</pre></>} {tab !== 'activity' && tab !== 'workspace' && <DataView tab={tab} session={selected}/>}</div>
    </section>
    {approval && <ApprovalDialog approval={approval} onDecision={decide}/>} 
    {error && <div className="toast" role="alert"><AlertTriangle size={17}/><span>{error}</span><button aria-label="오류 닫기" onClick={() => setError('')}><X size={15}/></button></div>}
  </main>
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) { return <div className="panel-title">{icon}<span>{title}</span></div> }
function Empty({ icon, title }: { icon: React.ReactNode; title: string }) { return <div className="empty">{icon}<p>{title}</p></div> }
function EventLog({ events }: { events: AgentEnvelope[] }) { return <div className="event-log">{events.slice(-20).reverse().map((event) => <div key={event.id}><span>{event.id}</span><strong>{event.type.replaceAll('_', ' ')}</strong><small>{new Date(event.occurred_at).toLocaleTimeString()}</small></div>)}</div> }
function DataView({ tab, session }: { tab: Tab; session?: string }) { const [data, setData] = useState<unknown>(); useEffect(() => { if (!session) return; const path = tab === 'documents' ? `/documents?session_id=${session}` : tab === 'research' ? `/sessions/${session}/research` : `/audit?session_id=${session}`; api(path).then(setData).catch((error) => setData({ error: String(error) })) }, [tab, session]); return <pre>{JSON.stringify(data ?? { status: 'loading' }, null, 2)}</pre> }
function shortId(id: string) { return id.slice(0, 8) }
function tabIcon(tab: Tab) { const props = { size: 15 }; return tab === 'activity' ? <Activity {...props}/> : tab === 'workspace' ? <FileCode2 {...props}/> : tab === 'documents' ? <FileSearch {...props}/> : tab === 'research' ? <Search {...props}/> : <ShieldCheck {...props}/> }
