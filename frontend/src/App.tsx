import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertTriangle, Bot, Braces, Check, ChevronRight, CircleGauge, FileCode2, FileSearch, FolderGit2, Gauge, Plus, Search, Send, ShieldCheck, X } from 'lucide-react'
import { api, reduceEvents, streamEvents, type AgentEnvelope, type Session } from './api'
import { AnswerContent } from './ImagePreview'
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
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [pending, setPending] = useState(false)
  const [documentsVersion, setDocumentsVersion] = useState(0)
  const selectedRef = useRef(selected)
  const composer = useRef<HTMLInputElement>(null)

  const showError = useCallback((value: unknown) => setError(value instanceof Error ? value.message : String(value)), [])
  const refreshSessions = useCallback(() => api<{ items: Session[] }>('/sessions').then((v) => { setSessions(v.items); setSelected((current) => current ?? v.items[0]?.id) }).catch(showError), [showError])

  useEffect(() => { refreshSessions(); api<{items: string[]; default: string}>('/models').then((value) => { setModels(value.items); setSelectedModel(value.default) }).catch(showError); api<Record<string, unknown>>('/models/status').then(setModel).catch(() => setModel({ available: false })) }, [refreshSessions, showError])
  useEffect(() => {
    if (!selected) return
    const controller = new AbortController(); let last = 0
    selectedRef.current = selected
    api<Record<string, unknown>>(`/sessions/${selected}`, { signal: controller.signal }).then((value) => { if (!controller.signal.aborted && selectedRef.current === selected) setDetail(value) }).catch((e) => { if (!controller.signal.aborted) showError(e) })
    api<{ items: TreeEntry[] }>(`/workspace/tree?session_id=${selected}`, { signal: controller.signal }).then((v) => { if (!controller.signal.aborted && selectedRef.current === selected) setTree(v.items) }).catch((e) => { if (!controller.signal.aborted) showError(e) })
    const connect = async () => { while (!controller.signal.aborted) { try { await Promise.resolve(); if (controller.signal.aborted) return; setConnection('connected'); await streamEvents(selected, last, controller.signal, (event) => { if (controller.signal.aborted || selectedRef.current !== selected) return; last = Math.max(last, event.id); setEvents((all) => reduceEvents(all, event)) }); if (!controller.signal.aborted) { setConnection('reconnecting'); await new Promise((resolve) => window.setTimeout(resolve, 1200)) } } catch (reason) { if (!controller.signal.aborted) { setConnection('reconnecting'); showError(reason); await new Promise((resolve) => window.setTimeout(resolve, 1200)) } } } }
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

  function selectSession(id: string) { selectedRef.current = id; setSelected(id); setEvents([]); setDetail({}); setTree([]); setPreview('파일을 선택하면 내용이 여기에 표시됩니다.'); setMessage(''); setError(''); setPending(false); setConnection('idle') }
  async function createSession() { try { const value = await api<{ id: string }>('/sessions', { method: 'POST', body: JSON.stringify({ workspace: '.' }) }); selectSession(value.id); await refreshSessions(); composer.current?.focus() } catch (reason) { showError(reason) } }
  async function submit(event: FormEvent) { event.preventDefault(); if (!selected || !message.trim() || pending) return; const value = message; const session = selected; setPending(true); setError(''); setMessage(''); try { await api(`/sessions/${session}/messages`, { method: 'POST', body: JSON.stringify({ message: value, model: selectedModel || undefined }) }) } catch (reason) { if (selectedRef.current === session) { setMessage(value); showError(reason) } } finally { if (selectedRef.current === session) setPending(false) } }
  async function decide(allow: boolean) { if (!approval) return; try { await api(`/approvals/${approval.approval_id}/${allow ? 'allow' : 'deny'}`, { method: 'POST' }) } catch (reason) { showError(reason) } }
  async function openFile(path: string) { if (!selected) return; setTab('workspace'); try { const value = await api<{ content: string }>(`/workspace/file?session_id=${selected}&path=${encodeURIComponent(path)}`); if (selectedRef.current === selected) setPreview(value.content) } catch (reason) { showError(reason) } }
  async function loadGit(kind: 'status' | 'diff') { if (!selected) return; try { const value = await api<{ output: string }>(`/workspace/git/${kind}?session_id=${selected}`); if (selectedRef.current === selected) setPreview(value.output || `${kind}: 변경 사항 없음`) } catch (reason) { showError(reason) } }
  async function upload(file?: File) { if (!file || !selected) return; const session = selected; const body = new FormData(); body.append('session_id', session); body.append('file', file); try { await api('/documents', { method: 'POST', body }); if (selectedRef.current === session) { setDocumentsVersion((v) => v + 1); setTab('documents') } } catch (reason) { if (selectedRef.current === session) showError(reason) } }
  function saveToken(value: string) { setToken(value); if (value) sessionStorage.setItem('veyra.token', value); else sessionStorage.removeItem('veyra.token') }

  return <main className="shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Braces size={19}/></span><strong>VEYRA</strong><span className="version">0.9</span></div>
      <div className="runtime"><label className="token-field">TOKEN<input type="password" value={token} onChange={(event) => saveToken(event.target.value)} onBlur={refreshSessions} aria-label="원격 API 토큰"/></label><span className={`pulse ${model.available ? 'online' : ''}`}/><span>{model.available ? 'MODEL READY' : 'MODEL OFFLINE'}</span><span className="divider"/><Gauge size={15}/><span>{String(context?.estimated_prompt_tokens ?? 0)} TOKENS</span><span className={`connection ${connection}`}>{connection}</span></div>
    </header>
    <section className="workspace-grid">
      <aside className="rail">
        <div className="section-title"><span>SESSIONS</span><button aria-label="새 세션" onClick={createSession}><Plus size={17}/></button></div>
        <nav aria-label="세션 목록">{sessions.map((session) => <button className={`session ${selected === session.id ? 'active' : ''}`} key={session.id} onClick={() => selectSession(session.id)}><span className="session-icon"><Bot size={17}/></span><span><strong>{session.recent_task || '새 세션'}</strong><small>{session.status} · {shortId(session.id)}</small></span><ChevronRight size={15}/></button>)}</nav>
        <div className="section-title workspace-title"><span>WORKSPACE</span><FolderGit2 size={15}/></div>
        <div className="tree">{tree.map((entry) => <button key={entry.name} onClick={() => !entry.directory && openFile(entry.name)}><span>{entry.directory ? '▸' : '·'}</span>{entry.name}</button>)}</div>
      </aside>
      <section className="conversation">
        <div className="conversation-head"><div><small>ACTIVE SESSION</small><strong>{selected ? shortId(selected) : '세션 없음'}</strong></div><span className="status-chip">{String((detail.session as Record<string, unknown>)?.status ?? 'ready')}</span></div>
        <div className="messages" key={selected} aria-live="polite">
          <div className="task-progress" role="status">{pending ? `모델 준비 중 · ${selectedModel || '기본 모델'} — 로드 및 연결 확인에 시간이 걸릴 수 있습니다.` : events.length ? eventLabel(events[events.length - 1]) : '새 명령을 받을 준비가 되었습니다.'}</div>
          {!selected && <Empty icon={<Bot/>} title="작업을 시작할 세션을 만드세요"/>}
          {selected && !answer && <Empty icon={<Activity/>} title="메시지를 보내면 실행 과정이 여기에 표시됩니다"/>}
          {answer && <article className="assistant-message"><div className="avatar"><Bot size={18}/></div><div><label>VEYRA</label><AnswerContent text={answer} session={selected!}/></div></article>}
        </div>
        <div className="model-picker"><label htmlFor="model-select">작업 모델</label><select id="model-select" value={selectedModel} disabled={pending} onChange={(e) => setSelectedModel(e.target.value)}>{models.map((name) => <option key={name} value={name}>{name}</option>)}</select></div>
        <form className="composer" onSubmit={submit}><input ref={composer} value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Agent에게 작업을 요청하세요…" aria-label="Agent 메시지" disabled={!selected}/><button aria-label="메시지 보내기" disabled={!selected || !message.trim() || pending}><Send size={18}/></button></form>
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
      <div className="dock-body">{tab === 'activity' && <EventLog events={events}/>} {tab === 'workspace' && <><div className="workspace-actions"><button onClick={() => loadGit('status')}>Git status</button><button onClick={() => loadGit('diff')}>Diff 보기</button></div><pre>{preview}</pre></>} {tab !== 'activity' && tab !== 'workspace' && <DataView key={`${selected}-${tab}-${documentsVersion}`} tab={tab} session={selected}/>}</div>
    </section>
    {approval && <ApprovalDialog approval={approval} onDecision={decide}/>} 
    {error && <div className="toast" role="alert"><AlertTriangle size={17}/><span>{error}</span><button aria-label="오류 닫기" onClick={() => setError('')}><X size={15}/></button></div>}
  </main>
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) { return <div className="panel-title">{icon}<span>{title}</span></div> }
function Empty({ icon, title }: { icon: React.ReactNode; title: string }) { return <div className="empty">{icon}<p>{title}</p></div> }
function EventLog({ events }: { events: AgentEnvelope[] }) {
  const entries = events.filter((event) => event.type !== 'token_delta').slice().reverse()
  return <div className="activity-list">{!entries.length && <p className="muted">아직 실행 기록이 없습니다.</p>}{entries.map((event) => <details className={event.type.includes('failed') ? 'log-error' : ''} key={event.id}><summary><time>{new Date(event.occurred_at).toLocaleTimeString()}</time><strong>{eventLabel(event)}</strong><small>#{event.id}</small></summary><pre>{JSON.stringify(event.payload, null, 2)}</pre></details>)}</div>
}
function eventLabel(event: AgentEnvelope) {
  const labels: Record<string, string> = { token_delta: '답변 생성 중', plan_created: '실행 계획 생성', context_built: '작업 문맥 준비 완료', context_usage_observed: '문맥 사용량 갱신', tool_requested: '도구 실행 요청', tool_started: '도구 실행 중', tool_completed: '도구 실행 완료', tool_failed: '도구 실행 실패', approval_requested: '사용자 승인 대기', approval_resolved: '승인 결정 반영', task_completed: '작업 완료', task_failed: '작업 실패', status_changed: '작업 상태', workflow_phase_changed: '진행 단계' }
  return [labels[event.type] ?? event.type.replaceAll('_', ' '), event.payload.name ?? event.payload.status ?? event.payload.phase ?? event.payload.error].filter(Boolean).join(' · ')
}
type DocumentItem = { id: string; path: string; title?: string; status: string; format: string; byte_size: number; chunk_count: number; indexed_at: string; error?: string }
function DataView({ tab, session }: { tab: Tab; session?: string }) {
  const [data, setData] = useState<{items?: DocumentItem[]; error?: string}>()
  const [chosen, setChosen] = useState<DocumentItem>()
  useEffect(() => {
    if (!session) return
    const controller = new AbortController()
    const path = tab === 'documents' ? `/documents?session_id=${session}` : tab === 'research' ? `/sessions/${session}/research` : `/audit?session_id=${session}`
    api<{items?: DocumentItem[]}>(path, { signal: controller.signal }).then((value) => { if (!controller.signal.aborted) setData(value) }).catch((error) => { if (!controller.signal.aborted) setData({ error: String(error) }) })
    return () => controller.abort()
  }, [tab, session])
  if (!session) return <p className="muted">세션을 선택하세요.</p>
  if (!data) return <p role="status">불러오는 중…</p>
  if (data.error) return <p role="alert">{data.error}</p>
  if (tab !== 'documents') return <pre>{JSON.stringify(data, null, 2)}</pre>
  return <div className="documents-view"><div className="document-list">{!data.items?.length && <p className="muted">등록된 문서가 없습니다. 문서를 업로드해 주세요.</p>}{data.items?.map((item) => <button key={item.id} aria-pressed={chosen?.id === item.id} onClick={() => setChosen(item)}><FileSearch size={20}/><span><strong>{item.title || item.path.split(/[/\\\\]/).pop()}</strong><small>{item.format} · {(item.byte_size / 1024).toFixed(1)} KB · {item.chunk_count} 청크</small></span><span className="status-chip">{item.status}</span></button>)}</div><section className="document-detail" aria-label="문서 상세">{chosen ? <><h3>{chosen.title || '문서 상세'}</h3><dl>{Object.entries({ '경로': chosen.path, '상태': chosen.status, '형식': chosen.format, '크기': `${chosen.byte_size.toLocaleString()} bytes`, '청크 수': chosen.chunk_count, '색인 시각': chosen.indexed_at, ...(chosen.error ? { '오류': chosen.error } : {}) }).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl></> : <p className="muted">목록에서 문서를 선택하면 상세 정보가 표시됩니다.</p>}</section></div>
}
function shortId(id: string) { return id.slice(0, 8) }
function tabIcon(tab: Tab) { const props = { size: 15 }; return tab === 'activity' ? <Activity {...props}/> : tab === 'workspace' ? <FileCode2 {...props}/> : tab === 'documents' ? <FileSearch {...props}/> : tab === 'research' ? <Search {...props}/> : <ShieldCheck {...props}/> }
