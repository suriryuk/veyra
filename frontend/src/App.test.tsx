import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { App } from './App'
import { type AgentEnvelope } from './api'

const mock = vi.hoisted(() => ({ api: vi.fn(), listeners: new Map<string, (event: AgentEnvelope) => void>() }))
afterEach(cleanup)
vi.mock('./api', async (original) => ({ ...await original<typeof import('./api')>(), api: mock.api, streamEvents: vi.fn((id: string, _after: number, signal: AbortSignal, receive: (event: AgentEnvelope) => void) => { mock.listeners.set(id, receive); return new Promise<void>((resolve) => signal.addEventListener('abort', () => resolve())) }) }))
beforeEach(() => {
  mock.listeners.clear()
  mock.api.mockReset()
  mock.api.mockImplementation(async (path: string, init?: RequestInit) => {
    if (path === '/sessions' && init?.method === 'POST') return { id: 'new' }
    if (path === '/sessions') return { items: [{ id: 'old', recent_task: '기존 작업' }] }
    if (path === '/models') return { default: 'model-a', items: ['model-a', 'model-b'] }
    if (path === '/models/status') return { available: true }
    if (path.includes('/workspace/tree')) return { items: [] }
    return { session: {} }
  })
})
it('clears replayed output on creation and ignores events from the previous session', async () => {
  render(<App />)
  await waitFor(() => expect(mock.listeners.has('old')).toBe(true))
  const oldEvent: AgentEnvelope = { id: 1, session_id: 'old', occurred_at: '', type: 'token_delta', payload: { text: '이전 답변' } }
  act(() => mock.listeners.get('old')?.(oldEvent))
  expect(screen.getByText('이전 답변')).toBeInTheDocument()
  fireEvent.click(screen.getByLabelText('새 세션'))
  await waitFor(() => expect(mock.listeners.has('new')).toBe(true))
  act(() => mock.listeners.get('old')?.({ ...oldEvent, id: 2 }))
  expect(screen.queryByText('이전 답변')).not.toBeInTheDocument()
  expect(screen.getByLabelText('Agent 메시지')).toHaveValue('')
})
it('sends the chosen model and shows preparation while the request is pending', async () => {
  render(<App />)
  await screen.findByRole('option', { name: 'model-b' })
  fireEvent.change(screen.getByLabelText('작업 모델'), { target: { value: 'model-b' } })
  mock.api.mockImplementationOnce(() => new Promise(() => {}))
  fireEvent.change(screen.getByLabelText('Agent 메시지'), { target: { value: '테스트 작업' } })
  fireEvent.click(screen.getByLabelText('메시지 보내기'))
  expect(screen.getByRole('status')).toHaveTextContent('모델 준비 중 · model-b')
  expect(mock.api).toHaveBeenLastCalledWith('/sessions/old/messages', expect.objectContaining({ body: JSON.stringify({ message: '테스트 작업', model: 'model-b' }) }))
})
it('shows readable document entries and opens metadata on selection', async () => {
  render(<App />)
  await waitFor(() => expect(mock.listeners.has('old')).toBe(true))
  mock.api.mockResolvedValueOnce({ items: [{ id: 'doc', title: '운영 안내', path: '.veyra/documents/guide.md', format: 'markdown', status: 'ready', byte_size: 2048, chunk_count: 3, indexed_at: '2026-09-05' }] })
  fireEvent.click(screen.getByRole('tab', { name: 'documents' }))
  fireEvent.click(await screen.findByRole('button', { name: /운영 안내/ }))
  expect(screen.getByLabelText('문서 상세')).toHaveTextContent('.veyra/documents/guide.md')
  expect(screen.getByLabelText('문서 상세')).toHaveTextContent('2,048 bytes')
})
