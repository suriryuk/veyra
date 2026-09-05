import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { AnswerContent } from './ImagePreview'

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })
it('loads a local citation with authentication and opens a centered dialog', async () => {
  sessionStorage.setItem('veyra.token', 'test-token')
  const fetcher = vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['image']) })
  vi.stubGlobal('fetch', fetcher)
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() })
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', '') }
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open') }
  render(<AnswerContent text={'분석 결과\n[smoke/network.png]'} session="s"/>)
  fireEvent.click(await screen.findByRole('button', { name: 'smoke/network.png 크게 보기' }))
  expect(screen.getByRole('dialog')).toHaveAttribute('open')
  expect(fetcher).toHaveBeenCalledWith(expect.stringContaining('path=smoke%2Fnetwork.png'), expect.objectContaining({ headers: { authorization: 'Bearer test-token' } }))
  fireEvent.click(screen.getByText('닫기 ×'))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  sessionStorage.clear()
})
it('does not fetch remote URLs or parent directory citations', async () => {
  const fetcher = vi.fn()
  vi.stubGlobal('fetch', fetcher)
  await act(async () => { render(<AnswerContent text={'![remote](https://example.com/a.png) [../secret.png]'} session="s"/>) })
  expect(fetcher).not.toHaveBeenCalled()
})
