import { useEffect, useRef, useState } from 'react'

export function ImagePreview({ session, path }: { session: string; path: string }) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const controller = new AbortController()
    let objectUrl = ''
    const token = sessionStorage.getItem('veyra.token')
    void fetch(`/api/v1/workspace/image?session_id=${encodeURIComponent(session)}&path=${encodeURIComponent(path)}`, { signal: controller.signal, headers: token ? { authorization: `Bearer ${token}` } : {} })
      .then(async (response) => { if (!response.ok) throw new Error('image unavailable'); return response.blob() })
      .then((blob) => { if (!controller.signal.aborted) { objectUrl = URL.createObjectURL(blob); setUrl(objectUrl) } })
      .catch(() => { if (!controller.signal.aborted) setError(true) })
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [session, path])
  return <figure className="image-preview">
    {url ? <button type="button" className="image-thumbnail" onClick={() => dialog.current?.showModal()} aria-label={`${path} 크게 보기`}><img src={url} alt={path} loading="lazy"/></button> : <span>{error ? '이미지를 불러올 수 없습니다.' : '이미지 불러오는 중…'}</span>}
    <figcaption>{path}</figcaption>
    <dialog ref={dialog} className="image-lightbox" aria-label="이미지 확대 보기" onClick={(event) => { if (event.target === event.currentTarget) dialog.current?.close() }}>
      <button type="button" className="image-close" onClick={() => dialog.current?.close()} autoFocus>닫기 ×</button>
      {url && <img src={url} alt={path}/>}<p>{path}</p>
    </dialog>
  </figure>
}

export function AnswerContent({ text, session }: { text: string; session: string }) {
  // Recognize local Markdown images/links and the agent's [path.png] citations.
  const pattern = /!?\[([^\]\n]+)\](?:\(([^)\n]+)\))?/g
  const parts: React.ReactNode[] = []
  let offset = 0
  for (const match of text.matchAll(pattern)) {
    const path = (match[2] ?? match[1]).trim()
    if (!/\.(png|jpe?g|webp)$/i.test(path) || /^(?:[a-z]+:|\/\/|\/|\\)/i.test(path) || path.split(/[/\\]/).includes('..')) continue
    parts.push(<span key={`text-${match.index}`}>{text.slice(offset, match.index)}</span>)
    parts.push(<ImagePreview key={`${session}-${path}-${match.index}`} session={session} path={path}/>)
    offset = match.index + match[0].length
  }
  parts.push(<span key="tail">{text.slice(offset)}</span>)
  return <div className="answer-content">{parts}</div>
}
