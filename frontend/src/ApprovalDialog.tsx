import { useEffect, useRef } from 'react'
import { AlertTriangle, Check, X } from 'lucide-react'

export type Approval = { approval_id: string; action: string; risk: string; target?: string; reason: string; expected_effect: string; warning?: string }

export function ApprovalDialog({ approval, onDecision }: { approval: Approval; onDecision: (allow: boolean) => void }) {
  const dialog = useRef<HTMLElement>(null)
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const buttons = dialog.current?.querySelectorAll<HTMLButtonElement>('button')
    buttons?.[1]?.focus()
    return () => previous?.focus()
  }, [])

  function trapFocus(event: React.KeyboardEvent) {
    if (event.key !== 'Tab') return
    const buttons = Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>('button') ?? [])
    if (!buttons.length) return
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement)
    const next = event.shiftKey ? (index <= 0 ? buttons.length - 1 : index - 1) : (index + 1) % buttons.length
    event.preventDefault(); buttons[next]?.focus()
  }

  return <div className="modal-backdrop" role="presentation"><section ref={dialog} onKeyDown={trapFocus} className="approval" role="alertdialog" aria-modal="true" aria-labelledby="approval-title"><div className="approval-icon"><AlertTriangle/></div><div><small>PERMISSION REQUIRED · {approval.risk}</small><h2 id="approval-title">{approval.action}</h2><p>{approval.reason}</p><code>{approval.target ?? approval.expected_effect}</code>{approval.warning && <p className="warning">{approval.warning}</p>}<div className="approval-actions"><button onClick={() => onDecision(false)}><X size={16}/>거부</button><button className="allow" onClick={() => onDecision(true)}><Check size={16}/>이번만 허용</button></div></div></section></div>
}
