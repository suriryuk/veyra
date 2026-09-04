import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ApprovalDialog } from './ApprovalDialog'

describe('ApprovalDialog', () => {
  it('focuses the safe decision surface, traps focus, and emits one decision', () => {
    const decide = vi.fn()
    render(<ApprovalDialog approval={{ approval_id: 'a', action: 'cargo_test', risk: 'EXECUTE', reason: '검증', expected_effect: 'tests' }} onDecision={decide}/>)
    const allow = screen.getByRole('button', { name: '이번만 허용' })
    const deny = screen.getByRole('button', { name: '거부' })
    expect(allow).toHaveFocus()
    fireEvent.keyDown(screen.getByRole('alertdialog'), { key: 'Tab' })
    expect(deny).toHaveFocus()
    fireEvent.click(deny)
    expect(decide).toHaveBeenCalledOnce()
    expect(decide).toHaveBeenCalledWith(false)
  })
})
