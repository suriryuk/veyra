import { describe, expect, it } from 'vitest'
import { reduceEvents, type AgentEnvelope } from './api'

const event = (id: number): AgentEnvelope => ({ id, type: 'token_delta', occurred_at: '', session_id: 's', payload: {} })
describe('event replay', () => {
  it('deduplicates reconnect overlap and preserves order', () => expect(reduceEvents(reduceEvents([event(2)], event(1)), event(2)).map((item) => item.id)).toEqual([1, 2]))
})
