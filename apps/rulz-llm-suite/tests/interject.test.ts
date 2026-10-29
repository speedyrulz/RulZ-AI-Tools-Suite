import { describe, expect, it } from 'vitest'
import type { ChatMessage } from '@shared/types'
import { interruptedAssistantContent, noteAfterInterrupt, partialReply, type QueuedNote, takeQueued } from '../src/renderer/src/lib/interject'
import { buildWire } from '../src/renderer/src/lib/wire'

const msg = (role: ChatMessage['role'], content: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id: Math.random().toString(36).slice(2),
  role,
  content,
  createdAt: 0,
  ...extra
})

describe('interrupted replies', () => {
  it('splits parsed and inline reasoning from the partial answer', () => {
    expect(partialReply(msg('assistant', 'Part', { reasoning: 'thinking' }))).toEqual({ reasoning: 'thinking', answer: 'Part' })
    expect(partialReply(msg('assistant', '<think>inline so far'))).toEqual({ reasoning: 'inline so far', answer: '' })
  })

  it('never sends an empty assistant turn', () => {
    expect(interruptedAssistantContent(msg('assistant', '', { reasoning: 'r' }))).toMatch(/still thinking/)
    expect(interruptedAssistantContent(msg('assistant', '<think>x</think>Half an ans'))).toBe('Half an ans')
  })

  it('quotes the partial thinking in the note', () => {
    const note = noteAfterInterrupt(msg('assistant', '', { reasoning: 'line one\nline two' }), 'It is 42.')
    expect(note).toContain('> line one\n> line two')
    expect(note.endsWith('It is 42.')).toBe(true)
    expect(noteAfterInterrupt(msg('assistant', 'Hi'), 'note')).not.toContain('>')
  })

  it('feeds the thinking back through the wire messages', () => {
    const wire = buildWire(
      [
        msg('user', 'What is x?'),
        msg('assistant', '', { reasoning: 'Maybe x is 7', interrupted: true }),
        msg('user', 'x is 42'),
        msg('assistant', 'Then x is 42.'),
        msg('user', 'Thanks')
      ],
      'sys',
      false,
      null
    )
    expect(wire.map((w) => w.role)).toEqual(['system', 'user', 'assistant', 'user', 'assistant', 'user'])
    expect(wire[2].content).toMatch(/still thinking/)
    expect(wire[3].content).toContain('> Maybe x is 7')
    expect(String(wire[3].content).endsWith('x is 42')).toBe(true)
    // Only the note right after the interruption carries the quote.
    expect(wire[5].content).toBe('Thanks')
  })
})

describe('takeQueued', () => {
  const q = (id: string, conversationId: string): QueuedNote => ({ id, conversationId, text: id, attachments: [] })
  it('takes the oldest note for the chat and keeps the rest in order', () => {
    const queue = [q('a', 'c1'), q('b', 'c2'), q('c', 'c1')]
    const first = takeQueued(queue, 'c1')
    expect(first.note?.id).toBe('a')
    expect(first.rest.map((n) => n.id)).toEqual(['b', 'c'])
    expect(takeQueued(first.rest, 'c1').note?.id).toBe('c')
    expect(takeQueued(queue, 'c3')).toEqual({ note: null, rest: queue })
  })
})
