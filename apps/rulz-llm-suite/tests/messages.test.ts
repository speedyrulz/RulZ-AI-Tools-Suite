import { describe, expect, it } from 'vitest'
import type { ChatMessage } from '@shared/types'
import { editMessages, idsToDelete, removeMessages, restoreMessages } from '../src/renderer/src/lib/messages'

const msg = (id: string, role: ChatMessage['role'], content = id): ChatMessage => ({ id, role, content, createdAt: 0 })

// A tool round makes two assistant messages for one question.
const chat = (): ChatMessage[] => [msg('u1', 'user'), msg('a1', 'assistant'), msg('u2', 'user'), msg('a2', 'assistant'), msg('a2b', 'assistant'), msg('u3', 'user')]

describe('deleting messages', () => {
  it('takes a user message with its whole reply, and a reply alone', () => {
    expect(idsToDelete(chat(), 'u2')).toEqual(['u2', 'a2', 'a2b'])
    expect(idsToDelete(chat(), 'u3')).toEqual(['u3'])
    expect(idsToDelete(chat(), 'a2')).toEqual(['a2'])
    expect(idsToDelete(chat(), 'nope')).toEqual([])
  })

  it('undoes a delete by putting the messages back where they were', () => {
    const { kept, removed } = removeMessages(chat(), idsToDelete(chat(), 'u2'))
    expect(kept.map((m) => m.id)).toEqual(['u1', 'a1', 'u3'])
    expect(restoreMessages(kept, removed).map((m) => m.id)).toEqual(chat().map((m) => m.id))
  })

  it('puts them back sensibly when the chat changed meanwhile', () => {
    const { kept, removed } = removeMessages(chat(), ['u3'])
    const later = [...kept, msg('u4', 'user')]
    expect(restoreMessages(later, removed).map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'a2b', 'u3', 'u4'])
    // Restoring twice doesn't duplicate.
    expect(restoreMessages(restoreMessages(later, removed), removed)).toHaveLength(7)
    // Everything before it gone: it goes at the end.
    expect(restoreMessages([], removed).map((m) => m.id)).toEqual(['u3'])
  })
})

describe('editing messages', () => {
  it('changes only the text when saving', () => {
    const out = editMessages(chat(), 'a1', 'better answer', false)
    expect(out.find((m) => m.id === 'a1')?.content).toBe('better answer')
    expect(out).toHaveLength(6)
  })

  it('drops what follows a user message sent again', () => {
    const out = editMessages(chat(), 'u2', 'new question', true)
    expect(out.map((m) => m.id)).toEqual(['u1', 'a1', 'u2'])
    expect(out[2].content).toBe('new question')
    // A reply is never cut off at.
    expect(editMessages(chat(), 'a1', 'x', true)).toHaveLength(6)
  })
})
