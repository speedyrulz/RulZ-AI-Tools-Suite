import type { ChatMessage } from '@shared/types'

/** A message taken out of a chat, with where it was, so Undo can put it back. */
export interface Removed {
  index: number
  message: ChatMessage
}

/**
 * What deleting a message takes with it. A user message goes with the reply that answers it (every
 * message up to the next user message), so no reply is left answering nothing; any other message
 * goes alone.
 */
export function idsToDelete(messages: ChatMessage[], id: string): string[] {
  const i = messages.findIndex((m) => m.id === id)
  if (i < 0) return []
  if (messages[i].role !== 'user') return [id]
  let end = i + 1
  while (end < messages.length && messages[end].role !== 'user') end++
  return messages.slice(i, end).map((m) => m.id)
}

export function removeMessages(messages: ChatMessage[], ids: string[]): { kept: ChatMessage[]; removed: Removed[] } {
  const drop = new Set(ids)
  const removed: Removed[] = []
  const kept = messages.filter((message, index) => {
    if (!drop.has(message.id)) return true
    removed.push({ index, message })
    return false
  })
  return { kept, removed }
}

/** Puts removed messages back where they were (or as near as the chat, changed since, allows). */
export function restoreMessages(messages: ChatMessage[], removed: Removed[]): ChatMessage[] {
  const out = messages.filter((m) => !removed.some((r) => r.message.id === m.id))
  for (const r of [...removed].sort((a, b) => a.index - b.index)) out.splice(Math.min(r.index, out.length), 0, r.message)
  return out
}

/**
 * The chat after editing a message. With `resend` (user messages only) everything after it goes,
 * ready to ask the model again; otherwise only the text changes.
 */
export function editMessages(messages: ChatMessage[], id: string, content: string, resend: boolean): ChatMessage[] {
  const i = messages.findIndex((m) => m.id === id)
  if (i < 0) return messages
  const edited = messages.map((m) => (m.id === id ? { ...m, content } : m))
  return resend && messages[i].role === 'user' ? edited.slice(0, i + 1) : edited
}
