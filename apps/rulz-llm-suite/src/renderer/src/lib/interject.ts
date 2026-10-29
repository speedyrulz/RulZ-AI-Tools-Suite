import type { Attachment, ChatMessage } from '@shared/types'
import { splitThink } from './thinking'

/** A message the user wrote while a reply was streaming, waiting to be sent when the reply ends. */
export interface QueuedNote {
  id: string
  conversationId: string
  text: string
  attachments: Attachment[]
}

/** The interrupted reply's thinking and visible answer, with inline <think> blocks split out. */
export function partialReply(m: ChatMessage): { reasoning: string; answer: string } {
  const split = splitThink(m.content)
  const reasoning = [m.reasoning?.trim(), split?.reasoning].filter(Boolean).join('\n\n')
  return { reasoning, answer: (split ? split.content : m.content).trim() }
}

/** What the model sees as its own interrupted turn. Never empty, so roles keep alternating. */
export function interruptedAssistantContent(m: ChatMessage): string {
  return partialReply(m).answer || '(I was still thinking when the user stopped me.)'
}

/**
 * The user's note as sent after an interrupted reply. Many chat templates drop the reasoning of
 * past assistant turns, so the partial thinking is quoted here in the user message, where every
 * template keeps it.
 */
export function noteAfterInterrupt(interrupted: ChatMessage, note: string): string {
  const { reasoning } = partialReply(interrupted)
  if (!reasoning) return `I stopped your reply to add something. Take this into account and carry on:\n\n${note}`
  const quoted = reasoning
    .split('\n')
    .map((l) => `> ${l}`)
    .join('\n')
  return `I stopped your reply to add something. This is what you had thought so far:\n\n${quoted}\n\nTake my note into account and carry on from there, without starting over:\n\n${note}`
}

/** Removes the first queued note for `conversationId`, returning it and the rest of the queue. */
export function takeQueued(queue: QueuedNote[], conversationId: string): { note: QueuedNote | null; rest: QueuedNote[] } {
  const i = queue.findIndex((q) => q.conversationId === conversationId)
  if (i < 0) return { note: null, rest: queue }
  return { note: queue[i], rest: [...queue.slice(0, i), ...queue.slice(i + 1)] }
}
