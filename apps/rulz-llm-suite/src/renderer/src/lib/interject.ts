import type { Attachment } from '@shared/types'

// The pure parts moved to src/shared/wire.ts (the main process builds the same wire messages).
export { interruptedAssistantContent, noteAfterInterrupt, partialReply } from '@shared/wire'

/** A message the user wrote while a reply was streaming, waiting to be sent when the reply ends. */
export interface QueuedNote {
  id: string
  conversationId: string
  text: string
  attachments: Attachment[]
}

/** Removes the first queued note for `conversationId`, returning it and the rest of the queue. */
export function takeQueued(queue: QueuedNote[], conversationId: string): { note: QueuedNote | null; rest: QueuedNote[] } {
  const i = queue.findIndex((q) => q.conversationId === conversationId)
  if (i < 0) return { note: null, rest: queue }
  return { note: queue[i], rest: [...queue.slice(0, i), ...queue.slice(i + 1)] }
}
