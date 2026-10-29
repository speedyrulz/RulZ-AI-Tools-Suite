// Deleting and redoing messages in the conversation. Shared so the window can grey out the buttons
// with the same rules the main process enforces.
import type { TaskRecord, TimelineItem, WireMessage } from './types'

/** What the model sees in place of a reply the owner deleted (kept so roles still alternate). */
export const DELETED_REPLY = '(The owner deleted this reply.)'

/** The request that started a task (later user items of the same task are notes). */
export function requestItem(timeline: TimelineItem[], taskId: string): Extract<TimelineItem, { kind: 'user' }> | undefined {
  return timeline.find((i): i is Extract<TimelineItem, { kind: 'user' }> => i.kind === 'user' && i.taskId === taskId && !i.note)
}

const isRequest = (timeline: TimelineItem[], item: TimelineItem): boolean => item.kind === 'user' && requestItem(timeline, item.taskId)?.id === item.id

/** Why a message can't be deleted right now, or null when it can. */
export function whyNotDelete(timeline: TimelineItem[], tasks: TaskRecord[], itemId: string, running: boolean): string | null {
  const item = timeline.find((i) => i.id === itemId)
  if (!item || item.kind === 'notice') return 'This message is gone.'
  if (running) return 'Wait for the task to finish (or press Stop) first.'
  if (item.kind === 'assistant') return null
  if (!isRequest(timeline, item)) return 'Notes belong to their request: delete the request to remove them.'
  const task = tasks.find((t) => t.id === item.taskId)
  if (task && task.files.length && !task.undone) {
    return 'This request changed files, and deleting it would make me forget what I did. Undo it first (Undo last task, or History), then delete it.'
  }
  return null
}

/** Why the request can't be edited and run again, or null when it can. Only the latest request can. */
export function whyNotRedo(timeline: TimelineItem[], tasks: TaskRecord[], itemId: string, running: boolean): string | null {
  const item = timeline.find((i) => i.id === itemId)
  if (!item || !isRequest(timeline, item)) return 'Only a request can be run again.'
  if (running) return 'Wait for the task to finish (or press Stop) first.'
  const task = tasks.at(-1)
  if (!task || task.id !== item.taskId) return 'Only your latest request can be run again.'
  if (task.files.length && !task.undone && !task.snapshot) {
    return "This request changed files and has no restore point, so I can't put them back before running it again."
  }
  return null
}

/** Where a task's messages are in the model's history: from the message that opens it to the next task's. */
export function exchangeRange(messages: WireMessage[], taskId: string): { from: number; to: number } | null {
  const from = messages.findIndex((m) => m.task === taskId)
  if (from < 0) return null
  const next = messages.findIndex((m, i) => i > from && m.task)
  return { from, to: next < 0 ? messages.length : next }
}

/**
 * Removes a task's request and everything that followed it from the history and the timeline.
 * Returns false when its messages were no longer in the history (folded into the summary, or saved
 * by an older version), so the model still needs telling.
 */
export function dropExchange(data: { messages: WireMessage[]; timeline: TimelineItem[] }, taskId: string): boolean {
  data.timeline = data.timeline.filter((i) => i.taskId !== taskId)
  const range = exchangeRange(data.messages, taskId)
  if (!range) return false
  data.messages = [...data.messages.slice(0, range.from), ...data.messages.slice(range.to)]
  return true
}

/**
 * Removes one reply from the timeline, and its text from the history. The message itself stays:
 * its tool calls go with steps that ran, and dropping it could put two user messages in a row.
 */
export function dropReply(data: { messages: WireMessage[]; timeline: TimelineItem[] }, itemId: string): void {
  data.timeline = data.timeline.filter((i) => i.id !== itemId)
  const m = data.messages.find((x) => x.item === itemId)
  if (m) m.content = m.tool_calls?.length ? '' : DELETED_REPLY
}

/** The history as sent to the model: without the links back to the timeline. */
export function forModel(messages: WireMessage[]): WireMessage[] {
  return messages.map(({ task: _task, item: _item, ...m }) => m)
}
