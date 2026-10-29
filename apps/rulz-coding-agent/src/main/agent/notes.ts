import { randomUUID } from 'node:crypto'
import type { PendingNote } from '@shared/types'

/**
 * Notes the owner writes while a task runs. They go into the conversation between steps, right
 * before the next model call. A note can also cut the model's current turn short, so it applies at
 * once; commands and tools that are running are never cut (only Stop does that).
 */
export class NoteQueue {
  private notes: PendingNote[] = []
  private turn: AbortController | null = null
  private cut = false

  get pending(): PendingNote[] {
    return [...this.notes]
  }

  get size(): number {
    return this.notes.length
  }

  add(text: string, interrupt: boolean): PendingNote | null {
    const trimmed = text.trim()
    if (!trimmed) return null
    const note = { id: randomUUID(), text: trimmed }
    this.notes.push(note)
    if (interrupt) this.interrupt()
    return note
  }

  cancel(id: string): boolean {
    const before = this.notes.length
    this.notes = this.notes.filter((n) => n.id !== id)
    return this.notes.length < before
  }

  /** Every waiting note, oldest first; the queue is empty afterwards. */
  take(): PendingNote[] {
    return this.notes.splice(0)
  }

  /** The signal for one model turn: it aborts with the task, or when a note interrupts the turn. */
  beginTurn(task: AbortSignal): AbortSignal {
    this.turn = new AbortController()
    this.cut = false
    return AbortSignal.any([task, this.turn.signal])
  }

  /** Ends the turn; true when a note cut it short (and the task itself goes on). */
  endTurn(): boolean {
    const cut = this.cut
    this.turn = null
    this.cut = false
    return cut
  }

  /** Cuts the model's current turn short. Does nothing between turns: the notes go in before the next one anyway. */
  interrupt(): void {
    if (!this.turn || this.turn.signal.aborted) return
    this.cut = true
    this.turn.abort()
  }
}

/** Quotes text as a Markdown block quote. */
function quote(text: string): string {
  return text
    .split('\n')
    .map((l) => `> ${l}`)
    .join('\n')
}

/**
 * The user message that carries the owner's notes. After a cut turn it also quotes the model's
 * thinking so far: chat templates often drop the reasoning of past turns, so this is the only way
 * the model gets to pick up where it was.
 */
export function noteMessage(notes: string[], cut?: { thinking: string }): string {
  const said = notes.length > 1 ? notes.map((n, i) => `${i + 1}. ${n}`).join('\n') : (notes[0] ?? '')
  if (!cut) return `[The owner added a note while you were working. Take it into account from here on:]\n${said}`
  const thinking = cut.thinking.trim() ? `\n\nYour thinking so far was:\n${quote(cut.thinking.trim())}` : ''
  const ask = said ? `\n\nThe note:\n${said}` : ''
  return `[The owner stopped your last reply to add a note. Take it into account and carry on from where you were, without starting over.]${thinking}${ask}`
}

/** What the model sees as its own cut-short turn. Never empty, so the roles keep alternating. */
export function cutTurnContent(answer: string): string {
  return answer.trim() || '(The owner stopped me before I had finished thinking.)'
}
