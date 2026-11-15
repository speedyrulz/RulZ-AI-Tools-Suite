// The agent's own checklist for a task (the set_plan tool): reading what the model sent, forgiving
// the usual variations, and the texts that go back to the model.
import type { PlanItem, WireMessage } from '@shared/types'

export const PLAN_TOOL = 'set_plan'
/** More steps than this are a to-do dump, not a plan; the rest are left out. */
export const MAX_PLAN_ITEMS = 20
const MAX_ITEM_CHARS = 160

const DONE = new Set(['done', 'complete', 'completed', 'finished', 'x', 'checked', 'true', 'yes'])
const DOING = new Set(['doing', 'in_progress', 'in-progress', 'in progress', 'inprogress', 'active', 'current', 'working', 'started', 'now', 'ongoing'])

function statusOf(value: unknown): PlanItem['status'] {
  if (value === true) return 'done'
  const s = typeof value === 'string' ? value.trim().toLowerCase() : ''
  return DONE.has(s) ? 'done' : DOING.has(s) ? 'doing' : 'todo'
}

const TODO = new Set(['', 'todo', 'to do', 'pending', 'not started', 'open', 'next', 'later'])

/** What a box at the start of a line says: "[x]", "[>]", "[ ]", or a word ("[done]"). Null when it isn't a box ("[UI] ..."). */
function boxStatus(box: string): PlanItem['status'] | null {
  const b = box.trim().toLowerCase()
  if (TODO.has(b)) return 'todo'
  if (DONE.has(b)) return 'done'
  if (DOING.has(b) || /^[>~/-]+$/.test(b)) return 'doing'
  return null
}

/** "[x] Build the page", "- [ ] Add the buttons" and "3. Check it" → the text and what the box says. */
function fromLine(line: string): PlanItem | null {
  const rest = line.replace(/^\s*(?:[-*•]\s+|\d+[.)]\s+)?/, '')
  const m = /^\[([^\]\n]{0,12})\]\s*(.*)$/.exec(rest)
  const status = m ? boxStatus(m[1]) : null
  const text = (status !== null && m ? m[2] : rest).trim()
  return text ? { text, status: status ?? 'todo' } : null
}

function fromEntry(entry: unknown): PlanItem | null {
  if (typeof entry === 'string') return fromLine(entry)
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return null
  const e = entry as Record<string, unknown>
  const raw = [e.text, e.step, e.title, e.content, e.task, e.description, e.name].find((v) => typeof v === 'string' && v.trim())
  if (typeof raw !== 'string') return null
  const flagged = e.done === true || e.completed === true || e.checked === true
  return { text: raw.trim(), status: flagged ? 'done' : statusOf(e.status ?? e.state) }
}

const clip = (text: string): string => {
  const one = text.replace(/\s+/g, ' ').trim()
  return one.length > MAX_ITEM_CHARS ? `${one.slice(0, MAX_ITEM_CHARS - 1)}…` : one
}

/**
 * Reads set_plan's arguments: one text with a line per step. Also accepts what models trained on
 * other tools send: the list under a few other names, as a list of lines or of objects, and their
 * status words ("pending", "in_progress", "completed").
 */
export function parsePlan(args: Record<string, unknown>): { plan: PlanItem[]; dropped: number } | { error: string } {
  let list: unknown = args.steps ?? args.plan ?? args.items ?? args.todos ?? args.tasks ?? args.checklist
  if (typeof list === 'string') {
    const text = list.trim()
    try {
      const parsed: unknown = JSON.parse(text)
      list = Array.isArray(parsed) ? parsed : text.split(/\r?\n/)
    } catch {
      list = text.split(/\r?\n/)
    }
  }
  if (!Array.isArray(list)) {
    return { error: 'set_plan needs "steps": the checklist as text, one step per line, like "[x] Build the main page\\n[>] Add the buttons\\n[ ] Check that it works".' }
  }
  const all = list.flatMap((entry) => {
    const item = fromEntry(entry)
    return item ? [{ text: clip(item.text), status: item.status }] : []
  })
  if (!all.length) return { error: 'set_plan needs at least one step with some text.' }
  return { plan: all.slice(0, MAX_PLAN_ITEMS), dropped: Math.max(0, all.length - MAX_PLAN_ITEMS) }
}

/** What set_plan answers: short, since the model sent the list itself. */
export function planResult(plan: PlanItem[], dropped = 0): string {
  const done = plan.filter((p) => p.status === 'done').length
  const now = plan.find((p) => p.status === 'doing') ?? plan.find((p) => p.status === 'todo')
  const where = done === plan.length ? `All ${plan.length} steps are done.` : `${done} of ${plan.length} steps done.${now ? ` ${now.status === 'doing' ? 'Now' : 'Next'}: ${now.text}` : ''}`
  const extra = dropped ? ` (Only the first ${MAX_PLAN_ITEMS} steps are kept: keep the checklist short.)` : ''
  return `Checklist saved. ${where}${extra}`
}

const BOX: Record<PlanItem['status'], string> = { done: '[x]', doing: '[>]', todo: '[ ]' }

const PLAN_NOTE_START = '[Note: Your checklist for this task, from before your notes were tidied up'
const PLAN_NOTE_END = `Carry on from there, and keep it up to date with ${PLAN_TOOL}.]`

/** The checklist as a note for the model, for when the call that held it is no longer in the conversation. */
export function planNote(plan: PlanItem[]): string {
  return (
    `${PLAN_NOTE_START} ([x] done, [>] the step you were on, [ ] still to do):\n` +
    `${plan.map((p) => `${BOX[p.status]} ${p.text}`).join('\n')}\n` +
    PLAN_NOTE_END
  )
}

/**
 * Takes the checklist notes out of the conversation, once their task is over: left in, they would
 * tell the model in the next task to carry on with a checklist that isn't that task's. Anything
 * else in the same message (a note about background commands added later) is kept.
 */
export function dropPlanNotes(messages: WireMessage[]): void {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role !== 'user' || typeof m.content !== 'string') continue
    let text = m.content
    for (let at = text.indexOf(PLAN_NOTE_START); at >= 0; at = text.indexOf(PLAN_NOTE_START)) {
      const end = text.indexOf(PLAN_NOTE_END, at)
      if (end < 0) break
      text = `${text.slice(0, at).trimEnd()}\n\n${text.slice(end + PLAN_NOTE_END.length).trimStart()}`.trim()
    }
    if (text === m.content) continue
    // A message with a task's request or a timeline item stays, whatever is left of it.
    if (text || m.task || m.item) m.content = text
    else messages.splice(i, 1)
  }
}

/** Whether the model can still see its checklist in these messages: its last set_plan call, or a note that repeats it. */
export function planInHistory(messages: WireMessage[]): boolean {
  return messages.some(
    (m) =>
      (m.role === 'assistant' && !!m.tool_calls?.some((c) => c.function.name === PLAN_TOOL)) ||
      (m.role === 'user' && m.content.includes(PLAN_NOTE_START))
  )
}
