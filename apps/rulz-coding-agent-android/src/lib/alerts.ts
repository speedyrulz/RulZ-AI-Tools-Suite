import type { AgentEvent, PendingPrompt, TimelineItem } from '@shared/types'

// Which agent events become phone notifications. Kept free of the phone's plugins so it can be tested.

export interface AlertPrefs {
  /** A task finished (or stopped with an error). */
  finished: boolean
  /** The agent waits for the owner: a permission request, a question, or a pause. */
  waiting: boolean
  /** Keep the connection in the background (needed for both while the app is closed). */
  background: boolean
}

export const DEFAULT_ALERT_PREFS: AlertPrefs = { finished: true, waiting: true, background: true }

/** Only one "waiting for you" notification at a time: a newer one replaces it, and an answer removes it. */
export const WAITING_ID = 1
/** Finished notifications rotate through a few ids, so several can show at once. */
const FINISHED_BASE = 100
const FINISHED_SLOTS = 20

export type AlertChannel = 'waiting' | 'finished'

export type AlertAction =
  | { kind: 'show'; id: number; channel: AlertChannel; title: string; body: string }
  | { kind: 'cancel'; id: number }

export interface AlertContext {
  prefs: AlertPrefs
  projectName: string
  /** The conversation so far, for the agent's last words in a finished task. */
  timeline: TimelineItem[]
  /** Counts finished notifications, for their ids. */
  finishedCount: number
}

const clip = (s: string, n = 180): string => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s)

/** Markdown to one line of plain text. */
export function plainText(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[*_~#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export interface StatusInput {
  computer: string
  link: 'connected' | 'connecting' | 'stopped'
  project?: string
  running: boolean
  /** The running task's title. */
  task?: string
  /** What the agent says it is doing ("Installing the pieces your app needs"). */
  doing?: string
  prompt: PendingPrompt | null
}

/** What the ongoing notification says: what the agent is doing right now. */
export function statusLine(s: StatusInput): { title: string; text: string } {
  if (s.link === 'connecting') return { title: `Reconnecting to ${s.computer}…`, text: 'I keep trying. Notifications resume once connected.' }
  if (s.link === 'stopped') return { title: `Not connected to ${s.computer}`, text: 'Open the app to see why.' }
  const where = s.project ? ` · ${s.project}` : ''
  const p = s.prompt
  if (p?.kind === 'approval') return { title: `Waiting for your OK${where}`, text: clip(plainText(p.action), 200) }
  if (p?.kind === 'question') return { title: `Question for you${where}`, text: clip(plainText(p.question), 200) }
  if (p?.kind === 'stuck') return { title: `Paused${where}`, text: clip(plainText(p.message), 200) }
  if (s.running) return { title: `Working on “${clip(s.task ?? 'your task', 50)}”`, text: clip(s.doing || 'Working…', 200) }
  return { title: `Connected to ${s.computer}`, text: 'Nothing running. You’ll get a notification when a task finishes or needs you.' }
}

export function alertFor(e: AgentEvent, ctx: AlertContext): AlertAction[] {
  const project = ctx.projectName
  if (e.type === 'prompt') {
    if (!e.prompt) return [{ kind: 'cancel', id: WAITING_ID }]
    if (!ctx.prefs.waiting) return []
    const p = e.prompt
    const [title, body] =
      p.kind === 'approval'
        ? [`Permission needed · ${project}`, p.risk ? `${p.action}\n${p.risk}` : p.action]
        : p.kind === 'question'
          ? [`Question · ${project}`, p.question]
          : [`Paused · ${project}`, p.message]
    return [{ kind: 'show', id: WAITING_ID, channel: 'waiting', title, body: clip(plainText(body), 300) }]
  }
  if (e.type === 'task-end') {
    const actions: AlertAction[] = [{ kind: 'cancel', id: WAITING_ID }]
    // Stopped tasks were stopped by someone on purpose.
    if (!ctx.prefs.finished || e.task.outcome === 'stopped' || e.task.outcome === 'running') return actions
    const said = [...ctx.timeline].reverse().find((i) => i.kind === 'assistant' && i.taskId === e.task.id && i.text.trim())
    const failed = e.task.outcome === 'error'
    const title = `${failed ? 'Didn’t finish' : 'Done'}: ${clip(e.task.title, 60)}`
    const body = said ? clip(plainText(said.text)) : failed ? `Something went wrong in ${project}.` : `Finished in ${project}.`
    actions.push({ kind: 'show', id: FINISHED_BASE + (ctx.finishedCount % FINISHED_SLOTS), channel: 'finished', title, body })
    return actions
  }
  return []
}
