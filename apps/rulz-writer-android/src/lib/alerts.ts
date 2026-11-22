import type { WritingJob } from '@shared/phone'

// Which writing events become phone notifications. Kept free of the phone's plugins so it can be tested.

export interface AlertPrefs {
  /** A job finished (or didn't). */
  finished: boolean
  /** Keep the connection in the background (needed for notifications while the app is closed). */
  background: boolean
}

export const DEFAULT_ALERT_PREFS: AlertPrefs = { finished: true, background: true }

/** Finished notifications rotate through a few ids, so several can show at once. */
const FINISHED_BASE = 100
const FINISHED_SLOTS = 20

export interface Alert {
  id: number
  channel: 'finished'
  title: string
  body: string
}

const clip = (s: string, n = 160): string => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s)

/** The notification for a finished job; null for one that was stopped on purpose. */
export function alertForJob(job: WritingJob, project: string, count: number): Alert | null {
  if (job.status === 'stopped') return null
  const id = FINISHED_BASE + (count % FINISHED_SLOTS)
  if (job.status === 'error') return { id, channel: 'finished', title: `${job.label} didn’t finish · ${project}`, body: clip(job.error ?? 'Something went wrong on the computer.') }
  return { id, channel: 'finished', title: `${job.label} is ready · ${project}`, body: resultLine(job) }
}

function resultLine(job: WritingJob): string {
  const r = job.result
  if (!r) return 'Open the app to see it.'
  switch (r.kind) {
    case 'prose': {
      const first = r.alts[0]?.generated.replace(/\s+/g, ' ').trim()
      return first ? clip(first) : 'Open the app to see it.'
    }
    case 'summary':
      return clip(r.reply.summary.replace(/\s+/g, ' ').trim())
    case 'chapters':
      return `${r.reply.chapters.length} chapters proposed.`
    case 'scenes':
      return `${r.reply.scenes.length} scenes proposed.`
    case 'check':
      return r.findings.length ? `${r.findings.length} possible problems found.` : 'Nothing found.'
    case 'state':
      return r.items.length ? `${r.items.length} changes to the tracked state proposed.` : 'No changes proposed.'
    case 'interview':
      return clip(r.reply.replace(/\s+/g, ' ').trim())
  }
}

export interface StatusInput {
  computer: string
  link: 'connected' | 'connecting' | 'stopped'
  project?: string
  /** The running job's line, e.g. "Continue · writing 2 of 3". */
  job?: string
}

/** What the ongoing notification says. */
export function statusLine(s: StatusInput): { title: string; text: string } {
  if (s.link === 'connecting') return { title: `Reconnecting to ${s.computer}…`, text: 'I keep trying. Notifications resume once connected.' }
  if (s.link === 'stopped') return { title: `Not connected to ${s.computer}`, text: 'Open the app to see why.' }
  const where = s.project ? ` · ${s.project}` : ''
  if (s.job) return { title: `Writing${where}`, text: s.job }
  return { title: `Connected to ${s.computer}`, text: 'Nothing running. You’ll get a notification when a writing job finishes.' }
}
