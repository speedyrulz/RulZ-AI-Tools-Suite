import type { WritingJob, WritingJobStatus } from '@shared/phone'

// The phone's list of writing jobs (kept per project on the computer until dismissed) and how events and
// reconnect listings update it. Pure, so it can be tested.

export type JobMap = Record<string, WritingJob>

export type JobAction =
  /** A `writing:job` event, or the answer to writing:run / writing:job. */
  | { type: 'update'; job: WritingJob }
  /**
   * The answer to writing:jobs(folder): the computer's truth for that folder as of `startedAt` (when the request
   * was sent). Anything that reached the phone after that (`receivedAt`: event or run answer times, `removedAt`:
   * dismissals) is newer than the listing and wins over it.
   */
  | { type: 'load'; folder: string; jobs: WritingJob[]; startedAt?: number; receivedAt?: Record<string, number>; removedAt?: Record<string, number> }
  | { type: 'remove'; id: string }
  | { type: 'clear' }

export const TERMINAL: readonly WritingJobStatus[] = ['done', 'error', 'stopped']

export function isTerminal(status: WritingJobStatus): boolean {
  return TERMINAL.includes(status)
}

export function isRunning(job: WritingJob): boolean {
  return !isTerminal(job.status)
}

export function reduceJobs(state: JobMap, action: JobAction): JobMap {
  switch (action.type) {
    case 'update': {
      const old = state[action.job.id]
      // Streaming events are throttled and can arrive after the final one; a finished job stays finished.
      if (old && isTerminal(old.status) && !isTerminal(action.job.status)) return state
      return { ...state, [action.job.id]: action.job }
    }
    case 'load': {
      const since = action.startedAt ?? Infinity
      const received = action.receivedAt ?? {}
      const removed = action.removedAt ?? {}
      const newer = (id: string): boolean => (received[id] ?? 0) > since
      const next: JobMap = {}
      for (const [id, job] of Object.entries(state)) {
        // Other folders stay; so does a job of this folder that arrived after the listing was asked for.
        if (job.folder !== action.folder || newer(id)) next[id] = job
      }
      for (const job of action.jobs) {
        if ((removed[job.id] ?? 0) > since) continue
        const old = state[job.id]
        if (old && newer(job.id)) continue
        next[job.id] = old && isTerminal(old.status) && !isTerminal(job.status) ? old : job
      }
      return next
    }
    case 'remove': {
      if (!(action.id in state)) return state
      const next = { ...state }
      delete next[action.id]
      return next
    }
    case 'clear':
      return {}
  }
}

/** Jobs of one project, newest first. */
export function jobsOf(state: JobMap, folder: string): WritingJob[] {
  return Object.values(state)
    .filter((j) => j.folder === folder)
    .sort((a, b) => b.startedAt - a.startedAt)
}

export function runningJob(state: JobMap, folder: string): WritingJob | null {
  return jobsOf(state, folder).find(isRunning) ?? null
}

/** What the job's header says: "Continue · writing 2 of 3", "Rewrite · reading the story 40%". */
export function jobLine(job: WritingJob): string {
  switch (job.status) {
    case 'queued':
      return `${job.label} · waiting in line`
    case 'preparing':
      return `${job.label} · getting ready`
    case 'waiting':
      return job.progress !== null ? `${job.label} · reading the story ${Math.round(job.progress * 100)}%` : `${job.label} · reading the story`
    case 'writing':
      return job.of > 1 ? `${job.label} · writing ${job.alt} of ${job.of}` : `${job.label} · writing`
    case 'done':
      return `${job.label} · done`
    case 'stopped':
      return `${job.label} · stopped`
    case 'error':
      return `${job.label} · didn’t finish`
  }
}

/** Jobs that finished since `before` (a map of what was seen), for notifications. */
export function newlyFinished(before: JobMap, after: JobMap): WritingJob[] {
  return Object.values(after).filter((j) => isTerminal(j.status) && (!before[j.id] || !isTerminal(before[j.id].status)))
}
