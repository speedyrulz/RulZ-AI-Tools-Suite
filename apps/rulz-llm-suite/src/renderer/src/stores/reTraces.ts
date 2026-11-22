// Recorded RE sessions, including the ones other apps (RulZ Coding Agent and its phone app) ran
// through the API server. Those have no chat here, so the practice counts need this list too.
import { create } from 'zustand'
import type { ReTraceInfo } from '@shared/types'
import { invoke, on } from '../lib/api'

interface ReTracesState {
  traces: ReTraceInfo[]
  refresh: () => void
  /** Reads the list and follows changes; safe to call more than once. */
  init: () => void
}

let listening = false

export const useReTraces = create<ReTracesState>((set, get) => ({
  traces: [],
  refresh: () => {
    void invoke('re:traces')
      .then((traces) => set({ traces }))
      .catch(() => undefined)
  },
  init: () => {
    get().refresh()
    if (listening) return
    listening = true
    on('re:changed', () => get().refresh())
  }
}))

/** One practice session: which task, and how it was graded. */
export interface PracticeRecord {
  taskId: string
  outcome: string
}

/** Tasks a practice session (here or in another app) has solved. */
export function solvedTasks(records: PracticeRecord[]): Set<string> {
  const out = new Set<string>()
  for (const r of records) if (r.outcome === 'solved') out.add(r.taskId)
  return out
}

/** Practice sessions from the chat list and from other apps' recorded sessions. */
export function practiceRecords(
  summaries: { id: string; rePracticeTask?: string; reOutcome?: string }[],
  traces: Pick<ReTraceInfo, 'conversationId' | 'api'>[]
): PracticeRecord[] {
  const out: PracticeRecord[] = []
  const chatIds = new Set<string>()
  for (const s of summaries) {
    chatIds.add(s.id)
    if (s.rePracticeTask) out.push({ taskId: s.rePracticeTask, outcome: s.reOutcome ?? 'unrated' })
  }
  for (const t of traces) {
    const p = t.api?.practice
    if (p && !chatIds.has(t.conversationId)) out.push({ taskId: p.taskId, outcome: t.api!.outcome })
  }
  return out
}
