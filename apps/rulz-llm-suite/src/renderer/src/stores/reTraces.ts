// Recorded RE sessions, including the ones other apps (RulZ Coding Agent and its phone app) ran
// through the API server. Those have no chat here, so the practice counts need this list too.
import { create } from 'zustand'
import { keyOfModel, modelKey, practiceOf, practiceRecords, solvedTasks, taskRecords } from '@shared/practice'
import type { PracticeRecord, PracticeSources, TaskRecord } from '@shared/practice'
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

// The practice helpers live in src/shared/practice.ts; re-exported for the pages that already use them from here.
export { keyOfModel, modelKey, practiceOf, practiceRecords, solvedTasks, taskRecords }
export type { PracticeRecord, PracticeSources, TaskRecord }
