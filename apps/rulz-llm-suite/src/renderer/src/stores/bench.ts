// Load-settings benchmark runs: the saved list, kept current by bench:changed, and start/stop/delete.
// The run itself happens in the main process.
import { create } from 'zustand'
import { toast } from 'sonner'
import type { BenchmarkRequest, BenchmarkRun } from '@shared/benchmark'
import { insertRunIfAbsent, mergeRun } from '../lib/bench'
import { errorMessage, invoke, on } from '../lib/api'
import { once } from '../lib/once'
import { WBENCH_BUSY, wbenchRunning } from '../lib/wbenchFlag'

interface BenchState {
  loaded: boolean
  /** Newest first. */
  runs: BenchmarkRun[]
  /** bench:start is being answered. */
  starting: boolean
  init: () => Promise<void>
  /** The new run, or null (after a toast) when it was refused. */
  start: (req: BenchmarkRequest) => Promise<BenchmarkRun | null>
  stop: () => Promise<void>
  remove: (runId: string) => Promise<void>
}

/** The run in progress, if any (one at a time). */
export function activeBenchRun(runs: BenchmarkRun[]): BenchmarkRun | undefined {
  return runs.find((r) => r.status === 'running')
}

export const useBench = create<BenchState>((set, get) => ({
  loaded: false,
  runs: [],
  starting: false,

  init: once(async () => {
    // An event that arrives before the list is newer than the list's copy of that run, so merge.
    on('bench:changed', (run) => set((s) => ({ runs: mergeRun(s.runs, run) })))
    try {
      const list = await invoke('bench:list')
      set((s) => ({ runs: list.reduce(insertRunIfAbsent, s.runs), loaded: true }))
    } catch (e) {
      set({ loaded: true })
      toast.error(`Could not read the benchmark runs: ${errorMessage(e)}`)
    }
  }),

  start: async (req) => {
    if (get().starting || activeBenchRun(get().runs)) return null
    if (wbenchRunning()) {
      toast(WBENCH_BUSY)
      return null
    }
    set({ starting: true })
    try {
      const run = await invoke('bench:start', req)
      set((s) => ({ runs: insertRunIfAbsent(s.runs, run) }))
      return run
    } catch (e) {
      toast.error(errorMessage(e))
      return null
    } finally {
      set({ starting: false })
    }
  },

  stop: async () => {
    try {
      await invoke('bench:stop')
    } catch (e) {
      toast.error(`Could not stop the benchmark: ${errorMessage(e)}`)
    }
  },

  remove: async (runId) => {
    try {
      await invoke('bench:delete', runId)
      set((s) => ({ runs: s.runs.filter((r) => r.id !== runId) }))
    } catch (e) {
      toast.error(`Could not delete the benchmark run: ${errorMessage(e)}`)
    }
  }
}))
