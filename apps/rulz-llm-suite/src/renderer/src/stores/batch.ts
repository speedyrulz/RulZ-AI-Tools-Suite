// Practice batch runs: the list of runs (the scoreboard) and the one in progress. The run itself is the practice
// pool in the main process (src/main/re/pool.ts); this store starts and stops it and follows its events.
import { create } from 'zustand'
import { toast } from 'sonner'
import { cleanBatchOptions, summarize, type BatchOptions, type BatchRun, type BatchStop } from '@shared/batch'
import { DEFAULT_BATCH_TARGET, type BatchTarget } from '@shared/remote'
import type { RePracticeTask } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'
import { once } from '../lib/once'
import { useApp } from './app'
import { useChat } from './chat'
import { WBENCH_BUSY, wbenchRunning } from '../lib/wbenchFlag'
import { useModels } from './models'
import { practiceRecords, solvedTasks, useReTraces } from './reTraces'

interface BatchState {
  loaded: boolean
  /** Newest first. */
  runs: BatchRun[]
  /** The run in progress. */
  activeId: string | null
  /** A run is being started: its model is loading on the hosts. */
  starting: boolean
  /** 'after': finish the tasks in progress, then stop. 'now': stop them too. */
  stopping: BatchStop | null
  init: () => Promise<void>
  /** Starts a run where `target` says (this PC, one or more rentals; several tasks at once). Refusals arrive as a toast. */
  start: (tasks: RePracticeTask[], options: BatchOptions, label: string, kitDir: string, target?: BatchTarget) => Promise<void>
  /** Runs the tasks a stopped run has not tried yet, where `target` says (default: where it ran last). */
  resume: (id: string, target?: BatchTarget) => Promise<void>
  stop: (mode: BatchStop) => void
  /** Stops one session of the run (its chat id); its task counts as stopped and the lane takes the next. */
  skip: (conversationId: string) => void
  /** Removes a run from the list. Its chats and recorded data stay. */
  remove: (id: string) => Promise<void>
}

/** A run in the list, newest first. */
function upsert(runs: BatchRun[], run: BatchRun): BatchRun[] {
  const next = runs.some((r) => r.id === run.id) ? runs.map((r) => (r.id === run.id ? run : r)) : [run, ...runs]
  return next.sort((a, b) => b.createdAt - a.createdAt)
}

export const useBatch = create<BatchState>((set, get) => {
  /** Said once when the run in progress ends. */
  const announce = (run: BatchRun): void => {
    const s = summarize(run)
    const head = run.status === 'done' ? 'Batch run finished' : 'Batch run stopped'
    toast(`${head}: ${s.solved} of ${s.scored} solved${s.finished < s.total ? `, ${s.total - s.finished} not run` : ''}.`, { duration: 15000 })
  }

  return {
    loaded: false,
    runs: [],
    activeId: null,
    starting: false,
    stopping: null,

    init: once(async () => {
      on('re:batchChanged', ({ run, active, stopping }) => {
        const wasActive = get().activeId === run.id
        set((s) => ({
          runs: upsert(s.runs, run),
          activeId: active ? run.id : s.activeId === run.id ? null : s.activeId,
          stopping: active ? stopping : s.activeId === run.id ? null : s.stopping
        }))
        if (wasActive && !active) announce(run)
      })
      try {
        const runs = await invoke('re:batches')
        // A run the pool is still running (this window was reloaded) comes back as running.
        set({ runs, loaded: true, activeId: runs.find((r) => r.status === 'running')?.id ?? null })
      } catch (e) {
        set({ loaded: true })
        toast.error(`Could not read the batch runs: ${errorMessage(e)}`)
      }
    }),

    start: async (tasks, options, label, kitDir, target = DEFAULT_BATCH_TARGET) => {
      if (get().activeId || get().starting) return
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      const settings = useApp.getState().settings
      if (!settings) return
      const models = useModels.getState()
      const wanted = settings.re.modelId
      const modelId = wanted && models.models.some((m) => m.id === wanted) ? wanted : models.activeModelId
      if (!modelId) return void toast.error('Load a model first, or choose one under New session.')
      const solved = [...solvedTasks(practiceRecords(useChat.getState().summaries, useReTraces.getState().traces))]
      set({ starting: true, stopping: null })
      try {
        const run = await invoke('re:batchStart', { tasks, options: cleanBatchOptions(options), label, kitDir, modelId, solved, target })
        set((s) => ({ runs: upsert(s.runs, run), activeId: run.id }))
      } catch (e) {
        toast.error(errorMessage(e))
      } finally {
        set({ starting: false })
      }
    },

    resume: async (id, target) => {
      if (get().activeId || get().starting) return
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      const run = get().runs.find((r) => r.id === id)
      if (!run || !run.items.some((i) => i.status === 'pending')) return
      set({ starting: true, stopping: null })
      try {
        const resumed = await invoke('re:batchResume', id, target ?? run.target ?? DEFAULT_BATCH_TARGET)
        set((s) => ({ runs: upsert(s.runs, resumed), activeId: resumed.id }))
      } catch (e) {
        toast.error(errorMessage(e))
      } finally {
        set({ starting: false })
      }
    },

    stop: (mode) => {
      if (!get().activeId) return
      // 'now' outranks 'after'; asking for 'after' later does not take 'now' back.
      set((s) => ({ stopping: s.stopping === 'now' ? 'now' : mode }))
      void invoke('re:batchStop', mode).catch((e) => toast.error(errorMessage(e)))
    },

    skip: (conversationId) => {
      if (!get().activeId) return
      void invoke('re:batchSkip', conversationId).catch((e) => toast.error(errorMessage(e)))
    },

    remove: async (id) => {
      // Not while it runs, nor while a run is starting: that one may be this run being resumed.
      if (get().activeId === id || get().starting) return
      try {
        await invoke('re:deleteBatch', id)
        set((s) => ({ runs: s.runs.filter((r) => r.id !== id) }))
      } catch (e) {
        toast.error(`Could not remove the batch run: ${errorMessage(e)}`)
      }
    }
  }
})
