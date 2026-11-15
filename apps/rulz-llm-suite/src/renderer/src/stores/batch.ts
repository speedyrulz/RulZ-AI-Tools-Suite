// Practice batch runs: the list of runs (the scoreboard) and the one in progress. The loop itself
// is lib/batchRunner.ts; this store gives it the chat, the models and the saved runs.
import { create } from 'zustand'
import { toast } from 'sonner'
import { cleanBatchOptions, planBatch, summarize, type BatchOptions, type BatchRun } from '@shared/batch'
import type { RePracticeTask } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { runBatchLoop, type BatchStop } from '../lib/batchRunner'
import { uid } from '../lib/format'
import { once } from '../lib/once'
import { useApp } from './app'
import { isChatBusy, isGrading, onTurnEnd, useChat } from './chat'
import { readyInstance, useModels } from './models'

interface BatchState {
  loaded: boolean
  /** Newest first. */
  runs: BatchRun[]
  /** The run in progress. */
  activeId: string | null
  /** The model for a new or resumed run is loading. */
  starting: boolean
  /** 'after': finish the task in progress, then stop. 'now': stop that task too. */
  stopping: BatchStop | null
  init: () => Promise<void>
  start: (tasks: RePracticeTask[], options: BatchOptions, label: string, kitDir: string) => Promise<void>
  /** Runs the tasks a stopped run has not tried yet. */
  resume: (id: string) => Promise<void>
  stop: (mode: BatchStop) => void
  /** Removes a run from the list. Its chats and recorded data stay. */
  remove: (id: string) => Promise<void>
}

/** Tasks a session in the chat list has solved. */
export function solvedTasks(summaries: { rePracticeTask?: string; reOutcome?: string }[]): Set<string> {
  const out = new Set<string>()
  for (const s of summaries) if (s.rePracticeTask && s.reOutcome === 'solved') out.add(s.rePracticeTask)
  return out
}

export const useBatch = create<BatchState>((set, get) => {
  let saveFailed = false

  const save = (run: BatchRun): void => {
    void invoke('re:saveBatch', run).then(
      () => {
        saveFailed = false
      },
      (e) => {
        // Once per failure streak: a run saves after every task.
        if (!saveFailed) toast.error(`Could not save the batch run: ${errorMessage(e)}`)
        saveFailed = true
      }
    )
  }

  /** Changes a run in the list and saves it; undefined if it was removed meanwhile. */
  const patchRun = (id: string, fn: (run: BatchRun) => BatchRun): BatchRun | undefined => {
    const cur = get().runs.find((r) => r.id === id)
    if (!cur) return undefined
    const next = fn(cur)
    set((s) => ({ runs: s.runs.map((r) => (r.id === id ? next : r)) }))
    save(next)
    return next
  }

  /** Loads the run's model if needed; null (after a toast) if it cannot be used. */
  const readyModel = async (modelId: string | null): Promise<{ modelName: string; contextLength: number | null } | null> => {
    if (!modelId) {
      toast.error('Load a model first, or choose one under New session.')
      return null
    }
    if (!useModels.getState().models.some((m) => m.id === modelId) && !readyInstance(modelId)) {
      toast.error(`${modelId} is no longer in your models folder.`)
      return null
    }
    // Messages go to the active model, so the run's model becomes it.
    useModels.getState().setActive(modelId)
    const inst = readyInstance(modelId) ?? (await useModels.getState().ensure(modelId))
    return inst ? { modelName: inst.modelName, contextLength: inst.contextLength } : null
  }

  const runBatch = async (runId: string): Promise<void> => {
    const { run, error } = await runBatchLoop({
      now: () => Date.now(),
      sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
      stopping: () => get().stopping,
      chatBusy: () => isChatBusy(useChat.getState()),
      chatIdle: () => {
        const c = useChat.getState()
        return !isChatBusy(c) && !c.handoff
      },
      replyingIn: () => useChat.getState().generation?.conversationId ?? null,
      settling: isGrading,
      currentChat: () => useChat.getState().current?.id ?? null,
      stopReply: () => useChat.getState().stop(),
      prepare: (kitDir, taskId) => invoke('re:practicePrepare', kitDir, taskId),
      startSession: (o) =>
        useChat.getState().startRe({ ...o, collect: useApp.getState().settings?.re.collect ?? true, background: true }),
      attended: () => useChat.getState().setUnattended(null),
      failPractice: (id, notes) => useChat.getState().failRePractice(id, notes),
      snapshot: (id) => useChat.getState().snapshot(id),
      onTurnEnd,
      getRun: () => get().runs.find((r) => r.id === runId),
      patchRun: (fn) => patchRun(runId, fn)
    })
    set({ activeId: null, stopping: null })
    if (error) toast.error(`The batch run stopped: ${error}`)
    if (run) {
      const s = summarize(run)
      const head = run.status === 'done' ? 'Batch run finished' : 'Batch run stopped'
      toast(`${head}: ${s.solved} of ${s.scored} solved${s.finished < s.total ? `, ${s.total - s.finished} not run` : ''}.`, { duration: 15000 })
    }
  }

  return {
    loaded: false,
    runs: [],
    activeId: null,
    starting: false,
    stopping: null,

    init: once(async () => {
      try {
        set({ runs: await invoke('re:batches'), loaded: true })
      } catch (e) {
        set({ loaded: true })
        toast.error(`Could not read the batch runs: ${errorMessage(e)}`)
      }
    }),

    start: async (tasks, options, label, kitDir) => {
      if (get().activeId || get().starting) return
      const settings = useApp.getState().settings
      if (!settings) return
      const opts = cleanBatchOptions(options)
      const items = planBatch(tasks, opts, solvedTasks(useChat.getState().summaries))
      if (!items.length) return void toast('No practice tasks match this selection.')
      const models = useModels.getState()
      const wanted = settings.re.modelId
      const modelId = wanted && models.models.some((m) => m.id === wanted) ? wanted : models.activeModelId
      set({ starting: true, stopping: null })
      let model: Awaited<ReturnType<typeof readyModel>>
      try {
        model = await readyModel(modelId)
      } finally {
        set({ starting: false })
      }
      if (!model || get().activeId) return
      const run: BatchRun = {
        id: uid(),
        label: label.trim().slice(0, 80),
        createdAt: Date.now(),
        status: 'running',
        modelId,
        modelName: model.modelName,
        contextLength: model.contextLength,
        maxToolRounds: settings.re.maxToolRounds,
        kitDir,
        options: opts,
        items
      }
      set((s) => ({ runs: [run, ...s.runs], activeId: run.id }))
      save(run)
      void runBatch(run.id)
    },

    resume: async (id) => {
      if (get().activeId || get().starting) return
      const run = get().runs.find((r) => r.id === id)
      if (!run || !run.items.some((i) => i.status === 'pending')) return
      set({ starting: true, stopping: null })
      let model: Awaited<ReturnType<typeof readyModel>>
      try {
        model = await readyModel(run.modelId)
      } finally {
        set({ starting: false })
      }
      if (!model || get().activeId) return
      if (!patchRun(id, (r) => ({ ...r, status: 'running', finishedAt: undefined }))) return
      set({ activeId: id })
      void runBatch(id)
    },

    stop: (mode) => {
      if (!get().activeId) return
      // 'now' outranks 'after'; asking for 'after' later does not take 'now' back.
      set((s) => ({ stopping: s.stopping === 'now' ? 'now' : mode }))
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
