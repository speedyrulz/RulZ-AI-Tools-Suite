// Writing benchmark: the saved runs (and damaged files), the user's own prompts, the run in progress and
// the vote panel's position. The loop is lib/wbenchRunner.ts; this store gives it the models, the chat
// stream and the saved files.
import { create } from 'zustand'
import { toast } from 'sonner'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { StreamEvent } from '@shared/types'
import {
  newRun,
  nextOutput,
  type BenchCandidate,
  type BenchPrompt,
  type BenchRunEntry,
  type BenchVote,
  type WritingBenchRun
} from '@shared/writingBench'
import { aiBusyReason } from '../lib/aiBusy'
import { errorMessage, invoke, on } from '../lib/api'
import { uid } from '../lib/format'
import { once } from '../lib/once'
import { setWbenchRunning } from '../lib/wbenchFlag'
import {
  adaptersLoaded,
  applyVote,
  requestAdapters,
  runProgress,
  runWbenchLoop,
  type BenchReply,
  type BenchReplyInput,
  type BenchReplyResult,
  type WbenchStop
} from '../lib/wbenchRunner'
import { useApp } from './app'
import { readyInstance, useModels } from './models'

export interface NewBenchRun {
  label: string
  prompts: BenchPrompt[]
  candidates: BenchCandidate[]
  seeds: number[]
  maxTokens: number
}

interface WbenchState {
  loaded: boolean
  /** Runs newest first, then the damaged files. */
  entries: BenchRunEntry[]
  /** The user's own prompts (the bundled ones live in code). */
  prompts: BenchPrompt[]
  /** The run in progress. */
  activeId: string | null
  /** The first model of a run is loading. */
  starting: boolean
  /** True from the start of a run to its end: what aiBusy reads. */
  running: boolean
  stopping: WbenchStop | null
  /** The output being written now: its key and the text so far. Never saved. */
  live: { key: string; text: string } | null
  /** The run opened in the view (null = the list). */
  openId: string | null
  /** The pair the vote panel shows; null = the first without a vote. */
  shownPairId: string | null
  init: () => Promise<void>
  open: (id: string | null) => void
  showPair: (id: string | null) => void
  create: (input: NewBenchRun) => Promise<void>
  resume: (id: string) => Promise<void>
  stop: (mode: WbenchStop) => void
  remove: (id: string) => Promise<void>
  vote: (pairId: string, vote: BenchVote) => void
  savePrompts: (prompts: BenchPrompt[]) => Promise<boolean>
}

/** Runs newest first (by creation), damaged files after them. */
export function sortEntries(entries: BenchRunEntry[]): BenchRunEntry[] {
  const ok = entries.filter((e): e is Extract<BenchRunEntry, { ok: true }> => e.ok).sort((a, b) => b.run.createdAt.localeCompare(a.run.createdAt))
  return [...ok, ...entries.filter((e) => !e.ok)]
}

export const entryId = (e: BenchRunEntry): string => (e.ok ? e.run.id : e.id)

// ---------------------------------------------------------------------------------------------
// One reply through chat:start, with its finish reason
// ---------------------------------------------------------------------------------------------

const ABORT_GRACE_MS = 5000
const FLUSH_MS = 90
const streams = new Map<string, (e: StreamEvent) => void>()
let subscribed = false

function ensureSubscribed(): void {
  if (subscribed) return
  subscribed = true
  on('chat:event', (e) => streams.get(e.requestId)?.(e))
}

/** Starts one benchmark reply: prefitted messages, thinking off, the candidate's sampling with the output's seed. */
function startBenchReply(input: BenchReplyInput): BenchReply {
  ensureSubscribed()
  const requestId = uid()
  let finish: (r: BenchReplyResult) => void = () => undefined
  const done = new Promise<BenchReplyResult>((resolve) => (finish = resolve))
  let text = ''
  let ended = false
  let lastFlush = 0
  const end = (r: BenchReplyResult): void => {
    if (ended) return
    ended = true
    streams.delete(requestId)
    input.onText(text)
    finish(r)
  }
  streams.set(requestId, (e) => {
    if (e.type === 'delta') {
      if (!e.content) return
      text += e.content
      // The page repaints on every update: about eleven a second is plenty.
      const now = Date.now()
      if (now - lastFlush >= FLUSH_MS) {
        lastFlush = now
        input.onText(text)
      }
    } else if (e.type === 'error') end({ text, error: e.error, stopped: false, finish: null })
    else if (e.type === 'done') {
      const reason = e.stats.stopReason
      if (reason === 'stopped') return end({ text, error: null, stopped: true, finish: null })
      // The server names the reason; if it did not, a reply that used its whole budget was cut off.
      const finishReason = reason === 'length' ? 'length' : reason === 'stop' ? 'stop' : (e.stats.predictedTokens ?? 0) >= input.maxTokens ? 'length' : 'stop'
      end({ text, error: null, stopped: false, finish: finishReason })
    }
  })
  invoke('chat:start', {
    requestId,
    modelId: input.modelId,
    messages: input.messages,
    sampling: { ...input.sampling, seed: input.seed, maxTokens: input.maxTokens, jsonSchema: '', thinking: 'off' },
    adapters: input.adapters,
    overflow: 'stop',
    disableThinking: true,
    prefitted: true
  }).catch((e) => end({ text, error: errorMessage(e), stopped: false, finish: null }))
  return {
    done,
    abort: () => {
      void invoke('chat:abort', requestId).catch(() => undefined)
      // The server normally ends the stream with a 'stopped' event; if none comes, do not wait for ever.
      setTimeout(() => end({ text, error: null, stopped: true, finish: null }), ABORT_GRACE_MS)
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------------------------

export const useWbench = create<WbenchState>((set, get) => {
  let saveFailed = false

  const flags = (activeId: string | null, starting: boolean): { activeId: string | null; starting: boolean; running: boolean } => {
    const running = activeId !== null || starting
    setWbenchRunning(running)
    return { activeId, starting, running }
  }

  /** Runs whose last save failed. A retry writes the run as it is in memory then (the newest), never an older copy. */
  const unsaved = new Set<string>()
  let retryTimer: ReturnType<typeof setTimeout> | null = null
  const scheduleRetry = (): void => {
    if (retryTimer) return
    retryTimer = setTimeout(() => {
      retryTimer = null
      for (const id of [...unsaved]) {
        const run = runOf(id)
        if (run) void save(run)
        else unsaved.delete(id)
      }
    }, 5000)
  }

  const save = (run: WritingBenchRun): Promise<void> =>
    invoke('wbench:save', run).then(
      () => {
        // Written as it is in memory now: nothing of it is waiting any more.
        if (runOf(run.id) === run) unsaved.delete(run.id)
        if (unsaved.size === 0) saveFailed = false
      },
      (e) => {
        // Once per failure streak: a run saves after every output.
        if (!saveFailed) toast.error(`Could not save the writing benchmark: ${errorMessage(e)} It is tried again.`)
        saveFailed = true
        unsaved.add(run.id)
        scheduleRetry()
      }
    )

  const runOf = (id: string): WritingBenchRun | undefined => {
    const e = get().entries.find((x) => x.ok && x.run.id === id)
    return e?.ok ? e.run : undefined
  }

  /** Changes a run in the list (memory only); undefined if it was removed meanwhile. */
  const patchRun = (id: string, fn: (run: WritingBenchRun) => WritingBenchRun): WritingBenchRun | undefined => {
    const cur = runOf(id)
    if (!cur) return undefined
    const next = fn(cur)
    set((s) => ({ entries: s.entries.map((e) => (e.ok && e.run.id === id ? { ok: true as const, run: next } : e)) }))
    return next
  }

  /** Loads the candidate's model with its adapters if needed; null (after a toast) if it cannot be used. */
  const prepare = async (c: BenchCandidate): Promise<{ adapters: ReturnType<typeof requestAdapters> } | null> => {
    const models = useModels.getState()
    let inst = readyInstance(c.modelId)
    if (!inst && !models.models.some((m) => m.id === c.modelId)) {
      toast.error(`${c.modelName} is no longer in your models folder.`)
      return null
    }
    if (!inst || !adaptersLoaded(inst.adapters, c.adapters)) {
      if (!inst && c.adapters.length === 0) inst = (await models.ensure(c.modelId)) ?? undefined
      else {
        const base = inst?.config ?? useApp.getState().settings?.defaultLoadConfig ?? DEFAULT_LOAD_CONFIG
        inst = (await models.load(c.modelId, { ...base, adapters: c.adapters })) ?? undefined
      }
    }
    if (!inst) return null
    if (!adaptersLoaded(inst.adapters, c.adapters)) {
      toast.error(`${c.label}: an adapter could not be loaded (missing or moved), so its outputs would not be its own.`)
      return null
    }
    return { adapters: requestAdapters(inst.adapters, c.adapters) }
  }

  const runLoop = async (id: string): Promise<void> => {
    const result = await runWbenchLoop({
      now: () => Date.now(),
      sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
      rand: Math.random,
      stopping: () => get().stopping,
      busyReason: () => aiBusyReason('wbench'),
      getRun: () => runOf(id),
      patchRun: (fn) => patchRun(id, fn),
      save,
      prepare,
      startReply: startBenchReply,
      onText: (key, text) => set({ live: key ? { key, text } : null })
    })
    set({ ...flags(null, false), stopping: null, live: null })
    if (result.error) toast.error(`The writing benchmark stopped: ${result.error}`)
    else if (result.busy) toast(`The writing benchmark paused: ${result.busy} Resume it when that is finished.`, { duration: 15000 })
    else if (result.run) {
      const p = runProgress(result.run)
      toast(result.run.status === 'done' ? `Writing benchmark finished: ${p.finished} outputs, ${p.pairs} pairs to vote on.` : 'Writing benchmark stopped. You can resume it.', { duration: 10000 })
    }
  }

  return {
    loaded: false,
    entries: [],
    prompts: [],
    ...flags(null, false),
    stopping: null,
    live: null,
    openId: null,
    shownPairId: null,

    init: once(async () => {
      try {
        const [list, prompts] = await Promise.all([invoke('wbench:list'), invoke('wbench:prompts')])
        // A run the file says is running was cut off: nothing runs on load.
        set((s) => ({
          entries: sortEntries(list.map((e) => (e.ok && e.run.status === 'running' && e.run.id !== s.activeId ? { ok: true as const, run: { ...e.run, status: 'stopped' as const } } : e))),
          prompts,
          loaded: true
        }))
      } catch (e) {
        set({ loaded: true })
        toast.error(`Could not read the writing benchmarks: ${errorMessage(e)}`)
      }
    }),

    open: (openId) => set({ openId, shownPairId: null }),
    showPair: (shownPairId) => set({ shownPairId }),

    create: async (input) => {
      if (get().running || !get().loaded) return
      const busy = aiBusyReason('wbench')
      if (busy) return void toast.error(`${busy} Wait for it to finish first.`)
      if (input.candidates.length < 2 || !input.prompts.length || !input.seeds.length) return
      const run = newRun({ ...input, label: input.label.trim().slice(0, 80) }, new Date().toISOString(), uid())
      const first = { ...run, status: 'running' as const }
      set((s) => ({ entries: sortEntries([{ ok: true as const, run: first }, ...s.entries]), ...flags(run.id, false), stopping: null, openId: run.id, shownPairId: null }))
      void save(first)
      void runLoop(run.id)
    },

    resume: async (id) => {
      if (get().running) return
      const run = runOf(id)
      if (!run || !nextOutput(run)) return
      const busy = aiBusyReason('wbench')
      if (busy) return void toast.error(`${busy} Wait for it to finish first.`)
      if (!patchRun(id, (r) => ({ ...r, status: 'running' }))) return
      set({ ...flags(id, false), stopping: null, openId: id })
      void runLoop(id)
    },

    stop: (mode) => {
      if (!get().activeId) return
      // 'now' outranks 'after'; asking for 'after' later does not take 'now' back.
      set((s) => ({ stopping: s.stopping === 'now' ? 'now' : mode }))
    },

    remove: async (id) => {
      if (get().activeId === id) return
      try {
        await invoke('wbench:delete', id)
        set((s) => ({ entries: s.entries.filter((e) => entryId(e) !== id), openId: s.openId === id ? null : s.openId }))
      } catch (e) {
        toast.error(`Could not delete the writing benchmark: ${errorMessage(e)}`)
      }
    },

    vote: (pairId, vote) => {
      const id = get().openId
      if (!id) return
      const cur = runOf(id)
      if (!cur) return
      const changed = applyVote(cur, pairId, vote, Date.now())
      // A pair is voted once: a second vote (a double key press) changes and saves nothing.
      if (changed === cur) return
      const next = patchRun(id, () => changed)
      if (next) void save(next)
    },

    savePrompts: async (prompts) => {
      if (!get().loaded) return false
      try {
        set({ prompts: await invoke('wbench:savePrompts', prompts) })
        return true
      } catch (e) {
        toast.error(`Could not save your prompts: ${errorMessage(e)}`)
        return false
      }
    }
  }
})
