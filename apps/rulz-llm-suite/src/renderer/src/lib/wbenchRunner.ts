// The loop of a writing benchmark run: one output after another, grouped by candidate, each saved when
// it is finished. Everything it touches comes in through `WbenchDeps`, so it runs in tests without the
// app. Also the small pure helpers the store and the views share (adapters for a request, what a vote
// panel shows, progress).
import type { InstanceAdapter, LoadAdapter, LoadConfig, SamplingParams } from '@shared/types'
import {
  addPairs,
  benchMessages,
  nextOutput,
  type BenchCandidate,
  type BenchFinish,
  type BenchOutput,
  type BenchPair,
  type BenchVote,
  type WritingBenchRun
} from '@shared/writingBench'

export type WbenchStop = 'after' | 'now'

export interface BenchReplyResult {
  text: string
  error: string | null
  /** The reply was aborted. */
  stopped: boolean
  /** How the model ended it; null when the chat path did not say. */
  finish: 'stop' | 'length' | null
}

export interface BenchReply {
  done: Promise<BenchReplyResult>
  abort: () => void
}

export interface BenchReplyInput {
  modelId: string
  messages: { role: 'system' | 'user'; content: string }[]
  sampling: SamplingParams
  adapters: LoadAdapter[] | null
  seed: number
  maxTokens: number
  onText: (text: string) => void
}

export interface WbenchDeps {
  now: () => number
  sleep: (ms: number) => Promise<void>
  rand: () => number
  /** 'after': finish the output in progress, then stop. 'now': abort it too. */
  stopping: () => WbenchStop | null
  /** Why something else is using the model now, or null. */
  busyReason: () => string | null
  /** The run as it is now; undefined once it was removed. */
  getRun: () => WritingBenchRun | undefined
  /** Changes the run in memory; undefined once it was removed. */
  patchRun: (fn: (run: WritingBenchRun) => WritingBenchRun) => WritingBenchRun | undefined
  /** Writes the run (wbench:save). The loop waits for it and carries on if it fails. */
  save: (run: WritingBenchRun) => Promise<void>
  /**
   * Makes the candidate's model ready with its adapters. Returns the adapter strengths to send with
   * each request, or null when it could not be used (the user was told).
   */
  prepare: (candidate: BenchCandidate) => Promise<{ adapters: LoadAdapter[] | null } | null>
  startReply: (input: BenchReplyInput) => BenchReply
  /** The text of the output being written (key null: nothing is being written). */
  onText: (key: string | null, text: string) => void
}

export interface WbenchLoopResult {
  run: WritingBenchRun | undefined
  /** The loop stopped because something else is using the model. */
  busy?: string
  error?: string
}

const WATCH_MS = 100

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

const iso = (ms: number): string => new Date(ms).toISOString()

/**
 * Runs the run's pending outputs until none are left, a stop is asked for, something else needs the
 * model, or the model cannot be loaded; then marks the run done or stopped. Never throws.
 */
export async function runWbenchLoop(deps: WbenchDeps): Promise<WbenchLoopResult> {
  let busy: string | undefined
  let error: string | undefined
  /** The candidate whose model is ready, and what to send with its requests. */
  const state: { ready: { id: string; adapters: LoadAdapter[] | null } | null } = { ready: null }

  const saveRun = async (run: WritingBenchRun | undefined): Promise<void> => {
    if (!run) return
    try {
      await deps.save(run)
    } catch {
      // The store tells the user; the loop goes on.
    }
  }

  try {
    for (;;) {
      if (deps.stopping()) break
      const run = deps.getRun()
      if (!run) break
      const out = nextOutput(run)
      if (!out) break
      const reason = deps.busyReason()
      if (reason) {
        busy = reason
        break
      }
      const candidate = run.candidates.find((c) => c.id === out.candidateId)
      const prompt = run.prompts.find((p) => p.id === out.promptId)
      if (!candidate || !prompt) {
        const run2 = finish(deps, out.key, { status: 'failed', finish: 'error', error: 'The run no longer has this prompt or candidate.' })
        await saveRun(run2)
        continue
      }
      if (state.ready?.id !== candidate.id) {
        state.ready = null
        const p = await deps.prepare(candidate)
        if (deps.stopping()) break
        if (!p) {
          error = `Could not load ${candidate.modelName} for ${candidate.label}.`
          break
        }
        state.ready = { id: candidate.id, adapters: p.adapters }
        // The model took a while to load: look again before using it.
        const again = deps.busyReason()
        if (again) {
          busy = again
          break
        }
      }

      const started = deps.now()
      const reply = deps.startReply({
        modelId: candidate.modelId,
        messages: benchMessages(prompt),
        sampling: candidate.sampling,
        adapters: state.ready?.adapters ?? null,
        seed: out.seed,
        maxTokens: run.maxTokens,
        onText: (t) => deps.onText(out.key, t)
      })
      let settled = false
      let aborted = false
      const watcher = (async (): Promise<void> => {
        while (!settled) {
          if (!aborted && deps.stopping() === 'now') {
            aborted = true
            reply.abort()
          }
          await deps.sleep(WATCH_MS)
        }
      })()
      let res: BenchReplyResult
      try {
        res = await reply.done
      } catch (e) {
        res = { text: '', error: errorText(e), stopped: false, finish: null }
      }
      settled = true
      await watcher
      deps.onText(null, '')
      const ms = Math.max(0, deps.now() - started)

      if (res.error) {
        await saveRun(finish(deps, out.key, { status: 'failed', text: res.text, finish: 'error', error: res.error, ms }))
      } else if (res.stopped && deps.stopping() === 'now') {
        // "Stop now" cut it off: it has not had its try, so a resumed run starts it again.
        deps.patchRun((r) => touch(deps, { ...r, outputs: r.outputs.map((o) => (o.key === out.key ? { ...o, status: 'pending', text: '', finish: null, error: null, ms: null } : o)) }))
        break
      } else if (res.stopped) {
        await saveRun(finish(deps, out.key, { status: 'failed', text: res.text, finish: 'stopped', error: 'The reply was stopped.', ms }))
      } else {
        await saveRun(finish(deps, out.key, { status: 'done', text: res.text, finish: res.finish ?? 'stop', error: null, ms }))
      }
    }
  } catch (e) {
    error = errorText(e)
  }
  deps.onText(null, '')
  const run = deps.patchRun((r) => touch(deps, { ...r, status: nextOutput(r) ? 'stopped' : 'done' }))
  await saveRun(run)
  return { run, busy, error }
}

function touch(deps: Pick<WbenchDeps, 'now'>, run: WritingBenchRun): WritingBenchRun {
  return { ...run, updatedAt: iso(deps.now()) }
}

/** Records how one output ended and adds the pairs it makes possible. */
function finish(deps: WbenchDeps, key: string, patch: Partial<BenchOutput>): WritingBenchRun | undefined {
  return deps.patchRun((r) =>
    addPairs(
      touch(deps, { ...r, outputs: r.outputs.map((o) => (o.key === key ? { ...o, ...patch } : o)) }),
      deps.rand
    )
  )
}

// ---------------------------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------------------------

/** The adapters a candidate's requests carry, given the adapters the model's instance has loaded. */
export function requestAdapters(loaded: InstanceAdapter[], wanted: LoadAdapter[]): LoadAdapter[] | null {
  if (!loaded.length) return null
  return loaded.map((a) => ({ id: a.id, scale: wanted.find((w) => w.id === a.id)?.scale ?? 0 }))
}

/** Whether a loaded instance already has every adapter the candidate wants (so it need not be loaded again). */
export function adaptersLoaded(loaded: InstanceAdapter[], wanted: LoadAdapter[]): boolean {
  return wanted.every((w) => loaded.some((a) => a.id === w.id))
}

/** A value with object keys in sorted order, so two configs compare by content, not by key order. */
function sortedJson(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(sortedJson).join(',')}]`
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>
    return `{${Object.keys(o).sort().filter((k) => o[k] !== undefined).map((k) => `${JSON.stringify(k)}:${sortedJson(o[k])}`).join(',')}}`
  }
  return JSON.stringify(v) ?? 'null'
}

/** Whether two load configs are the same apart from their adapters (which a benchmark candidate sets on its own). */
export function sameLoadConfig(a: LoadConfig, b: LoadConfig): boolean {
  const { adapters: _a, ...x } = a
  const { adapters: _b, ...y } = b
  return sortedJson(x) === sortedJson(y)
}

/** The sampling and load settings a candidate is saved with: the edited values, else the select's choice and the model's saved load settings. */
export function candidateSettings(edited: { sampling: SamplingParams | null; load: LoadConfig | null }, chosenSampling: SamplingParams): { sampling: SamplingParams; load: LoadConfig | null } {
  return { sampling: edited.sampling ?? chosenSampling, load: edited.load ? { ...edited.load, adapters: [] } : null }
}

export interface RunProgress {
  /** Outputs that are done or failed. */
  finished: number
  /** Outputs that failed. */
  failed: number
  total: number
  /** Pairs with a vote (a skip counts as voted). */
  voted: number
  pairs: number
}

export function runProgress(run: WritingBenchRun): RunProgress {
  const failed = run.outputs.filter((o) => o.status === 'failed').length
  const finished = run.outputs.filter((o) => o.status === 'done' || o.status === 'failed').length
  return { finished, failed, total: run.outputs.length, voted: run.pairs.filter((p) => p.vote !== null).length, pairs: run.pairs.length }
}

/** The run with one vote recorded. A pair is voted once; a second vote, or an unknown pair, changes nothing. */
export function applyVote(run: WritingBenchRun, pairId: string, vote: BenchVote, nowMs: number): WritingBenchRun {
  const pair = run.pairs.find((p) => p.id === pairId)
  if (!pair || pair.vote !== null) return run
  // Votes count in votedAt order, so each is later than every earlier one even within one millisecond.
  const last = run.pairs.reduce((m, p) => (p.votedAt ? Math.max(m, Date.parse(p.votedAt) || 0) : m), 0)
  const at = iso(Math.max(nowMs, last + 1))
  return {
    ...run,
    updatedAt: at,
    pairs: run.pairs.map((p) => (p.id === pairId ? { ...p, vote, votedAt: at } : p))
  }
}

export interface PairSide {
  /** 'A' or 'B'. */
  letter: 'A' | 'B'
  text: string
  finish: BenchFinish | null
  /** The candidate's label: only once the pair is voted, never before. */
  candidate: string | null
}

export interface PairView {
  pair: BenchPair
  promptTitle: string
  promptText: string
  a: PairSide
  b: PairSide
  voted: boolean
  /** The user's vote, once voted. */
  vote: BenchVote | null
}

/** What the vote panel shows for a pair. Candidate names are left out until the pair has a vote. */
export function pairView(run: WritingBenchRun, pair: BenchPair): PairView {
  const voted = pair.vote !== null
  const side = (letter: 'A' | 'B', key: string): PairSide => {
    const o = run.outputs.find((x) => x.key === key)
    const c = o ? run.candidates.find((x) => x.id === o.candidateId) : undefined
    return { letter, text: o?.text ?? '', finish: o?.finish ?? null, candidate: voted ? (c?.label ?? 'Unknown') : null }
  }
  const prompt = run.prompts.find((p) => p.id === pair.promptId)
  return {
    pair,
    promptTitle: prompt?.title ?? '',
    promptText: prompt?.text ?? '',
    a: side('A', pair.a),
    b: side('B', pair.b),
    voted,
    vote: pair.vote
  }
}

/** The pair to show: the one the user is looking at if it still exists, else the first without a vote. */
export function pairToShow(run: WritingBenchRun, currentId: string | null): BenchPair | null {
  const cur = currentId ? run.pairs.find((p) => p.id === currentId) : undefined
  if (cur) return cur
  return run.pairs.find((p) => p.vote === null) ?? null
}
