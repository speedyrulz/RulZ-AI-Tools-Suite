/// <reference lib="dom" />
// Writing benchmark, renderer side: the shared busy check, the run loop with fake deps, the vote and
// what the vote panel may show. No component is rendered.
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '../src/shared/defaults'
import type { InstanceAdapter } from '../src/shared/types'
import { BUILTIN_PROMPTS, newRun, type BenchCandidate, type WritingBenchRun } from '../src/shared/writingBench'
import {
  adaptersLoaded,
  applyVote,
  candidateSettings,
  pairToShow,
  pairView,
  requestAdapters,
  runProgress,
  runWbenchLoop,
  sameLoadConfig,
  type BenchReply,
  type BenchReplyInput,
  type BenchReplyResult,
  type WbenchDeps,
  type WbenchStop
} from '../src/renderer/src/lib/wbenchRunner'

const invokeMock = vi.fn(async (_channel: string, ..._args: unknown[]) => undefined as unknown)
vi.mock('sonner', () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), message: vi.fn(), success: vi.fn(), warning: vi.fn() }) }))
vi.mock('../src/renderer/src/lib/api', () => ({
  invoke: (channel: string, ...args: unknown[]) => invokeMock(channel, ...args),
  on: () => () => undefined,
  errorMessage: (e: unknown) => (e instanceof Error ? e.message : String(e))
}))

const cand = (id: string): BenchCandidate => ({ id, label: `Label ${id}`, modelId: `${id}.gguf`, modelName: id, adapters: [], sampling: DEFAULT_SAMPLING, load: null })

function makeRun(): WritingBenchRun {
  // 2 prompts x 2 candidates x 2 seeds = 8 outputs, grouped by candidate.
  return newRun({ label: 't', prompts: BUILTIN_PROMPTS.slice(0, 2), candidates: [cand('c1'), cand('c2')], seeds: [11, 22], maxTokens: 100 }, '2026-01-01T00:00:00.000Z', 'run1')
}

interface Rig {
  deps: WbenchDeps
  run: () => WritingBenchRun
  saves: WritingBenchRun[]
  prepared: string[]
  requests: BenchReplyInput[]
  stopping: { value: WbenchStop | null }
  busy: { value: string | null }
  texts: (string | null)[]
}

/** Fake deps. `reply` decides what each request returns. */
function rig(reply: (n: number, input: BenchReplyInput, rigRef: Rig) => BenchReply, start = makeRun()): Rig {
  let current = start
  const r = {
    saves: [] as WritingBenchRun[],
    prepared: [] as string[],
    requests: [] as BenchReplyInput[],
    stopping: { value: null as WbenchStop | null },
    busy: { value: null as string | null },
    texts: [] as (string | null)[]
  } as Rig
  r.run = () => current
  r.deps = {
    now: () => 1_700_000_000_000,
    sleep: (ms) => new Promise((res) => setTimeout(res, Math.min(ms, 2))),
    rand: () => 0.1,
    stopping: () => r.stopping.value,
    busyReason: () => r.busy.value,
    getRun: () => current,
    patchRun: (fn) => (current = fn(current)),
    save: async (run) => void r.saves.push(run),
    prepare: async (c) => {
      r.prepared.push(c.id)
      return { adapters: null }
    },
    startReply: (input) => {
      r.requests.push(input)
      return reply(r.requests.length, input, r)
    },
    onText: (k, t) => void r.texts.push(k === null ? null : t)
  }
  return r
}

const ok = (text = 'Some prose.', finish: BenchReplyResult['finish'] = 'stop'): BenchReply => ({ done: Promise.resolve({ text, error: null, stopped: false, finish }), abort: () => undefined })

describe('writing benchmark loop', () => {
  it('writes outputs in stored order, loads each candidate once, saves after every output and adds pairs', async () => {
    const r = rig((n) => ok(`text ${n}`))
    const res = await runWbenchLoop(r.deps)
    expect(res.error).toBeUndefined()
    expect(res.busy).toBeUndefined()
    // Order = stored order: candidate c1's four outputs, then c2's four.
    expect(r.requests.map((q) => q.modelId)).toEqual(['c1.gguf', 'c1.gguf', 'c1.gguf', 'c1.gguf', 'c2.gguf', 'c2.gguf', 'c2.gguf', 'c2.gguf'])
    expect(r.requests.map((q) => q.seed)).toEqual([11, 22, 11, 22, 11, 22, 11, 22])
    expect(r.requests.every((q) => q.maxTokens === 100)).toBe(true)
    // One load per candidate group.
    expect(r.prepared).toEqual(['c1', 'c2'])
    // One save per output, plus the final one.
    expect(r.saves.length).toBe(9)
    expect(r.saves[0].outputs.filter((o) => o.status === 'done')).toHaveLength(1)
    expect(r.saves[3].outputs.filter((o) => o.status === 'done')).toHaveLength(4)
    expect(r.saves[3].pairs).toHaveLength(0)
    // Pairs appear as soon as the second candidate's output for the same prompt and seed is done.
    expect(r.saves[4].pairs).toHaveLength(1)
    const end = r.run()
    expect(end.status).toBe('done')
    expect(end.pairs).toHaveLength(4)
    expect(end.outputs.every((o) => o.status === 'done' && o.finish === 'stop' && o.ms !== null)).toBe(true)
    expect(end.outputs[0].text).toBe('text 1')
    // The live text is cleared when the loop ends.
    expect(r.texts.at(-1)).toBeNull()
  })

  it('records a reply cut off at the token limit and a failed reply', async () => {
    const r = rig((n) =>
      n === 1
        ? ok('cut', 'length')
        : n === 2
          ? { done: Promise.resolve({ text: 'part', error: 'boom', stopped: false, finish: null }), abort: () => undefined }
          : ok()
    )
    await runWbenchLoop(r.deps)
    const outs = r.run().outputs
    expect(outs[0]).toMatchObject({ status: 'done', finish: 'length' })
    expect(outs[1]).toMatchObject({ status: 'failed', finish: 'error', error: 'boom', text: 'part' })
    // The failed output makes no pair; the run is still done (nothing is left pending).
    expect(r.run().status).toBe('done')
    expect(r.run().pairs.length).toBe(3)
  })

  it('stops with a message when something else is using the model', async () => {
    const r = rig((n, _i, rg) => {
      if (n === 2) rg.busy.value = 'A chat reply is running.'
      return ok()
    })
    const res = await runWbenchLoop(r.deps)
    expect(res.busy).toBe('A chat reply is running.')
    expect(r.requests).toHaveLength(2)
    expect(r.run().status).toBe('stopped')
    expect(r.run().outputs.filter((o) => o.status === 'done')).toHaveLength(2)
    expect(r.run().outputs.filter((o) => o.status === 'pending')).toHaveLength(6)
  })

  it('refuses to start a step while busy', async () => {
    const r = rig(() => ok())
    r.busy.value = 'A batch run is running.'
    const res = await runWbenchLoop(r.deps)
    expect(res.busy).toBe('A batch run is running.')
    expect(r.requests).toHaveLength(0)
    expect(r.prepared).toHaveLength(0)
    expect(r.run().status).toBe('stopped')
  })

  it('Stop after this finishes the output in progress and stops', async () => {
    const r = rig((n, _i, rg) => {
      if (n === 3) rg.stopping.value = 'after'
      return ok()
    })
    const res = await runWbenchLoop(r.deps)
    expect(res.error).toBeUndefined()
    expect(r.requests).toHaveLength(3)
    expect(r.run().outputs.filter((o) => o.status === 'done')).toHaveLength(3)
    expect(r.run().status).toBe('stopped')
    // The last save carries the stopped status.
    expect(r.saves.at(-1)?.status).toBe('stopped')
  })

  it('Stop now aborts the reply and puts the output back to pending', async () => {
    let aborted = 0
    const r = rig((n, _i, rg) => {
      if (n < 3) return ok()
      let finish: (v: BenchReplyResult) => void = () => undefined
      const done = new Promise<BenchReplyResult>((res) => (finish = res))
      // The user presses Stop now while the third output is being written.
      setTimeout(() => (rg.stopping.value = 'now'), 5)
      return {
        done,
        abort: () => {
          aborted++
          finish({ text: 'half a sentence', error: null, stopped: true, finish: null })
        }
      }
    })
    const res = await runWbenchLoop(r.deps)
    expect(res.error).toBeUndefined()
    expect(aborted).toBe(1)
    const run = r.run()
    expect(run.status).toBe('stopped')
    expect(run.outputs.filter((o) => o.status === 'done')).toHaveLength(2)
    expect(run.outputs[2]).toMatchObject({ status: 'pending', text: '', finish: null, error: null, ms: null })
    // Resuming picks the same output up again.
    expect(run.outputs.find((o) => o.status === 'pending')?.key).toBe(run.outputs[2].key)
    // The partial text is not left on screen.
    expect(r.texts.at(-1)).toBeNull()
  })

  it('stops with an error, without touching outputs, when the model cannot be loaded', async () => {
    const r = rig(() => ok())
    r.deps.prepare = async () => null
    const res = await runWbenchLoop(r.deps)
    expect(res.error).toMatch(/Could not load c1/)
    expect(r.requests).toHaveLength(0)
    expect(r.run().outputs.every((o) => o.status === 'pending')).toBe(true)
    expect(r.run().status).toBe('stopped')
  })

  it('keeps going when a save fails', async () => {
    const r = rig(() => ok())
    r.deps.save = async () => {
      throw new Error('disk full')
    }
    const res = await runWbenchLoop(r.deps)
    expect(res.error).toBeUndefined()
    expect(r.run().status).toBe('done')
  })
})

describe('adapters for a request', () => {
  const loaded: InstanceAdapter[] = [
    { id: 'a.gguf', name: 'a', path: 'p/a.gguf', scale: 1 },
    { id: 'b.gguf', name: 'b', path: 'p/b.gguf', scale: 1 }
  ]
  it('sends the wanted strengths and switches the other loaded adapters off', () => {
    expect(requestAdapters(loaded, [{ id: 'b.gguf', scale: 0.7 }])).toEqual([
      { id: 'a.gguf', scale: 0 },
      { id: 'b.gguf', scale: 0.7 }
    ])
    expect(requestAdapters([], [])).toBeNull()
  })
  it('knows when the loaded instance already has what a candidate wants', () => {
    expect(adaptersLoaded(loaded, [])).toBe(true)
    expect(adaptersLoaded(loaded, [{ id: 'a.gguf', scale: 1 }])).toBe(true)
    expect(adaptersLoaded(loaded, [{ id: 'c.gguf', scale: 1 }])).toBe(false)
    expect(adaptersLoaded([], [{ id: 'a.gguf', scale: 1 }])).toBe(false)
  })
})

describe('load settings of a candidate', () => {
  it('compares load configs apart from their adapters', () => {
    const a = { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 1 }] }
    expect(sameLoadConfig(a, { ...DEFAULT_LOAD_CONFIG, adapters: [] })).toBe(true)
    expect(sameLoadConfig(a, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'b.gguf', scale: 0.5 }] })).toBe(true)
    expect(sameLoadConfig(a, { ...a, contextLength: 4096 })).toBe(false)
    expect(sameLoadConfig(a, { ...a, gpuLayers: 12 })).toBe(false)
    // Key order does not matter; a changed nested value does.
    const shuffled = Object.fromEntries(Object.entries(DEFAULT_LOAD_CONFIG).reverse()) as typeof DEFAULT_LOAD_CONFIG
    expect(sameLoadConfig(DEFAULT_LOAD_CONFIG, shuffled)).toBe(true)
    expect(sameLoadConfig(DEFAULT_LOAD_CONFIG, { ...DEFAULT_LOAD_CONFIG, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 0.5 } })).toBe(false)
  })
  it('saves the edited values, else the chosen sampling, and clears adapters from the load config', () => {
    const chosen = { ...DEFAULT_SAMPLING, temperature: 0.9 }
    expect(candidateSettings({ sampling: null, load: null }, chosen)).toEqual({ sampling: chosen, load: null })
    const edited = { ...DEFAULT_SAMPLING, temperature: 1.4 }
    const load = { ...DEFAULT_LOAD_CONFIG, contextLength: 4096, adapters: [{ id: 'a.gguf', scale: 1 }] }
    const out = candidateSettings({ sampling: edited, load }, chosen)
    expect(out.sampling).toBe(edited)
    expect(out.load).toEqual({ ...load, adapters: [] })
  })
})

describe('votes and what the panel shows', () => {
  async function finished(): Promise<WritingBenchRun> {
    const r = rig(() => ok())
    await runWbenchLoop(r.deps)
    return r.run()
  }

  it('hides candidate names until the pair has a vote', async () => {
    const run = await finished()
    const pair = run.pairs[0]
    const before = pairView(run, pair)
    expect(before.voted).toBe(false)
    expect(before.a.candidate).toBeNull()
    expect(before.b.candidate).toBeNull()
    expect(before.a.letter).toBe('A')
    expect(before.promptText).not.toBe('')
    const after = pairView(applyVote(run, pair.id, 'a', Date.now()), applyVote(run, pair.id, 'a', Date.now()).pairs[0])
    expect(after.voted).toBe(true)
    expect(after.vote).toBe('a')
    expect([after.a.candidate, after.b.candidate].sort()).toEqual(['Label c1', 'Label c2'])
  })

  it('A and B follow the stored order, not the candidate order', async () => {
    const run = await finished()
    for (const p of run.pairs) {
      const v = pairView(applyVote(run, p.id, 'tie', 1), { ...p, vote: 'tie', votedAt: 'x' })
      const aKey = run.outputs.find((o) => o.key === p.a)
      expect(v.a.candidate).toBe(`Label ${aKey?.candidateId}`)
    }
  })

  it('records a vote once, with strictly increasing times, and ignores a second vote', async () => {
    const run = await finished()
    const [p1, p2] = run.pairs
    const once = applyVote(run, p1.id, 'b', 5_000)
    expect(once.pairs[0]).toMatchObject({ vote: 'b' })
    expect(once.pairs[0].votedAt).toBe(new Date(5_000).toISOString())
    expect(applyVote(once, p1.id, 'a', 9_000)).toBe(once)
    expect(applyVote(once, 'missing', 'a', 9_000)).toBe(once)
    // A second vote in the same millisecond still sorts after the first.
    const twice = applyVote(once, p2.id, 'a', 5_000)
    expect(Date.parse(twice.pairs[1].votedAt as string)).toBeGreaterThan(Date.parse(twice.pairs[0].votedAt as string))
  })

  it('shows the pair being looked at, else the first without a vote', async () => {
    const run = await finished()
    expect(pairToShow(run, null)?.id).toBe(run.pairs[0].id)
    const voted = applyVote(run, run.pairs[0].id, 'a', 1)
    expect(pairToShow(voted, null)?.id).toBe(run.pairs[1].id)
    // After voting, the panel keeps the voted pair on screen until Next.
    expect(pairToShow(voted, run.pairs[0].id)?.id).toBe(run.pairs[0].id)
    expect(runProgress(voted)).toMatchObject({ finished: 8, total: 8, voted: 1, pairs: 4 })
  })
})

describe('stores', () => {
  beforeEach(() => {
    invokeMock.mockClear()
    ;(globalThis as unknown as { window: unknown }).window = {
      rulz: { invoke: invokeMock, on: () => () => undefined },
      addEventListener: () => undefined,
      matchMedia: () => ({ matches: true, addEventListener: () => undefined })
    }
    ;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }
  })

  it('aiBusyReason names what is running, and lets a caller ignore its own kind', async () => {
    const { aiBusyReason } = await import('../src/renderer/src/lib/aiBusy')
    const { useBatch } = await import('../src/renderer/src/stores/batch')
    const { useChat } = await import('../src/renderer/src/stores/chat')
    const { useBench } = await import('../src/renderer/src/stores/bench')
    const { useNovel } = await import('../src/renderer/src/stores/novel')
    const { useWbench } = await import('../src/renderer/src/stores/wbench')
    expect(aiBusyReason()).toBeNull()

    useChat.setState({ autoLoadingModelId: 'm' })
    expect(aiBusyReason()).toBe('A chat reply is running.')
    useChat.setState({ autoLoadingModelId: null })
    expect(aiBusyReason()).toBeNull()

    useBatch.setState({ activeId: 'b1' })
    expect(aiBusyReason()).toBe('A batch run is running.')
    useBatch.setState({ activeId: null })

    useBench.setState({ starting: true })
    expect(aiBusyReason()).toBe('A hardware benchmark is running.')
    useBench.setState({ starting: false })

    const gen = { path: null, action: 'continue', label: 'Continue', phase: 'waiting', alt: 1, of: 1, progress: null, text: '' }
    useNovel.setState({ gen } as never)
    expect(aiBusyReason()).toBe('A Writing action is running.')
    // The Writing code asks with 'writing': its own action does not block it, the others do.
    expect(aiBusyReason('writing')).toBeNull()
    expect(aiBusyReason('wbench')).toBe('A Writing action is running.')
    useNovel.setState({ gen: null })

    useWbench.setState({ running: true })
    expect(aiBusyReason()).toBe('A writing benchmark is running.')
    expect(aiBusyReason('writing')).toBe('A writing benchmark is running.')
    expect(aiBusyReason('wbench')).toBeNull()
    useWbench.setState({ running: false })
    expect(aiBusyReason()).toBeNull()
  })

  it('vote saves the run through wbench:save and keeps names hidden until then', async () => {
    const { useWbench } = await import('../src/renderer/src/stores/wbench')
    const r = rig(() => ok())
    await runWbenchLoop(r.deps)
    const run = r.run()
    useWbench.setState({ entries: [{ ok: true, run }], openId: run.id, running: false, activeId: null })
    const pair = run.pairs[0]
    expect(pairView(run, pair).a.candidate).toBeNull()

    useWbench.getState().vote(pair.id, 'a')
    const e = useWbench.getState().entries[0]
    expect(e.ok && e.run.pairs[0].vote).toBe('a')
    const saved = invokeMock.mock.calls.filter((c) => c[0] === 'wbench:save')
    expect(saved).toHaveLength(1)
    const sentRun = saved[0][1] as WritingBenchRun
    expect(sentRun.pairs[0]).toMatchObject({ vote: 'a' })
    expect(sentRun.pairs[0].votedAt).not.toBeNull()
    expect(e.ok && pairView(e.run, e.run.pairs[0]).a.candidate).not.toBeNull()

    // A second vote on the same pair neither changes it nor is written again.
    useWbench.getState().vote(pair.id, 'b')
    const again = useWbench.getState().entries[0]
    expect(again.ok && again.run.pairs[0].vote).toBe('a')
    expect(invokeMock.mock.calls.filter((c) => c[0] === 'wbench:save')).toHaveLength(1)
  })

  it('a vote whose save failed is written again later, with the run as it is in memory', async () => {
    const { useWbench } = await import('../src/renderer/src/stores/wbench')
    const r = rig(() => ok())
    await runWbenchLoop(r.deps)
    const run = r.run()
    useWbench.setState({ entries: [{ ok: true, run }], openId: run.id, running: false, activeId: null })
    vi.useFakeTimers()
    try {
      let fail = true
      invokeMock.mockImplementation(async (channel: string) => {
        if (channel === 'wbench:save' && fail) throw new Error('locked')
        return undefined
      })
      useWbench.getState().vote(run.pairs[0].id, 'a')
      await vi.advanceTimersByTimeAsync(0)
      fail = false
      await vi.advanceTimersByTimeAsync(5000)
      const saves = invokeMock.mock.calls.filter((c) => c[0] === 'wbench:save')
      expect(saves).toHaveLength(2)
      expect((saves[1][1] as WritingBenchRun).pairs[0].vote).toBe('a')
      // Written: no more retries.
      await vi.advanceTimersByTimeAsync(10000)
      expect(invokeMock.mock.calls.filter((c) => c[0] === 'wbench:save')).toHaveLength(2)
    } finally {
      vi.useRealTimers()
      invokeMock.mockImplementation(async () => undefined)
    }
  })

  it('refuses to start a run while anything else is busy, and lists damaged runs after the good ones', async () => {
    const { useWbench, sortEntries } = await import('../src/renderer/src/stores/wbench')
    const { useChat } = await import('../src/renderer/src/stores/chat')
    useWbench.setState({ entries: [], running: false, activeId: null, starting: false })
    useChat.setState({ autoLoadingModelId: 'm' })
    const input = { label: 'x', prompts: BUILTIN_PROMPTS.slice(0, 1), candidates: [cand('c1'), cand('c2')], seeds: [1], maxTokens: 50 }
    await useWbench.getState().create(input)
    expect(useWbench.getState().entries).toHaveLength(0)
    expect(useWbench.getState().running).toBe(false)
    useChat.setState({ autoLoadingModelId: null })

    const older = newRun({ ...input, label: 'old' }, '2026-01-01T00:00:00.000Z', 'old')
    const newer = newRun({ ...input, label: 'new' }, '2026-02-01T00:00:00.000Z', 'new')
    const sorted = sortEntries([
      { ok: false, id: 'bad', file: 'bad.json', problem: 'Unreadable' },
      { ok: true, run: older },
      { ok: true, run: newer }
    ])
    expect(sorted.map((x) => (x.ok ? x.run.id : x.id))).toEqual(['new', 'old', 'bad'])
  })
})
