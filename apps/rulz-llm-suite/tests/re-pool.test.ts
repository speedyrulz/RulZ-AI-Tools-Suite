// The practice pool (src/main/re/pool.ts): lanes over hosts, claiming, stops, losses, cost and saving, with a
// scripted session runner in place of runSession and fakes for everything the pool reaches.
import { describe, expect, it } from 'vitest'
import { DEFAULT_BATCH_OPTIONS, summarize, type BatchRun } from '@shared/batch'
import { DEFAULT_BATCH_TARGET, type BatchTarget } from '@shared/remote'
import type { BatchStartRequest } from '@shared/ipc'
import type { Conversation, RePracticeTask } from '@shared/types'
import { LANE_STAGGER_MS, PracticePool, type PoolDeps, type PoolHost, type PoolScope } from '../src/main/re/pool'
import { poolHandlers } from '../src/main/re/poolIpc'
import type { SessionDeps, SessionResult, SessionSpec } from '../src/main/re/session'

const task = (id: string, family = 'xor'): RePracticeTask => ({ id, family, tier: 'easy', binary: `${id}.exe`, prompt: `Solve {exe}` })
const tasks = (n: number): RePracticeTask[] => Array.from({ length: n }, (_, i) => task(`t${String(i + 1).padStart(2, '0')}`))

const host = (n: number, o: Partial<PoolHost> = {}): PoolHost => ({
  instanceId: `inst-${n}`,
  hostLabel: n === 0 ? 'This PC' : `RTX 4090 (${n})`,
  dollarsPerHour: n === 0 ? 0 : 0.5,
  ...(n === 0 ? {} : { rentalId: n }),
  modelName: 'model',
  contextLength: 131072,
  adapters: undefined,
  ...o
})

const target = (hosts: number[], sessions: number, extra: Partial<BatchTarget> = {}): BatchTarget => ({
  hosts: hosts.map((n) => (n === 0 ? { kind: 'local' as const } : { kind: 'vast' as const, rentalId: n })),
  sessionsPerHost: sessions,
  contextPerSession: sessions > 1 ? 65536 : null,
  stopRentalsAfter: false,
  ...extra
})

const request = (n: number, t: BatchTarget = DEFAULT_BATCH_TARGET, extra: Partial<BatchStartRequest> = {}): BatchStartRequest => ({
  tasks: tasks(n),
  options: DEFAULT_BATCH_OPTIONS,
  label: 'test',
  kitDir: 'C:\\kit',
  modelId: 'm/model.gguf',
  solved: [],
  target: t,
  ...extra
})

type Behavior = 'solved' | 'wrong' | 'lost' | 'hang' | 'error'

function convOf(id: string, taskId: string, outcome: 'solved' | 'failed'): Conversation {
  return {
    id,
    messages: [{ id: 'm', role: 'assistant', content: 'ANSWER: 42', createdAt: 0, stats: { promptTokens: 100, predictedTokens: 10 } }],
    re: { target: 't', goal: 'g', collect: true, outcome, autoGraded: true, practice: { taskId, family: 'xor', tier: 'easy', answer: '42', expected: '42' } }
  } as unknown as Conversation
}

interface Setup {
  hosts?: PoolHost[]
  /** Per task id. */
  behave?: (taskId: string, instanceId: string, spec: SessionSpec) => Behavior
  busy?: string | null
  openScope?: (opts: { alone: boolean }) => PoolScope
  prepareHosts?: PoolDeps['prepareHosts']
  states?: Record<string, 'ready' | 'reconnecting' | 'gone'>
  /** Lanes start this far apart; default none (the test for the real stagger passes 'default'). */
  stagger?: 'default'
  /** Each session takes this long (real time) before it ends. */
  delayMs?: number
}

function make(o: Setup = {}) {
  const hosts = o.hosts ?? [host(0)]
  const clock = { t: 1_000_000 }
  const log = {
    runs: [] as { taskId: string; instanceId: string; conversationId: string }[],
    saves: [] as BatchRun[],
    changes: [] as { run: BatchRun; active: boolean; stopping: string | null }[],
    chats: [] as string[][],
    owned: new Set<string>(),
    released: [] as string[],
    ended: [] as { run: BatchRun; rentals: number[] }[],
    scopesOpened: 0,
    scopesClosed: 0,
    beginTasks: 0,
    sleeps: [] as number[],
    usedHosts: new Set<string>(),
    maxRunning: 0,
    maxPerHost: new Map<string, number>(),
    warn: [] as string[],
    prepared: [] as { target: BatchTarget; load: unknown }[]
  }
  const running = new Map<string, number>()
  const stored = new Map<string, BatchRun>()
  const states = o.states ?? {}
  let n = 0
  const signals = new Map<string, AbortSignal>()
  const deps: PoolDeps = {
    now: () => clock.t,
    sleep: async (ms) => {
      log.sleeps.push(ms)
      await new Promise((r) => setImmediate(r))
    },
    uid: () => `u${++n}`,
    store: {
      save: async (run) => {
        log.saves.push(structuredClone(run))
        stored.set(run.id, structuredClone(run))
      },
      get: async (id) => stored.get(id)
    },
    conversations: {
      own: (id) => void log.owned.add(id),
      release: (id) => {
        log.owned.delete(id)
        log.released.push(id)
      },
      savePool: async (c) => ({ id: c.id }) as never,
      get: async () => null
    },
    events: {
      batchChanged: (run, active, stopping) => void log.changes.push({ run, active, stopping }),
      poolChats: (ids) => void log.chats.push(ids),
      chatLive: () => undefined,
      conversationSaved: () => undefined
    },
    busy: () => o.busy ?? null,
    prepareHosts: async (t, _model, load) => {
      log.prepared.push({ target: t, load })
      if (o.prepareHosts) return o.prepareHosts(t, _model, load)
      return hosts
    },
    useHost: (id) => {
      log.usedHosts.add(id)
      return () => void log.usedHosts.delete(id)
    },
    servers: () => [],
    openScope: (opts) => {
      if (o.openScope) return o.openScope(opts)
      log.scopesOpened++
      return {
        beginTask: () => void log.beginTasks++,
        callTool: async () => ({ text: '', isError: false }),
        close: async () => void log.scopesClosed++
      }
    },
    settings: () => ({ re: { maxToolRounds: 80, collect: true, kitDir: 'C:\\kit' } }) as never,
    chat: { start: async () => undefined, abort: () => undefined, releaseTurn: () => undefined },
    instance: () => ({ hasVision: false }),
    instanceState: (id) => states[id] ?? 'ready',
    prepare: async (_kit, taskId) => ({ target: `C:\\work\\${taskId}.exe`, prompt: `Solve ${taskId}`, family: 'xor', tier: 'easy' }),
    grade: async () => ({ answer: '42', passed: true, expected: '42' }),
    onEnded: (run, rentals) => void log.ended.push({ run, rentals }),
    warn: (m) => void log.warn.push(m),
    staggerMs: o.stagger === 'default' ? undefined : 0,
    runSession: async (_d: SessionDeps, spec: SessionSpec): Promise<SessionResult> => {
      const taskId = spec.practice.taskId
      log.runs.push({ taskId, instanceId: spec.instanceId, conversationId: spec.conversationId })
      signals.set(spec.conversationId, spec.signal)
      running.set(spec.instanceId, (running.get(spec.instanceId) ?? 0) + 1)
      const total = [...running.values()].reduce((a, b) => a + b, 0)
      log.maxRunning = Math.max(log.maxRunning, total)
      log.maxPerHost.set(spec.instanceId, Math.max(log.maxPerHost.get(spec.instanceId) ?? 0, running.get(spec.instanceId)!))
      try {
        const b = o.behave?.(taskId, spec.instanceId, spec) ?? 'solved'
        const base = { timedOut: false, lost: false, aborted: null, replyAt: clock.t }
        if (b === 'hang') {
          await new Promise<void>((resolve) => spec.signal.addEventListener('abort', () => resolve()))
          const reason = spec.signal.reason as 'stopNow' | 'skip'
          return { ...base, conversation: convOf(spec.conversationId, taskId, 'failed'), end: 'stopped', aborted: reason }
        }
        // Let the other lanes run in between.
        await new Promise((r) => setTimeout(r, o.delayMs ?? 0))
        clock.t += 60_000
        if (b === 'lost') return { ...base, conversation: convOf(spec.conversationId, taskId, 'failed'), end: 'error', lost: true }
        if (b === 'error') throw new Error('boom')
        return { ...base, conversation: convOf(spec.conversationId, taskId, b === 'solved' ? 'solved' : 'failed'), end: 'done' }
      } finally {
        running.set(spec.instanceId, running.get(spec.instanceId)! - 1)
      }
    }
  }
  const pool = new PracticePool(deps)
  const waitFor = async (test: () => boolean, ms = 2000): Promise<void> => {
    const t0 = Date.now()
    while (!test()) {
      if (Date.now() - t0 > ms) throw new Error('timed out waiting')
      await new Promise((r) => setTimeout(r, 2))
    }
  }
  return { pool, deps, log, clock, stored, waitFor, signals, states }
}

const items = (r: BatchRun) => r.items.map((i) => `${i.taskId}:${i.status}`)

describe('PracticePool', () => {
  it('runs one local lane through every task in order, as the old loop did', async () => {
    const t = make({ behave: (id) => (id === 't02' ? 'wrong' : 'solved') })
    const run = await t.pool.start(request(3))
    expect(run).toMatchObject({ status: 'running', modelId: 'm/model.gguf', modelName: 'model', contextLength: 131072, maxToolRounds: 80, kitDir: 'C:\\kit', label: 'test' })
    expect(run.hosts).toEqual([{ label: 'This PC', dollarsPerHour: 0 }])
    await t.pool.whenIdle()
    expect(t.log.runs.map((r) => r.taskId)).toEqual(['t01', 't02', 't03'])
    const final = t.stored.get(run.id)!
    expect(final.status).toBe('done')
    expect(items(final)).toEqual(['t01:solved', 't02:wrong', 't03:solved'])
    expect(final.items[0]).toMatchObject({ answer: '42', toolRounds: 0, promptTokens: 100, outputTokens: 10, host: 'This PC' })
    expect(final.items[0].conversationId).toBe(t.log.runs[0].conversationId)
    expect(final.finishedAt).toBeDefined()
    expect(summarize(final)).toMatchObject({ solved: 2, scored: 3 })
    // Loaded as it is (no parallel settings) for one session.
    expect(t.log.prepared[0].load).toBeNull()
    expect(t.log.ended).toHaveLength(1)
    expect(t.log.ended[0].rentals).toEqual([])
    expect(t.log.usedHosts.size).toBe(0)
  })

  it('runs 3 sessions on each of 2 hosts, loading the model for them, and never gives a task to two lanes', async () => {
    const t = make({ hosts: [host(1), host(2)] })
    const run = await t.pool.start(request(12, target([1, 2], 3)))
    expect(t.log.prepared[0].load).toEqual({ parallel: 3, contextLength: 65536 * 3 })
    expect(t.log.scopesOpened).toBe(6)
    await t.pool.whenIdle()
    const taskIds = t.log.runs.map((r) => r.taskId)
    expect(taskIds).toHaveLength(12)
    expect(new Set(taskIds).size).toBe(12)
    expect(t.log.maxRunning).toBe(6)
    expect([...t.log.maxPerHost.values()]).toEqual([3, 3])
    const final = t.stored.get(run.id)!
    expect(final.status).toBe('done')
    expect(final.items.every((i) => i.status === 'solved')).toBe(true)
    expect(new Set(final.items.map((i) => i.host))).toEqual(new Set(['RTX 4090 (1)', 'RTX 4090 (2)']))
    // Every lane's scope was closed, once.
    expect(t.log.scopesClosed).toBe(6)
    // Every task begins in a fresh scope state.
    expect(t.log.beginTasks).toBe(12)
  })

  it('runs a real session end to end on a lane: chat saved by the pool, graded, scored', async () => {
    const t = make()
    delete t.deps.runSession
    const savedChats: Conversation[] = []
    t.deps.conversations.savePool = async (c) => {
      savedChats.push(structuredClone(c))
      return { id: c.id } as never
    }
    t.deps.settings = () =>
      ({
        toolsEnabled: true, toolMode: 'auto', toolBudgetTokens: 6000, disabledTools: [], confirmToolCalls: true, allowedTools: [],
        defaultSampling: {}, contextOverflow: 'truncateMiddle', compactToolResults: false, keepFullToolResults: 3,
        re: { maxToolRounds: 80, toolBudgetTokens: 16000, kitDir: 'C:\kit', autoApproveStatic: true, collect: true }
      }) as never
    t.deps.chat = {
      start: async (req, onEvent) => {
        onEvent({ type: 'delta', requestId: req.requestId, content: 'ANSWER: 42' })
        onEvent({ type: 'done', requestId: req.requestId, stats: { stopReason: 'stop', promptTokens: 50, predictedTokens: 5 } })
      },
      abort: () => undefined,
      releaseTurn: () => undefined
    }
    const run = await t.pool.start(request(2))
    await t.pool.whenIdle()
    const final = t.stored.get(run.id)!
    expect(items(final)).toEqual(['t01:solved', 't02:solved'])
    expect(final.items[0]).toMatchObject({ answer: '42', promptTokens: 50, outputTokens: 5, toolRounds: 0 })
    // The chat went through the pool's save with a rising revision, and the last copy is graded.
    const mine = savedChats.filter((c) => c.id === final.items[0].conversationId)
    expect(mine.map((c) => c.poolRev)).toEqual([1, 2, 3])
    expect(mine.at(-1)!.title).toBe('Practice: t01')
    expect(mine.at(-1)!.re).toMatchObject({ outcome: 'solved', autoGraded: true })
    expect(t.log.owned.size).toBe(0)
  })

  it('starts lanes 3 seconds apart', async () => {
    const t = make({ hosts: [host(1)], stagger: 'default' })
    await t.pool.start(request(4, target([1], 4)))
    await t.pool.whenIdle()
    expect(LANE_STAGGER_MS).toBe(3000)
    expect(t.log.sleeps.filter((s) => s >= 3000).sort((a, b) => a - b)).toEqual([3000, 6000, 9000])
  })

  it('keeps the order of attempts: every task has its first try before any second', async () => {
    const t = make({ hosts: [host(1)] })
    const run = await t.pool.start(request(2, target([1], 2), { options: { ...DEFAULT_BATCH_OPTIONS, attempts: 2 } }))
    await t.pool.whenIdle()
    expect(t.log.runs.map((r) => r.taskId)).toEqual(['t01', 't02', 't01', 't02'])
    expect(t.stored.get(run.id)!.items.map((i) => i.attempt)).toEqual([1, 1, 2, 2])
  })

  it('"stop after" lets the tasks in progress finish and starts no more', async () => {
    const t = make({ hosts: [host(1)], behave: () => 'solved', delayMs: 30 })
    const run = await t.pool.start(request(10, target([1], 2)))
    await t.waitFor(() => t.log.runs.length >= 2)
    t.pool.stop('after')
    expect(t.pool.stopping).toBe('after')
    await t.pool.whenIdle()
    const final = t.stored.get(run.id)!
    expect(final.status).toBe('stopped')
    const done = final.items.filter((i) => i.status === 'solved')
    expect(done.length).toBeGreaterThanOrEqual(2)
    expect(done.length).toBeLessThan(10)
    expect(final.items.filter((i) => i.status === 'pending').length).toBe(10 - done.length)
    expect(t.pool.activeId).toBeNull()
  })

  it('"stop now" stops the sessions and puts their tasks back to pending, in a chat they leave behind', async () => {
    const t = make({ hosts: [host(1)], behave: () => 'hang' })
    const run = await t.pool.start(request(6, target([1], 2)))
    await t.waitFor(() => t.log.runs.length >= 2)
    t.pool.stop('now')
    await t.pool.whenIdle()
    const final = t.stored.get(run.id)!
    expect(final.status).toBe('stopped')
    expect(final.items.every((i) => i.status === 'pending')).toBe(true)
    expect(final.items.every((i) => i.conversationId === undefined && i.startedAt === undefined)).toBe(true)
    expect([...t.signals.values()].every((s) => s.aborted && s.reason === 'stopNow')).toBe(true)
    // 'now' outranks a later 'after'.
    t.pool.stop('after')
    expect(t.pool.stopping).toBeNull()
  })

  it('"skip" stops one session: its task is stopped and the lane takes the next', async () => {
    const t = make({ behave: (id) => (id === 't01' ? 'hang' : 'solved') })
    const run = await t.pool.start(request(3))
    await t.waitFor(() => t.log.runs.length === 1)
    t.pool.skip(t.log.runs[0].conversationId)
    await t.pool.whenIdle()
    expect(items(t.stored.get(run.id)!)).toEqual(['t01:stopped', 't02:solved', 't03:solved'])
    expect(t.stored.get(run.id)!.status).toBe('done')
  })

  it('owns each chat from before its session starts until its last save, and tells the window', async () => {
    const t = make({ delayMs: 30 })
    await t.pool.start(request(2))
    await t.waitFor(() => t.log.runs.length === 1)
    const id = t.log.runs[0].conversationId
    expect(t.log.owned.has(id)).toBe(true)
    expect(t.pool.poolChats()).toEqual([id])
    await t.pool.whenIdle()
    expect(t.log.owned.size).toBe(0)
    expect(t.pool.poolChats()).toEqual([])
    expect(t.log.chats[0]).toEqual([id])
    expect(t.log.chats.at(-1)).toEqual([])
  })

  describe('taking the pool', () => {
    const stoppedRun = async (t: ReturnType<typeof make>): Promise<string> => {
      const run = await t.pool.start(request(3))
      await t.pool.whenIdle()
      const saved = t.stored.get(run.id)!
      t.stored.set(run.id, { ...saved, status: 'stopped', items: saved.items.map((i) => ({ ...i, status: 'pending' as const })) })
      return run.id
    }

    it('lets only one of two resumes (or a resume and a start) through, though the store answers slowly', async () => {
      let hang = false
      const t = make({ behave: () => (hang ? 'hang' : 'solved') })
      const id = await stoppedRun(t)
      hang = true
      const get = t.deps.store.get
      t.deps.store.get = async (x) => {
        await new Promise((r) => setTimeout(r, 10))
        return get(x)
      }
      const results = await Promise.allSettled([t.pool.resume(id, DEFAULT_BATCH_TARGET), t.pool.resume(id, DEFAULT_BATCH_TARGET), t.pool.start(request(1))])
      expect(results.filter((r) => r.status === 'fulfilled')).toHaveLength(1)
      expect(results.filter((r) => r.status === 'rejected').every((r) => /already running/.test(String((r as PromiseRejectedResult).reason)))).toBe(true)
      t.pool.stop('now')
      await t.pool.whenIdle()
    })

    it('stays taken until the last save of a run is written, though the window was told it ended', async () => {
      const t = make()
      const save = t.deps.store.save
      let release!: () => void
      const gate = new Promise<void>((r) => (release = r))
      t.deps.store.save = async (run) => {
        if (run.status === 'done') await gate
        return save(run)
      }
      const run = await t.pool.start(request(2))
      await t.waitFor(() => t.log.changes.some((c) => c.run.id === run.id && !c.active))
      expect(t.pool.activeId).toBeNull()
      expect(t.pool.busy).toBe(true)
      await expect(t.pool.start(request(1))).rejects.toThrow('A batch run is already running.')
      release()
      await t.pool.whenIdle()
      expect(t.pool.busy).toBe(false)
      expect(t.log.scopesClosed).toBe(1)
      await expect(t.pool.start(request(1))).resolves.toBeDefined()
      await t.pool.whenIdle()
    })

    it('does not begin a run when the app closes while the model loads, and acquires nothing', async () => {
      let release!: () => void
      const gate = new Promise<void>((r) => (release = r))
      const t = make({ hosts: [host(1)], prepareHosts: async () => (await gate, [host(1)]) })
      const started = t.pool.start(request(2, target([1], 1)))
      const outcome = started.then(
        () => 'started',
        (e: Error) => e.message
      )
      await new Promise((r) => setTimeout(r, 5))
      await t.pool.dispose()
      release()
      expect(await outcome).toMatch(/closing/)
      expect(t.log.scopesOpened).toBe(0)
      expect(t.log.usedHosts.size).toBe(0)
      expect(t.log.saves).toHaveLength(0)
      expect(t.pool.busy).toBe(false)
    })

    it('also checks the app is not closing before a resume begins', async () => {
      let release!: () => void
      const gate = new Promise<void>((r) => (release = r))
      const t = make()
      const id = await stoppedRun(t)
      t.deps.prepareHosts = async () => (await gate, [host(0)])
      const resumed = t.pool.resume(id, DEFAULT_BATCH_TARGET).then(
        () => 'resumed',
        (e: Error) => e.message
      )
      await new Promise((r) => setTimeout(r, 5))
      await t.pool.dispose()
      release()
      expect(await resumed).toMatch(/closing/)
      expect(t.log.usedHosts.size).toBe(0)
    })

    it('tells openScope whether the lane is alone', async () => {
      const seen: boolean[] = []
      const open = (opts: { alone: boolean }): PoolScope => {
        seen.push(opts.alone)
        return { beginTask: () => undefined, callTool: async () => ({ text: '', isError: false }), close: async () => undefined }
      }
      const one = make({ openScope: open })
      await one.pool.start(request(1))
      await one.pool.whenIdle()
      const many = make({ hosts: [host(1)], openScope: open })
      await many.pool.start(request(2, target([1], 2)))
      await many.pool.whenIdle()
      expect(seen).toEqual([true, false, false])
    })

    it('marks a task whose chat could not be saved as an error instead of recording its grade', async () => {
      const t = make()
      const inner = t.deps.runSession!
      t.deps.runSession = async (d, spec) => ({ ...(await inner(d, spec)), saveError: 'EBUSY: resource busy' })
      const run = await t.pool.start(request(1))
      await t.pool.whenIdle()
      const [item] = t.stored.get(run.id)!.items
      expect(item).toMatchObject({ status: 'error', error: 'The chat could not be saved: EBUSY: resource busy' })
      expect(item.answer).toBeUndefined()
      // The chat is let go after its last save, not before.
      expect(t.log.owned.size).toBe(0)
    })
  })

  describe('losing an instance', () => {
    it('returns the task to pending, ends that host\'s lane, and the run goes on with the other host', async () => {
      const t = make({ hosts: [host(1), host(2)], behave: (_id, inst) => (inst === 'inst-1' ? 'lost' : 'solved') })
      // The first host is gone once its first session is reported lost.
      t.states['inst-1'] = 'gone'
      const run = await t.pool.start(request(6, target([1, 2], 1)))
      await t.pool.whenIdle()
      const final = t.stored.get(run.id)!
      expect(final.status).toBe('done')
      expect(final.items.every((i) => i.status === 'solved' && i.host === 'RTX 4090 (2)')).toBe(true)
      expect(t.log.runs.filter((r) => r.instanceId === 'inst-1').length).toBeLessThanOrEqual(1)
    })

    it('counts the loss on the item, and a task lost twice becomes an error', async () => {
      const t = make({ hosts: [host(1)], behave: () => 'lost' })
      // Reachable again each time, so the lane carries on and loses the same task twice.
      const run = await t.pool.start(request(1))
      await t.pool.whenIdle()
      const [item] = t.stored.get(run.id)!.items
      expect(item).toMatchObject({ status: 'error', lost: 2 })
      expect(item.error).toMatch(/instance was lost 2 times/)
    })

    it('stops the run, resumable, when no lane is left', async () => {
      const t = make({ hosts: [host(1), host(2)], behave: () => 'lost' })
      t.states['inst-1'] = 'gone'
      t.states['inst-2'] = 'gone'
      const run = await t.pool.start(request(4, target([1, 2], 1)))
      await t.pool.whenIdle()
      const final = t.stored.get(run.id)!
      expect(final.status).toBe('stopped')
      expect(final.items.filter((i) => i.status === 'pending')).toHaveLength(4 - final.items.filter((i) => i.status !== 'pending').length)
      expect(final.items.filter((i) => i.status === 'pending').length).toBeGreaterThanOrEqual(2)
    })

    it('ends a lane at once whose instance is gone before it claims anything', async () => {
      const t = make({ hosts: [host(1), host(2)] })
      t.states['inst-1'] = 'gone'
      const run = await t.pool.start(request(4, target([1, 2], 1)))
      await t.pool.whenIdle()
      expect(t.log.runs.every((r) => r.instanceId === 'inst-2')).toBe(true)
      expect(t.stored.get(run.id)!.status).toBe('done')
    })
  })

  it('records a task that cannot be prepared or whose session throws, and carries on', async () => {
    const t = make({ behave: (id) => (id === 't01' ? 'error' : 'solved') })
    t.deps.prepare = async (_k, id) => {
      if (id === 't02') throw new Error('No copy of t02.')
      return { target: 'x', prompt: 'p', family: 'xor', tier: 'easy' }
    }
    const run = await t.pool.start(request(3))
    await t.pool.whenIdle()
    const final = t.stored.get(run.id)!
    expect(items(final)).toEqual(['t01:error', 't02:error', 't03:solved'])
    expect(final.items[0].error).toBe('boom')
    expect(final.items[1].error).toBe('No copy of t02.')
    // A chat that was owned is let go even when its session threw.
    expect(t.log.owned.size).toBe(0)
  })

  describe('refusing', () => {
    it('refuses a second run, an empty selection, and a busy PC with a sentence', async () => {
      const t = make({ behave: () => 'hang' })
      await t.pool.start(request(2))
      await expect(t.pool.start(request(2))).rejects.toThrow('A batch run is already running.')
      t.pool.stop('now')
      await t.pool.whenIdle()
      await expect(t.pool.start({ ...request(2), tasks: [] })).rejects.toThrow('No practice tasks match this selection.')
      const busy = make({ busy: 'A benchmark is running.' })
      await expect(busy.pool.start(request(2))).rejects.toThrow('A benchmark is running.')
      expect(busy.log.prepared).toHaveLength(0)
    })

    it('fails before it exists when a scope cannot open, and lets everything go', async () => {
      let opened = 0
      let closed = 0
      const t = make({
        hosts: [host(1)],
        openScope: () => {
          if (opened === 2) throw new Error('REA is not connected. Open the MCP page, fix it, and start the run again.')
          opened++
          return { beginTask: () => undefined, callTool: async () => ({ text: '', isError: false }), close: async () => void closed++ }
        }
      })
      await expect(t.pool.start(request(4, target([1], 4)))).rejects.toThrow(/REA is not connected/)
      expect(closed).toBe(2)
      expect(t.log.usedHosts.size).toBe(0)
      expect(t.pool.activeId).toBeNull()
      expect(t.log.saves).toHaveLength(0)
      // The pool is free again.
      expect(t.pool.busy).toBe(false)
    })

    it('fails with the hosts\' sentence when they cannot be prepared, and with different adapters', async () => {
      const t = make({ prepareHosts: async () => Promise.reject(new Error('Rental 5 is not running.')) })
      await expect(t.pool.start(request(2))).rejects.toThrow('Rental 5 is not running.')
      const mixed = make({ hosts: [host(1, { adapters: [{ name: 'a', scale: 1 }] }), host(2)] })
      await expect(mixed.pool.start(request(2, target([1, 2], 1)))).rejects.toThrow(/different adapters/)
    })
  })

  describe('resuming', () => {
    it('runs the pending tasks of a stopped run and keeps its results and cost', async () => {
      const t = make({ hosts: [host(1)], behave: (id) => (id === 't01' ? 'solved' : 'hang') })
      const run = await t.pool.start(request(3))
      await t.waitFor(() => t.log.runs.length >= 2)
      t.pool.stop('now')
      await t.pool.whenIdle()
      const stopped = t.stored.get(run.id)!
      expect(items(stopped)).toEqual(['t01:solved', 't02:pending', 't03:pending'])
      const costBefore = stopped.cost
      t.log.runs.length = 0
      const o = { behave: undefined }
      void o
      const again = await t.pool.resume(run.id, DEFAULT_BATCH_TARGET)
      expect(again.status).toBe('running')
      await t.waitFor(() => t.log.runs.length >= 1)
      t.pool.stop('now')
      await t.pool.whenIdle()
      expect(t.stored.get(run.id)!.items[0].status).toBe('solved')
      expect(t.log.runs.map((r) => r.taskId)).not.toContain('t01')
      if (costBefore !== undefined) expect(t.stored.get(run.id)!.cost).toBeGreaterThanOrEqual(costBefore)
    })

    it('refuses other adapters, a finished run, an unknown run, and hosts that cannot be prepared', async () => {
      const t = make({ hosts: [host(1, { adapters: [{ name: 'lora', scale: 0.8 }] })] })
      const run = await t.pool.start(request(2))
      expect(run.adapters).toEqual([{ name: 'lora', scale: 0.8 }])
      t.pool.stop('now')
      await t.pool.whenIdle()
      const saved = t.stored.get(run.id)!
      t.stored.set(run.id, { ...saved, status: 'stopped', items: saved.items.map((i) => ({ ...i, status: 'pending' as const })) })
      const other = make({ hosts: [host(1, { adapters: [{ name: 'other', scale: 1 }] })] })
      other.stored.set(run.id, structuredClone(t.stored.get(run.id)!))
      await expect(other.pool.resume(run.id, DEFAULT_BATCH_TARGET)).rejects.toThrow(/This run used other adapters \(\+ lora x0\.8\)/)
      await expect(t.pool.resume('nope', DEFAULT_BATCH_TARGET)).rejects.toThrow('That batch run is gone.')
      t.stored.set(run.id, { ...saved, status: 'done', items: saved.items.map((i) => ({ ...i, status: 'solved' as const })) })
      await expect(t.pool.resume(run.id, DEFAULT_BATCH_TARGET)).rejects.toThrow('Every task of this run has been tried.')
      t.stored.set(run.id, { ...saved, status: 'stopped' })
      t.deps.prepareHosts = async () => Promise.reject(new Error('Rental 1 is stopped.'))
      await expect(t.pool.resume(run.id, DEFAULT_BATCH_TARGET)).rejects.toThrow('Rental 1 is stopped.')
    })
  })

  describe('cost and rentals', () => {
    it('counts each rented host once, for as long as its lanes ran, not once per lane', async () => {
      const t = make({ hosts: [host(1, { dollarsPerHour: 3.6 })] })
      // Two lanes; each task advances the clock a minute.
      const run = await t.pool.start(request(4, target([1], 2)))
      await t.pool.whenIdle()
      const final = t.stored.get(run.id)!
      const hours = (t.clock.t - run.createdAt) / 3_600_000
      expect(final.cost).toBeCloseTo(3.6 * hours, 3)
      expect(final.cost).toBeGreaterThan(0)
    })

    it('records no cost for this PC', async () => {
      const t = make()
      const run = await t.pool.start(request(2))
      await t.pool.whenIdle()
      expect(t.stored.get(run.id)!.cost).toBeUndefined()
    })

    it('asks for the rentals to be stopped only when the target says so', async () => {
      const a = make({ hosts: [host(1), host(2)] })
      await a.pool.start(request(2, target([1, 2], 1, { stopRentalsAfter: true })))
      await a.pool.whenIdle()
      expect(a.log.ended[0].rentals.sort()).toEqual([1, 2])
      const b = make({ hosts: [host(1)] })
      await b.pool.start(request(2, target([1], 1)))
      await b.pool.whenIdle()
      expect(b.log.ended[0].rentals).toEqual([])
    })

    it('releases a host as soon as its lanes are done', async () => {
      const t = make({ hosts: [host(1), host(2)], behave: (_id, inst) => (inst === 'inst-1' ? 'solved' : 'hang') })
      await t.pool.start(request(2, target([1, 2], 1)))
      await t.waitFor(() => t.log.runs.length === 2)
      await t.waitFor(() => !t.log.usedHosts.has('inst-1'))
      expect(t.log.usedHosts.has('inst-2')).toBe(true)
      t.pool.stop('now')
      await t.pool.whenIdle()
      expect(t.log.usedHosts.size).toBe(0)
    })
  })

  describe('saving', () => {
    it('saves the run at most once a second while it runs, and always at the end', async () => {
      const t = make({ hosts: [host(1)] })
      const run = await t.pool.start(request(40, target([1], 4)))
      await t.pool.whenIdle()
      // The clock only moved in whole minutes, but never while a save was due: far fewer saves than changes.
      expect(t.log.changes.length).toBeGreaterThan(80)
      expect(t.log.saves.length).toBeLessThan(t.log.changes.length / 2)
      expect(t.log.saves.at(-1)).toMatchObject({ id: run.id, status: 'done' })
    })

    it('tells the window about every change, with the stop mode', async () => {
      const t = make({ behave: () => 'hang' })
      await t.pool.start(request(2))
      await t.waitFor(() => t.log.runs.length === 1)
      t.pool.stop('after')
      expect(t.log.changes.at(-1)).toMatchObject({ active: true, stopping: 'after' })
      t.pool.stop('now')
      await t.pool.whenIdle()
      expect(t.log.changes.at(-1)).toMatchObject({ active: false, stopping: null })
      expect(t.log.changes.at(-1)!.run.status).toBe('stopped')
    })

    it('on app quit stops the sessions, puts running tasks back to pending and saves', async () => {
      const t = make({ behave: () => 'hang' })
      const run = await t.pool.start(request(3))
      await t.waitFor(() => t.log.runs.length === 1)
      await t.pool.dispose()
      const final = t.stored.get(run.id)!
      expect(final.status).toBe('stopped')
      expect(items(final)).toEqual(['t01:pending', 't02:pending', 't03:pending'])
      await t.pool.whenIdle()
      // The lane ending afterwards changes nothing.
      expect(t.stored.get(run.id)!.status).toBe('stopped')
      await expect(t.pool.start(request(1))).rejects.toThrow(/closing/)
    })
  })

  describe('watching', () => {
    it('hands the chat as saved to the window that starts watching it', async () => {
      const t = make()
      t.deps.conversations.get = async (id) => ({ id }) as Conversation
      expect(await t.pool.watch('abc')).toEqual({ id: 'abc' })
      expect(await t.pool.watch(null)).toBeNull()
    })
  })

  describe('the IPC handlers', () => {
    it('check what the window sends', async () => {
      const t = make()
      const h = poolHandlers(t.pool, async () => undefined)
      await expect(h['re:batchStart'](null as never)).rejects.toThrow('Not a batch run request.')
      await expect(h['re:batchStart']({ ...request(1), tasks: 'x' } as never)).rejects.toThrow('no tasks')
      await expect(h['re:batchStart']({ ...request(1), kitDir: '' })).rejects.toThrow(/kit folder/)
      await expect(h['re:batchStart']({ ...request(1), modelId: '' })).rejects.toThrow(/no model/)
      await expect(h['re:batchStart']({ ...request(1), tasks: Array.from({ length: 20001 }, () => task('x')) })).rejects.toThrow(/Too many tasks/)
      await expect(h['re:batchResume']('../evil', DEFAULT_BATCH_TARGET)).rejects.toThrow('Not a batch run id.')
      expect(() => h['re:batchStop']('whenever' as never)).toThrow(/Stop how/)
      expect(() => h['re:batchSkip']('a/b')).toThrow('Not a chat id.')
      await expect(h['re:watch']('a b' as never)).rejects.toThrow('Not a chat id.')
      // A target outside what is allowed is cleaned, not trusted: 99 sessions become at most 8.
      const run = await h['re:batchStart']({ ...request(1), target: { hosts: [{ kind: 'local' }], sessionsPerHost: 99, contextPerSession: 'x' } as never })
      expect(run.target).toMatchObject({ sessionsPerHost: 8, contextPerSession: null })
      await t.pool.whenIdle()
      // Bad task rows are dropped; good ones kept.
      const t2 = make()
      const h2 = poolHandlers(t2.pool, async () => undefined)
      const r2 = await h2['re:batchStart']({ ...request(1), tasks: [task('ok-1'), { id: '../x' }, 7] as never })
      expect(r2.items.map((i) => i.taskId)).toEqual(['ok-1'])
      await t2.pool.whenIdle()
    })

    it('refuse to delete the run that is running', async () => {
      const t = make({ behave: () => 'hang' })
      const deleted: string[] = []
      const h = poolHandlers(t.pool, async (id) => void deleted.push(id))
      const run = await h['re:batchStart'](request(1))
      await expect(h['re:deleteBatch'](run.id)).rejects.toThrow('This batch run is running; stop it first.')
      h['re:batchStop']('now')
      await t.pool.whenIdle()
      await h['re:deleteBatch'](run.id)
      expect(deleted).toEqual([run.id])
      expect(h['re:poolChats']()).toEqual([])
    })
  })
})
