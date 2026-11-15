import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  cleanBatchRun,
  DEFAULT_BATCH_OPTIONS,
  isScored,
  itemResult,
  median,
  planBatch,
  sessionStats,
  settleRun,
  summarize,
  UNATTENDED_DENIAL,
  type BatchItem,
  type BatchOptions,
  type BatchRun,
  type TurnEndReason
} from '@shared/batch'
import { runsWithoutAsking, starterToolsThatAsk } from '@shared/re'
import type { ChatMessage, Conversation, RePracticeTask, ReSession } from '@shared/types'
import { BatchStore, MAX_BATCH_RUNS } from '../src/main/re/batches'
import { runBatchLoop, type BatchDeps, type BatchSessionOptions, type BatchStop } from '../src/renderer/src/lib/batchRunner'

const task = (id: string, family: string, tier = 'easy'): RePracticeTask => ({ id, family, tier, binary: `${id}.exe`, prompt: `Solve {exe}` })
const opts = (o: Partial<BatchOptions> = {}): BatchOptions => ({ ...DEFAULT_BATCH_OPTIONS, ...o })
const TASKS = [task('xor-2', 'xor'), task('xor-10', 'xor'), task('xor-1', 'xor'), task('hash-1', 'hash'), task('hash-2', 'hash'), task('cfg-1', 'config')]

const item = (taskId: string, status: BatchItem['status'], extra: Partial<BatchItem> = {}): BatchItem => ({
  taskId,
  family: taskId.split('-')[0],
  tier: 'easy',
  attempt: 1,
  status,
  ...extra
})

const makeRun = (items: BatchItem[], extra: Partial<BatchRun> = {}): BatchRun => ({
  id: 'run-1',
  label: '',
  createdAt: 1000,
  status: 'running',
  modelId: 'm/model.gguf',
  modelName: 'model',
  contextLength: 65536,
  maxToolRounds: 80,
  kitDir: 'C:\\kit',
  options: opts(),
  items,
  ...extra
})

const conv = (re: Partial<ReSession> | undefined, messages: Partial<ChatMessage>[] = []): Conversation =>
  ({
    id: 'c',
    messages: messages.map((m, i) => ({ id: `m${i}`, role: 'assistant', content: '', createdAt: 0, ...m })),
    re: re ? { target: 't', goal: 'g', collect: true, outcome: 'unrated', ...re } : undefined
  }) as Conversation

describe('planning a batch run', () => {
  it('takes every task, family by family in turn, in id order', () => {
    const ids = planBatch(TASKS, opts(), new Set()).map((i) => i.taskId)
    expect(ids).toEqual(['cfg-1', 'hash-1', 'xor-1', 'hash-2', 'xor-2', 'xor-10'])
  })

  it('spreads a task limit over the families', () => {
    expect(planBatch(TASKS, opts({ maxTasks: 4 }), new Set()).map((i) => i.taskId)).toEqual(['cfg-1', 'hash-1', 'xor-1', 'hash-2'])
    expect(planBatch(TASKS, opts({ maxTasks: 99 }), new Set())).toHaveLength(6)
  })

  it('gives every task its first try before any second try', () => {
    const items = planBatch(TASKS, opts({ maxTasks: 2, attempts: 2 }), new Set())
    expect(items.map((i) => `${i.taskId}#${i.attempt}`)).toEqual(['cfg-1#1', 'hash-1#1', 'cfg-1#2', 'hash-1#2'])
    expect(items.every((i) => i.status === 'pending')).toBe(true)
  })

  it('filters by family and by solved, and ignores a task listed twice', () => {
    expect(planBatch(TASKS, opts({ family: 'hash' }), new Set()).map((i) => i.taskId)).toEqual(['hash-1', 'hash-2'])
    expect(planBatch(TASKS, opts({ family: 'hash', onlyUnsolved: true }), new Set(['hash-1'])).map((i) => i.taskId)).toEqual(['hash-2'])
    // Solved tasks still run when comparing models.
    expect(planBatch(TASKS, opts({ family: 'hash' }), new Set(['hash-1']))).toHaveLength(2)
    expect(planBatch([...TASKS, task('cfg-1', 'config')], opts(), new Set())).toHaveLength(6)
    expect(planBatch(TASKS, opts({ family: 'nope' }), new Set())).toEqual([])
  })

  it('repairs options that are out of range', () => {
    const items = planBatch(TASKS, { family: '', onlyUnsolved: false, maxTasks: -5, attempts: 0, minutesPerTask: 30 }, new Set())
    expect(items).toHaveLength(6)
  })
})

describe('tools a session nobody watches would refuse', () => {
  const t = (server: string, name: string) => ({ qualifiedName: `${server}__${name}`, server, name })
  const available = [t('rea', 'open_binary'), t('rea', 'search_strings'), t('rea', 'xrefs'), t('calc', 'calc'), t('rea', 'start_gdb_session')]
  const s = (o: Partial<{ confirmToolCalls: boolean; allowedTools: string[]; autoApproveStatic: boolean }> = {}) => ({
    confirmToolCalls: o.confirmToolCalls ?? true,
    allowedTools: o.allowedTools ?? [],
    re: { autoApproveStatic: o.autoApproveStatic ?? true }
  })

  it('applies the same rule as a tool call in the chat', () => {
    expect(runsWithoutAsking(s(), t('rea', 'xrefs'), true)).toBe(true)
    // The same tool in an ordinary chat asks.
    expect(runsWithoutAsking(s(), t('rea', 'xrefs'), false)).toBe(false)
    expect(runsWithoutAsking(s(), t('rea', 'start_gdb_session'), true)).toBe(false)
    expect(runsWithoutAsking(s({ allowedTools: ['rea__start_gdb_session'] }), t('rea', 'start_gdb_session'), false)).toBe(true)
    expect(runsWithoutAsking(s({ confirmToolCalls: false }), t('rea', 'start_gdb_session'), false)).toBe(true)
    expect(runsWithoutAsking(s({ autoApproveStatic: false }), t('rea', 'xrefs'), true)).toBe(false)
  })

  it('lists the starter tools that would ask, and only those that exist', () => {
    // open_binary is not on the run-unasked list: it asks until the user allows it.
    expect(starterToolsThatAsk(s(), available)).toEqual(['rea__open_binary'])
    expect(starterToolsThatAsk(s({ allowedTools: ['rea__open_binary'] }), available)).toEqual([])
    expect(starterToolsThatAsk(s({ confirmToolCalls: false }), available)).toEqual([])
    expect(starterToolsThatAsk(s({ autoApproveStatic: false }), available)).toEqual(['rea__open_binary', 'rea__search_strings', 'rea__xrefs', 'calc__calc'])
    expect(starterToolsThatAsk(s(), [])).toEqual([])
  })
})

describe('reading a finished session', () => {
  it('counts tool rounds, tokens and refused tool calls', () => {
    const c = conv({}, [
      { role: 'user' },
      { toolCalls: [{ id: '1', name: 'rea__xrefs', arguments: '{}', status: 'done' }], stats: { promptTokens: 900, predictedTokens: 40 } },
      {
        toolCalls: [
          { id: '2', name: 'rea__run', arguments: '{}', status: 'denied', result: UNATTENDED_DENIAL },
          { id: '3', name: 'rea__run', arguments: '{}', status: 'denied', result: 'The user declined to run this tool.' }
        ],
        stats: { promptTokens: 2400, predictedTokens: 60 }
      },
      { content: 'ANSWER: 1', stats: { promptTokens: 2100, predictedTokens: 15 } }
    ])
    expect(sessionStats(c)).toEqual({ toolRounds: 2, promptTokens: 2400, outputTokens: 115, denied: 1 })
  })

  it('reads the grade of a session that ended on its own', () => {
    const practice = { taskId: 'x', family: 'f', tier: 't' }
    expect(itemResult(conv({ outcome: 'solved', autoGraded: true, practice: { ...practice, answer: '42' } }), 'done')).toEqual({ status: 'solved', answer: '42' })
    expect(itemResult(conv({ outcome: 'failed', autoGraded: true, practice: { ...practice, answer: '41' } }), 'done')).toEqual({ status: 'wrong', answer: '41' })
    expect(itemResult(conv({ outcome: 'failed', autoGraded: true, practice }), 'done')).toEqual({ status: 'no-answer' })
    // Grading failed (the kit moved, say): not the model's fault.
    expect(itemResult(conv({ outcome: 'unrated', practice }), 'done').status).toBe('error')
  })

  it('tells limits, stops and errors apart', () => {
    const c = conv({ practice: { taskId: 'x', family: 'f', tier: 't' } }, [{ error: 'context is full' }])
    expect(itemResult(c, 'rounds')).toEqual({ status: 'rounds' })
    expect(itemResult(c, 'stopped')).toEqual({ status: 'stopped' })
    expect(itemResult(c, 'stopped', { timedOut: true })).toEqual({ status: 'timeout' })
    expect(itemResult(null, null, { timedOut: true })).toEqual({ status: 'timeout' })
    expect(itemResult(c, 'error')).toEqual({ status: 'error', error: 'context is full' })
    expect(itemResult(conv({}), 'failed').error).toMatch(/could not start/)
    expect(itemResult(conv({}), null).status).toBe('error')
    expect(itemResult(null, 'done').status).toBe('error')
  })
})

describe('scoring a run', () => {
  it('leaves errors and stops out of the score', () => {
    expect(['solved', 'wrong', 'no-answer', 'rounds', 'timeout'].every((s) => isScored(s as BatchItem['status']))).toBe(true)
    expect(['pending', 'running', 'error', 'stopped'].some((s) => isScored(s as BatchItem['status']))).toBe(false)
    const s = summarize(
      makeRun([
        item('xor-1', 'solved', { toolRounds: 10, ms: 60_000, outputTokens: 500 }),
        item('xor-2', 'wrong', { toolRounds: 30, ms: 90_000, outputTokens: 700, denied: 2 }),
        item('hash-1', 'solved', { toolRounds: 20, ms: 120_000 }),
        item('hash-2', 'error'),
        item('cfg-1', 'timeout'),
        item('cfg-2', 'pending')
      ])
    )
    expect(s).toMatchObject({ total: 6, finished: 5, solved: 2, scored: 4, rate: 0.5, medianRounds: 15, medianMs: 90_000, outputTokens: 1200, denied: 2 })
    expect(s.counts).toMatchObject({ solved: 2, wrong: 1, error: 1, timeout: 1, pending: 1 })
    expect(s.families).toEqual([
      { family: 'cfg', score: { solved: 0, scored: 1, rate: 0 } },
      { family: 'hash', score: { solved: 1, scored: 1, rate: 1 } },
      { family: 'xor', score: { solved: 1, scored: 2, rate: 0.5 } }
    ])
  })

  it('counts a task solved in any try once', () => {
    const s = summarize(makeRun([item('xor-1', 'wrong'), item('xor-2', 'wrong'), item('xor-1', 'solved', { attempt: 2 }), item('xor-2', 'error', { attempt: 2 })]))
    expect(s).toMatchObject({ solved: 1, scored: 3, tasksSolved: 1, tasksScored: 2 })
  })

  it('has no rate before anything counted', () => {
    expect(summarize(makeRun([item('a-1', 'pending'), item('a-2', 'error')]))).toMatchObject({ rate: null, medianRounds: null, medianMs: null })
    expect(median([])).toBeNull()
    expect(median([3, 1, 2])).toBe(2)
  })
})

describe('saved batch runs', () => {
  let dir: string
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-batch-'))
  })
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true })
  })

  it('drops what is not a run and repairs the rest', () => {
    expect(cleanBatchRun(null)).toBeNull()
    expect(cleanBatchRun({ id: '../x', items: [] })).toBeNull()
    expect(cleanBatchRun({ id: 'ok' })).toBeNull()
    const r = cleanBatchRun({
      id: 'ok',
      status: 'weird',
      items: [{ taskId: 'a', status: 'nope', ms: -1, toolRounds: 3, extra: 1 }, { status: 'solved' }, 7],
      options: { attempts: 99 },
      junk: true
    })!
    expect(r).toMatchObject({ status: 'stopped', modelId: null, contextLength: null, options: { attempts: 10, minutesPerTask: 30 } })
    expect(r.items).toEqual([{ taskId: 'a', family: '', tier: '', attempt: 1, status: 'pending', toolRounds: 3 }])
    expect(r).not.toHaveProperty('junk')
  })

  it('shows a run cut off by a closed app as stopped, its task as not run', () => {
    const r = settleRun(makeRun([item('a-1', 'solved'), item('a-2', 'running'), item('a-3', 'pending')]))
    expect(r.status).toBe('stopped')
    expect(r.items.map((i) => i.status)).toEqual(['solved', 'pending', 'pending'])
    const done = makeRun([item('a-1', 'solved')], { status: 'done' })
    expect(settleRun(done)).toBe(done)
  })

  it('saves, replaces, lists newest first and deletes', async () => {
    const path = join(dir, 're-batches.json')
    const store = new BatchStore(path)
    expect(await store.list()).toEqual([])
    await store.save(makeRun([item('a-1', 'pending')], { id: 'old', createdAt: 1 }))
    await store.save(makeRun([item('a-1', 'pending')], { id: 'new', createdAt: 2 }))
    await store.save(makeRun([item('a-1', 'solved')], { id: 'old', createdAt: 1, status: 'done' }))
    await expect(store.save({ nope: true })).rejects.toThrow()

    // The window was reloaded: it asks for the list again, and nothing can be running.
    expect((await store.list()).map((r) => `${r.id}:${r.status}`)).toEqual(['new:stopped', 'old:done'])
    await store.save(makeRun([item('a-1', 'running')], { id: 'new', createdAt: 2 }))

    // A fresh start: the run still marked running was cut off.
    const again = new BatchStore(path)
    const listed = await again.list()
    expect(listed.map((r) => `${r.id}:${r.status}`)).toEqual(['new:stopped', 'old:done'])
    expect(listed[0].items[0].status).toBe('pending')
    await again.delete('new')
    await again.delete('missing')
    expect(JSON.parse(await readFile(path, 'utf8')).map((r: BatchRun) => r.id)).toEqual(['old'])
  }, 20_000)

  it('keeps only the newest runs and survives a file that is not a list', async () => {
    const path = join(dir, 're-batches.json')
    await writeFile(path, '{"not":"a list"}', 'utf8')
    expect(MAX_BATCH_RUNS).toBeGreaterThanOrEqual(50)
    const store = new BatchStore(path, 2)
    expect(await store.list()).toEqual([])
    // Not awaited one by one: the store runs them in order itself.
    await Promise.all([0, 1, 2, 3].map((i) => store.save(makeRun([], { id: `r${i}`, createdAt: i }))))
    expect((await store.list()).map((r) => r.id)).toEqual(['r3', 'r2'])
  }, 20_000)
})

/** What a fake session does: ends by itself after `after` ms with `end`, or never (undefined). */
interface Script {
  after?: number
  end?: TurnEndReason
  re?: Partial<ReSession>
  /** The model loads for this long first: busy, but no reply to stop yet. */
  loading?: number
  /** After the final reply, checking the answer takes this long: not busy, the end not yet reported. */
  grading?: number
  /** The model the session ended up running on, if not the run's. */
  ranOn?: string
  prepareError?: string
}

function harness(run: BatchRun, scripts: Record<string, Script>, o: { stopAt?: [number, BatchStop]; busyUntil?: number; watching?: boolean } = {}) {
  let now = 0
  let current: string | null = 'user-chat'
  let state: BatchRun | undefined = run
  const listeners = new Set<(id: string, reason: TurnEndReason) => void>()
  const sessions: { id: string; script: Script; startedAt: number; ended: boolean; end?: TurnEndReason }[] = []
  const log = { started: [] as BatchSessionOptions[], stops: 0, failed: [] as string[], attendedCalls: 0, patches: 0 }
  const finish = (s: (typeof sessions)[number], reason: TurnEndReason): void => {
    if (s.ended) return
    s.ended = true
    s.end = reason
    for (const fn of listeners) fn(s.id, reason)
  }
  /** When the session's last reply is in (grading may still follow); Infinity if it never answers. */
  const repliedAt = (s: (typeof sessions)[number]): number => (s.script.after === undefined ? Infinity : s.startedAt + (s.script.loading ?? 0) + s.script.after)
  const tick = (): void => {
    for (const s of sessions) if (!s.ended && now >= repliedAt(s) + (s.script.grading ?? 0)) finish(s, s.script.end ?? 'done')
  }
  const active = (): (typeof sessions)[number] | undefined => sessions.find((s) => !s.ended && now < repliedAt(s))
  const externalBusy = (): boolean => o.busyUntil !== undefined && now < o.busyUntil
  const deps: BatchDeps = {
    now: () => now,
    sleep: async (ms) => {
      now += ms
      tick()
      await Promise.resolve()
    },
    stopping: () => (o.stopAt && now >= o.stopAt[0] ? o.stopAt[1] : null),
    chatBusy: () => !!active() || externalBusy(),
    chatIdle: () => !active() && !externalBusy(),
    replyingIn: () => {
      const s = active()
      return s && now >= s.startedAt + (s.script.loading ?? 0) ? s.id : null
    },
    settling: (id) => sessions.some((s) => s.id === id && !s.ended && now >= repliedAt(s)),
    currentChat: () => current,
    stopReply: () => {
      log.stops++
      const s = active()
      if (s) finish(s, 'stopped')
    },
    prepare: async (_kit, taskId) => {
      const err = scripts[taskId]?.prepareError
      if (err) throw new Error(err)
      return { target: `C:\\work\\${taskId}.exe`, prompt: `Solve ${taskId}`, family: 'f', tier: 't' }
    },
    startSession: async (opts) => {
      if (externalBusy()) return null
      const id = `c${sessions.length + 1}`
      log.started.push(opts)
      sessions.push({ id, script: scripts[opts.practice.taskId] ?? { after: 1000 }, startedAt: now, ended: false })
      if (opts.follow && o.watching !== false) current = id
      return id
    },
    attended: () => void log.attendedCalls++,
    failPractice: async (id) => void log.failed.push(id),
    snapshot: async (id) => {
      const s = sessions.find((x) => x.id === id)
      if (!s) return null
      const c = conv({ practice: { taskId: 'x', family: 'f', tier: 't' }, ...s.script.re }, [{ toolCalls: [{ id: '1', name: 't', arguments: '{}', status: 'done' }] }, {}])
      return { ...c, modelId: s.script.ranOn ?? 'm/model.gguf' }
    },
    onTurnEnd: (fn) => {
      listeners.add(fn)
      return () => void listeners.delete(fn)
    },
    getRun: () => state,
    patchRun: (fn) => {
      if (!state) return undefined
      log.patches++
      state = fn(state)
      return state
    }
  }
  return { deps, log, listeners, get run() { return state }, remove: () => void (state = undefined), get now() { return now } }
}

const SOLVED: Script = { after: 5000, re: { outcome: 'solved', autoGraded: true, practice: { taskId: 'x', family: 'f', tier: 't', answer: '42' } } }
const WRONG: Script = { after: 3000, re: { outcome: 'failed', autoGraded: true, practice: { taskId: 'x', family: 'f', tier: 't', answer: '7' } } }

describe('the batch run loop', () => {
  it('runs every task in order and records each result', async () => {
    const h = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending'), item('a-3', 'pending')]), {
      'a-1': SOLVED,
      'a-2': WRONG,
      'a-3': { after: 2000, end: 'error' }
    })
    const { run, error } = await runBatchLoop(h.deps)
    expect(error).toBeUndefined()
    expect(run).toMatchObject({ status: 'done' })
    expect(run!.finishedAt).toBeGreaterThan(0)
    expect(run!.items.map((i) => i.status)).toEqual(['solved', 'wrong', 'error'])
    expect(run!.items.map((i) => i.conversationId)).toEqual(['c1', 'c2', 'c3'])
    expect(run!.items[0]).toMatchObject({ answer: '42', toolRounds: 1, ms: 5000, startedAt: 0 })
    expect(h.log.started.map((s) => s.title)).toEqual(['Practice: a-1', 'Practice: a-2', 'Practice: a-3'])
    expect(h.log.started[0]).toMatchObject({ target: 'C:\\work\\a-1.exe', goal: 'Solve a-1', modelId: 'm/model.gguf', practice: { taskId: 'a-1', kitDir: 'C:\\kit' } })
    expect(h.log.failed).toEqual([])
    expect(h.listeners.size).toBe(0)
  })

  it('grades a session that hit the round limit as failed', async () => {
    const h = harness(makeRun([item('a-1', 'pending')]), { 'a-1': { after: 4000, end: 'rounds' } })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items[0].status).toBe('rounds')
    expect(h.log.failed).toEqual(['c1'])
    expect(h.log.stops).toBe(0)
  })

  it('stops a task at the time limit and goes on to the next', async () => {
    const h = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending')], { options: opts({ minutesPerTask: 1 }) }), { 'a-1': {}, 'a-2': SOLVED })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items.map((i) => i.status)).toEqual(['timeout', 'solved'])
    expect(run!.items[0].ms).toBeGreaterThanOrEqual(60_000)
    expect(run!.items[0].ms).toBeLessThan(62_000)
    expect(h.log.stops).toBe(1)
    expect(h.log.failed).toEqual(['c1'])
    expect(run!.status).toBe('done')
  })

  it('has no time limit at 0 minutes', async () => {
    const h = harness(makeRun([item('a-1', 'pending')], { options: opts({ minutesPerTask: 0 }) }), { 'a-1': { ...SOLVED, after: 5 * 3_600_000 } })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items[0].status).toBe('solved')
    expect(h.log.stops).toBe(0)
  })

  it('counts the time limit from the first reply, not from the model loading', async () => {
    // Loading takes longer than the limit; the task still gets its full minute, and solves in it.
    const slow = harness(makeRun([item('a-1', 'pending')], { options: opts({ minutesPerTask: 1 }) }), { 'a-1': { ...SOLVED, loading: 75_000, after: 50_000 } })
    const a = await runBatchLoop(slow.deps)
    expect(a.run!.items[0].status).toBe('solved')
    expect(slow.log.stops).toBe(0)
    // The time recorded is the model's working time.
    expect(a.run!.items[0].ms).toBeGreaterThanOrEqual(49_000)
    expect(a.run!.items[0].ms).toBeLessThanOrEqual(51_000)

    const never = harness(makeRun([item('a-1', 'pending')], { options: opts({ minutesPerTask: 1 }) }), { 'a-1': { loading: 75_000 } })
    const b = await runBatchLoop(never.deps)
    expect(b.run!.items[0].status).toBe('timeout')
    expect(never.log.stops).toBe(1)
    expect(never.now).toBeGreaterThanOrEqual(135_000)
    expect(never.now).toBeLessThan(140_000)
  })

  it('lets a final reply that beat the time limit be graded', async () => {
    // The reply is in at 59 s; checking it runs past the 60 s limit.
    const h = harness(makeRun([item('a-1', 'pending')], { options: opts({ minutesPerTask: 1 }) }), { 'a-1': { ...SOLVED, after: 59_000, grading: 8000 } })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items[0].status).toBe('solved')
    expect(h.log.stops).toBe(0)
    expect(h.log.failed).toEqual([])
  })

  it('"Stop now" stops the task in progress and leaves it and the rest to resume', async () => {
    const h = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending'), item('a-3', 'pending')]), { 'a-1': SOLVED, 'a-2': {} }, { stopAt: [20_000, 'now'] })
    const { run } = await runBatchLoop(h.deps)
    // The cut-off task has not had its try: it runs again when the run is resumed.
    expect(run!.items.map((i) => i.status)).toEqual(['solved', 'pending', 'pending'])
    expect(run!.items[1].conversationId).toBeUndefined()
    expect(run!.items[1].startedAt).toBeUndefined()
    expect(run!.status).toBe('stopped')
    expect(h.log.stops).toBe(1)
    // Stopped by the user: no verdict on the model.
    expect(h.log.failed).toEqual([])
  })

  it('does not start a task when a stop is asked for while it is being prepared', async () => {
    const h = harness(makeRun([item('a-1', 'pending')]), {})
    let stop: BatchStop | null = null
    h.deps.stopping = () => stop
    const prepare = h.deps.prepare
    h.deps.prepare = async (k, t) => {
      stop = 'after'
      return prepare(k, t)
    }
    const { run } = await runBatchLoop(h.deps)
    expect(h.log.started).toHaveLength(0)
    expect(run).toMatchObject({ status: 'stopped', items: [{ status: 'pending' }] })
  })

  it('does not count a session that ran on another model', async () => {
    const h = harness(makeRun([item('a-1', 'pending')]), { 'a-1': { ...SOLVED, ranOn: 'other/model.gguf' } })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items[0].status).toBe('error')
    expect(run!.items[0].error).toMatch(/other\/model\.gguf/)
  })

  it('"Stop after this task" lets the task finish', async () => {
    const h = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending')]), { 'a-1': SOLVED }, { stopAt: [1000, 'after'] })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items.map((i) => i.status)).toEqual(['solved', 'pending'])
    expect(run!.status).toBe('stopped')
    expect(h.log.stops).toBe(0)
  })

  it('resumes: tasks already run are left alone', async () => {
    const h = harness(makeRun([item('a-1', 'wrong', { conversationId: 'old' }), item('a-2', 'pending')], { status: 'stopped' }), { 'a-2': SOLVED })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items).toMatchObject([{ status: 'wrong', conversationId: 'old' }, { status: 'solved', conversationId: 'c1' }])
    expect(h.log.started).toHaveLength(1)
  })

  it('waits while another reply is running, then starts', async () => {
    const h = harness(makeRun([item('a-1', 'pending')]), { 'a-1': SOLVED }, { busyUntil: 30_000 })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items[0]).toMatchObject({ status: 'solved' })
    expect(run!.items[0].startedAt).toBeGreaterThanOrEqual(30_000)
  })

  it('tries a task again when another reply got in first', async () => {
    const h = harness(makeRun([item('a-1', 'pending')]), { 'a-1': SOLVED })
    let refused = 0
    const start = h.deps.startSession
    h.deps.startSession = async (o) => (refused++ < 2 ? null : start(o))
    const { run } = await runBatchLoop(h.deps)
    expect(refused).toBe(3)
    expect(run!.items[0].status).toBe('solved')
  })

  it('records a task that cannot be prepared and carries on', async () => {
    const h = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending')]), { 'a-1': { prepareError: 'Practice task a-1 is not in C:\\kit.' }, 'a-2': SOLVED })
    const { run } = await runBatchLoop(h.deps)
    expect(run!.items[0]).toMatchObject({ status: 'error', error: 'Practice task a-1 is not in C:\\kit.' })
    expect(run!.items[0].conversationId).toBeUndefined()
    expect(run!.items[1].status).toBe('solved')
  })

  it('shows each new session only while the user is watching the one before', async () => {
    const watching = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending')]), {})
    await runBatchLoop(watching.deps)
    expect(watching.log.started.map((s) => s.follow)).toEqual([true, true])
    const away = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending')]), {}, { watching: false })
    await runBatchLoop(away.deps)
    expect(away.log.started.map((s) => s.follow)).toEqual([true, false])
  })

  it('ends quietly when the run is removed, and reports a failure instead of throwing', async () => {
    const gone = harness(makeRun([item('a-1', 'pending')]), {})
    gone.remove()
    expect(await runBatchLoop(gone.deps)).toEqual({ run: undefined, error: undefined })

    const broken = harness(makeRun([item('a-1', 'pending'), item('a-2', 'pending')]), {})
    broken.deps.chatIdle = () => {
      throw new Error('store is gone')
    }
    const r = await runBatchLoop(broken.deps)
    expect(r.error).toBe('store is gone')
    expect(r.run).toMatchObject({ status: 'stopped' })
    expect(r.run!.items.map((i) => i.status)).toEqual(['pending', 'pending'])
    expect(broken.listeners.size).toBe(0)
  })
})
