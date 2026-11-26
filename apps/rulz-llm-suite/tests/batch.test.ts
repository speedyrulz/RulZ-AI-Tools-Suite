import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  batchAdapters,
  cleanBatchRun,
  DEFAULT_BATCH_OPTIONS,
  isRemoteDenial,
  isScored,
  isStaleChatError,
  itemResult,
  median,
  planBatch,
  REMOTE_DENIAL,
  sameBatchAdapters,
  sessionStats,
  settleRun,
  summarize,
  UNATTENDED_DENIAL,
  type BatchItem,
  type BatchOptions,
  type BatchRun
} from '@shared/batch'
import { argsStayInside, remoteRoots, runsWithoutAsking, starterToolsThatAsk } from '@shared/re'
import type { ChatMessage, Conversation, RePracticeTask, ReSession } from '@shared/types'
import { BatchStore, MAX_BATCH_RUNS } from '../src/main/re/batches'

const task = (id: string, family: string, tier = 'easy'): RePracticeTask => ({ id, family, tier, binary: `${id}.exe`, prompt: `Solve {exe}`, set: 'original' })
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

describe('adapters of a run and the host of a task', () => {
  it('records what the model loaded, and tells the same adapters from other ones', () => {
    expect(batchAdapters(undefined)).toBeUndefined()
    expect(batchAdapters([])).toBeUndefined()
    expect(batchAdapters([{ name: 'a', scale: 0.8 }])).toEqual([{ name: 'a', scale: 0.8 }])
    expect(sameBatchAdapters(undefined, [])).toBe(true)
    expect(sameBatchAdapters([{ name: 'a', scale: 1 }], [{ name: 'a', scale: 1 }])).toBe(true)
    expect(sameBatchAdapters([{ name: 'a', scale: 1 }], [{ name: 'a', scale: 0.5 }])).toBe(false)
    expect(sameBatchAdapters([{ name: 'a', scale: 1 }], undefined)).toBe(false)
  })

  it('keeps the host a task ran on', () => {
    const r = cleanBatchRun({ id: 'ok', items: [{ taskId: 'a', host: 'RTX 4090 (12)', lost: 1 }, { taskId: 'b', host: 7 }] })!
    expect(r.items[0]).toMatchObject({ host: 'RTX 4090 (12)', lost: 1 })
    expect(r.items[1]).not.toHaveProperty('host')
  })

  it('keeps why a run stopped by itself, and drops anything that is not a sentence', () => {
    const base = { id: 'ok', items: [], status: 'stopped' }
    expect(cleanBatchRun({ ...base, stopReason: 'The daily limit is used up.' })!.stopReason).toBe('The daily limit is used up.')
    expect(cleanBatchRun({ ...base, stopReason: 'x'.repeat(900) })!.stopReason).toHaveLength(500)
    for (const bad of [7, null, {}, ['a'], '', true]) expect(cleanBatchRun({ ...base, stopReason: bad })).not.toHaveProperty('stopReason')
    expect(cleanBatchRun({ ...base })).not.toHaveProperty('stopReason')
    // A finished run has nothing to explain.
    expect(cleanBatchRun({ ...base, status: 'done', stopReason: 'old' })).not.toHaveProperty('stopReason')
  })

  it('counts a call refused with the old wording of the rental fence, and with the new one', () => {
    const oldText =
      'Not run: without asking, a model on a rented instance may only use static analysis, on absolute paths in this task' +
      String.fromCharCode(0x2019) +
      's folder. Continue with the given target.'
    expect(oldText).not.toBe(REMOTE_DENIAL)
    expect(isRemoteDenial(oldText)).toBe(true)
    expect(isRemoteDenial(REMOTE_DENIAL)).toBe(true)
    expect(isRemoteDenial('The user declined to run this tool.')).toBe(false)
    expect(isRemoteDenial(undefined)).toBe(false)
    // It names both kinds of model it is shown to.
    expect(REMOTE_DENIAL).toContain('rented instance')
    expect(REMOTE_DENIAL).toContain('OpenRouter')
    const c = conv({}, [
      { role: 'user' },
      {
        toolCalls: [
          { id: '1', name: 'rea__run', arguments: '{}', status: 'denied', result: oldText },
          { id: '2', name: 'rea__run', arguments: '{}', status: 'denied', result: REMOTE_DENIAL },
          { id: '3', name: 'rea__run', arguments: '{}', status: 'denied', result: 'The user declined to run this tool.' }
        ],
        stats: { promptTokens: 10, predictedTokens: 1 }
      }
    ])
    expect(sessionStats(c).denied).toBe(2)
  })

  it('recognises the refusal of a stale chat copy, however the error was wrapped on its way over IPC', () => {
    expect(isStaleChatError("Error invoking remote method 'conversations:save': Error: The batch run is using this chat, so this copy was not saved.")).toBe(true)
    expect(isStaleChatError('ENOSPC: no space left on device')).toBe(false)
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

  it('does not settle the run the pool is running when a reloaded window asks for the list', async () => {
    const path = join(dir, 're-batches.json')
    const store = new BatchStore(path)
    await store.save(makeRun([item('a-1', 'running'), item('a-2', 'pending')], { id: 'live', createdAt: 5 }))
    await store.save(makeRun([item('a-1', 'running')], { id: 'cut', createdAt: 4 }))
    let active: string | null = 'live'
    store.setActive(() => active)
    const first = await store.list()
    expect(first.map((r) => `${r.id}:${r.status}`)).toEqual(['live:running', 'cut:stopped'])
    expect(first[0].items.map((i) => i.status)).toEqual(['running', 'pending'])
    // Once the pool is done with it, a list settles whatever is still marked running.
    active = null
    expect((await store.list()).map((r) => `${r.id}:${r.status}`)).toEqual(['live:stopped', 'cut:stopped'])
    expect((await store.get('live'))?.id).toBe('live')
    expect(await store.get('none')).toBeUndefined()
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

describe('tool calls of a model on a rented instance', () => {
  const roots = ['C:\\work\\t\\1']
  const tool = (server: string, name: string) => ({ qualifiedName: `${server}__${name}`, server, name })
  const open = { confirmToolCalls: false, allowedTools: ['rea__start_gdb_session', 'shell__run'], re: { autoApproveStatic: true } }
  const call = (args: unknown) => ({ args: typeof args === 'string' ? args : JSON.stringify(args), roots, rea: 'private' as const })

  it('does not count "do not ask" or the always-allowed tools, inside or outside RE', () => {
    expect(runsWithoutAsking(open, tool('rea', 'start_gdb_session'), true, call({}))).toBe(false)
    expect(runsWithoutAsking(open, tool('shell', 'run'), true, call({}))).toBe(false)
    expect(runsWithoutAsking(open, tool('shell', 'run'), false, call({}))).toBe(false)
    expect(runsWithoutAsking(open, tool('rea', 'xrefs'), false, call({}))).toBe(false)
    const ask = { confirmToolCalls: true, allowedTools: [] as string[], re: { autoApproveStatic: false } }
    expect(runsWithoutAsking(ask, tool('rea', 'xrefs'), true, call({}))).toBe(false)
    expect(runsWithoutAsking(ask, tool('rea', 'open_binary'), true, call({ path: 'C:\\work\\t\\1\\a.exe' }))).toBe(false)
    expect(runsWithoutAsking({ ...ask, allowedTools: ['rea__open_binary'] }, tool('rea', 'open_binary'), true, call({ path: 'C:\\work\\t\\1\\a.exe' }))).toBe(true)
    expect(runsWithoutAsking(open, tool('rea', 'xrefs'), true, call({ address: '0x1000' }))).toBe(true)
    expect(runsWithoutAsking(open, tool('calc', 'calc'), true, call({ expression: '~5 / 2' }))).toBe(true)
    // A local model keeps its settings.
    expect(runsWithoutAsking(open, tool('shell', 'run'), false)).toBe(true)
  })

  it('lets static analysis open the task folder only', () => {
    const opens = (path: string): boolean => runsWithoutAsking(open, tool('rea', 'open_binary'), true, call({ path }))
    for (const p of ['C:\\work\\t\\1\\a.exe', 'c:/WORK/t/1/a.exe', 'C:\\work\\t\\1', 'C:\\work\\\\t\\1\\sub\\b.dll']) expect(opens(p), p).toBe(true)
    for (const p of [
      'C:\\work\\t\\1\\..\\2\\a.exe', 'C:\\work\\t\\1\\.. \\x', 'C:\\work\\t\\10\\a.exe', 'C:\\work\\t', 'C:\\Users\\u\\.ssh\\id_ed25519', 'D:\\work\\t\\1\\a.exe',
      '\\\\host\\share\\a', '//host/share/a', '\\\\?\\C:\\work\\t\\1\\a.exe', '\\work\\t\\1\\a.exe', 'C:a.exe', 'C:\\work\\t\\1\\a.exe:stream', '%USERPROFILE%\\a', '~/a', 'a.exe', '', '/mnt/c/work/t/1/a.exe', 'file:///C:/work/t/1/a.exe', 'http://127.0.0.1:9222'
    ]) expect(opens(p), p).toBe(false)
    expect(runsWithoutAsking(open, tool('rulz', 'list_directory'), true, call({ path: 'C:\\Users' }))).toBe(false)
    expect(runsWithoutAsking(open, tool('rulz', 'find_files'), true, call({ root: 'C:\\work\\t\\1', pattern: '*.exe' }))).toBe(true)
    expect(runsWithoutAsking(open, tool('rulz', 'find_files'), true, call({ root: 'C:\\', pattern: '*.key' }))).toBe(false)
  })

  it('does not read a program another chat opened, or files next to the target in a clone', () => {
    const other = { ...call({}), rea: 'other' as const }
    expect(runsWithoutAsking(open, tool('rea', 'list_strings'), true, other)).toBe(false)
    expect(runsWithoutAsking(open, tool('rea', 'list_strings'), true, call({}))).toBe(true)
    expect(runsWithoutAsking(open, tool('rea', 'open_binary'), true, { ...call({ path: 'C:\\work\\t\\1\\a.exe' }), rea: 'other' })).toBe(true)
    for (const n of ['get_evidence_bundle', 'compare_functions']) {
      expect(runsWithoutAsking(open, tool('rea', n), true, { ...call({}), rea: 'opened' }), n).toBe(false)
      expect(runsWithoutAsking(open, tool('rea', n), true, call({})), n).toBe(true)
    }
    const reads = (path: string): boolean => runsWithoutAsking(open, tool('rulz', 'read_file'), true, call({ path }))
    for (const p of ['src/main.c', 'NOTES.md', 'C:\\work\\t\\1\\src\\a.c']) expect(reads(p), p).toBe(true)
    for (const p of ['C:\\work\\t\\secrets.txt', '..\\x', 'a/../../x', 'C:x', '\\x', '"C:\\x"', "'\\x'", '%TEMP%\\x', '\\\\host\\share\\x', '']) expect(reads(p), p).toBe(false)
    expect(runsWithoutAsking(open, tool('rulz', 'write_file'), true, call({ path: 'src/main.c', content: 'C:\\Windows' }))).toBe(true)
  })

  it('asks before read_pdf: not static, never unasked on a rented instance, and only the normal settings let it run', () => {
    const ask = { confirmToolCalls: true, allowedTools: [], re: { autoApproveStatic: true } }
    const pdf = tool('rulz', 'read_pdf')
    const inside = call({ path: 'C:\\work\\t\\1\\doc.pdf' })
    // RE session with auto-approve of static tools on: other rulz tools run, read_pdf asks.
    expect(runsWithoutAsking(ask, tool('rulz', 'list_directory'), true)).toBe(true)
    expect(runsWithoutAsking(ask, pdf, true)).toBe(false)
    expect(runsWithoutAsking(ask, pdf, false)).toBe(false)
    // The user's own settings still work as for any tool.
    expect(runsWithoutAsking({ ...ask, allowedTools: ['rulz__read_pdf'] }, pdf, true)).toBe(true)
    expect(runsWithoutAsking({ ...ask, confirmToolCalls: false }, pdf, false)).toBe(true)
    // A model on a rented instance never gets it unasked, not even for a file inside the session's folder, with every setting open.
    expect(runsWithoutAsking({ ...ask, allowedTools: ['rulz__read_pdf'], confirmToolCalls: false }, pdf, true, inside)).toBe(false)
    expect(runsWithoutAsking({ ...ask, allowedTools: ['rulz__read_pdf'], confirmToolCalls: false }, pdf, false, inside)).toBe(false)
    expect(runsWithoutAsking(open, pdf, true, inside)).toBe(false)
  })

  it('checks nested and list arguments and paths under any key, but not search text', () => {
    expect(argsStayInside('{"pattern":"C:\\\\Windows"}', roots)).toBe(true)
    expect(argsStayInside('{"query":"http://example"}', roots)).toBe(true)
    expect(argsStayInside('{"anything":"C:\\\\Windows\\\\x"}', roots)).toBe(false)
    expect(argsStayInside('{"cdp_endpoint":"http://127.0.0.1:9222","target_id":"1"}', roots)).toBe(false)
    expect(argsStayInside('{"additional_paths":["C:\\\\work\\\\t\\\\1\\\\a.js","C:\\\\x.js"]}', roots)).toBe(false)
    expect(argsStayInside('{"a":{"b":[{"local_path":"C:\\\\x"}]}}', roots)).toBe(false)
    expect(argsStayInside('', roots)).toBe(true)
    expect(argsStayInside('not json', roots)).toBe(false)
    expect(argsStayInside('{"overwrite":true}', roots)).toBe(false)
    expect(argsStayInside('{"overwrite":false}', roots)).toBe(true)
    expect(argsStayInside('[]', roots)).toBe(false)
    expect(argsStayInside('{"path":"C:\\\\work\\\\t\\\\1\\\\a"}', [])).toBe(false)
  })

  it('roots: the target, the work folder of a practice task, the output folder of a clone', () => {
    const practice = { taskId: 'a', family: 'f', tier: 'easy', kitDir: 'C:\\kit' }
    expect(remoteRoots({ target: 'C:\\dl\\a.exe' })).toEqual(['C:\\dl\\a.exe'])
    expect(remoteRoots({ target: '"C:\\work\\t\\1\\a.exe"', practice })).toEqual(['C:\\work\\t\\1\\a.exe', 'C:\\work\\t\\1'])
    expect(remoteRoots(undefined)).toEqual([])
  })
})
