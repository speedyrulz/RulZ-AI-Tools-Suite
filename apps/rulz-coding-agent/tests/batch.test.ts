// Practice batch runs: the run loop, stopping, the time limit, the saved runs and the Suite call.
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_BATCH_OPTIONS } from '@shared/batch'
import { DEFAULT_SETTINGS } from '@shared/defaults'
import type { BatchOptions, BatchRun, BatchState, PendingPrompt } from '@shared/types'
import { BATCH_ACTIVE_MESSAGE, BatchRunner, BatchStore, MAX_BATCH_RUNS } from '../src/main/agent/batch'
import { ProjectSession } from '../src/main/agent/session'
import { CopySnapshots } from '../src/main/project/snapshots'
import { emptyProject, ProjectStore } from '../src/main/project/store'
import { RulzClient } from '../src/main/rulz/client'
import { startMockRulz, type MockTurn } from './helpers/mockRulz'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-batch-'))
afterAll(() => rmSync(base, { recursive: true, force: true }))
let counter = 0

const MODEL = 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M'
const call = (name: string, args: unknown) => ({ name, args })
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
const until = async (test: () => boolean, ms = 10_000): Promise<void> => {
  const end = Date.now() + ms
  while (!test()) {
    if (Date.now() > end) throw new Error('timed out waiting')
    await sleep(10)
  }
}

async function harness(turns: MockTurn[], opts: { msPerMinute?: number } = {}) {
  const mock = await startMockRulz({ turns })
  const root = join(base, `project-${++counter}`)
  mkdirSync(root, { recursive: true })
  const dataDir = join(base, `data-${counter}`)
  const store = new ProjectStore(dataDir)
  const settings = { ...DEFAULT_SETTINGS, firstRunDone: true, model: MODEL }
  const prompts: PendingPrompt[] = []
  const states: BatchState[] = []
  const trash = async (abs: string): Promise<void> => {
    mkdirSync(join(dataDir, 'trash'), { recursive: true })
    await fs.rename(abs, join(dataDir, 'trash', `${Date.now()}-${basename(abs)}`))
  }
  const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
  const batchStore = new BatchStore(join(dataDir, 'batch-runs.json'))
  let runner!: BatchRunner
  const session = new ProjectSession(emptyProject(root), {
    client,
    settings: () => settings,
    store,
    emit: (e) => {
      if (e.type === 'prompt' && e.prompt) prompts.push(e.prompt)
      runner.onAgentEvent(e)
    },
    activity: () => undefined,
    models: () => [{ id: MODEL, loaded: true, loadedContext: 65536 }],
    refreshModels: () => undefined,
    snapshotter: async () => new CopySnapshots(store.backupDir(root), root, trash),
    trash
  })
  runner = new BatchRunner({
    client,
    store: batchStore,
    session: () => session,
    modelName: () => MODEL,
    broadcast: (s) => states.push(s),
    msPerMinute: opts.msPerMinute ?? 60_000
  })
  const options = (o: Partial<BatchOptions> = {}): BatchOptions => ({ ...DEFAULT_BATCH_OPTIONS, ...o })
  return {
    mock,
    root,
    session,
    runner,
    batchStore,
    prompts,
    states,
    options,
    dataDir,
    runFile: join(dataDir, 'batch-runs.json'),
    close: async () => {
      session.stop()
      await runner.idle()
      await runner.flush()
      await session.close()
      await mock.close()
    }
  }
}

const finished = async (h: Awaited<ReturnType<typeof harness>>): Promise<BatchRun> => {
  await h.runner.idle()
  await h.runner.flush()
  return h.batchStore.list()[0]
}

describe('batch runs', () => {
  it('runs a batch of two tasks, solving one and failing one', async () => {
    const h = await harness([
      { content: 'Looking.', toolCalls: [call('list_files', {})] },
      { content: 'The key is 42.\nANSWER: 42' },
      { content: 'ANSWER: 7' }
    ])
    try {
      const state = await h.runner.start(h.root, ' base ', h.options(), true)
      expect(state.activeId).not.toBeNull()
      expect(h.runner.active).toBe(true)
      const run = await finished(h)
      expect(h.runner.active).toBe(false)
      expect(run).toMatchObject({ label: 'base', status: 'done', modelName: MODEL, kitDir: 'k', record: true, projectPath: h.root })
      expect(run.finishedAt).toBeGreaterThanOrEqual(run.createdAt)
      expect(run.items.map((i) => [i.taskId, i.status])).toEqual([
        ['xor-1', 'solved'],
        ['xor-2', 'wrong']
      ])
      expect(run.items[0]).toMatchObject({ answer: '42', expected: '42', toolRounds: 1 })
      expect(run.items[0].ms).toBeGreaterThanOrEqual(0)
      expect(run.items[1]).toMatchObject({ answer: '7', expected: '42', toolRounds: 0 })
      expect(run.items.every((i) => i.taskRecordId)).toBe(true)
      expect(h.session.data.tasks.map((t) => t.re?.practice?.taskId)).toEqual(['xor-1', 'xor-2'])
      // Recorded in the Suite, graded.
      expect(h.mock.reSessions.filter((r) => r.body.outcome).map((r) => r.body.outcome)).toEqual(['solved', 'failed'])
      // The window heard about every change, ending with the finished run.
      expect(h.states.length).toBeGreaterThan(4)
      expect(h.states.at(-1)).toMatchObject({ activeId: null, stopping: null })
      expect(h.states.at(-1)!.runs[0].status).toBe('done')
      // Saved to disk, newest first.
      const onDisk = JSON.parse(readFileSync(h.runFile, 'utf8')) as BatchRun[]
      expect(onDisk[0].id).toBe(run.id)
      expect((await h.runner.state()).runs[0].id).toBe(run.id)
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stops a task at the time limit and goes on with the next', async () => {
    const h = await harness([{ delay: 20_000, content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], { msPerMinute: 300 })
    try {
      await h.runner.start(h.root, '', h.options({ minutesPerTask: 1 }), true)
      const run = await finished(h)
      expect(run.status).toBe('done')
      expect(run.items.map((i) => i.status)).toEqual(['timeout', 'solved'])
      expect(run.items[0].ms).toBeGreaterThanOrEqual(250)
      expect(h.session.data.tasks[0].outcome).toBe('stopped')
      // The stopped task was not graded, so the Suite is told it failed.
      expect(h.mock.reSessions.find((r) => r.id === `agent-${h.session.data.tasks[0].id}` && r.body.outcome === 'failed')?.body).toMatchObject({
        outcome: 'failed',
        practice: { taskId: 'xor-1', family: 'xor', tier: 't1' }
      })
    } finally {
      await h.close()
    }
  }, 60_000)

  it('does not tell the Suite about a timeout when the run is not recorded', async () => {
    const h = await harness([{ delay: 20_000, content: 'x' }, { content: 'ANSWER: 42' }], { msPerMinute: 200 })
    try {
      await h.runner.start(h.root, '', h.options({ minutesPerTask: 1, maxTasks: 1 }), false)
      const run = await finished(h)
      expect(run.items.map((i) => i.status)).toEqual(['timeout'])
      expect(h.mock.reSessions.filter((r) => r.body.outcome)).toEqual([])
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stop after this task finishes the task in progress, then stops', async () => {
    const h = await harness([{ delay: 400, content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }])
    try {
      await h.runner.start(h.root, '', h.options(), true)
      await until(() => h.mock.requests.length > 0)
      const s = h.runner.stop('after')
      expect(s.stopping).toBe('after')
      const run = await finished(h)
      expect(run.status).toBe('stopped')
      expect(run.items.map((i) => i.status)).toEqual(['solved', 'pending'])
      expect(h.states.at(-1)).toMatchObject({ activeId: null, stopping: null })
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stop now stops the task too, which goes back to pending', async () => {
    const h = await harness([{ delay: 20_000, content: 'ANSWER: 42' }])
    try {
      await h.runner.start(h.root, '', h.options(), true)
      await until(() => h.mock.requests.length > 0)
      expect(h.session.isRunning).toBe(true)
      h.runner.stop('now')
      const run = await finished(h)
      expect(run.status).toBe('stopped')
      expect(run.items.map((i) => i.status)).toEqual(['pending', 'pending'])
      expect(run.items[0].taskRecordId).toBeUndefined()
      expect(h.session.isRunning).toBe(false)
    } finally {
      await h.close()
    }
  }, 60_000)

  it('records the owner pressing Stop on a task as stopped, and carries on with the next', async () => {
    const h = await harness([{ delay: 20_000, content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }])
    try {
      await h.runner.start(h.root, '', h.options(), true)
      await until(() => h.mock.requests.length > 0)
      h.session.stop()
      const run = await finished(h)
      expect(run.status).toBe('done')
      expect(run.items.map((i) => i.status)).toEqual(['stopped', 'solved'])
    } finally {
      await h.close()
    }
  }, 60_000)

  it('approves tools automatically in its tasks, and asks again in a normal task', async () => {
    const h = await harness([
      { toolCalls: [call('run_command', { command: 'Get-Content a.txt', summary: 'Reading' })] },
      { content: 'ANSWER: 42' },
      { toolCalls: [call('run_command', { command: 'Get-Content a.txt', summary: 'Reading' })] },
      { content: 'Done.' }
    ])
    try {
      writeFileSync(join(h.root, 'a.txt'), 'hello')
      await h.runner.start(h.root, '', h.options({ maxTasks: 1 }), true)
      const run = await finished(h)
      expect(run.items[0].status).toBe('solved')
      expect(h.prompts).toEqual([])
      // A normal task is asked as before (nobody answers here, so Stop ends it).
      h.session.start('Look at a.txt')
      await until(() => h.prompts.length > 0)
      expect(h.prompts[0]).toMatchObject({ kind: 'approval' })
      h.session.stop()
    } finally {
      await h.close()
    }
  }, 60_000)

  it('refuses what would collide with a run, and what makes no sense', async () => {
    const h = await harness([{ delay: 20_000, content: 'ANSWER: 42' }])
    try {
      await expect(h.runner.start(join(base, 'elsewhere'), '', h.options(), true)).rejects.toThrow('That project is not open.')
      await expect(h.runner.start(h.root, '', h.options({ family: 'nothing' }), true)).rejects.toThrow('No tasks match.')
      expect(() => h.runner.assertIdle()).not.toThrow()
      const state = await h.runner.start(h.root, '', h.options(), true)
      expect(() => h.runner.assertIdle()).toThrow(BATCH_ACTIVE_MESSAGE)
      expect(BATCH_ACTIVE_MESSAGE).toBe('A batch run is going. Stop it first.')
      await expect(h.runner.start(h.root, '', h.options(), true)).rejects.toThrow('already going')
      await expect(h.runner.delete(state.activeId!)).rejects.toThrow('Stop it first')
      h.runner.stop('now')
      await finished(h)
      expect(() => h.runner.assertIdle()).not.toThrow()
    } finally {
      await h.close()
    }
  }, 60_000)

  it('refuses to start while a task is running', async () => {
    const h = await harness([{ delay: 20_000, content: 'x' }])
    try {
      h.session.start('Do something')
      await until(() => h.mock.requests.length > 0)
      await expect(h.runner.start(h.root, '', h.options(), true)).rejects.toThrow('A task is already running')
      h.session.stop()
    } finally {
      await h.close()
    }
  }, 30_000)

  it('deletes a saved run', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }])
    try {
      await h.runner.start(h.root, 'one', h.options({ maxTasks: 1 }), true)
      const run = await finished(h)
      const after = await h.runner.delete(run.id)
      expect(after.runs).toEqual([])
      expect(JSON.parse(readFileSync(h.runFile, 'utf8'))).toEqual([])
    } finally {
      await h.close()
    }
  }, 30_000)
})

describe('saved batch runs', () => {
  const run = (id: string, createdAt: number, status: BatchRun['status']): BatchRun => ({
    id,
    label: id,
    createdAt,
    status,
    projectPath: 'C:\\p',
    modelName: 'm',
    kitDir: 'k',
    record: true,
    options: DEFAULT_BATCH_OPTIONS,
    items: [
      { taskId: 'xor-1', family: 'xor', tier: 't1', attempt: 1, status: 'solved' },
      { taskId: 'xor-2', family: 'xor', tier: 't1', attempt: 1, status: 'running', startedAt: 5 }
    ]
  })

  it('reloads a run left running as stopped, with its running task pending, newest first', async () => {
    const dir = mkdtempSync(join(base, 'store-'))
    const file = join(dir, 'batch-runs.json')
    writeFileSync(file, JSON.stringify([run('old', 1, 'done'), run('cut', 3, 'running'), { nonsense: true }]))
    const store = new BatchStore(file)
    await store.load()
    expect(store.list().map((r) => [r.id, r.status])).toEqual([
      ['cut', 'stopped'],
      ['old', 'done']
    ])
    expect(store.get('cut')!.items.map((i) => i.status)).toEqual(['solved', 'pending'])
    // Through the runner too.
    const state = await new BatchRunner({
      client: { reOverview: async () => Promise.reject(new Error('x')), preparePractice: async () => Promise.reject(new Error('x')), describeReSession: async () => false },
      store: new BatchStore(file),
      session: () => null,
      modelName: () => 'm',
      broadcast: () => undefined
    }).state()
    expect(state).toMatchObject({ activeId: null, stopping: null })
    expect(state.runs[0].status).toBe('stopped')
  })

  it('starts empty without a file, and keeps a corrupt file aside', async () => {
    const dir = mkdtempSync(join(base, 'store-'))
    const none = new BatchStore(join(dir, 'none.json'))
    await none.load()
    expect(none.list()).toEqual([])
    const bad = join(dir, 'bad.json')
    writeFileSync(bad, '{ not json')
    const store = new BatchStore(bad)
    await store.load()
    expect(store.list()).toEqual([])
  })

  it('keeps at most the newest runs, and saves in order', async () => {
    const dir = mkdtempSync(join(base, 'store-'))
    const file = join(dir, 'batch-runs.json')
    const store = new BatchStore(file, 3)
    await store.load()
    for (let i = 1; i <= 5; i++) void store.put(run(`r${i}`, i, 'done'))
    await store.flush()
    expect(store.list().map((r) => r.id)).toEqual(['r5', 'r4', 'r3'])
    expect((JSON.parse(readFileSync(file, 'utf8')) as BatchRun[]).map((r) => r.id)).toEqual(['r5', 'r4', 'r3'])
    expect(MAX_BATCH_RUNS).toBe(200)
  })
})

describe('changing the kit folder', () => {
  it('posts the folder to RulZ LLM Suite and returns its tasks', async () => {
    const mock = await startMockRulz()
    try {
      const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      const practice = await client.setReKit('C:\\kits\\one')
      expect(mock.kits).toEqual([{ kitDir: 'C:\\kits\\one' }])
      expect(practice).toEqual({ kitDir: 'C:\\kits\\one', tasks: [] })
    } finally {
      await mock.close()
    }
  })

  it('says to update an older Suite that does not know the call', async () => {
    const mock = await startMockRulz({ lmStudio: true })
    try {
      const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      await expect(client.setReKit('C:\\kits\\one')).rejects.toThrow('Update RulZ LLM Suite to 0.27.0 or later to change the kit folder from here.')
    } finally {
      await mock.close()
    }
  })
})
