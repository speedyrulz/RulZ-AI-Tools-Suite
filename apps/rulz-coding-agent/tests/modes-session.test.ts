// Tasks and batch runs with LLM Suite's modes and rentals: the headers of every request, the sampling
// and prompt a task gets, which model it may use, and what a batch run does about rentals.
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_BATCH_OPTIONS } from '@shared/batch'
import { DEFAULT_SETTINGS } from '@shared/defaults'
import { CLOUD_FATAL_CODES, CLOUD_NEEDS_YES, CLOUD_OLD_APP, chooseModel, RENTAL_NEEDS_YES, RENTAL_OLD_APP } from '@shared/modes'
import type { AgentEvent, AppSettings, BatchOptions, BatchRun, ModelInfo, ReTaskInfo, TaskRecord } from '@shared/types'
import { BatchRunner, BatchStore } from '../src/main/agent/batch'
import { isContextOverflow, ProjectSession } from '../src/main/agent/session'
import { CopySnapshots } from '../src/main/project/snapshots'
import { emptyProject, ProjectStore } from '../src/main/project/store'
import { RulzClient, ServerError } from '../src/main/rulz/client'
import { startMockRulz, type MockOptions, type MockTurn } from './helpers/mockRulz'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-modes-'))
afterAll(() => rmSync(base, { recursive: true, force: true }))
let counter = 0

const LOCAL = 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M'
const RENTED = 'Big-Rented-Model'
const call = (name: string, args: unknown) => ({ name, args })
const rental = { label: 'RTX 5090', reconnecting: false }

/** The body LLM Suite 0.35 answers for /modes. */
const modesBody = (patch: { coding?: Record<string, unknown>; re?: Record<string, unknown>; api_override?: unknown } = {}): Record<string, unknown> => ({
  object: 'model.modes',
  id: LOCAL,
  state: 'ready',
  instance: 'i1',
  api_override: patch.api_override ?? null,
  modes: {
    coding: { own_load: false, sampling: null, thinking: null, system_prompt: '', matches: true, ...patch.coding },
    re: { own_load: false, sampling: null, thinking: null, system_prompt: '', matches: true, ...patch.re }
  }
})

async function harness(
  turns: MockTurn[],
  opts: {
    mock?: Partial<MockOptions>
    models?: MockOptions['models']
    settings?: Partial<AppSettings>
    nCtx?: number
    /** Called when a task ends, before the batch runner hears of it (to act between two tasks). */
    onTaskEnd?: (h: { settings: AppSettings; setList: (l: ModelInfo[]) => void }) => void
    /** The model list the app believes, when it should differ from the server's (a stale list). */
    list?: ModelInfo[]
  } = {}
) {
  const mock = await startMockRulz({ turns, nCtx: opts.nCtx, models: opts.models ?? [{ id: LOCAL, state: 'loaded', loaded_context_length: opts.nCtx ?? 65536 }], ...opts.mock })
  const root = join(base, `project-${++counter}`)
  mkdirSync(root, { recursive: true })
  const dataDir = join(base, `data-${counter}`)
  const store = new ProjectStore(dataDir)
  const settings: AppSettings = { ...DEFAULT_SETTINGS, firstRunDone: true, model: LOCAL, approval: 'handsoff', ...opts.settings }
  const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
  let list = opts.list ?? (await client.listModels())
  const events: AgentEvent[] = []
  const trash = async (abs: string): Promise<void> => {
    mkdirSync(join(dataDir, 'trash'), { recursive: true })
    await fs.rename(abs, join(dataDir, 'trash', `${Date.now()}-${basename(abs)}`))
  }
  const ended: ((t: TaskRecord) => void)[] = []
  let runner!: BatchRunner
  const setList = (l: ModelInfo[]): void => {
    list = l
  }
  const session = new ProjectSession(emptyProject(root), {
    client,
    settings: () => settings,
    store,
    emit: (e) => {
      events.push(e)
      if (e.type === 'task-end') {
        opts.onTaskEnd?.({ settings, setList })
        ended.shift()?.(e.task)
        runner?.onAgentEvent(e)
      }
    },
    activity: () => undefined,
    models: () => list,
    refreshModels: () => undefined,
    snapshotter: async () => new CopySnapshots(store.backupDir(root), root, trash),
    trash
  })
  const batchStore = new BatchStore(join(dataDir, 'batch-runs.json'))
  runner = new BatchRunner({
    client,
    store: batchStore,
    session: () => session,
    modelName: () => {
      const c = chooseModel(settings.model, settings.modelRental, list, settings.modelCloud)
      return c.ok ? c.model : settings.model
    },
    model: () => chooseModel(settings.model, settings.modelRental, list, settings.modelCloud),
    broadcast: () => undefined
  })
  const notices = (): { text: string; tone: string }[] =>
    session.data.timeline.flatMap((i) => (i.kind === 'notice' ? [{ text: i.text, tone: i.tone }] : []))
  return {
    mock,
    session,
    settings,
    runner,
    batchStore,
    root,
    notices,
    setList,
    options: (o: Partial<BatchOptions> = {}): BatchOptions => ({ ...DEFAULT_BATCH_OPTIONS, ...o }),
    run: (text: string, startOpts?: { re?: ReTaskInfo; batch?: boolean; rentalOk?: boolean }): Promise<TaskRecord> =>
      new Promise((resolve) => {
        ended.push(resolve)
        session.start(text, startOpts)
      }),
    finished: async (): Promise<BatchRun> => {
      await runner.idle()
      await runner.flush()
      return batchStore.list()[0]
    },
    close: async () => {
      session.stop()
      await runner.idle()
      await runner.flush()
      await session.close()
      await mock.close()
    }
  }
}

describe('headers on every request of a task', () => {
  it('a coding task sends the mode and refuses rentals on turns and /props, and says nothing with an older LLM Suite', async () => {
    const h = await harness([{ toolCalls: [call('list_files', {})] }, { content: 'Done.' }])
    try {
      const task = await h.run('Look around')
      expect(task.outcome).toBe('done')
      expect(h.mock.requests).toHaveLength(2)
      for (const r of h.mock.requests) expect(r).toMatchObject({ mode: 'coding', rental: 'deny' })
      const props = h.mock.headerLog.filter((l) => l.path === '/props')
      expect(props).toEqual([{ path: '/props', method: 'GET', model: LOCAL, mode: 'coding', rental: 'deny' }])
      // The modes route is a 404 here (an older LLM Suite): no notice, and the body is what it always was.
      expect(h.notices()).toEqual([])
      expect(h.mock.requests[0].body).toMatchObject({ temperature: 0.7, top_p: 0.8, top_k: 20, repeat_penalty: 1.05, model: LOCAL })
    } finally {
      await h.close()
    }
  }, 30_000)

  it('the summary request carries the same headers, and the modes were read once for the task', async () => {
    const big = (n: number) => ({ toolCalls: [call('run_command', { command: `1..${n} | ForEach-Object { 'output line number ' + $_ }` })] })
    const h = await harness([big(400), big(401), big(402), { content: 'first done' }, { content: 'second done' }], { nCtx: 4000, mock: { modes: modesBody() } })
    try {
      await h.run('First task')
      await h.run('Second task')
      const summary = h.mock.requests.filter((r) => r.body.stream === false)
      expect(summary.length).toBeGreaterThan(0)
      for (const r of summary) expect(r).toMatchObject({ mode: 'coding', rental: 'deny' })
      for (const r of h.mock.requests) expect(r).toMatchObject({ mode: 'coding', rental: 'deny' })
      // One read per task, however many requests it made.
      expect(h.mock.headerLog.filter((l) => l.path.endsWith('/modes'))).toHaveLength(2)
    } finally {
      await h.close()
    }
  }, 90_000)

  it('a reverse-engineering task is the re mode', async () => {
    const h = await harness([{ content: 'Found it.' }], { mock: { modes: modesBody() } })
    try {
      await h.run('Look at x', { re: { kind: 'analyse', target: 'C:\\x.exe', goal: 'g', record: false } })
      expect(h.mock.requests[0]).toMatchObject({ mode: 're', rental: 'deny' })
      expect(h.mock.headerLog.every((l) => l.mode === 're')).toBe(true)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('a failure other than 404 reading the modes goes on as before, with a notice', async () => {
    const h = await harness([{ content: 'Done.' }], { mock: { modes: () => ({ status: 500, body: { error: { message: 'boom' } } }) } })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('done')
      expect(h.notices()).toEqual([{ text: expect.stringContaining("couldn't read RulZ LLM Suite's settings"), tone: 'warn' }])
      expect(h.mock.requests[0]).toMatchObject({ mode: 'coding', rental: 'deny' })
      expect(h.mock.requests[0].body.temperature).toBe(0.7)
    } finally {
      await h.close()
    }
  }, 30_000)
})

describe('the sampling and prompt a task gets from LLM Suite', () => {
  const coding = {
    own_load: true,
    sampling: { temperature: 0.6, top_k: 40, max_tokens: 5, model: 'evil', stream: false, messages: [] },
    thinking: { chat_template_kwargs: { enable_thinking: true } },
    system_prompt: 'Use tabs, never spaces.',
    matches: true
  }

  it('merges recommended, LLM Suite and own values; the longest answer stays this app\'s; one notice says so', async () => {
    const h = await harness([{ content: 'Done.' }], {
      mock: { modes: modesBody({ coding }) },
      settings: { modelSampling: { [LOCAL]: { temperature: 0.9, maxOutputTokens: 2048 } } }
    })
    try {
      await h.run('Make it')
      const body = h.mock.requests[0].body
      expect(body).toMatchObject({ model: LOCAL, temperature: 0.9, top_p: 0.8, top_k: 40, repeat_penalty: 1.05, chat_template_kwargs: { enable_thinking: true }, stream: true })
      expect(body.max_tokens).toBeLessThanOrEqual(2048)
      expect(body.max_tokens).not.toBe(5)
      expect(body.messages!.length).toBeGreaterThan(0)
      // The mode's prompt is the very end of the system prompt, which the context budget counts.
      const system = body.messages![0]
      expect(system.role).toBe('system')
      expect(system.content.trimEnd().endsWith('Use tabs, never spaces.')).toBe(true)
      expect(system.content).toContain('Extra instructions from RulZ LLM Suite (Coding settings)')
      expect(h.notices()).toEqual([{ text: "Uses LLM Suite's Coding sampling and prompt. 1 value set here overrides it.", tone: 'info' }])
    } finally {
      await h.close()
    }
  }, 30_000)

  it('thinking turned off here removes LLM Suite\'s thinking fields', async () => {
    const h = await harness([{ content: 'Done.' }], {
      mock: { modes: modesBody({ coding: { ...coding, thinking: { chat_template_kwargs: { enable_thinking: true }, reasoning_effort: 'high' } } }) },
      settings: { modelSampling: { [LOCAL]: { thinking: 'off' } } }
    })
    try {
      await h.run('Make it')
      const body = h.mock.requests[0].body
      expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
      expect(body).not.toHaveProperty('reasoning_effort')
      expect(body.top_k).toBe(40)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('a reverse-engineering task takes LLM Suite\'s RE sampling as it is', async () => {
    const h = await harness([{ content: 'Found it.' }], {
      mock: { modes: modesBody({ re: { own_load: false, sampling: { temperature: 0.1 }, thinking: null, system_prompt: 'Be exact.', matches: true } }) },
      settings: { modelSampling: { [LOCAL]: { temperature: 0.9, topK: 3 } } }
    })
    try {
      await h.run('Look at x', { re: { kind: 'analyse', target: 'C:\\x.exe', goal: 'g', record: false } })
      const body = h.mock.requests[0].body
      expect(body.temperature).toBe(0.1)
      expect(body).not.toHaveProperty('top_p')
      expect(body).not.toHaveProperty('top_k')
      expect(body.messages![0].content.trimEnd().endsWith('Be exact.')).toBe(true)
      expect(h.notices()).toEqual([{ text: "Uses LLM Suite's Reverse engineering sampling and prompt.", tone: 'info' }])
    } finally {
      await h.close()
    }
  }, 30_000)

  it('a mode without settings of its own changes nothing and says nothing; the API server override is named', async () => {
    const plain = await harness([{ content: 'Done.' }], { mock: { modes: modesBody() } })
    const api = await harness([{ content: 'Done.' }], { mock: { modes: modesBody({ api_override: { temperature: 1 } }) } })
    try {
      await plain.run('Hi')
      expect(plain.notices()).toEqual([])
      expect(plain.mock.requests[0].body).toMatchObject({ temperature: 0.7, top_p: 0.8, top_k: 20 })
      expect(plain.mock.requests[0].body.messages![0].content).not.toContain('Extra instructions from RulZ LLM Suite')
      await api.run('Hi')
      expect(api.notices()).toEqual([{ text: 'API server sampling overrides temperature.', tone: 'info' }])
    } finally {
      await plain.close()
      await api.close()
    }
  }, 30_000)

  it('treats an unreadable modes body as a failure to read, not as settings', async () => {
    const h = await harness([{ content: 'Done.' }], { mock: { modes: () => ({ status: 200, body: { modes: { coding: { sampling: 'hot', system_prompt: 7 } } } }) } })
    try {
      await h.run('Hi')
      expect(h.notices()).toEqual([])
      expect(h.mock.requests[0].body.temperature).toBe(0.7)
    } finally {
      await h.close()
    }
  }, 30_000)
})

describe('rentals', () => {
  const both: MockOptions['models'] = [
    { id: 'gemma', state: 'not-loaded' },
    { id: RENTED, state: 'loaded', loaded_context_length: 65536, rental }
  ]

  it('never picks a rental\'s model by itself, and refuses its answers', async () => {
    const h = await harness([{ content: 'Done.' }], { models: both, settings: { model: '' } })
    try {
      await h.run('Hi')
      expect(h.mock.requests[0].body.model).toBe('gemma')
      expect(h.mock.requests[0].rental).toBe('deny')
    } finally {
      await h.close()
    }
  }, 30_000)

  it('uses a rental\'s model the owner chose, without the deny header', async () => {
    const h = await harness([{ content: 'Done.' }], { models: both, settings: { model: RENTED, modelRental: true } })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('done')
      expect(h.mock.requests[0].body.model).toBe(RENTED)
      expect(h.mock.requests[0].rental).toBeUndefined()
      expect(h.mock.requests[0].mode).toBe('coding')
      expect(h.mock.headerLog.find((l) => l.path === '/props')?.rental).toBeUndefined()
    } finally {
      await h.close()
    }
  }, 30_000)

  it('says so and starts nothing when the chosen rental\'s model is gone; a local model that is gone is swapped as before', async () => {
    const gone = await harness([{ content: 'Done.' }], { models: [{ id: 'gemma', state: 'loaded' }], settings: { model: RENTED, modelRental: true } })
    const local = await harness([{ content: 'Done.' }], { models: [{ id: 'gemma', state: 'loaded' }], settings: { model: 'old-local' } })
    try {
      const task = await gone.run('Hi')
      expect(task.outcome).toBe('error')
      expect(gone.notices()).toEqual([
        { text: "Big-Rented-Model is not loaded on a rental now. Load it on LLM Suite's vast.ai page, or choose another model.", tone: 'error' }
      ])
      expect(gone.mock.requests).toHaveLength(0)
      expect(gone.mock.headerLog).toHaveLength(0)
      expect((await local.run('Hi')).outcome).toBe('done')
      expect(local.mock.requests[0].body.model).toBe('gemma')
    } finally {
      await gone.close()
      await local.close()
    }
  }, 30_000)

  it('a model that became a rental behind a stale list is refused by the deny header, in the server\'s words', async () => {
    // The app believes the model is local; the server knows it runs on a rental.
    const h = await harness([{ content: 'Done.' }], {
      models: [{ id: LOCAL, state: 'loaded', rental }],
      list: [{ id: LOCAL, loaded: true }]
    })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('error')
      expect(h.notices().at(-1)).toMatchObject({ tone: 'error', text: expect.stringContaining('runs on a rented computer') })
      expect(h.notices().at(-1)!.text).not.toContain('The model server reported a problem')
      expect(h.session.modelRefusal).toContain('runs on a rented computer')
      expect(h.mock.requests).toHaveLength(0)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('also reads a rental from the vast/ path of LLM Suite 0.29 to 0.34', async () => {
    const h = await harness([{ content: 'Done.' }], { models: [{ id: 'old-rental', state: 'loaded', path: 'vast/123/m.gguf' }, { id: 'gemma' }], settings: { model: '' } })
    try {
      await h.run('Hi')
      expect(h.mock.requests[0].body.model).toBe('gemma')
    } finally {
      await h.close()
    }
  }, 30_000)
})

describe('batch runs and rentals', () => {
  const both: MockOptions['models'] = [
    { id: LOCAL, state: 'loaded', loaded_context_length: 65536 },
    { id: RENTED, state: 'loaded', loaded_context_length: 65536, rental }
  ]
  const rentedSettings = { model: RENTED, modelRental: true }

  it('is refused for a rental\'s model without the yes, with its own sentence for an older phone app', async () => {
    const h = await harness([], { models: both, settings: rentedSettings })
    try {
      await expect(h.runner.start(h.root, '', h.options(), true, false)).rejects.toThrow(RENTAL_NEEDS_YES)
      await expect(h.runner.start(h.root, '', h.options(), true)).rejects.toThrow(RENTAL_OLD_APP)
      expect(h.runner.active).toBe(false)
      expect(h.batchStore.list()).toEqual([])
      expect(h.mock.requests).toHaveLength(0)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('runs with the yes: tasks go to the rental without the deny header, and the run remembers it', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }], { models: both, settings: rentedSettings, mock: { practicePerModel: true } })
    try {
      await h.runner.start(h.root, 'r', h.options({ maxTasks: 1 }), true, true)
      const run = await h.finished()
      expect(run).toMatchObject({ status: 'done', modelName: RENTED, rentalOk: true })
      expect(run.items[0].status).toBe('solved')
      expect(h.mock.requests[0].body.model).toBe(RENTED)
      expect(h.mock.requests[0].mode).toBe('re')
      expect(h.mock.requests[0].rental).toBeUndefined()
    } finally {
      await h.close()
    }
  }, 60_000)

  it('a local model in a batch still refuses rentals', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }], { models: both })
    try {
      await h.runner.start(h.root, '', h.options({ maxTasks: 1 }), true, false)
      const run = await h.finished()
      expect(run.rentalOk).toBeUndefined()
      expect(h.mock.requests[0]).toMatchObject({ mode: 're', rental: 'deny' })
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stops between two tasks when the model in use becomes a rental\'s, instead of failing the rest', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], {
      models: both,
      onTaskEnd: ({ settings }) => {
        settings.model = RENTED
        settings.modelRental = true
      }
    })
    try {
      await h.runner.start(h.root, '', h.options(), true, false)
      const run = await h.finished()
      expect(run.status).toBe('stopped')
      expect(run.items.map((i) => i.status)).toEqual(['solved', 'pending'])
      expect(run.stopNote).toMatch(/now on a rented computer/)
      expect(h.mock.requests).toHaveLength(1)
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stops when the chosen rental\'s model is gone between two tasks', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], {
      models: both,
      settings: rentedSettings,
      onTaskEnd: ({ setList }) => setList([{ id: LOCAL, loaded: true }])
    })
    try {
      await h.runner.start(h.root, '', h.options(), true, true)
      const run = await h.finished()
      expect(run.items.map((i) => i.status)).toEqual(['solved', 'pending'])
      expect(run.status).toBe('stopped')
      expect(run.stopNote).toContain('is not loaded on a rental now')
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stops when a task is refused by the server (a switch the app did not see), keeping the rest pending', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }], {
      models: [{ id: LOCAL, state: 'loaded', rental }],
      list: [{ id: LOCAL, loaded: true }]
    })
    try {
      await h.runner.start(h.root, '', h.options(), true, false)
      const run = await h.finished()
      expect(run.status).toBe('stopped')
      expect(run.items.map((i) => i.status)).toEqual(['error', 'pending'])
      expect(run.items[0].error).toContain('runs on a rented computer')
      expect(run.stopNote).toContain('runs on a rented computer')
    } finally {
      await h.close()
    }
  }, 60_000)

  it('never uses a rental\'s model by itself, and counts practice for the model it uses', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }], {
      models: [{ id: RENTED, state: 'loaded', rental }, { id: 'gemma', state: 'loaded' }],
      settings: { model: '' },
      mock: { practicePerModel: true }
    })
    try {
      await h.runner.start(h.root, '', h.options({ onlyUnsolved: true }), true, false)
      const run = await h.finished()
      expect(run.modelName).toBe('gemma')
      expect(h.mock.overviewQueries).toEqual(['gemma'])
      // xor-1 counts as solved for that model, so only xor-2 is planned.
      expect(run.items.map((i) => i.taskId)).toEqual(['xor-2'])
      expect(h.mock.requests[0].body.model).toBe('gemma')
    } finally {
      await h.close()
    }
  }, 60_000)

  it('with an older LLM Suite the overview is for all models together, as before', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], { models: [{ id: LOCAL, state: 'loaded' }] })
    try {
      await h.runner.start(h.root, '', h.options({ onlyUnsolved: true }), true)
      const run = await h.finished()
      expect(run.items.map((i) => i.taskId)).toEqual(['xor-1', 'xor-2'])
      expect(h.mock.overviewQueries).toEqual([LOCAL])
    } finally {
      await h.close()
    }
  }, 60_000)
})

describe('OpenRouter models in tasks', () => {
  const CLOUD = 'openrouter/vendor/model-x'
  const cloudModel = { id: CLOUD, state: 'loaded' as const, loaded_context_length: 32768, cloud: { provider: 'openrouter', free: false } }
  const both: MockOptions['models'] = [{ id: 'gemma', state: 'loaded', loaded_context_length: 65536 }, cloudModel]
  const chosen = { model: CLOUD, modelCloud: true }
  const SENTENCE = 'OpenRouter says there are no credits left. Add some at openrouter.ai, then try again.'

  it('is never picked by itself: a task without a choice uses the local model and still refuses rentals', async () => {
    const h = await harness([{ content: 'Done.' }], { models: both, settings: { model: '' } })
    const only = await harness([{ content: 'Done.' }], { models: [cloudModel], settings: { model: '' } })
    try {
      await h.run('Hi')
      expect(h.mock.requests[0].body.model).toBe('gemma')
      expect(h.mock.requests[0].rental).toBe('deny')
      const task = await only.run('Hi')
      expect(task.outcome).toBe('error')
      expect(only.mock.requests).toHaveLength(0)
      expect(only.mock.headerLog).toHaveLength(0)
    } finally {
      await h.close()
      await only.close()
    }
  }, 30_000)

  it('a chosen cloud model gets every request of the task by its exact id, with no rental header and no further yes', async () => {
    const big = (n: number) => ({ toolCalls: [call('run_command', { command: `1..${n} | ForEach-Object { 'output line number ' + $_ }` })] })
    const h = await harness([big(400), big(401), big(402), { content: 'first done' }, { content: 'second done' }], {
      nCtx: 4000,
      models: [{ ...cloudModel, loaded_context_length: 4000 }, { id: 'gemma', state: 'loaded' }],
      settings: chosen,
      mock: { modes: modesBody() }
    })
    try {
      await h.run('First task')
      const task = await h.run('Second task')
      expect(task.outcome).toBe('done')
      // Summary (compaction) requests are among them, and they name the chosen model too.
      expect(h.mock.requests.some((r) => r.body.stream === false)).toBe(true)
      for (const r of h.mock.requests) {
        expect(r.body.model).toBe(CLOUD)
        expect(r.rental).toBeUndefined()
        expect(r.mode).toBe('coding')
      }
      expect(h.mock.headerLog.length).toBeGreaterThan(h.mock.requests.length)
      for (const l of h.mock.headerLog) {
        expect(l.model).toBe(CLOUD)
        expect(l.rental).toBeUndefined()
      }
      expect(h.notices().filter((n) => n.tone === 'error')).toEqual([])
    } finally {
      await h.close()
    }
  }, 90_000)

  it.each([...CLOUD_FATAL_CODES])('%s ends the task at once with the server\'s sentence and no second request', async (code) => {
    const h = await harness([{ status: 402, error: SENTENCE, code }, { content: 'Should never be asked.' }], { models: both, settings: chosen })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('error')
      expect(h.notices().at(-1)).toEqual({ text: SENTENCE, tone: 'error' })
      expect(h.mock.requests).toHaveLength(1)
      expect(h.session.modelRefusal).toBe(SENTENCE)
      expect(h.session.modelRefusalIsCloud).toBe(true)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('an error in the middle of a reply ends the task too, without a second request', async () => {
    const h = await harness([{ streamError: { message: SENTENCE, code: 'cloud_limit' } }, { content: 'Should never be asked.' }], { models: both, settings: chosen })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('error')
      expect(h.notices().at(-1)).toEqual({ text: SENTENCE, tone: 'error' })
      expect(h.mock.requests).toHaveLength(1)
      expect(h.session.modelRefusalIsCloud).toBe(true)
    } finally {
      await h.close()
    }
  }, 30_000)

  it.each(['cloud_policy', 'cloud_request', 'cloud_error'])('%s ends the task as an ordinary error, and is not tried again', async (code) => {
    // The sentence of a too-long prompt must not set off the "try again with less history" step either: that would cost money.
    const message = `The request exceeds the available context size (${code}).`
    const h = await harness([{ status: 400, error: message, code }, { content: 'Should never be asked.' }], { models: both, settings: chosen })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('error')
      expect(h.notices().at(-1)).toEqual({ text: message, tone: 'error' })
      expect(h.mock.requests).toHaveLength(1)
      expect(h.session.modelRefusal).toBeNull()
      expect(h.session.modelRefusalIsCloud).toBe(false)
    } finally {
      await h.close()
    }
  }, 30_000)

  it.each([
    ['cloud_credits', 402],
    ['cloud_error', 500]
  ])('a %s from the summary request ends the task: no mechanical summary, no second summary request', async (code, status) => {
    const big = (n: number) => ({ toolCalls: [call('run_command', { command: `1..${n} | ForEach-Object { 'output line number ' + $_ }` })] })
    const h = await harness([big(400), big(401), big(402), { content: 'first done' }, { content: 'Should never be asked.' }], {
      nCtx: 4000,
      models: [{ ...cloudModel, loaded_context_length: 4000 }],
      settings: chosen,
      mock: { complete: () => ({ status, error: SENTENCE, code }) }
    })
    try {
      await h.run('First task')
      const task = await h.run('Second task')
      expect(task.outcome).toBe('error')
      expect(h.notices().at(-1)).toEqual({ text: SENTENCE, tone: 'error' })
      expect(h.mock.requests.filter((r) => r.body.stream === false)).toHaveLength(1)
    } finally {
      await h.close()
    }
  }, 90_000)

  it('sums the cost of the turns and of the summary on the task, and records none for a local model', async () => {
    const h = await harness([{ toolCalls: [call('list_files', {})], cost: 0.01 }, { content: 'Done.', cost: 0.0042 }], { models: both, settings: chosen })
    const local = await harness([{ content: 'Done.' }])
    try {
      const task = await h.run('Hi')
      expect(task.cost).toBeCloseTo(0.0142, 8)
      expect(h.session.data.tasks[0].cost).toBeCloseTo(0.0142, 8)
      expect((await local.run('Hi')).cost).toBeUndefined()
    } finally {
      await h.close()
      await local.close()
    }
  }, 30_000)

  it('also counts what a summary request cost', async () => {
    const big = (n: number) => ({ toolCalls: [call('run_command', { command: `1..${n} | ForEach-Object { 'output line number ' + $_ }` })] })
    const h = await harness([big(400), big(401), big(402), { content: 'first done' }, { content: 'second done' }], {
      nCtx: 4000,
      models: [{ ...cloudModel, loaded_context_length: 4000 }],
      settings: chosen,
      mock: { complete: () => ({ cost: 0.5 }) }
    })
    try {
      await h.run('First task')
      const task = await h.run('Second task')
      expect(task.outcome).toBe('done')
      expect(task.cost).toBeGreaterThanOrEqual(0.5)
    } finally {
      await h.close()
    }
  }, 90_000)

  it('a chosen cloud model that is gone is never swapped for another one', async () => {
    const h = await harness([{ content: 'Done.' }], { models: [{ id: 'gemma', state: 'loaded' }], settings: chosen })
    try {
      const task = await h.run('Hi')
      expect(task.outcome).toBe('error')
      expect(h.notices().at(-1)).toMatchObject({ tone: 'error', text: expect.stringContaining('not available on OpenRouter now') })
      expect(h.mock.requests).toHaveLength(0)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('keeps the context-overflow retry for everything that is not an OpenRouter error, rentals included', () => {
    const text = 'the request exceeds the available context size, try increasing it'
    expect(isContextOverflow(new ServerError(text, 400))).toBe(true)
    expect(isContextOverflow(new ServerError(text, 409))).toBe(true)
    expect(isContextOverflow(new ServerError(text, 400, 'invalid_request'))).toBe(true)
    for (const code of ['cloud_limit', 'cloud_request', 'cloud_policy', 'cloud_error']) expect(isContextOverflow(new ServerError(text, 400, code))).toBe(false)
  })
})

describe('batch runs and OpenRouter models', () => {
  const CLOUD = 'openrouter/vendor/model-x'
  const cloudModel = { id: CLOUD, state: 'loaded' as const, loaded_context_length: 32768, cloud: { provider: 'openrouter', free: true } }
  const both: MockOptions['models'] = [{ id: LOCAL, state: 'loaded', loaded_context_length: 65536 }, cloudModel]
  const chosen = { model: CLOUD, modelCloud: true }
  const SENTENCE = 'Your daily OpenRouter spending limit is reached. Raise it on LLM Suite\'s OpenRouter page.'

  it('is refused without the yes, in words that name OpenRouter, also for a phone app that sends nothing', async () => {
    const h = await harness([], { models: both, settings: chosen })
    try {
      await expect(h.runner.start(h.root, '', h.options(), true, false)).rejects.toThrow(CLOUD_NEEDS_YES)
      await expect(h.runner.start(h.root, '', h.options(), true)).rejects.toThrow(CLOUD_OLD_APP)
      expect(CLOUD_NEEDS_YES).toContain('OpenRouter')
      expect(CLOUD_OLD_APP).toContain('OpenRouter')
      expect(CLOUD_NEEDS_YES).not.toContain('rented')
      expect(h.runner.active).toBe(false)
      expect(h.batchStore.list()).toEqual([])
      expect(h.mock.requests).toHaveLength(0)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('runs with the yes: every task names the model, there is no rental header, and the run is marked as OpenRouter', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }], { models: both, settings: chosen, mock: { practicePerModel: true } })
    try {
      await h.runner.start(h.root, 'c', h.options({ maxTasks: 1 }), true, true)
      const run = await h.finished()
      expect(run).toMatchObject({ status: 'done', modelName: CLOUD, modelCloud: true, rentalOk: true })
      expect(run.items[0].status).toBe('solved')
      expect(h.mock.requests[0].body.model).toBe(CLOUD)
      expect(h.mock.requests[0].rental).toBeUndefined()
      expect(h.mock.requests[0].mode).toBe('re')
      expect(h.mock.overviewQueries).toEqual([CLOUD])
    } finally {
      await h.close()
    }
  }, 60_000)

  it('a fatal OpenRouter error stops the run with the server\'s sentence and leaves that task pending, without another request', async () => {
    const h = await harness([{ status: 429, error: SENTENCE, code: 'cloud_limit' }, { content: 'ANSWER: 42' }], { models: both, settings: chosen })
    try {
      await h.runner.start(h.root, '', h.options(), true, true)
      const run = await h.finished()
      expect(run.status).toBe('stopped')
      expect(run.items.map((i) => i.status)).toEqual(['pending', 'pending'])
      expect(run.items[0]).toMatchObject({ startedAt: undefined, taskRecordId: undefined })
      expect(run.stopNote).toBe(SENTENCE)
      expect(h.mock.requests).toHaveLength(1)
    } finally {
      await h.close()
    }
  }, 60_000)

  it('a non-fatal OpenRouter error fails that task only: the run goes on and nothing is retried', async () => {
    const h = await harness([{ status: 400, error: 'OpenRouter refused this request.', code: 'cloud_request' }, { content: 'ANSWER: 42' }], { models: both, settings: chosen })
    try {
      await h.runner.start(h.root, '', h.options(), true, true)
      const run = await h.finished()
      expect(run.items.map((i) => i.status)).toEqual(['error', 'solved'])
      expect(run.items[0].error).toBe('OpenRouter refused this request.')
      expect(run.stopNote).toBeUndefined()
      expect(h.mock.requests).toHaveLength(2)
    } finally {
      await h.close()
    }
  }, 60_000)

  it('stops between two tasks when the model changes from the cloud one, and when a local run moves to a cloud one', async () => {
    const away = await harness([{ content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], {
      models: both,
      settings: chosen,
      onTaskEnd: ({ settings }) => {
        settings.model = LOCAL
        settings.modelCloud = false
      }
    })
    const toCloud = await harness([{ content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], {
      models: both,
      onTaskEnd: ({ settings }) => {
        settings.model = CLOUD
        settings.modelCloud = true
      }
    })
    try {
      await away.runner.start(away.root, '', away.options(), true, true)
      const run = await away.finished()
      expect(run.items.map((i) => i.status)).toEqual(['solved', 'pending'])
      expect(run.status).toBe('stopped')
      expect(run.stopNote).toMatch(/model changed/)
      expect(away.mock.requests).toHaveLength(1)

      await toCloud.runner.start(toCloud.root, '', toCloud.options(), true, false)
      const run2 = await toCloud.finished()
      expect(run2.items.map((i) => i.status)).toEqual(['solved', 'pending'])
      expect(run2.stopNote).toMatch(/model changed/)
      expect(toCloud.mock.requests).toHaveLength(1)
    } finally {
      await away.close()
      await toCloud.close()
    }
  }, 60_000)

  it('a run on a local model still goes on when the owner picks another local model (as before)', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }, { content: 'ANSWER: 42' }], {
      models: [...both, { id: 'gemma', state: 'loaded' }],
      onTaskEnd: ({ settings }) => {
        settings.model = 'gemma'
      }
    })
    try {
      await h.runner.start(h.root, '', h.options(), true, false)
      const run = await h.finished()
      expect(run.items.map((i) => i.status)).toEqual(['solved', 'solved'])
      expect(h.mock.requests.map((r) => r.body.model)).toEqual([LOCAL, 'gemma'])
    } finally {
      await h.close()
    }
  }, 60_000)

  it('a rental refusal still leaves its task as an error, not pending (as before)', async () => {
    const h = await harness([{ content: 'ANSWER: 42' }], { models: [{ id: LOCAL, state: 'loaded', rental }], list: [{ id: LOCAL, loaded: true }] })
    try {
      await h.runner.start(h.root, '', h.options(), true, false)
      const run = await h.finished()
      expect(run.items.map((i) => i.status)).toEqual(['error', 'pending'])
    } finally {
      await h.close()
    }
  }, 60_000)
})
