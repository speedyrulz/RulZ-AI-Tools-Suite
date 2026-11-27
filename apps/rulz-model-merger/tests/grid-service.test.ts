import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { GridJob, GridRecord, GridRequest, GridTemplateInfo } from '@shared/grid'
import { DEFAULT_RECIPE, type MergeRecipe } from '@shared/merge'
import type { ComfyEventHandlers, HistoryEntry, HistoryImage } from '../src/main/comfy/client'
import { GridService, type GridClient } from '../src/main/grid/service'
import type { MergeConnection } from '../src/main/merge/runner'

const validate = vi.fn<(...a: unknown[]) => string[]>(() => [])
vi.mock('../src/main/merge/graph', () => ({
  validateRecipe: (...a: unknown[]) => validate(...a),
  completeBlocks: (r: MergeRecipe, inputs: string[]) => ({ ...r, blocks: { ...Object.fromEntries(inputs.map((i) => [i, r.ratio])), ...r.blocks } })
}))
vi.mock('../src/main/merge/inspect', () => ({
  blockNodeOptions: () => [{ node: 'ModelMergeBlocks', inputs: ['input', 'middle', 'out'] }]
}))

const recipe: MergeRecipe = { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors' }
const TEMPLATE = { '1': { class_type: 'UNETLoader', inputs: {} }, '9': { class_type: 'PreviewImage', inputs: {} } }

const info = (path: string): GridTemplateInfo => ({
  path,
  name: 'tpl.json',
  error: null,
  warnings: [],
  modelLoader: { id: '1', classType: 'UNETLoader' },
  promptInput: { id: '2', input: 'text', classType: 'CLIPTextEncode' },
  seedInputs: [],
  imageOutputs: ['9'],
  defaultPrompt: null
})

type Script = (n: number, promptId: string, h: ComfyEventHandlers, fake: Fake['client']) => void

interface Fake {
  client: GridClient & {
    handlers: ComfyEventHandlers | null
    closed: number
    queued: Array<{ prompt: Record<string, unknown>; id: string }>
    deleted: string[][]
    interrupted: Array<string | undefined>
    histories: Map<string, HistoryEntry | null>
    script: Script
    queueThrows: Error | null
    historyThrows: Error | null
  }
  conn: MergeConnection
}

const png = (tag: string): Buffer => Buffer.from(`PNG-${tag}`)
const okEntry = (id: string, nodeId = '9', filename = 'out.png'): HistoryEntry => ({
  promptId: id,
  completed: true,
  statusStr: 'success',
  images: [{ nodeId, image: { filename, subfolder: '', type: 'temp' } }],
  error: null
})
const succeed: Script = (_n, id, h, f) => {
  f.histories.set(id, okEntry(id))
  h.onSuccess?.(id)
}

function makeFake(): Fake {
  const conn: MergeConnection = { url: 'http://127.0.0.1:8188', reachable: true, own: true, sharedQueueAccepted: true, version: '0.46' }
  const client: Fake['client'] = {
    handlers: null,
    closed: 0,
    queued: [],
    deleted: [],
    interrupted: [],
    histories: new Map(),
    script: succeed,
    queueThrows: null,
    historyThrows: null,
    async objectInfo() {
      return {}
    },
    async queuePrompt(prompt) {
      if (client.queueThrows) throw client.queueThrows
      const id = `p${client.queued.length + 1}`
      client.queued.push({ prompt, id })
      const n = client.queued.length
      setTimeout(() => client.handlers && client.script(n, id, client.handlers, client), 1)
      return { promptId: id, number: n }
    },
    async history(id) {
      if (client.historyThrows) throw client.historyThrows
      return client.histories.get(id) ?? null
    },
    async queue() {
      return { running: [], pending: [] }
    },
    async deletePending(ids) {
      client.deleted.push(ids)
    },
    async interrupt(id) {
      client.interrupted.push(id)
    },
    async viewImage(img: HistoryImage) {
      return png(img.filename)
    },
    openEvents(_cid, handlers) {
      client.handlers = handlers
      return {
        close() {
          client.closed++
        }
      }
    }
  }
  return { client, conn }
}

let root: string
let gridsDir: string
let tplPath: string
let fake: Fake
let seq: number
let trashed: string[]
let opened: string[]
let savePath: string | null
let picked: string | null
let tplCheck: string | null
const built: Array<{ recipe: MergeRecipe; prompt: string; seed: number }> = []

function service(extra: Partial<ConstructorParameters<typeof GridService>[0]> = {}): GridService {
  return new GridService({
    client: () => fake.client,
    getConnection: async () => fake.conn,
    appVersion: '9.9.9',
    gridsDir,
    pickFile: async () => picked,
    pickSavePath: async () => savePath,
    trashItem: async (p) => {
      trashed.push(p)
    },
    openPath: async (p) => {
      opened.push(p)
    },
    now: () => 1_000_000 + seq * 1000,
    newId: () => `g${++seq}`,
    template: {
      inspectTemplate: (_json, path) => info(path),
      checkTemplateForRecipe: () => tplCheck,
      buildGridPrompt: (_t, _i, r, prompt, seed) => {
        built.push({ recipe: r, prompt, seed })
        if (prompt === 'BAD') throw new Error('The template does not fit.')
        return { '1': { class_type: 'UNETLoader', inputs: {} }, '9': { class_type: 'PreviewImage', inputs: {} } }
      }
    },
    pollMs: 5,
    historyRetryMs: 1,
    progressIntervalMs: 200,
    ...extra
  })
}

const req = (over: Partial<GridRequest> = {}): GridRequest => ({
  recipe,
  axis: { kind: 'ratio', loraIndex: null, values: [0.2, 0.8] },
  prompts: ['one', 'two'],
  seed: 5,
  templatePath: tplPath,
  ...over
})

async function waitDone(svc: GridService, id: string): Promise<GridJob> {
  for (let i = 0; i < 400; i++) {
    const j = (await svc.list()).find((x) => x.id === id)
    if (j && j.status !== 'running') return j
    await new Promise((r) => setTimeout(r, 5))
  }
  throw new Error('grid did not finish')
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'grid-svc-'))
  gridsDir = join(root, 'grids')
  tplPath = join(root, 'tpl.json')
  await writeFile(tplPath, JSON.stringify(TEMPLATE))
  fake = makeFake()
  seq = 0
  trashed = []
  opened = []
  savePath = null
  picked = null
  tplCheck = null
  built.length = 0
  validate.mockReset()
  validate.mockImplementation(() => [])
})
afterEach(async () => {
  await rm(root, { recursive: true, force: true })
})

describe('loadTemplate', () => {
  it('returns null when the dialog is cancelled', async () => {
    expect(await service().loadTemplate()).toBeNull()
  })
  it('uses the dialog path when none is given', async () => {
    picked = tplPath
    expect((await service().loadTemplate())?.path).toBe(tplPath)
  })
  it('reports a parse error as an info with an error', async () => {
    const bad = join(root, 'bad.json')
    await writeFile(bad, '{ nope')
    const r = await service().loadTemplate(bad)
    expect(r?.error).toMatch(/not valid JSON/)
    expect(r?.name).toBe('bad.json')
  })
  it('refuses a file over 5 MB and a missing file', async () => {
    const big = join(root, 'big.json')
    await writeFile(big, Buffer.alloc(5 * 1024 * 1024 + 1, 0x20))
    expect((await service().loadTemplate(big))?.error).toMatch(/5 MB/)
    expect((await service().loadTemplate(join(root, 'none.json')))?.error).toMatch(/could not be read/)
  })
})

describe('start validation', () => {
  const bad = async (over: Partial<GridRequest>, re: RegExp): Promise<void> => {
    await expect(service().start(req(over))).rejects.toThrow(re)
    expect(fake.client.queued).toHaveLength(0)
  }
  it('rejects the wrong number of values and non-finite values', async () => {
    await bad({ axis: { kind: 'ratio', loraIndex: null, values: [0.5] } }, /2 to 8 values/)
    await bad({ axis: { kind: 'ratio', loraIndex: null, values: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] } }, /2 to 8 values/)
    await bad({ axis: { kind: 'ratio', loraIndex: null, values: [0.1, Number.NaN] } }, /must be a number/)
  })
  it('checks ranges per axis', async () => {
    await bad({ axis: { kind: 'ratio', loraIndex: null, values: [0.1, 1.2] } }, /ratio must be between 0 and 1/)
    await bad({ recipe: { ...recipe, mode: 'addDifference', c: 'c.safetensors' }, axis: { kind: 'multiplier', loraIndex: null, values: [0, 2.5] } }, /multiplier must be between 0 and 2/)
    await bad({ recipe: { ...recipe, loras: [{ name: 'l.safetensors', strength: 1, clipStrength: 1 }] }, axis: { kind: 'loraStrength', loraIndex: 0, values: [-11, 1] } }, /strength must be between -10 and 10/)
  })
  it('ties each axis to its mode', async () => {
    await bad({ recipe: { ...recipe, mode: 'addDifference' } }, /ratio can only vary in weighted or blocks/)
    await bad({ axis: { kind: 'multiplier', loraIndex: null, values: [0, 1] } }, /multiplier can only vary in add difference/)
    await bad({ axis: { kind: 'loraStrength', loraIndex: 0, values: [0, 1] } }, /Choose which LoRA/)
    await bad({ axis: { kind: 'loraStrength', loraIndex: null, values: [0, 1] } }, /Choose which LoRA/)
  })
  it('checks prompts and the seed', async () => {
    await bad({ prompts: [] }, /1 to 4 prompts/)
    await bad({ prompts: ['a', 'b', 'c', 'd', 'e'] }, /1 to 4 prompts/)
    await bad({ prompts: ['a', '  '] }, /needs text/)
    await bad({ prompts: ['x'.repeat(2001)] }, /at most 2000/)
    await bad({ seed: -1 }, /seed/)
    await bad({ seed: 1.5 }, /seed/)
    await bad({ seed: 2 ** 53 }, /seed/)
  })
  it('refuses a template that does not load or does not fit the recipe', async () => {
    await bad({ templatePath: join(root, 'missing.json') }, /could not be read/)
    tplCheck = 'The template has no sampler.'
    await bad({}, /no sampler/)
  })
  it('refuses a template with an error', async () => {
    await expect(
      service({ template: { inspectTemplate: (_j, p) => ({ ...info(p), error: 'No loader.' }), checkTemplateForRecipe: () => null, buildGridPrompt: () => ({}) } }).start(req())
    ).rejects.toThrow(/No loader/)
  })
  it('reports recipe problems and ignores the output name', async () => {
    validate.mockImplementation(() => ['Pick model B.'])
    await bad({}, /Pick model B/)
    validate.mockImplementation(() => [])
    const svc = service()
    await svc.start(req({ recipe: { ...recipe, outputName: '' } }))
    expect((validate.mock.calls.at(-1)?.[0] as MergeRecipe).outputName).toBe('grid')
  })
  it('ignores the missing ModelSave / CheckpointSave problem, since grids save no model', async () => {
    validate.mockImplementation(() => ['The server has no ModelSave node.', 'The server has no CheckpointSave node.'])
    const svc = service()
    const job = await svc.start(req({ prompts: ['one'] }))
    await waitDone(svc, job.id)
    fake.client.queued.length = 0
    validate.mockImplementation(() => ['The server has no ModelSave node.', 'The server has no UNETLoader node.'])
    await bad({}, /no UNETLoader node/)
  })
  it('writes template.json from the text that was checked', async () => {
    const text = '{\n  "1": {"class_type": "UNETLoader", "inputs": {}},\n  "9": {"class_type": "PreviewImage", "inputs": {}}\n}\n'
    await writeFile(tplPath, text)
    const svc = service()
    const job = await svc.start(req({ prompts: ['one'] }))
    expect(await readFile(join(gridsDir, job.id, 'template.json'), 'utf8')).toBe(text)
    await waitDone(svc, job.id)
  })
  it('refuses a ratio axis when every block has its own value, but allows it when some are default', async () => {
    const blocks = (b: Record<string, number>): MergeRecipe => ({ ...recipe, mode: 'blocks', blockNode: 'ModelMergeBlocks', blocks: b })
    await bad({ recipe: blocks({ input: 0.1, middle: 0.2, out: 0.3 }) }, /Every block has its own value.*Clear some block values or sweep a LoRA strength\./)
    const svc = service()
    const job = await svc.start(req({ recipe: blocks({ input: 0.1, middle: 0.2 }) }))
    await waitDone(svc, job.id)
  })
  it('refuses an unreachable server and a shared queue that was not accepted', async () => {
    fake.conn.reachable = false
    await bad({}, /No ComfyUI server is reachable/)
    fake.conn = { ...fake.conn, reachable: true, own: false, sharedQueueAccepted: false }
    await bad({}, /Accept its shared queue/)
  })
  it('allows a server the app did not start once its queue was accepted', async () => {
    fake.conn = { ...fake.conn, own: false, sharedQueueAccepted: true }
    const svc = service()
    const job = await svc.start(req())
    expect(job.own).toBe(false)
    await waitDone(svc, job.id)
  })
  it('allows one grid at a time', async () => {
    fake.client.script = () => undefined
    const svc = service()
    const job = await svc.start(req())
    await expect(svc.start(req())).rejects.toThrow(/already running/)
    await svc.cancel(job.id)
    const again = await svc.start(req())
    expect(again.id).not.toBe(job.id)
    await svc.cancel(again.id)
  })
})

describe('running a grid', () => {
  it('runs column by column and keeps the cells row-major', async () => {
    const svc = service()
    const job = await svc.start(req({ axis: { kind: 'ratio', loraIndex: null, values: [0.1234567, 0.9] }, prompts: ['one', 'two', 'three'] }))
    expect(job.cells.map((c) => [c.col, c.row])).toEqual([
      [0, 0], [1, 0],
      [0, 1], [1, 1],
      [0, 2], [1, 2]
    ])
    expect(job.cells.every((c) => c.status === 'pending')).toBe(true)
    expect(job.request.axis.values).toEqual([0.1235, 0.9])
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('done')
    expect(built.map((b) => [b.recipe.ratio, b.prompt, b.seed])).toEqual([
      [0.1235, 'one', 5], [0.1235, 'two', 5], [0.1235, 'three', 5],
      [0.9, 'one', 5], [0.9, 'two', 5], [0.9, 'three', 5]
    ])
    expect(done.cells.map((c) => [c.col, c.row])).toEqual(job.cells.map((c) => [c.col, c.row]))
    expect(done.cells.every((c) => c.status === 'done' && c.file === `c${c.col}-r${c.row}.png`)).toBe(true)
    expect(fake.client.closed).toBeGreaterThan(0)
  })

  it('writes the images, template.json and grid.json (atomically, with no temp file left)', async () => {
    const svc = service()
    const job = await svc.start(req())
    const done = await waitDone(svc, job.id)
    const dir = join(gridsDir, job.id)
    expect((await readdir(dir)).sort()).toEqual(['c0-r0.png', 'c0-r1.png', 'c1-r0.png', 'c1-r1.png', 'grid.json', 'template.json'])
    expect((await readFile(join(dir, 'c0-r0.png'))).toString()).toBe('PNG-out.png')
    expect(JSON.parse(await readFile(join(dir, 'template.json'), 'utf8'))).toEqual(TEMPLATE)
    const rec = JSON.parse(await readFile(join(dir, 'grid.json'), 'utf8')) as GridRecord
    expect(rec).toMatchObject({ app: 'RulZ Model Merger', appVersion: '9.9.9', serverVersion: '0.46' })
    expect(rec.job.status).toBe('done')
    expect(rec.job.cells.every((c) => c.status === 'done')).toBe(true)
    expect(done.finishedAt).not.toBeNull()
  })

  it('writes grid.json after every cell', async () => {
    const svc = service()
    const seen: number[] = []
    svc.on('job', () => undefined)
    fake.client.script = (n, id, h, f) => {
      void (async () => {
        const rec = JSON.parse(await readFile(join(gridsDir, 'g1', 'grid.json'), 'utf8')) as GridRecord
        seen.push(rec.job.cells.filter((c) => c.status === 'done').length)
        succeed(n, id, h, f)
      })()
    }
    const job = await svc.start(req())
    await waitDone(svc, job.id)
    expect(seen).toEqual([0, 1, 2, 3])
  })

  it('completes block inputs and applies the lora strength axis', async () => {
    const svc = service()
    const job = await svc.start(
      req({
        recipe: { ...recipe, mode: 'blocks', blockNode: 'ModelMergeBlocks', blocks: { out: 0.3 } },
        axis: { kind: 'ratio', loraIndex: null, values: [0.25, 0.75] },
        prompts: ['one']
      })
    )
    await waitDone(svc, job.id)
    expect(built[0].recipe.blocks).toEqual({ input: 0.25, middle: 0.25, out: 0.3 })
    expect(built[1].recipe.blocks).toEqual({ input: 0.75, middle: 0.75, out: 0.3 })

    const lora = { name: 'l.safetensors', strength: 1, clipStrength: 1 }
    built.length = 0
    const j2 = await svc.start(req({ recipe: { ...recipe, loras: [lora, { ...lora, name: 'm.safetensors' }] }, axis: { kind: 'loraStrength', loraIndex: 1, values: [-1, 2] }, prompts: ['one'] }))
    await waitDone(svc, j2.id)
    expect(built.map((b) => b.recipe.loras.map((l) => l.strength))).toEqual([[1, -1], [1, 2]])
  })

  it('keeps the real extension ComfyUI returns', async () => {
    fake.client.script = (_n, id, h, f) => {
      f.histories.set(id, okEntry(id, '9', 'pic.webp'))
      h.onSuccess?.(id)
    }
    const svc = service()
    const job = await svc.start(req({ prompts: ['one'] }))
    const done = await waitDone(svc, job.id)
    expect(done.cells[0].file).toBe('c0-r0.webp')
    expect((await readdir(join(gridsDir, job.id))).includes('c0-r0.webp')).toBe(true)
  })

  it('marks a cell without an image as an error and goes on', async () => {
    fake.client.script = (n, id, h, f) => {
      f.histories.set(id, n === 2 ? okEntry(id, '5') : okEntry(id))
      h.onSuccess?.(id)
    }
    const svc = service()
    const job = await svc.start(req())
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('done')
    const bad = done.cells.filter((c) => c.status === 'error')
    expect(bad).toHaveLength(1)
    expect(bad[0].error).toMatch(/without returning an image/)
    expect(done.cells.filter((c) => c.status === 'done')).toHaveLength(3)
  })

  it('waits for the history entry that ComfyUI writes just after execution_success', async () => {
    let calls = 0
    fake.client.script = (_n, id, h, f) => {
      f.histories.set(id, okEntry(id))
      h.onSuccess?.(id)
    }
    const real = fake.client.history
    fake.client.history = async (id) => (++calls <= 2 ? null : real(id))
    const svc = service({ pollMs: 1000 })
    const job = await svc.start(req({ prompts: ['one'], axis: { kind: 'ratio', loraIndex: null, values: [0.2, 0.8] } }))
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('done')
    expect(done.cells.every((c) => c.status === 'done')).toBe(true)
  })

  it('gives up on a missing history entry after 8 looks', async () => {
    let calls = 0
    fake.client.script = (_n, id, h) => h.onSuccess?.(id)
    fake.client.history = async () => (calls++, null)
    const svc = service({ pollMs: 1000 })
    const job = await svc.start(req({ prompts: ['one'] }))
    const done = await waitDone(svc, job.id)
    expect(done.cells.every((c) => c.status === 'error')).toBe(true)
    expect(calls).toBe(16)
  })

  it('records an execution error on one cell and continues', async () => {
    fake.client.script = (n, id, h, f) => (n === 1 ? h.onError?.({ promptId: id, message: 'out of memory', nodeId: '3' }) : succeed(n, id, h, f))
    const svc = service()
    const job = await svc.start(req())
    const done = await waitDone(svc, job.id)
    const c = done.cells.find((x) => x.col === 0 && x.row === 0)
    expect(c).toMatchObject({ status: 'error', error: 'out of memory' })
    expect(done.status).toBe('done')
    expect(done.cells.filter((x) => x.status === 'done')).toHaveLength(3)
  })

  it('records a template that does not fit as a cell error', async () => {
    const svc = service()
    const job = await svc.start(req({ prompts: ['BAD', 'good'] }))
    const done = await waitDone(svc, job.id)
    expect(done.cells.filter((c) => c.row === 0).every((c) => c.status === 'error' && /does not fit/.test(c.error ?? ''))).toBe(true)
    expect(done.cells.filter((c) => c.row === 1).every((c) => c.status === 'done')).toBe(true)
  })

  it('becomes an error when every cell failed', async () => {
    fake.client.script = (_n, id, h) => h.onError?.({ promptId: id, message: 'boom', nodeId: null })
    const svc = service()
    const job = await svc.start(req())
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('error')
  })

  it('learns the end of a cell from history when no websocket event arrives', async () => {
    fake.client.script = (_n, id, _h, f) => {
      f.histories.set(id, okEntry(id))
    }
    const svc = service()
    const job = await svc.start(req({ prompts: ['one'] }))
    expect((await waitDone(svc, job.id)).status).toBe('done')
  })

  it('stops with a job error when the connection is lost, cancelling the rest', async () => {
    fake.client.script = (n, id, h, f) => {
      if (n === 1) return succeed(n, id, h, f)
      f.historyThrows = new Error('connect ECONNREFUSED')
    }
    const svc = service()
    const job = await svc.start(req())
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toMatch(/Lost the connection.*ECONNREFUSED/)
    expect(done.cells.filter((c) => c.status === 'done')).toHaveLength(1)
    expect(done.cells.filter((c) => c.status === 'cancelled')).toHaveLength(3)
    expect(fake.client.queued).toHaveLength(2)
    const rec = JSON.parse(await readFile(join(gridsDir, job.id, 'grid.json'), 'utf8')) as GridRecord
    expect(rec.job.status).toBe('error')
  })

  it('treats a queue request with no answer as a lost connection, but an HTTP 400 as a cell error', async () => {
    fake.client.queueThrows = new Error('Could not reach http://x')
    const svc = service()
    const job = await svc.start(req())
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('error')
    expect(done.cells.every((c) => c.status === 'cancelled')).toBe(true)

    fake.client.queueThrows = Object.assign(new Error('node 3: bad input'), { status: 400 })
    const j2 = await svc.start(req({ prompts: ['one'] }))
    const d2 = await waitDone(svc, j2.id)
    expect(d2.status).toBe('error')
    expect(d2.cells.every((c) => c.status === 'error' && /bad input/.test(c.error ?? ''))).toBe(true)
  })

  it('emits status changes at once and throttles progress events', async () => {
    let t = 1_000_000
    fake.client.script = (n, id, h, f) => {
      for (let i = 1; i <= 10; i++) {
        t += 10
        h.onProgress?.({ promptId: id, node: '1', value: i, max: 10 })
      }
      succeed(n, id, h, f)
    }
    const svc = service({ now: () => t })
    const events: GridJob[] = []
    svc.on('job', (j: GridJob) => events.push(j))
    const job = await svc.start(req({ prompts: ['one'] }))
    await waitDone(svc, job.id)
    expect(events.at(-1)?.status).toBe('done')
    expect(events[0].status).toBe('running')
    const withProgress = events.filter((e) => e.progress && e.progress.max === 10)
    expect(withProgress.length).toBeGreaterThan(0)
    expect(withProgress.length).toBeLessThan(20)
  })
})

describe('cancel', () => {
  it('deletes only its own prompt, interrupts by its id and marks the rest cancelled', async () => {
    fake.client.script = () => undefined
    const svc = service()
    const job = await svc.start(req())
    for (let i = 0; i < 100 && fake.client.queued.length === 0; i++) await new Promise((r) => setTimeout(r, 5))
    const cancelled = await svc.cancel(job.id)
    expect(cancelled.status).toBe('cancelled')
    expect(fake.client.deleted).toEqual([['p1']])
    expect(fake.client.interrupted).toEqual(['p1'])
    expect(cancelled.cells.every((c) => c.status === 'cancelled')).toBe(true)
    await new Promise((r) => setTimeout(r, 30))
    expect(fake.client.queued).toHaveLength(1)
    const rec = JSON.parse(await readFile(join(gridsDir, job.id, 'grid.json'), 'utf8')) as GridRecord
    expect(rec.job.status).toBe('cancelled')
    expect((await svc.cancel(job.id)).status).toBe('cancelled')
    expect(fake.client.deleted).toHaveLength(1)
  })

  it('keeps finished cells and cancels the later ones', async () => {
    fake.client.script = (n, id, h, f) => {
      if (n === 1) succeed(n, id, h, f)
    }
    const svc = service()
    const job = await svc.start(req())
    for (let i = 0; i < 200 && fake.client.queued.length < 2; i++) await new Promise((r) => setTimeout(r, 5))
    const c = await svc.cancel(job.id)
    expect(c.cells.filter((x) => x.status === 'done')).toHaveLength(1)
    expect(c.cells.filter((x) => x.status === 'cancelled')).toHaveLength(3)
    expect(fake.client.deleted).toEqual([['p2']])
  })

  it('throws and keeps running when the server could not be reached', async () => {
    fake.client.script = () => undefined
    const svc = service()
    const job = await svc.start(req())
    for (let i = 0; i < 100 && fake.client.queued.length === 0; i++) await new Promise((r) => setTimeout(r, 5))
    fake.client.deletePending = async () => {
      throw new Error('down')
    }
    fake.client.interrupt = async () => {
      throw new Error('down')
    }
    fake.client.queue = async () => ({ running: ['p1'], pending: [] })
    await expect(svc.cancel(job.id)).rejects.toThrow('down')
    expect((await svc.list()).find((j) => j.id === job.id)?.status).toBe('running')
    fake.client.deletePending = async () => undefined
    fake.client.interrupt = async () => undefined
    await svc.cancel(job.id)
  })

  it('keeps running to the end after a cancel that failed (the loop never exits on it)', async () => {
    let release: (() => void) | null = null
    fake.client.script = (n, id, h, f) => {
      if (n === 1) release = () => succeed(n, id, h, f)
      else succeed(n, id, h, f)
    }
    const svc = service()
    const job = await svc.start(req())
    for (let i = 0; i < 100 && fake.client.queued.length === 0; i++) await new Promise((r) => setTimeout(r, 5))
    fake.client.deletePending = async () => {
      throw new Error('down')
    }
    fake.client.interrupt = async () => {
      throw new Error('down')
    }
    fake.client.queue = async () => ({ running: ['p1'], pending: [] })
    await expect(svc.cancel(job.id)).rejects.toThrow('down')
    await new Promise((r) => setTimeout(r, 20))
    expect(fake.client.queued).toHaveLength(1)
    release!()
    const done = await waitDone(svc, job.id)
    expect(done.status).toBe('done')
    expect(done.cells.every((c) => c.status === 'done')).toBe(true)
  })

  it('counts a cancel as done when both calls fail but the prompt is no longer in the queue', async () => {
    fake.client.script = () => undefined
    const svc = service()
    const job = await svc.start(req())
    for (let i = 0; i < 100 && fake.client.queued.length === 0; i++) await new Promise((r) => setTimeout(r, 5))
    fake.client.deletePending = async () => {
      throw new Error('down')
    }
    fake.client.interrupt = async () => {
      throw new Error('down')
    }
    expect((await svc.cancel(job.id)).status).toBe('cancelled')
  })

  it('cancels a prompt whose submission is still in flight, by its own id only', async () => {
    let answer: ((v: { promptId: string; number: number }) => void) | null = null
    fake.client.queuePrompt = async (prompt) => {
      fake.client.queued.push({ prompt, id: 'late' })
      return new Promise((r) => {
        answer = r
      })
    }
    const svc = service()
    const job = await svc.start(req())
    for (let i = 0; i < 100 && fake.client.queued.length === 0; i++) await new Promise((r) => setTimeout(r, 5))
    const pending = svc.cancel(job.id)
    await new Promise((r) => setTimeout(r, 20))
    expect(fake.client.deleted).toEqual([])
    answer!({ promptId: 'late', number: 1 })
    const c = await pending
    expect(c.status).toBe('cancelled')
    expect(fake.client.deleted).toEqual([['late']])
    expect(fake.client.interrupted).toEqual(['late'])
    expect(c.cells.every((x) => x.status === 'cancelled')).toBe(true)
    await new Promise((r) => setTimeout(r, 30))
    expect(fake.client.queued).toHaveLength(1)
  })

  describe('on a server the app did not start', () => {
    beforeEach(() => {
      fake.conn = { ...fake.conn, own: false, sharedQueueAccepted: true }
      fake.client.script = () => undefined
    })
    const startAndWait = async (svc: GridService): Promise<GridJob> => {
      const job = await svc.start(req())
      for (let i = 0; i < 100 && fake.client.queued.length === 0; i++) await new Promise((r) => setTimeout(r, 5))
      return job
    }

    it('deletes by id and never interrupts', async () => {
      const svc = service()
      const job = await startAndWait(svc)
      const c = await svc.cancel(job.id)
      expect(c.status).toBe('cancelled')
      expect(fake.client.deleted).toEqual([['p1']])
      expect(fake.client.interrupted).toEqual([])
    })

    it('lets a prompt that already runs there finish and ignores its result', async () => {
      const svc = service()
      const job = await startAndWait(svc)
      fake.client.deletePending = async () => {
        throw new Error('cannot delete a running prompt')
      }
      fake.client.queue = async () => ({ running: ['p1'], pending: [] })
      const c = await svc.cancel(job.id)
      expect(c.status).toBe('cancelled')
      expect(c.cells.every((x) => x.status === 'cancelled')).toBe(true)
      expect(fake.client.interrupted).toEqual([])
      fake.client.histories.set('p1', okEntry('p1'))
      fake.client.handlers?.onSuccess?.('p1')
      await new Promise((r) => setTimeout(r, 30))
      expect((await svc.list()).find((j) => j.id === job.id)?.cells.every((x) => x.status === 'cancelled')).toBe(true)
      expect(fake.client.queued).toHaveLength(1)
    })

    it('throws when the delete failed and the prompt still waits in the queue', async () => {
      const svc = service()
      const job = await startAndWait(svc)
      fake.client.deletePending = async () => {
        throw new Error('down')
      }
      fake.client.queue = async () => ({ running: [], pending: ['p1'] })
      await expect(svc.cancel(job.id)).rejects.toThrow('down')
      expect((await svc.list()).find((j) => j.id === job.id)?.status).toBe('running')
      fake.client.deletePending = async () => undefined
      await svc.cancel(job.id)
    })
  })

  it('refuses an unknown grid', async () => {
    await expect(service().cancel('nope')).rejects.toThrow(/not running/)
  })
})

describe('list', () => {
  const record = (id: string, status: GridJob['status'], createdAt: number): GridRecord => ({
    app: 'RulZ Model Merger',
    appVersion: '1',
    serverVersion: null,
    job: {
      id,
      request: req(),
      templateName: 't.json',
      status,
      serverUrl: 'http://x',
      own: true,
      createdAt,
      finishedAt: status === 'running' ? null : createdAt + 1,
      cells: [
        { col: 0, row: 0, status: 'done', file: 'c0-r0.png', error: null, seconds: 1 },
        { col: 1, row: 0, status: status === 'running' ? 'running' : 'done', file: null, error: null, seconds: null },
        { col: 0, row: 1, status: status === 'running' ? 'pending' : 'done', file: null, error: null, seconds: null }
      ],
      progress: null,
      error: null
    }
  })
  const put = async (id: string, text: string): Promise<void> => {
    await mkdir(join(gridsDir, id), { recursive: true })
    await writeFile(join(gridsDir, id, 'grid.json'), text)
  }

  it('merges disk and memory newest first without duplicates, skipping broken folders', async () => {
    await put('old', JSON.stringify(record('old', 'done', 10)))
    await put('mid', JSON.stringify(record('mid', 'error', 20)))
    await put('broken', '{ nope')
    await put('wrongid', JSON.stringify(record('other', 'done', 99)))
    await mkdir(join(gridsDir, 'empty'), { recursive: true })
    const svc = service()
    const job = await svc.start(req())
    await waitDone(svc, job.id)
    const list = await svc.list()
    expect(list.map((j) => j.id)).toEqual([job.id, 'mid', 'old'])
    expect(list.filter((j) => j.id === job.id)).toHaveLength(1)
  })

  it('shows a grid left running on disk as cancelled with its unfinished cells cancelled', async () => {
    await put('stale', JSON.stringify(record('stale', 'running', 10)))
    const [j] = await service().list()
    expect(j.status).toBe('cancelled')
    expect(j.cells.map((c) => c.status)).toEqual(['done', 'cancelled', 'cancelled'])
    expect(j.finishedAt).not.toBeNull()
  })

  it('returns an empty list when the folder does not exist', async () => {
    expect(await service().list()).toEqual([])
  })
})

describe('image, saveImage, openFolder, delete', () => {
  async function finished(): Promise<{ svc: GridService; id: string }> {
    const svc = service()
    const job = await svc.start(req({ prompts: ['one'] }))
    await waitDone(svc, job.id)
    return { svc, id: job.id }
  }

  it('returns a data URL for a valid cell image', async () => {
    const { svc, id } = await finished()
    const url = await svc.image(id, 'c0-r0.png')
    expect(url).toBe(`data:image/png;base64,${Buffer.from('PNG-out.png').toString('base64')}`)
    await writeFile(join(gridsDir, id, 'c1-r0.jpg'), 'J')
    expect(await svc.image(id, 'c1-r0.jpg')).toMatch(/^data:image\/jpeg;base64,/)
    await writeFile(join(gridsDir, id, 'c2-r0.webp'), 'W')
    expect(await svc.image(id, 'c2-r0.webp')).toMatch(/^data:image\/webp;base64,/)
  })

  it('refuses names that are not cell images and ids that are not folders', async () => {
    const { svc, id } = await finished()
    for (const bad of ['grid.json', 'template.json', '../grid.json', '..\\x.png', 'c0-r0.png.exe', 'c0-r0.gif', 'a/c0-r0.png', 'c-r.png', '']) {
      await expect(svc.image(id, bad)).rejects.toThrow()
    }
    await writeFile(join(root, 'secret.png'), 'S')
    for (const badId of ['..', '../', `..${'/'}${'grids'}`, 'a/b', 'a\\b', '', 'nope', '.']) {
      await expect(svc.image(badId, 'c0-r0.png')).rejects.toThrow(/not known|not a grid/)
    }
  })

  it('knows a grid that exists only on disk', async () => {
    await mkdir(join(gridsDir, 'disk'), { recursive: true })
    await writeFile(join(gridsDir, 'disk', 'grid.json'), '{}')
    await writeFile(join(gridsDir, 'disk', 'c0-r0.png'), 'D')
    expect(await service().image('disk', 'c0-r0.png')).toMatch(/^data:image\/png;base64,/)
  })

  it('saves only PNG data URLs under the size cap, through the save dialog', async () => {
    const svc = service()
    const ok = `data:image/png;base64,${Buffer.from('hello').toString('base64')}`
    await expect(svc.saveImage('data:image/jpeg;base64,AAAA', 'x.png')).rejects.toThrow(/Only PNG/)
    await expect(svc.saveImage('data:text/html;base64,AAAA', 'x.png')).rejects.toThrow(/Only PNG/)
    await expect(svc.saveImage('file:///c:/x.png', 'x.png')).rejects.toThrow(/Only PNG/)
    await expect(svc.saveImage('data:image/png;base64,@@@', 'x.png')).rejects.toThrow(/not valid/)
    await expect(svc.saveImage(`data:image/png;base64,${'A'.repeat(90 * 1024 * 1024)}`, 'x.png')).rejects.toThrow(/64 MB/)
    expect(await svc.saveImage(ok, 'x.png')).toBeNull()
    const target = join(root, 'out.png')
    savePath = target
    expect(await svc.saveImage(ok, 'x.png')).toBe(target)
    expect((await readFile(target)).toString()).toBe('hello')
  })

  it('sanitizes the suggested file name', async () => {
    const names: string[] = []
    const svc = service({ pickSavePath: async (n) => (names.push(n), null) })
    const ok = `data:image/png;base64,${Buffer.from('h').toString('base64')}`
    await svc.saveImage(ok, '..\\..\\evil:name?.PNG')
    await svc.saveImage(ok, '')
    expect(names).toEqual(['.._.._evil_name_.png'.replace(/^\.+/, ''), 'grid.png'])
    expect(names[0]).not.toMatch(/[\\/:?]/)
  })

  it('opens the folder of a known grid only', async () => {
    const { svc, id } = await finished()
    await svc.openFolder(id)
    expect(opened).toEqual([join(gridsDir, id)])
    await expect(svc.openFolder('../x')).rejects.toThrow()
    expect(opened).toHaveLength(1)
  })

  it('moves a finished grid to the recycle bin and refuses while it runs', async () => {
    fake.client.script = () => undefined
    const svc = service()
    const job = await svc.start(req())
    await expect(svc.delete(job.id)).rejects.toThrow(/still running/)
    expect(trashed).toEqual([])
    await svc.cancel(job.id)
    await svc.delete(job.id)
    expect(trashed).toEqual([join(gridsDir, job.id)])
    await expect(svc.delete('../x')).rejects.toThrow()
  })
})
