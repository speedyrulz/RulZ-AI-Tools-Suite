import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_RECIPE, type MergeRecipe } from '@shared/merge'
import { DEFAULT_SETTINGS } from '@shared/types'
import type { ComfyEventHandlers } from '../src/main/comfy/client'
import { buildMergeGraph } from '../src/main/merge/graph'
import { GridService, type GridClient } from '../src/main/grid/service'
import { MergeRunner, type MergeClient, type MergeConnection } from '../src/main/merge/runner'
import type { SafetensorsHeader, SafetensorsTensor } from '../src/main/models/safetensors'
import type { WslRunner } from '../src/main/server/wsl'

const files = new Map<string, unknown>()
const graphSpy = vi.fn()

vi.mock('../src/main/models/safetensors', () => ({
  readSafetensorsHeader: async (p: string) => {
    if (!files.has(p)) throw new Error(`no such file ${p}`)
    return files.get(p)
  }
}))
vi.mock('../src/main/merge/inspect', async () => {
  const actual = await vi.importActual<typeof import('../src/main/merge/inspect')>('../src/main/merge/inspect')
  return {
    ...actual,
    describeModelFile: (name: string, header: { fileBytes: number } | null) => ({ name, readable: !!header, error: null, sizeBytes: header?.fileBytes ?? null }),
    normalizeKeys: (h: { tensors: Record<string, unknown> }) => ({ prefix: '', keys: new Map(Object.entries(h.tensors)) }),
    blockNodeOptions: () => [],
    checkMerge: () => ({ problems: [] })
  }
})
vi.mock('../src/main/merge/graph', async () => {
  const actual = await vi.importActual<typeof import('../src/main/merge/graph')>('../src/main/merge/graph')
  return {
    ...actual,
    validateRecipe: () => [],
    buildMergeGraph: (...a: Parameters<typeof actual.buildMergeGraph>) => {
      graphSpy(...a)
      return actual.buildMergeGraph(...a)
    }
  }
})

const hdr = (keys: string[], metadata: Record<string, string> = {}): unknown => ({
  tensors: Object.fromEntries(keys.map((k) => [k, { dtype: 'BF16', shape: [1], offsets: [0, 2] }])),
  metadata,
  headerBytes: 100,
  fileBytes: 1000,
  headerSha256: 'sha'
})

const BASE = '/home/jeff/.rulz/comfy'
const MODELS = '/home/jeff/ComfyUI/models'
const recipe: MergeRecipe = { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors', outputName: 'merged' }
const A_PATH = `UNC:${MODELS}/diffusion_models/a.safetensors`
const OUT_PATH = `UNC:${BASE}/output/diffusion_models/merged_00001_.safetensors`

function makeClient() {
  const c = {
    handlers: null as ComfyEventHandlers | null,
    queued: [] as Array<{ prompt: Record<string, { class_type: string; inputs: Record<string, unknown> }> }>,
    queueState: { running: [] as string[], pending: [] as string[] },
    objectInfoCalls: 0,
    deleteThrows: null as Error | null,
    queueThrows: null as Error | null,
    interrupted: [] as Array<string | undefined>,
    objectInfo: vi.fn(async () => {
      c.objectInfoCalls++
      return { ModelMergeSimple: {} }
    }),
    queuePrompt: vi.fn(async (prompt: never) => {
      c.queued.push({ prompt })
      return { promptId: 'p1', number: 1 }
    }),
    history: vi.fn(async () => null),
    queue: vi.fn(async () => {
      if (c.queueThrows) throw c.queueThrows
      return c.queueState
    }),
    deletePending: vi.fn(async () => {
      if (c.deleteThrows) throw c.deleteThrows
    }),
    interrupt: vi.fn(async (id?: string) => {
      c.interrupted.push(id)
    }),
    openEvents: vi.fn((_id: string, h: ComfyEventHandlers) => {
      c.handlers = h
      return { close: () => undefined }
    })
  }
  return c
}

function makeRunner(client: ReturnType<typeof makeClient>, ids: string[] = ['job-1']) {
  const conn: MergeConnection = { url: 'http://127.0.0.1:8288', reachable: true, own: true, sharedQueueAccepted: true, version: '0.46.0' }
  const outputs: string[] = []
  const wsl: WslRunner = {
    run: async (_d, script) => {
      if (script.includes('RULZ_FILE')) {
        return { code: 0, stdout: outputs.map((n) => `RULZ_FILE|1000|1700000000|${n}\n`).join(''), stderr: '' }
      }
      if (script.includes('MemAvailable')) return { code: 0, stdout: 'RULZ_MEM=1024\n', stderr: '' }
      return { code: 0, stdout: `${MODELS}\n`, stderr: '' }
    },
    spawn: () => {
      throw new Error('not used')
    }
  }
  const runner = new MergeRunner({
    client: () => client as unknown as MergeClient,
    wsl,
    getConnection: async () => conn,
    getOwnRecord: () => ({ distro: 'Ubuntu-24.04', baseDir: BASE }),
    ownSettings: () => DEFAULT_SETTINGS.ownServer,
    appVersion: '0.2.0',
    now: () => 1700000000000,
    newId: () => ids.shift() ?? 'job-x',
    toUnc: (_d, p) => `UNC:${p}`,
    writeText: async () => undefined,
    pollMs: 3600000,
    findRetryMs: 1
  })
  return { runner, outputs }
}

beforeEach(() => {
  files.clear()
  graphSpy.mockClear()
  files.set(A_PATH, hdr(['diffusion_model.x.weight']))
  files.set(`UNC:${MODELS}/diffusion_models/b.safetensors`, hdr(['diffusion_model.x.weight']))
})

describe('B8: save node is unique per job', () => {
  it('two builds with different job ids differ only in one input of the save node', () => {
    const a = buildMergeGraph(recipe, { jobId: 'job-1' })
    const b = buildMergeGraph(recipe, { jobId: 'job-2' })
    expect(JSON.stringify(a.prompt)).not.toBe(JSON.stringify(b.prompt))
    const strip = (g: typeof a): unknown => {
      const copy = structuredClone(g.prompt) as Record<string, { inputs: Record<string, unknown> }>
      delete copy[g.saveNode === 'ModelSave' ? Object.keys(copy).find((k) => (copy[k] as { class_type?: string }).class_type === 'ModelSave')! : ''].inputs.rulz_job_id
      return copy
    }
    expect(strip(a)).toEqual(strip(b))
    expect(a.filenamePrefix).toBe(b.filenamePrefix)
  })
  it('also for checkpoints, and unchanged without a job id', () => {
    const ck = { ...recipe, kind: 'checkpoint' as const }
    const g = buildMergeGraph(ck, { jobId: 'j' })
    const save = (Object.values(g.prompt) as Array<{ class_type: string; inputs: Record<string, unknown> }>).find((n) => n.class_type === 'CheckpointSave')!
    expect(save.inputs.rulz_job_id).toBe('j')
    const plain = (Object.values(buildMergeGraph(ck).prompt) as Array<{ class_type: string; inputs: Record<string, unknown> }>).find((n) => n.class_type === 'CheckpointSave')!
    expect('rulz_job_id' in plain.inputs).toBe(false)
  })
  it('the runner passes its job id, so a repeat of the same recipe is a different prompt', async () => {
    const client = makeClient()
    const { runner } = makeRunner(client, ['job-1', 'job-2'])
    await runner.start(recipe)
    const first = client.queued[0].prompt
    // Finish the first job so a second may start.
    client.handlers!.onError!({ promptId: 'p1', message: 'x', nodeId: null })
    await runner.start(recipe)
    const second = client.queued[1].prompt
    expect(graphSpy.mock.calls.map((c) => c[1])).toEqual([{ jobId: 'job-1' }, { jobId: 'job-2' }])
    const saveOf = (p: typeof first): Record<string, unknown> => Object.values(p).find((n) => n.class_type === 'ModelSave')!.inputs
    expect(saveOf(first).rulz_job_id).toBe('job-1')
    expect(saveOf(second).rulz_job_id).toBe('job-2')
  })
})

describe('B9: cancel when the prompt is still queued', () => {
  async function started() {
    const client = makeClient()
    const { runner } = makeRunner(client)
    await runner.start(recipe)
    return { client, runner }
  }
  it('throws and keeps the job when delete failed and the prompt is still pending', async () => {
    const { client, runner } = await started()
    client.deleteThrows = new Error('delete refused')
    client.queueState = { running: [], pending: ['p1'] }
    await expect(runner.cancel('job-1')).rejects.toThrow(/still waiting/)
    expect(runner.jobs()[0].status).toBe('queued')
  })
  it('throws when delete failed and the queue cannot be read', async () => {
    const { client, runner } = await started()
    client.deleteThrows = new Error('delete refused')
    client.queueThrows = new Error('queue down')
    await expect(runner.cancel('job-1')).rejects.toThrow(/queue could not be read/)
    expect(runner.jobs()[0].status).toBe('queued')
  })
  it('cancels when delete failed but the prompt is no longer pending', async () => {
    const { client, runner } = await started()
    client.deleteThrows = new Error('delete refused')
    client.queueState = { running: ['p1'], pending: [] }
    const j = await runner.cancel('job-1')
    expect(j.status).toBe('cancelled')
  })
})

describe('O1: only .weight and .bias keys are real', () => {
  async function verifyWith(aKeys: string[], outKeys: string[]) {
    files.set(A_PATH, hdr(aKeys))
    const client = makeClient()
    const { runner, outputs } = makeRunner(client)
    await runner.start(recipe)
    files.set(OUT_PATH, hdr(outKeys, { modelmerger_recipe: JSON.stringify({ jobId: 'job-1' }) }))
    outputs.push('merged_00001_.safetensors')
    client.handlers!.onSuccess!('p1')
    const end = Date.now() + 2000
    while (runner.jobs()[0].status !== 'done') {
      if (Date.now() > end) throw new Error('timed out')
      await new Promise((r) => setTimeout(r, 5))
    }
    return runner.jobs()[0].output!.verify!
  }
  it('a key such as "xweight" is a buffer, not a missing weight', async () => {
    const v = await verifyWith(['diffusion_model.l.weight', 'diffusion_model.xweight'], ['diffusion_model.l.weight'])
    expect(v).toMatchObject({ ok: true, missingKeys: 0 })
  })
  it('"a.weight" still counts as missing', async () => {
    const v = await verifyWith(['diffusion_model.l.weight', 'diffusion_model.a.weight'], ['diffusion_model.l.weight'])
    expect(v).toMatchObject({ ok: false, missingKeys: 1 })
  })
})

describe('B10: mixed storage formats are not shape problems', () => {
  const t = (dtype: string, shape: number[]): SafetensorsTensor => ({ dtype, shape, offsets: [0, 0] })
  const h = (tensors: Record<string, SafetensorsTensor>): SafetensorsHeader => ({ tensors, metadata: {}, headerBytes: 10, headerSha256: 'x', fileBytes: 1000 })
  // The file info comes from the real describeModelFile, as in the app (it decides whether a file is quantized).
  const side = (actual: typeof import('../src/main/merge/inspect'), header: SafetensorsHeader) => ({ info: actual.describeModelFile('x', header), header })
  it('packed 4-bit weights and their scales against bf16 give no shape problem', async () => {
    const actual = await vi.importActual<typeof import('../src/main/merge/inspect')>('../src/main/merge/inspect')
    const a: Record<string, SafetensorsTensor> = {}
    const b: Record<string, SafetensorsTensor> = {}
    for (let i = 0; i < 10; i++) {
      a[`diffusion_model.blocks.${i}.weight`] = t('U8', [64, 32])
      a[`diffusion_model.blocks.${i}.weight_scale`] = t('F8_E4M3', [64, 4])
      a[`diffusion_model.blocks.${i}.weight_scale_2`] = t('F32', [])
      b[`diffusion_model.blocks.${i}.weight`] = t('BF16', [64, 64])
      b[`diffusion_model.blocks.${i}.weight_scale`] = t('BF16', [1])
      b[`diffusion_model.blocks.${i}.weight_scale_2`] = t('BF16', [1])
    }
    const r = actual.checkMerge({ kind: 'diffusion', mode: 'simple', a: side(actual, h(a)), b: side(actual, h(b)), c: null, objectInfo: { ModelMergeSimple: { input: { required: { model1: ['MODEL'], model2: ['MODEL'], ratio: ['FLOAT', {}] } } } }, ramFreeBytes: null, loras: [] } as never)
    expect(r.problems.filter((p) => /different shape/.test(p))).toEqual([])
  })
  it('two bf16 files with a real shape difference still fail', async () => {
    const actual = await vi.importActual<typeof import('../src/main/merge/inspect')>('../src/main/merge/inspect')
    const a: Record<string, SafetensorsTensor> = {}
    const b: Record<string, SafetensorsTensor> = {}
    for (let i = 0; i < 10; i++) {
      a[`diffusion_model.blocks.${i}.weight`] = t('BF16', [64, 64])
      b[`diffusion_model.blocks.${i}.weight`] = t('BF16', [64, 32])
    }
    const r = actual.checkMerge({ kind: 'diffusion', mode: 'simple', a: side(actual, h(a)), b: side(actual, h(b)), c: null, objectInfo: { ModelMergeSimple: { input: { required: { model1: ['MODEL'], model2: ['MODEL'], ratio: ['FLOAT', {}] } } } }, ramFreeBytes: null, loras: [] } as never)
    expect(r.problems.some((p) => /different shape/.test(p))).toBe(true)
  })
  const run = async (a: Record<string, SafetensorsTensor>, b: Record<string, SafetensorsTensor>) => {
    const actual = await vi.importActual<typeof import('../src/main/merge/inspect')>('../src/main/merge/inspect')
    return actual.checkMerge({ kind: 'diffusion', mode: 'simple', a: side(actual, h(a)), b: side(actual, h(b)), c: null, objectInfo: { ModelMergeSimple: { input: { required: { model1: ['MODEL'], model2: ['MODEL'], ratio: ['FLOAT', {}] } } } }, ramFreeBytes: null, loras: [] } as never)
  }
  it('review: a "scale" key of two unquantized files with different shapes still fails', async () => {
    const a: Record<string, SafetensorsTensor> = {}
    const b: Record<string, SafetensorsTensor> = {}
    for (let i = 0; i < 10; i++) {
      a[`diffusion_model.blocks.${i}.scale_shift_table`] = t('BF16', [6, 64])
      b[`diffusion_model.blocks.${i}.scale_shift_table`] = t('BF16', [6, 128])
    }
    expect((await run(a, b)).problems.some((p) => /different shape/.test(p))).toBe(true)
  })
  it('review: a layer with "scale" in its name is still compared in mixed-format files', async () => {
    const a: Record<string, SafetensorsTensor> = {}
    const b: Record<string, SafetensorsTensor> = {}
    for (let i = 0; i < 10; i++) {
      a[`diffusion_model.blocks.${i}.weight`] = t('U8', [64, 32])
      a[`diffusion_model.blocks.${i}.weight_scale_2`] = t('F32', [])
      a[`diffusion_model.blocks.${i}.upscale.weight`] = t('BF16', [64, 64])
      b[`diffusion_model.blocks.${i}.weight`] = t('BF16', [64, 64])
      b[`diffusion_model.blocks.${i}.upscale.weight`] = t('BF16', [32, 64])
    }
    const r = await run(a, b)
    expect(r.problems.some((p) => /different shape/.test(p) && /upscale/.test(p))).toBe(true)
  })
  it('review: a packed weight whose leading dimensions differ still fails', async () => {
    const a: Record<string, SafetensorsTensor> = {}
    const b: Record<string, SafetensorsTensor> = {}
    for (let i = 0; i < 10; i++) {
      a[`diffusion_model.blocks.${i}.weight`] = t('U8', [128, 32])
      a[`diffusion_model.blocks.${i}.weight_scale_2`] = t('F32', [])
      b[`diffusion_model.blocks.${i}.weight`] = t('BF16', [64, 64])
    }
    expect((await run(a, b)).problems.some((p) => /different shape/.test(p))).toBe(true)
  })
})

describe('B13: clearObjectInfoCache', () => {
  it('MergeRunner fetches the node list again after the cache was cleared', async () => {
    const client = makeClient()
    const { runner } = makeRunner(client)
    await runner.start(recipe)
    client.handlers!.onError!({ promptId: 'p1', message: 'x', nodeId: null })
    await runner.start(recipe)
    expect(client.objectInfoCalls).toBe(1)
    client.handlers!.onError!({ promptId: 'p1', message: 'x', nodeId: null })
    runner.clearObjectInfoCache()
    await runner.start(recipe)
    expect(client.objectInfoCalls).toBe(2)
  })
  it('review: a node-list request that was under way when the cache was cleared does not refill it', async () => {
    const client = makeClient()
    const { runner } = makeRunner(client)
    const pending: Array<() => void> = []
    client.objectInfo.mockImplementationOnce(
      () =>
        new Promise((resolve) =>
          pending.push(() => {
            client.objectInfoCalls++
            resolve({ ModelMergeSimple: {} })
          })
        )
    )
    const first = runner.start(recipe)
    await vi.waitFor(() => expect(pending.length).toBe(1))
    runner.clearObjectInfoCache()
    pending.shift()!()
    await first
    client.handlers!.onError!({ promptId: 'p1', message: 'x', nodeId: null })
    const before = client.objectInfoCalls
    await runner.start(recipe)
    expect(client.objectInfoCalls).toBe(before + 1)
  })
  it('GridService has the method', () => {
    const s = new GridService({} as never)
    expect(() => s.clearObjectInfoCache()).not.toThrow()
    void (null as unknown as GridClient)
  })
})

// Variable paths keep the node tsconfig from pulling the renderer (DOM) sources into its program.
const HISTORY_STORE = '../src/renderer/src/stores/history'
const GRID_STORE = '../src/renderer/src/stores/grid'

describe('B11 and P1: renderer stores', () => {
  const invokeMock = vi.fn()
  beforeEach(() => {
    invokeMock.mockReset()
    ;(globalThis as unknown as { window: unknown }).window = {
      rulz: { invoke: invokeMock, on: () => () => undefined },
      setTimeout,
      clearTimeout
    }
  })

  it('history remove keeps the row when the main process refuses', async () => {
    const { useHistory } = await import(/* @vite-ignore */ HISTORY_STORE)
    const entry = { kind: 'merge', job: { id: 'm1' } } as never
    useHistory.setState({ entries: [entry] } as never)
    invokeMock.mockRejectedValueOnce(new Error('disk locked'))
    await useHistory.getState().remove(entry)
    expect(useHistory.getState().entries).toHaveLength(1)
    invokeMock.mockResolvedValueOnce(undefined)
    await useHistory.getState().remove(entry)
    expect(useHistory.getState().entries).toHaveLength(0)
  })

  it('grid images are kept for the 3 newest grids plus the selected one', async () => {
    const { useGrid } = await import(/* @vite-ignore */ GRID_STORE)
    const grids = ['g1', 'g2', 'g3', 'g4', 'g5'].map((id) => ({ id })) as never
    useGrid.setState({ grids, images: {}, selectedId: 'g1' } as never)
    invokeMock.mockImplementation(async (_c: string, id: string, file: string) => `data:${id}/${file}`)
    for (const id of ['g1', 'g2', 'g3', 'g4', 'g5']) await useGrid.getState().ensureImage(id, 'a.png')
    const have = (id: string): boolean => `${id}/a.png` in useGrid.getState().images
    // g1 is selected so it stays; g2 was the least recently used of the rest.
    expect(['g1', 'g2', 'g3', 'g4', 'g5'].map(have)).toEqual([true, false, true, true, true])
    // The grid being fetched for is never evicted by its own add.
    await useGrid.getState().ensureImage('g2', 'b.png')
    expect('g2/b.png' in useGrid.getState().images).toBe(true)
  })

  it('review: selecting a grid counts as using it', async () => {
    const { useGrid } = await import(/* @vite-ignore */ GRID_STORE)
    const ids = ['h1', 'h2', 'h3', 'h4', 'h5']
    useGrid.setState({ grids: ids.map((id) => ({ id })), images: {}, selectedId: null } as never)
    invokeMock.mockImplementation(async (_c: string, id: string, file: string) => `data:${id}/${file}`)
    for (const id of ['h1', 'h2', 'h3']) await useGrid.getState().ensureImage(id, 'a.png')
    // h1 is viewed again (selected), then the user moves on to h4 and h5.
    useGrid.getState().select('h1')
    useGrid.getState().select('h4')
    await useGrid.getState().ensureImage('h4', 'a.png')
    await useGrid.getState().ensureImage('h5', 'a.png')
    const have = (id: string): boolean => `${id}/a.png` in useGrid.getState().images
    // h1 was used more recently than h2 and h3, so h2 and h3 go first.
    expect(ids.map(have)).toEqual([true, false, false, true, true])
  })
})
