import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_RECIPE, type MergeJob, type MergeRecipe } from '@shared/merge'
import { DEFAULT_SETTINGS } from '@shared/types'
import type { ComfyEventHandlers } from '../src/main/comfy/client'
import { MergeRunner, URL_MODE_MESSAGE, parseLoraNotLoaded, type MergeClient, type MergeConnection } from '../src/main/merge/runner'
import type { WslResult, WslRunner } from '../src/main/server/wsl'

const hdr = (keys: string[], metadata: Record<string, string> = {}): unknown => ({
  tensors: Object.fromEntries(keys.map((k) => [k, { dtype: 'BF16', shape: [1], offsets: [0, 2] }])),
  metadata,
  headerBytes: 100,
  fileBytes: 1000,
  headerSha256: `sha-${keys.length}`
})

const outHdr = (keys: string[], jobId = 'job-1'): unknown => hdr(keys, { modelmerger_recipe: JSON.stringify({ jobId }) })
const checkProblems = vi.fn<() => string[]>(() => [])
const files = new Map<string, unknown>()
const blockOpts = vi.fn<(...a: unknown[]) => unknown[]>(() => [])
const validate = vi.fn<(...a: unknown[]) => string[]>(() => [])

vi.mock('../src/main/models/safetensors', () => ({
  readSafetensorsHeader: async (p: string) => {
    if (!files.has(p)) throw new Error(`no such file ${p}`)
    return files.get(p)
  }
}))
vi.mock('../src/main/merge/inspect', () => ({
  describeModelFile: (name: string, header: { fileBytes: number } | null, error?: string) => ({ name, readable: !!header, error: error ?? null, sizeBytes: header?.fileBytes ?? null }),
  normalizeKeys: (h: { tensors: Record<string, unknown> }) => ({ prefix: '', keys: new Map(Object.entries(h.tensors)) }),
  blockNodeOptions: (...a: unknown[]) => blockOpts(...a),
  describeLoraFile: (name: string, header: { fileBytes: number } | null, aHeader: unknown, error: string | null) => ({ name, readable: !!header, error, sizeBytes: header?.fileBytes ?? null, aHeader }),
  isTextEncoderLoraKey: (k: string) => /lora_te|text_encoder/.test(k),
  checkMerge: (input: unknown) => ({ problems: checkProblems(), input })
}))
vi.mock('../src/main/merge/graph', () => ({
  validateRecipe: (...a: unknown[]) => validate(...a),
  completeBlocks: (r: MergeRecipe, inputs: string[]) => ({ ...r, blocks: { ...Object.fromEntries(inputs.map((i) => [i, r.ratio])), ...r.blocks } }),
  sanitizeOutputName: (s: string) => s,
  buildMergeGraph: (r: MergeRecipe) => ({
    prompt: { '1': { class_type: 'UNETLoader', inputs: {} }, '2': { class_type: 'ModelMergeSimple', inputs: {} }, '3': { class_type: 'ModelSave', inputs: {} } },
    saveNode: 'ModelSave',
    outputSubfolder: 'diffusion_models',
    filenamePrefix: `diffusion_models/${r.outputName}`
  })
}))

const BASE = '/home/jeff/.rulz/comfy'
const MODELS = '/home/jeff/ComfyUI/models'
const recipe: MergeRecipe = { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors', outputName: 'merged' }
const unc = (_d: string, p: string): string => `UNC:${p}`

interface Fake {
  client: MergeClient & { handlers: ComfyEventHandlers | null; closed: number; queued: unknown[]; deleted: string[][]; interrupted: Array<string | undefined>; queueState: { running: string[]; pending: string[] }; historyResult: unknown }
  wsl: WslRunner
  outputs: string[]
  moveReply: Partial<WslResult>
  sidecar: { path: string; text: string } | null
  conn: MergeConnection
  ownRecord: { distro: string; baseDir: string } | null
  runner: MergeRunner
  events: MergeJob[]
  logListeners: Set<(text: string) => void>
  emitLog: (text: string) => void
}

function setup(over: { conn?: Partial<MergeConnection>; own?: boolean; pollMs?: number } = {}): Fake {
  const f = {} as Fake
  f.outputs = []
  f.moveReply = { stdout: `RULZ_SIDECAR=${MODELS}/diffusion_models/merged/merged_00001_.safetensors.recipe.json\nRULZ_MOVED=${MODELS}/diffusion_models/merged/merged_00001_.safetensors\n` }
  f.sidecar = null
  f.conn = { url: 'http://127.0.0.1:8288', reachable: true, own: true, sharedQueueAccepted: true, version: '0.46.0', ...over.conn }
  f.ownRecord = over.own === false ? null : { distro: 'Ubuntu-24.04', baseDir: BASE }
  f.events = []
  const client = {
    handlers: null as ComfyEventHandlers | null,
    closed: 0,
    queued: [] as unknown[],
    deleted: [] as string[][],
    interrupted: [] as Array<string | undefined>,
    queueState: { running: [] as string[], pending: [] as string[] },
    historyResult: null as unknown,
    objectInfo: vi.fn(async () => ({ ModelMergeSimple: {} })),
    queuePrompt: vi.fn(async (prompt: Record<string, unknown>, clientId: string, extra?: Record<string, unknown>) => {
      client.queued.push({ prompt, clientId, extra })
      return { promptId: 'p1', number: 1 }
    }),
    history: vi.fn(async () => client.historyResult),
    queue: vi.fn(async () => client.queueState),
    deletePending: vi.fn(async (ids: string[]) => {
      client.deleted.push(ids)
    }),
    interrupt: vi.fn(async (id?: string) => {
      client.interrupted.push(id)
    }),
    openEvents: vi.fn((_id: string, h: ComfyEventHandlers) => {
      client.handlers = h
      return {
        close: () => {
          client.closed++
        }
      }
    })
  }
  f.client = client as unknown as Fake['client']
  f.wsl = {
    run: async (_d, script) => {
      if (script.includes('RULZ_FILE')) {
        const stdout = f.outputs.map((n) => `RULZ_FILE|1000|${1700000000 + f.outputs.indexOf(n)}|${n}\n`).join('')
        return { code: 0, stdout, stderr: '' }
      }
      if (script.includes('mv -n')) return { code: 0, stdout: '', stderr: '', ...f.moveReply }
      if (script.includes('MemAvailable')) return { code: 0, stdout: 'RULZ_MEM=1024\n', stderr: '' }
      return { code: 0, stdout: `${MODELS}\n`, stderr: '' }
    },
    spawn: () => {
      throw new Error('not used')
    }
  }
  f.logListeners = new Set()
  f.runner = new MergeRunner({
    client: () => f.client,
    wsl: f.wsl,
    getConnection: async () => f.conn,
    getOwnRecord: () => f.ownRecord,
    ownSettings: () => DEFAULT_SETTINGS.ownServer,
    onServerLog: (l) => {
      f.logListeners.add(l)
      return () => {
        f.logListeners.delete(l)
      }
    },
    appVersion: '0.2.0',
    now: () => 1700000000000,
    newId: () => 'job-1',
    toUnc: unc,
    writeText: async (p, t) => {
      f.sidecar = { path: p, text: t }
    },
    pollMs: over.pollMs ?? 3600000,
    findRetryMs: 1
  })
  f.emitLog = (text) => {
    for (const l of [...f.logListeners]) l(text)
  }
  f.runner.on('job', (j: MergeJob) => f.events.push(j))
  return f
}

async function waitFor(fn: () => boolean, ms = 2000): Promise<void> {
  const end = Date.now() + ms
  while (!fn()) {
    if (Date.now() > end) throw new Error('timed out waiting')
    await new Promise((r) => setTimeout(r, 5))
  }
}

const A_PATH = `UNC:${MODELS}/diffusion_models/a.safetensors`
const OUT_PATH = `UNC:${BASE}/output/diffusion_models/merged_00001_.safetensors`

beforeEach(() => {
  files.clear()
  blockOpts.mockReset()
  blockOpts.mockReturnValue([])
  checkProblems.mockReset()
  checkProblems.mockReturnValue([])
  validate.mockReset()
  validate.mockReturnValue([])
  files.set(A_PATH, hdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
  files.set(`UNC:${MODELS}/diffusion_models/b.safetensors`, hdr(['diffusion_model.x.weight']))
})

describe('start: refusals', () => {
  it('refuses when the server is not reachable', async () => {
    const f = setup({ conn: { reachable: false } })
    await expect(f.runner.start(recipe)).rejects.toThrow(/reachable/)
    expect(f.client.queued).toHaveLength(0)
  })
  it('refuses a shared server whose queue was not accepted', async () => {
    const f = setup({ conn: { own: false, sharedQueueAccepted: false } })
    await expect(f.runner.start(recipe)).rejects.toThrow(/shared queue/)
    expect(f.client.queued).toHaveLength(0)
  })
  it('refuses an invalid recipe with the problems', async () => {
    validate.mockReturnValue(['Model B is missing.', 'Ratio out of range.'])
    const f = setup()
    await expect(f.runner.start(recipe)).rejects.toThrow('Model B is missing. Ratio out of range.')
    expect(f.client.queued).toHaveLength(0)
  })
  it('passes the block node inputs to validateRecipe and stores the completed recipe', async () => {
    const f = setup()
    blockOpts.mockReturnValue([{ node: 'ModelMergeBlocks', inputs: ['input', 'middle', 'out'], match: null }])
    const job = await f.runner.start({ ...recipe, mode: 'blocks', blockNode: 'ModelMergeBlocks', ratio: 0.3, blocks: { middle: 0.9 } })
    expect(blockOpts.mock.calls[0][1]).toBeNull()
    expect(validate.mock.calls[0][2]).toEqual(['input', 'middle', 'out'])
    expect(job.recipe.blocks).toEqual({ input: 0.3, middle: 0.9, out: 0.3 })
    const q = f.client.queued[0] as { extra: { extra_pnginfo: { modelmerger_recipe: { recipe: MergeRecipe } } } }
    expect(q.extra.extra_pnginfo.modelmerger_recipe.recipe.blocks).toEqual({ input: 0.3, middle: 0.9, out: 0.3 })
  })
  it('allows one merge at a time', async () => {
    const f = setup()
    await f.runner.start(recipe)
    await expect(f.runner.start(recipe)).rejects.toThrow(/already queued or running/)
    await f.runner.cancel('job-1')
    await expect(f.runner.start(recipe)).resolves.toBeTruthy()
  })
})

describe('start: own server', () => {
  it('sends extra_data with the recipe record, then tracks progress to done with verify and sidecar', async () => {
    const f = setup()
    f.outputs = ['merged_00000_.safetensors']
    const job = await f.runner.start(recipe)
    expect(job.status).toBe('queued')
    expect(job.promptId).toBe('p1')
    // The websocket was opened before the prompt was queued.
    expect((f.client.openEvents as ReturnType<typeof vi.fn>).mock.invocationCallOrder[0]).toBeLessThan((f.client.queuePrompt as ReturnType<typeof vi.fn>).mock.invocationCallOrder[0])
    const q = f.client.queued[0] as { extra: { extra_pnginfo: { modelmerger_recipe: { app: string; sourceHeaderSha256: unknown; serverVersion: string; appVersion: string } } } }
    const rec = q.extra.extra_pnginfo.modelmerger_recipe
    expect(rec).toMatchObject({ app: 'RulZ Model Merger', appVersion: '0.2.0', serverVersion: '0.46.0', sourceHeaderSha256: { a: 'sha-2', b: 'sha-1', c: null, loras: [] } })

    f.client.handlers!.onExecuting!({ promptId: 'p1', node: '2' })
    f.client.handlers!.onProgress!({ promptId: 'p1', node: '2', value: 3, max: 10 })
    expect(f.runner.jobs()[0].status).toBe('running')
    expect(f.runner.jobs()[0].progress).toEqual({ node: 'ModelMergeSimple', value: 3, max: 10 })
    // Events of other prompts are ignored.
    f.client.handlers!.onError!({ promptId: 'other', message: 'x', nodeId: null })
    expect(f.runner.jobs()[0].status).toBe('running')

    files.set(`UNC:${BASE}/output/diffusion_models/merged_00001_.safetensors`, outHdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
    f.outputs = ['merged_00000_.safetensors', 'merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    const done = f.runner.jobs()[0]
    expect(done.output).toMatchObject({
      relPath: 'diffusion_models/merged_00001_.safetensors',
      wslPath: `${BASE}/output/diffusion_models/merged_00001_.safetensors`,
      sidecarPath: `${BASE}/output/diffusion_models/merged_00001_.safetensors.recipe.json`,
      movedTo: null,
      verify: { ok: true, missingKeys: 0, extraKeys: 0, recipeInHeader: true, tensorCount: 2 }
    })
    expect(f.sidecar!.path).toBe(`UNC:${BASE}/output/diffusion_models/merged_00001_.safetensors.recipe.json`)
    expect(JSON.parse(f.sidecar!.text).app).toBe('RulZ Model Merger')
    expect(f.client.closed).toBeGreaterThan(0)
    expect(done.finishedAt).not.toBeNull()
    expect(f.events.at(-1)!.status).toBe('done')
  })

  it('reports missing and extra keys against A, and unreadable A', async () => {
    const f = setup()
    f.client.handlers = null
    await f.runner.start(recipe)
    files.set(OUT_PATH, outHdr(['diffusion_model.x.weight', 'diffusion_model.z.weight']))
    f.outputs = ['merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    expect(f.runner.jobs()[0].output!.verify).toMatchObject({ ok: false, missingKeys: 1, extraKeys: 1, recipeInHeader: true })

    const g = setup()
    files.delete(A_PATH)
    await g.runner.start(recipe)
    g.outputs = ['merged_00001_.safetensors']
    g.client.handlers!.onSuccess!('p1')
    await waitFor(() => g.runner.jobs()[0].status === 'done')
    expect(g.runner.jobs()[0].output!.verify!.ok).toBe(true)
    expect(g.runner.jobs()[0].output!.verify!.message).toMatch(/could not be read/)
    expect(g.client.queued[0]).toMatchObject({ extra: { extra_pnginfo: { modelmerger_recipe: { sourceHeaderSha256: { a: null } } } } })
  })

  it('goes to error when no new file appears', async () => {
    const f = setup()
    await f.runner.start(recipe)
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'error')
    expect(f.runner.jobs()[0].error).toMatch(/could not be found or is not this merge/)
  })

  it('turns an error event into status error and closes the socket', async () => {
    const f = setup()
    await f.runner.start(recipe)
    f.client.handlers!.onError!({ promptId: 'p1', message: 'Out of memory', nodeId: '2' })
    const j = f.runner.jobs()[0]
    expect(j.status).toBe('error')
    expect(j.error).toBe('Out of memory')
    expect(f.client.closed).toBeGreaterThan(0)
    // a later success event changes nothing
    f.client.handlers!.onSuccess!('p1')
    expect(f.runner.jobs()[0].status).toBe('error')
  })

  it('falls back to polling history when websocket events are missed', async () => {
    const f = setup({ pollMs: 10 })
    await f.runner.start(recipe)
    f.client.queueState = { running: ['p1'], pending: [] }
    f.client.historyResult = { promptId: 'p1', completed: false, statusStr: 'success', images: [], error: null }
    files.set(OUT_PATH, outHdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
    f.outputs = ['merged_00001_.safetensors']
    await new Promise((r) => setTimeout(r, 30))
    expect(f.runner.jobs()[0].status).toBe('queued')
    f.client.historyResult = { promptId: 'p1', completed: true, statusStr: 'success', images: [], error: null }
    await waitFor(() => f.runner.jobs()[0].status === 'done')
  })

  it('history error finishes the job as error', async () => {
    const f = setup({ pollMs: 10 })
    await f.runner.start(recipe)
    f.client.historyResult = { promptId: 'p1', completed: true, statusStr: 'error', images: [], error: 'ModelSave: disk full' }
    await waitFor(() => f.runner.jobs()[0].status === 'error')
    expect(f.runner.jobs()[0].error).toBe('ModelSave: disk full')
  })
})

describe('start: safety', () => {
  it('refuses when the compatibility check finds problems', async () => {
    checkProblems.mockReturnValue(['A and B have different architectures.', 'B is quantized.'])
    const f = setup()
    await expect(f.runner.start(recipe)).rejects.toThrow('A and B have different architectures. B is quantized.')
    expect(f.client.queued).toHaveLength(0)
    expect(f.client.openEvents).not.toHaveBeenCalled()
  })

  it('stores the job id in the recipe record', async () => {
    const f = setup()
    await f.runner.start(recipe)
    const q = f.client.queued[0] as { extra: { extra_pnginfo: { modelmerger_recipe: { jobId: string } } } }
    expect(q.extra.extra_pnginfo.modelmerger_recipe.jobId).toBe('job-1')
  })

  it("ignores another job's file and takes the newest one that carries this job's id", async () => {
    const f = setup()
    await f.runner.start(recipe)
    const dir = `UNC:${BASE}/output/diffusion_models/`
    files.set(`${dir}merged_00003_.safetensors`, hdr(['diffusion_model.x.weight']))
    files.set(`${dir}merged_00002_.safetensors`, outHdr(['diffusion_model.x.weight'], 'someone-else'))
    files.set(`${dir}merged_00001_.safetensors`, outHdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
    f.outputs = ['merged_00001_.safetensors', 'merged_00002_.safetensors', 'merged_00003_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    expect(f.runner.jobs()[0].output!.relPath).toBe('diffusion_models/merged_00001_.safetensors')
  })

  it("errors when no new file is this merge's, writes no sidecar and allows no move", async () => {
    const f = setup()
    await f.runner.start(recipe)
    files.set(`UNC:${BASE}/output/diffusion_models/merged_00002_.safetensors`, outHdr(['diffusion_model.x.weight'], 'someone-else'))
    f.outputs = ['merged_00002_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'error')
    expect(f.runner.jobs()[0].error).toBe('ComfyUI finished, but the saved file could not be found or is not this merge\u2019s file.')
    expect(f.runner.jobs()[0].output).toBeNull()
    expect(f.sidecar).toBeNull()
    await expect(f.runner.moveToModels('job-1')).rejects.toThrow(/not finished/)
  })

  it('never overwrites an existing sidecar and says so', async () => {
    const f = setup()
    await f.runner.start(recipe)
    files.set(OUT_PATH, outHdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
    f.outputs = ['merged_00001_.safetensors']
    // Replace the writer with one that behaves like flag wx on an existing file.
    ;(f.runner as unknown as { d: { writeText: (p: string, t: string) => Promise<void> } }).d.writeText = async () => {
      throw Object.assign(new Error('exists'), { code: 'EEXIST' })
    }
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    const out = f.runner.jobs()[0].output!
    expect(out.sidecarPath).toBeNull()
    expect(out.verify!.message).toMatch(/already exists; it was left as it is/)
  })

  it('creates the real sidecar exclusively (flag wx)', async () => {
    const { mkdtempSync, writeFileSync, readFileSync, existsSync } = await import('node:fs')
    const { tmpdir } = await import('node:os')
    const { join } = await import('node:path')
    const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
    const f = setup()
    const deps = (f.runner as unknown as { d: { writeText?: unknown; toUnc: (d: string, p: string) => string } }).d
    deps.writeText = undefined
    deps.toUnc = (_d, p) => join(dir, p.split('/').pop()!)
    writeFileSync(join(dir, 'merged_00001_.safetensors.recipe.json'), 'KEEP')
    await f.runner.start(recipe)
    files.set(join(dir, 'merged_00001_.safetensors'), outHdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
    f.outputs = ['merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    expect(readFileSync(join(dir, 'merged_00001_.safetensors.recipe.json'), 'utf8')).toBe('KEEP')
    expect(f.runner.jobs()[0].output!.sidecarPath).toBeNull()
    expect(existsSync(join(dir, 'x'))).toBe(false)
  })
})

describe('start: events and polling', () => {
  it('replays events that arrive before the prompt id is known', async () => {
    const f = setup()
    let release!: () => void
    ;(f.client.queuePrompt as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      // The server runs the prompt and emits events before the HTTP answer reaches us.
      f.client.handlers!.onExecuting!({ promptId: 'p1', node: '2' })
      f.client.handlers!.onProgress!({ promptId: 'p1', node: '2', value: 4, max: 8 })
      f.client.handlers!.onProgress!({ promptId: 'zzz', node: '2', value: 9, max: 9 })
      await new Promise<void>((r) => (release = r))
      return { promptId: 'p1', number: 1 }
    })
    const p = f.runner.start(recipe)
    await waitFor(() => !!release)
    release()
    await p
    const j = f.runner.jobs()[0]
    expect(j.status).toBe('running')
    expect(j.progress).toEqual({ node: 'ModelMergeSimple', value: 4, max: 8 })
  })

  it('replays an early success event too', async () => {
    const f = setup({ conn: { own: false }, own: false })
    ;(f.client.queuePrompt as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      f.client.handlers!.onSuccess!('p1')
      return { promptId: 'p1', number: 1 }
    })
    await f.runner.start(recipe)
    await waitFor(() => f.runner.jobs()[0].status === 'done')
  })

  it('errors after three polls where history and queue do not know the prompt', async () => {
    const f = setup({ pollMs: 10 })
    await f.runner.start(recipe)
    f.client.queueState = { running: [], pending: [] }
    await waitFor(() => f.runner.jobs()[0].status === 'error')
    expect(f.runner.jobs()[0].error).toMatch(/no longer knows this merge/)
    expect(f.client.closed).toBeGreaterThan(0)
  })

  it('does not error while the queue still lists the prompt', async () => {
    const f = setup({ pollMs: 10 })
    await f.runner.start(recipe)
    f.client.queueState = { running: [], pending: ['p1'] }
    await new Promise((r) => setTimeout(r, 80))
    expect(f.runner.jobs()[0].status).toBe('queued')
  })
})

describe('verify: buffers', () => {
  it('does not fail on buffers ComfyUI does not save, but names them', async () => {
    const f = setup()
    files.set(A_PATH, hdr(['diffusion_model.l.weight', 'diffusion_model.l.bias', 'diffusion_model.rope.bands']))
    await f.runner.start(recipe)
    files.set(OUT_PATH, outHdr(['diffusion_model.l.weight', 'diffusion_model.l.bias']))
    f.outputs = ['merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    const v = f.runner.jobs()[0].output!.verify!
    expect(v).toMatchObject({ ok: true, missingKeys: 0, extraKeys: 0 })
    expect(v.message).toBe('File checked: 2 layers, recipe stored. 1 buffer not saved (diffusion_model.rope.bands); ComfyUI rebuilds it when loading.')
  })
  it('still fails when a weight, bias or scale is missing', async () => {
    const f = setup()
    files.set(A_PATH, hdr(['diffusion_model.l.weight', 'diffusion_model.l.weight_scale', 'diffusion_model.rope.bands']))
    await f.runner.start(recipe)
    files.set(OUT_PATH, outHdr(['diffusion_model.l.weight']))
    f.outputs = ['merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    expect(f.runner.jobs()[0].output!.verify).toMatchObject({ ok: false, missingKeys: 1 })
  })
})

describe('start: URL server', () => {
  it('finishes with only the prefix known and no verify', async () => {
    const f = setup({ conn: { own: false, sharedQueueAccepted: true }, own: false })
    await f.runner.start(recipe)
    expect(f.client.queued[0]).toMatchObject({ extra: { extra_pnginfo: { modelmerger_recipe: { sourceHeaderSha256: { a: null, b: null, c: null, loras: [] } } } } })
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    expect(f.runner.jobs()[0].output).toEqual({
      relPath: 'diffusion_models/merged_*.safetensors',
      wslPath: null,
      sizeBytes: null,
      sidecarPath: null,
      verify: null,
      movedTo: null
    })
    expect(f.sidecar).toBeNull()
  })
})

describe('cancel', () => {
  it('deletes the prompt by id and interrupts by id, whatever the queue looked like', async () => {
    const f = setup()
    await f.runner.start(recipe)
    f.client.queueState = { running: ['other'], pending: ['p1'] }
    const j = await f.runner.cancel('job-1')
    expect(j.status).toBe('cancelled')
    expect(f.client.deleted).toEqual([['p1']])
    expect(f.client.interrupted).toEqual(['p1'])
    expect(f.client.closed).toBeGreaterThan(0)
  })
  it('stops a running prompt by its id only', async () => {
    const f = setup()
    await f.runner.start(recipe)
    f.client.queueState = { running: ['p1'], pending: [] }
    await f.runner.cancel('job-1')
    expect(f.client.interrupted).toEqual(['p1'])
    expect(f.client.deleted).toEqual([['p1']])
    // a late interrupted event does not turn it into an error
    f.client.handlers?.onError?.({ promptId: 'p1', message: 'Interrupted', nodeId: null })
    expect(f.runner.jobs()[0].status).toBe('cancelled')
  })
  it('is a no-op for a finished job and rejects unknown ids', async () => {
    const f = setup()
    await f.runner.start(recipe)
    f.client.handlers!.onError!({ promptId: 'p1', message: 'x', nodeId: null })
    await f.runner.cancel('job-1')
    expect(f.client.deleted).toEqual([])
    expect(f.client.interrupted).toEqual([])
    await expect(f.runner.cancel('nope')).rejects.toThrow(/not known/)
  })
})

describe('moveToModels', () => {
  async function finished(f: Fake): Promise<void> {
    await f.runner.start(recipe)
    files.set(OUT_PATH, outHdr(['diffusion_model.x.weight', 'diffusion_model.y.weight']))
    f.outputs = ['merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
  }
  it('moves into <models>/<kind>/merged without overwriting and records the new paths', async () => {
    const f = setup()
    const scripts: string[] = []
    const orig = f.wsl.run
    f.wsl.run = async (d, s, o) => {
      scripts.push(s)
      return orig(d, s, o)
    }
    f.moveReply = {
      stdout: `RULZ_SIDECAR=${MODELS}/diffusion_models/merged/merged_00001_-2.safetensors.recipe.json\nRULZ_MOVED=${MODELS}/diffusion_models/merged/merged_00001_-2.safetensors\n`
    }
    await finished(f)
    const j = await f.runner.moveToModels('job-1')
    const mv = scripts.find((s) => s.includes('mv -n'))!
    expect(mv).toContain(`DIR='${MODELS}/diffusion_models/merged'`)
    expect(mv).toContain(`SRC='${BASE}/output/diffusion_models/merged_00001_.safetensors'`)
    expect(j.output!.movedTo).toBe(`${MODELS}/diffusion_models/merged/merged_00001_-2.safetensors`)
    expect(j.output!.sidecarPath).toBe(`${MODELS}/diffusion_models/merged/merged_00001_-2.safetensors.recipe.json`)
    await expect(f.runner.moveToModels('job-1')).rejects.toThrow(/already moved/)
  })
  it('refuses before the merge is done and for a URL server', async () => {
    const f = setup()
    await f.runner.start(recipe)
    await expect(f.runner.moveToModels('job-1')).rejects.toThrow(/not finished/)
    const g = setup({ conn: { own: false }, own: false })
    await g.runner.start(recipe)
    g.client.handlers!.onSuccess!('p1')
    await waitFor(() => g.runner.jobs()[0].status === 'done')
    await expect(g.runner.moveToModels('job-1')).rejects.toThrow(/own server/)
  })
  it('surfaces a failed move', async () => {
    const f = setup()
    f.moveReply = { code: 3, stdout: 'RULZ_ERR=The file is no longer there.\n' }
    await finished(f)
    await expect(f.runner.moveToModels('job-1')).rejects.toThrow('no longer there')
    expect(f.runner.jobs()[0].output!.movedTo).toBeNull()
  })
})

describe('inspect', () => {
  it('reads headers of a and b on the own server and reports free RAM', async () => {
    const f = setup()
    const res = (await f.runner.inspect({ kind: 'diffusion', mode: 'weighted', a: 'a.safetensors', b: 'b.safetensors', c: null, loras: [] })) as unknown as {
      input: { a: { info: { readable: boolean } }; b: { info: { readable: boolean } }; c: unknown; ramFreeBytes: number | null }
    }
    expect(res.input.a.info.readable).toBe(true)
    expect(res.input.b.info.readable).toBe(true)
    expect(res.input.c).toBeNull()
    expect(res.input.ramFreeBytes).toBe(1024 * 1024)
  })
  it('turns an unreadable file into an error entry', async () => {
    const f = setup()
    const res = (await f.runner.inspect({ kind: 'diffusion', mode: 'addDifference', a: 'a.safetensors', b: 'missing.safetensors', c: '../evil', loras: [] })) as unknown as {
      input: { b: { info: { readable: boolean; error: string } }; c: { info: { error: string } } }
    }
    expect(res.input.b.info.readable).toBe(false)
    expect(res.input.b.info.error).toMatch(/no such file/)
    expect(res.input.c.info.error).toMatch(/not a name inside/)
  })
  it('does not read files on a URL server', async () => {
    const f = setup({ conn: { own: false }, own: false })
    const res = (await f.runner.inspect({ kind: 'checkpoint', mode: 'weighted', a: 'a.safetensors', b: 'b.safetensors', c: null, loras: [] })) as unknown as {
      input: { a: { info: { error: string } }; ramFreeBytes: null }
    }
    expect(res.input.a.info.error).toBe(URL_MODE_MESSAGE)
    expect(res.input.ramFreeBytes).toBeNull()
  })
  it('caches object_info per URL', async () => {
    const f = setup()
    const req = { kind: 'diffusion' as const, mode: 'weighted' as const, a: 'a.safetensors', b: 'b.safetensors', c: null, loras: [] }
    await f.runner.inspect(req)
    await f.runner.inspect(req)
    expect(f.client.objectInfo).toHaveBeenCalledTimes(1)
  })
})

describe('parseLoraNotLoaded', () => {
  it('finds both line forms and dedupes', () => {
    const text = [
      'got prompt',
      'lora key not loaded: diffusion_model.a.lora_up.weight',
      'NOT LOADED lora_unet_b.alpha',
      'lora key not loaded: diffusion_model.a.lora_up.weight',
      'unrelated line'
    ].join('\n')
    expect(parseLoraNotLoaded(text)).toEqual(['diffusion_model.a.lora_up.weight', 'lora_unet_b.alpha'])
  })
  it('handles Windows line endings and empty text', () => {
    expect(parseLoraNotLoaded('x\r\nlora key not loaded: k1\r\nNOT LOADED k2\r\n')).toEqual(['k1', 'k2'])
    expect(parseLoraNotLoaded('')).toEqual([])
  })
})

describe('LoRAs', () => {
  const lora = (name: string): MergeRecipe['loras'][number] => ({ name, strength: 1, clipStrength: 1 })
  const L1 = `UNC:${MODELS}/loras/one.safetensors`
  const loraRecipe = (over: Partial<MergeRecipe> = {}): MergeRecipe => ({ ...recipe, loras: [lora('one.safetensors'), lora('two.safetensors')], ...over })
  const finish = async (f: Fake, keys = ['diffusion_model.x.weight', 'diffusion_model.y.weight']): Promise<MergeJob> => {
    files.set(OUT_PATH, outHdr(keys))
    f.outputs = ['merged_00001_.safetensors']
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => ['done', 'error'].includes(f.runner.jobs()[0].status))
    return f.runner.jobs()[0]
  }
  beforeEach(() => {
    files.set(L1, hdr(['k1', 'k2', 'k3']))
  })

  it('inspect in bake mode does not read B and reads the LoRA headers with A', async () => {
    const f = setup()
    const res = (await f.runner.inspect({ kind: 'diffusion', mode: 'bake', a: 'a.safetensors', b: 'missing.safetensors', c: null, loras: ['one.safetensors', 'two.safetensors'] })) as unknown as {
      input: { a: { info: { readable: boolean } }; b: unknown; c: unknown; loras: Array<{ name: string; readable: boolean; error: string | null; aHeader: unknown }> }
    }
    expect(res.input.a.info.readable).toBe(true)
    expect(res.input.b).toBeNull()
    expect(res.input.c).toBeNull()
    expect(res.input.loras.map((l) => [l.name, l.readable])).toEqual([['one.safetensors', true], ['two.safetensors', false]])
    expect(res.input.loras[1].error).toMatch(/no such file/)
    expect(res.input.loras[0].aHeader).toMatchObject({ headerSha256: 'sha-2' })
  })
  it('turns a LoRA outside the models folder into an error entry', async () => {
    const f = setup()
    const res = (await f.runner.inspect({ kind: 'diffusion', mode: 'bake', a: 'a.safetensors', b: '', c: null, loras: ['../evil'] })) as unknown as {
      input: { loras: Array<{ readable: boolean; error: string }> }
    }
    expect(res.input.loras[0].readable).toBe(false)
    expect(res.input.loras[0].error).toMatch(/not a name inside/)
  })
  it('inspect on a URL server gives unreadable LoRA infos with the URL message', async () => {
    const f = setup({ conn: { own: false }, own: false })
    const res = (await f.runner.inspect({ kind: 'diffusion', mode: 'weighted', a: 'a.safetensors', b: 'b.safetensors', c: null, loras: ['one.safetensors'] })) as unknown as {
      input: { loras: Array<{ readable: boolean; error: string }> }
    }
    expect(res.input.loras).toMatchObject([{ readable: false, error: URL_MODE_MESSAGE }])
  })

  it('start stores the LoRA header hashes in recipe order (null when unreadable) and skips B in bake mode', async () => {
    const f = setup()
    files.delete(`UNC:${MODELS}/diffusion_models/b.safetensors`)
    await f.runner.start(loraRecipe({ mode: 'bake', b: '' }))
    const q = f.client.queued[0] as { extra: { extra_pnginfo: { modelmerger_recipe: { sourceHeaderSha256: unknown } } } }
    expect(q.extra.extra_pnginfo.modelmerger_recipe.sourceHeaderSha256).toEqual({ a: 'sha-2', b: null, c: null, loras: ['sha-3', null] })
  })
  it('start refuses on check problems that include LoRAs', async () => {
    checkProblems.mockReturnValue(['LoRA one is for another model.'])
    const f = setup()
    await expect(f.runner.start(loraRecipe())).rejects.toThrow('LoRA one is for another model.')
    expect(f.client.queued).toHaveLength(0)
    expect(f.logListeners.size).toBe(0)
  })

  it('collects not-loaded keys across split chunks and separates text-encoder keys for diffusion', async () => {
    const f = setup()
    await f.runner.start(loraRecipe())
    expect(f.logListeners.size).toBe(1)
    f.emitLog('lora key not loaded: diffusion_model.q.lora_up.we')
    f.emitLog('ight\r\nlora key not loaded: lora_te1_x.lora_down.weight\nNOT LOADED diffusion_model.r.alpha\nlora key not loaded: diffusion_model.q.lora_up.weight\n')
    f.emitLog('NOT LOADED text_encoder.y.weight')
    const job = await finish(f)
    expect(job.status).toBe('done')
    expect(job.output!.verify).toMatchObject({
      ok: false,
      loraKeysNotLoaded: 2,
      loraKeysNotLoadedSample: ['diffusion_model.q.lora_up.weight', 'diffusion_model.r.alpha'],
      loraTextEncoderKeysSkipped: 2
    })
    expect(job.output!.verify!.message).toContain(' 2 LoRA weights were not applied (for example diffusion_model.q.lora_up.weight, diffusion_model.r.alpha); the LoRA may be for another model.')
    expect(job.output!.verify!.message).toContain(' 2 text-encoder weights left out (a diffusion-model file has no text encoder).')
    expect(f.logListeners.size).toBe(0)
  })
  it('counts text-encoder keys as not loaded for a checkpoint', async () => {
    const f = setup()
    files.set(`UNC:${MODELS}/checkpoints/a.safetensors`, hdr(['model.diffusion_model.x.weight']))
    files.set(`UNC:${MODELS}/checkpoints/b.safetensors`, hdr(['model.diffusion_model.x.weight']))
    await f.runner.start(loraRecipe({ kind: 'checkpoint' }))
    f.emitLog('lora key not loaded: lora_te1_x.lora_down.weight\n')
    const job = await finish(f, ['model.diffusion_model.x.weight'])
    expect(job.output!.verify).toMatchObject({ loraKeysNotLoaded: 1, loraKeysNotLoadedSample: ['lora_te1_x.lora_down.weight'], loraTextEncoderKeysSkipped: 0, ok: false })
  })
  it('says all weights were applied when nothing was skipped', async () => {
    const f = setup()
    await f.runner.start(loraRecipe())
    f.emitLog('some other line\n')
    const job = await finish(f)
    expect(job.output!.verify).toMatchObject({ ok: true, loraKeysNotLoaded: 0, loraKeysNotLoadedSample: [], loraTextEncoderKeysSkipped: 0 })
    expect(job.output!.verify!.message).toMatch(/ LoRAs applied: all weights\.$/)
  })
  it('leaves the LoRA fields unknown without LoRAs', async () => {
    const f = setup()
    await f.runner.start(recipe)
    expect(f.logListeners.size).toBe(0)
    const job = await finish(f)
    expect(job.output!.verify).toMatchObject({ loraKeysNotLoaded: null, loraKeysNotLoadedSample: [], loraTextEncoderKeysSkipped: 0 })
  })
  it('does not collect on a URL server', async () => {
    const f = setup({ conn: { own: false }, own: false })
    await f.runner.start(loraRecipe())
    expect(f.logListeners.size).toBe(0)
    f.client.handlers!.onSuccess!('p1')
    await waitFor(() => f.runner.jobs()[0].status === 'done')
    expect(f.runner.jobs()[0].output!.verify).toBeNull()
  })

  it('stops listening after an error, after cancel and when the prompt is lost', async () => {
    const f = setup()
    await f.runner.start(loraRecipe())
    expect(f.logListeners.size).toBe(1)
    f.client.handlers!.onError!({ promptId: 'p1', message: 'boom', nodeId: null })
    expect(f.logListeners.size).toBe(0)

    const g = setup()
    await g.runner.start(loraRecipe())
    expect(g.logListeners.size).toBe(1)
    await g.runner.cancel('job-1')
    expect(g.logListeners.size).toBe(0)

    const h = setup({ pollMs: 10 })
    await h.runner.start(loraRecipe())
    h.client.queueState = { running: [], pending: [] }
    await waitFor(() => h.runner.jobs()[0].status === 'error')
    expect(h.logListeners.size).toBe(0)
  })
  it('stops listening when queueing fails', async () => {
    const f = setup()
    ;(f.client.queuePrompt as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('refused'))
    await expect(f.runner.start(loraRecipe())).rejects.toThrow('refused')
    expect(f.logListeners.size).toBe(0)
    expect(f.client.closed).toBeGreaterThan(0)
  })
})
