import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LORA_MERGE_RECIPE, type LoraMergeJob, type LoraMergeRecipe } from '@shared/loraMerge'
import { DEFAULT_SETTINGS } from '@shared/types'
import { LoraMergeService, type WorkerHandle } from '../src/main/loras/service'
import type { LoraWorkerMessage, LoraWorkerRequest } from '../src/main/loras/workerTypes'
import type { WslRunner } from '../src/main/server/wsl'

const problems = vi.fn<() => string[]>(() => [])
vi.mock('../src/main/loras/merge', () => ({
  describeMergeInput: (name: string, header: { headerSha256: string } | null, error: string | null) => ({
    name,
    readable: !!header,
    error,
    sizeBytes: header ? 10 : null
  }),
  checkLoraMerge: (infos: unknown[]) => ({ inputs: infos, problems: problems(), warnings: [], totalModules: 1, sharedModules: 1, outputRank: null, estimatedBytes: null })
}))

const MODELS = '/home/jeff/ComfyUI/models'
const STAGING = '/home/jeff/.rulz/comfy/lora-merge'
const unc = (_d: string, p: string): string => `UNC:${p}`

const recipe = (over: Partial<LoraMergeRecipe> = {}): LoraMergeRecipe => ({
  ...DEFAULT_LORA_MERGE_RECIPE,
  inputs: [
    { name: 'a.safetensors', weight: 1 },
    { name: 'sub/b.safetensors', weight: 0.5 }
  ],
  outputName: 'my merge',
  ...over
})

interface Fake {
  svc: LoraMergeService
  scripts: string[]
  events: LoraMergeJob[]
  written: Array<{ path: string; text: string }>
  unlinked: string[]
  rmdirs: string[]
  mkdirs: string[]
  clock: { t: number }
  worker: { req: LoraWorkerRequest | null; onMessage: (m: LoraWorkerMessage) => void; finish: (m: LoraWorkerMessage) => void; terminated: number }
  moveStdout: string
}

const doneMsg = (over: Partial<Extract<LoraWorkerMessage, { type: 'done' }>> = {}): LoraWorkerMessage => ({
  type: 'done',
  sizeBytes: 1234,
  modules: 7,
  rank: { min: 8, max: 16 },
  energyKept: null,
  verify: { ok: true, message: 'ok' },
  metadata: {},
  ...over
})

function setup(): Fake {
  const f = {} as Fake
  f.scripts = []
  f.events = []
  f.written = []
  f.unlinked = []
  f.rmdirs = []
  f.mkdirs = []
  f.clock = { t: 1000 }
  f.moveStdout = `RULZ_SIDECAR=${MODELS}/loras/merged/my merge.safetensors.recipe.json\nRULZ_MOVED=${MODELS}/loras/merged/my merge.safetensors\n`
  f.worker = { req: null, onMessage: () => undefined, finish: () => undefined, terminated: 0 }
  const wsl: WslRunner = {
    run: async (_d, script) => {
      f.scripts.push(script)
      if (script.includes('RULZ_STAGING')) return { code: 0, stdout: `RULZ_STAGING=${STAGING}\n`, stderr: '' }
      if (script.includes('mv -n')) return { code: 0, stdout: f.moveStdout, stderr: '' }
      if (script.includes('RULZ_LORA')) return { code: 0, stdout: 'RULZ_LORA|a.safetensors\n', stderr: '' }
      return { code: 0, stdout: `${MODELS}\n`, stderr: '' }
    },
    spawn: () => {
      throw new Error('no spawn')
    }
  }
  f.svc = new LoraMergeService({
    wsl,
    settings: () => DEFAULT_SETTINGS.ownServer,
    appVersion: '9.9.9',
    toUnc: unc,
    now: () => f.clock.t,
    newId: () => 'job1',
    progressMs: 200,
    readHeader: async (p) => {
      if (p.includes('broken')) throw new Error('cannot read it')
      return { tensors: {}, metadata: {}, headerBytes: 1, fileBytes: 10, headerSha256: `sha:${p}` }
    },
    writeText: async (path, text) => void f.written.push({ path, text }),
    unlinkFile: async (p) => void f.unlinked.push(p),
    removeDir: async (p) => void f.rmdirs.push(p),
    makeDir: async (p) => void f.mkdirs.push(p),
    runWorker: (req, onMessage) => {
      f.worker.req = req
      f.worker.onMessage = onMessage
      let resolve!: (m: LoraWorkerMessage) => void
      const done = new Promise<LoraWorkerMessage>((r) => (resolve = r))
      f.worker.finish = resolve
      const h: WorkerHandle = {
        terminate: () => {
          f.worker.terminated++
          resolve({ type: 'error', message: 'The merge was stopped.' })
        },
        done
      }
      return h
    }
  })
  f.svc.on('job', (j: LoraMergeJob) => f.events.push(j))
  return f
}

const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0))
async function settle(): Promise<void> {
  for (let i = 0; i < 10; i++) await tick()
}

beforeEach(() => problems.mockReset().mockReturnValue([]))

describe('start validation', () => {
  it.each([
    [{ inputs: [{ name: 'a', weight: 1 }] }, 'at least two'],
    [{ inputs: Array.from({ length: 11 }, (_, i) => ({ name: `n${i}`, weight: 1 })) }, 'At most 10'],
    [{ inputs: [{ name: ' ', weight: 1 }, { name: 'b', weight: 1 }] }, 'file name'],
    [{ inputs: [{ name: 'a', weight: 11 }, { name: 'b', weight: 1 }] }, 'from -10 to 10'],
    [{ inputs: [{ name: 'a', weight: NaN }, { name: 'b', weight: 1 }] }, 'from -10 to 10'],
    [{ rank: 0 }, 'rank'],
    [{ rank: 2000 }, 'rank'],
    [{ outputName: '   ' }, 'name'],
    [{ outputName: '...' }, 'usable']
  ])('refuses %j', async (over, msg) => {
    const f = setup()
    await expect(f.svc.start(recipe(over as Partial<LoraMergeRecipe>))).rejects.toThrow(msg)
    expect(f.worker.req).toBeNull()
    // The failed start did not block the next one.
    const j = await f.svc.start(recipe())
    expect(j.status).toBe('running')
  })

  it('refuses when the check has problems', async () => {
    const f = setup()
    problems.mockReturnValue(['LoKr is not supported.', 'Second problem.'])
    await expect(f.svc.start(recipe())).rejects.toThrow('LoKr is not supported. Second problem.')
  })

  it('refuses a second merge while one runs', async () => {
    const f = setup()
    await f.svc.start(recipe())
    await expect(f.svc.start(recipe())).rejects.toThrow('A LoRA merge is already running.')
  })

})

describe('inspect', () => {
  it('turns unreadable files and ".." names into error infos', async () => {
    const f = setup()
    const c = await f.svc.inspect({ names: ['ok.safetensors', 'broken.safetensors', '../etc/passwd', '/abs.safetensors'], mode: 'concat', rank: 64, dtype: 'source' })
    const infos = c.inputs as unknown as Array<{ name: string; readable: boolean; error: string | null }>
    expect(infos.map((i) => i.readable)).toEqual([true, false, false, false])
    expect(infos[1].error).toBe('cannot read it')
    expect(infos[2].error).toContain('not a name inside the models folder')
    expect(infos[3].error).toContain('not a name inside the models folder')
  })
})

describe('list', () => {
  it('lists LoRAs of the resolved models folder', async () => {
    const f = setup()
    expect(await f.svc.list()).toEqual(['a.safetensors'])
    expect(f.scripts[1]).toContain(`'${MODELS}/loras'`)
  })
})

describe('a merge', () => {
  it('runs, throttles progress, saves, moves and reports the output', async () => {
    const f = setup()
    const started = await f.svc.start(recipe())
    expect(started.status).toBe('running')
    expect(f.mkdirs).toEqual([`UNC:${STAGING}/job1`])
    await settle()
    const req = f.worker.req!
    expect(req.outUncPath).toBe(`UNC:${STAGING}/job1/my merge.safetensors`)
    expect(req.inputs).toEqual([
      { name: 'a.safetensors', uncPath: `UNC:${MODELS}/loras/a.safetensors`, weight: 1 },
      { name: 'sub/b.safetensors', uncPath: `UNC:${MODELS}/loras/sub/b.safetensors`, weight: 0.5 }
    ])
    expect(req.appVersion).toBe('9.9.9')
    expect(req.inputHeaderSha256).toEqual([`sha:UNC:${MODELS}/loras/a.safetensors`, `sha:UNC:${MODELS}/loras/sub/b.safetensors`])

    const before = f.events.length
    f.worker.onMessage({ type: 'progress', done: 1, total: 10 }) // t=1000, first: emitted
    f.clock.t += 50
    f.worker.onMessage({ type: 'progress', done: 2, total: 10 }) // throttled
    f.clock.t += 50
    f.worker.onMessage({ type: 'progress', done: 3, total: 10 }) // throttled
    f.clock.t += 300
    f.worker.onMessage({ type: 'progress', done: 4, total: 10 }) // emitted
    f.worker.onMessage({ type: 'progress', done: 10, total: 10 }) // last: always emitted
    expect(f.events.length - before).toBe(3)
    expect(f.events[f.events.length - 1].progress).toEqual({ done: 10, total: 10 })

    f.worker.finish(doneMsg({ energyKept: { mean: 0.9, min: 0.8 } }))
    await settle()

    const job = f.svc.jobs()[0]
    expect(job.status).toBe('done')
    expect(job.error).toBeNull()
    expect(job.output).toEqual({
      wslPath: `${MODELS}/loras/merged/my merge.safetensors`,
      loraName: 'merged/my merge.safetensors',
      sizeBytes: 1234,
      modules: 7,
      rank: { min: 8, max: 16 },
      energyKept: { mean: 0.9, min: 0.8 },
      verify: { ok: true, message: 'ok' },
      sidecarPath: `${MODELS}/loras/merged/my merge.safetensors.recipe.json`
    })
    expect(f.events.map((e) => e.status)).toContain('saving')

    // sidecar written next to the staged file, with the record
    expect(f.written).toHaveLength(1)
    expect(f.written[0].path).toBe(`UNC:${STAGING}/job1/my merge.safetensors.recipe.json`)
    const rec = JSON.parse(f.written[0].text)
    expect(rec).toMatchObject({ app: 'RulZ Model Merger', appVersion: '9.9.9', kind: 'lora-merge' })
    expect(rec.recipe.outputName).toBe('my merge')
    expect(rec.inputHeaderSha256).toHaveLength(2)

    // the move script: from the staging file into <models>/loras/merged
    const move = f.scripts.find((s) => s.includes('mv -n'))!
    expect(move).toContain(`SRC='${STAGING}/job1/my merge.safetensors'`)
    expect(move).toContain(`DIR='${MODELS}/loras/merged'`)
    // the then-empty job folder is removed with rmdir only
    expect(f.rmdirs).toEqual([`UNC:${STAGING}/job1`])
    expect(f.unlinked).toEqual([])
  })

  it('uses the metadata record for the sidecar when the worker sent one', async () => {
    const f = setup()
    await f.svc.start(recipe())
    await settle()
    const record = { app: 'RulZ Model Merger', appVersion: '1', kind: 'lora-merge', createdAt: 'x', recipe: recipe(), inputHeaderSha256: ['h1', 'h2'] }
    f.worker.finish(doneMsg({ metadata: { rulz_model_merger: JSON.stringify(record) } }))
    await settle()
    expect(JSON.parse(f.written[0].text)).toEqual(record)
  })

  it('sanitises the output name and starts the next merge after one ends', async () => {
    const f = setup()
    await f.svc.start(recipe({ outputName: 'a/b:c*' }))
    await settle()
    expect(f.worker.req!.outUncPath).toBe(`UNC:${STAGING}/job1/a_b_c_.safetensors`)
    f.worker.finish(doneMsg())
    await settle()
    await expect(f.svc.start(recipe())).resolves.toBeDefined()
  })

  it('keeps a file whose verify failed, status done, with the message', async () => {
    const f = setup()
    await f.svc.start(recipe())
    await settle()
    f.worker.finish(doneMsg({ verify: { ok: false, message: 'Layer x differs by 12%.' } }))
    await settle()
    const job = f.svc.jobs()[0]
    expect(job.status).toBe('done')
    expect(job.output?.verify).toEqual({ ok: false, message: 'Layer x differs by 12%.' })
    expect(f.scripts.some((s) => s.includes('mv -n'))).toBe(true)
  })

  it('a worker error becomes an error job and removes only its own files', async () => {
    const f = setup()
    await f.svc.start(recipe())
    await settle()
    f.worker.finish({ type: 'error', message: 'Out of memory.' })
    await settle()
    const job = f.svc.jobs()[0]
    expect(job.status).toBe('error')
    expect(job.error).toBe('Out of memory.')
    expect(f.unlinked).toEqual([`UNC:${STAGING}/job1/my merge.safetensors`, `UNC:${STAGING}/job1/my merge.safetensors.recipe.json`])
    expect(f.rmdirs).toEqual([`UNC:${STAGING}/job1`])
    expect(f.scripts.some((s) => s.includes('mv -n'))).toBe(false)
    // and a new merge may start
    await expect(f.svc.start(recipe())).resolves.toBeDefined()
  })

  it('a failed move is an error job', async () => {
    const f = setup()
    f.moveStdout = 'RULZ_ERR=The move failed.\n'
    await f.svc.start(recipe())
    await settle()
    f.worker.finish(doneMsg())
    await settle()
    const job = f.svc.jobs()[0]
    expect(job.status).toBe('error')
    expect(job.error).toBe('The move failed.')
  })

  it('cancel stops the worker and deletes only this job files', async () => {
    const f = setup()
    await f.svc.start(recipe())
    await settle()
    const job = await f.svc.cancel('job1')
    expect(job.status).toBe('cancelled')
    expect(f.worker.terminated).toBe(1)
    expect(f.unlinked).toEqual([`UNC:${STAGING}/job1/my merge.safetensors`, `UNC:${STAGING}/job1/my merge.safetensors.recipe.json`])
    expect(f.rmdirs.every((p) => p === `UNC:${STAGING}/job1`)).toBe(true)
    expect(f.scripts.some((s) => s.includes('mv -n'))).toBe(false)
    expect(f.svc.jobs()[0].status).toBe('cancelled')
    expect(f.events[f.events.length - 1].status).toBe('cancelled')
    await expect(f.svc.start(recipe())).resolves.toBeDefined()
  })

  it('cancel of an unknown id fails, of a finished job returns it', async () => {
    const f = setup()
    await expect(f.svc.cancel('nope')).rejects.toThrow('not known')
    await f.svc.start(recipe())
    await settle()
    f.worker.finish(doneMsg())
    await settle()
    expect((await f.svc.cancel('job1')).status).toBe('done')
  })

  it('jobs() is newest first', async () => {
    const f = setup()
    let n = 0
    // two jobs with different ids and times
    const svcIds = ['j1', 'j2']
    ;(f.svc as unknown as { d: { newId: () => string } }).d.newId = () => svcIds[n++]
    await f.svc.start(recipe())
    await settle()
    f.worker.finish(doneMsg())
    await settle()
    f.clock.t += 5000
    await f.svc.start(recipe())
    expect(f.svc.jobs().map((j) => j.id)).toEqual(['j2', 'j1'])
  })
})
