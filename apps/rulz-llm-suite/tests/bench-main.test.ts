import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MAX_BENCH_RUNS, type BenchAxis, type BenchmarkRequest, type BenchmarkRun } from '@shared/benchmark'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { GpuStat, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import { BenchmarkRunner, type BenchDeps } from '../src/main/bench/runner'
import { BenchStore } from '../src/main/bench/store'

const gpu = (index: number, used: number): GpuStat => ({ index, name: `GPU ${index}`, memUsedMiB: used, memTotalMiB: 24000, utilPct: 0 })

const model = (extra: Partial<ModelEntry> = {}): ModelEntry => ({
  id: 'm/model.gguf',
  name: 'model',
  folder: 'm',
  path: 'C:/models/m/model.gguf',
  files: ['C:/models/m/model.gguf'],
  sizeBytes: 1e9,
  info: { architecture: 'llama', hasChatTemplate: true },
  modifiedAt: 1,
  ...extra
})

const est = (verdict: MemoryEstimate['verdict'] = 'fits'): MemoryEstimate => ({
  weightsBytes: 1,
  kvBytes: 1,
  overheadBytes: 1,
  gpuBytes: 111,
  totalBytes: 222,
  contextLength: 4096,
  freeVramBytes: null,
  freeRamBytes: 1,
  verdict,
  perGpu: [],
  cpuExpertBytes: 5,
  adapterBytes: 0
})

const request = (axes: BenchAxis[] = [], extra: Partial<BenchmarkRequest> = {}): BenchmarkRequest => ({
  modelId: 'm/model.gguf',
  base: { ...DEFAULT_LOAD_CONFIG },
  axes,
  prompt: { promptTokens: 64, genTokens: 16, repeats: 1 },
  skipLikelyOom: false,
  label: 'test',
  ...extra
})

interface DoneChunk {
  ttft?: number
  promptTps?: number
  genTps?: number
  draftN?: number
  draftAccepted?: number
  noTimings?: boolean
  errorChunk?: string
}

const sse = (pieces: string[]): Response => {
  const enc = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const p of pieces) c.enqueue(enc.encode(p))
      c.close()
    }
  })
  return new Response(body, { status: 200 })
}

interface Harness {
  deps: BenchDeps
  runner: BenchmarkRunner
  loads: LoadConfig[]
  completions: Array<Record<string, unknown>>
  tokenizeBodies: Array<Record<string, unknown>>
  changes: BenchmarkRun[]
  saved: BenchmarkRun[]
  instances: {
    load: ReturnType<typeof vi.fn>
    unload: ReturnType<typeof vi.fn>
    unloadAll: ReturnType<typeof vi.fn>
    acquire: ReturnType<typeof vi.fn>
    baseUrl: ReturnType<typeof vi.fn>
  }
  released: number
  clock: { t: number }
}

function harness(
  opts: {
    models?: ModelEntry[]
    estimate?: (cfg: LoadConfig) => MemoryEstimate | null
    gpus?: (call: number) => GpuStat[]
    /** One entry per completion request, in order; the last one repeats. */
    done?: DoneChunk[]
    contextLength?: number | null
    failLoadAt?: number
    hangCompletion?: boolean
    onCompletion?: () => void
    draftFor?: BenchDeps['draftFor']
    tokens?: number
  } = {}
): Harness {
  const h = { loads: [] as LoadConfig[], completions: [] as Array<Record<string, unknown>>, tokenizeBodies: [] as Array<Record<string, unknown>>, changes: [] as BenchmarkRun[], saved: [] as BenchmarkRun[], released: 0, clock: { t: 1000 } } as Harness
  let loadN = 0
  let sampleN = 0
  let completionN = 0
  h.instances = {
    load: vi.fn(async (_id: string, cfg: LoadConfig | null) => {
      loadN++
      if (opts.failLoadAt === loadN) throw new Error('out of memory')
      h.loads.push(cfg as LoadConfig)
      return {
        instanceId: `inst-${loadN}`,
        modelId: 'm/model.gguf',
        modelName: 'model',
        port: 8080,
        pid: 1,
        status: 'ready',
        progress: 1,
        stage: 'Ready',
        config: cfg as LoadConfig,
        startedAt: 0,
        contextLength: opts.contextLength === undefined ? 8192 : opts.contextLength,
        hasVision: false,
        adapters: [],
        estimate: null
      } as LoadedInstance
    }),
    unload: vi.fn(async () => undefined),
    unloadAll: vi.fn(async () => undefined),
    acquire: vi.fn(() => () => {
      h.released++
    }),
    baseUrl: vi.fn(() => 'http://127.0.0.1:8080')
  }
  const fakeFetch = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const u = String(url)
    const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
    if (u.endsWith('/tokenize')) {
      h.tokenizeBodies.push(body)
      const n = opts.tokens ?? 5000
      return new Response(JSON.stringify({ tokens: Array.from({ length: n }, (_, i) => i + 1) }))
    }
    h.completions.push(body)
    opts.onCompletion?.()
    if (opts.hangCompletion) {
      return new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
      })
    }
    const list = opts.done ?? [{}]
    const d = list[Math.min(completionN++, list.length - 1)]
    h.clock.t += d.ttft ?? 20
    const timings = d.noTimings
      ? undefined
      : { prompt_per_second: d.promptTps ?? 500, predicted_per_second: d.genTps ?? 50, ...(d.draftN !== undefined ? { draft_n: d.draftN, draft_n_accepted: d.draftAccepted ?? 0 } : {}) }
    const first = JSON.stringify({ content: '', stop: false })
    const second = JSON.stringify({ content: 'Hi', stop: false })
    const last = JSON.stringify({ content: '', stop: true, ...(timings ? { timings } : {}) })
    if (d.errorChunk) return sse([`data: ${JSON.stringify({ error: d.errorChunk })}\n\n`])
    // The second line is split across two chunks to exercise the line buffer.
    const text = `data: ${first}\n\ndata: ${second}\n\ndata: ${last}\n\n`
    const cut = text.indexOf('Hi') + 1
    return sse([text.slice(0, cut), text.slice(cut)])
  })
  h.deps = {
    instances: h.instances as unknown as BenchDeps['instances'],
    getModel: (id) => (opts.models ?? [model()]).find((m) => m.id === id),
    estimate: (_m, cfg) => (opts.estimate ? opts.estimate(cfg) : est()),
    draftFor: opts.draftFor ?? (() => undefined),
    sampleGpus: async () => (opts.gpus ? opts.gpus(sampleN++) : []),
    engineVersion: async () => 'b9999',
    onChange: (run) => h.changes.push(run),
    store: { save: async (run) => void h.saved.push(run) },
    fetch: fakeFetch as unknown as typeof fetch,
    sleep: async () => undefined,
    now: () => h.clock.t,
    settleMs: 0,
    sampleIntervalMs: 60_000
  }
  h.runner = new BenchmarkRunner(h.deps)
  return h
}

const finish = (h: Harness): Promise<void> => vi.waitFor(() => expect(h.runner.isRunning()).toBe(false))
const last = (h: Harness): BenchmarkRun => h.changes[h.changes.length - 1]

describe('BenchmarkRunner', () => {
  it('runs the combinations in order and records medians', async () => {
    const h = harness({ done: [{}, { genTps: 10, promptTps: 100, ttft: 30 }, { genTps: 30, promptTps: 300, ttft: 10 }, { genTps: 20, promptTps: 200, ttft: 20 }] })
    const req = request(
      [
        { key: 'cacheType', values: ['f16', 'q8_0'] },
        { key: 'batchSize', values: [256, 512] }
      ],
      { prompt: { promptTokens: 64, genTokens: 16, repeats: 3 } }
    )
    const started = await h.runner.start(req)
    expect(started.status).toBe('running')
    expect(started.results).toHaveLength(4)
    expect(started.results.every((r) => r.status === 'pending')).toBe(true)
    expect(started.engineVersion).toBe('b9999')
    await finish(h)

    expect(h.loads.map((c) => [c.cacheTypeK, c.cacheTypeV, c.batchSize])).toEqual([
      ['f16', 'f16', 256],
      ['f16', 'f16', 512],
      ['q8_0', 'q8_0', 256],
      ['q8_0', 'q8_0', 512]
    ])
    const run = last(h)
    expect(run.status).toBe('done')
    expect(run.finishedAt).toBeTypeOf('number')
    expect(run.results.map((r) => r.status)).toEqual(['ok', 'ok', 'ok', 'ok'])
    // Combination 1: warm-up uses the first scripted answer; the timed ones are 10/30/20 tok/s (median 20).
    expect(run.results[0].genTps).toBe(20)
    expect(run.results[0].promptTps).toBe(200)
    expect(run.results[0].ttftMs).toBe(20)
    expect(run.results[0].estimate).toEqual({ gpuBytes: 111, totalBytes: 222, verdict: 'fits', cpuExpertBytes: 5 })
    expect(run.results[0].contextLength).toBe(8192)
    expect(run.results[0].loadMs).toBe(0)
    expect(run.results[0].draftAcceptance).toBeNull()
    expect(h.instances.unload.mock.calls.map((c) => c[0])).toEqual(['inst-1', 'inst-2', 'inst-3', 'inst-4'])
    expect(h.released).toBe(4)
    expect(h.instances.unloadAll).toHaveBeenCalledTimes(2)
    expect(h.saved[h.saved.length - 1].status).toBe('done')
    // Running states were broadcast along the way.
    expect(h.changes.some((r) => r.results.some((x) => x.status === 'running'))).toBe(true)
  })

  it('skips combinations the estimate calls tight when asked to', async () => {
    const h = harness({ estimate: (cfg) => est(cfg.cacheTypeK === 'f32' ? 'tight' : 'fits') })
    await h.runner.start(request([{ key: 'cacheType', values: ['f32', 'q8_0'] }], { skipLikelyOom: true }))
    await finish(h)
    const run = last(h)
    expect(run.results.map((r) => r.status)).toEqual(['skipped', 'ok'])
    expect(run.results[0].estimate?.verdict).toBe('tight')
    expect(h.loads).toHaveLength(1)
    expect(run.status).toBe('done')
  })

  it('tries tight combinations when skipping is off', async () => {
    const h = harness({ estimate: () => est('tight') })
    await h.runner.start(request([{ key: 'cacheType', values: ['f32'] }]))
    await finish(h)
    expect(last(h).results[0].status).toBe('ok')
  })

  it('works out peak VRAM from the baseline per GPU', async () => {
    // Per combination (repeats 1) the sampler is read: baseline, after load, after warm-up, after the timed request, at the end.
    const script: GpuStat[][] = [
      [gpu(0, 1000), gpu(1, 2000)],
      [gpu(0, 1000), gpu(1, 2000)],
      [gpu(0, 5000), gpu(1, 2500)],
      [gpu(0, 6000), gpu(1, 4000)],
      [gpu(0, 5500), gpu(1, 3000)],
      // second combination: the baseline is whatever is in use after the unload
      [gpu(0, 1200), gpu(1, 2100)],
      [gpu(0, 1100), gpu(1, 2000)]
    ]
    const h = harness({ gpus: (n) => script[Math.min(n, script.length - 1)] })
    await h.runner.start(request([{ key: 'cacheType', values: ['f16', 'q8_0'] }]))
    await finish(h)
    const [a, b] = last(h).results
    expect(a.perGpu).toEqual([
      { index: 0, name: 'GPU 0', peakMiB: 5000 },
      { index: 1, name: 'GPU 1', peakMiB: 2000 }
    ])
    expect(a.peakVramMiB).toBe(7000)
    // Never below zero: later readings under the baseline give 0.
    expect(b.perGpu?.map((g) => g.peakMiB)).toEqual([0, 0])
    expect(b.peakVramMiB).toBe(0)
  })

  it('records null VRAM when no GPU is reported', async () => {
    const h = harness({ gpus: () => [] })
    await h.runner.start(request())
    await finish(h)
    expect(last(h).results[0].peakVramMiB).toBeNull()
    expect(last(h).results[0].perGpu).toEqual([])
  })

  it('builds exactly promptTokens tokens and parses the stream (ttft, speeds, draft acceptance)', async () => {
    const h = harness({
      tokens: 400,
      done: [{}, { ttft: 40, promptTps: 800, genTps: 60, draftN: 10, draftAccepted: 7 }, { ttft: 60, promptTps: 900, genTps: 80, draftN: 30, draftAccepted: 25 }]
    })
    await h.runner.start(request([], { prompt: { promptTokens: 64, genTokens: 32, repeats: 2 } }))
    await finish(h)

    expect(h.tokenizeBodies).toHaveLength(1)
    expect(typeof h.tokenizeBodies[0].content).toBe('string')
    expect((h.tokenizeBodies[0].content as string).length).toBeGreaterThanOrEqual(64 * 6)
    expect(h.completions).toHaveLength(3)
    expect((h.completions[0].prompt as number[]).length).toBe(64)
    expect(h.completions[0].n_predict).toBe(8)
    for (const c of h.completions.slice(1)) {
      expect(c).toMatchObject({ n_predict: 32, cache_prompt: false, stream: true, temperature: 0, ignore_eos: true })
      expect(c.prompt).toEqual(Array.from({ length: 64 }, (_, i) => i + 1))
    }
    const r = last(h).results[0]
    expect(r.ttftMs).toBe(50)
    expect(r.promptTps).toBe(850)
    expect(r.genTps).toBe(70)
    expect(r.draftAcceptance).toBeCloseTo(32 / 40)
  })

  it('fails a combination whose request reports no timings, an error, or an HTTP error', async () => {
    for (const done of [[{ noTimings: true }], [{ errorChunk: 'boom' }]]) {
      const h = harness({ done })
      await h.runner.start(request())
      await finish(h)
      expect(last(h).results[0].status).toBe('failed')
      expect(last(h).results[0].error).toBeTruthy()
      expect(last(h).status).toBe('done')
    }
    const h = harness()
    h.deps.fetch = (async (url: string | URL | Request) =>
      String(url).endsWith('/tokenize') ? new Response(JSON.stringify({ tokens: [1, 2, 3] })) : new Response('bad', { status: 500 })) as unknown as typeof fetch
    h.runner = new BenchmarkRunner({ ...h.deps, fetch: h.deps.fetch })
    await h.runner.start(request([], { prompt: { promptTokens: 16, genTokens: 8, repeats: 1 } }))
    await finish(h)
    // Only 3 tokens can ever be built, so the prompt cannot reach the wanted length.
    expect(last(h).results[0].status).toBe('failed')
  })

  it('marks a failed load and carries on with the next combination', async () => {
    const h = harness({ failLoadAt: 1 })
    await h.runner.start(request([{ key: 'cacheType', values: ['f16', 'q8_0'] }]))
    await finish(h)
    const run = last(h)
    expect(run.results.map((r) => r.status)).toEqual(['failed', 'ok'])
    expect(run.results[0].error).toContain('out of memory')
    expect(run.status).toBe('done')
  })

  it('fails a combination whose context is too small for the prompt', async () => {
    const h = harness({ contextLength: 90 })
    await h.runner.start(request([], { prompt: { promptTokens: 64, genTokens: 16, repeats: 1 } }))
    await finish(h)
    const r = last(h).results[0]
    expect(r.status).toBe('failed')
    expect(r.error).toMatch(/context is too small/)
    expect(h.completions).toHaveLength(0)
    expect(h.instances.unload).toHaveBeenCalledWith('inst-1')
  })

  it('divides the context between parallel slots', async () => {
    const h = harness({ contextLength: 190 })
    await h.runner.start(request([], { base: { ...DEFAULT_LOAD_CONFIG, parallel: 2 }, prompt: { promptTokens: 64, genTokens: 16, repeats: 1 } }))
    await finish(h)
    expect(last(h).results[0].status).toBe('failed')
  })

  it('stops mid-run: the rest becomes stopped and everything is unloaded', async () => {
    let started!: () => void
    const inFlight = new Promise<void>((r) => (started = r))
    const h = harness({ hangCompletion: true, onCompletion: () => started() })
    await h.runner.start(request([{ key: 'cacheType', values: ['f16', 'q8_0', 'q4_0'] }]))
    await inFlight
    expect(h.instances.unloadAll).toHaveBeenCalledTimes(1)
    await h.runner.stop()
    const run = last(h)
    expect(run.status).toBe('stopped')
    expect(run.finishedAt).toBeTypeOf('number')
    expect(run.results.map((r) => r.status)).toEqual(['stopped', 'stopped', 'stopped'])
    expect(h.instances.unloadAll.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(h.instances.unload).toHaveBeenCalledWith('inst-1')
    expect(h.released).toBe(1)
    expect(h.loads).toHaveLength(1)
    expect(h.saved[h.saved.length - 1].status).toBe('stopped')
    expect(h.runner.isRunning()).toBe(false)
  })

  it('stops while a load is still in progress', async () => {
    const h = harness()
    let rejectLoad!: (e: Error) => void
    h.instances.load.mockImplementation(() => new Promise((_, reject) => (rejectLoad = reject)))
    h.instances.unloadAll.mockImplementation(async () => rejectLoad?.(new Error('Load cancelled')))
    await h.runner.start(request([{ key: 'cacheType', values: ['f16', 'q8_0'] }]))
    await vi.waitFor(() => expect(h.instances.load).toHaveBeenCalled())
    await h.runner.stop()
    expect(last(h).results.map((r) => r.status)).toEqual(['stopped', 'stopped'])
    expect(last(h).status).toBe('stopped')
  })

  it('a stop during start-up ends the run as stopped without loading anything', async () => {
    const h = harness()
    let release!: () => void
    const gate = new Promise<void>((r) => (release = r))
    let asked!: () => void
    const askedP = new Promise<void>((r) => (asked = r))
    h.deps.engineVersion = async () => {
      asked()
      await gate
      return 'b1'
    }
    h.runner = new BenchmarkRunner(h.deps)
    const starting = h.runner.start(request([{ key: 'cacheType', values: ['f16', 'q8_0'] }]))
    await askedP
    const stopping = h.runner.stop()
    release()
    await starting
    await stopping
    expect(h.runner.isRunning()).toBe(false)
    expect(h.instances.load).not.toHaveBeenCalled()
    expect(last(h).status).toBe('stopped')
    expect(last(h).results.map((r) => r.status)).toEqual(['stopped', 'stopped'])
    expect(h.saved[h.saved.length - 1].status).toBe('stopped')
  })

  it('a stop during the baseline reading does not start a load', async () => {
    const h = harness()
    let release!: () => void
    const gate = new Promise<void>((r) => (release = r))
    let asked!: () => void
    const askedP = new Promise<void>((r) => (asked = r))
    h.deps.sampleGpus = async () => {
      asked()
      await gate
      return []
    }
    h.runner = new BenchmarkRunner(h.deps)
    await h.runner.start(request())
    await askedP
    const stopping = h.runner.stop()
    release()
    await stopping
    expect(h.instances.load).not.toHaveBeenCalled()
    expect(last(h).status).toBe('stopped')
    expect(last(h).results[0].status).toBe('stopped')
  })

  it('times out a prompt tokenizing that never answers, and can be stopped while it waits', async () => {
    const hang = (_u: string | URL | Request, init?: RequestInit): Promise<Response> =>
      new Promise((_, reject) => init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))
    const h = harness()
    h.deps.fetch = hang as unknown as typeof fetch
    h.runner = new BenchmarkRunner({ ...h.deps, tokenizeTimeoutMs: 20 })
    await h.runner.start(request())
    await finish(h)
    expect(last(h).results[0].status).toBe('failed')
    expect(last(h).results[0].error).toMatch(/Tokenizing the prompt took longer/)

    const h2 = harness()
    h2.deps.fetch = hang as unknown as typeof fetch
    h2.runner = new BenchmarkRunner(h2.deps)
    await h2.runner.start(request())
    await vi.waitFor(() => expect(h2.instances.baseUrl).toHaveBeenCalled())
    await h2.runner.stop()
    expect(last(h2).status).toBe('stopped')
  })

  it('waits for a reading in flight before the final one', async () => {
    let calls = 0
    const h = harness()
    h.deps.sampleGpus = async () => {
      const n = calls++
      if (n > 0) await new Promise((r) => setTimeout(r, 15))
      return [gpu(0, 1000 + n * 100)]
    }
    h.runner = new BenchmarkRunner({ ...h.deps, sampleIntervalMs: 3 })
    await h.runner.start(request())
    await finish(h)
    const total = calls
    // The memory rises with every reading, so the last one taken must be in the peak.
    expect(last(h).results[0].peakVramMiB).toBe((total - 1) * 100)
    await new Promise((r) => setTimeout(r, 40))
    expect(calls).toBe(total)
  })

  it('keeps going when saving fails and says so on the finished run', async () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    try {
      const h = harness()
      h.deps.store = { save: async () => Promise.reject(new Error('disk full')) }
      h.runner = new BenchmarkRunner(h.deps)
      await h.runner.start(request())
      await finish(h)
      const run = last(h)
      expect(run.status).toBe('done')
      expect(run.results[0].status).toBe('ok')
      expect(run.error).toBe('The results could not be saved: disk full')
      expect(spy).toHaveBeenCalled()
    } finally {
      spy.mockRestore()
    }
  })

  it('ends as failed on an unexpected error and still unloads', async () => {
    const h = harness()
    h.instances.unloadAll.mockRejectedValueOnce(new Error('cannot kill'))
    await h.runner.start(request())
    await finish(h)
    const run = last(h)
    expect(run.status).toBe('failed')
    expect(run.error).toContain('cannot kill')
    expect(h.instances.unloadAll).toHaveBeenCalledTimes(2)
  })

  describe('refuses in start()', () => {
    it('while a run is going', async () => {
      const h = harness({ hangCompletion: true })
      await h.runner.start(request())
      await expect(h.runner.start(request())).rejects.toThrow(/already running/)
      await h.runner.stop()
    })
    it('a bad request', async () => {
      const h = harness()
      await expect(h.runner.start(request([{ key: 'draft', values: [] }]))).rejects.toThrow(/at least one value/i)
      expect(h.runner.isRunning()).toBe(false)
    })
    it('a missing model and a model without info', async () => {
      await expect(harness({ models: [] }).runner.start(request())).rejects.toThrow(/not found/i)
      await expect(harness({ models: [model({ info: null })] }).runner.start(request())).rejects.toThrow(/readable file information/)
    })
    it('MoE offload on a model without experts', async () => {
      const h = harness()
      await expect(h.runner.start(request([{ key: 'cpuMoe', values: ['off', 'all'] }]))).rejects.toThrow(/mixture-of-experts/)
      await expect(harness({ models: [model({ info: { architecture: 'qwen3moe', hasChatTemplate: true, expertCount: 8 } })] }).runner.start(request([{ key: 'cpuMoe', values: ['all'] }]))).resolves.toBeTruthy()
      // 'off' alone is fine on any model.
      await expect(harness().runner.start(request([{ key: 'cpuMoe', values: ['off'] }]))).resolves.toBeTruthy()
    })
    it('draft tokens without a draft', async () => {
      const h = harness()
      await expect(h.runner.start(request([{ key: 'draft', values: ['off', 3] }]))).rejects.toThrow(/draft/i)
      const withDraft = harness({ draftFor: () => ({ path: 'd.gguf', sizeBytes: 1 }) })
      await expect(withDraft.runner.start(request([{ key: 'draft', values: ['off', 3] }]))).resolves.toBeTruthy()
      await withDraft.runner.stop()
    })
  })
})

describe('BenchStore', () => {
  let dir: string
  let file: string
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'bench-store-'))
    file = join(dir, 'benchmarks.json')
  })
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true })
  })

  const run = (id: string, startedAt: number, extra: Partial<BenchmarkRun> = {}): BenchmarkRun => ({
    id,
    modelId: 'm/model.gguf',
    modelName: 'model',
    label: id,
    startedAt,
    status: 'done',
    base: { ...DEFAULT_LOAD_CONFIG },
    axes: [{ key: 'cacheType', values: ['f16', 'q8_0'] }],
    prompt: { promptTokens: 64, genTokens: 16, repeats: 1 },
    skipLikelyOom: false,
    results: [
      { combo: { cacheType: 'f16' }, status: 'ok', genTps: 12.5, peakVramMiB: 100, perGpu: [{ index: 0, name: 'GPU 0', peakMiB: 100 }] },
      { combo: { cacheType: 'q8_0' }, status: 'ok', genTps: 11 }
    ],
    ...extra
  })

  it('lists newest first and replaces by id', async () => {
    const store = new BenchStore(file)
    await store.save(run('a', 1))
    await store.save(run('b', 3))
    await store.save(run('c', 2))
    await store.save(run('a', 1, { label: 'changed' }))
    const list = await store.list()
    expect(list.map((r) => r.id)).toEqual(['b', 'c', 'a'])
    expect(list[2].label).toBe('changed')
    // It survives a restart.
    const again = await new BenchStore(file).list()
    expect(again.map((r) => r.id)).toEqual(['b', 'c', 'a'])
    expect(again[2].results[0].genTps).toBe(12.5)
  })

  it('keeps only the newest runs', async () => {
    const store = new BenchStore(file, 3)
    for (let i = 1; i <= 5; i++) await store.save(run(`r${i}`, i))
    expect((await store.list()).map((r) => r.id)).toEqual(['r5', 'r4', 'r3'])
    expect(MAX_BENCH_RUNS).toBeGreaterThan(3)
  })

  it('refuses to delete the running run, deletes others, ignores unknown ids', async () => {
    const store = new BenchStore(file)
    await store.list()
    await store.save(run('live', 5, { status: 'running' }))
    await store.save(run('old', 1))
    await expect(store.delete('live')).rejects.toThrow(/Stop the benchmark/)
    await store.delete('old')
    await store.delete('nope')
    expect((await store.list()).map((r) => r.id)).toEqual(['live'])
  })

  it('turns a run left running into stopped when the file is first read', async () => {
    const cut = run('cut', 5, {
      status: 'running',
      results: [
        { combo: { cacheType: 'f16' }, status: 'ok', genTps: 9 },
        { combo: { cacheType: 'q8_0' }, status: 'running' },
        { combo: { cacheType: 'q4_0' }, status: 'pending' }
      ]
    })
    await writeFile(file, JSON.stringify([cut, run('fine', 1)]), 'utf8')
    const list = await new BenchStore(file).list()
    expect(list[0].id).toBe('cut')
    expect(list[0].status).toBe('stopped')
    expect(list[0].results.map((r) => r.status)).toEqual(['ok', 'stopped', 'stopped'])
    expect(list[1].status).toBe('done')
  })

  it('does not cache an empty history when the file cannot be read, and reads it later', async () => {
    const { mkdir } = await import('node:fs/promises')
    await mkdir(file) // a directory where the file should be: reading it fails (EISDIR)
    const store = new BenchStore(file)
    await expect(store.list()).rejects.toThrow()
    await expect(store.save(run('x', 1))).rejects.toThrow()
    await rm(file, { recursive: true })
    await writeFile(file, JSON.stringify([run('real', 1)]), 'utf8')
    expect((await store.list()).map((r) => r.id)).toEqual(['real'])
  })

  it('drops damaged entries and survives a damaged file', async () => {
    await writeFile(file, JSON.stringify([run('ok', 1), { id: 'bad' }, null, 7, { ...run('worse', 2), axes: [{ key: 'nope', values: [1] }] }, { ...run('nobase', 3), base: null }]), 'utf8')
    expect((await new BenchStore(file).list()).map((r) => r.id)).toEqual(['ok'])
    await writeFile(file, '{ not json', 'utf8')
    expect(await new BenchStore(file).list()).toEqual([])
    const store = new BenchStore(file)
    await expect(store.save({ id: 1 })).rejects.toThrow(/Not a benchmark run/)
    await store.save(run('x', 1))
    expect(JSON.parse(await readFile(file, 'utf8'))).toHaveLength(1)
  })
})
