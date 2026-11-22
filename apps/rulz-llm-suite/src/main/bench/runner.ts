// Load-settings benchmark runner: loads a model once per combination of settings, measures speed and VRAM
// with one fixed prompt, and records the numbers. Everything it touches is injected so tests can stub it.
import { randomUUID } from 'node:crypto'
import {
  benchRequestProblem,
  comboConfig,
  expandCombos,
  median,
  type BenchGpuPeak,
  type BenchmarkRequest,
  type BenchmarkRun,
  type BenchResult
} from '@shared/benchmark'
import type { GpuStat, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry, SidecarFile } from '@shared/types'

/** Wait after an unload before reading the baseline, so the previous model's memory is released. */
const SETTLE_MS = 1500
const SAMPLE_INTERVAL_MS = 500
const REQUEST_TIMEOUT_MS = 10 * 60 * 1000
const TOKENIZE_TIMEOUT_MS = 60 * 1000
const WARMUP_TOKENS = 8
const FILLER =
  'The quick brown fox jumps over the lazy dog while a gentle river carries autumn leaves past the old stone bridge, ' +
  'and the villagers talk about weather, harvests, music and the long road to the distant harbour town. '

export interface BenchInstances {
  load(modelId: string, config: LoadConfig | null): Promise<LoadedInstance>
  unload(instanceId: string): Promise<void>
  unloadAll(): Promise<void>
  acquire(instanceId: string): () => void
  baseUrl(instance: LoadedInstance): string
}

export interface BenchRunStore {
  save(run: BenchmarkRun): Promise<void>
}

export interface BenchDeps {
  instances: BenchInstances
  getModel: (id: string) => ModelEntry | undefined
  estimate: (model: ModelEntry, cfg: LoadConfig) => MemoryEstimate | null
  draftFor: (model: ModelEntry, cfg: LoadConfig) => SidecarFile | undefined
  /** One reading of every GPU's memory (nvidia-smi); [] when there is none. */
  sampleGpus: () => Promise<GpuStat[]>
  engineVersion: () => Promise<string | undefined>
  onChange: (run: BenchmarkRun) => void
  store: BenchRunStore
  fetch?: typeof fetch
  sleep?: (ms: number) => Promise<void>
  now?: () => number
  settleMs?: number
  sampleIntervalMs?: number
  requestTimeoutMs?: number
  tokenizeTimeoutMs?: number
}

class StopSignal extends Error {
  constructor() {
    super('Stopped')
  }
}

interface Measurement {
  ttftMs: number
  promptTps: number
  genTps: number
  drafted: number
  accepted: number
}

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

/** Reads GPU memory during one combination: the baseline before loading, then the highest use seen. */
class VramMeter {
  private baseline = new Map<number, { name: string; used: number }>()
  private peak = new Map<number, number>()
  private timer: ReturnType<typeof setInterval> | null = null
  /** The newest reading still being taken; readings run one after another. */
  private inflight: Promise<void> | null = null

  constructor(
    private readonly sample: () => Promise<GpuStat[]>,
    private readonly intervalMs: number
  ) {}

  async begin(): Promise<void> {
    const gpus = await this.sample().catch(() => [] as GpuStat[])
    for (const g of gpus) {
      this.baseline.set(g.index, { name: g.name, used: g.memUsedMiB })
      this.peak.set(g.index, g.memUsedMiB)
    }
    // A timer reading is skipped while another is still running.
    this.timer = setInterval(() => {
      if (!this.inflight) void this.tick()
    }, this.intervalMs)
  }

  /** Takes one reading (after any reading still running); used at the key moments and by the timer. */
  tick(): Promise<void> {
    const take = async (): Promise<void> => {
      try {
        for (const g of await this.sample()) {
          const prev = this.peak.get(g.index)
          if (prev !== undefined && g.memUsedMiB > prev) this.peak.set(g.index, g.memUsedMiB)
        }
      } catch {
        // a missed reading only lowers the peak
      }
    }
    const p: Promise<void> = (this.inflight ?? Promise.resolve()).then(take).finally(() => {
      if (this.inflight === p) this.inflight = null
    })
    this.inflight = p
    return p
  }

  async end(): Promise<{ peakVramMiB: number | null; perGpu: BenchGpuPeak[] }> {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
    await this.tick()
    if (!this.baseline.size) return { peakVramMiB: null, perGpu: [] }
    const perGpu: BenchGpuPeak[] = [...this.baseline.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([index, b]) => ({ index, name: b.name, peakMiB: Math.max(0, (this.peak.get(index) ?? b.used) - b.used) }))
    return { peakVramMiB: perGpu.reduce((s, g) => s + g.peakMiB, 0), perGpu }
  }

  cancel(): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
  }
}

export class BenchmarkRunner {
  private current: BenchmarkRun | null = null
  private stopping = false
  private ctrl: AbortController | null = null
  private loopDone: Promise<void> = Promise.resolve()
  private tokens: number[] | null = null
  /** The reason the latest save failed, if any; reported on the run when it ends. */
  private saveError: string | null = null

  constructor(private readonly deps: BenchDeps) {}

  isRunning(): boolean {
    return this.current !== null
  }

  private now(): number {
    return (this.deps.now ?? Date.now)()
  }

  private sleep(ms: number): Promise<void> {
    return (this.deps.sleep ?? ((n) => new Promise<void>((r) => setTimeout(r, n))))(ms)
  }

  async start(req: BenchmarkRequest): Promise<BenchmarkRun> {
    if (this.current) throw new Error('A benchmark is already running.')
    const problem = benchRequestProblem(req)
    if (problem) throw new Error(problem)
    const model = this.deps.getModel(req.modelId)
    if (!model) throw new Error('Model not found.')
    if (!model.info) throw new Error('This model has no readable file information, so it cannot be benchmarked.')
    for (const axis of req.axes) {
      if (axis.key === 'cpuMoe' && axis.values.some((v) => v !== 'off') && !(model.info.expertCount && model.info.expertCount > 0))
        throw new Error('MoE offload needs a mixture-of-experts model; this model has no experts.')
      if (axis.key === 'draft')
        for (const v of axis.values) {
          if (v === 'off') continue
          if (!this.deps.draftFor(model, comboConfig(req.base, { draft: v })))
            throw new Error('Draft tokens need a draft for this model, and there is none.')
        }
    }
    // Reserve the slot before the first await, so two quick starts cannot both run.
    const run: BenchmarkRun = {
      id: randomUUID(),
      modelId: req.modelId,
      modelName: model.name,
      label: req.label,
      startedAt: this.now(),
      status: 'running',
      base: req.base,
      axes: req.axes,
      prompt: req.prompt,
      skipLikelyOom: req.skipLikelyOom,
      results: expandCombos(req.axes).map((combo): BenchResult => ({ combo, status: 'pending' }))
    }
    this.current = run
    this.stopping = false
    this.tokens = null
    this.ctrl = new AbortController()
    this.saveError = null
    // Made before the first await, so a stop() during the awaits below waits for the whole run.
    let finished!: () => void
    this.loopDone = new Promise<void>((r) => (finished = r))
    try {
      try {
        run.engineVersion = (await this.deps.engineVersion().catch(() => undefined)) ?? undefined
      } catch {
        // the version is informational only
      }
      await this.emit(run)
    } catch (err) {
      // Nothing started, so nothing is left to clean up.
      this.current = null
      this.ctrl = null
      finished()
      throw err
    }
    const snapshot = structuredClone(run)
    void this.execute(run, model)
      .catch(() => undefined)
      .finally(finished)
    return snapshot
  }

  /** Stops the run: the current request is aborted, everything is unloaded, the rest becomes 'stopped'. */
  async stop(): Promise<void> {
    if (!this.current) return
    this.stopping = true
    this.ctrl?.abort()
    await this.deps.instances.unloadAll().catch(() => undefined)
    await this.loopDone
  }

  private async emit(run: BenchmarkRun): Promise<void> {
    const copy = structuredClone(run)
    try {
      this.deps.onChange(copy)
    } catch {
      // a broadcast problem must not stop the run
    }
    await this.deps.store.save(copy).catch((err) => {
      console.error('Could not save the benchmark run:', err)
      this.saveError = errText(err)
    })
  }

  private async execute(run: BenchmarkRun, model: ModelEntry): Promise<void> {
    try {
      if (!this.stopping) await this.deps.instances.unloadAll()
      for (const result of run.results) {
        if (this.stopping) break
        try {
          await this.runCombo(run, model, result)
        } catch (err) {
          if (err instanceof StopSignal || this.stopping) {
            result.status = 'stopped'
            break
          }
          result.status = 'failed'
          result.error = errText(err)
        }
        await this.emit(run)
      }
      if (this.stopping) {
        for (const r of run.results) if (r.status === 'pending' || r.status === 'running') r.status = 'stopped'
        run.status = 'stopped'
      } else run.status = 'done'
    } catch (err) {
      for (const r of run.results) if (r.status === 'pending' || r.status === 'running') r.status = 'stopped'
      run.status = this.stopping ? 'stopped' : 'failed'
      if (!this.stopping) run.error = errText(err)
    } finally {
      await this.deps.instances.unloadAll().catch(() => undefined)
      run.finishedAt = this.now()
      const noteSaveError = (): void => {
        if (this.saveError && !run.error) run.error = `The results could not be saved: ${this.saveError}`
      }
      noteSaveError()
      await this.emit(run)
      // This very save may have been the one that failed: tell the page, without another save attempt.
      if (this.saveError && !run.error) {
        noteSaveError()
        try {
          this.deps.onChange(structuredClone(run))
        } catch {
          // a broadcast problem must not matter here
        }
      }
      this.current = null
      this.ctrl = null
      this.stopping = false
    }
  }

  private checkStop(): void {
    if (this.stopping) throw new StopSignal()
  }

  private async runCombo(run: BenchmarkRun, model: ModelEntry, result: BenchResult): Promise<void> {
    const cfg = comboConfig(run.base, result.combo)
    const est = this.deps.estimate(model, cfg)
    if (est)
      result.estimate = { gpuBytes: est.gpuBytes, totalBytes: est.totalBytes, verdict: est.verdict, cpuExpertBytes: est.cpuExpertBytes }
    if (run.skipLikelyOom && est?.verdict === 'tight') {
      result.status = 'skipped'
      result.error = 'Skipped: the estimate says this would not fit.'
      return
    }
    result.status = 'running'
    await this.emit(run)

    await this.sleep(this.deps.settleMs ?? SETTLE_MS)
    this.checkStop()
    const meter = new VramMeter(this.deps.sampleGpus, this.deps.sampleIntervalMs ?? SAMPLE_INTERVAL_MS)
    await meter.begin()
    if (this.stopping) {
      meter.cancel()
      throw new StopSignal()
    }
    let instance: LoadedInstance | null = null
    let release: () => void = () => undefined
    try {
      this.checkStop()
      const loadStart = this.now()
      try {
        instance = await this.deps.instances.load(run.modelId, cfg)
      } catch (err) {
        this.checkStop()
        throw new Error(`Load failed: ${errText(err)}`)
      }
      result.loadMs = this.now() - loadStart
      result.contextLength = instance.contextLength
      release = this.deps.instances.acquire(instance.instanceId)
      await meter.tick()
      this.checkStop()

      const { promptTokens, genTokens, repeats } = run.prompt
      const perSlot = instance.contextLength !== null ? Math.floor(instance.contextLength / Math.max(1, cfg.parallel)) : null
      if (perSlot !== null && perSlot < promptTokens + genTokens + 16)
        throw new Error(
          `The context is too small: ${perSlot} tokens per slot, but the prompt needs ${promptTokens + genTokens + 16}. Raise the context or lower the prompt length.`
        )

      const base = this.deps.instances.baseUrl(instance)
      const tokens = await this.promptTokens(base, promptTokens)
      this.checkStop()
      await this.request(base, tokens, WARMUP_TOKENS)
      await meter.tick()
      const timed: Measurement[] = []
      for (let i = 0; i < repeats; i++) {
        this.checkStop()
        timed.push(await this.request(base, tokens, genTokens))
        await meter.tick()
      }
      result.ttftMs = median(timed.map((m) => m.ttftMs))
      result.promptTps = median(timed.map((m) => m.promptTps))
      result.genTps = median(timed.map((m) => m.genTps))
      const drafted = timed.reduce((s, m) => s + m.drafted, 0)
      result.draftAcceptance = drafted > 0 ? timed.reduce((s, m) => s + m.accepted, 0) / drafted : null
      result.status = 'ok'
    } finally {
      release()
      const vram = await meter.end().catch(() => {
        meter.cancel()
        return null
      })
      if (vram) {
        result.peakVramMiB = vram.peakVramMiB
        result.perGpu = vram.perGpu
      }
      if (instance) await this.deps.instances.unload(instance.instanceId).catch(() => undefined)
    }
  }

  /** The fixed prompt as exactly `count` token ids of the loaded model's own tokenizer. */
  private async promptTokens(base: string, count: number): Promise<number[]> {
    if (this.tokens && this.tokens.length >= count) return this.tokens.slice(0, count)
    const f = this.deps.fetch ?? fetch
    let chars = count * 6
    for (let attempt = 0; attempt < 3; attempt++) {
      const text = FILLER.repeat(Math.ceil(chars / FILLER.length)).slice(0, chars)
      // One timeout covers the request and reading its body; stopping the run aborts it too.
      const ctrl = new AbortController()
      const onStop = (): void => ctrl.abort()
      this.ctrl?.signal.addEventListener('abort', onStop)
      let timedOut = false
      const timeoutMs = this.deps.tokenizeTimeoutMs ?? TOKENIZE_TIMEOUT_MS
      const timer = setTimeout(() => {
        timedOut = true
        ctrl.abort()
      }, timeoutMs)
      let body: { tokens?: unknown }
      try {
        const res = await f(`${base}/tokenize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: text }),
          signal: ctrl.signal
        })
        if (!res.ok) throw new Error(`Tokenizing the prompt failed (HTTP ${res.status}).`)
        body = (await res.json()) as { tokens?: unknown }
      } catch (err) {
        if (this.stopping) throw new StopSignal()
        if (timedOut) throw new Error(`Tokenizing the prompt took longer than ${Math.round(timeoutMs / 1000)} seconds and was cancelled.`)
        throw err
      } finally {
        clearTimeout(timer)
        this.ctrl?.signal.removeEventListener('abort', onStop)
      }
      const tokens = Array.isArray(body.tokens) ? body.tokens.filter((t): t is number => typeof t === 'number') : []
      if (tokens.length >= count) {
        this.tokens = tokens.slice(0, count)
        return this.tokens
      }
      chars *= 2
    }
    throw new Error('Could not build a prompt of the wanted length.')
  }

  /** One streamed completion; throws on an HTTP error, an error chunk, a timeout or a missing timings block. */
  private async request(base: string, tokens: number[], nPredict: number): Promise<Measurement> {
    const f = this.deps.fetch ?? fetch
    const ctrl = new AbortController()
    const onStop = (): void => ctrl.abort()
    this.ctrl?.signal.addEventListener('abort', onStop)
    const timeoutMs = this.deps.requestTimeoutMs ?? REQUEST_TIMEOUT_MS
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      ctrl.abort()
    }, timeoutMs)
    try {
      const sent = this.now()
      const res = await f(`${base}/completion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: tokens,
          n_predict: nPredict,
          cache_prompt: false,
          stream: true,
          temperature: 0,
          ignore_eos: true
        }),
        signal: ctrl.signal
      })
      if (!res.ok) throw new Error(`The server answered HTTP ${res.status}: ${(await res.text().catch(() => '')).slice(0, 200)}`)
      if (!res.body) throw new Error('The server sent no response body.')
      let ttft: number | null = null
      let timings: { prompt_per_second?: number; predicted_per_second?: number; draft_n?: number; draft_n_accepted?: number } | null = null
      const handle = (line: string): void => {
        if (!line.startsWith('data:')) return
        const payload = line.slice(5).trim()
        if (!payload || payload === '[DONE]') return
        let chunk: { content?: string; stop?: boolean; error?: unknown; timings?: typeof timings }
        try {
          chunk = JSON.parse(payload)
        } catch {
          return
        }
        if (chunk.error) throw new Error(typeof chunk.error === 'string' ? chunk.error : JSON.stringify(chunk.error))
        if (ttft === null && typeof chunk.content === 'string' && chunk.content.length > 0) ttft = this.now() - sent
        if (chunk.stop && chunk.timings) timings = chunk.timings
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let nl: number
        while ((nl = buf.indexOf('\n')) >= 0) {
          handle(buf.slice(0, nl).replace(/\r$/, ''))
          buf = buf.slice(nl + 1)
        }
      }
      buf += decoder.decode()
      if (buf) handle(buf.replace(/\r$/, ''))
      const t = timings as { prompt_per_second?: number; predicted_per_second?: number; draft_n?: number; draft_n_accepted?: number } | null
      if (!t || typeof t.predicted_per_second !== 'number') throw new Error('The server did not report timings for the request.')
      if (ttft === null) throw new Error('The model generated no text.')
      return {
        ttftMs: ttft,
        promptTps: t.prompt_per_second ?? 0,
        genTps: t.predicted_per_second,
        drafted: t.draft_n ?? 0,
        accepted: t.draft_n_accepted ?? 0
      }
    } catch (err) {
      if (this.stopping) throw new StopSignal()
      if (timedOut) throw new Error(`The request took longer than ${Math.round(timeoutMs / 60000)} minutes and was cancelled.`)
      throw err
    } finally {
      clearTimeout(timer)
      this.ctrl?.signal.removeEventListener('abort', onStop)
    }
  }
}
