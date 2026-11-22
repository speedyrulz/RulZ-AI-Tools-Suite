// Saved benchmark runs, one JSON file in the app's data folder.
import { benchRequestProblem, MAX_BENCH_RUNS, type BenchGpuPeak, type BenchmarkRun, type BenchResult } from '@shared/benchmark'
import type { MemoryEstimate } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

const RUN_STATUSES = ['running', 'done', 'stopped', 'failed']
const RESULT_STATUSES = ['pending', 'running', 'ok', 'failed', 'skipped', 'stopped']
const VERDICTS = ['fits', 'partial', 'tight', 'unknown']

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)

function cleanResult(raw: unknown): BenchResult | null {
  if (!isObj(raw) || !isObj(raw.combo) || typeof raw.status !== 'string' || !RESULT_STATUSES.includes(raw.status)) return null
  const r: BenchResult = { combo: raw.combo as BenchResult['combo'], status: raw.status as BenchResult['status'] }
  if (typeof raw.error === 'string') r.error = raw.error.slice(0, 2000)
  for (const k of ['loadMs', 'ttftMs', 'promptTps', 'genTps'] as const) {
    const v = num(raw[k])
    if (v !== undefined) r[k] = v
  }
  if (raw.contextLength === null || num(raw.contextLength) !== undefined) r.contextLength = raw.contextLength as number | null
  if (raw.draftAcceptance === null || num(raw.draftAcceptance) !== undefined) r.draftAcceptance = raw.draftAcceptance as number | null
  if (raw.peakVramMiB === null || num(raw.peakVramMiB) !== undefined) r.peakVramMiB = raw.peakVramMiB as number | null
  if (Array.isArray(raw.perGpu)) {
    r.perGpu = raw.perGpu.flatMap((g): BenchGpuPeak[] =>
      isObj(g) && num(g.index) !== undefined && typeof g.name === 'string' && num(g.peakMiB) !== undefined
        ? [{ index: g.index as number, name: g.name, peakMiB: g.peakMiB as number }]
        : []
    )
  }
  const e = raw.estimate
  if (isObj(e) && num(e.gpuBytes) !== undefined && num(e.totalBytes) !== undefined && typeof e.verdict === 'string' && VERDICTS.includes(e.verdict)) {
    r.estimate = {
      gpuBytes: e.gpuBytes as number,
      totalBytes: e.totalBytes as number,
      verdict: e.verdict as MemoryEstimate['verdict'],
      cpuExpertBytes: num(e.cpuExpertBytes) ?? 0
    }
  }
  return r
}

/** A saved run read back from disk (or handed to save): null when it is not a usable run. */
export function cleanBenchRun(raw: unknown): BenchmarkRun | null {
  if (!isObj(raw)) return null
  if (typeof raw.id !== 'string' || !raw.id || typeof raw.modelId !== 'string') return null
  if (typeof raw.status !== 'string' || !RUN_STATUSES.includes(raw.status)) return null
  const startedAt = num(raw.startedAt)
  if (startedAt === undefined) return null
  if (!isObj(raw.base) || !Array.isArray(raw.base.adapters) || !isObj(raw.base.sampling)) return null
  if (!Array.isArray(raw.axes) || !isObj(raw.prompt) || !Array.isArray(raw.results)) return null
  if (benchRequestProblem({ axes: raw.axes as BenchmarkRun['axes'], prompt: raw.prompt as unknown as BenchmarkRun['prompt'] })) return null
  const results = raw.results.flatMap((r) => {
    const c = cleanResult(r)
    return c ? [c] : []
  })
  const run: BenchmarkRun = {
    id: raw.id,
    modelId: raw.modelId,
    modelName: typeof raw.modelName === 'string' ? raw.modelName : raw.modelId,
    label: typeof raw.label === 'string' ? raw.label : '',
    startedAt,
    status: raw.status as BenchmarkRun['status'],
    base: raw.base as unknown as BenchmarkRun['base'],
    axes: raw.axes as BenchmarkRun['axes'],
    prompt: raw.prompt as unknown as BenchmarkRun['prompt'],
    skipLikelyOom: raw.skipLikelyOom === true,
    results
  }
  const finishedAt = num(raw.finishedAt)
  if (finishedAt !== undefined) run.finishedAt = finishedAt
  if (typeof raw.error === 'string') run.error = raw.error.slice(0, 2000)
  if (typeof raw.engineVersion === 'string') run.engineVersion = raw.engineVersion.slice(0, 200)
  return run
}

/** A run cut off by closing the app: it is over, so it reads as stopped. */
export function settleBenchRun(run: BenchmarkRun): BenchmarkRun {
  if (run.status !== 'running') return run
  return {
    ...run,
    status: 'stopped',
    results: run.results.map((r) => (r.status === 'running' || r.status === 'pending' ? { ...r, status: 'stopped' as const } : r))
  }
}

export class BenchStore {
  private runs: BenchmarkRun[] | null = null
  /** Reads, saves and deletes run one after another, so two saves cannot write each other's list away. */
  private queue: Promise<unknown> = Promise.resolve()

  constructor(
    private readonly path: string,
    private readonly maxRuns = MAX_BENCH_RUNS
  ) {}

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.queue.catch(() => undefined).then(fn)
    this.queue = next
    return next
  }

  private async load(): Promise<BenchmarkRun[]> {
    if (!this.runs) {
      // A read error is let through (and nothing cached), so a locked file is retried later instead of
      // reading as an empty history that the next save would overwrite.
      const raw = await readJson<unknown>(this.path, [])
      const list = Array.isArray(raw) ? raw : []
      // Nothing is running when the file is first read: a run still marked so was cut off.
      this.runs = list.flatMap((r) => {
        const run = cleanBenchRun(r)
        return run ? [settleBenchRun(run)] : []
      })
    }
    return this.runs
  }

  /** Newest first. */
  list(): Promise<BenchmarkRun[]> {
    return this.enqueue(async () => [...(await this.load())].sort((a, b) => b.startedAt - a.startedAt))
  }

  save(raw: unknown): Promise<void> {
    return this.enqueue(async () => {
      const run = cleanBenchRun(raw)
      if (!run) throw new Error('Not a benchmark run.')
      const runs = (await this.load()).filter((r) => r.id !== run.id)
      runs.push(run)
      runs.sort((a, b) => b.startedAt - a.startedAt)
      this.runs = runs.slice(0, this.maxRuns)
      await writeJsonAtomic(this.path, this.runs)
    })
  }

  delete(id: string): Promise<void> {
    return this.enqueue(async () => {
      const runs = await this.load()
      const run = runs.find((r) => r.id === id)
      if (!run) return
      if (run.status === 'running') throw new Error('Stop the benchmark before deleting it.')
      this.runs = runs.filter((r) => r.id !== id)
      await writeJsonAtomic(this.path, this.runs)
    })
  }
}
