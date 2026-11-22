// Load-settings benchmark: one fixed prompt run over every combination of a few load settings, measuring
// speed and VRAM, so the settings can be chosen by measurement instead of by the estimate alone.
// Pure; shared by main and renderer.
import type { KvCacheType, LoadConfig, MemoryEstimate } from './types'
import { KV_CACHE_TYPES } from './types'

/** The settings a benchmark can vary. */
export type BenchAxisKey =
  | 'gpuLayers'
  | 'cpuMoe'
  /** Sets both cacheTypeK and cacheTypeV. */
  | 'cacheType'
  /** 'off' = useDraft false; a number = useDraft true with that many draft tokens per step. */
  | 'draft'
  | 'contextLength'
  | 'batchSize'
  | 'ubatchSize'

export interface BenchAxisValues {
  gpuLayers: LoadConfig['gpuLayers']
  cpuMoe: LoadConfig['cpuMoe']
  cacheType: KvCacheType
  draft: 'off' | number
  /** null = Auto (fit). */
  contextLength: number | null
  /** null = llama.cpp's default. */
  batchSize: number | null
  ubatchSize: number | null
}

export const BENCH_AXIS_KEYS: readonly BenchAxisKey[] = [
  'gpuLayers',
  'cpuMoe',
  'cacheType',
  'draft',
  'contextLength',
  'batchSize',
  'ubatchSize'
]

/** One varied setting and the values to try, in the order given. */
export type BenchAxis = { [K in BenchAxisKey]: { key: K; values: BenchAxisValues[K][] } }[BenchAxisKey]

/** One combination: a value for each varied setting. Settings not in it keep the base config's value. */
export type BenchCombo = Partial<BenchAxisValues>

export interface BenchPrompt {
  /** Prompt length in tokens (the prompt is built and cut to this many with the model's own tokenizer). */
  promptTokens: number
  /** Tokens generated per measurement (end-of-text is ignored, so every measurement writes this many). */
  genTokens: number
  /** Timed measurements per combination, after one untimed warm-up request; results are medians. */
  repeats: number
}

export const BENCH_PROMPT_DEFAULTS: BenchPrompt = { promptTokens: 2048, genTokens: 128, repeats: 2 }
export const BENCH_PROMPT_TOKENS = [512, 2048, 8192] as const
export const MAX_BENCH_COMBOS = 48
export const MAX_BENCH_AXIS_VALUES = 8
/** Saved runs kept (newest first). */
export const MAX_BENCH_RUNS = 100

export interface BenchmarkRequest {
  modelId: string
  /** Settings every combination starts from (normally the model's saved defaults or the form's values). */
  base: LoadConfig
  axes: BenchAxis[]
  prompt: BenchPrompt
  /** Skip combinations the memory estimate calls likely out of memory ('tight'), instead of trying them. */
  skipLikelyOom: boolean
  /** Shown in the run list. */
  label: string
}

export type BenchResultStatus =
  | 'pending'
  | 'running'
  | 'ok'
  /** The model failed to load or a request failed. */
  | 'failed'
  /** Skipped because the estimate said it would not fit (skipLikelyOom). */
  | 'skipped'
  /** The run was stopped before or during this combination. */
  | 'stopped'

export interface BenchGpuPeak {
  index: number
  name: string
  /** Highest memory.used seen while this combination was loaded, minus memory.used before it loaded. */
  peakMiB: number
}

export interface BenchResult {
  combo: BenchCombo
  status: BenchResultStatus
  error?: string
  /** From starting llama-server until it answered as ready. */
  loadMs?: number
  /** Context size the server reported once ready. */
  contextLength?: number | null
  /** Median wall time from sending the request until the first generated token arrived. */
  ttftMs?: number
  /** Median prompt processing speed (llama-server timings.prompt_per_second). */
  promptTps?: number
  /** Median generation speed (timings.predicted_per_second). */
  genTps?: number
  /** Drafted tokens accepted / drafted, summed over the timed measurements; null when nothing was drafted. */
  draftAcceptance?: number | null
  /** Sum over GPUs of BenchGpuPeak.peakMiB. Null when nvidia-smi is unavailable. */
  peakVramMiB?: number | null
  perGpu?: BenchGpuPeak[]
  /** The memory estimate for this combination's config, made just before loading it. */
  estimate?: Pick<MemoryEstimate, 'gpuBytes' | 'totalBytes' | 'verdict' | 'cpuExpertBytes'>
}

export type BenchRunStatus = 'running' | 'done' | 'stopped' | 'failed'

export interface BenchmarkRun {
  id: string
  modelId: string
  modelName: string
  label: string
  startedAt: number
  finishedAt?: number
  status: BenchRunStatus
  /** Why the whole run failed (engine missing, model gone); per-combination errors are on the results. */
  error?: string
  base: LoadConfig
  axes: BenchAxis[]
  prompt: BenchPrompt
  skipLikelyOom: boolean
  /** One per combination, in expandCombos order. */
  results: BenchResult[]
  /** llama.cpp build the run used (EngineInfo.version), when known. */
  engineVersion?: string
}

/** Every combination of the axes' values (cartesian product), first axis varying slowest. [{}] for no axes. */
export function expandCombos(axes: BenchAxis[]): BenchCombo[] {
  let out: BenchCombo[] = [{}]
  for (const axis of axes) {
    if (!axis.values.length) continue
    const next: BenchCombo[] = []
    for (const combo of out) for (const v of axis.values) next.push({ ...combo, [axis.key]: v })
    out = next
  }
  return out
}

/** How many combinations the axes make, without building them. */
export function comboCount(axes: BenchAxis[]): number {
  return axes.reduce((n, a) => (a.values.length ? n * a.values.length : n), 1)
}

/** The load config a combination runs with. */
export function comboConfig(base: LoadConfig, combo: BenchCombo): LoadConfig {
  const cfg: LoadConfig = { ...base, sampling: { ...base.sampling }, adapters: base.adapters.map((a) => ({ ...a })) }
  if (combo.gpuLayers !== undefined) cfg.gpuLayers = combo.gpuLayers
  if (combo.cpuMoe !== undefined) cfg.cpuMoe = combo.cpuMoe
  if (combo.cacheType !== undefined) {
    cfg.cacheTypeK = combo.cacheType
    cfg.cacheTypeV = combo.cacheType
  }
  if (combo.draft !== undefined) {
    if (combo.draft === 'off') cfg.useDraft = false
    else {
      cfg.useDraft = true
      cfg.draftTokens = combo.draft
    }
  }
  if (combo.contextLength !== undefined) cfg.contextLength = combo.contextLength
  if (combo.batchSize !== undefined) cfg.batchSize = combo.batchSize
  if (combo.ubatchSize !== undefined) cfg.ubatchSize = combo.ubatchSize
  return cfg
}

/** Short text for one axis value, e.g. "all", "q8_0", "draft 3", "auto". */
export function axisValueLabel<K extends BenchAxisKey>(key: K, value: BenchAxisValues[K]): string {
  if (value === null) return key === 'contextLength' ? 'auto' : 'default'
  if (key === 'draft') return value === 'off' ? 'off' : `${value}`
  if (key === 'contextLength' && typeof value === 'number') return value % 1024 === 0 ? `${value / 1024}K` : `${value}`
  return String(value)
}

export const BENCH_AXIS_NAMES: Record<BenchAxisKey, string> = {
  gpuLayers: 'GPU layers',
  cpuMoe: 'MoE offload',
  cacheType: 'KV cache',
  draft: 'Draft tokens',
  contextLength: 'Context',
  batchSize: 'Batch',
  ubatchSize: 'Micro-batch'
}

/** "GPU layers all · KV cache q8_0" for a combination, in BENCH_AXIS_KEYS order; "Base settings" for {}. */
export function comboLabel(combo: BenchCombo): string {
  const parts: string[] = []
  for (const key of BENCH_AXIS_KEYS) {
    if (!(key in combo)) continue
    const v = combo[key] as BenchAxisValues[typeof key]
    parts.push(`${BENCH_AXIS_NAMES[key]} ${axisValueLabel(key, v)}`)
  }
  return parts.length ? parts.join(' · ') : 'Base settings'
}

/** Median of the finite numbers; undefined when there are none. */
export function median(values: number[]): number | undefined {
  const xs = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
  if (!xs.length) return undefined
  const mid = Math.floor(xs.length / 2)
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2
}

const isIntIn = (v: unknown, lo: number, hi: number): v is number =>
  typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi

/** Whether one axis value is allowed for its key. */
export function validAxisValue(key: BenchAxisKey, v: unknown): boolean {
  switch (key) {
    case 'gpuLayers':
      return v === 'auto' || v === 'all' || isIntIn(v, 0, 1000)
    case 'cpuMoe':
      return v === 'off' || v === 'all' || isIntIn(v, 1, 1000)
    case 'cacheType':
      return typeof v === 'string' && (KV_CACHE_TYPES as string[]).includes(v)
    case 'draft':
      return v === 'off' || isIntIn(v, 1, 64)
    case 'contextLength':
      return v === null || isIntIn(v, 512, 4_194_304)
    case 'batchSize':
    case 'ubatchSize':
      return v === null || isIntIn(v, 1, 65_536)
  }
}

/**
 * Checks a request's axes and prompt. Returns the problem as text, or null when it can run. Each key may
 * appear once, with 1..MAX_BENCH_AXIS_VALUES distinct valid values, and the product must stay within
 * MAX_BENCH_COMBOS.
 */
export function benchRequestProblem(req: Pick<BenchmarkRequest, 'axes' | 'prompt'>): string | null {
  const seen = new Set<BenchAxisKey>()
  for (const axis of req.axes) {
    if (!BENCH_AXIS_KEYS.includes(axis.key)) return `Unknown setting: ${String(axis.key)}`
    if (seen.has(axis.key)) return `${BENCH_AXIS_NAMES[axis.key]} is listed twice.`
    seen.add(axis.key)
    const values = axis.values as unknown[]
    if (!values.length) return `Pick at least one value for ${BENCH_AXIS_NAMES[axis.key]}.`
    if (values.length > MAX_BENCH_AXIS_VALUES)
      return `At most ${MAX_BENCH_AXIS_VALUES} values for ${BENCH_AXIS_NAMES[axis.key]}.`
    if (new Set(values.map((v) => JSON.stringify(v))).size !== values.length)
      return `${BENCH_AXIS_NAMES[axis.key]} has the same value twice.`
    const bad = values.find((v) => !validAxisValue(axis.key, v))
    if (bad !== undefined) return `${BENCH_AXIS_NAMES[axis.key]}: ${JSON.stringify(bad)} is not allowed.`
  }
  const n = comboCount(req.axes)
  if (n > MAX_BENCH_COMBOS) return `${n} combinations; at most ${MAX_BENCH_COMBOS}. Remove some values.`
  const p = req.prompt
  if (!isIntIn(p.promptTokens, 16, 131_072)) return 'Prompt length must be 16 to 131072 tokens.'
  if (!isIntIn(p.genTokens, 8, 4096)) return 'Generated tokens must be 8 to 4096.'
  if (!isIntIn(p.repeats, 1, 10)) return 'Repeats must be 1 to 10.'
  return null
}

/** The fastest finished result by generation speed (ties: lower peak VRAM), or null when none finished. */
export function bestResult(results: BenchResult[]): BenchResult | null {
  let best: BenchResult | null = null
  for (const r of results) {
    if (r.status !== 'ok' || r.genTps === undefined) continue
    if (
      !best ||
      r.genTps > (best.genTps ?? 0) ||
      (r.genTps === best.genTps && (r.peakVramMiB ?? Infinity) < (best.peakVramMiB ?? Infinity))
    )
      best = r
  }
  return best
}
