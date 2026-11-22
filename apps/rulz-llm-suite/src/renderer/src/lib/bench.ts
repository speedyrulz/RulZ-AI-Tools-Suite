// Load-settings benchmark, renderer side: the setup form's state, turning it into a request, and
// formatting results. Pure (no DOM, no IPC), so it is unit-tested; the contract is shared/benchmark.ts.
import {
  BENCH_AXIS_KEYS,
  BENCH_AXIS_NAMES,
  BENCH_PROMPT_DEFAULTS,
  benchRequestProblem,
  type BenchAxis,
  type BenchAxisKey,
  type BenchAxisValues,
  type BenchmarkRequest,
  type BenchmarkRun,
  type BenchResult
} from '@shared/benchmark'
import { KV_CACHE_TYPES, type LoadConfig } from '@shared/types'

/** Result of reading a typed list of whole numbers. */
export interface NumberList {
  numbers: number[]
  /** Entries that are not whole numbers in range, as typed. */
  bad: string[]
}

/**
 * Reads "8192, 32k  65536" (commas, semicolons or spaces between entries). A trailing k means x1024.
 * Duplicates are dropped (first kept); entries that are not whole numbers >= min are reported in `bad`.
 */
export function parseNumberList(text: string, min = 1): NumberList {
  const numbers: number[] = []
  const bad: string[] = []
  for (const raw of text.split(/[\s,;]+/)) {
    if (!raw) continue
    const m = /^(\d+)(k?)$/i.exec(raw)
    const n = m ? Number(m[1]) * (m[2] ? 1024 : 1) : NaN
    if (!m || !Number.isSafeInteger(n) || n < min) {
      bad.push(raw)
      continue
    }
    if (!numbers.includes(n)) numbers.push(n)
  }
  return { numbers, bad }
}

/** One setting's part of the form. */
export interface AxisForm {
  vary: boolean
  /** Ticked fixed choices, by id (see AXIS_SPECS). */
  picks: string[]
  /** Typed numbers. */
  custom: string
}

export interface BenchForm {
  axes: Record<BenchAxisKey, AxisForm>
  promptTokens: number
  genTokens: number
  repeats: number
  skipLikelyOom: boolean
  label: string
}

export interface AxisSpec {
  key: BenchAxisKey
  /** Fixed choices (ticked), in the order they are tried. */
  choices: { id: string; label: string }[]
  /** Whether typed numbers are allowed, with the smallest accepted one. */
  numbers: boolean
  min: number
  numberHint: string
}

export const AXIS_SPECS: Record<BenchAxisKey, AxisSpec> = {
  gpuLayers: {
    key: 'gpuLayers',
    choices: [
      { id: 'auto', label: 'Auto' },
      { id: 'all', label: 'All' }
    ],
    numbers: true,
    min: 0,
    numberHint: 'Layer counts, e.g. 20, 30'
  },
  cpuMoe: {
    key: 'cpuMoe',
    choices: [
      { id: 'off', label: 'Off' },
      { id: 'all', label: 'All' }
    ],
    numbers: true,
    min: 1,
    numberHint: 'Layers whose experts stay in RAM, e.g. 10, 20'
  },
  cacheType: {
    key: 'cacheType',
    choices: KV_CACHE_TYPES.map((t) => ({ id: t, label: t })),
    numbers: false,
    min: 0,
    numberHint: ''
  },
  draft: {
    key: 'draft',
    choices: [{ id: 'off', label: 'Off' }],
    numbers: true,
    min: 1,
    numberHint: 'Tokens drafted per step, e.g. 2, 3'
  },
  contextLength: {
    key: 'contextLength',
    choices: [{ id: 'auto', label: 'Auto' }],
    numbers: true,
    min: 512,
    numberHint: 'Tokens, e.g. 8192, 32k'
  },
  batchSize: {
    key: 'batchSize',
    choices: [{ id: 'default', label: 'Default' }],
    numbers: true,
    min: 1,
    numberHint: 'e.g. 512, 2048'
  },
  ubatchSize: {
    key: 'ubatchSize',
    choices: [{ id: 'default', label: 'Default' }],
    numbers: true,
    min: 1,
    numberHint: 'e.g. 256, 512'
  }
}

/** The form a model starts with: nothing varied, sensible values preselected for when a setting is. */
export function defaultBenchForm(): BenchForm {
  const a = (picks: string[], custom = ''): AxisForm => ({ vary: false, picks, custom })
  return {
    axes: {
      gpuLayers: a(['auto', 'all']),
      cpuMoe: a(['off', 'all']),
      cacheType: a(['f16', 'q8_0']),
      draft: a(['off'], '2, 3'),
      contextLength: a(['auto'], '8192, 32768'),
      batchSize: a(['default'], '512, 2048'),
      ubatchSize: a(['default'], '256, 512')
    },
    promptTokens: BENCH_PROMPT_DEFAULTS.promptTokens,
    genTokens: BENCH_PROMPT_DEFAULTS.genTokens,
    repeats: BENCH_PROMPT_DEFAULTS.repeats,
    skipLikelyOom: true,
    label: ''
  }
}

/** The settings this model can vary: MoE offload only for expert models, draft tokens only with a draft. */
export function availableAxes(model: { moe: boolean; hasDraft: boolean }): BenchAxisKey[] {
  return BENCH_AXIS_KEYS.filter((k) => (k === 'cpuMoe' ? model.moe : k === 'draft' ? model.hasDraft : true))
}

/** The value a ticked fixed choice stands for. */
export function pickValue<K extends BenchAxisKey>(key: K, id: string): BenchAxisValues[K] {
  const v: unknown =
    key === 'contextLength' || key === 'batchSize' || key === 'ubatchSize' ? null : id // auto / default
  return v as BenchAxisValues[K]
}

/** One setting's values in try order: ticked choices (in spec order), then the typed numbers. */
export function axisValues(key: BenchAxisKey, f: AxisForm): { values: unknown[]; bad: string[] } {
  const spec = AXIS_SPECS[key]
  const values: unknown[] = spec.choices.filter((c) => f.picks.includes(c.id)).map((c) => pickValue(key, c.id))
  let bad: string[] = []
  if (spec.numbers) {
    const list = parseNumberList(f.custom, spec.min)
    values.push(...list.numbers)
    bad = list.bad
  }
  return { values, bad }
}

export interface BuiltBench {
  request: BenchmarkRequest
  /** Why it cannot start (unreadable entries first, then benchRequestProblem), or null. */
  problem: string | null
}

/** Builds the request from the form. Settings the model cannot vary are left out whatever the form says. */
export function buildBenchRequest(
  form: BenchForm,
  ctx: { modelId: string; base: LoadConfig; available: BenchAxisKey[] }
): BuiltBench {
  const axes: BenchAxis[] = []
  let parseProblem: string | null = null
  for (const key of ctx.available) {
    const f = form.axes[key]
    if (!f.vary) continue
    const { values, bad } = axisValues(key, f)
    if (bad.length && !parseProblem) parseProblem = `${BENCH_AXIS_NAMES[key]}: cannot use ${bad.map((b) => `"${b}"`).join(', ')}.`
    axes.push({ key, values } as BenchAxis)
  }
  const request: BenchmarkRequest = {
    modelId: ctx.modelId,
    base: ctx.base,
    axes,
    prompt: { promptTokens: form.promptTokens, genTokens: form.genTokens, repeats: form.repeats },
    skipLikelyOom: form.skipLikelyOom,
    label: form.label.trim()
  }
  return { request, problem: parseProblem ?? benchRequestProblem(request) }
}

/** A short line describing the settings a benchmark starts from. */
export function baseSummary(c: LoadConfig): string {
  const parts = [
    `GPU layers ${c.gpuLayers}`,
    c.cpuMoe !== 'off' ? `MoE offload ${c.cpuMoe}` : null,
    `context ${c.contextLength ? c.contextLength : 'auto'}`,
    c.cacheTypeK === c.cacheTypeV ? `KV cache ${c.cacheTypeK}` : `KV cache ${c.cacheTypeK}/${c.cacheTypeV}`,
    `flash attention ${c.flashAttention}`,
    c.useDraft ? `draft ${c.draftTokens ?? 'default'}` : 'no draft',
    `batch ${c.batchSize ?? 'default'}/${c.ubatchSize ?? 'default'}`
  ]
  return parts.filter(Boolean).join(' · ')
}

// ---- formatting ----

const dash = '—'
const ok = (n: number | null | undefined): n is number => typeof n === 'number' && Number.isFinite(n)

/** 850 ms, 1.25 s, 2m 5s. */
export function fmtMs(ms: number | null | undefined): string {
  if (!ok(ms)) return dash
  if (ms < 1000) return `${Math.round(ms)} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

/** Tokens per second with one decimal. */
export function fmtTps(n: number | null | undefined): string {
  return ok(n) ? n.toFixed(1) : dash
}

/** A 0..1 ratio as a whole percent. */
export function fmtRatioPct(r: number | null | undefined): string {
  return ok(r) ? `${Math.round(r * 100)}%` : dash
}

/** MiB as GB (1024 MiB per GB), one decimal. */
export function fmtMiB(mib: number | null | undefined): string {
  return ok(mib) ? `${(mib / 1024).toFixed(1)} GB` : dash
}

/** Bytes as GB (1024^3), one decimal. */
export function fmtGbFromBytes(bytes: number | null | undefined): string {
  return ok(bytes) ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : dash
}

/** Measured peak VRAM against the estimate, in percent of the estimate (positive: used more than estimated). */
export function vramDiffPct(peakMiB: number | null | undefined, estimateBytes: number | null | undefined): number | null {
  if (!ok(peakMiB) || !ok(estimateBytes) || estimateBytes <= 0) return null
  return ((peakMiB * 1024 * 1024 - estimateBytes) / estimateBytes) * 100
}

/** "+5%", "-12%", "0%". */
export function fmtSignedPct(p: number | null | undefined): string {
  if (!ok(p)) return dash
  const r = Math.round(p)
  return r > 0 ? `+${r}%` : `${r}%`
}

/** Combinations that are over (finished, failed, skipped or stopped) and the total. */
export function runProgress(run: Pick<BenchmarkRun, 'results'>): { done: number; total: number } {
  const total = run.results.length
  const done = run.results.filter((r) => r.status !== 'pending' && r.status !== 'running').length
  return { done, total }
}

/**
 * Time left: the average time per combination over those done so far, times those still to go. Null until one
 * is done or when the run is over. Skipped combinations are quick, so the estimate runs long when more skip.
 */
export function etaMs(run: Pick<BenchmarkRun, 'results' | 'startedAt' | 'status'>, now: number): number | null {
  if (run.status !== 'running') return null
  const { done, total } = runProgress(run)
  if (done < 1 || done >= total) return null
  const elapsed = Math.max(0, now - run.startedAt)
  return (elapsed / done) * (total - done)
}

export function resultVramDiff(r: BenchResult): number | null {
  return vramDiffPct(r.peakVramMiB, r.estimate?.gpuBytes)
}

/** Runs of one model, newest first. */
export function runsForModel(runs: BenchmarkRun[], modelId: string): BenchmarkRun[] {
  return runs.filter((r) => r.modelId === modelId).sort((a, b) => b.startedAt - a.startedAt)
}

/** Puts a changed run in the list (replacing the one with its id), newest first. */
export function upsertRun(runs: BenchmarkRun[], run: BenchmarkRun): BenchmarkRun[] {
  const next = runs.some((r) => r.id === run.id) ? runs.map((r) => (r.id === run.id ? run : r)) : [run, ...runs]
  return next.sort((a, b) => b.startedAt - a.startedAt)
}

/** Whether `a` is at least as far along as `b` (same run): over beats running, then more combinations over. */
function atLeastAsNew(a: BenchmarkRun, b: BenchmarkRun): boolean {
  if (a.status !== 'running' && b.status === 'running') return true
  if (a.status === 'running' && b.status !== 'running') return false
  return runProgress(a).done >= runProgress(b).done
}

/** Merges a run that may be older than the list's copy (a late reply): keeps the list's copy when it is further along. */
export function mergeRun(runs: BenchmarkRun[], run: BenchmarkRun): BenchmarkRun[] {
  const cur = runs.find((r) => r.id === run.id)
  if (cur && atLeastAsNew(cur, run) && !(cur.status === run.status && runProgress(cur).done === runProgress(run).done)) return runs
  return upsertRun(runs, run)
}

/** Adds a run only when the list has none with its id (the bench:start reply; events are at least as new). */
export function insertRunIfAbsent(runs: BenchmarkRun[], run: BenchmarkRun): BenchmarkRun[] {
  return runs.some((r) => r.id === run.id) ? runs : upsertRun(runs, run)
}
