// Writing benchmark: types shared by the main-process store and the renderer.
// Design: docs/writing-mode-design.md, "Stage 4 design" (Writing benchmark). Contract: change only with both sides.
import type { LoadAdapter, LoadConfig, SamplingParams } from './types'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from './defaults'

export const WBENCH_FORMAT = 1

export interface BenchPrompt {
  id: string
  title: string
  /** The scene brief sent to the model. */
  text: string
  builtin: boolean
}

export interface BenchCandidate {
  id: string
  label: string
  modelId: string
  modelName: string
  adapters: LoadAdapter[]
  sampling: SamplingParams
  /** Load settings for this candidate in this run only; null = the model's saved load settings. Adapters come from `adapters`. */
  load: LoadConfig | null
}

export type BenchFinish = 'stop' | 'length' | 'stopped' | 'error'

export type BenchOutputStatus = 'pending' | 'running' | 'done' | 'failed'

export interface BenchOutput {
  /** Unique within the run: `${promptId}|${candidateId}|${seed}`. */
  key: string
  promptId: string
  candidateId: string
  seed: number
  status: BenchOutputStatus
  text: string
  finish: BenchFinish | null
  error: string | null
  ms: number | null
  /** Generation speed the server measured, tokens per second; null when not measured (or an old run file). */
  tokensPerSecond: number | null
  /** Prompt processing speed, tokens per second; null when not measured. */
  promptTokensPerSecond: number | null
  /** Tokens generated; null when not measured. */
  tokens: number | null
  /** Prompt tokens actually processed (cached ones left out); weights the prompt speed average. Null when not measured. */
  promptTokens: number | null
}

export type BenchVote = 'a' | 'b' | 'tie' | 'skip'

export interface BenchPair {
  id: string
  promptId: string
  seed: number
  /** Output key shown as A. */
  a: string
  /** Output key shown as B. */
  b: string
  vote: BenchVote | null
  votedAt: string | null
}

export type WritingBenchStatus = 'pending' | 'running' | 'stopped' | 'done'

export interface WritingBenchRun {
  format: 1
  id: string
  label: string
  createdAt: string
  updatedAt: string
  status: WritingBenchStatus
  /** Snapshot of the prompts used, so later edits do not change the run. */
  prompts: BenchPrompt[]
  candidates: BenchCandidate[]
  /** One seed list shared by every candidate. */
  seeds: number[]
  maxTokens: number
  outputs: BenchOutput[]
  pairs: BenchPair[]
}

/** A run file that could not be read or validated is listed, never rewritten by the app. */
export type BenchRunEntry =
  | { ok: true; run: WritingBenchRun }
  | { ok: false; id: string; file: string; problem: string }

export interface BenchScore {
  candidateId: string
  rating: number
  wins: number
  losses: number
  ties: number
  votes: number
}

export interface WritingBenchIpcInvoke {
  'wbench:list': () => BenchRunEntry[]
  /** Validates and writes runs/<id>.json; returns the run as stored. */
  'wbench:save': (run: WritingBenchRun) => WritingBenchRun
  /** Deletes runs/<id>.json (also a damaged one). */
  'wbench:delete': (id: string) => void
  /** The user's own prompts (prompts.json); the bundled ones live in code. */
  'wbench:prompts': () => BenchPrompt[]
  'wbench:savePrompts': (prompts: BenchPrompt[]) => BenchPrompt[]
}

export const BENCH_SYSTEM_PROMPT = 'Write the requested scene as polished prose. Do not include a title, notes, or lists.'
export const BUILTIN_PROMPTS: BenchPrompt[] = [
  ['The return', 'A woman returns to her childhood home to sort her father’s belongings. During a power cut, a neighbor brings her a letter he never sent. Write the encounter as one scene in prose.'],
  ['The witness', 'A detective interviews a baker who claims to have seen a robbery through a flour-dusted window. The witness changes one small detail each time the clock strikes. Write one scene in prose.'],
  ['The toll', 'A ferryman demands a memory instead of a coin to carry a traveler across an enchanted river. The traveler has already forgotten why the journey matters. Write one scene in prose.'],
  ['The signal', 'An engineer alone on a distant relay station receives a message from a ship that has not launched yet. The message asks her to shut down the beacon before sunrise. Write one scene in prose.'],
  ['The rehearsal', 'Two former partners must rehearse a dance for a mutual friend’s wedding. One has brought a pair of shoes the other thought lost. Write their reunion as one scene in prose.'],
  ['The upstairs room', 'A night caretaker hears furniture being dragged across the locked upstairs room of an empty museum. The sound stops whenever she approaches the stairs. Write one scene in prose.'],
  ['The prize', 'A town’s annual prize for the best vegetable is accidentally awarded to a papier-mache pumpkin. The winner must give a speech before the judges discover the mistake. Write one scene in prose.'],
  ['The mapmaker', 'In a busy port centuries ago, a mapmaker is asked to erase an island from a chart before a ship sails. The captain insists the island is real and offers a curious proof. Write one scene in prose.']
].map(([title, text], i) => ({ id: `builtin-${i + 1}`, title, text, builtin: true }))

export function benchMessages(prompt: BenchPrompt): { role: 'system' | 'user'; content: string }[] {
  return [{ role: 'system', content: BENCH_SYSTEM_PROMPT }, { role: 'user', content: prompt.text }]
}
export function outputKey(promptId: string, candidateId: string, seed: number): string {
  return `${promptId}|${candidateId}|${seed}`
}
export function newRun(input: { label: string; prompts: BenchPrompt[]; candidates: BenchCandidate[]; seeds: number[]; maxTokens: number }, now: string, id: string): WritingBenchRun {
  const snapshot = structuredClone(input)
  return { format: 1, id, ...snapshot, createdAt: now, updatedAt: now, status: 'pending', pairs: [],
    outputs: snapshot.candidates.flatMap((c) => snapshot.prompts.flatMap((p) => snapshot.seeds.map((seed) => ({
      key: outputKey(p.id, c.id, seed), promptId: p.id, candidateId: c.id, seed, status: 'pending' as const,
      text: '', finish: null, error: null, ms: null, tokensPerSecond: null, promptTokensPerSecond: null, tokens: null, promptTokens: null
    })))) }
}
export function nextOutput(run: WritingBenchRun): BenchOutput | null { return run.outputs.find((o) => o.status === 'pending') ?? null }
export function addPairs(run: WritingBenchRun, rand: () => number): WritingBenchRun {
  const pairs = [...run.pairs]
  const done = run.outputs.filter((o) => o.status === 'done')
  for (let i = 0; i < done.length; i++) for (let j = i + 1; j < done.length; j++) {
    const x = done[i], y = done[j]
    if (x.promptId !== y.promptId || x.seed !== y.seed || x.candidateId === y.candidateId) continue
    if (pairs.some((p) => (p.a === x.key && p.b === y.key) || (p.a === y.key && p.b === x.key))) continue
    const [a, b] = rand() < 0.5 ? [x.key, y.key] : [y.key, x.key]
    pairs.push({ id: JSON.stringify([x.key, y.key].sort()), promptId: x.promptId, seed: x.seed, a, b, vote: null, votedAt: null })
  }
  return { ...run, pairs }
}
export function nextPair(run: WritingBenchRun): BenchPair | null { return run.pairs.find((p) => p.vote === null) ?? null }
export function scores(run: WritingBenchRun): BenchScore[] {
  const result = run.candidates.map((c) => ({ candidateId: c.id, rating: 1000, wins: 0, losses: 0, ties: 0, votes: 0 }))
  const byCandidate = new Map(result.map((s) => [s.candidateId, s]))
  const outputs = new Map(run.outputs.map((o) => [o.key, o]))
  for (const p of [...run.pairs].filter((p) => p.vote && p.vote !== 'skip').sort((a, b) => (a.votedAt ?? '').localeCompare(b.votedAt ?? ''))) {
    const a = byCandidate.get(outputs.get(p.a)?.candidateId ?? ''), b = byCandidate.get(outputs.get(p.b)?.candidateId ?? '')
    if (!a || !b || a === b) continue
    const actual = p.vote === 'a' ? 1 : p.vote === 'b' ? 0 : 0.5
    const change = 32 * (actual - 1 / (1 + 10 ** ((b.rating - a.rating) / 400)))
    a.rating += change; b.rating -= change; a.votes++; b.votes++
    if (actual === 0.5) { a.ties++; b.ties++ } else if (actual === 1) { a.wins++; b.losses++ } else { b.wins++; a.losses++ }
  }
  return result
}
export interface BenchSpeed {
  candidateId: string
  /** Plain mean of the outputs' generation tokens/s; null when none was measured. */
  tokensPerSecond: number | null
  /** Plain mean of the outputs' prompt tokens/s; null when none was measured. */
  promptTokensPerSecond: number | null
  /** Done outputs with a generation speed. */
  measured: number
}
/**
 * Average speeds per candidate over its done outputs, weighted by tokens: total tokens / total time (time = tokens / rate).
 * Weighting matters for the prompt speed: after the first output the server reuses the cached prompt, so later outputs
 * process a few tokens in a few milliseconds, and a plain mean of those rates would mean little. Outputs without a token
 * count (older run files) fall back to a plain mean.
 */
export function speedByCandidate(run: WritingBenchRun): BenchSpeed[] {
  const average = (items: { rate: number | null | undefined; n: number | null | undefined }[]): number | null => {
    const rated = items.filter((i): i is { rate: number; n: number | null | undefined } => typeof i.rate === 'number' && i.rate > 0)
    const counted = rated.filter((i): i is { rate: number; n: number } => typeof i.n === 'number' && i.n > 0)
    if (counted.length) {
      const tokens = counted.reduce((s, i) => s + i.n, 0)
      const seconds = counted.reduce((s, i) => s + i.n / i.rate, 0)
      return seconds > 0 ? tokens / seconds : null
    }
    return rated.length ? rated.reduce((s, i) => s + i.rate, 0) / rated.length : null
  }
  return run.candidates.map((c) => {
    const done = run.outputs.filter((o) => o.candidateId === c.id && o.status === 'done')
    const gen = done.map((o) => ({ rate: o.tokensPerSecond, n: o.tokens }))
    const prompt = done.map((o) => ({ rate: o.promptTokensPerSecond, n: o.promptTokens }))
    return { candidateId: c.id, tokensPerSecond: average(gen), promptTokensPerSecond: average(prompt), measured: gen.filter((g) => typeof g.rate === 'number' && g.rate > 0).length }
  })
}
export function randomSeeds(n: number, rand: () => number): number[] {
  const seeds: number[] = [], used = new Set<number>()
  for (let i = 0; i < n; i++) {
    let seed = Math.floor(rand() * 0x80000000) & 0x7fffffff
    while (used.has(seed)) seed = (seed + 1) % 0x80000000
    used.add(seed); seeds.push(seed)
  }
  return seeds
}

const obj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const str = (v: unknown): v is string => typeof v === 'string'
const integer = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0
const promptOk = (v: unknown): v is BenchPrompt => obj(v) && str(v.id) && !!v.id && !v.id.includes('|') && str(v.title) && str(v.text) && typeof v.builtin === 'boolean'
export function cleanPrompts(raw: unknown): BenchPrompt[] {
  return Array.isArray(raw) ? raw.filter(promptOk).filter((p) => !p.builtin).map((p) => ({ ...p, builtin: false })) : []
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
/** A load config as a run file stores it: every key present must have the type of its default; a missing key is filled from the defaults. */
function loadOk(v: unknown): boolean {
  if (!obj(v)) return false
  if (v.adapters !== undefined && !(Array.isArray(v.adapters) && v.adapters.every((a: unknown) => obj(a) && str(a.id) && finite(a.scale)))) return false
  return Object.entries(DEFAULT_LOAD_CONFIG).every(([key, sample]) => {
    const value = v[key]
    if (value === undefined || key === 'adapters') return true
    if (key === 'gpuLayers' || key === 'cpuMoe') return str(value) || finite(value)
    if (sample === null) return value === null || finite(value) || str(value)
    if (Array.isArray(sample)) return Array.isArray(value)
    if (obj(sample)) return obj(value)
    // Numeric settings may be null ("Auto (fit)" context, empty draft tokens): the main process checks them again on load.
    if (typeof sample === 'number') return value === null || finite(value)
    return typeof value === typeof sample
  })
}
export function cleanRun(raw: unknown): { ok: true; run: WritingBenchRun } | { ok: false; problem: string } {
  const bad = (): { ok: false; problem: string } => ({ ok: false, problem: 'Invalid writing benchmark run.' })
  if (!obj(raw) || raw.format !== 1 || !str(raw.id) || !/^[A-Za-z0-9_-]{1,64}$/.test(raw.id) || !str(raw.label) || !str(raw.createdAt) || !str(raw.updatedAt) || !str(raw.status) || !['pending', 'running', 'stopped', 'done'].includes(raw.status)) return bad()
  if (!Array.isArray(raw.prompts) || !raw.prompts.every(promptOk) || !Array.isArray(raw.candidates) || !Array.isArray(raw.seeds) || !raw.seeds.every((s: unknown) => integer(s) && s <= 0x7fffffff) || !integer(raw.maxTokens) || raw.maxTokens === 0 || !Array.isArray(raw.outputs) || !Array.isArray(raw.pairs)) return bad()
  if (!raw.candidates.every((c: unknown) => obj(c) && str(c.id) && !!c.id && !c.id.includes('|') && str(c.label) && str(c.modelId) && str(c.modelName) && Array.isArray(c.adapters) && c.adapters.every((a: unknown) => obj(a) && str(a.id) && typeof a.scale === 'number' && Number.isFinite(a.scale)) && obj(c.sampling) && Object.entries(DEFAULT_SAMPLING).every(([key, sample]) => {
    const value = (c.sampling as Record<string, unknown>)[key]
    if (sample === null) return value === null || (typeof value === 'number' && Number.isFinite(value))
    if (Array.isArray(sample)) return Array.isArray(value) && value.every(str)
    if (typeof sample === 'number') return typeof value === 'number' && Number.isFinite(value)
    return typeof value === typeof sample
  }) && ['auto', 'on', 'off'].includes(c.sampling.thinking as string) && ['default', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'].includes(c.sampling.reasoningEffort as string) && (c.load === undefined || c.load === null || loadOk(c.load)))) return bad()
  const unique = (items: string[]): boolean => new Set(items).size === items.length
  const prompts = raw.prompts as BenchPrompt[], candidates = raw.candidates as BenchCandidate[], seeds = raw.seeds as number[]
  if (!unique(prompts.map((p) => p.id)) || !unique(candidates.map((c) => c.id)) || new Set(seeds).size !== seeds.length) return bad()
  const keys = new Map<string, BenchOutput>()
  for (const o of raw.outputs) {
    if (!obj(o) || !str(o.key) || !str(o.promptId) || !str(o.candidateId) || !integer(o.seed) || o.key !== outputKey(o.promptId, o.candidateId, o.seed) || keys.has(o.key) || !prompts.some((p) => p.id === o.promptId) || !candidates.some((c) => c.id === o.candidateId) || !seeds.includes(o.seed) || !str(o.status) || !['pending', 'running', 'done', 'failed'].includes(o.status) || !str(o.text) || !(o.finish === null || (str(o.finish) && ['stop', 'length', 'stopped', 'error'].includes(o.finish))) || !(o.error === null || str(o.error)) || !(o.ms === null || (typeof o.ms === 'number' && Number.isFinite(o.ms) && o.ms >= 0))) return bad()
    // Speeds: absent in run files from before they were measured (read as null); present ones must be numbers >= 0 or null.
    const speed = (v: unknown): boolean => v === undefined || v === null || (finite(v) && v >= 0)
    if (!speed(o.tokensPerSecond) || !speed(o.promptTokensPerSecond) || !speed(o.tokens) || !speed(o.promptTokens)) return bad()
    keys.set(o.key, { ...o, tokensPerSecond: o.tokensPerSecond ?? null, promptTokensPerSecond: o.promptTokensPerSecond ?? null, tokens: o.tokens ?? null, promptTokens: o.promptTokens ?? null } as unknown as BenchOutput)
  }
  if (keys.size !== prompts.length * candidates.length * seeds.length) return bad()
  const ids = new Set<string>(), pairKeys = new Set<string>()
  for (const p of raw.pairs) {
    if (!obj(p) || !str(p.id) || !p.id || ids.has(p.id) || !str(p.a) || !str(p.b) || !str(p.promptId) || !integer(p.seed) || !(p.vote === null || (str(p.vote) && ['a', 'b', 'tie', 'skip'].includes(p.vote))) || !(p.votedAt === null || str(p.votedAt)) || (p.vote === null) !== (p.votedAt === null)) return bad()
    const a = keys.get(p.a), b = keys.get(p.b), key = JSON.stringify([p.a, p.b].sort())
    if (!a || !b || a.candidateId === b.candidateId || a.promptId !== p.promptId || b.promptId !== p.promptId || a.seed !== p.seed || b.seed !== p.seed || pairKeys.has(key)) return bad()
    ids.add(p.id); pairKeys.add(key)
  }
  // Run files from before per-candidate load settings have no `load`: that means the model's saved settings.
  const cleaned = candidates.map((c) => ({ ...c, load: c.load ? ({ ...DEFAULT_LOAD_CONFIG, ...c.load } as LoadConfig) : null }))
  const outputs = (raw.outputs as unknown[]).map((o) => keys.get((o as { key: string }).key) as BenchOutput)
  return { ok: true, run: { ...(raw as unknown as WritingBenchRun), candidates: cleaned, outputs } }
}
