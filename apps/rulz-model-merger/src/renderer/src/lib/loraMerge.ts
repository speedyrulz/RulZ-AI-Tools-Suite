// Pure helpers for the LoRA merge page (no React, no IPC).
import {
  MAX_LORA_MERGE_INPUTS,
  type LoraMergeCheck,
  type LoraMergeDtype,
  type LoraMergeInspectRequest,
  type LoraMergeJob,
  type LoraMergeMode,
  type LoraMergeRecipe,
  type LoraMergeStatus
} from '@shared/loraMerge'
import { splitName } from './merge'

export const LORA_WEIGHT_LIMIT = 10
export const LORA_RANK_CHOICES = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
export const NAME_MAX = 80

export interface LoraRow {
  name: string
  weight: number
}

export const MODE_TEXT: Record<LoraMergeMode, { label: string; text: string }> = {
  concat: { label: 'Keep every rank (exact)', text: "The result is exactly the weighted sum; its rank is the sum of the inputs' ranks." },
  shrink: { label: 'Shrink to a smaller rank', text: 'Smaller file; the app reports how much of each layer it kept.' }
}

export const DTYPE_CHOICES: Array<{ value: LoraMergeDtype; label: string }> = [
  { value: 'source', label: 'Same as the first LoRA' },
  { value: 'bf16', label: 'bf16' },
  { value: 'fp16', label: 'fp16' },
  { value: 'fp32', label: 'fp32' }
]

export function emptyRows(): LoraRow[] {
  return [
    { name: '', weight: 1 },
    { name: '', weight: 1 }
  ]
}

export function addRow(rows: LoraRow[]): LoraRow[] {
  if (rows.length >= MAX_LORA_MERGE_INPUTS) return rows
  return [...rows, { name: '', weight: 1 }]
}

export function removeRow(rows: LoraRow[], i: number): LoraRow[] {
  if (i < 0 || i >= rows.length) return rows
  return rows.filter((_, k) => k !== i)
}

export function moveRow(rows: LoraRow[], i: number, dir: -1 | 1): LoraRow[] {
  const j = i + dir
  if (i < 0 || i >= rows.length || j < 0 || j >= rows.length) return rows
  const next = rows.slice()
  ;[next[i], next[j]] = [next[j], next[i]]
  return next
}

export function setRow(rows: LoraRow[], i: number, patch: Partial<LoraRow>): LoraRow[] {
  if (i < 0 || i >= rows.length) return rows
  const next = rows.slice()
  next[i] = { ...next[i], ...patch }
  return next
}

/** Rows that have a LoRA picked, in order. */
export function namedRows(rows: LoraRow[]): LoraRow[] {
  return rows.filter((r) => r.name)
}

export function namedList(rows: LoraRow[]): string[] {
  return namedRows(rows).map((r) => r.name)
}

export function validLoraOutputName(name: string): boolean {
  const t = name.trim()
  return t.length > 0 && t.length <= NAME_MAX && /^[A-Za-z0-9._ -]+$/.test(t)
}

function stem(name: string): string {
  return splitName(name).file.replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, '')
}

/** Stems of the named inputs joined with "_", cleaned to the allowed characters, at most NAME_MAX long. */
export function suggestLoraOutputName(rows: LoraRow[]): string {
  const parts = namedRows(rows)
    .map((r) => stem(r.name))
    .filter(Boolean)
  if (parts.length === 0) return 'merged-lora'
  const out = parts
    .join('_')
    .replace(/[^A-Za-z0-9 _.-]+/g, '_')
    .replace(/^[.\s_]+/, '')
    .slice(0, NAME_MAX)
    .trim()
  return out === '' ? 'merged-lora' : out
}

export interface BlockState {
  rows: LoraRow[]
  check: LoraMergeCheck | null
  checking: boolean
  checkError: string | null
  jobs: Array<Pick<LoraMergeJob, 'status'>>
  starting: boolean
  outputName: string
}

/** Why the Merge button is off; null when it can be pressed. */
export function loraMergeBlockReason(s: BlockState): string | null {
  if (s.jobs.some((j) => loraJobIsActive(j.status))) return 'A LoRA merge is running.'
  if (s.starting) return 'Starting.'
  const named = namedRows(s.rows).length
  if (named < 2) return 'Pick at least two LoRAs.'
  if (named < s.rows.length) return 'Pick a LoRA in every row or remove the empty ones.'
  if (!validLoraOutputName(s.outputName)) return 'Give the output a valid name.'
  if (s.checking) return 'The LoRAs are still being checked.'
  if (s.checkError) return `The LoRAs could not be checked: ${s.checkError}. Try again.`
  if (!s.check) return 'The LoRAs have not been checked yet.'
  if (s.check.problems.length > 0) return 'Fix the problems above first.'
  return null
}

export function buildRecipe(rows: LoraRow[], opts: { mode: LoraMergeMode; rank: number; dtype: LoraMergeDtype; outputName: string }): LoraMergeRecipe {
  return {
    inputs: namedRows(rows).map((r) => ({ name: r.name, weight: r.weight })),
    mode: opts.mode,
    rank: opts.rank,
    dtype: opts.dtype,
    outputName: opts.outputName.trim()
  }
}

export function buildInspectRequest(rows: LoraRow[], opts: { mode: LoraMergeMode; rank: number; dtype: LoraMergeDtype }): LoraMergeInspectRequest {
  return { names: namedList(rows), mode: opts.mode, rank: opts.rank, dtype: opts.dtype }
}

export function formatRankRange(r: { min: number; max: number } | null): string {
  if (!r) return 'unknown'
  return r.min === r.max ? `r${r.min}` : `r${r.min}–${r.max}`
}

/** "LoRA r32", "LoRA r16-64"; the bare format when the rank is unknown. */
export function formatInputKind(format: string, rank: { min: number; max: number } | null): string {
  return rank ? `${format} ${formatRankRange(rank)}` : format
}

export function formatSize(bytes: number | null): string {
  if (bytes === null) return 'unknown'
  const mib = bytes / 1024 ** 2
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${Math.max(0, Math.round(mib))} MiB`
}

/** 0.987 -> "98.7%". */
export function formatPercent(frac: number): string {
  return `${(frac * 100).toFixed(1)}%`
}

export function energyText(e: { mean: number; min: number }): string {
  return `Kept on average ${formatPercent(e.mean)} of each layer (lowest ${formatPercent(e.min)})`
}

export function verifyLine(v: { ok: boolean; message: string }): { tone: 'ok' | 'warn'; text: string } {
  return v.ok ? { tone: 'ok', text: v.message } : { tone: 'warn', text: `The check after saving found a problem: ${v.message}` }
}

export function recipeSummary(r: LoraMergeRecipe): string {
  const inputs = r.inputs.map((i) => `${stem(i.name)} × ${Number(i.weight.toFixed(2))}`).join(', ')
  const mode = r.mode === 'shrink' ? `shrunk to rank ${r.rank}` : 'every rank kept'
  return `${inputs} (${mode})`
}

export function loraJobIsActive(status: LoraMergeStatus): boolean {
  return status === 'running' || status === 'saving'
}

export const LORA_JOB_STATUS: Record<LoraMergeStatus, { label: string; tone: 'ok' | 'warn' | 'danger' | 'idle'; busy: boolean }> = {
  running: { label: 'Running', tone: 'warn', busy: true },
  saving: { label: 'Saving', tone: 'warn', busy: true },
  done: { label: 'Done', tone: 'ok', busy: false },
  error: { label: 'Failed', tone: 'danger', busy: false },
  cancelled: { label: 'Cancelled', tone: 'idle', busy: false }
}

const terminal = (s: LoraMergeStatus): boolean => s === 'done' || s === 'error' || s === 'cancelled'

/** Replaces or adds a job by id, newest first. A finished job is never replaced by an unfinished one. */
export function upsertLoraJob(jobs: LoraMergeJob[], job: LoraMergeJob): LoraMergeJob[] {
  const i = jobs.findIndex((j) => j.id === job.id)
  if (i >= 0) {
    const next = jobs.slice()
    next[i] = terminal(jobs[i].status) && !terminal(job.status) ? jobs[i] : job
    return next
  }
  return [job, ...jobs].sort((a, b) => b.createdAt - a.createdAt)
}

export function jobFraction(p: { done: number; total: number } | null): number | null {
  if (!p || !(p.total > 0)) return null
  return Math.min(1, Math.max(0, p.done / p.total))
}
