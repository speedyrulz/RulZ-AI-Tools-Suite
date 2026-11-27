// Pure helpers for the Merge page (no React, no IPC).
import { MAX_LORAS, type LoraEntry, type MergeJob, type MergeJobStatus, type MergeRecipe, type MergeVerify } from '@shared/merge'

const GIB = 1024 ** 3

/** "folder/sub/file.safetensors" -> { folder: "folder/sub", file: "file.safetensors" }. Both slash kinds count. */
export function splitName(name: string): { folder: string; file: string } {
  const i = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'))
  return i < 0 ? { folder: '', file: name } : { folder: name.slice(0, i), file: name.slice(i + 1) }
}

/** Case-insensitive match: every space-separated word must appear in the name. */
export function filterNames(names: string[], query: string): string[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return names
  return names.filter((n) => {
    const low = n.toLowerCase()
    return words.every((w) => low.includes(w))
  })
}

export function formatBytesGiB(bytes: number | null): string {
  return bytes === null ? 'unknown' : `${(bytes / GIB).toFixed(1)} GiB`
}

const n2 = (v: number): string => v.toFixed(2)

/** Live formula text for the chosen mode. */
export function formulaText(r: Pick<MergeRecipe, 'mode' | 'ratio' | 'multiplier' | 'blocks'>): string {
  if (r.mode === 'bake') return 'A + LoRAs'
  if (r.mode === 'weighted') return `A × ${n2(r.ratio)} + B × ${n2(1 - r.ratio)}`
  if (r.mode === 'addDifference') return `A + (B − C) × ${n2(r.multiplier)}`
  const count = Object.keys(r.blocks).length
  return `Per block (${count} changed, the rest A × ${n2(r.ratio)} + B × ${n2(1 - r.ratio)})`
}

/** Group name of a block input: the text before the first number ("double_blocks.3." -> "double_blocks"). */
export function blockGroupName(input: string): string {
  const m = input.match(/^(.*?)[._]?\d/)
  const base = (m ? m[1] : input).replace(/[._]+$/, '')
  return base || input
}

export interface BlockGroup {
  name: string
  inputs: string[]
}

/** Groups block inputs in server order. Groups of a single input are collected under "other". */
export function groupBlockInputs(inputs: string[]): BlockGroup[] {
  const map = new Map<string, string[]>()
  for (const i of inputs) {
    const g = blockGroupName(i)
    const list = map.get(g)
    if (list) list.push(i)
    else map.set(g, [i])
  }
  const groups: BlockGroup[] = []
  const other: string[] = []
  for (const [name, list] of map) {
    if (list.length > 1) groups.push({ name, inputs: list })
    else other.push(...list)
  }
  if (other.length > 0) groups.push({ name: 'other', inputs: other })
  return groups
}

/** Share of A per input, falling back to the recipe's ratio. */
export function blockValue(blocks: Record<string, number>, input: string, ratio: number): number {
  return blocks[input] ?? ratio
}

export function groupAverage(blocks: Record<string, number>, inputs: string[], ratio: number): number {
  if (inputs.length === 0) return ratio
  return inputs.reduce((s, i) => s + blockValue(blocks, i, ratio), 0) / inputs.length
}

/** Output name stem rule of the contract: letters, digits, - _ . and spaces. */
export function validOutputName(name: string): boolean {
  return /^[A-Za-z0-9._ -]+$/.test(name.trim()) && name.trim().length > 0
}

export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const mm = String(m).padStart(h > 0 ? 2 : 1, '0')
  const ss = String(sec).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

export function jobElapsed(job: Pick<MergeJob, 'createdAt' | 'finishedAt'>, now: number): number {
  return (job.finishedAt ?? now) - job.createdAt
}

export const JOB_STATUS: Record<MergeJobStatus, { label: string; tone: 'ok' | 'warn' | 'danger' | 'idle'; busy: boolean }> = {
  queued: { label: 'Queued', tone: 'idle', busy: false },
  running: { label: 'Running', tone: 'warn', busy: true },
  verifying: { label: 'Checking file', tone: 'warn', busy: true },
  done: { label: 'Done', tone: 'ok', busy: false },
  error: { label: 'Failed', tone: 'danger', busy: false },
  cancelled: { label: 'Cancelled', tone: 'idle', busy: false }
}

export function jobIsActive(status: MergeJobStatus): boolean {
  return status === 'queued' || status === 'running' || status === 'verifying'
}

export function jobIsTerminal(status: MergeJobStatus): boolean {
  return status === 'done' || status === 'error' || status === 'cancelled'
}

/** Which of two versions of the same job to keep. A finished job is never replaced by an unfinished one. */
export function newerJob(old: MergeJob, incoming: MergeJob): MergeJob {
  const oldDone = jobIsTerminal(old.status)
  const newDone = jobIsTerminal(incoming.status)
  if (oldDone && !newDone) return old
  if (!oldDone && newDone) return incoming
  if (oldDone && newDone) {
    const oldMoved = !!old.output?.movedTo
    const newMoved = !!incoming.output?.movedTo
    if (oldMoved !== newMoved) return oldMoved ? old : incoming
    return (incoming.finishedAt ?? 0) >= (old.finishedAt ?? 0) ? incoming : old
  }
  return incoming
}

/** Replaces or adds a job by id, newest first (see newerJob for which version wins). */
export function upsertJob(jobs: MergeJob[], job: MergeJob): MergeJob[] {
  const i = jobs.findIndex((j) => j.id === job.id)
  if (i >= 0) {
    const next = jobs.slice()
    next[i] = newerJob(jobs[i], job)
    return next
  }
  return [job, ...jobs].sort((a, b) => b.createdAt - a.createdAt)
}

export function progressFraction(p: { value: number; max: number } | null): number | null {
  if (!p || !(p.max > 0)) return null
  return Math.min(1, Math.max(0, p.value / p.max))
}

export function verifyText(v: { ok: boolean; tensorCount: number; recipeInHeader: boolean; message: string }): string {
  if (!v.ok) return v.message
  const layers = v.tensorCount.toLocaleString('en-US')
  return `File checked: ${layers} layers, ${v.recipeInHeader ? 'recipe stored' : 'recipe not stored in the file'}`
}

// ---- LoRAs ----

export const LORA_STRENGTH_LIMIT = 10

export function addLora(loras: LoraEntry[]): LoraEntry[] {
  if (loras.length >= MAX_LORAS) return loras
  return [...loras, { name: '', strength: 1, clipStrength: 1 }]
}

export function removeLora(loras: LoraEntry[], i: number): LoraEntry[] {
  return loras.filter((_, k) => k !== i)
}

/** Moves entry i one place up (dir -1) or down (dir 1); out of range returns the list unchanged. */
export function moveLora(loras: LoraEntry[], i: number, dir: -1 | 1): LoraEntry[] {
  const j = i + dir
  if (i < 0 || i >= loras.length || j < 0 || j >= loras.length) return loras
  const next = loras.slice()
  ;[next[i], next[j]] = [next[j], next[i]]
  return next
}

export function setLora(loras: LoraEntry[], i: number, patch: Partial<LoraEntry>): LoraEntry[] {
  if (i < 0 || i >= loras.length) return loras
  const next = loras.slice()
  next[i] = { ...next[i], ...patch }
  return next
}

/** The LoRA names the check reads: non-empty, in order. */
export function loraNames(loras: LoraEntry[]): string[] {
  return loras.map((l) => l.name).filter(Boolean)
}

function stem(name: string): string {
  return splitName(name).file.replace(/\.(safetensors|ckpt|pt|pth|bin|gguf)$/i, '')
}

/** "moss_image_base_moss-2500": A's stem and the LoRA stems, joined with "_" ("+" is not allowed in output names). */
export function suggestOutputName(r: Pick<MergeRecipe, 'a' | 'loras'>): string {
  const parts = [r.a ? stem(r.a) : '', ...r.loras.filter((l) => l.name).map((l) => stem(l.name))].filter(Boolean)
  if (parts.length === 0) return 'merged'
  const out = parts
    .join('_')
    .replace(/[^A-Za-z0-9 _.-]+/g, '_')
    .replace(/^[.\s_]+/, '')
    .slice(0, 120)
    .trim()
  return out === '' ? 'merged' : out
}

export function loraSummary(loras: LoraEntry[], kind: MergeRecipe['kind']): string {
  return loras
    .filter((l) => l.name)
    .map((l) => `${stem(l.name)} × ${l.strength}${kind === 'checkpoint' && l.clipStrength !== l.strength ? ` (text ${l.clipStrength})` : ''}`)
    .join(', ')
}

/** The LoRA line of a finished job: null when there is nothing to say. */
export function loraVerifyText(
  v: Pick<MergeVerify, 'loraKeysNotLoaded' | 'loraTextEncoderKeysSkipped'>,
  hasLoras: boolean
): { tone: 'ok' | 'warn'; text: string } | null {
  if (v.loraKeysNotLoaded === null) return null
  if (!hasLoras && v.loraKeysNotLoaded === 0 && v.loraTextEncoderKeysSkipped === 0) return null
  if (v.loraKeysNotLoaded > 0) {
    return {
      tone: 'warn',
      text: `${v.loraKeysNotLoaded.toLocaleString('en-US')} LoRA weight${v.loraKeysNotLoaded === 1 ? '' : 's'} not applied`
    }
  }
  return { tone: 'ok', text: 'LoRAs: all weights applied' }
}
