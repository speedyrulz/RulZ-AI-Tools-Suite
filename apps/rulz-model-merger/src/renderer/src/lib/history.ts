// Pure helpers for the History page (no React, no IPC): labels, summaries, filtering, and what Re-run checks.
import { MAX_LORA_MERGE_INPUTS, type LoraMergeRecipe } from '@shared/loraMerge'
import type { HistoryEntry } from '@shared/history'
import type { MergeRecipe } from '@shared/merge'
import { GRID_STATUS, formatDate, round4 } from './grid'
import { LORA_JOB_STATUS, loraJobIsActive, recipeSummary as loraRecipeSummary, validLoraOutputName, type LoraRow } from './loraMerge'
import { JOB_STATUS, jobIsActive, splitName, validOutputName } from './merge'
import { gridRowKind, type GridJob, type GridRequest } from '@shared/grid'

export type HistoryFilter = 'all' | 'merges' | 'loraMerges' | 'grids'

export const HISTORY_FILTERS: Array<{ value: HistoryFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'merges', label: 'Merges' },
  { value: 'loraMerges', label: 'LoRA merges' },
  { value: 'grids', label: 'Grids' }
]

// ---- tolerant readers: one broken entry must never blank the page ----

const asArr = <T>(v: T[] | null | undefined): T[] => (Array.isArray(v) ? v : [])
const finite = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0)
function safe<T>(fn: () => T, fallback: T): T {
  try {
    return fn()
  } catch {
    return fallback
  }
}

const KINDS = ['merge', 'loraMerge', 'grid']

/** True when the entry has the shape the page needs at all: a known kind and a job with an id. */
export function entryReadable(e: HistoryEntry | null | undefined): boolean {
  return safe(() => !!e && typeof e === 'object' && KINDS.includes(e.kind) && !!e.job && typeof e.job === 'object' && typeof e.job.id === 'string' && e.job.id !== '', false)
}

export function entryId(e: HistoryEntry): string {
  return `${e?.kind}:${e?.job?.id}`
}

/** createdAt as a finite number (0 when missing or not a number). */
export function entryCreatedAt(e: HistoryEntry): number {
  return safe(() => finite(e.job.createdAt), 0)
}

/** Newest first (stable for equal times); values that are not entries are dropped. */
export function sortEntries(entries: HistoryEntry[]): HistoryEntry[] {
  return asArr(entries)
    .filter((e) => !!e && typeof e === 'object')
    .sort((a, b) => entryCreatedAt(b) - entryCreatedAt(a))
}

const MERGE_KIND_LABEL: Record<MergeRecipe['mode'], string> = {
  weighted: 'Merge',
  addDifference: 'Add difference',
  blocks: 'Per block',
  bake: 'Bake'
}

/** Merge / Bake / Add difference / Per block / LoRA merge / Grid. */
export function entryKindLabel(e: HistoryEntry): string {
  return safe(() => {
    if (e.kind === 'merge') return MERGE_KIND_LABEL[e.job.recipe.mode] ?? 'Merge'
    return e.kind === 'loraMerge' ? 'LoRA merge' : e.kind === 'grid' ? 'Grid' : 'Entry'
  }, 'Entry')
}

const UNKNOWN_STATUS = { label: 'Unknown', tone: 'idle' as const, busy: false }

export function entryStatus(e: HistoryEntry): { label: string; tone: 'ok' | 'warn' | 'danger' | 'idle'; busy: boolean } {
  return safe(() => {
    if (e.kind === 'merge') return JOB_STATUS[e.job.status] ?? UNKNOWN_STATUS
    if (e.kind === 'loraMerge') return LORA_JOB_STATUS[e.job.status] ?? UNKNOWN_STATUS
    return GRID_STATUS[e.job.status] ?? UNKNOWN_STATUS
  }, UNKNOWN_STATUS)
}

const stem = (name: string | null | undefined): string =>
  typeof name === 'string' && name ? splitName(name).file.replace(/\.(safetensors|ckpt|pt|pth|bin|gguf)$/i, '') : '?'
const num = (v: unknown): string => String(round4(finite(v)))

/** The setting a grid sweeps; its value is left out of the merge summary because every column differs. */
export interface SweepOmit {
  kind: 'ratio' | 'multiplier' | 'loraStrength'
  loraIndex: number | null
}

function loraText(r: MergeRecipe, omitIndex: number | null): string {
  return asArr(r.loras)
    .map((l, i) => ({ l, i }))
    .filter(({ l }) => !!l && !!l.name)
    .map(({ l, i }) => {
      if (i === omitIndex) return stem(l.name)
      const clip = r.kind === 'checkpoint' && l.clipStrength !== l.strength ? ` (text ${num(l.clipStrength)})` : ''
      return `${stem(l.name)} × ${num(l.strength)}${clip}`
    })
    .join(', ')
}

/** One line about a merge recipe: its sources and numbers. `omit` leaves a swept setting out. */
export function mergeRecipeSummary(r: MergeRecipe, omit: SweepOmit | null = null): string {
  if (!r || typeof r !== 'object') return ''
  const loras = loraText(r, omit?.kind === 'loraStrength' ? omit.loraIndex : null)
  if (r.mode === 'bake') return loras ? `${stem(r.a)} + ${loras}` : stem(r.a)
  let base: string
  if (r.mode === 'addDifference') base = `${stem(r.a)} + (${stem(r.b)} − ${stem(r.c)})${omit?.kind === 'multiplier' ? '' : ` × ${num(r.multiplier)}`}`
  else if (r.mode === 'blocks') {
    const rest = omit?.kind === 'ratio' ? '' : `, the rest ${num(r.ratio)}`
    base = `${stem(r.a)} / ${stem(r.b)} per block (${Object.keys(r.blocks ?? {}).length} changed${rest})`
  } else if (omit?.kind === 'ratio') base = `${stem(r.a)} + ${stem(r.b)}`
  else base = `${stem(r.a)} × ${num(r.ratio)} + ${stem(r.b)} × ${num(1 - finite(r.ratio))}`
  return loras ? `${base}; LoRAs: ${loras}` : base
}

function gridRows(req: GridRequest | undefined): number {
  return gridRowKind(req ?? ({} as GridRequest)) === 'seeds' ? asArr(req?.seeds).length : asArr(req?.prompts).length
}

function axisName(req: GridRequest | undefined): string {
  const a = req?.axis
  if (!a) return 'Setting'
  if (a.kind === 'ratio') return 'Ratio'
  if (a.kind === 'multiplier') return 'Multiplier'
  const l = a.loraIndex !== null && a.loraIndex !== undefined ? asArr(req?.recipe?.loras)[a.loraIndex] : undefined
  return `${l ? stem(l.name) : 'LoRA'} strength`
}

/** A short recipe summary (sources and numbers) for any entry; '' when it cannot be read. */
export function entrySummary(e: HistoryEntry): string {
  return safe(() => {
    if (e.kind === 'merge') return mergeRecipeSummary(e.job.recipe)
    if (e.kind === 'loraMerge') return loraRecipeSummary(e.job.recipe)
    const req = e.job.request
    const values = asArr(req?.axis?.values)
    const rows = gridRows(req)
    const rowText = `${rows} ${gridRowKind(req) === 'seeds' ? 'seed' : 'prompt'} row${rows === 1 ? '' : 's'}`
    const range = values.length > 0 ? ` ${num(values[0])} to ${num(values[values.length - 1])}` : ''
    const omit = req?.axis ? { kind: req.axis.kind, loraIndex: req.axis.loraIndex ?? null } : null
    return `${axisName(req)}${range} (${values.length} values), ${rowText}. ${mergeRecipeSummary(req?.recipe, omit)}`
  }, '')
}

/** What the entry produced: the saved file, or the grid's size; null when it produced nothing. */
export function entryOutput(e: HistoryEntry): string | null {
  return safe(() => {
    if (e.kind === 'merge') {
      const o = e.job.output
      return o ? (o.movedTo ?? o.wslPath ?? o.relPath ?? null) : null
    }
    if (e.kind === 'loraMerge') return e.job.output?.wslPath ?? null
    const done = asArr(e.job.cells).filter((c) => c?.status === 'done').length
    if (done === 0) return null
    return `${asArr(e.job.request?.axis?.values).length} x ${gridRows(e.job.request)} grid, ${done} image${done === 1 ? '' : 's'}`
  }, null)
}

/** The text a search looks through: names of sources, LoRAs, output and workflow (lower case). */
export function entrySearchText(e: HistoryEntry): string {
  return safe(() => {
    const parts: string[] = []
    const recipe = (r: MergeRecipe | undefined): void => {
      parts.push(r?.a ?? '', r?.b ?? '', r?.c ?? '', r?.outputName ?? '', ...asArr(r?.loras).map((l) => l?.name ?? ''))
    }
    if (e.kind === 'merge') {
      recipe(e.job.recipe)
      parts.push(e.job.output?.relPath ?? '', e.job.output?.wslPath ?? '', e.job.output?.movedTo ?? '')
    } else if (e.kind === 'loraMerge') {
      parts.push(e.job.recipe?.outputName ?? '', ...asArr(e.job.recipe?.inputs).map((i) => i?.name ?? ''), e.job.output?.wslPath ?? '')
    } else {
      recipe(e.job.request?.recipe)
      parts.push(e.job.templateName ?? '')
    }
    parts.push(entryKindLabel(e))
    return parts.filter((p) => typeof p === 'string').join(' ').toLowerCase()
  }, '')
}

export function matchesFilter(e: HistoryEntry, filter: HistoryFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'merges') return e?.kind === 'merge'
  if (filter === 'loraMerges') return e?.kind === 'loraMerge'
  return e?.kind === 'grid'
}

/** Entries of the chosen kind whose names contain every word of the query (case-insensitive). */
export function filterEntries(entries: HistoryEntry[], filter: HistoryFilter, query: string): HistoryEntry[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  return entries.filter((e) => matchesFilter(e, filter) && (words.length === 0 || words.every((w) => entrySearchText(e).includes(w))))
}

export const entryDate = (e: HistoryEntry): string => {
  const t = entryCreatedAt(e)
  return t > 0 ? formatDate(t) : 'unknown date'
}

// ---- change detection (Re-run and the history refresh) ----

/** A string that is equal for equal values (what Re-run compares before it starts). */
export function snapshotOf(value: unknown): string {
  return JSON.stringify(value) ?? ''
}

/** True when the form no longer holds what was reopened. */
export function snapshotChanged(before: string, now: unknown): boolean {
  return before !== snapshotOf(now)
}

/** True when `next` is `prev` with at most other favourites (starring a cell must not rescan grid folders). */
export function onlyFavouritesChanged(prev: GridJob, next: GridJob): boolean {
  return safe(() => snapshotOf({ ...prev, favourites: null }) === snapshotOf({ ...next, favourites: null }), false)
}

/** A recipe with nothing chosen yet (the Merge page as it opens). */
export function isEmptyRecipe(r: MergeRecipe): boolean {
  return !r.a && !r.b && !r.c && asArr(r.loras).length === 0
}

/** True when loading `incoming` into the Merge page would replace a recipe the user may care about
 *  (the output name is ignored: the Merge page keeps its own). */
export function mergePageWillChange(current: MergeRecipe, incoming: MergeRecipe): boolean {
  if (isEmptyRecipe(current)) return false
  return snapshotOf({ ...current, outputName: '' }) !== snapshotOf({ ...incoming, outputName: '' })
}

/** The form rows of a saved LoRA merge recipe; at least two rows, so the form keeps its usual shape. */
export function loraRowsFromRecipe(recipe: LoraMergeRecipe): LoraRow[] {
  const rows = recipe.inputs.slice(0, MAX_LORA_MERGE_INPUTS).map((i) => ({ name: i.name, weight: i.weight }))
  while (rows.length < 2) rows.push({ name: '', weight: 1 })
  return rows
}

const NODES_BY_MODE: Record<MergeRecipe['mode'], string[]> = {
  weighted: ['ModelMergeSimple'],
  addDifference: ['ModelMergeSubtract', 'ModelMergeAdd'],
  blocks: [],
  bake: []
}

export interface MergeStartState {
  recipe: MergeRecipe
  /** ComfyUI reachable, and (own server, or shared queue accepted). */
  connected: boolean
  /** Server capabilities loaded; `nodes[name] === false` means the server lacks that node. */
  nodes: Record<string, boolean> | null
  jobs: Array<{ status: Parameters<typeof jobIsActive>[0] }>
  check: { problems: string[] } | null
  checking: boolean
  checkError: string | null
  starting: boolean
}

/** Why the Merge page's "Merge and save" button is off, or null. Same conditions and order as MergePage. */
export function mergeStartReason(s: MergeStartState): string | null {
  const r = s.recipe
  if (!s.connected || !s.nodes) return 'Connect to ComfyUI on the Server page first.'
  const missing = NODES_BY_MODE[r.mode].filter((n) => s.nodes?.[n] === false)
  if (s.jobs.some((j) => jobIsActive(j.status))) return 'A merge of this app is still running. Wait for it or cancel it.'
  if (r.mode === 'bake' ? !r.a : !r.a || !r.b) return r.mode === 'bake' ? 'Choose A.' : 'Choose A and B.'
  if (r.mode === 'bake' && !r.loras.some((l) => l.name)) return 'Add at least one LoRA to bake.'
  if (r.loras.some((l) => !l.name)) return 'Pick a LoRA in every row or remove the empty ones.'
  if (r.mode === 'addDifference' && !r.c) return 'Choose C.'
  if (r.mode === 'blocks' && !r.blockNode) return 'Choose a block node.'
  if (!validOutputName(r.outputName)) return 'Give the output a valid name.'
  if (missing.length > 0) return `This server has no ${missing.join(' or ')} node.`
  if (s.checking) return 'The files are still being checked.'
  if (s.checkError) return `The files could not be checked: ${s.checkError}. Try again.`
  if (!s.check) return 'The files have not been checked yet.'
  if (s.check.problems.length > 0) return 'Fix the problems in the check first.'
  if (s.starting) return 'Starting.'
  return null
}

export interface RerunContext {
  /** ComfyUI reachable and (own server, or shared queue accepted). */
  connected: boolean
  merges: Array<{ status: Parameters<typeof jobIsActive>[0] }>
  loraMerges: Array<{ status: Parameters<typeof loraJobIsActive>[0] }>
}

/** Why Re-run would be refused before the page's own start action is pressed, or null. Grids are checked on the Sweep
 *  form after Reopen (gridBlockReason), because they need the workflow loaded first. */
export function rerunBlockReason(e: HistoryEntry, ctx: RerunContext): string | null {
  if (e.kind === 'merge') {
    if (ctx.merges.some((j) => jobIsActive(j.status))) return 'A merge of this app is still running. Wait for it or cancel it.'
    if (!ctx.connected) return 'Connect to ComfyUI on the Server page first.'
    if (!validOutputName(e.job.recipe.outputName)) return 'The output name of this merge is not valid. Reopen it and give the output a valid name.'
    return null
  }
  if (e.kind === 'loraMerge') {
    if (ctx.loraMerges.some((j) => loraJobIsActive(j.status))) return 'A LoRA merge is running.'
    if (!validLoraOutputName(e.job.recipe.outputName)) return 'The output name of this merge is not valid. Reopen it and give the output a valid name.'
    return null
  }
  return null
}
