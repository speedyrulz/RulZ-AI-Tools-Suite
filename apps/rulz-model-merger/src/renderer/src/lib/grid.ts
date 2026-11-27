// Pure helpers for the Sweep page (no React, no IPC): axes, values, prompts, labels, job merging, image cache, layout.
import {
  GRID_MAX_PROMPTS,
  GRID_MAX_SEEDS,
  GRID_MAX_VALUES,
  GRID_MIN_VALUES,
  GRID_PROMPT_MAX,
  axisLabel,
  gridRowCount,
  gridRowKind,
  type GridAxis,
  type GridAxisKind,
  type GridCell,
  type GridJob,
  type GridRequest,
  type GridRowKind,
  type GridStatus,
  type GridTemplateInfo
} from '@shared/grid'
import type { MergeRecipe } from '@shared/merge'
import { loraSummary, splitName } from './merge'

export const TEMPLATE_STORAGE_KEY = 'rulz-merger.grid.template'
/** Widest composed grid image (pixels). */
export const GRID_IMAGE_MAX_WIDTH = 4096

// ---- axes and values ----

export interface AxisOption {
  /** 'ratio', 'multiplier' or 'lora:<index>'. */
  key: string
  axis: Omit<GridAxis, 'values'>
  label: string
}

export function axisKeyOf(axis: Pick<GridAxis, 'kind' | 'loraIndex'>): string {
  return axis.kind === 'loraStrength' ? `lora:${axis.loraIndex ?? 0}` : axis.kind
}

const stem = (name: string): string => splitName(name).file.replace(/\.safetensors$/i, '')

/** The settings that can be swept for this recipe: only options that change something. */
export function axisOptions(recipe: MergeRecipe): AxisOption[] {
  const out: AxisOption[] = []
  if (recipe.mode === 'weighted' || recipe.mode === 'blocks') out.push({ key: 'ratio', axis: { kind: 'ratio', loraIndex: null }, label: 'Ratio (share of A)' })
  if (recipe.mode === 'addDifference') out.push({ key: 'multiplier', axis: { kind: 'multiplier', loraIndex: null }, label: 'Multiplier' })
  recipe.loras.forEach((l, i) => {
    if (l.name) out.push({ key: `lora:${i}`, axis: { kind: 'loraStrength', loraIndex: i }, label: `Strength of ${stem(l.name)}` })
  })
  return out
}

/** The option for a key, or the first one when the key no longer applies; null when nothing can be swept. */
export function resolveAxisOption(recipe: MergeRecipe, key: string): AxisOption | null {
  const opts = axisOptions(recipe)
  return opts.find((o) => o.key === key) ?? opts[0] ?? null
}

export const AXIS_RANGE: Record<GridAxisKind, { min: number; max: number }> = {
  ratio: { min: 0, max: 1 },
  multiplier: { min: 0, max: 2 },
  loraStrength: { min: -10, max: 10 }
}

/** Starting from / to for a fresh axis. */
export const AXIS_DEFAULT: Record<GridAxisKind, { from: number; to: number }> = {
  ratio: { from: 0, to: 1 },
  multiplier: { from: 0.5, to: 1.5 },
  loraStrength: { from: 0, to: 1 }
}

export const round4 = (v: number): number => Math.round(v * 10000) / 10000

export function clampSteps(steps: number): number {
  if (!Number.isFinite(steps)) return GRID_MIN_VALUES
  return Math.min(GRID_MAX_VALUES, Math.max(GRID_MIN_VALUES, Math.round(steps)))
}

/** `steps` evenly spaced values from `from` to `to`, rounded to 4 decimals. */
export function generateValues(from: number, to: number, steps: number): number[] {
  const n = clampSteps(steps)
  const out: number[] = []
  for (let i = 0; i < n; i++) out.push(round4(from + ((to - from) * i) / (n - 1)))
  return out
}

export function formatValues(values: number[]): string {
  return values.map((v) => String(round4(v))).join(', ')
}

/** Parses "0, 0.25 0.5; 1" into numbers (rounded to 4 decimals). error is set when a part is not a number. */
export function parseValues(text: string): { values: number[]; error: string | null } {
  const parts = text.split(/[,;\s]+/).filter(Boolean)
  const values: number[] = []
  for (const p of parts) {
    const n = Number(p.replace(',', '.'))
    if (!Number.isFinite(n)) return { values, error: `"${p.length > 12 ? p.slice(0, 12) + '...' : p}" is not a number.` }
    values.push(round4(n))
  }
  return { values, error: null }
}

/** Why the value list cannot be used, or null. */
export function valuesError(kind: GridAxisKind, text: string): string | null {
  const { values, error } = parseValues(text)
  if (error) return error
  if (values.length < GRID_MIN_VALUES || values.length > GRID_MAX_VALUES) {
    return `Give ${GRID_MIN_VALUES} to ${GRID_MAX_VALUES} values. You have ${values.length}.`
  }
  const r = AXIS_RANGE[kind]
  const bad = values.find((v) => v < r.min || v > r.max)
  if (bad !== undefined) return `${bad} is outside ${r.min} to ${r.max}.`
  return null
}

// ---- prompts and seed ----

/** Prompt texts that count (trimmed, non-empty). */
export function activePrompts(prompts: string[]): string[] {
  return prompts.map((p) => p.trim()).filter(Boolean)
}

export function promptsError(prompts: string[]): string | null {
  const act = activePrompts(prompts)
  if (act.length === 0) return 'Write at least one prompt.'
  if (act.length > GRID_MAX_PROMPTS) return `Use at most ${GRID_MAX_PROMPTS} prompts.`
  if (act.some((p) => p.length > GRID_PROMPT_MAX)) return `A prompt is longer than ${GRID_PROMPT_MAX} characters.`
  return null
}

export function validSeed(seed: number): boolean {
  return Number.isInteger(seed) && seed >= 0 && seed <= Number.MAX_SAFE_INTEGER
}

export function randomSeed(rand: () => number = Math.random): number {
  return Math.floor(rand() * 2 ** 32)
}

export function cellCount(valueCount: number, promptCount: number): number {
  return valueCount * promptCount
}

export function cellCountText(n: number): string {
  return `${n} image${n === 1 ? '' : 's'}`
}

/** Row header: the first `max` characters of the prompt. */
export function promptHeader(prompt: string, max = 60): string {
  const one = prompt.replace(/\s+/g, ' ').trim()
  return one.length > max ? `${one.slice(0, max - 1)}…` : one
}

// ---- seed rows ----

/** Why the seed list cannot be used (seed rows), or null. */
export function seedsError(seeds: number[]): string | null {
  if (seeds.length === 0) return 'Give at least one seed.'
  if (seeds.length > GRID_MAX_SEEDS) return `Use at most ${GRID_MAX_SEEDS} seeds.`
  if (seeds.some((s) => !validSeed(s))) return 'Every seed must be a whole number from 0 up.'
  if (new Set(seeds).size !== seeds.length) return 'Every seed must be different.'
  return null
}

/** Why the rows cannot be used: prompt rows need 1 to 4 prompts, seed rows need one prompt and 1 to 4 seeds. */
export function rowsError(kind: GridRowKind, prompts: string[], seeds: number[]): string | null {
  if (kind === 'seeds') {
    const pe = promptsError(prompts.slice(0, 1))
    return pe ?? seedsError(seeds)
  }
  return promptsError(prompts)
}

/** Number of rows the form would give: valid prompts, or seeds. */
export function rowCountOf(kind: GridRowKind, prompts: string[], seeds: number[]): number {
  return kind === 'seeds' ? seeds.length : activePrompts(prompts).length
}

/** `count` distinct random seeds that are not in `exclude`. */
export function randomSeeds(count: number, rand: () => number = Math.random, exclude: number[] = []): number[] {
  const used = new Set(exclude)
  const out: number[] = []
  while (out.length < count) {
    const s = randomSeed(rand)
    if (used.has(s)) continue
    used.add(s)
    out.push(s)
  }
  return out
}

/** Adds one new distinct random seed (unchanged at GRID_MAX_SEEDS). */
export function addSeed(seeds: number[], rand: () => number = Math.random): number[] {
  if (seeds.length >= GRID_MAX_SEEDS) return seeds
  return [...seeds, ...randomSeeds(1, rand, seeds)]
}

/** Removes seed i, never the last one. */
export function removeSeed(seeds: number[], i: number): number[] {
  if (seeds.length <= 1 || i < 0 || i >= seeds.length) return seeds
  return seeds.filter((_, k) => k !== i)
}

export function setSeedAt(seeds: number[], i: number, value: number): number[] {
  if (i < 0 || i >= seeds.length) return seeds
  return seeds.map((s, k) => (k === i ? value : s))
}

/** "seed 5" for prompt rows, "seeds 5, 9" for seed rows. */
export function seedText(req: GridRequest): string {
  if (gridRowKind(req) === 'seeds') {
    const seeds = req.seeds ?? []
    return `seed${seeds.length === 1 ? '' : 's'} ${seeds.join(', ')}`
  }
  return `seed ${req.seed}`
}

export interface GridFormState {
  recipe: MergeRecipe
  axis: AxisOption
  valuesText: string
  rowKind: GridRowKind
  prompts: string[]
  seeds: number[]
  seed: number
  templatePath: string
}

/** The request the Start button sends (the caller has checked gridBlockReason). Seed rows send one prompt and the seeds. */
export function buildGridRequest(s: GridFormState): GridRequest {
  const axis = { ...s.axis.axis, values: parseValues(s.valuesText).values }
  if (s.rowKind === 'seeds') {
    const prompt = (s.prompts[0] ?? '').trim()
    return { recipe: s.recipe, axis, rowKind: 'seeds', prompts: [prompt], seed: s.seeds[0] ?? s.seed, seeds: s.seeds.slice(), templatePath: s.templatePath }
  }
  return { recipe: s.recipe, axis, rowKind: 'prompts', prompts: activePrompts(s.prompts), seed: s.seed, templatePath: s.templatePath }
}

/** What the Sweep form holds after Reopen of a request. `seeds` is null when the request has none (the form keeps its own). */
export interface GridFormFields {
  axisKey: string
  from: number
  to: number
  steps: number
  valuesText: string
  rowKind: GridRowKind
  prompts: string[]
  seeds: number[] | null
  seed: number
}

export function formFromRequest(req: GridRequest): GridFormFields {
  const values = req.axis.values
  const kind = gridRowKind(req)
  const seeds = kind === 'seeds' ? (req.seeds ?? []).slice(0, GRID_MAX_SEEDS) : null
  const prompts = req.prompts.slice(0, kind === 'seeds' ? 1 : GRID_MAX_PROMPTS)
  const d = AXIS_DEFAULT[req.axis.kind]
  return {
    axisKey: axisKeyOf(req.axis),
    from: values[0] ?? d.from,
    to: values[values.length - 1] ?? d.to,
    steps: clampSteps(values.length),
    valuesText: formatValues(values),
    rowKind: kind,
    prompts: prompts.length > 0 ? prompts : [''],
    seeds: seeds && seeds.length > 0 ? seeds : null,
    seed: kind === 'seeds' ? (seeds?.[0] ?? req.seed) : req.seed
  }
}

// ---- template and recipe ----

/** Why the template cannot render this recipe, or null. */
export function templateFit(template: GridTemplateInfo | null, recipe: MergeRecipe): string | null {
  if (!template || template.error || !template.modelLoader) return null
  const c = template.modelLoader.classType
  if (c === 'UNETLoader' && recipe.kind !== 'diffusion') {
    return 'This workflow loads a diffusion model (UNETLoader), but the merge on the Merge page uses checkpoints. Choose a workflow with a checkpoint loader or change the merge.'
  }
  if (c === 'CheckpointLoaderSimple' && recipe.kind !== 'checkpoint') {
    return 'This workflow loads a checkpoint (CheckpointLoaderSimple), but the merge on the Merge page uses diffusion models. Choose a workflow with a diffusion model loader or change the merge.'
  }
  return null
}

/** Why the recipe on the Merge page is not ready for a grid, or null. */
export function recipeReadiness(recipe: MergeRecipe): string | null {
  if (recipe.mode === 'bake') {
    if (!recipe.a) return 'Choose A on the Merge page.'
    if (!recipe.loras.some((l) => l.name)) return 'Add at least one LoRA on the Merge page.'
  } else {
    if (!recipe.a || !recipe.b) return 'Choose A and B on the Merge page.'
    if (recipe.mode === 'addDifference' && !recipe.c) return 'Choose C on the Merge page.'
    if (recipe.mode === 'blocks' && !recipe.blockNode) return 'Choose a block node on the Merge page.'
  }
  if (recipe.loras.some((l) => !l.name)) return 'Pick a LoRA in every row on the Merge page, or remove the empty ones.'
  return null
}

const MODE_LABEL: Record<MergeRecipe['mode'], string> = {
  weighted: 'Weighted',
  addDifference: 'Add difference',
  blocks: 'Per block',
  bake: 'Bake LoRAs'
}

export interface RecipeSummaryLine {
  label: string
  value: string
}

/** Read-only lines for the Merge card. */
export function recipeLines(recipe: MergeRecipe): RecipeSummaryLine[] {
  const lines: RecipeSummaryLine[] = [
    { label: 'Kind', value: recipe.kind === 'diffusion' ? 'Diffusion models' : 'Checkpoints' },
    { label: 'Mode', value: MODE_LABEL[recipe.mode] },
    { label: 'A', value: recipe.a ? splitName(recipe.a).file : 'not chosen' }
  ]
  if (recipe.mode !== 'bake') lines.push({ label: 'B', value: recipe.b ? splitName(recipe.b).file : 'not chosen' })
  if (recipe.mode === 'addDifference') lines.push({ label: 'C', value: recipe.c ? splitName(recipe.c).file : 'not chosen' })
  if (recipe.mode === 'weighted' || recipe.mode === 'blocks') lines.push({ label: 'Ratio', value: String(recipe.ratio) })
  if (recipe.mode === 'addDifference') lines.push({ label: 'Multiplier', value: String(recipe.multiplier) })
  const loras = loraSummary(recipe.loras, recipe.kind)
  if (loras) lines.push({ label: 'LoRAs', value: loras })
  return lines
}

// ---- start gating ----

export interface GridBlockState {
  connected: boolean
  template: GridTemplateInfo | null
  recipe: MergeRecipe
  axis: AxisOption | null
  valuesText: string
  prompts: string[]
  seed: number
  /** Absent means prompt rows. */
  rowKind?: GridRowKind
  seeds?: number[]
  grids: GridJob[]
  starting: boolean
}

export function gridIsActive(status: GridStatus): boolean {
  return status === 'running'
}

/** Why Start sweep is off, most useful reason first; null when it can start. */
export function gridBlockReason(s: GridBlockState): string | null {
  if (s.grids.some((g) => gridIsActive(g.status))) return 'A grid is still running. Wait for it or cancel it.'
  if (!s.connected) return 'Connect to ComfyUI on the Server page first.'
  if (!s.template) return 'Choose a workflow first.'
  if (s.template.error) return 'Fix the workflow first.'
  const ready = recipeReadiness(s.recipe)
  if (ready) return ready
  const fit = templateFit(s.template, s.recipe)
  if (fit) return 'The workflow does not fit the merge.'
  if (!s.axis) return 'The merge has nothing to vary. Use a ratio, a multiplier or a LoRA.'
  const ve = valuesError(s.axis.axis.kind, s.valuesText)
  if (ve) return ve
  const kind = s.rowKind ?? 'prompts'
  const pe = rowsError(kind, s.prompts, s.seeds ?? [])
  if (pe) return pe
  if (kind === 'prompts' && !validSeed(s.seed)) return 'The seed must be a whole number from 0 up.'
  if (s.starting) return 'Starting.'
  return null
}

// ---- jobs ----

export const GRID_STATUS: Record<GridStatus, { label: string; tone: 'ok' | 'warn' | 'danger' | 'idle'; busy: boolean }> = {
  running: { label: 'Running', tone: 'warn', busy: true },
  done: { label: 'Done', tone: 'ok', busy: false },
  error: { label: 'Failed', tone: 'danger', busy: false },
  cancelled: { label: 'Cancelled', tone: 'idle', busy: false }
}

/** Which of two versions of the same grid to keep: a finished one is never replaced by a running one. */
export function newerGrid(old: GridJob, incoming: GridJob): GridJob {
  const oldDone = old.status !== 'running'
  const newDone = incoming.status !== 'running'
  if (oldDone && !newDone) return old
  return incoming
}

/** Replaces or adds a grid by id, newest first. */
export function upsertGrid(grids: GridJob[], job: GridJob): GridJob[] {
  const i = grids.findIndex((g) => g.id === job.id)
  if (i >= 0) {
    const next = grids.slice()
    next[i] = newerGrid(grids[i], job)
    return next
  }
  return [job, ...grids].sort((a, b) => b.createdAt - a.createdAt)
}

/** A full list from grid:list merged with what events already told us. */
export function mergeGridList(current: GridJob[], listed: GridJob[]): GridJob[] {
  let next = current
  for (const j of listed) next = upsertGrid(next, j)
  return next
}

export function removeGrid(grids: GridJob[], id: string): GridJob[] {
  return grids.filter((g) => g.id !== id)
}

/** The grid to show: the picked one if it still exists, else the running one, else the newest. */
export function pickShownGrid(grids: GridJob[], selectedId: string | null): GridJob | null {
  return grids.find((g) => g.id === selectedId) ?? grids.find((g) => g.status === 'running') ?? grids[0] ?? null
}

export function cellAt(job: GridJob, col: number, row: number): GridCell | undefined {
  return job.cells.find((c) => c.col === col && c.row === row)
}

export function gridDims(job: GridJob): { cols: number; rows: number } {
  return { cols: job.request.axis.values.length, rows: gridRowCount(job.request) }
}

/** Fraction 0..1 of the running cell, null when unknown. */
export function cellFraction(p: GridJob['progress']): number | null {
  if (!p || !(p.max > 0)) return null
  return Math.min(1, Math.max(0, p.value / p.max))
}

export function cellsDone(job: GridJob): number {
  return job.cells.filter((c) => c.status === 'done').length
}

/** "Ratio", "Multiplier" or "<lora> strength" without a value. */
export function axisLabelShort(job: GridJob): string {
  return axisLabel(job.request.axis, job.request.recipe, Number.NaN).replace(/ NaN$/, '')
}

export function formatDate(ms: number): string {
  const d = new Date(ms)
  const p = (n: number): string => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

// ---- image cache ----

export const imageKey = (gridId: string, file: string): string => `${gridId}/${file}`

/** The cache without images of grids that are gone (deleted or no longer listed). */
export function pruneImages(images: Record<string, string>, grids: GridJob[]): Record<string, string> {
  const ids = new Set(grids.map((g) => g.id))
  const next: Record<string, string> = {}
  for (const [k, v] of Object.entries(images)) if (ids.has(k.slice(0, k.indexOf('/')))) next[k] = v
  return next
}

export function dropGridImages(images: Record<string, string>, gridId: string): Record<string, string> {
  const next: Record<string, string> = {}
  for (const [k, v] of Object.entries(images)) if (!k.startsWith(`${gridId}/`)) next[k] = v
  return next
}

/** Files of a grid that are done and not yet cached. */
export function missingImages(job: GridJob, images: Record<string, string>): string[] {
  return job.cells.filter((c) => c.status === 'done' && c.file && !(imageKey(job.id, c.file) in images)).map((c) => c.file as string)
}

/** Done cells whose image is not cached yet (column, row, file), in row-major order. */
export function doneCellsMissing(job: GridJob, images: Record<string, string>): Array<{ col: number; row: number; file: string }> {
  return job.cells
    .filter((c) => c.status === 'done' && c.file && !(imageKey(job.id, c.file) in images))
    .map((c) => ({ col: c.col, row: c.row, file: c.file as string }))
    .sort((a, b) => a.row - b.row || a.col - b.col)
}

/** "column 2, row 1; column 3, row 1" for the cells that could not be fetched. */
export function cellListText(cells: Array<{ col: number; row: number }>): string {
  return cells.map((c) => `column ${c.col + 1}, row ${c.row + 1}`).join('; ')
}

// ---- composed image ----

export interface GridLayout {
  pad: number
  gap: number
  labelW: number
  headerH: number
  cellW: number
  cellH: number
  width: number
  height: number
}

/** Sizes of the composed picture: cells scaled down (never up) so the whole picture is at most `maxWidth` wide. */
export function layoutGrid(cols: number, rows: number, naturalW: number, naturalH: number, maxWidth = GRID_IMAGE_MAX_WIDTH): GridLayout {
  const pad = 24
  const gap = 8
  const labelW = 240
  const headerH = 56
  const avail = maxWidth - pad * 2 - labelW - gap * (cols - 1) - gap
  const cellW = Math.max(16, Math.min(naturalW, Math.floor(avail / cols)))
  const cellH = Math.max(16, Math.round((naturalH * cellW) / naturalW))
  return {
    pad,
    gap,
    labelW,
    headerH,
    cellW,
    cellH,
    width: pad * 2 + labelW + gap + cols * cellW + gap * (cols - 1),
    height: pad * 2 + headerH + rows * cellH + gap * (rows - 1)
  }
}

export function gridImageName(job: GridJob): string {
  const day = formatDate(job.createdAt).slice(0, 10)
  return `grid_${axisLabelShort(job).replace(/[^A-Za-z0-9_-]+/g, '_')}_${day}.png`
}

// ---- remembered template ----

export function rememberTemplatePath(path: string | null): void {
  try {
    if (path) localStorage.setItem(TEMPLATE_STORAGE_KEY, path)
    else localStorage.removeItem(TEMPLATE_STORAGE_KEY)
  } catch {
    /* storage may be unavailable */
  }
}

export function recalledTemplatePath(): string | null {
  try {
    return localStorage.getItem(TEMPLATE_STORAGE_KEY) || null
  } catch {
    return null
  }
}
