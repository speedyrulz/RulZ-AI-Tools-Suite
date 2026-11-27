// Sample grids (milestone 6): render a merge at several values of one setting, for 1 to 4 prompts, with the user's own
// ComfyUI workflow as the template. Nothing is saved as a model file: each cell builds the merge inside the prompt
// (loaders -> merge nodes -> LoRAs) and feeds it to the template's sampler in place of the template's own model loader.
import type { MergeRecipe } from './merge'

export const GRID_MIN_VALUES = 2
export const GRID_MAX_VALUES = 8
export const GRID_MAX_PROMPTS = 4
/** Longest prompt text accepted (characters). */
export const GRID_PROMPT_MAX = 2000

/**
 * The swept setting.
 * - ratio: MergeRecipe.ratio (weighted, blocks: the default share; values 0..1).
 * - multiplier: MergeRecipe.multiplier (addDifference only; values 0..2).
 * - loraStrength: MergeRecipe.loras[loraIndex].strength (any mode with LoRAs; values -10..10). For checkpoints the
 *   clip strength is left as it is in the recipe.
 */
export type GridAxisKind = 'ratio' | 'multiplier' | 'loraStrength'

export interface GridAxis {
  kind: GridAxisKind
  /** loraStrength only: index into recipe.loras. */
  loraIndex: number | null
  /** GRID_MIN_VALUES..GRID_MAX_VALUES values, in column order (rounded to 4 decimals by the app). */
  values: number[]
}

/** What the app found in a template workflow (API format, as saved by ComfyUI's "Export (API)"). */
export interface GridTemplateInfo {
  /** Windows path of the JSON file. */
  path: string
  /** File name for display. */
  name: string
  /** null when the template is usable. */
  error: string | null
  warnings: string[]
  /** Node id and class of the model loader the merge replaces (UNETLoader or CheckpointLoaderSimple), or null. */
  modelLoader: { id: string; classType: string } | null
  /** Node id + input name that receives the positive prompt text, or null. */
  promptInput: { id: string; input: string; classType: string } | null
  /** Node ids + input names holding a seed (seed / noise_seed), all set to the grid's seed. */
  seedInputs: Array<{ id: string; input: string }>
  /** Image output nodes (SaveImage / PreviewImage); SaveImage nodes are turned into PreviewImage for grid cells. */
  imageOutputs: string[]
  /** The template's own prompt text (shown as the first prompt by default), or null. */
  defaultPrompt: string | null
}

/**
 * What the rows are.
 * - prompts: one row per prompt (1..GRID_MAX_PROMPTS), every cell uses `seed`.
 * - seeds: one row per seed in `seeds` (1..GRID_MAX_SEEDS), every cell uses `prompts[0]` (exactly one prompt).
 * Grids saved before 0.6.0 have no `rowKind`: read them as 'prompts'.
 */
export type GridRowKind = 'prompts' | 'seeds'

export const GRID_MAX_SEEDS = 4

export interface GridRequest {
  /** The merge (as on the Merge page). mode 'bake' is allowed (A + LoRAs). outputName is ignored. */
  recipe: MergeRecipe
  axis: GridAxis
  /** Absent means 'prompts' (grids from 0.5.0). */
  rowKind?: GridRowKind
  /** rowKind 'prompts': 1..GRID_MAX_PROMPTS non-empty prompts, rows in order. rowKind 'seeds': exactly 1 prompt. */
  prompts: string[]
  /** rowKind 'prompts': the seed of every cell. rowKind 'seeds': ignored (kept for display). 0..2^53-1 (integer). */
  seed: number
  /** rowKind 'seeds' only: 1..GRID_MAX_SEEDS distinct seeds, rows in order (integers 0..2^53-1). */
  seeds?: number[]
  /** Windows path of the template JSON (must have loaded without error). */
  templatePath: string
}

export type GridStatus = 'running' | 'done' | 'error' | 'cancelled'

export interface GridCell {
  /** Column (value index) and row (prompt or seed index). */
  col: number
  row: number
  status: 'pending' | 'running' | 'done' | 'error' | 'cancelled'
  /** File name of the cell image inside the grid folder, once done. */
  file: string | null
  error: string | null
  /** Seconds the cell took on the server (queue to done), once done. */
  seconds: number | null
}

export interface GridJob {
  id: string
  request: GridRequest
  /** Template file name (the template itself is copied into the grid folder as template.json). */
  templateName: string
  status: GridStatus
  serverUrl: string
  own: boolean
  createdAt: number
  finishedAt: number | null
  /** values.length * gridRowCount(request) cells, row-major (row 0 first). */
  cells: GridCell[]
  /** Progress of the running cell from the websocket. */
  progress: { node: string | null; value: number; max: number } | null
  error: string | null
  /** Cells the user starred on the Results page (absent in grids from 0.5.0 = none). Saved in grid.json. */
  favourites?: GridCellRef[]
}

export interface GridCellRef {
  col: number
  row: number
}

/** 'prompts' for requests without rowKind (0.5.0 grids). */
export function gridRowKind(req: GridRequest): GridRowKind {
  return req.rowKind === 'seeds' ? 'seeds' : 'prompts'
}

/** Number of rows: prompts or seeds. */
export function gridRowCount(req: GridRequest): number {
  return gridRowKind(req) === 'seeds' ? (req.seeds ?? []).length : req.prompts.length
}

/** Prompt of a row. */
export function gridCellPrompt(req: GridRequest, row: number): string {
  return gridRowKind(req) === 'seeds' ? (req.prompts[0] ?? '') : (req.prompts[row] ?? '')
}

/** Seed of a row. */
export function gridCellSeed(req: GridRequest, row: number): number {
  return gridRowKind(req) === 'seeds' ? (req.seeds?.[row] ?? req.seed) : req.seed
}

/** Row label for the grid: the prompt, or "Seed 1234". */
export function gridRowLabel(req: GridRequest, row: number): string {
  return gridRowKind(req) === 'seeds' ? `Seed ${gridCellSeed(req, row)}` : gridCellPrompt(req, row)
}

/** grid.json in each grid folder: the finished (or stopped) job, kept across sessions. */
export interface GridRecord {
  app: 'RulZ Model Merger'
  appVersion: string
  serverVersion: string | null
  job: GridJob
}

/** Label of an axis for column headers, e.g. "Ratio 0.25", "my_lora strength 0.8". */
export function axisLabel(axis: GridAxis, recipe: MergeRecipe, value: number): string {
  const v = String(Math.round(value * 10000) / 10000)
  if (axis.kind === 'ratio') return `Ratio ${v}`
  if (axis.kind === 'multiplier') return `Multiplier ${v}`
  const l = axis.loraIndex !== null ? recipe.loras[axis.loraIndex] : undefined
  const name = l ? (l.name.split('/').pop() ?? l.name).replace(/\.safetensors$/i, '') : 'LoRA'
  return `${name} strength ${v}`
}

/** The recipe of one column: the base recipe with the axis value applied. */
export function recipeForValue(recipe: MergeRecipe, axis: GridAxis, value: number): MergeRecipe {
  if (axis.kind === 'ratio') return { ...recipe, ratio: value }
  if (axis.kind === 'multiplier') return { ...recipe, multiplier: value }
  return {
    ...recipe,
    loras: recipe.loras.map((l, i) => (i === axis.loraIndex ? { ...l, strength: value } : l))
  }
}
