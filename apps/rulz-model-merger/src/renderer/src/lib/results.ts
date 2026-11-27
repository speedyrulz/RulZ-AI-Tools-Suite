// Pure helpers for the Results page (no React, no IPC): picking a grid, moving between cells, the A/B selection, stars
// and the zoom/pan maths shared by the compared images.
import { axisLabel, gridRowCount, gridRowLabel, type GridCellRef, type GridJob } from '@shared/grid'
import { axisLabelShort, formatDate, promptHeader } from './grid'

// ---- which grid ----

/** A grid the viewer can show: not running (a running one belongs to the Sweep page). */
export function isViewable(job: GridJob): boolean {
  return job.status !== 'running'
}

/** Grids the picker lists, newest first. */
export function viewableGrids(grids: GridJob[]): GridJob[] {
  return grids.filter(isViewable).sort((a, b) => b.createdAt - a.createdAt)
}

/** The grid to show: the selected one when it is viewable, else the newest viewable one, else null. */
export function defaultGridId(grids: GridJob[], selectedId: string | null): string | null {
  const list = viewableGrids(grids)
  return list.find((g) => g.id === selectedId)?.id ?? list[0]?.id ?? null
}

export function gridDimensions(job: GridJob): { cols: number; rows: number } {
  return { cols: job.request.axis.values.length, rows: gridRowCount(job.request) }
}

/** One line for the picker: date, template, axis, size. */
export function pickerLabel(job: GridJob): string {
  const { cols, rows } = gridDimensions(job)
  return `${formatDate(job.createdAt)} · ${job.templateName} · ${axisLabelShort(job)} · ${cols} x ${rows}`
}

// ---- cells ----

export const sameCell = (a: GridCellRef | null | undefined, b: GridCellRef | null | undefined): boolean =>
  !!a && !!b && a.col === b.col && a.row === b.row

export const cellId = (c: GridCellRef): string => `${c.col}/${c.row}`

/** Label of a column header. */
export function columnLabel(job: GridJob, col: number): string {
  return axisLabel(job.request.axis, job.request.recipe, job.request.axis.values[col] ?? Number.NaN)
}

/** Label of a row: the prompt (shortened) or "Seed 1234". */
export function rowLabel(job: GridJob, row: number, max = 80): string {
  return promptHeader(gridRowLabel(job.request, row), max)
}

/** "Ratio 0.5, Seed 42": names a cell for buttons and the overlay. */
export function cellTitle(job: GridJob, c: GridCellRef): string {
  return `${columnLabel(job, c.col)}, ${rowLabel(job, c.row)}`
}

/** File name for saving one cell image. */
export function cellImageName(job: GridJob, c: GridCellRef): string {
  const day = formatDate(job.createdAt).slice(0, 10)
  return `${axisLabelShort(job).replace(/[^A-Za-z0-9_-]+/g, '_')}_col${c.col + 1}_row${c.row + 1}_${day}.png`
}

export type Direction = 'left' | 'right' | 'up' | 'down'

/** The cell next to `from` in a direction; at an edge it stays where it is. */
export function neighbour(cols: number, rows: number, from: GridCellRef, dir: Direction): GridCellRef {
  const col = dir === 'left' ? from.col - 1 : dir === 'right' ? from.col + 1 : from.col
  const row = dir === 'up' ? from.row - 1 : dir === 'down' ? from.row + 1 : from.row
  if (col < 0 || row < 0 || col >= cols || row >= rows) return { col: from.col, row: from.row }
  return { col, row }
}

/** The direction a keyboard key means, or null. */
export function keyDirection(key: string): Direction | null {
  if (key === 'ArrowLeft') return 'left'
  if (key === 'ArrowRight') return 'right'
  if (key === 'ArrowUp') return 'up'
  if (key === 'ArrowDown') return 'down'
  return null
}

// ---- A/B selection ----

export const MAX_SELECTED = 2

/**
 * Toggles a cell in the selection (first = A, second = B). A cell that is selected is removed; a new one is added and,
 * with MAX_SELECTED cells already picked, the oldest pick drops out.
 */
export function toggleSelection(sel: GridCellRef[], cell: GridCellRef, max = MAX_SELECTED): GridCellRef[] {
  if (sel.some((s) => sameCell(s, cell))) return sel.filter((s) => !sameCell(s, cell))
  const next = [...sel, { col: cell.col, row: cell.row }]
  return next.length > max ? next.slice(next.length - max) : next
}

export type Slot = 'A' | 'B'

/** Puts a cell in slot A or B. When it already sits in the other slot, the two swap. B needs an A first. */
export function assignSlot(sel: GridCellRef[], slot: Slot, cell: GridCellRef): GridCellRef[] {
  const c = { col: cell.col, row: cell.row }
  const a = sel[0]
  const b = sel[1]
  if (slot === 'A') {
    if (b && sameCell(b, c)) return a ? [c, a] : [c]
    return b ? [c, b] : [c]
  }
  if (!a) return [c]
  if (sameCell(a, c)) return [a]
  return [a, c]
}

/** Slot of a cell in the selection, or null. */
export function slotOf(sel: GridCellRef[], cell: GridCellRef): Slot | null {
  if (sameCell(sel[0], cell)) return 'A'
  if (sameCell(sel[1], cell)) return 'B'
  return null
}

/** Drops picks that are not cells of the grid (a different grid was opened). */
export function validSelection(sel: GridCellRef[], cols: number, rows: number): GridCellRef[] {
  return sel.filter((c) => c.col >= 0 && c.row >= 0 && c.col < cols && c.row < rows).slice(0, MAX_SELECTED)
}

// ---- stars ----

/** Favourites sorted by row, then column, without duplicates. */
export function normalizeFavourites(favs: GridCellRef[] | undefined): GridCellRef[] {
  const seen = new Set<string>()
  const out: GridCellRef[] = []
  for (const f of favs ?? []) {
    const id = cellId(f)
    if (seen.has(id)) continue
    seen.add(id)
    out.push({ col: f.col, row: f.row })
  }
  return out.sort((a, b) => a.row - b.row || a.col - b.col)
}

export function isFavourite(favs: GridCellRef[] | undefined, cell: GridCellRef): boolean {
  return (favs ?? []).some((f) => sameCell(f, cell))
}

/** Adds the cell, or removes it when it is starred: the new sorted, unique list. */
export function toggleFavourite(favs: GridCellRef[] | undefined, cell: GridCellRef): GridCellRef[] {
  const list = normalizeFavourites(favs)
  if (isFavourite(list, cell)) return list.filter((f) => !sameCell(f, cell))
  return normalizeFavourites([...list, cell])
}

/** Starred cells that exist in the grid, in row-major order. */
export function starredCells(job: GridJob, favs: GridCellRef[] | undefined = job.favourites): GridCellRef[] {
  const { cols, rows } = gridDimensions(job)
  return normalizeFavourites(favs).filter((f) => f.col >= 0 && f.row >= 0 && f.col < cols && f.row < rows)
}

// ---- zoom and pan ----
// A view is the scale of the picture and where its top-left corner sits, as fractions of the frame (so it does not
// depend on pixel sizes and two frames of different sizes show the same region). At scale s the corner can be moved
// between 1 - s (far edge in view) and 0 (near edge in view).

export interface ZoomView {
  scale: number
  x: number
  y: number
}

export const MIN_SCALE = 1
export const MAX_SCALE = 8
export const RESET_VIEW: ZoomView = { scale: 1, x: 0, y: 0 }

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v))

/** Keeps the scale within limits and the picture covering the frame. */
export function clampView(v: ZoomView): ZoomView {
  const scale = clamp(Number.isFinite(v.scale) ? v.scale : 1, MIN_SCALE, MAX_SCALE)
  const lo = 1 - scale
  const x = clamp(Number.isFinite(v.x) ? v.x : 0, lo, 0)
  const y = clamp(Number.isFinite(v.y) ? v.y : 0, lo, 0)
  // Avoid -0 so equal views compare equal.
  return { scale, x: x === 0 ? 0 : x, y: y === 0 ? 0 : y }
}

/** Zooms by `factor` keeping the point (fx, fy), given as fractions of the frame, under the pointer. */
export function zoomAt(v: ZoomView, factor: number, fx: number, fy: number): ZoomView {
  const from = clampView(v)
  const scale = clamp(from.scale * factor, MIN_SCALE, MAX_SCALE)
  // Picture point under the pointer, in picture fractions.
  const ux = (fx - from.x) / from.scale
  const uy = (fy - from.y) / from.scale
  return clampView({ scale, x: fx - ux * scale, y: fy - uy * scale })
}

/** Moves the picture by (dx, dy) frame fractions. */
export function panBy(v: ZoomView, dx: number, dy: number): ZoomView {
  const from = clampView(v)
  return clampView({ scale: from.scale, x: from.x + dx, y: from.y + dy })
}

/** Zoom factor of one wheel event (deltaY > 0 zooms out). */
export function wheelFactor(deltaY: number): number {
  return Math.exp(-clamp(deltaY, -400, 400) * 0.0018)
}

/** CSS transform of the picture for a view. */
export function viewTransform(v: ZoomView): string {
  const c = clampView(v)
  return `translate(${c.x * 100}%, ${c.y * 100}%) scale(${c.scale})`
}
