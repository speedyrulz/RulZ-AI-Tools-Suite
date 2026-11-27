import { describe, expect, it } from 'vitest'
import type { GridCell, GridJob } from '../src/shared/grid'
import {
  MAX_SCALE,
  MIN_SCALE,
  RESET_VIEW,
  assignSlot,
  cellImageName,
  clampView,
  cellTitle,
  defaultGridId,
  keyDirection,
  neighbour,
  normalizeFavourites,
  panBy,
  rowLabel,
  slotOf,
  starredCells,
  toggleFavourite,
  toggleSelection,
  validSelection,
  viewTransform,
  viewableGrids,
  wheelFactor,
  zoomAt
} from '../src/renderer/src/lib/results'

function job(id: string, status: GridJob['status'], createdAt: number, extra: Partial<GridJob> = {}): GridJob {
  const cells: GridCell[] = []
  for (let row = 0; row < 2; row++)
    for (let col = 0; col < 3; col++) cells.push({ col, row, status: 'done', file: `c${col}_${row}.png`, error: null, seconds: 1 })
  return {
    id,
    request: {
      recipe: { loras: [] } as never,
      axis: { kind: 'ratio', loraIndex: null, values: [0, 0.5, 1] },
      prompts: ['a red fox in the snow', 'a blue bird'],
      seed: 7,
      templatePath: 'x.json'
    },
    templateName: 'flux.json',
    status,
    serverUrl: 'http://127.0.0.1:8188',
    own: true,
    createdAt,
    finishedAt: createdAt + 1,
    cells,
    progress: null,
    error: null,
    ...extra
  }
}

describe('default grid', () => {
  const grids = [job('old', 'done', 1), job('run', 'running', 9), job('new', 'cancelled', 5)]
  it('lists only grids that are not running, newest first', () => {
    expect(viewableGrids(grids).map((g) => g.id)).toEqual(['new', 'old'])
  })
  it('keeps the selected grid when it can be shown', () => {
    expect(defaultGridId(grids, 'old')).toBe('old')
  })
  it('falls back to the newest finished grid for no, unknown or running selection', () => {
    expect(defaultGridId(grids, null)).toBe('new')
    expect(defaultGridId(grids, 'gone')).toBe('new')
    expect(defaultGridId(grids, 'run')).toBe('new')
  })
  it('is null when nothing is finished', () => {
    expect(defaultGridId([], null)).toBeNull()
    expect(defaultGridId([job('run', 'running', 1)], null)).toBeNull()
  })
})

describe('neighbour navigation', () => {
  it('moves one cell in each direction', () => {
    const c = { col: 1, row: 1 }
    expect(neighbour(3, 3, c, 'left')).toEqual({ col: 0, row: 1 })
    expect(neighbour(3, 3, c, 'right')).toEqual({ col: 2, row: 1 })
    expect(neighbour(3, 3, c, 'up')).toEqual({ col: 1, row: 0 })
    expect(neighbour(3, 3, c, 'down')).toEqual({ col: 1, row: 2 })
  })
  it('stays put at the edges', () => {
    expect(neighbour(3, 2, { col: 0, row: 0 }, 'left')).toEqual({ col: 0, row: 0 })
    expect(neighbour(3, 2, { col: 0, row: 0 }, 'up')).toEqual({ col: 0, row: 0 })
    expect(neighbour(3, 2, { col: 2, row: 1 }, 'right')).toEqual({ col: 2, row: 1 })
    expect(neighbour(3, 2, { col: 2, row: 1 }, 'down')).toEqual({ col: 2, row: 1 })
  })
  it('maps arrow keys to directions', () => {
    expect(keyDirection('ArrowLeft')).toBe('left')
    expect(keyDirection('ArrowDown')).toBe('down')
    expect(keyDirection('a')).toBeNull()
  })
})

describe('selection of two cells', () => {
  const a = { col: 0, row: 0 }
  const b = { col: 1, row: 0 }
  const c = { col: 2, row: 1 }
  it('adds and removes by toggling', () => {
    expect(toggleSelection([], a)).toEqual([a])
    expect(toggleSelection([a], b)).toEqual([a, b])
    expect(toggleSelection([a, b], a)).toEqual([b])
  })
  it('never holds more than two: the oldest pick drops out', () => {
    expect(toggleSelection([a, b], c)).toEqual([b, c])
  })
  it('gives slots A and B in pick order', () => {
    expect(slotOf([a, b], a)).toBe('A')
    expect(slotOf([a, b], b)).toBe('B')
    expect(slotOf([a, b], c)).toBeNull()
  })
  it('assigns a slot and swaps when the cell is in the other slot', () => {
    expect(assignSlot([a, b], 'A', c)).toEqual([c, b])
    expect(assignSlot([a, b], 'B', c)).toEqual([a, c])
    expect(assignSlot([a, b], 'A', b)).toEqual([b, a])
    expect(assignSlot([a], 'B', b)).toEqual([a, b])
    expect(assignSlot([a], 'B', a)).toEqual([a])
    expect(assignSlot([], 'B', a)).toEqual([a])
  })
  it('drops picks outside the grid', () => {
    expect(validSelection([a, { col: 5, row: 0 }], 3, 2)).toEqual([a])
    expect(validSelection([{ col: 0, row: -1 }], 3, 2)).toEqual([])
  })
})

describe('favourites', () => {
  it('adds a star and keeps the list sorted by row then column', () => {
    let f = toggleFavourite(undefined, { col: 2, row: 1 })
    f = toggleFavourite(f, { col: 0, row: 0 })
    f = toggleFavourite(f, { col: 1, row: 1 })
    expect(f).toEqual([
      { col: 0, row: 0 },
      { col: 1, row: 1 },
      { col: 2, row: 1 }
    ])
  })
  it('removes a star that is set', () => {
    expect(toggleFavourite([{ col: 0, row: 0 }, { col: 1, row: 0 }], { col: 0, row: 0 })).toEqual([{ col: 1, row: 0 }])
  })
  it('removes duplicates from a saved list', () => {
    expect(normalizeFavourites([{ col: 1, row: 0 }, { col: 1, row: 0 }, { col: 0, row: 0 }])).toEqual([
      { col: 0, row: 0 },
      { col: 1, row: 0 }
    ])
  })
  it('lists only starred cells that exist in the grid', () => {
    const j = job('g', 'done', 1, { favourites: [{ col: 9, row: 0 }, { col: 2, row: 1 }, { col: 0, row: 0 }] })
    expect(starredCells(j)).toEqual([
      { col: 0, row: 0 },
      { col: 2, row: 1 }
    ])
  })
})

describe('zoom and pan', () => {
  it('keeps the scale within limits and the picture covering the frame', () => {
    expect(clampView({ scale: 20, x: 0, y: 0 }).scale).toBe(MAX_SCALE)
    expect(clampView({ scale: 0.2, x: -1, y: -1 })).toEqual(RESET_VIEW)
    expect(clampView({ scale: 2, x: -5, y: 3 })).toEqual({ scale: 2, x: -1, y: 0 })
    expect(clampView({ scale: Number.NaN, x: Number.NaN, y: 0 })).toEqual(RESET_VIEW)
  })
  it('zooms at the pointer: the picture point under it does not move', () => {
    const fx = 0.3
    const fy = 0.6
    const v = zoomAt(RESET_VIEW, 2, fx, fy)
    expect(v.scale).toBe(2)
    // Picture point under the pointer before (fx, fy) and after ((fx - x) / scale).
    expect((fx - v.x) / v.scale).toBeCloseTo(fx, 10)
    expect((fy - v.y) / v.scale).toBeCloseTo(fy, 10)
  })
  it('keeps the point fixed on a second zoom too', () => {
    const v1 = zoomAt(RESET_VIEW, 2, 0.5, 0.5)
    const pt = (0.7 - v1.x) / v1.scale
    const v2 = zoomAt(v1, 1.5, 0.7, 0.4)
    expect((0.7 - v2.x) / v2.scale).toBeCloseTo(pt, 10)
  })
  it('zooming in at a corner stays inside the frame', () => {
    const v = zoomAt(RESET_VIEW, 8, 1, 1)
    expect(v).toEqual({ scale: 8, x: -7, y: -7 })
    expect(zoomAt(RESET_VIEW, 8, 0, 0)).toEqual({ scale: 8, x: 0, y: 0 })
  })
  it('zooming out to 1 resets the position', () => {
    const z = zoomAt(RESET_VIEW, 4, 0.8, 0.8)
    expect(zoomAt(z, 0.01, 0.5, 0.5)).toEqual(RESET_VIEW)
    expect(MIN_SCALE).toBe(1)
  })
  it('pans within bounds and not at all at scale 1', () => {
    expect(panBy(RESET_VIEW, 0.5, 0.5)).toEqual(RESET_VIEW)
    const z: { scale: number; x: number; y: number } = { scale: 2, x: -0.5, y: -0.5 }
    expect(panBy(z, 0.2, -0.2)).toEqual({ scale: 2, x: -0.3, y: -0.7 })
    expect(panBy(z, 5, 5)).toEqual({ scale: 2, x: 0, y: 0 })
    expect(panBy(z, -5, -5)).toEqual({ scale: 2, x: -1, y: -1 })
  })
  it('two frames given the same view show the same region (sizes do not matter)', () => {
    const v = zoomAt(RESET_VIEW, 3, 0.25, 0.75)
    expect(viewTransform(v)).toBe(viewTransform({ ...v }))
    expect(viewTransform(v)).toContain('scale(3)')
  })
  it('wheel up zooms in, wheel down zooms out, and one event is bounded', () => {
    expect(wheelFactor(-100)).toBeGreaterThan(1)
    expect(wheelFactor(100)).toBeLessThan(1)
    expect(wheelFactor(100000)).toBeCloseTo(wheelFactor(400), 10)
  })
})

describe('labels', () => {
  const prompts = job('p', 'done', Date.UTC(2026, 10, 27, 12))
  const seeds = job('s', 'done', Date.UTC(2026, 10, 27, 12), {
    request: { ...prompts.request, rowKind: 'seeds', prompts: ['a red fox'], seeds: [11, 22] }
  })
  it('names rows by prompt or by seed', () => {
    expect(rowLabel(prompts, 1)).toBe('a blue bird')
    expect(rowLabel(seeds, 1)).toBe('Seed 22')
  })
  it('lists only starred cells from a given (pending) list', () => {
    expect(starredCells(prompts, [{ col: 1, row: 1 }])).toEqual([{ col: 1, row: 1 }])
  })
  it('names a cell and its image file', () => {
    expect(cellTitle(prompts, { col: 1, row: 0 })).toBe('Ratio 0.5, a red fox in the snow')
    expect(cellImageName(prompts, { col: 1, row: 0 })).toMatch(/^Ratio_col2_row1_\d{4}-\d{2}-\d{2}\.png$/)
  })
})
