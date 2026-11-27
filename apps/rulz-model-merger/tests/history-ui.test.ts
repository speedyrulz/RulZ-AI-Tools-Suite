import { describe, expect, it } from 'vitest'
import type { HistoryEntry } from '../src/shared/history'
import type { GridJob } from '../src/shared/grid'
import type { LoraMergeJob } from '../src/shared/loraMerge'
import { DEFAULT_RECIPE, type MergeJob, type MergeRecipe } from '../src/shared/merge'
import {
  entryDate,
  entryId,
  entryReadable,
  isEmptyRecipe,
  mergePageWillChange,
  onlyFavouritesChanged,
  snapshotChanged,
  snapshotOf,
  entryKindLabel,
  entryOutput,
  entryStatus,
  entrySummary,
  filterEntries,
  loraRowsFromRecipe,
  mergeRecipeSummary,
  mergeStartReason,
  rerunBlockReason,
  type MergeStartState,
  sortEntries
} from '../src/renderer/src/lib/history'

const recipe = (over: Partial<MergeRecipe> = {}): MergeRecipe => ({ ...DEFAULT_RECIPE, a: 'f/alpha.safetensors', b: 'beta.safetensors', ...over })
const mergeJob = (id: string, over: Partial<MergeRecipe> = {}, createdAt = 1, status: MergeJob['status'] = 'done', output: MergeJob['output'] = null): MergeJob => ({
  id,
  recipe: recipe(over),
  status,
  serverUrl: 'x',
  own: true,
  promptId: null,
  createdAt,
  finishedAt: createdAt + 1,
  progress: null,
  output,
  error: null
})
const loraJob = (id: string, createdAt = 1): LoraMergeJob => ({
  id,
  recipe: { inputs: [{ name: 'x/one.safetensors', weight: 0.5 }, { name: 'two.safetensors', weight: 1 }], mode: 'concat', rank: 64, dtype: 'source', outputName: 'mix' },
  status: 'done',
  createdAt,
  finishedAt: createdAt + 1,
  progress: null,
  error: null,
  output: { wslPath: '/m/loras/merged/mix.safetensors', loraName: 'merged/mix.safetensors', sizeBytes: 1, modules: 2, rank: { min: 8, max: 8 }, energyKept: null, verify: { ok: true, message: '' }, sidecarPath: null }
})
const gridJob = (id: string, createdAt = 1): GridJob => ({
  id,
  request: { recipe: recipe(), axis: { kind: 'ratio', loraIndex: null, values: [0, 0.5, 1] }, rowKind: 'seeds', prompts: ['cat'], seeds: [1, 2], seed: 1, templatePath: 'T' },
  templateName: 'flow.json',
  status: 'done',
  serverUrl: 'x',
  own: true,
  createdAt,
  finishedAt: null,
  cells: [
    { col: 0, row: 0, status: 'done', file: 'a.png', error: null, seconds: 1 },
    { col: 1, row: 0, status: 'error', file: null, error: 'x', seconds: null }
  ],
  progress: null,
  error: null
})
const m = (j: MergeJob): HistoryEntry => ({ kind: 'merge', job: j })
const l = (j: LoraMergeJob): HistoryEntry => ({ kind: 'loraMerge', job: j })
const g = (j: GridJob): HistoryEntry => ({ kind: 'grid', job: j })

describe('labels and status', () => {
  it('names the kind', () => {
    expect(entryKindLabel(m(mergeJob('a')))).toBe('Merge')
    expect(entryKindLabel(m(mergeJob('a', { mode: 'bake' })))).toBe('Bake')
    expect(entryKindLabel(m(mergeJob('a', { mode: 'addDifference' })))).toBe('Add difference')
    expect(entryKindLabel(m(mergeJob('a', { mode: 'blocks' })))).toBe('Per block')
    expect(entryKindLabel(l(loraJob('b')))).toBe('LoRA merge')
    expect(entryKindLabel(g(gridJob('c')))).toBe('Grid')
  })
  it('gives a status word for each kind', () => {
    expect(entryStatus(m(mergeJob('a', {}, 1, 'error'))).label).toBe('Failed')
    expect(entryStatus(l(loraJob('b'))).label).toBe('Done')
    expect(entryStatus(g(gridJob('c'))).tone).toBe('ok')
  })
  it('builds ids that differ across kinds', () => {
    expect(entryId(m(mergeJob('x')))).not.toBe(entryId(l(loraJob('x'))))
  })
})

describe('summaries and output', () => {
  it('summarises each merge mode with sources and numbers', () => {
    expect(mergeRecipeSummary(recipe({ ratio: 0.25 }))).toBe('alpha × 0.25 + beta × 0.75')
    expect(mergeRecipeSummary(recipe({ mode: 'addDifference', c: 'c.safetensors', multiplier: 1.5 }))).toBe('alpha + (beta − c) × 1.5')
    expect(mergeRecipeSummary(recipe({ mode: 'bake', loras: [{ name: 'l/x.safetensors', strength: 0.8, clipStrength: 0.8 }] }))).toBe('alpha + x × 0.8')
    expect(mergeRecipeSummary(recipe({ mode: 'blocks', blocks: { a: 1, b: 0 } }))).toContain('2 changed')
  })
  it('summarises LoRA merges and grids', () => {
    expect(entrySummary(l(loraJob('b')))).toContain('one × 0.5')
    const s = entrySummary(g(gridJob('c')))
    expect(s).toContain('Ratio 0 to 1 (3 values), 2 seed rows')
    expect(s).toContain('alpha')
  })
  it('names what was produced', () => {
    const out = { relPath: 'r.safetensors', wslPath: '/w/r.safetensors', sizeBytes: 1, sidecarPath: null, verify: null, movedTo: null }
    expect(entryOutput(m(mergeJob('a', {}, 1, 'done', out)))).toBe('/w/r.safetensors')
    expect(entryOutput(m(mergeJob('a', {}, 1, 'done', { ...out, movedTo: '/models/r.safetensors' })))).toBe('/models/r.safetensors')
    expect(entryOutput(m(mergeJob('a', {}, 1, 'error')))).toBeNull()
    expect(entryOutput(l(loraJob('b')))).toBe('/m/loras/merged/mix.safetensors')
    expect(entryOutput(g(gridJob('c')))).toBe('3 x 2 grid, 1 image')
  })
})

describe('filter and search', () => {
  const all = [m(mergeJob('a', {}, 1)), l(loraJob('b', 3)), g(gridJob('c', 2))]
  it('sorts newest first', () => {
    expect(sortEntries(all).map((e) => e.job.id)).toEqual(['b', 'c', 'a'])
  })
  it('filters by kind', () => {
    expect(filterEntries(all, 'all', '')).toHaveLength(3)
    expect(filterEntries(all, 'merges', '').map((e) => e.job.id)).toEqual(['a'])
    expect(filterEntries(all, 'loraMerges', '').map((e) => e.job.id)).toEqual(['b'])
    expect(filterEntries(all, 'grids', '').map((e) => e.job.id)).toEqual(['c'])
  })
  it('searches names, every word, any case', () => {
    expect(filterEntries(all, 'all', 'ALPHA').map((e) => e.job.id)).toEqual(['a', 'c'])
    expect(filterEntries(all, 'all', 'alpha flow').map((e) => e.job.id)).toEqual(['c'])
    expect(filterEntries(all, 'all', 'mix one').map((e) => e.job.id)).toEqual(['b'])
    expect(filterEntries(all, 'grids', 'mix')).toEqual([])
    expect(filterEntries(all, 'all', '   ')).toHaveLength(3)
  })
})

describe('summaries leave the swept setting out', () => {
  const lora = (name: string, strength: number) => ({ name, strength, clipStrength: strength })
  const gridWith = (axis: GridJob['request']['axis'], r: Partial<MergeRecipe>): HistoryEntry => {
    const j = gridJob('s')
    return g({ ...j, request: { ...j.request, axis, recipe: recipe(r) } })
  }
  it('bake lists the LoRAs once', () => {
    expect(mergeRecipeSummary(recipe({ mode: 'bake', a: 'harbor.safetensors', loras: [lora('w.safetensors', 0.8), lora('g.safetensors', 0.6)] }))).toBe('harbor + w × 0.8, g × 0.6')
    expect(mergeRecipeSummary(recipe({ mode: 'bake' }))).toBe('alpha')
  })
  it('ratio sweep: just A + B', () => {
    const s = entrySummary(gridWith({ kind: 'ratio', loraIndex: null, values: [0.4, 0.6, 0.8] }, { ratio: 0.6 }))
    expect(s).toBe('Ratio 0.4 to 0.8 (3 values), 2 seed rows. alpha + beta')
  })
  it('multiplier sweep omits the multiplier', () => {
    const s = entrySummary(gridWith({ kind: 'multiplier', loraIndex: null, values: [0.5, 1] }, { mode: 'addDifference', c: 'c.safetensors', multiplier: 1.5 }))
    expect(s).toContain('alpha + (beta − c)')
    expect(s).not.toContain('1.5')
  })
  it('LoRA-strength sweep omits only that LoRA strength', () => {
    const s = entrySummary(gridWith({ kind: 'loraStrength', loraIndex: 1, values: [0, 1] }, { loras: [lora('a/one.safetensors', 0.3), lora('two.safetensors', 0.9)] }))
    expect(s).toContain('two strength 0 to 1')
    expect(s).toContain('LoRAs: one × 0.3, two')
    expect(s).not.toContain('0.9')
  })
  it('a plain merge keeps its numbers', () => {
    expect(mergeRecipeSummary(recipe({ ratio: 0.4 }))).toBe('alpha × 0.4 + beta × 0.6')
  })
})

describe('broken entries do not throw', () => {
  const bad = [
    { kind: 'merge', job: { id: 'a', createdAt: Number.NaN } },
    { kind: 'merge', job: { id: 'a', recipe: { mode: 'weighted', a: 'x.safetensors' }, status: 'weird' } },
    { kind: 'loraMerge', job: { id: 'b', recipe: {} } },
    { kind: 'grid', job: { id: 'c', request: { axis: {}, recipe: {} }, cells: null } },
    { kind: 'grid', job: { id: 'c', request: {} } },
    { kind: 'what', job: null },
    null
  ] as unknown as HistoryEntry[]
  it('reads every helper without throwing', () => {
    for (const e of bad) {
      expect(() => {
        entryKindLabel(e)
        entryStatus(e)
        entrySummary(e)
        entryOutput(e)
        filterEntries([e], 'all', 'x')
        entryId(e)
        entryDate(e)
      }).not.toThrow()
    }
    expect(entryStatus(bad[1]).label).toBe('Unknown')
    expect(entryDate(bad[0])).toBe('unknown date')
    expect(entrySummary(bad[1])).toContain('x')
  })
  it('sorts with missing or non-finite times and drops non-entries', () => {
    const sorted = sortEntries([...bad, m(mergeJob('new', {}, 50))])
    expect(sorted[0].job.id).toBe('new')
    expect(sorted).toHaveLength(bad.length) // the null is dropped, the good one added
  })
  it('knows which entries are readable', () => {
    expect(entryReadable(m(mergeJob('a')))).toBe(true)
    expect(entryReadable(bad[0])).toBe(true)
    expect(entryReadable(bad[5])).toBe(false)
    expect(entryReadable(bad[6])).toBe(false)
    expect(entryReadable({ kind: 'merge', job: { id: '' } } as unknown as HistoryEntry)).toBe(false)
  })
})

describe('change detection', () => {
  it('spots an edited form', () => {
    const before = snapshotOf({ recipe: recipe(), n: 1 })
    expect(snapshotChanged(before, { recipe: recipe(), n: 1 })).toBe(false)
    expect(snapshotChanged(before, { recipe: recipe({ ratio: 0.9 }), n: 1 })).toBe(true)
  })
  it('ignores a grid update that only changed favourites', () => {
    const a = gridJob('x')
    expect(onlyFavouritesChanged(a, { ...a, favourites: [{ col: 0, row: 0 }] })).toBe(true)
    expect(onlyFavouritesChanged(a, { ...a, status: 'error' })).toBe(false)
    expect(onlyFavouritesChanged(a, { ...a, cells: [] })).toBe(false)
  })
  it('flags a Merge page replacement only when it holds something different', () => {
    expect(mergePageWillChange(recipe({ a: '', b: '' }), recipe())).toBe(false)
    expect(mergePageWillChange(recipe(), recipe({ outputName: 'other' }))).toBe(false)
    expect(mergePageWillChange(recipe(), recipe({ ratio: 0.1 }))).toBe(true)
    expect(isEmptyRecipe(recipe({ a: '', b: '' }))).toBe(true)
  })
})

describe('merge start gating (same conditions as the Merge button)', () => {
  const ok = (over: Partial<MergeStartState> = {}): MergeStartState => ({
    recipe: recipe(),
    connected: true,
    nodes: { ModelMergeSimple: true },
    jobs: [],
    check: { problems: [] },
    checking: false,
    checkError: null,
    starting: false,
    ...over
  })
  it('passes when connected, checked and valid', () => expect(mergeStartReason(ok())).toBeNull())
  it('refuses, with a reason, in the button order', () => {
    expect(mergeStartReason(ok({ connected: false }))).toMatch(/Connect/)
    expect(mergeStartReason(ok({ nodes: null }))).toMatch(/Connect/)
    expect(mergeStartReason(ok({ jobs: [{ status: 'queued' }] }))).toMatch(/still running/)
    expect(mergeStartReason(ok({ recipe: recipe({ b: '' }) }))).toBe('Choose A and B.')
    expect(mergeStartReason(ok({ recipe: recipe({ mode: 'bake', a: '' }) }))).toBe('Choose A.')
    expect(mergeStartReason(ok({ recipe: recipe({ mode: 'bake' }) }))).toMatch(/at least one LoRA/)
    expect(mergeStartReason(ok({ recipe: recipe({ loras: [{ name: '', strength: 1, clipStrength: 1 }] }) }))).toMatch(/every row/)
    expect(mergeStartReason(ok({ recipe: recipe({ mode: 'addDifference' }), nodes: {} }))).toBe('Choose C.')
    expect(mergeStartReason(ok({ recipe: recipe({ mode: 'blocks' }) }))).toMatch(/block node/)
    expect(mergeStartReason(ok({ recipe: recipe({ outputName: 'a/b' }) }))).toMatch(/valid name/)
    expect(mergeStartReason(ok({ nodes: { ModelMergeSimple: false } }))).toMatch(/no ModelMergeSimple/)
    expect(mergeStartReason(ok({ checking: true }))).toMatch(/still being checked/)
    expect(mergeStartReason(ok({ checkError: 'boom' }))).toMatch(/boom/)
    expect(mergeStartReason(ok({ check: null }))).toMatch(/not been checked/)
    expect(mergeStartReason(ok({ check: { problems: ['x'] } }))).toMatch(/Fix the problems/)
    expect(mergeStartReason(ok({ starting: true }))).toBe('Starting.')
  })
})

describe('reopen and re-run helpers', () => {
  it('turns a LoRA merge recipe into form rows with at least two rows', () => {
    expect(loraRowsFromRecipe(loraJob('b').recipe)).toEqual([
      { name: 'x/one.safetensors', weight: 0.5 },
      { name: 'two.safetensors', weight: 1 }
    ])
    expect(loraRowsFromRecipe({ ...loraJob('b').recipe, inputs: [] })).toHaveLength(2)
  })
  it('refuses a re-run that cannot start', () => {
    const ctx = { connected: true, merges: [], loraMerges: [] }
    expect(rerunBlockReason(m(mergeJob('a')), ctx)).toBeNull()
    expect(rerunBlockReason(m(mergeJob('a')), { ...ctx, connected: false })).toMatch(/Connect/)
    expect(rerunBlockReason(m(mergeJob('a')), { ...ctx, merges: [{ status: 'running' }] })).toMatch(/still running/)
    expect(rerunBlockReason(m(mergeJob('a', { outputName: 'a/b' })), ctx)).toMatch(/not valid/)
    expect(rerunBlockReason(l(loraJob('b')), { ...ctx, connected: false })).toBeNull()
    expect(rerunBlockReason(l(loraJob('b')), { ...ctx, loraMerges: [{ status: 'saving' }] })).toMatch(/running/)
    expect(rerunBlockReason(g(gridJob('c')), ctx)).toBeNull()
  })
})
