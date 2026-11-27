import { describe, expect, it } from 'vitest'
import {
  axisKeyOf,
  axisOptions,
  cellCount,
  cellCountText,
  clampSteps,
  cellListText,
  doneCellsMissing,
  dropGridImages,
  generateValues,
  gridBlockReason,
  imageKey,
  layoutGrid,
  mergeGridList,
  missingImages,
  parseValues,
  pickShownGrid,
  promptHeader,
  promptsError,
  pruneImages,
  recipeLines,
  recipeReadiness,
  removeGrid,
  resolveAxisOption,
  templateFit,
  upsertGrid,
  validSeed,
  valuesError,
  type GridBlockState
} from '../src/renderer/src/lib/grid'
import { axisLabel, recipeForValue, type GridJob, type GridTemplateInfo } from '../src/shared/grid'
import { DEFAULT_RECIPE, type MergeRecipe } from '../src/shared/merge'

const recipe = (over: Partial<MergeRecipe> = {}): MergeRecipe => ({ ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors', ...over })
const lora = (name: string, strength = 1) => ({ name, strength, clipStrength: strength })
const template = (over: Partial<GridTemplateInfo> = {}): GridTemplateInfo => ({
  path: 'C:\\w\\flow.json',
  name: 'flow.json',
  error: null,
  warnings: [],
  modelLoader: { id: '1', classType: 'UNETLoader' },
  promptInput: { id: '2', input: 'text', classType: 'CLIPTextEncode' },
  seedInputs: [],
  imageOutputs: ['9'],
  defaultPrompt: null,
  ...over
})
const job = (id: string, status: GridJob['status'], createdAt = 1, over: Partial<GridJob> = {}): GridJob => ({
  id,
  request: { recipe: recipe(), axis: { kind: 'ratio', loraIndex: null, values: [0, 1] }, prompts: ['p'], seed: 1, templatePath: 'x' },
  templateName: 'flow.json',
  status,
  serverUrl: 'http://127.0.0.1:8188',
  own: true,
  createdAt,
  finishedAt: null,
  cells: [
    { col: 0, row: 0, status: 'done', file: 'c0_r0.png', error: null, seconds: 1 },
    { col: 1, row: 0, status: 'pending', file: null, error: null, seconds: null }
  ],
  progress: null,
  error: null,
  ...over
})

describe('axis options', () => {
  it('offers ratio for weighted and blocks, multiplier for addDifference', () => {
    expect(axisOptions(recipe({ mode: 'weighted' })).map((o) => o.key)).toEqual(['ratio'])
    expect(axisOptions(recipe({ mode: 'blocks' })).map((o) => o.key)).toEqual(['ratio'])
    expect(axisOptions(recipe({ mode: 'addDifference' })).map((o) => o.key)).toEqual(['multiplier'])
  })
  it('adds one option per named LoRA and none for bake without LoRAs', () => {
    const r = recipe({ mode: 'weighted', loras: [lora('x/my_lora.safetensors'), lora('')] })
    const opts = axisOptions(r)
    expect(opts.map((o) => o.key)).toEqual(['ratio', 'lora:0'])
    expect(opts[1].label).toBe('Strength of my_lora')
    expect(axisOptions(recipe({ mode: 'bake' }))).toEqual([])
    expect(axisOptions(recipe({ mode: 'bake', loras: [lora('l.safetensors')] })).map((o) => o.key)).toEqual(['lora:0'])
  })
  it('falls back to the first option when the key is gone', () => {
    expect(resolveAxisOption(recipe({ mode: 'weighted' }), 'multiplier')?.key).toBe('ratio')
    expect(resolveAxisOption(recipe({ mode: 'bake' }), 'ratio')).toBeNull()
  })
  it('keys an axis', () => {
    expect(axisKeyOf({ kind: 'loraStrength', loraIndex: 2 })).toBe('lora:2')
    expect(axisKeyOf({ kind: 'ratio', loraIndex: null })).toBe('ratio')
  })
})

describe('values', () => {
  it('generates evenly spaced values rounded to 4 decimals', () => {
    expect(generateValues(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(generateValues(0, 1, 4)).toEqual([0, 0.3333, 0.6667, 1])
    expect(generateValues(1, 0, 2)).toEqual([1, 0])
  })
  it('clamps steps to 2..8', () => {
    expect(clampSteps(1)).toBe(2)
    expect(clampSteps(20)).toBe(8)
    expect(clampSteps(Number.NaN)).toBe(2)
    expect(generateValues(0, 1, 99)).toHaveLength(8)
  })
  it('parses comma, space and semicolon lists and rounds', () => {
    expect(parseValues('0, 0.25;0.123456 1').values).toEqual([0, 0.25, 0.1235, 1])
    expect(parseValues('0,5').values).toEqual([0, 5])
    expect(parseValues('0, abc').error).toMatch(/not a number/)
  })
  it('validates count and range per axis kind', () => {
    expect(valuesError('ratio', '0, 0.5, 1')).toBeNull()
    expect(valuesError('ratio', '0.5')).toMatch(/2 to 8/)
    expect(valuesError('ratio', '0,1,2,3,4,5,6,7,8')).toMatch(/2 to 8/)
    expect(valuesError('ratio', '0, 1.5')).toMatch(/outside 0 to 1/)
    expect(valuesError('multiplier', '0, 2')).toBeNull()
    expect(valuesError('multiplier', '0, 2.5')).toMatch(/outside 0 to 2/)
    expect(valuesError('loraStrength', '-10, 10')).toBeNull()
    expect(valuesError('loraStrength', '-11, 1')).toMatch(/outside -10 to 10/)
    expect(valuesError('ratio', '0, x')).toMatch(/not a number/)
  })
})

describe('prompts and seed', () => {
  it('needs 1 to 4 non-empty prompts', () => {
    expect(promptsError([''])).toMatch(/at least one/)
    expect(promptsError(['  ', 'a'])).toBeNull()
    expect(promptsError(['a', 'b', 'c', 'd', 'e'])).toMatch(/at most 4/)
    expect(promptsError(['x'.repeat(2001)])).toMatch(/longer/)
  })
  it('checks the seed', () => {
    expect(validSeed(0)).toBe(true)
    expect(validSeed(2 ** 53 - 1)).toBe(true)
    expect(validSeed(2 ** 53)).toBe(false)
    expect(validSeed(-1)).toBe(false)
    expect(validSeed(1.5)).toBe(false)
  })
  it('counts cells and shortens headers', () => {
    expect(cellCount(5, 2)).toBe(10)
    expect(cellCountText(10)).toBe('10 images')
    expect(cellCountText(1)).toBe('1 image')
    expect(promptHeader('a  b\n c')).toBe('a b c')
    expect(promptHeader('x'.repeat(80)).length).toBe(60)
  })
})

describe('template fit and recipe', () => {
  it('matches loader class and kind', () => {
    expect(templateFit(template(), recipe({ kind: 'diffusion' }))).toBeNull()
    expect(templateFit(template(), recipe({ kind: 'checkpoint' }))).toMatch(/UNETLoader/)
    const ck = template({ modelLoader: { id: '1', classType: 'CheckpointLoaderSimple' } })
    expect(templateFit(ck, recipe({ kind: 'checkpoint' }))).toBeNull()
    expect(templateFit(ck, recipe({ kind: 'diffusion' }))).toMatch(/CheckpointLoaderSimple/)
    expect(templateFit(null, recipe())).toBeNull()
  })
  it('reports what is missing in the recipe', () => {
    expect(recipeReadiness(recipe({ a: '' }))).toMatch(/A and B/)
    expect(recipeReadiness(recipe({ mode: 'addDifference' }))).toMatch(/C/)
    expect(recipeReadiness(recipe({ mode: 'bake', b: '' }))).toMatch(/LoRA/)
    expect(recipeReadiness(recipe({ loras: [lora('')] }))).toMatch(/every row/)
    expect(recipeReadiness(recipe())).toBeNull()
  })
  it('summarises the recipe', () => {
    const lines = recipeLines(recipe({ loras: [lora('f/l.safetensors', 0.8)] }))
    expect(lines.find((l) => l.label === 'A')?.value).toBe('a.safetensors')
    expect(lines.find((l) => l.label === 'Ratio')?.value).toBe('0.5')
    expect(lines.find((l) => l.label === 'LoRAs')?.value).toContain('0.8')
  })
})

describe('start gating', () => {
  const ok = (over: Partial<GridBlockState> = {}): GridBlockState => ({
    connected: true,
    template: template(),
    recipe: recipe(),
    axis: resolveAxisOption(recipe(), 'ratio'),
    valuesText: '0, 0.5, 1',
    prompts: ['a cat'],
    seed: 5,
    grids: [],
    starting: false,
    ...over
  })
  it('can start when everything is valid', () => expect(gridBlockReason(ok())).toBeNull())
  it('blocks, with a reason, in order', () => {
    expect(gridBlockReason(ok({ grids: [job('g', 'running')] }))).toMatch(/still running/)
    expect(gridBlockReason(ok({ connected: false }))).toMatch(/Connect/)
    expect(gridBlockReason(ok({ template: null }))).toMatch(/Choose a workflow/)
    expect(gridBlockReason(ok({ template: template({ error: 'bad' }) }))).toMatch(/Fix the workflow/)
    expect(gridBlockReason(ok({ recipe: recipe({ b: '' }) }))).toMatch(/A and B/)
    expect(gridBlockReason(ok({ recipe: recipe({ kind: 'checkpoint' }) }))).toMatch(/does not fit/)
    expect(gridBlockReason(ok({ axis: null }))).toMatch(/nothing to vary/)
    expect(gridBlockReason(ok({ valuesText: '0' }))).toMatch(/2 to 8/)
    expect(gridBlockReason(ok({ prompts: [' '] }))).toMatch(/prompt/)
    expect(gridBlockReason(ok({ seed: -2 }))).toMatch(/seed/)
    expect(gridBlockReason(ok({ starting: true }))).toBe('Starting.')
  })
  it('does not block on finished grids', () => {
    expect(gridBlockReason(ok({ grids: [job('g', 'done'), job('h', 'error')] }))).toBeNull()
  })
})

describe('grid list merging', () => {
  it('adds newest first and replaces by id', () => {
    let g = upsertGrid([], job('a', 'running', 1))
    g = upsertGrid(g, job('b', 'running', 2))
    expect(g.map((x) => x.id)).toEqual(['b', 'a'])
    g = upsertGrid(g, job('a', 'done', 1))
    expect(g.find((x) => x.id === 'a')?.status).toBe('done')
  })
  it('never replaces a finished grid with a running one', () => {
    const g = upsertGrid([job('a', 'done', 1)], job('a', 'running', 1))
    expect(g[0].status).toBe('done')
  })
  it('merges a listing without losing newer events', () => {
    const cur = [job('a', 'done', 1)]
    const merged = mergeGridList(cur, [job('a', 'running', 1), job('c', 'done', 0)])
    expect(merged.map((x) => `${x.id}:${x.status}`)).toEqual(['a:done', 'c:done'])
  })
  it('removes and picks the shown grid', () => {
    const g = [job('b', 'done', 2), job('a', 'running', 1)]
    expect(removeGrid(g, 'b').map((x) => x.id)).toEqual(['a'])
    expect(pickShownGrid(g, 'b')?.id).toBe('b')
    expect(pickShownGrid(g, 'zzz')?.id).toBe('a')
    expect(pickShownGrid([job('x', 'done')], null)?.id).toBe('x')
    expect(pickShownGrid([], null)).toBeNull()
  })
})

describe('image cache', () => {
  it('finds files still to fetch', () => {
    const j = job('g1', 'running')
    expect(missingImages(j, {})).toEqual(['c0_r0.png'])
    expect(missingImages(j, { [imageKey('g1', 'c0_r0.png')]: 'data:' })).toEqual([])
  })
  it('drops images of grids that are gone', () => {
    const imgs = { 'g1/a.png': 'x', 'g2/b.png': 'y' }
    expect(pruneImages(imgs, [job('g1', 'done')])).toEqual({ 'g1/a.png': 'x' })
    expect(dropGridImages(imgs, 'g1')).toEqual({ 'g2/b.png': 'y' })
  })
})

describe('save preparation', () => {
  it('lists done cells missing from the cache', () => {
    const j = job('g1', 'running', 1, {
      cells: [
        { col: 1, row: 1, status: 'done', file: 'c1_r1.png', error: null, seconds: 1 },
        { col: 0, row: 0, status: 'done', file: 'c0_r0.png', error: null, seconds: 1 },
        { col: 1, row: 0, status: 'running', file: null, error: null, seconds: null }
      ]
    })
    expect(doneCellsMissing(j, {}).map((m) => m.file)).toEqual(['c0_r0.png', 'c1_r1.png'])
    expect(doneCellsMissing(j, { [imageKey('g1', 'c0_r0.png')]: 'data:' }).map((m) => m.file)).toEqual(['c1_r1.png'])
    expect(
      cellListText([
        { col: 1, row: 0 },
        { col: 2, row: 1 }
      ])
    ).toBe('column 2, row 1; column 3, row 2')
  })
})

describe('layout of the saved picture', () => {
  it('keeps natural size when it fits and never exceeds 4096 wide', () => {
    const small = layoutGrid(2, 1, 512, 768)
    expect(small.cellW).toBe(512)
    expect(small.cellH).toBe(768)
    const wide = layoutGrid(8, 4, 1024, 1024)
    expect(wide.width).toBeLessThanOrEqual(4096)
    expect(wide.cellW).toBeLessThan(1024)
    expect(wide.cellH).toBe(wide.cellW)
    expect(wide.height).toBeGreaterThan(wide.cellH * 4)
  })
})

describe('labels', () => {
  it('labels columns and builds the recipe of a column', () => {
    const r = recipe({ loras: [lora('f/my.safetensors', 0.5)] })
    const axis = { kind: 'loraStrength' as const, loraIndex: 0, values: [0, 1] }
    expect(axisLabel(axis, r, 0.8)).toBe('my strength 0.8')
    expect(recipeForValue(r, axis, 0.8).loras[0].strength).toBe(0.8)
    expect(recipeForValue(r, { kind: 'ratio', loraIndex: null, values: [] }, 0.25).ratio).toBe(0.25)
    expect(recipeForValue(r, { kind: 'multiplier', loraIndex: null, values: [] }, 1.5).multiplier).toBe(1.5)
  })
})
