import { describe, expect, it } from 'vitest'
import {
  NAME_MAX,
  addRow,
  buildInspectRequest,
  buildRecipe,
  emptyRows,
  energyText,
  formatInputKind,
  formatPercent,
  formatRankRange,
  formatSize,
  jobFraction,
  loraMergeBlockReason,
  moveRow,
  recipeSummary,
  removeRow,
  setRow,
  suggestLoraOutputName,
  upsertLoraJob,
  validLoraOutputName,
  verifyLine,
  type BlockState,
  type LoraRow
} from '../src/renderer/src/lib/loraMerge'
import { MAX_LORA_MERGE_INPUTS, type LoraMergeCheck, type LoraMergeJob } from '../src/shared/loraMerge'

const row = (name: string, weight = 1): LoraRow => ({ name, weight })
const okCheck: LoraMergeCheck = { inputs: [], problems: [], warnings: [], totalModules: 10, sharedModules: 10, outputRank: { min: 64, max: 64 }, estimatedBytes: 1024 }
const base = (over: Partial<BlockState> = {}): BlockState => ({
  rows: [row('a.safetensors'), row('b.safetensors')],
  check: okCheck,
  checking: false,
  checkError: null,
  jobs: [],
  starting: false,
  outputName: 'a_b',
  ...over
})
const job = (id: string, status: LoraMergeJob['status'], createdAt = 1): LoraMergeJob => ({
  id,
  recipe: { inputs: [], mode: 'concat', rank: 64, dtype: 'source', outputName: 'x' },
  status,
  createdAt,
  finishedAt: null,
  progress: null,
  error: null,
  output: null
})

describe('lora merge rows', () => {
  it('starts with two empty rows', () => {
    expect(emptyRows()).toEqual([row(''), row('')])
  })
  it('adds up to the limit and then returns the same array', () => {
    let rows: LoraRow[] = []
    for (let i = 0; i < MAX_LORA_MERGE_INPUTS + 3; i++) rows = addRow(rows)
    expect(rows).toHaveLength(MAX_LORA_MERGE_INPUTS)
    expect(addRow(rows)).toBe(rows)
  })
  it('is immutable', () => {
    const rows = [row('a'), row('b'), row('c')]
    const copy = JSON.parse(JSON.stringify(rows))
    removeRow(rows, 0)
    moveRow(rows, 0, 1)
    setRow(rows, 1, { weight: 2 })
    expect(rows).toEqual(copy)
  })
  it('moves, removes and sets', () => {
    const rows = [row('a'), row('b'), row('c')]
    expect(moveRow(rows, 0, 1).map((r) => r.name)).toEqual(['b', 'a', 'c'])
    expect(moveRow(rows, 0, -1)).toBe(rows)
    expect(moveRow(rows, 2, 1)).toBe(rows)
    expect(removeRow(rows, 1).map((r) => r.name)).toEqual(['a', 'c'])
    expect(removeRow(rows, 9)).toBe(rows)
    expect(setRow(rows, 1, { weight: -0.5 })[1]).toEqual({ name: 'b', weight: -0.5 })
    expect(setRow(rows, 5, { weight: 1 })).toBe(rows)
  })
})

describe('lora merge name', () => {
  it('joins the stems of the named inputs', () => {
    expect(suggestLoraOutputName([row('x/one.safetensors'), row(''), row('two.safetensors')])).toBe('one_two')
  })
  it('falls back, cleans and limits', () => {
    expect(suggestLoraOutputName(emptyRows())).toBe('merged-lora')
    expect(suggestLoraOutputName([row('a+b (v2).safetensors'), row('c.safetensors')])).toBe('a_b _v2__c')
    const long = suggestLoraOutputName([row('x'.repeat(60) + '.safetensors'), row('y'.repeat(60) + '.safetensors')])
    expect(long.length).toBeLessThanOrEqual(NAME_MAX)
    expect(validLoraOutputName(long)).toBe(true)
  })
  it('validates names', () => {
    expect(validLoraOutputName('ok name-1_2.x')).toBe(true)
    expect(validLoraOutputName('')).toBe(false)
    expect(validLoraOutputName('a/b')).toBe(false)
    expect(validLoraOutputName('x'.repeat(NAME_MAX + 1))).toBe(false)
  })
})

describe('lora merge block reason', () => {
  it('allows a good state', () => {
    expect(loraMergeBlockReason(base())).toBeNull()
  })
  it('gives the reasons', () => {
    expect(loraMergeBlockReason(base({ rows: [row('a'), row('b'), row('')] }))).toBe('Pick a LoRA in every row or remove the empty ones.')
    expect(loraMergeBlockReason(base({ rows: [row('a')] }))).toBe('Pick at least two LoRAs.')
    expect(loraMergeBlockReason(base({ rows: emptyRows() }))).toBe('Pick at least two LoRAs.')
    expect(loraMergeBlockReason(base({ jobs: [{ status: 'running' }] }))).toBe('A LoRA merge is running.')
    expect(loraMergeBlockReason(base({ jobs: [{ status: 'done' }] }))).toBeNull()
    expect(loraMergeBlockReason(base({ checking: true }))).toMatch(/still being checked/)
    expect(loraMergeBlockReason(base({ check: null }))).toMatch(/not been checked/)
    expect(loraMergeBlockReason(base({ checkError: 'boom' }))).toMatch(/boom/)
    expect(loraMergeBlockReason(base({ check: { ...okCheck, problems: ['bad'] } }))).toBe('Fix the problems above first.')
    expect(loraMergeBlockReason(base({ outputName: 'a/b' }))).toMatch(/valid name/)
  })
})

describe('lora merge requests', () => {
  const rows = [row('a.safetensors', 0.5), row(''), row('b.safetensors', -1)]
  it('sends only named inputs', () => {
    expect(buildInspectRequest(rows, { mode: 'shrink', rank: 32, dtype: 'bf16' })).toEqual({ names: ['a.safetensors', 'b.safetensors'], mode: 'shrink', rank: 32, dtype: 'bf16' })
    const r = buildRecipe(rows, { mode: 'concat', rank: 64, dtype: 'source', outputName: ' out ' })
    expect(r.inputs).toEqual([
      { name: 'a.safetensors', weight: 0.5 },
      { name: 'b.safetensors', weight: -1 }
    ])
    expect(r.outputName).toBe('out')
  })
})

describe('lora merge formatting', () => {
  it('formats ranks', () => {
    expect(formatRankRange({ min: 32, max: 32 })).toBe('r32')
    expect(formatRankRange({ min: 16, max: 64 })).toBe('r16–64')
    expect(formatRankRange(null)).toBe('unknown')
    expect(formatInputKind('LoRA', { min: 32, max: 32 })).toBe('LoRA r32')
    expect(formatInputKind('LoKr', null)).toBe('LoKr')
  })
  it('formats sizes and energy', () => {
    expect(formatSize(null)).toBe('unknown')
    expect(formatSize(5 * 1024 ** 2)).toBe('5 MiB')
    expect(formatSize(1.5 * 1024 ** 3)).toBe('1.5 GiB')
    expect(formatPercent(0.987)).toBe('98.7%')
    expect(energyText({ mean: 0.987, min: 0.912 })).toBe('Kept on average 98.7% of each layer (lowest 91.2%)')
  })
  it('words the verify result', () => {
    expect(verifyLine({ ok: true, message: 'fine' })).toEqual({ tone: 'ok', text: 'fine' })
    expect(verifyLine({ ok: false, message: 'x' })).toEqual({ tone: 'warn', text: 'The check after saving found a problem: x' })
  })
  it('summarises a recipe', () => {
    const r = { inputs: [{ name: 'f/a.safetensors', weight: 0.5 }, { name: 'b.safetensors', weight: 1 }], mode: 'shrink' as const, rank: 32, dtype: 'source' as const, outputName: 'x' }
    expect(recipeSummary(r)).toBe('a × 0.5, b × 1 (shrunk to rank 32)')
  })
})

describe('lora merge jobs', () => {
  it('upserts newest first and never un-finishes a job', () => {
    let jobs = upsertLoraJob([], job('a', 'running', 1))
    jobs = upsertLoraJob(jobs, job('b', 'running', 2))
    expect(jobs.map((j) => j.id)).toEqual(['b', 'a'])
    jobs = upsertLoraJob(jobs, job('a', 'done', 1))
    jobs = upsertLoraJob(jobs, job('a', 'running', 1))
    expect(jobs.find((j) => j.id === 'a')?.status).toBe('done')
  })
  it('computes progress', () => {
    expect(jobFraction(null)).toBeNull()
    expect(jobFraction({ done: 0, total: 0 })).toBeNull()
    expect(jobFraction({ done: 5, total: 10 })).toBe(0.5)
    expect(jobFraction({ done: 20, total: 10 })).toBe(1)
  })
})
