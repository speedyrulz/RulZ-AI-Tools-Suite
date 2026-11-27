import { describe, expect, it } from 'vitest'
import {
  blockGroupName,
  filterNames,
  formatElapsed,
  formulaText,
  groupAverage,
  groupBlockInputs,
  splitName,
  upsertJob,
  validOutputName,
  verifyText
} from '../src/renderer/src/lib/merge'
import type { MergeJob } from '../src/shared/merge'

describe('merge ui helpers', () => {
  it('splits folder and file', () => {
    expect(splitName('flux/dev/model.safetensors')).toEqual({ folder: 'flux/dev', file: 'model.safetensors' })
    expect(splitName('a\\b.safetensors')).toEqual({ folder: 'a', file: 'b.safetensors' })
    expect(splitName('x.safetensors')).toEqual({ folder: '', file: 'x.safetensors' })
  })

  it('filters by all words', () => {
    const l = ['flux/Dev-fp8.safetensors', 'sdxl/base.safetensors']
    expect(filterNames(l, 'FLUX fp8')).toEqual([l[0]])
    expect(filterNames(l, '')).toEqual(l)
    expect(filterNames(l, 'nope')).toEqual([])
  })

  it('writes the formula', () => {
    expect(formulaText({ mode: 'weighted', ratio: 0.7, multiplier: 1, blocks: {} })).toBe('A × 0.70 + B × 0.30')
    expect(formulaText({ mode: 'addDifference', ratio: 0.5, multiplier: 1, blocks: {} })).toBe('A + (B − C) × 1.00')
  })

  it('groups block inputs', () => {
    expect(blockGroupName('double_blocks.3.')).toBe('double_blocks')
    expect(blockGroupName('input_blocks.0.')).toBe('input_blocks')
    expect(blockGroupName('img_in.')).toBe('img_in')
    const g = groupBlockInputs(['img_in.', 'double_blocks.0.', 'double_blocks.1.', 'single_blocks.0.', 'single_blocks.1.', 'final_layer.'])
    expect(g.map((x) => x.name)).toEqual(['double_blocks', 'single_blocks', 'other'])
    expect(g[2].inputs).toEqual(['img_in.', 'final_layer.'])
  })

  it('averages a group with the ratio as fallback', () => {
    expect(groupAverage({ a: 1 }, ['a', 'b'], 0.5)).toBe(0.75)
  })

  it('validates names', () => {
    expect(validOutputName('my merge-1.0')).toBe(true)
    expect(validOutputName('')).toBe(false)
    expect(validOutputName('a/b')).toBe(false)
  })

  it('formats elapsed time', () => {
    expect(formatElapsed(65_000)).toBe('1:05')
    expect(formatElapsed(3_725_000)).toBe('1:02:05')
  })

  it('upserts jobs newest first', () => {
    const j = (id: string, createdAt: number): MergeJob => ({ id, createdAt }) as MergeJob
    const list = upsertJob([j('a', 1)], j('b', 2))
    expect(list.map((x) => x.id)).toEqual(['b', 'a'])
    expect(upsertJob(list, j('a', 1)).length).toBe(2)
  })

  it('words the verify result', () => {
    expect(verifyText({ ok: true, tensorCount: 1284, recipeInHeader: true, message: '' })).toBe('File checked: 1,284 layers, recipe stored')
    expect(verifyText({ ok: false, tensorCount: 0, recipeInHeader: false, message: 'Keys missing.' })).toBe('Keys missing.')
  })
})

describe('job versions', () => {
  const mk = (status: MergeJob['status'], extra: Partial<MergeJob> = {}): MergeJob =>
    ({ id: 'a', createdAt: 1, finishedAt: null, output: null, status, ...extra }) as MergeJob

  it('never replaces a finished job with an unfinished one', () => {
    const done = mk('done', { finishedAt: 5 })
    expect(upsertJob([done], mk('running'))[0]).toBe(done)
    expect(upsertJob([mk('running')], done)[0]).toBe(done)
  })

  it('prefers the moved or later finished version', () => {
    const moved = mk('done', { finishedAt: 5, output: { movedTo: '/x' } as MergeJob['output'] })
    const plain = mk('done', { finishedAt: 9 })
    expect(upsertJob([moved], plain)[0]).toBe(moved)
    expect(upsertJob([plain], moved)[0]).toBe(moved)
    const later = mk('error', { finishedAt: 10 })
    expect(upsertJob([plain], later)[0]).toBe(later)
  })
})
