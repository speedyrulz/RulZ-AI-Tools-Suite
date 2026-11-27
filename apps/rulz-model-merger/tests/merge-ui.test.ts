import { describe, expect, it } from 'vitest'
import {
  addLora,
  blockGroupName,
  filterNames,
  formatElapsed,
  formulaText,
  groupAverage,
  groupBlockInputs,
  loraSummary,
  loraVerifyText,
  moveLora,
  removeLora,
  setLora,
  splitName,
  suggestOutputName,
  upsertJob,
  validOutputName,
  verifyText
} from '../src/renderer/src/lib/merge'
import { MAX_LORAS, type LoraEntry, type MergeJob } from '../src/shared/merge'

describe('merge ui helpers', () => {
  it('splits folder and file', () => {
    expect(splitName('flux/dev/model.safetensors')).toEqual({
      folder: 'flux/dev',
      file: 'model.safetensors'
    })
    expect(splitName('a\\b.safetensors')).toEqual({
      folder: 'a',
      file: 'b.safetensors'
    })
    expect(splitName('x.safetensors')).toEqual({
      folder: '',
      file: 'x.safetensors'
    })
  })

  it('filters by all words', () => {
    const l = ['flux/Dev-fp8.safetensors', 'sdxl/base.safetensors']
    expect(filterNames(l, 'FLUX fp8')).toEqual([l[0]])
    expect(filterNames(l, '')).toEqual(l)
    expect(filterNames(l, 'nope')).toEqual([])
  })

  it('writes the formula', () => {
    expect(formulaText({ mode: 'weighted', ratio: 0.7, multiplier: 1, blocks: {} })).toBe('A × 0.70 + B × 0.30')
    expect(
      formulaText({
        mode: 'addDifference',
        ratio: 0.5,
        multiplier: 1,
        blocks: {}
      })
    ).toBe('A + (B − C) × 1.00')
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
    expect(
      verifyText({
        ok: true,
        tensorCount: 1284,
        recipeInHeader: true,
        message: ''
      })
    ).toBe('File checked: 1,284 layers, recipe stored')
    expect(
      verifyText({
        ok: false,
        tensorCount: 0,
        recipeInHeader: false,
        message: 'Keys missing.'
      })
    ).toBe('Keys missing.')
  })
})

describe('job versions', () => {
  const mk = (status: MergeJob['status'], extra: Partial<MergeJob> = {}): MergeJob =>
    ({
      id: 'a',
      createdAt: 1,
      finishedAt: null,
      output: null,
      status,
      ...extra
    }) as MergeJob

  it('never replaces a finished job with an unfinished one', () => {
    const done = mk('done', { finishedAt: 5 })
    expect(upsertJob([done], mk('running'))[0]).toBe(done)
    expect(upsertJob([mk('running')], done)[0]).toBe(done)
  })

  it('prefers the moved or later finished version', () => {
    const moved = mk('done', {
      finishedAt: 5,
      output: { movedTo: '/x' } as MergeJob['output']
    })
    const plain = mk('done', { finishedAt: 9 })
    expect(upsertJob([moved], plain)[0]).toBe(moved)
    expect(upsertJob([plain], moved)[0]).toBe(moved)
    const later = mk('error', { finishedAt: 10 })
    expect(upsertJob([plain], later)[0]).toBe(later)
  })
})

describe('lora list helpers', () => {
  const L = (name: string, strength = 1): LoraEntry => ({
    name,
    strength,
    clipStrength: 1
  })

  it('adds an empty row with strength 1 and stops at the limit', () => {
    const one = addLora([])
    expect(one).toEqual([{ name: '', strength: 1, clipStrength: 1 }])
    let list: LoraEntry[] = []
    for (let i = 0; i < MAX_LORAS + 3; i++) list = addLora(list)
    expect(list.length).toBe(MAX_LORAS)
  })

  it('never changes the list it was given', () => {
    const list = [L('a'), L('b')]
    const frozen = JSON.stringify(list)
    moveLora(list, 0, 1)
    removeLora(list, 0)
    setLora(list, 1, { strength: 0.5 })
    addLora(list)
    expect(JSON.stringify(list)).toBe(frozen)
  })

  it('removes, moves and patches rows', () => {
    const list = [L('a'), L('b'), L('c')]
    expect(removeLora(list, 1).map((l) => l.name)).toEqual(['a', 'c'])
    expect(moveLora(list, 0, 1).map((l) => l.name)).toEqual(['b', 'a', 'c'])
    expect(moveLora(list, 2, -1).map((l) => l.name)).toEqual(['a', 'c', 'b'])
    expect(moveLora(list, 0, -1)).toBe(list)
    expect(moveLora(list, 2, 1)).toBe(list)
    expect(setLora(list, 1, { strength: -0.5 })[1]).toEqual({
      name: 'b',
      strength: -0.5,
      clipStrength: 1
    })
    expect(setLora(list, 9, { strength: 2 })).toBe(list)
  })

  it('suggests an output name that the main process keeps as it is', () => {
    const name = suggestOutputName({
      a: 'moss/moss_image_base.safetensors',
      loras: [L('x/moss-2500.safetensors'), L(''), L('Other LoRA.safetensors')]
    })
    expect(name).toBe('moss_image_base_moss-2500_Other LoRA')
    expect(validOutputName(name)).toBe(true)
    expect(suggestOutputName({ a: '', loras: [] })).toBe('merged')
    expect(validOutputName(suggestOutputName({ a: 'we!rd+name.safetensors', loras: [] }))).toBe(true)
  })

  it('summarises LoRAs for a job', () => {
    expect(loraSummary([L('f/a.safetensors', 0.8), L('')], 'diffusion')).toBe('a × 0.8')
    expect(loraSummary([{ name: 'a.safetensors', strength: 1, clipStrength: 0.5 }], 'checkpoint')).toBe('a × 1 (text 0.5)')
  })

  it('words the LoRA result of a finished job', () => {
    const v = (n: number | null, te = 0) => ({
      loraKeysNotLoaded: n,
      loraTextEncoderKeysSkipped: te
    })
    expect(loraVerifyText(v(null), true)).toBeNull()
    expect(loraVerifyText(v(0), false)).toBeNull()
    expect(loraVerifyText(v(0), true)).toEqual({
      tone: 'ok',
      text: 'LoRAs: all weights applied'
    })
    expect(loraVerifyText(v(3), true)).toEqual({
      tone: 'warn',
      text: '3 LoRA weights not applied'
    })
    expect(loraVerifyText(v(1), true)?.text).toBe('1 LoRA weight not applied')
  })
})
