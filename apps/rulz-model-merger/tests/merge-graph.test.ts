import { describe, expect, it } from 'vitest'
import { DEFAULT_RECIPE, type MergeRecipe } from '@shared/merge'
import { KNOWN_NODES, type ServerCapabilities } from '@shared/types'
import { buildMergeGraph, completeBlocks, sanitizeOutputName, validateRecipe } from '../src/main/merge/graph'

function recipe(over: Partial<MergeRecipe>): MergeRecipe {
  return { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors', outputName: 'my merge', ...over }
}

function caps(over: Partial<ServerCapabilities> = {}): ServerCapabilities {
  const nodes: Record<string, boolean> = {}
  for (const n of KNOWN_NODES) nodes[n] = true
  return {
    checkpoints: ['a.safetensors', 'b.safetensors', 'c.safetensors'],
    diffusionModels: ['a.safetensors', 'b.safetensors', 'c.safetensors'],
    loras: [],
    textEncoders: [],
    vaes: [],
    nodes,
    blockMergeNodes: ['ModelMergeFakeFlux'],
    ...over
  }
}

describe('sanitizeOutputName', () => {
  it('keeps safe characters and collapses the rest', () => {
    expect(sanitizeOutputName('My Merge v1.2_final-a')).toBe('My Merge v1.2_final-a')
    expect(sanitizeOutputName('a/b\\c:d')).toBe('a_b_c_d')
    expect(sanitizeOutputName('a***b')).toBe('a_b')
    expect(sanitizeOutputName('  padded  ')).toBe('padded')
    expect(sanitizeOutputName('..hidden')).toBe('hidden')
    expect(sanitizeOutputName('../../etc')).toBe('_.._etc')
  })
  it('handles empty, only-dots and long values', () => {
    expect(sanitizeOutputName('')).toBe('merged')
    expect(sanitizeOutputName('   ')).toBe('merged')
    expect(sanitizeOutputName('...')).toBe('merged')
    expect(sanitizeOutputName('x'.repeat(300))).toHaveLength(120)
  })
})

describe('buildMergeGraph', () => {
  it('diffusion weighted', () => {
    const g = buildMergeGraph(recipe({ ratio: 0.33333, weightDtype: 'fp8_e4m3fn' }))
    expect(g.saveNode).toBe('ModelSave')
    expect(g.outputSubfolder).toBe('diffusion_models')
    expect(g.filenamePrefix).toBe('diffusion_models/my merge')
    expect(g.prompt).toEqual({
      '1': { class_type: 'UNETLoader', inputs: { unet_name: 'a.safetensors', weight_dtype: 'fp8_e4m3fn' } },
      '2': { class_type: 'UNETLoader', inputs: { unet_name: 'b.safetensors', weight_dtype: 'fp8_e4m3fn' } },
      '3': { class_type: 'ModelMergeSimple', inputs: { model1: ['1', 0], model2: ['2', 0], ratio: 0.3333 } },
      '4': { class_type: 'ModelSave', inputs: { model: ['3', 0], filename_prefix: 'diffusion_models/my merge' } }
    })
  })

  it('diffusion addDifference', () => {
    const g = buildMergeGraph(recipe({ mode: 'addDifference', c: 'c.safetensors', multiplier: 0.75 }))
    expect(g.prompt).toEqual({
      '1': { class_type: 'UNETLoader', inputs: { unet_name: 'a.safetensors', weight_dtype: 'default' } },
      '2': { class_type: 'UNETLoader', inputs: { unet_name: 'b.safetensors', weight_dtype: 'default' } },
      '3': { class_type: 'UNETLoader', inputs: { unet_name: 'c.safetensors', weight_dtype: 'default' } },
      '4': { class_type: 'ModelMergeSubtract', inputs: { model1: ['2', 0], model2: ['3', 0], multiplier: 0.75 } },
      '5': { class_type: 'ModelMergeAdd', inputs: { model1: ['1', 0], model2: ['4', 0] } },
      '6': { class_type: 'ModelSave', inputs: { model: ['5', 0], filename_prefix: 'diffusion_models/my merge' } }
    })
  })

  it('diffusion blocks, values clamped and rounded', () => {
    const g = buildMergeGraph(
      recipe({ mode: 'blocks', blockNode: 'ModelMergeBlocks', blocks: { input: 0.123456, middle: 1.7, out: -3 } })
    )
    expect(g.prompt['3']).toEqual({
      class_type: 'ModelMergeBlocks',
      inputs: { model1: ['1', 0], model2: ['2', 0], input: 0.1235, middle: 1, out: 0 }
    })
    expect(g.prompt['4']).toEqual({
      class_type: 'ModelSave',
      inputs: { model: ['3', 0], filename_prefix: 'diffusion_models/my merge' }
    })
  })

  it('checkpoint weighted with each clip and vae choice', () => {
    const merge = buildMergeGraph(recipe({ kind: 'checkpoint', ratio: 0.25, clip: 'merge', vae: 'a' }))
    expect(merge.saveNode).toBe('CheckpointSave')
    expect(merge.outputSubfolder).toBe('checkpoints')
    expect(merge.prompt).toEqual({
      '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'a.safetensors' } },
      '2': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'b.safetensors' } },
      '3': { class_type: 'ModelMergeSimple', inputs: { model1: ['1', 0], model2: ['2', 0], ratio: 0.25 } },
      '4': { class_type: 'CLIPMergeSimple', inputs: { clip1: ['1', 1], clip2: ['2', 1], ratio: 0.25 } },
      '5': {
        class_type: 'CheckpointSave',
        inputs: { model: ['3', 0], clip: ['4', 0], vae: ['1', 2], filename_prefix: 'checkpoints/my merge' }
      }
    })
    const a = buildMergeGraph(recipe({ kind: 'checkpoint', clip: 'a', vae: 'b' }))
    expect(a.prompt['4']).toEqual({
      class_type: 'CheckpointSave',
      inputs: { model: ['3', 0], clip: ['1', 1], vae: ['2', 2], filename_prefix: 'checkpoints/my merge' }
    })
    const b = buildMergeGraph(recipe({ kind: 'checkpoint', clip: 'b' }))
    expect((b.prompt['4'] as { inputs: Record<string, unknown> }).inputs.clip).toEqual(['2', 1])
  })

  it('checkpoint addDifference merges clips by difference', () => {
    const g = buildMergeGraph(recipe({ kind: 'checkpoint', mode: 'addDifference', c: 'c.safetensors', multiplier: 1.5 }))
    expect(g.prompt).toEqual({
      '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'a.safetensors' } },
      '2': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'b.safetensors' } },
      '3': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'c.safetensors' } },
      '4': { class_type: 'ModelMergeSubtract', inputs: { model1: ['2', 0], model2: ['3', 0], multiplier: 1.5 } },
      '5': { class_type: 'ModelMergeAdd', inputs: { model1: ['1', 0], model2: ['4', 0] } },
      '6': { class_type: 'CLIPMergeSubtract', inputs: { clip1: ['2', 1], clip2: ['3', 1], multiplier: 1.5 } },
      '7': { class_type: 'CLIPMergeAdd', inputs: { clip1: ['1', 1], clip2: ['6', 0] } },
      '8': {
        class_type: 'CheckpointSave',
        inputs: { model: ['5', 0], clip: ['7', 0], vae: ['1', 2], filename_prefix: 'checkpoints/my merge' }
      }
    })
  })

  it('checkpoint blocks uses CLIPMergeSimple with the ratio', () => {
    const g = buildMergeGraph(
      recipe({ kind: 'checkpoint', mode: 'blocks', blockNode: 'ModelMergeBlocks', blocks: { input: 0.5 }, ratio: 0.6 })
    )
    expect(g.prompt['3']).toMatchObject({ class_type: 'ModelMergeBlocks', inputs: { input: 0.5 } })
    expect(g.prompt['4']).toEqual({
      class_type: 'CLIPMergeSimple',
      inputs: { clip1: ['1', 1], clip2: ['2', 1], ratio: 0.6 }
    })
  })

  it('clamps ratio and multiplier and sanitizes the file name', () => {
    const g = buildMergeGraph(recipe({ ratio: 4, outputName: 'a/b' }))
    expect((g.prompt['3'] as { inputs: { ratio: number } }).inputs.ratio).toBe(1)
    expect(g.filenamePrefix).toBe('diffusion_models/a_b')
    const d = buildMergeGraph(recipe({ mode: 'addDifference', c: 'c.safetensors', multiplier: 9 }))
    expect((d.prompt['4'] as { inputs: { multiplier: number } }).inputs.multiplier).toBe(2)
  })
})

describe('completeBlocks', () => {
  it('fills omitted inputs with the ratio and keeps given ones', () => {
    const r = completeBlocks(recipe({ ratio: 0.4, blocks: { input: 0.9 } }), ['input', 'middle', 'out'])
    expect(r.blocks).toEqual({ input: 0.9, middle: 0.4, out: 0.4 })
  })
})

describe('validateRecipe', () => {
  it('accepts a good recipe for every mode and kind', () => {
    expect(validateRecipe(recipe({}), caps(), null)).toEqual([])
    expect(validateRecipe(recipe({ kind: 'checkpoint' }), caps(), null)).toEqual([])
    expect(validateRecipe(recipe({ mode: 'addDifference', c: 'c.safetensors' }), caps(), null)).toEqual([])
    expect(
      validateRecipe(recipe({ mode: 'blocks', blockNode: 'ModelMergeFakeFlux', blocks: { 'double_blocks.0.': 0.5 } }), caps(), [
        'double_blocks.0.',
        'double_blocks.1.'
      ])
    ).toEqual([])
    expect(validateRecipe(recipe({ mode: 'blocks', blockNode: 'ModelMergeBlocks' }), caps(), null)).toEqual([])
  })

  it('reports names missing from the server lists, using the list of the chosen kind', () => {
    const c = caps({ diffusionModels: ['a.safetensors'] })
    expect(validateRecipe(recipe({}), c, null)).toEqual(['Model B "b.safetensors" is not in the server\'s diffusion models list.'])
    expect(validateRecipe(recipe({ kind: 'checkpoint' }), c, null)).toEqual([])
    expect(validateRecipe(recipe({ a: '' }), caps(), null)).toEqual(['Choose model A.'])
  })

  it('reports A equals B, and C rules', () => {
    expect(validateRecipe(recipe({ b: 'a.safetensors' }), caps(), null)).toEqual([
      'A and B are the same file. Choose two different models.'
    ])
    expect(validateRecipe(recipe({ mode: 'addDifference', c: null }), caps(), null)).toEqual(['Choose model C.'])
    expect(validateRecipe(recipe({ mode: 'addDifference', c: 'a.safetensors' }), caps(), null)).toEqual([
      'C is the same file as A. Choose a different model C.'
    ])
  })

  it('reports required nodes the server lacks, only those the recipe uses', () => {
    const c = caps()
    c.nodes.ModelMergeSimple = false
    c.nodes.CheckpointSave = false
    expect(validateRecipe(recipe({}), c, null)).toEqual(['The server has no ModelMergeSimple node.'])
    expect(validateRecipe(recipe({ kind: 'checkpoint' }), c, null)).toEqual([
      'The server has no CheckpointSave node.',
      'The server has no ModelMergeSimple node.'
    ])
    // CLIPMergeSubtract is not in caps.nodes: addDifference with merged clips is accepted.
    expect(validateRecipe(recipe({ kind: 'checkpoint', mode: 'addDifference', c: 'c.safetensors' }), caps(), null)).toEqual([])
  })

  it('checks the block node and block names', () => {
    expect(validateRecipe(recipe({ mode: 'blocks' }), caps(), null)).toEqual(['Choose a block merge node.'])
    expect(validateRecipe(recipe({ mode: 'blocks', blockNode: 'ModelMergeNope' }), caps(), null)).toEqual([
      'The server has no ModelMergeNope node.'
    ])
    const c = caps()
    c.nodes.ModelMergeBlocks = false
    expect(validateRecipe(recipe({ mode: 'blocks', blockNode: 'ModelMergeBlocks' }), c, null)).toEqual([
      'The server has no ModelMergeBlocks node.'
    ])
    expect(
      validateRecipe(recipe({ mode: 'blocks', blockNode: 'ModelMergeBlocks', blocks: { input: 0.5, bogus: 0.5 } }), caps(), [
        'input',
        'middle',
        'out'
      ])
    ).toEqual(['These block names are not inputs of ModelMergeBlocks: bogus.'])
    expect(validateRecipe(recipe({ mode: 'blocks', blockNode: 'ModelMergeBlocks', blocks: { input: 1.5 } }), caps(), null)).toEqual([
      'Block "input" must be between 0 and 1.'
    ])
  })

  it('reports out-of-range numbers and an empty output name', () => {
    expect(validateRecipe(recipe({ ratio: 1.2 }), caps(), null)).toEqual(['The ratio must be between 0 and 1.'])
    expect(validateRecipe(recipe({ ratio: Number.NaN }), caps(), null)).toEqual(['The ratio must be between 0 and 1.'])
    expect(validateRecipe(recipe({ mode: 'addDifference', c: 'c.safetensors', multiplier: 2.5 }), caps(), null)).toEqual([
      'The multiplier must be between 0 and 2.'
    ])
    expect(validateRecipe(recipe({ outputName: '  ' }), caps(), null)).toEqual(['Enter a name for the merged file.'])
  })
})
