import { describe, expect, it } from 'vitest'
import { DEFAULT_RECIPE, MAX_LORAS, type LoraEntry, type MergeRecipe } from '@shared/merge'
import { KNOWN_NODES, type ServerCapabilities } from '@shared/types'
import { checkMerge, describeLoraFile, describeModelFile, isFusedInto, isTextEncoderLoraKey } from '../src/main/merge/inspect'
import { buildMergeGraph, validateRecipe } from '../src/main/merge/graph'
import type { SafetensorsHeader, SafetensorsTensor } from '../src/main/models/safetensors'

function header(tensors: Record<string, [string, number[]]>, metadata: Record<string, string> = {}, fileBytes = 1000): SafetensorsHeader {
  const t: Record<string, SafetensorsTensor> = {}
  for (const [k, [dtype, shape]] of Object.entries(tensors)) t[k] = { dtype, shape, offsets: [0, 0] }
  return { tensors: t, metadata, headerBytes: 10, fileBytes, headerSha256: 'x' }
}

/** Model A: blocks.0..n-1 with proj weights, under the checkpoint prefix. */
function modelA(n = 4, prefix = 'model.diffusion_model.'): SafetensorsHeader {
  const t: Record<string, [string, number[]]> = {}
  for (let i = 0; i < n; i++) t[`${prefix}layers.${i}.proj.k.weight`] = ['BF16', [8, 8]]
  return header(t)
}

function comfyLora(n: number, over: Record<string, [string, number[]]> = {}): SafetensorsHeader {
  const t: Record<string, [string, number[]]> = {}
  for (let i = 0; i < n; i++) {
    t[`diffusion_model.layers.${i}.proj.k.lora_down.weight`] = ['BF16', [16, 64]]
    t[`diffusion_model.layers.${i}.proj.k.lora_up.weight`] = ['BF16', [64, 16]]
    t[`diffusion_model.layers.${i}.proj.k.alpha`] = ['F32', []]
  }
  return header({ ...t, ...over })
}

const noObjects: Record<string, unknown> = {}

describe('isTextEncoderLoraKey', () => {
  it('recognises text-encoder prefixes, also behind base_model.model.', () => {
    for (const k of [
      'lora_te_text_model_x',
      'lora_te1_a',
      'lora_te2_a',
      'lora_te3_a',
      'text_encoder.layers.0',
      'text_encoders.t5.x',
      'text_model.encoder',
      'te1.x',
      'te2.x',
      'text_encoder_2.x',
      'clip_l.x',
      'clip_g.x',
      't5xxl.x',
      'conditioner.x',
      'base_model.model.text_encoder.x'
    ]) {
      expect(isTextEncoderLoraKey(k), k).toBe(true)
    }
    for (const k of ['diffusion_model.layers.0', 'lora_unet_x', 'transformer.x', 'base_model.model.transformer.x', 'text']) {
      expect(isTextEncoderLoraKey(k), k).toBe(false)
    }
  })
})

describe('describeLoraFile', () => {
  it('reads a comfy-style LoRA and matches it against A', () => {
    const info = describeLoraFile('moss.safetensors', comfyLora(4), modelA(4), null)
    expect(info).toMatchObject({
      name: 'moss.safetensors',
      readable: true,
      error: null,
      format: 'LoRA',
      rank: 16,
      keyStyle: 'comfy',
      modules: 4,
      match: 1,
      textEncoderModules: 0,
      baseModelHint: null,
      tensorCount: 12,
      sizeBytes: 1000
    })
  })

  it('gives the share of matching modules, and 0 when none match', () => {
    expect(describeLoraFile('l', comfyLora(4), modelA(2), null).match).toBe(0.5)
    expect(describeLoraFile('l', comfyLora(4), modelA(0), null).match).toBe(0)
    expect(describeLoraFile('l', comfyLora(4), null, null).match).toBeNull()
  })

  it('accepts the model.diffusion_model. prefix and A without any prefix', () => {
    const t: Record<string, [string, number[]]> = {
      'model.diffusion_model.layers.0.proj.k.lora_A.weight': ['BF16', [4, 64]],
      'model.diffusion_model.layers.0.proj.k.lora_B.weight': ['BF16', [64, 4]]
    }
    const info = describeLoraFile('l', header(t), modelA(1, ''), null)
    expect(info).toMatchObject({ keyStyle: 'comfy', match: 1, rank: 4, format: 'LoRA' })
  })

  it('reads a kohya LoRA', () => {
    const t: Record<string, [string, number[]]> = {
      'lora_unet_layers_0_proj_k.lora_down.weight': ['F16', [32, 64]],
      'lora_unet_layers_0_proj_k.lora_up.weight': ['F16', [64, 32]],
      'lora_unet_layers_0_proj_k.alpha': ['F32', []],
      'lora_unet_other_layer.lora_down.weight': ['F16', [32, 64]],
      'lora_unet_other_layer.lora_up.weight': ['F16', [64, 32]]
    }
    const info = describeLoraFile('k', header(t), modelA(2), null)
    expect(info).toMatchObject({ keyStyle: 'kohya', modules: 2, match: 0.5, rank: 32, format: 'LoRA' })
  })

  it('reads a diffusers LoRA without a match', () => {
    const t: Record<string, [string, number[]]> = {
      'transformer.transformer_blocks.0.attn.to_k.lora.down.weight': ['BF16', [16, 64]],
      'transformer.transformer_blocks.0.attn.to_k.lora.up.weight': ['BF16', [64, 16]]
    }
    const info = describeLoraFile('d', header(t), modelA(2), null)
    expect(info).toMatchObject({ keyStyle: 'diffusers', modules: 1, match: null, rank: 16, format: 'LoRA' })
    const peft = describeLoraFile('p', header({ 'base_model.model.x.lora_A.weight': ['BF16', [4, 64]] }), null, null)
    expect(peft.keyStyle).toBe('diffusers')
  })

  it('detects LoKr, LoHa, DoRA and mixed', () => {
    const lokr = describeLoraFile(
      'lokr',
      header({
        'diffusion_model.layers.0.proj.k.lokr_w1': ['F32', [4, 4]],
        'diffusion_model.layers.0.proj.k.lokr_w2': ['F32', [4, 4]],
        'diffusion_model.layers.1.proj.k.lokr_w1_a': ['F32', [4, 2]],
        'diffusion_model.layers.1.proj.k.lokr_w1_b': ['F32', [2, 4]],
        'diffusion_model.layers.1.proj.k.lokr_w2': ['F32', [4, 4]]
      }),
      modelA(2),
      null
    )
    expect(lokr).toMatchObject({ format: 'LoKr', rank: null, modules: 2, match: 1, keyStyle: 'comfy' })

    const loha = describeLoraFile(
      'loha',
      header({ 'diffusion_model.a.hada_w1_a': ['F32', [4, 2]], 'diffusion_model.a.hada_w2_b': ['F32', [2, 4]] }),
      null,
      null
    )
    expect(loha.format).toBe('LoHa')

    const dora = describeLoraFile('dora', comfyLora(1, { 'diffusion_model.layers.0.proj.k.dora_scale': ['F32', [8]] }), null, null)
    expect(dora.format).toBe('DoRA')
    expect(dora.modules).toBe(1)

    const mixed = describeLoraFile(
      'mixed',
      comfyLora(1, { 'diffusion_model.layers.5.proj.k.lokr_w1': ['F32', [4, 4]] }),
      null,
      null
    )
    expect(mixed.format).toBe('mixed')
  })

  it('counts text-encoder modules separately', () => {
    const t = comfyLora(2).tensors
    const h: SafetensorsHeader = {
      ...comfyLora(2),
      tensors: {
        ...t,
        'lora_te1_text_model_a.lora_down.weight': { dtype: 'F32', shape: [4, 4], offsets: [0, 0] },
        'lora_te1_text_model_a.lora_up.weight': { dtype: 'F32', shape: [4, 4], offsets: [0, 0] },
        'text_encoder.b.lora_A.weight': { dtype: 'F32', shape: [4, 4], offsets: [0, 0] }
      }
    }
    const info = describeLoraFile('te', h, modelA(2), null)
    expect(info.textEncoderModules).toBe(2)
    expect(info.modules).toBe(2)
    expect(info.match).toBe(1)
  })

  it('ignores keys that match no suffix, and reports unknown format and style', () => {
    const info = describeLoraFile('x', header({ 'something.weight': ['F32', [2]] }), modelA(1), null)
    expect(info).toMatchObject({ format: 'unknown', keyStyle: 'unknown', modules: 0, match: null, rank: null })
  })

  it('takes the base model hint from the metadata, trimmed and capped', () => {
    expect(describeLoraFile('m', comfyLora(1), null, null).baseModelHint).toBeNull()
    expect(describeLoraFile('m', { ...comfyLora(1), metadata: { ss_base_model: '  ', 'modelspec.architecture': ' Moss-1 ' } }, null, null).baseModelHint).toBe('Moss-1')
    expect(describeLoraFile('m', { ...comfyLora(1), metadata: { ss_base_model: 'x'.repeat(200) } }, null, null).baseModelHint).toHaveLength(80)
    expect(describeLoraFile('m', { ...comfyLora(1), metadata: { base_model: 'b' } }, null, null).baseModelHint).toBe('b')
  })

  it('reports an unreadable file', () => {
    const info = describeLoraFile('gone', null, modelA(1), 'File not found.')
    expect(info).toEqual({
      name: 'gone',
      readable: false,
      error: 'File not found.',
      sizeBytes: null,
      tensorCount: 0,
      format: 'unknown',
      rank: null,
      keyStyle: 'unknown',
      modules: 0,
      match: null,
      textEncoderModules: 0,
      baseModelHint: null
    })
  })
})

describe('checkMerge with LoRAs', () => {
  const aSrc = { info: describeModelFile('a.safetensors', modelA(4)), header: modelA(4) }
  const bake = { kind: 'diffusion' as const, mode: 'bake' as const, a: aSrc, b: null, c: null, objectInfo: noObjects, ramFreeBytes: null }
  const lora = (name: string, h: SafetensorsHeader | null, err: string | null = null) => describeLoraFile(name, h, aSrc.header, err)

  it('bake needs only A, passes with a fitting LoRA, and returns the LoRAs in order', () => {
    const l1 = lora('one.safetensors', comfyLora(4))
    const l2 = lora('two.safetensors', comfyLora(4))
    const res = checkMerge({ ...bake, loras: [l1, l2] })
    expect(res.problems).toEqual([])
    expect(res.warnings).toEqual([])
    expect(res.compatible).toBe(true)
    expect(res.loras).toEqual([l1, l2])
    expect(res.b).toBeNull()
  })

  it('bake: no B checks, even for an empty B or the same name', () => {
    const res = checkMerge({ ...bake, b: { info: describeModelFile('a.safetensors', null), header: null }, loras: [lora('l', comfyLora(4))] })
    expect(res.problems).toEqual([])
    expect(res.warnings).toEqual([])
  })

  it('bake: asks for model A and for a LoRA', () => {
    const none = checkMerge({ ...bake, a: null, loras: [] })
    expect(none.problems).toEqual(['Choose model A.', 'Add at least one LoRA to bake.'])
    expect(none.compatible).toBe(false)
    expect(checkMerge({ ...bake, loras: [] }).problems).toEqual(['Add at least one LoRA to bake.'])
    // other modes do not need a LoRA
    const merge = checkMerge({ ...bake, mode: 'weighted', b: aSrc, loras: [] })
    expect(merge.problems.some((p) => /LoRA/.test(p))).toBe(false)
  })

  it('blocks a LoRA that matches nothing in A', () => {
    const res = checkMerge({ ...bake, loras: [lora('wrong.safetensors', comfyLora(4, {}) && header({ 'diffusion_model.zzz.lora_down.weight': ['F32', [4, 4]] }))] })
    expect(res.problems).toEqual([
      'wrong.safetensors does not fit model A: none of its 1 layers are in A. It is probably for another model type.'
    ])
    expect(res.compatible).toBe(false)
  })

  it('warns when a LoRA fits only part of A, and not at 90 percent or more', () => {
    const partial = checkMerge({ ...bake, loras: [lora('p.safetensors', comfyLora(8))] }) // 4 of 8 modules match
    expect(partial.problems).toEqual([])
    expect(partial.warnings).toEqual(["p.safetensors fits only 50% of model A's layers; the rest is skipped."])
    const t: Record<string, [string, number[]]> = {}
    for (let i = 0; i < 10; i++) t[`diffusion_model.layers.${i}.proj.k.lora_down.weight`] = ['F32', [4, 8]]
    const nine = checkMerge({ ...bake, a: { info: aSrc.info, header: modelA(9) }, loras: [describeLoraFile('n', header(t), modelA(9), null)] })
    expect(nine.warnings).toEqual([]) // 9 of 10 = 90%
  })

  it('only warns (no problem) when a diffusers LoRA cannot be matched', () => {
    const d = lora('d.safetensors', header({ 'transformer.blocks.0.attn.to_k.lora_A.weight': ['F32', [4, 8]] }))
    const res = checkMerge({ ...bake, loras: [d] })
    expect(res.problems).toEqual([])
    expect(res.warnings).toHaveLength(1)
    expect(res.compatible).toBe(true)
  })

  it('warns about text-encoder layers only for a diffusion-model file', () => {
    const te = lora(
      'te.safetensors',
      header({
        ...Object.fromEntries(Object.entries(comfyLora(4).tensors).map(([k, v]) => [k, [v.dtype, v.shape] as [string, number[]]])),
        'lora_te1_a.lora_down.weight': ['F32', [4, 4]],
        'lora_te1_b.lora_down.weight': ['F32', [4, 4]]
      })
    )
    const diffusion = checkMerge({ ...bake, loras: [te] })
    expect(diffusion.warnings).toEqual([
      'te.safetensors also changes the text encoder (2 layers); a diffusion-model file has none, so that part is left out.'
    ])
    const ckpt = checkMerge({ ...bake, kind: 'checkpoint', loras: [te] })
    expect(ckpt.warnings).toEqual([])
  })

  it('blocks a LoRA with no diffusion-model layers', () => {
    const empty = lora('empty.safetensors', header({ 'lora_te1_a.lora_down.weight': ['F32', [4, 4]] }))
    expect(checkMerge({ ...bake, loras: [empty] }).problems).toEqual(['empty.safetensors has no layers for the diffusion model.'])
  })

  it('warns about an unreadable LoRA and returns compatible null', () => {
    const gone = lora('gone.safetensors', null, 'It is on a remote server.')
    const res = checkMerge({ ...bake, loras: [gone] })
    expect(res.problems).toEqual([])
    expect(res.warnings).toEqual(['gone.safetensors could not be read (It is on a remote server.), so it was not checked.'])
    expect(res.compatible).toBeNull()
  })

  it('adds LoRA sizes to the RAM estimate; bake counts only A and the LoRAs', () => {
    const big = (n: number): SafetensorsHeader => ({ ...comfyLora(4), fileBytes: n })
    const a = { info: describeModelFile('a', { ...modelA(4), fileBytes: 1000 }), header: { ...modelA(4), fileBytes: 1000 } }
    const b = { info: describeModelFile('b', { ...modelA(4), fileBytes: 5000 }), header: { ...modelA(4), fileBytes: 5000 } }
    const l = describeLoraFile('l', big(200), a.header, null)
    const baked = checkMerge({ ...bake, a, b, loras: [l] })
    expect(baked.ramNeededBytes).toBe(Math.round(1200 * 1.2))
    const merged = checkMerge({ ...bake, mode: 'weighted', a, b, loras: [l] })
    expect(merged.ramNeededBytes).toBe(Math.round(6200 * 1.2))
  })
})

function recipe(over: Partial<MergeRecipe>): MergeRecipe {
  return { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors', outputName: 'my merge', ...over }
}

function caps(over: Partial<ServerCapabilities> = {}): ServerCapabilities {
  const nodes: Record<string, boolean> = {}
  for (const n of KNOWN_NODES) nodes[n] = true
  return {
    checkpoints: ['a.safetensors', 'b.safetensors'],
    diffusionModels: ['a.safetensors', 'b.safetensors'],
    loras: [],
    textEncoders: [],
    vaes: [],
    nodes,
    blockMergeNodes: [],
    ...over
  }
}

const lo = (name: string, strength = 1, clipStrength = 1): LoraEntry => ({ name, strength, clipStrength })

describe('validateRecipe with LoRAs', () => {
  it('accepts a bake without B and ignores B and C', () => {
    expect(validateRecipe(recipe({ mode: 'bake', b: '', loras: [lo('x.safetensors')] }), caps(), null)).toEqual([])
    expect(validateRecipe(recipe({ mode: 'bake', b: 'nope', c: 'nope', ratio: 9, loras: [lo('x')] }), caps(), null)).toEqual([])
    expect(validateRecipe(recipe({ mode: 'bake', kind: 'checkpoint', loras: [lo('x')] }), caps(), null)).toEqual([])
  })

  it('bake needs A and at least one LoRA', () => {
    expect(validateRecipe(recipe({ mode: 'bake', a: '', loras: [lo('x')] }), caps(), null)).toEqual(['Choose model A.'])
    expect(validateRecipe(recipe({ mode: 'bake', loras: [] }), caps(), null)).toEqual(['Add at least one LoRA to bake.'])
    const c = caps()
    c.nodes.ModelMergeSimple = false
    expect(validateRecipe(recipe({ mode: 'bake', loras: [lo('x')] }), c, null)).toEqual([])
  })

  it('limits the number of LoRAs in every mode', () => {
    const many = Array.from({ length: MAX_LORAS + 1 }, (_, i) => lo(`l${i}.safetensors`))
    expect(validateRecipe(recipe({ loras: many }), caps(), null)).toEqual([`At most ${MAX_LORAS} LoRAs can be baked at once.`])
    expect(validateRecipe(recipe({ loras: many.slice(0, MAX_LORAS) }), caps(), null)).toEqual([])
  })

  it('checks names against the server list only when the list is not empty', () => {
    const c = caps({ loras: ['x.safetensors'] })
    expect(validateRecipe(recipe({ loras: [lo('x.safetensors')] }), c, null)).toEqual([])
    expect(validateRecipe(recipe({ loras: [lo('y.safetensors')] }), c, null)).toEqual([
      'LoRA "y.safetensors" is not in the server\'s loras list.'
    ])
    expect(validateRecipe(recipe({ loras: [lo('y.safetensors')] }), caps(), null)).toEqual([])
    expect(validateRecipe(recipe({ loras: [lo('  ')] }), caps(), null)).toEqual(['Choose a file for LoRA 1.'])
  })

  it('checks strengths', () => {
    expect(validateRecipe(recipe({ loras: [lo('x', 10, -10)] }), caps(), null)).toEqual([])
    expect(validateRecipe(recipe({ loras: [lo('x', 10.5)] }), caps(), null)).toEqual([
      'The strength of LoRA "x" must be between -10 and 10.'
    ])
    expect(validateRecipe(recipe({ loras: [lo('x', 1, Number.NaN)] }), caps(), null)).toEqual([
      'The text encoder strength of LoRA "x" must be between -10 and 10.'
    ])
  })

  it('needs the LoRA loader node of the kind', () => {
    const c = caps()
    c.nodes.LoraLoaderModelOnly = false
    expect(validateRecipe(recipe({ loras: [lo('x')] }), c, null)).toEqual(['The server has no LoraLoaderModelOnly node.'])
    expect(validateRecipe(recipe({ kind: 'checkpoint', loras: [lo('x')] }), c, null)).toEqual([])
    const d = caps()
    d.nodes.LoraLoader = false
    expect(validateRecipe(recipe({ kind: 'checkpoint', loras: [lo('x')] }), d, null)).toEqual(['The server has no LoraLoader node.'])
    expect(validateRecipe(recipe({ loras: [] }), d, null)).toEqual([])
  })
})

describe('buildMergeGraph with LoRAs', () => {
  it('bake on a diffusion model: A, LoRA chain, save', () => {
    const g = buildMergeGraph(recipe({ mode: 'bake', weightDtype: 'fp8_e4m3fn', loras: [lo('x.safetensors', 0.8), lo('y.safetensors', -1.23456)] }))
    expect(g.saveNode).toBe('ModelSave')
    expect(g.prompt).toEqual({
      '1': { class_type: 'UNETLoader', inputs: { unet_name: 'a.safetensors', weight_dtype: 'fp8_e4m3fn' } },
      '2': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['1', 0], lora_name: 'x.safetensors', strength_model: 0.8 } },
      '3': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['2', 0], lora_name: 'y.safetensors', strength_model: -1.2346 } },
      '4': { class_type: 'ModelSave', inputs: { model: ['3', 0], filename_prefix: 'diffusion_models/my merge' } }
    })
  })

  it('bake on a checkpoint: LoraLoader chain, A\'s VAE', () => {
    const g = buildMergeGraph(recipe({ mode: 'bake', kind: 'checkpoint', vae: 'b', loras: [lo('x', 0.5, 0.25), lo('y', 1, 0)] }))
    expect(g.saveNode).toBe('CheckpointSave')
    expect(g.prompt).toEqual({
      '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'a.safetensors' } },
      '2': {
        class_type: 'LoraLoader',
        inputs: { model: ['1', 0], clip: ['1', 1], lora_name: 'x', strength_model: 0.5, strength_clip: 0.25 }
      },
      '3': {
        class_type: 'LoraLoader',
        inputs: { model: ['2', 0], clip: ['2', 1], lora_name: 'y', strength_model: 1, strength_clip: 0 }
      },
      '4': {
        class_type: 'CheckpointSave',
        inputs: { model: ['3', 0], clip: ['3', 1], vae: ['1', 2], filename_prefix: 'checkpoints/my merge' }
      }
    })
  })

  it('weighted diffusion merge with two LoRAs applies them after the merge', () => {
    const g = buildMergeGraph(recipe({ ratio: 0.5, loras: [lo('x'), lo('y', 0.5)] }))
    expect(g.prompt).toEqual({
      '1': { class_type: 'UNETLoader', inputs: { unet_name: 'a.safetensors', weight_dtype: 'default' } },
      '2': { class_type: 'UNETLoader', inputs: { unet_name: 'b.safetensors', weight_dtype: 'default' } },
      '3': { class_type: 'ModelMergeSimple', inputs: { model1: ['1', 0], model2: ['2', 0], ratio: 0.5 } },
      '4': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['3', 0], lora_name: 'x', strength_model: 1 } },
      '5': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['4', 0], lora_name: 'y', strength_model: 0.5 } },
      '6': { class_type: 'ModelSave', inputs: { model: ['5', 0], filename_prefix: 'diffusion_models/my merge' } }
    })
  })

  it('weighted checkpoint merge with two LoRAs applies them to the merged model and clip', () => {
    const g = buildMergeGraph(recipe({ kind: 'checkpoint', ratio: 0.25, clip: 'merge', vae: 'b', loras: [lo('x', 1, 0.5), lo('y', -1, 0)] }))
    expect(g.prompt).toEqual({
      '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'a.safetensors' } },
      '2': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'b.safetensors' } },
      '3': { class_type: 'ModelMergeSimple', inputs: { model1: ['1', 0], model2: ['2', 0], ratio: 0.25 } },
      '4': { class_type: 'CLIPMergeSimple', inputs: { clip1: ['1', 1], clip2: ['2', 1], ratio: 0.25 } },
      '5': {
        class_type: 'LoraLoader',
        inputs: { model: ['3', 0], clip: ['4', 0], lora_name: 'x', strength_model: 1, strength_clip: 0.5 }
      },
      '6': {
        class_type: 'LoraLoader',
        inputs: { model: ['5', 0], clip: ['5', 1], lora_name: 'y', strength_model: -1, strength_clip: 0 }
      },
      '7': {
        class_type: 'CheckpointSave',
        inputs: { model: ['6', 0], clip: ['6', 1], vae: ['2', 2], filename_prefix: 'checkpoints/my merge' }
      }
    })
  })

  it('addDifference with a LoRA keeps the existing node ids', () => {
    const g = buildMergeGraph(recipe({ mode: 'addDifference', c: 'c.safetensors', loras: [lo('x')] }))
    expect(g.prompt['5']).toMatchObject({ class_type: 'ModelMergeAdd' })
    expect(g.prompt['6']).toEqual({ class_type: 'LoraLoaderModelOnly', inputs: { model: ['5', 0], lora_name: 'x', strength_model: 1 } })
    expect(g.prompt['7']).toMatchObject({ class_type: 'ModelSave', inputs: { model: ['6', 0] } })
  })

  it('clamps strengths to -10..10 and leaves a graph without LoRAs unchanged', () => {
    const g = buildMergeGraph(recipe({ mode: 'bake', loras: [lo('x', 99)] }))
    expect((g.prompt['2'] as { inputs: { strength_model: number } }).inputs.strength_model).toBe(10)
    expect(Object.keys(buildMergeGraph(recipe({})).prompt)).toEqual(['1', '2', '3', '4'])
  })
})

describe('isFusedInto', () => {
  const kids = new Map([
    ['layers.0.proj', ['qkv', 'o']],
    ['blocks.0.attn', ['to_qkv', 'to_out']],
    ['blocks.0.ffn', ['up_gate', 'down']]
  ])
  it('finds q/k/v stored as one fused weight', () => {
    expect(isFusedInto('layers.0.proj.k', kids)).toBe(true)
    expect(isFusedInto('blocks.0.attn.to_v', kids)).toBe(true)
    expect(isFusedInto('blocks.0.ffn.gate', kids)).toBe(true)
  })
  it('does not match unrelated layers', () => {
    expect(isFusedInto('layers.0.proj.x', kids)).toBe(false)
    expect(isFusedInto('layers.1.proj.k', kids)).toBe(false)
    expect(isFusedInto('blocks.0.ffn.down2', kids)).toBe(false)
  })
})

describe('review fixes', () => {
  const fused = header({ 'model.diffusion_model.blocks.0.attn.to_qkv.weight': ['BF16', [24, 8]], 'model.diffusion_model.blocks.0.attn.to_out.weight': ['BF16', [8, 8]] })
  it('kohya LoRA for q/k/v matches a model that stores them fused', () => {
    const t: Record<string, [string, number[]]> = {}
    for (const p of ['to_q', 'to_k', 'to_v', 'to_out']) {
      t[`lora_unet_blocks_0_attn_${p}.lora_down.weight`] = ['BF16', [4, 8]]
      t[`lora_unet_blocks_0_attn_${p}.lora_up.weight`] = ['BF16', [8, 4]]
    }
    const info = describeLoraFile('k.safetensors', header(t), fused, null)
    expect(info.keyStyle).toBe('kohya')
    expect(info.match).toBe(1)
  })
  const aSrc = { info: describeModelFile('a.safetensors', fused), header: fused }
  const base = { mode: 'bake' as const, a: aSrc, b: null, c: null, objectInfo: noObjects, ramFreeBytes: null }
  it('diffusers LoRA gets a not-checked warning', () => {
    const l = describeLoraFile('d.safetensors', header({ 'transformer.blocks.0.attn.to_q.lora.down.weight': ['BF16', [4, 8]], 'transformer.blocks.0.attn.to_q.lora.up.weight': ['BF16', [8, 4]] }), fused, null)
    const res = checkMerge({ ...base, kind: 'diffusion', loras: [l] })
    expect(res.problems).toEqual([])
    expect(res.warnings.some((w) => w.includes('cannot check it before baking'))).toBe(true)
  })
  it('text-encoder-only LoRA: allowed for a checkpoint, refused for a diffusion model', () => {
    const l = describeLoraFile('te.safetensors', header({ 'lora_te1_text_model_x.lora_down.weight': ['BF16', [4, 8]], 'lora_te1_text_model_x.lora_up.weight': ['BF16', [8, 4]] }), fused, null)
    expect(checkMerge({ ...base, kind: 'checkpoint', loras: [l] }).problems).toEqual([])
    expect(checkMerge({ ...base, kind: 'diffusion', loras: [l] }).problems).toEqual(['te.safetensors has no layers for the diffusion model.'])
  })
})
