import { describe, expect, it } from 'vitest'
import { DEFAULT_RECIPE, type MergeRecipe } from '@shared/merge'
import { buildMergeGraph, buildMergeModelSubgraph } from '../src/main/merge/graph'
import { buildGridPrompt, checkTemplateForRecipe, inspectTemplate } from '../src/main/grid/template'

type Api = Record<string, { class_type: string; inputs: Record<string, unknown> }>

function recipe(over: Partial<MergeRecipe> = {}): MergeRecipe {
  return { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors', outputName: 'm', ...over }
}

function diffusionTemplate(): Api {
  return {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: 'x.safetensors', weight_dtype: 'default' } },
    '2': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['1', 0], lora_name: 'l.safetensors', strength_model: 1 } },
    '3': { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['2', 0], shift: 3 } },
    '4': { class_type: 'CLIPLoader', inputs: { clip_name: 'te.safetensors', type: 'lumina2' } },
    '5': { class_type: 'CLIPTextEncode', inputs: { text: 'a red fox', clip: ['4', 0] } },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry', clip: ['4', 0] } },
    '7': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '8': {
      class_type: 'KSampler',
      inputs: { model: ['3', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['7', 0], seed: 42, steps: 20, cfg: 4 }
    },
    '9': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
    '10': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['9', 0] } },
    '11': { class_type: 'SaveImage', inputs: { images: ['10', 0], filename_prefix: 'out' } }
  }
}

function checkpointTemplate(): Api {
  return {
    '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'x.safetensors' } },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: 'a castle', clip: ['4', 1] } },
    '7': { class_type: 'CLIPTextEncode', inputs: { text: 'ugly', clip: ['4', 1] } },
    '5': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '3': {
      class_type: 'KSampler',
      inputs: { model: ['4', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0], seed: 1, steps: 20 }
    },
    '8': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
    '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'ComfyUI' } }
  }
}

function guiderTemplate(): Api {
  return {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: 'x.safetensors', weight_dtype: 'default' } },
    '2': { class_type: 'DualCLIPLoader', inputs: { clip_name1: 'a', clip_name2: 'b', type: 'flux' } },
    '3': { class_type: 'CLIPTextEncode', inputs: { text: 'a dragon', clip: ['2', 0] } },
    '4': { class_type: 'FluxGuidance', inputs: { conditioning: ['3', 0], guidance: 3.5 } },
    '5': { class_type: 'CFGGuider', inputs: { model: ['1', 0], conditioning: ['4', 0], cfg: 1 } },
    '6': { class_type: 'RandomNoise', inputs: { noise_seed: 7 } },
    '7': { class_type: 'KSamplerSelect', inputs: { sampler_name: 'euler' } },
    '8': { class_type: 'BasicScheduler', inputs: { model: ['1', 0], scheduler: 'simple', steps: 20, denoise: 1 } },
    '9': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '10': {
      class_type: 'SamplerCustomAdvanced',
      inputs: { noise: ['6', 0], guider: ['5', 0], sampler: ['7', 0], sigmas: ['8', 0], latent_image: ['9', 0] }
    },
    '11': { class_type: 'VAELoader', inputs: { vae_name: 'ae' } },
    '12': { class_type: 'VAEDecode', inputs: { samples: ['10', 0], vae: ['11', 0] } },
    '13': { class_type: 'PreviewImage', inputs: { images: ['12', 0] } }
  }
}

const uiFormat = { last_node_id: 3, last_link_id: 2, nodes: [{ id: 1, type: 'KSampler' }], links: [], version: 0.4 }

function allLinksValid(p: Record<string, unknown>): boolean {
  const nodes = p as Api
  for (const n of Object.values(nodes)) {
    for (const v of Object.values(n.inputs)) {
      if (Array.isArray(v) && v.length === 2 && typeof v[0] === 'string') {
        if (!nodes[v[0]]) return false
      }
    }
  }
  return true
}

describe('inspectTemplate', () => {
  it('reads a diffusion template', () => {
    const info = inspectTemplate(diffusionTemplate(), 'C:\\wf\\fox.json')
    expect(info.error).toBeNull()
    expect(info.name).toBe('fox.json')
    expect(info.path).toBe('C:\\wf\\fox.json')
    expect(info.modelLoader).toEqual({ id: '1', classType: 'UNETLoader' })
    expect(info.promptInput).toEqual({ id: '5', input: 'text', classType: 'CLIPTextEncode' })
    expect(info.defaultPrompt).toBe('a red fox')
    expect(info.seedInputs).toEqual([{ id: '8', input: 'seed' }])
    expect(info.imageOutputs).toEqual(['11'])
    expect(info.warnings.some((w) => /LoRA/.test(w))).toBe(true)
    expect(info.warnings.some((w) => /diffusion model/.test(w))).toBe(true)
  })

  it('reads a checkpoint template', () => {
    const info = inspectTemplate(checkpointTemplate(), '/tmp/castle.json')
    expect(info.error).toBeNull()
    expect(info.name).toBe('castle.json')
    expect(info.modelLoader).toEqual({ id: '4', classType: 'CheckpointLoaderSimple' })
    expect(info.promptInput?.id).toBe('6')
    expect(info.defaultPrompt).toBe('a castle')
    expect(info.warnings.some((w) => /only checkpoint merges/.test(w))).toBe(true)
    expect(info.warnings.some((w) => /LoRA/.test(w))).toBe(false)
  })

  it('reads a guider template with noise_seed and walks back through FluxGuidance', () => {
    const info = inspectTemplate(guiderTemplate(), 'g.json')
    expect(info.error).toBeNull()
    expect(info.promptInput).toEqual({ id: '3', input: 'text', classType: 'CLIPTextEncode' })
    expect(info.defaultPrompt).toBe('a dragon')
    expect(info.seedInputs).toEqual([{ id: '6', input: 'noise_seed' }])
    expect(info.imageOutputs).toEqual(['13'])
  })

  it('rejects the UI format', () => {
    const info = inspectTemplate(uiFormat, 'ui.json')
    expect(info.error).toBe(
      'This is a workflow file, not the API format. In ComfyUI use Workflow > Export (API) and pick that file.'
    )
  })

  it('rejects things that are not API format', () => {
    for (const bad of [null, 5, 'x', [], {}, { '1': { class_type: 'A' } }, { '1': 'nope' }]) {
      expect(inspectTemplate(bad, 'x.json').error).toMatch(/API format/)
    }
  })

  it('errors when there is no supported loader and names unsupported ones', () => {
    const t = diffusionTemplate()
    t['1'] = { class_type: 'UnetLoaderGGUF', inputs: { unet_name: 'x.gguf' } }
    const info = inspectTemplate(t, 'x.json')
    expect(info.error).toMatch(/UNETLoader or CheckpointLoaderSimple/)
    expect(info.error).toMatch(/UnetLoaderGGUF/)
    expect(info.error).toMatch(/not supported/)
    const t2 = diffusionTemplate()
    delete t2['1']
    t2['2'].inputs = { lora_name: 'x', strength_model: 1 }
    t2['3'].inputs = { shift: 3 }
    const e2 = inspectTemplate(t2, 'x.json').error
    expect(e2).toMatch(/no UNETLoader or CheckpointLoaderSimple/)
    expect(e2).not.toMatch(/Found/)
  })

  it('picks the loader that reaches the image when there are two', () => {
    const t = diffusionTemplate()
    t['20'] = { class_type: 'UNETLoader', inputs: { unet_name: 'unused.safetensors', weight_dtype: 'default' } }
    const info = inspectTemplate(t, 'x.json')
    expect(info.error).toBeNull()
    expect(info.modelLoader?.id).toBe('1')
  })

  it('errors when two loaders are ambiguous', () => {
    const t = diffusionTemplate()
    t['20'] = { class_type: 'UNETLoader', inputs: { unet_name: 'other.safetensors', weight_dtype: 'default' } }
    t['21'] = { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['20', 0], shift: 3 } }
    t['8'].inputs.model = ['3', 0]
    t['22'] = {
      class_type: 'KSampler',
      inputs: { model: ['21', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['7', 0], seed: 1 }
    }
    t['23'] = { class_type: 'VAEDecode', inputs: { samples: ['22', 0], vae: ['9', 0] } }
    t['24'] = { class_type: 'PreviewImage', inputs: { images: ['23', 0] } }
    const info = inspectTemplate(t, 'x.json')
    expect(info.error).toMatch(/2 model loaders/)
  })

  it('errors without an image output', () => {
    const t = diffusionTemplate()
    delete t['11']
    expect(inspectTemplate(t, 'x.json').error).toMatch(/SaveImage or PreviewImage/)
  })

  it('errors when the positive prompt is not found', () => {
    const t = diffusionTemplate()
    t['5'].inputs = { clip: ['4', 0] }
    expect(inspectTemplate(t, 'x.json').error).toMatch(/could not find the positive prompt/)
    const t2 = diffusionTemplate()
    t2['8'].inputs = { model: ['3', 0], latent_image: ['7', 0], seed: 1 }
    expect(inspectTemplate(t2, 'x.json').error).toMatch(/could not find the positive prompt/)
  })

  it('accepts text_g and t5xxl fallbacks and prompt', () => {
    const t = diffusionTemplate()
    t['5'] = { class_type: 'CLIPTextEncodeSDXL', inputs: { clip: ['4', 0], text_g: 'g text', text_l: 'l text' } }
    expect(inspectTemplate(t, 'x.json').promptInput?.input).toBe('text_g')
    t['5'] = { class_type: 'CLIPTextEncodeFlux', inputs: { clip: ['4', 0], t5xxl: 't5 text' } }
    expect(inspectTemplate(t, 'x.json').promptInput?.input).toBe('t5xxl')
    t['5'] = { class_type: 'PromptNode', inputs: { prompt: 'p text' } }
    expect(inspectTemplate(t, 'x.json').promptInput).toEqual({ id: '5', input: 'prompt', classType: 'PromptNode' })
  })

  it('walks back to a text node several hops away', () => {
    const t = diffusionTemplate()
    t['30'] = { class_type: 'PrimitiveStringMultiline', inputs: { value: 'far text' } }
    t['5'] = { class_type: 'ConditioningSetTimestepRange', inputs: { conditioning: ['31', 0], start: 0 } }
    t['31'] = { class_type: 'CLIPTextEncode', inputs: { clip: ['4', 0], text: ['30', 0] } }
    const info = inspectTemplate(t, 'x.json')
    expect(info.promptInput).toEqual({ id: '31', input: 'text', classType: 'CLIPTextEncode' })
    expect(info.defaultPrompt).toBe('far text')
    const p = buildGridPrompt(t, info, recipe(), 'mine', 1) as Api
    expect(p['31'].inputs.text).toBe('mine')
    expect(p['30']).toBeUndefined()
  })

  it('ignores linked seeds and warns about a shared text node', () => {
    const t = diffusionTemplate()
    t['8'].inputs.seed = ['50', 0]
    t['50'] = { class_type: 'Seed', inputs: { seed: 3 } }
    t['8'].inputs.negative = ['5', 0]
    const info = inspectTemplate(t, 'x.json')
    expect(info.seedInputs).toEqual([{ id: '50', input: 'seed' }])
    expect(info.warnings.some((w) => /negative prompt uses the same text node/.test(w))).toBe(true)
  })
})

describe('checkTemplateForRecipe', () => {
  it('fits matching kinds', () => {
    expect(checkTemplateForRecipe(inspectTemplate(diffusionTemplate(), 'a'), recipe())).toBeNull()
    expect(checkTemplateForRecipe(inspectTemplate(checkpointTemplate(), 'a'), recipe({ kind: 'checkpoint' }))).toBeNull()
  })
  it('refuses mismatches', () => {
    expect(checkTemplateForRecipe(inspectTemplate(diffusionTemplate(), 'a'), recipe({ kind: 'checkpoint' }))).toMatch(/UNETLoader/)
    expect(checkTemplateForRecipe(inspectTemplate(checkpointTemplate(), 'a'), recipe())).toMatch(/CheckpointLoaderSimple/)
  })
  it('returns the template error', () => {
    const info = inspectTemplate(uiFormat, 'a')
    expect(checkTemplateForRecipe(info, recipe())).toBe(info.error)
  })
})

describe('buildGridPrompt', () => {
  it('replaces a diffusion loader with the merge', () => {
    const t = diffusionTemplate()
    const info = inspectTemplate(t, 'a.json')
    const p = buildGridPrompt(t, info, recipe({ ratio: 0.25 }), 'a blue fox', 99) as Api
    expect(p['1']).toBeUndefined()
    expect(p.mm1.class_type).toBe('UNETLoader')
    expect(p.mm1.inputs.unet_name).toBe('a.safetensors')
    expect(p.mm2.inputs.unet_name).toBe('b.safetensors')
    expect(p.mm3).toEqual({ class_type: 'ModelMergeSimple', inputs: { model1: ['mm1', 0], model2: ['mm2', 0], ratio: 0.25 } })
    expect(p['2'].inputs.model).toEqual(['mm3', 0])
    expect(allLinksValid(p)).toBe(true)
    expect(p['5'].inputs.text).toBe('a blue fox')
    expect(p['6'].inputs.text).toBe('blurry')
    expect(p['8'].inputs.seed).toBe(99)
    expect(p['11']).toEqual({ class_type: 'PreviewImage', inputs: { images: ['10', 0] } })
    expect(Object.values(p).some((n) => n.class_type === 'ModelSave')).toBe(false)
  })

  it('rewires model, clip and vae of a checkpoint loader', () => {
    const t = checkpointTemplate()
    const info = inspectTemplate(t, 'a.json')
    const r = recipe({ kind: 'checkpoint', clip: 'b', vae: 'b' })
    const p = buildGridPrompt(t, info, r, 'x', 5) as Api
    expect(p['4']).toBeUndefined()
    const sub = buildMergeModelSubgraph(r, 'mm')
    expect(p['3'].inputs.model).toEqual(sub.model)
    expect(p['6'].inputs.clip).toEqual(sub.clip)
    expect(p['7'].inputs.clip).toEqual(sub.clip)
    expect(p['8'].inputs.vae).toEqual(sub.vae)
    expect(sub.clip).toEqual(['mm2', 1])
    expect(sub.vae).toEqual(['mm2', 2])
    expect(allLinksValid(p)).toBe(true)
    expect(p['9'].class_type).toBe('PreviewImage')
    expect(p['9'].inputs).toEqual({ images: ['8', 0] })
  })

  it('applies LoRA strength through the chain', () => {
    const t = diffusionTemplate()
    const info = inspectTemplate(t, 'a.json')
    const r = recipe({ loras: [{ name: 'my.safetensors', strength: 0.8, clipStrength: 1 }] })
    const p = buildGridPrompt(t, info, r, 'x', 1) as Api
    const lora = Object.values(p).find((n) => n.inputs.lora_name === 'my.safetensors')
    expect(lora?.class_type).toBe('LoraLoaderModelOnly')
    expect(lora?.inputs.strength_model).toBe(0.8)
    expect(p['2'].inputs.model).toEqual(['mm4', 0])
    expect(allLinksValid(p)).toBe(true)
  })

  it('applies checkpoint LoRAs to model and clip', () => {
    const t = checkpointTemplate()
    const info = inspectTemplate(t, 'a.json')
    const r = recipe({ kind: 'checkpoint', loras: [{ name: 'my.safetensors', strength: 0.5, clipStrength: 0.7 }] })
    const p = buildGridPrompt(t, info, r, 'x', 1) as Api
    const lora = Object.entries(p).find(([, n]) => n.inputs.lora_name === 'my.safetensors')
    expect(lora?.[1].class_type).toBe('LoraLoader')
    expect(lora?.[1].inputs.strength_clip).toBe(0.7)
    expect(p['3'].inputs.model).toEqual([lora?.[0], 0])
    expect(p['6'].inputs.clip).toEqual([lora?.[0], 1])
  })

  it('works for the guider template and keeps PreviewImage', () => {
    const t = guiderTemplate()
    const info = inspectTemplate(t, 'a.json')
    const p = buildGridPrompt(t, info, recipe(), 'hello', 12) as Api
    expect(p['1']).toBeUndefined()
    expect(p['5'].inputs.model).toEqual(['mm3', 0])
    expect(p['8'].inputs.model).toEqual(['mm3', 0])
    expect(p['3'].inputs.text).toBe('hello')
    expect(p['6'].inputs.noise_seed).toBe(12)
    expect(p['13']).toEqual({ class_type: 'PreviewImage', inputs: { images: ['12', 0] } })
    expect(allLinksValid(p)).toBe(true)
  })

  it('avoids id collisions with the template', () => {
    const t = diffusionTemplate()
    t.mm1 = { class_type: 'Note', inputs: {} }
    t.mm_5 = { class_type: 'Note', inputs: {} }
    t['7'].inputs.note1 = ['mm1', 0]
    t['7'].inputs.note2 = ['mm_5', 0]
    const info = inspectTemplate(t, 'a.json')
    const p = buildGridPrompt(t, info, recipe(), 'x', 1) as Api
    expect(p.mm1).toEqual({ class_type: 'Note', inputs: {} })
    expect(p.mm_5).toEqual({ class_type: 'Note', inputs: {} })
    expect(p.mm__1.class_type).toBe('UNETLoader')
    expect((p['2'].inputs.model as string[])[0]).toMatch(/^mm__/)
    expect(allLinksValid(p)).toBe(true)
  })

  it('sets every seed input', () => {
    const t = diffusionTemplate()
    t['40'] = { class_type: 'KSampler', inputs: { model: ['3', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['8', 0], seed: 8 } }
    t['10'].inputs.samples = ['40', 0]
    const info = inspectTemplate(t, 'a.json')
    expect(info.seedInputs).toHaveLength(2)
    const p = buildGridPrompt(t, info, recipe(), 'x', 777) as Api
    expect(p['8'].inputs.seed).toBe(777)
    expect(p['40'].inputs.seed).toBe(777)
  })

  it('does not mutate the template', () => {
    const t = diffusionTemplate()
    const before = JSON.stringify(t)
    const info = inspectTemplate(t, 'a.json')
    const infoBefore = JSON.stringify(info)
    buildGridPrompt(t, info, recipe(), 'changed', 5)
    expect(JSON.stringify(t)).toBe(before)
    expect(JSON.stringify(info)).toBe(infoBefore)
  })

  it('throws on a kind mismatch and on a broken template', () => {
    const t = diffusionTemplate()
    const info = inspectTemplate(t, 'a.json')
    expect(() => buildGridPrompt(t, info, recipe({ kind: 'checkpoint' }), 'x', 1)).toThrow(/UNETLoader/)
    const bad = inspectTemplate(uiFormat, 'ui.json')
    expect(() => buildGridPrompt(t, bad, recipe(), 'x', 1)).toThrow(/API format/)
  })

  it('throws when a diffusion loader output other than 0 is read', () => {
    const t = diffusionTemplate()
    t['10'].inputs.vae = ['1', 2]
    const info = inspectTemplate(t, 'a.json')
    expect(() => buildGridPrompt(t, info, recipe(), 'x', 1)).toThrow(/output 2/)
  })
})

describe('buildMergeModelSubgraph', () => {
  it('keeps buildMergeGraph unchanged and shares its nodes', () => {
    const r = recipe({ ratio: 0.4 })
    const g = buildMergeGraph(r)
    expect(g.prompt).toEqual({
      '1': { class_type: 'UNETLoader', inputs: { unet_name: 'a.safetensors', weight_dtype: 'default' } },
      '2': { class_type: 'UNETLoader', inputs: { unet_name: 'b.safetensors', weight_dtype: 'default' } },
      '3': { class_type: 'ModelMergeSimple', inputs: { model1: ['1', 0], model2: ['2', 0], ratio: 0.4 } },
      '4': { class_type: 'ModelSave', inputs: { model: ['3', 0], filename_prefix: 'diffusion_models/m' } }
    })
    const s = buildMergeModelSubgraph(r, 'p')
    expect(Object.keys(s.nodes)).toEqual(['p1', 'p2', 'p3'])
    expect(s.model).toEqual(['p3', 0])
    expect(s.clip).toBeNull()
    expect(s.vae).toBeNull()
  })

  it('checkpoint graph still ends in CheckpointSave', () => {
    const g = buildMergeGraph(recipe({ kind: 'checkpoint' }))
    expect(g.prompt['4']).toEqual({ class_type: 'CLIPMergeSimple', inputs: { clip1: ['1', 1], clip2: ['2', 1], ratio: 0.5 } })
    expect(g.prompt['5']).toEqual({
      class_type: 'CheckpointSave',
      inputs: { model: ['3', 0], clip: ['4', 0], vae: ['1', 2], filename_prefix: 'checkpoints/m' }
    })
    const s = buildMergeModelSubgraph(recipe({ kind: 'checkpoint' }), 'q')
    expect(s.clip).toEqual(['q4', 0])
    expect(s.vae).toEqual(['q1', 2])
  })
})

function qwenEditTemplate(): Api {
  return {
    '5:1': { class_type: 'UNETLoader', inputs: { unet_name: 'q.safetensors', weight_dtype: 'default' } },
    '5:2': { class_type: 'CLIPLoader', inputs: { clip_name: 'te', type: 'qwen_image' } },
    '5:3': { class_type: 'VAELoader', inputs: { vae_name: 'vae' } },
    '9': { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['5:1', 0], shift: 3 } },
    '13': { class_type: 'LoadImage', inputs: { image: 'in.png' } },
    '11': { class_type: 'RulZLoadImagePath', inputs: { path: 'in.png' } },
    '117': { class_type: 'PreviewImage', inputs: { images: ['11', 0] } },
    '10': {
      class_type: 'TextEncodeQwenImage21',
      inputs: { prompt: ['21', 0], negative_prompt: '', clip: ['5:2', 0], 'images.image_1': ['13', 0], vae: ['5:3', 0] }
    },
    '21': { class_type: 'PreviewAny', inputs: { source: ['15', 0] } },
    '15': { class_type: 'RulZOpenRouterVision', inputs: { system_prompt: ['17', 0], prompt: ['120', 0], seed: ['118', 0] } },
    '17': { class_type: 'PrimitiveStringMultiline', inputs: { value: 'system' } },
    '120': { class_type: 'StringConcatenate', inputs: { string_a: ['119', 0], string_b: ['12', 0] } },
    '119': { class_type: 'PrimitiveString', inputs: { value: 'a' } },
    '12': { class_type: 'PrimitiveString', inputs: { value: 'b' } },
    '118': { class_type: 'SeedNode', inputs: { seed: 91 } },
    '7': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '6': {
      class_type: 'KSampler',
      inputs: { seed: ['118', 0], model: ['9', 0], positive: ['10', 0], negative: ['10', 1], latent_image: ['7', 0], steps: 8 }
    },
    '20': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['5:3', 0] } },
    '19': { class_type: 'Image Comparer (rgthree)', inputs: { image_a: ['11', 0], image_b: ['20', 0] } },
    '18': { class_type: 'SaveImage', inputs: { filename_prefix: ['11', 2], images: ['20', 0] } }
  }
}

function kleinEditTemplate(): Api {
  return {
    '155': { class_type: 'UNETLoader', inputs: { unet_name: 'k.safetensors', weight_dtype: 'default' } },
    '178': { class_type: 'ModelAttentionBackend', inputs: { model: ['155', 0], backend: 'x' } },
    '156': { class_type: 'CLIPLoader', inputs: { clip_name: 'te', type: 'flux2' } },
    '174': { class_type: 'PrimitiveString', inputs: { value: 'llm prompt' } },
    '171': { class_type: 'RulZGenerateText', inputs: { prompt: ['174', 0] } },
    '175': { class_type: 'PreviewAny', inputs: { source: ['171', 0] } },
    '165': { class_type: 'CLIPTextEncode', inputs: { text: ['175', 0], clip: ['156', 0] } },
    '164': { class_type: 'ReferenceLatent', inputs: { conditioning: ['165', 0] } },
    '166': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['165', 0] } },
    '183': { class_type: 'Flux2KleinRefLatentController', inputs: { model: ['178', 0], conditioning: ['164', 0] } },
    '158': { class_type: 'CFGGuider', inputs: { model: ['183', 0], positive: ['183', 1], negative: ['166', 0], cfg: 1 } },
    '154': { class_type: 'RandomNoise', inputs: { noise_seed: 5 } },
    '153': { class_type: 'KSamplerSelect', inputs: { sampler_name: 'euler' } },
    '184': { class_type: 'BasicScheduler', inputs: { model: ['183', 0], scheduler: 'simple', steps: 4, denoise: 1 } },
    '185': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '152': {
      class_type: 'SamplerCustomAdvanced',
      inputs: { noise: ['154', 0], guider: ['158', 0], sampler: ['153', 0], sigmas: ['184', 0], latent_image: ['185', 0] }
    },
    '157': { class_type: 'VAELoader', inputs: { vae_name: 'vae' } },
    '176': { class_type: 'VAEDecode', inputs: { samples: ['152', 0], vae: ['157', 0] } },
    '9': { class_type: 'SaveImage', inputs: { images: ['176', 0], filename_prefix: 'k' } },
    '179:77': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sam3.1.safetensors' } },
    '179:75': { class_type: 'SAM3_Detect', inputs: { model: ['179:77', 0], image: ['176', 0] } },
    '180': { class_type: 'MaskIntelligentCrop', inputs: { mask: ['179:75', 0], image: ['176', 0] } },
    '181': { class_type: 'PreviewImage', inputs: { images: ['180', 0] } }
  }
}

function kreaTwoPassTemplate(): Api {
  return {
    '4': { class_type: 'UNETLoader', inputs: { unet_name: 'krea.safetensors', weight_dtype: 'default' } },
    '21': { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['4', 0], shift: 3 } },
    '5': { class_type: 'CLIPLoader', inputs: { clip_name: 'te', type: 'x' } },
    '8': { class_type: 'VAELoader', inputs: { vae_name: 'vae' } },
    '9': { class_type: 'CLIPTextEncode', inputs: { text: 'a harbor', clip: ['5', 0] } },
    '10': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['9', 0] } },
    '7': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '6': {
      class_type: 'KSampler',
      inputs: { model: ['21', 0], positive: ['9', 0], negative: ['10', 0], latent_image: ['7', 0], seed: 3 }
    },
    '22': { class_type: 'RulZVAEDecodeGrid', inputs: { samples: ['6', 0], vae: ['8', 0] } },
    '13': { class_type: 'SaveImage', inputs: { images: ['22', 0], filename_prefix: 'k' } },
    '24': { class_type: 'PreviewImage', inputs: { images: ['22', 1] } },
    '14': { class_type: 'ImageScaleBy', inputs: { image: ['22', 0], scale_by: 2 } },
    '16': { class_type: 'VAEEncode', inputs: { pixels: ['14', 0], vae: ['8', 0] } },
    '17': {
      class_type: 'KSampler',
      inputs: { model: ['21', 0], positive: ['9', 0], negative: ['10', 0], latent_image: ['16', 0], denoise: 0.3, seed: 3 }
    },
    '23': { class_type: 'RulZVAEDecodeGrid', inputs: { samples: ['17', 0], vae: ['8', 0] } },
    '25': { class_type: 'PreviewImage', inputs: { images: ['23', 1] } }
  }
}

const sorted = (p: Record<string, unknown>): string[] => Object.keys(p).sort()

describe('real-world shapes', () => {
  it('Qwen image edit: prompt via linked encoder, LLM branch pruned, seed through SeedNode', () => {
    const t = qwenEditTemplate()
    const info = inspectTemplate(t, 'qwen.json')
    expect(info.error).toBeNull()
    expect(info.modelLoader).toEqual({ id: '5:1', classType: 'UNETLoader' })
    expect(info.promptInput).toEqual({ id: '10', input: 'prompt', classType: 'TextEncodeQwenImage21' })
    expect(info.defaultPrompt).toBeNull()
    expect(info.warnings).toContain('The prompt comes from PreviewAny; the grid replaces it with your prompts.')
    expect(info.warnings.some((w) => /negative prompt uses the same/.test(w))).toBe(false)
    expect(info.imageOutputs).toEqual(['18'])
    expect(info.warnings.some((w) => /Only 18 \(SaveImage\) is rendered; other outputs are left out/.test(w))).toBe(true)
    expect(info.seedInputs).toEqual([{ id: '118', input: 'seed' }])
    const p = buildGridPrompt(t, info, recipe(), 'edit it', 1234) as Api
    expect(p['10'].inputs.prompt).toBe('edit it')
    expect(p['118'].inputs.seed).toBe(1234)
    expect(p['18']).toEqual({ class_type: 'PreviewImage', inputs: { images: ['20', 0] } })
    const classes = Object.values(p).map((n) => n.class_type)
    for (const bad of ['RulZOpenRouterVision', 'PreviewAny', 'Image Comparer (rgthree)', 'StringConcatenate']) {
      expect(classes).not.toContain(bad)
    }
    expect(classes.filter((c) => c === 'PreviewImage')).toHaveLength(1)
    expect(p['5:1']).toBeUndefined()
    expect(sorted(p)).toEqual(['10', '118', '13', '18', '20', '5:2', '5:3', '6', '7', '9', 'mm1', 'mm2', 'mm3'].sort())
    expect(p['9'].inputs.model).toEqual(['mm3', 0])
    expect(allLinksValid(p)).toBe(true)
  })

  it('Flux2 Klein edit: a SAM 3 crop that feeds the reference latent keeps its loader but is not the model loader', () => {
    const t = kleinEditTemplate()
    t['76'] = { class_type: 'LoadImage', inputs: { image: 'in.png' } }
    t['179:75'] = { class_type: 'SAM3_Detect', inputs: { model: ['179:77', 0], image: ['76', 0] } }
    t['180'] = { class_type: 'MaskIntelligentCrop', inputs: { mask: ['179:75', 0], image: ['76', 0] } }
    t['168'] = { class_type: 'ResizeImageV2Advanced', inputs: { image_1: ['180', 0], width: 512 } }
    t['167'] = { class_type: 'VAEEncode', inputs: { pixels: ['168', 0], vae: ['157', 0] } }
    t['164'] = { class_type: 'ReferenceLatent', inputs: { conditioning: ['165', 0], latent: ['167', 0] } }
    const info = inspectTemplate(t, 'klein-sam.json')
    expect(info.error).toBeNull()
    expect(info.modelLoader).toEqual({ id: '155', classType: 'UNETLoader' })
    expect(info.warnings.some((w) => /CheckpointLoaderSimple 179:77/.test(w))).toBe(true)
    const p = buildGridPrompt(t, info, recipe(), 'klein', 8) as Api
    expect(p['179:77']).toBeDefined()
    expect(p['179:75'].inputs.model).toEqual(['179:77', 0])
    expect(p['178'].inputs.model).toEqual(['mm3', 0])
    expect(p['155']).toBeUndefined()
    expect(allLinksValid(p)).toBe(true)
  })

  it('Flux2 Klein edit: the sam3 loader is ignored, guider prompt found, side branches pruned', () => {
    const t = kleinEditTemplate()
    const info = inspectTemplate(t, 'klein.json')
    expect(info.error).toBeNull()
    expect(info.modelLoader).toEqual({ id: '155', classType: 'UNETLoader' })
    expect(info.warnings.some((w) => /CheckpointLoaderSimple 179:77/.test(w) && /ignored/.test(w))).toBe(true)
    expect(info.promptInput).toEqual({ id: '165', input: 'text', classType: 'CLIPTextEncode' })
    expect(info.defaultPrompt).toBeNull()
    expect(info.warnings).toContain('The prompt comes from PreviewAny; the grid replaces it with your prompts.')
    expect(info.warnings.some((w) => /negative prompt uses the same/.test(w))).toBe(false)
    expect(info.imageOutputs).toEqual(['9'])
    expect(info.warnings.some((w) => /Only 9 \(SaveImage\)/.test(w))).toBe(true)
    expect(info.seedInputs).toEqual([{ id: '154', input: 'noise_seed' }])
    expect(checkTemplateForRecipe(info, recipe())).toBeNull()
    const p = buildGridPrompt(t, info, recipe(), 'klein', 8) as Api
    expect(p['165'].inputs.text).toBe('klein')
    expect(p['154'].inputs.noise_seed).toBe(8)
    for (const gone of ['155', '175', '171', '174', '179:77', '179:75', '180', '181']) expect(p[gone]).toBeUndefined()
    expect(p['178'].inputs.model).toEqual(['mm3', 0])
    expect(p['9'].class_type).toBe('PreviewImage')
    expect(allLinksValid(p)).toBe(true)
  })

  it('Krea two-pass: no shared-negative warning, only pass 1 and its output kept', () => {
    const t = kreaTwoPassTemplate()
    const info = inspectTemplate(t, 'krea.json')
    expect(info.error).toBeNull()
    expect(info.promptInput).toEqual({ id: '9', input: 'text', classType: 'CLIPTextEncode' })
    expect(info.defaultPrompt).toBe('a harbor')
    expect(info.warnings.some((w) => /negative prompt uses the same/.test(w))).toBe(false)
    expect(info.imageOutputs).toEqual(['13'])
    expect(info.seedInputs).toEqual([{ id: '6', input: 'seed' }])
    const p = buildGridPrompt(t, info, recipe(), 'x', 9) as Api
    for (const gone of ['4', '24', '25', '23', '17', '16', '14']) expect(p[gone]).toBeUndefined()
    for (const kept of ['21', '5', '9', '10', '7', '6', '22', '13', '8']) expect(p[kept]).toBeDefined()
    expect(p['13'].class_type).toBe('PreviewImage')
    expect(p['6'].inputs.seed).toBe(9)
    expect(allLinksValid(p)).toBe(true)
  })

  it('ignores an unrelated sampler and text node earlier in the file', () => {
    const t = diffusionTemplate()
    const withOther: Api = {
      '0': { class_type: 'CLIPTextEncode', inputs: { text: 'other branch', clip: ['4', 0] } },
      '00': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'z.safetensors' } },
      '000': {
        class_type: 'KSampler',
        inputs: { model: ['00', 0], positive: ['0', 0], negative: ['0', 0], latent_image: ['7', 0], seed: 1 }
      },
      ...t
    }
    const info = inspectTemplate(withOther, 'x.json')
    expect(info.error).toBeNull()
    expect(info.modelLoader?.id).toBe('1')
    expect(info.promptInput?.id).toBe('5')
    expect(info.defaultPrompt).toBe('a red fox')
    expect(info.seedInputs).toEqual([{ id: '8', input: 'seed' }])
  })

  it('takes the first pass when two passes use different prompts, and errors when it cannot tell', () => {
    const t = kreaTwoPassTemplate()
    t['30'] = { class_type: 'CLIPTextEncode', inputs: { text: 'second', clip: ['5', 0] } }
    t['17'].inputs.positive = ['30', 0]
    t['13'].inputs.images = ['23', 0]
    const info = inspectTemplate(t, 'x.json')
    expect(info.error).toBeNull()
    expect(info.promptInput?.id).toBe('9')
    const t2 = kreaTwoPassTemplate()
    t2['30'] = { class_type: 'CLIPTextEncode', inputs: { text: 'second', clip: ['5', 0] } }
    t2['31'] = {
      class_type: 'KSampler',
      inputs: { model: ['21', 0], positive: ['30', 0], negative: ['10', 0], latent_image: ['7', 0], seed: 1 }
    }
    t2['32'] = { class_type: 'ImageBlend', inputs: { image1: ['22', 0], image2: ['31', 0] } }
    t2['13'].inputs.images = ['32', 0]
    expect(inspectTemplate(t2, 'x.json').error).toMatch(/several samplers/)
  })
})

describe('encoders with sibling prompt inputs', () => {
  it('SDXL encoder: text_g and text_l both get the grid prompt', () => {
    const t = diffusionTemplate()
    t['30'] = { class_type: 'PrimitiveString', inputs: { value: 'linked l' } }
    t['5'] = {
      class_type: 'CLIPTextEncodeSDXL',
      inputs: { clip: ['4', 0], text_g: 'g text', text_l: ['30', 0], width: 1024, height: 1024 }
    }
    const info = inspectTemplate(t, 'x.json')
    expect(info.error).toBeNull()
    expect(info.promptInput).toEqual({ id: '5', input: 'text_g', classType: 'CLIPTextEncodeSDXL' })
    const p = buildGridPrompt(t, info, recipe(), 'new prompt', 1) as Api
    expect(p['5'].inputs.text_g).toBe('new prompt')
    expect(p['5'].inputs.text_l).toBe('new prompt')
    expect(p['30']).toBeUndefined()
    expect(allLinksValid(p)).toBe(true)
    expect((t['5'].inputs.text_l as string[])[0]).toBe('30')
  })

  it('Flux encoder: clip_l and t5xxl both get the grid prompt, other inputs stay', () => {
    const t = diffusionTemplate()
    t['5'] = {
      class_type: 'CLIPTextEncodeFlux',
      inputs: { clip: ['4', 0], clip_l: 'old l', t5xxl: 'old t5', guidance: 3.5 }
    }
    const info = inspectTemplate(t, 'x.json')
    expect(info.promptInput?.input).toBe('t5xxl')
    expect(info.defaultPrompt).toBe('old t5')
    const p = buildGridPrompt(t, info, recipe(), 'fresh', 1) as Api
    expect(p['5'].inputs.t5xxl).toBe('fresh')
    expect(p['5'].inputs.clip_l).toBe('fresh')
    expect(p['5'].inputs.guidance).toBe(3.5)
    expect(p['6'].inputs.text).toBe('blurry')
  })

  it('does not touch negative_prompt of a two-prompt encoder', () => {
    const t = qwenEditTemplate()
    const info = inspectTemplate(t, 'q.json')
    const p = buildGridPrompt(t, info, recipe(), 'edit', 1) as Api
    expect(p['10'].inputs.prompt).toBe('edit')
    expect(p['10'].inputs.negative_prompt).toBe('')
  })
})
