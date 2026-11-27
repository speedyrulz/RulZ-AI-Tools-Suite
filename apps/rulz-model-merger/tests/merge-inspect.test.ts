import { describe, expect, it } from 'vitest'
import {
  blockNodeOptions,
  checkMerge,
  describeModelFile,
  normalizeKeys
} from '../src/main/merge/inspect'
import type { SafetensorsHeader, SafetensorsTensor } from '../src/main/models/safetensors'

function header(
  tensors: Record<string, [string, number[]]>,
  metadata: Record<string, string> = {},
  fileBytes = 1000
): SafetensorsHeader {
  const t: Record<string, SafetensorsTensor> = {}
  for (const [k, [dtype, shape]] of Object.entries(tensors)) t[k] = { dtype, shape, offsets: [0, 0] }
  return { tensors: t, metadata, headerBytes: 10, fileBytes, headerSha256: 'x' }
}

/** n layers named `<prefix>block.<i>.weight`, all BF16 [4,4]. */
function layers(prefix: string, n: number, shape = [4, 4]): Record<string, [string, number[]]> {
  const out: Record<string, [string, number[]]> = {}
  for (let i = 0; i < n; i++) out[`${prefix}blocks.${i}.weight`] = ['BF16', shape]
  return out
}

const objectInfo: Record<string, unknown> = {
  ModelMergeSimple: { input: { required: { model1: ['MODEL'], model2: ['MODEL'], ratio: ['FLOAT', {}] } } },
  ModelMergeAdd: { input: { required: { model1: ['MODEL'], model2: ['MODEL'] } } },
  ModelMergeSubtract: { input: { required: { model1: ['MODEL'], model2: ['MODEL'], multiplier: ['FLOAT', {}] } } },
  ModelMergeBlocks: {
    input: { required: { model1: ['MODEL'], model2: ['MODEL'], input: ['FLOAT', {}], middle: ['FLOAT', {}], out: ['FLOAT', {}] } }
  },
  ModelMergeFakeFlux: {
    input: {
      required: {
        model1: ['MODEL'],
        model2: ['MODEL'],
        'double_blocks.0.': ['FLOAT', {}],
        'double_blocks.1.': ['FLOAT', {}],
        'single_blocks.0.': ['FLOAT', {}],
        'single_blocks.1.': ['FLOAT', {}]
      }
    }
  },
  ModelMergeNoFloat: { input: { required: { model1: ['MODEL'], model2: ['MODEL'] } } },
  ModelMergeOneModel: { input: { required: { model1: ['MODEL'], x: ['FLOAT', {}] } } },
  KSampler: { input: { required: { model: ['MODEL'] } } }
}

function src(name: string, h: SafetensorsHeader | null, error?: string) {
  return { info: describeModelFile(name, h, error), header: h }
}

describe('normalizeKeys', () => {
  it('detects the three layouts', () => {
    const ckpt = header({ ...layers('model.diffusion_model.', 3), 'first_stage_model.x': ['F32', [1]], 'cond_stage_model.y': ['F32', [1]] })
    const n1 = normalizeKeys(ckpt)
    expect(n1.prefix).toBe('model.diffusion_model.')
    expect([...n1.keys.keys()]).toEqual(['blocks.0.weight', 'blocks.1.weight', 'blocks.2.weight']) // vae / clip dropped

    const n2 = normalizeKeys(header(layers('diffusion_model.', 2)))
    expect(n2.prefix).toBe('diffusion_model.')
    expect([...n2.keys.keys()]).toEqual(['blocks.0.weight', 'blocks.1.weight'])

    const n3 = normalizeKeys(header(layers('', 2)))
    expect(n3.prefix).toBe('')
    expect([...n3.keys.keys()]).toEqual(['blocks.0.weight', 'blocks.1.weight'])
  })
})

describe('describeModelFile', () => {
  it('describes a plain file', () => {
    const info = describeModelFile('m.safetensors', header(layers('', 3), {}, 4242))
    expect(info).toMatchObject({
      name: 'm.safetensors',
      readable: true,
      error: null,
      sizeBytes: 4242,
      tensorCount: 3,
      dtypes: { BF16: 3 },
      quantized: false,
      quantLabel: null,
      keyPrefix: ''
    })
  })

  it('reports an unreadable file with the reason', () => {
    const info = describeModelFile('gone.safetensors', null, 'File not found.')
    expect(info.readable).toBe(false)
    expect(info.error).toBe('File not found.')
    expect(info.sizeBytes).toBeNull()
  })

  it('detects int8 weights with scale tensors', () => {
    const info = describeModelFile(
      'plain-name.safetensors',
      header({
        'layer.weight': ['I8', [4, 4]],
        'layer.weight_scale': ['F32', [4]],
        'norm.weight': ['BF16', [4]]
      })
    )
    expect(info.quantized).toBe(true)
    expect(info.quantLabel).toBe('int8')
    expect(info.dtypes).toEqual({ I8: 1, F32: 1, BF16: 1 })
  })

  it('does not call an I8 weight without scales quantized', () => {
    expect(describeModelFile('m.safetensors', header({ 'layer.weight': ['I8', [4, 4]] })).quantized).toBe(false)
  })

  it('detects fp8 and fp4 / nvfp4', () => {
    const fp8 = describeModelFile('m.safetensors', header({ 'a.weight': ['F8_E4M3', [4, 4]], 'a.weight_scale': ['F32', [1]] }))
    expect(fp8.quantLabel).toBe('fp8')
    const packed = header({ 'a.weight': ['U8', [4, 2]], 'a.weight_scale': ['F8_E4M3', [4, 1]] })
    expect(describeModelFile('m.safetensors', packed).quantLabel).toBe('fp4') // unsure: conservative name
    expect(describeModelFile('thing_nvfp4_v1.safetensors', packed).quantLabel).toBe('nvfp4') // weak name hint
    const withTensorScale = header({
      'a.weight': ['U8', [4, 2]],
      'a.weight_scale': ['F8_E4M3', [4, 1]],
      'a.weight_scale_2': ['F32', [1]]
    })
    expect(describeModelFile('m.safetensors', withTensorScale).quantLabel).toBe('nvfp4')
  })

  it('labels several formats as mixed', () => {
    const info = describeModelFile(
      'm.safetensors',
      header({
        'a.weight': ['I8', [4, 4]],
        'a.weight_scale': ['F32', [1]],
        'b.weight': ['F8_E4M3', [4, 4]],
        'b.weight_scale': ['F32', [1]]
      })
    )
    expect(info.quantized).toBe(true)
    expect(info.quantLabel).toBe('mixed')
  })

  it('uses a quantization entry in the metadata', () => {
    const info = describeModelFile('m.safetensors', header({ 'a.weight': ['BF16', [4, 4]] }, { _quantization_metadata: '{"format":"int8"}' }))
    expect(info.quantized).toBe(true)
    expect(info.quantLabel).toBe('int8')
  })

  it('ignores quantization-like keys outside the diffusion model of a checkpoint', () => {
    const info = describeModelFile(
      'm.safetensors',
      header({
        ...layers('model.diffusion_model.', 2),
        'cond_stage_model.w.weight': ['I8', [4, 4]],
        'cond_stage_model.w.weight_scale': ['F32', [1]]
      })
    )
    expect(info.quantized).toBe(false)
  })
})

describe('blockNodeOptions', () => {
  it('lists block nodes, excludes simple/add/subtract and nodes without two MODELs or FLOATs', () => {
    const opts = blockNodeOptions(objectInfo, null)
    expect(opts.map((o) => o.node)).toEqual(['ModelMergeBlocks', 'ModelMergeFakeFlux'])
    expect(opts[0].inputs).toEqual(['input', 'middle', 'out'])
    expect(opts.every((o) => o.match === null)).toBe(true)
  })

  it('scores against model A and sorts best first', () => {
    const fluxLike = header({
      'model.diffusion_model.double_blocks.0.img.weight': ['BF16', [2]],
      'model.diffusion_model.double_blocks.1.img.weight': ['BF16', [2]],
      'model.diffusion_model.single_blocks.0.l.weight': ['BF16', [2]],
      'model.diffusion_model.single_blocks.1.l.weight': ['BF16', [2]]
    })
    const opts = blockNodeOptions(objectInfo, fluxLike)
    expect(opts[0]).toMatchObject({ node: 'ModelMergeFakeFlux', match: 1 })
    expect(opts[1]).toMatchObject({ node: 'ModelMergeBlocks', match: 0 })

    const unet = header({
      'input_blocks.0.weight': ['BF16', [2]],
      'middle_block.0.weight': ['BF16', [2]],
      'output_blocks.0.weight': ['BF16', [2]]
    })
    const o2 = blockNodeOptions(objectInfo, unet)
    expect(o2[0]).toMatchObject({ node: 'ModelMergeBlocks', match: 1 })
    expect(o2[1].match).toBe(0)
  })

  it('fits a model with fewer blocks than the node has inputs, and lists only the inputs it uses', () => {
    const fewer = header({ 'double_blocks.0.w': ['BF16', [2]], 'single_blocks.0.w': ['BF16', [2]] })
    const o = blockNodeOptions(objectInfo, fewer).find((x) => x.node === 'ModelMergeFakeFlux')
    expect(o?.match).toBe(1)
    expect(o?.activeInputs).toEqual(['double_blocks.0.', 'single_blocks.0.'])
    expect(o?.inputs).toHaveLength(4)
  })

  it('gives the share of A layers a node controls', () => {
    const partial = header({ 'double_blocks.0.w': ['BF16', [2]], 'txt_in.w': ['BF16', [2]], 'img_in.w': ['BF16', [2]], 'final.w': ['BF16', [2]] })
    expect(blockNodeOptions(objectInfo, partial).find((x) => x.node === 'ModelMergeFakeFlux')?.match).toBe(0.25)
  })

  it('lists every input as active when A was not read', () => {
    const o = blockNodeOptions(objectInfo, null).find((x) => x.node === 'ModelMergeFakeFlux')
    expect(o?.activeInputs).toEqual(o?.inputs)
  })
})

describe('checkMerge', () => {
  const base = { kind: 'diffusion' as const, objectInfo, ramFreeBytes: null, loras: [] }

  it('passes for two matching files and suggests the block node', () => {
    const a = header(layers('double_blocks.0.x.', 10, [4, 4]))
    const res = checkMerge({
      ...base,
      mode: 'weighted',
      a: src('a.safetensors', header({ 'model.diffusion_model.double_blocks.0.w': ['BF16', [1]], 'model.diffusion_model.double_blocks.1.w': ['BF16', [1]], 'model.diffusion_model.single_blocks.0.w': ['BF16', [1]], 'model.diffusion_model.single_blocks.1.w': ['BF16', [1]] })),
      b: src('b.safetensors', header({ 'model.diffusion_model.double_blocks.0.w': ['BF16', [1]], 'model.diffusion_model.double_blocks.1.w': ['BF16', [1]], 'model.diffusion_model.single_blocks.0.w': ['BF16', [1]], 'model.diffusion_model.single_blocks.1.w': ['BF16', [1]] })),
      c: null
    })
    expect(a).toBeTruthy()
    expect(res.problems).toEqual([])
    expect(res.compatible).toBe(true)
    expect(res.suggestedBlockNode).toBe('ModelMergeFakeFlux')
    expect(res.blockNodes[0].node).toBe('ModelMergeFakeFlux')
  })

  it('blocks missing names, the same file twice, and addDifference without C', () => {
    const h = header(layers('', 5))
    const none = checkMerge({ ...base, mode: 'weighted', a: null, b: null, c: null })
    expect(none.problems).toEqual(['Choose model A.', 'Choose model B.'])
    expect(none.compatible).toBe(false)

    const same = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', h), b: src('a.safetensors', h), c: null })
    expect(same.problems.some((p) => /same file/.test(p))).toBe(true)

    const noC = checkMerge({ ...base, mode: 'addDifference', a: src('a.safetensors', h), b: src('b.safetensors', h), c: null })
    expect(noC.problems.some((p) => /model C/.test(p))).toBe(true)
    expect(noC.compatible).toBe(false)
  })

  it('flags different model types by key overlap', () => {
    const res = checkMerge({
      ...base,
      mode: 'weighted',
      a: src('a.safetensors', header(layers('x.', 10))),
      b: src('b.safetensors', header({ ...layers('x.', 5), ...layers('y.', 5) })),
      c: null
    })
    expect(res.problems).toEqual(['A and B look like different model types (only 50% of their layers match).'])
    expect(res.compatible).toBe(false)
  })

  it('compares only the diffusion model of checkpoints', () => {
    const res = checkMerge({
      ...base,
      kind: 'checkpoint',
      mode: 'weighted',
      a: src('a.safetensors', header({ ...layers('model.diffusion_model.', 10), 'cond.a': ['F32', [1]] })),
      b: src('b.safetensors', header({ ...layers('model.diffusion_model.', 10), 'cond.zzz': ['F32', [1]], 'vae.q': ['F32', [1]] })),
      c: null
    })
    expect(res.problems).toEqual([])
    expect(res.compatible).toBe(true)
  })

  it('names up to three keys with a different shape', () => {
    const a = layers('', 10)
    const bT = layers('', 10)
    for (let i = 0; i < 5; i++) bT[`blocks.${i}.weight`] = ['BF16', [8, 8]]
    const res = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', header(a)), b: src('b.safetensors', header(bT)), c: null })
    expect(res.problems).toHaveLength(1)
    expect(res.problems[0]).toMatch(/5 layers with a different shape/)
    expect(res.problems[0]).toContain('blocks.0.weight, blocks.1.weight, blocks.2.weight')
    expect(res.problems[0]).not.toContain('blocks.3.weight')
  })

  it('checks B against C in difference mode', () => {
    const h = header(layers('', 10))
    const res = checkMerge({
      ...base,
      mode: 'addDifference',
      a: src('a.safetensors', h),
      b: src('b.safetensors', h),
      c: src('c.safetensors', header(layers('other.', 10)))
    })
    expect(res.problems[0]).toMatch(/^B and C look like different model types/)
  })

  it('warns about an unreadable source and returns compatible null', () => {
    const res = checkMerge({
      ...base,
      mode: 'weighted',
      a: src('a.safetensors', header(layers('', 4))),
      b: src('b.safetensors', null, 'It is on a remote server.'),
      c: null
    })
    expect(res.problems).toEqual([])
    expect(res.compatible).toBeNull()
    expect(res.warnings[0]).toMatch(/^B could not be read \(It is on a remote server\.\)/)
    expect(res.warnings[0]).toMatch(/ComfyUI will report a mismatch itself/)
  })

  it('warns about quantized sources and different formats', () => {
    const q = header({
      'blocks.0.weight': ['I8', [4, 4]],
      'blocks.0.weight_scale': ['F32', [1]],
      'blocks.1.weight': ['I8', [4, 4]],
      'blocks.1.weight_scale': ['F32', [1]]
    })
    const plain = header({ 'blocks.0.weight': ['BF16', [4, 4]], 'blocks.1.weight': ['BF16', [4, 4]] })
    const res = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', q), b: src('b.safetensors', plain), c: null })
    expect(res.problems).toEqual([])
    expect(res.warnings).toContain(
      'A is int8: weights are merged, then saved as int8 again, which adds a little loss. Merging the bf16 originals is best.'
    )
    expect(res.warnings.some((w) => /A: int8, B: unquantized.*A's format/.test(w))).toBe(true)
    expect(res.compatible).toBe(true)
  })

  it('estimates RAM and warns above 80 percent of the free memory', () => {
    const GiB = 1024 ** 3
    const a = header(layers('', 4), {}, 10 * GiB)
    const b = header(layers('', 4), {}, 10 * GiB)
    const res = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', a), b: src('b.safetensors', b), c: null, ramFreeBytes: 25 * GiB })
    expect(res.ramNeededBytes).toBe(Math.round(20 * GiB * 1.2))
    expect(res.ramFreeBytes).toBe(25 * GiB)
    expect(res.warnings.some((w) => /about 24\.0 GiB .* 25\.0 GiB is free/.test(w))).toBe(true)

    const roomy = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', a), b: src('b.safetensors', b), c: null, ramFreeBytes: 64 * GiB })
    expect(roomy.warnings).toEqual([])
    const unknown = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', a), b: src('b.safetensors', b), c: null })
    expect(unknown.warnings).toEqual([])
  })

  it('uses factor 1.5 when a source is quantized and null RAM when nothing is readable', () => {
    const q = header({ 'blocks.0.weight': ['I8', [4, 4]], 'blocks.0.weight_scale': ['F32', [1]] }, {}, 1000)
    const res = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', q), b: src('b.safetensors', q), c: null })
    expect(res.ramNeededBytes).toBe(3000)
    const none = checkMerge({ ...base, mode: 'weighted', a: src('a.safetensors', null), b: src('b.safetensors', null), c: null })
    expect(none.ramNeededBytes).toBeNull()
    expect(none.compatible).toBeNull()
  })

  it('suggests no block node when no option matches 90 percent', () => {
    const res = checkMerge({
      ...base,
      mode: 'blocks',
      a: src('a.safetensors', header(layers('', 4))),
      b: src('b.safetensors', header(layers('', 4))),
      c: null
    })
    expect(res.suggestedBlockNode).toBeNull()
  })
})
