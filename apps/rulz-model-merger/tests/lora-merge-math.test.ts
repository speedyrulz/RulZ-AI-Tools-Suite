import { describe, expect, it } from 'vitest'
import type { LoraMergeRecipe } from '../src/shared/loraMerge'
import { shrinkLayer, svdSmall, symmetricEigen } from '../src/main/loras/linalg'
import {
  checkLoraMerge,
  describeMergeInput,
  mergeLoras,
  parseLoraLayers,
  verifyMergedLora
} from '../src/main/loras/merge'
import type { SafetensorsHeader } from '../src/main/models/safetensors'
import { encodeValues, tensorValues } from '../src/main/models/tensors'

// ---------- helpers ----------

function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function randMat(rows: number, cols: number, seed: number): Float64Array {
  const r = rng(seed)
  const m = new Float64Array(rows * cols)
  for (let i = 0; i < m.length; i++) m[i] = r() * 2 - 1
  return m
}

function matmul(a: Float64Array, b: Float64Array, m: number, k: number, n: number): Float64Array {
  const c = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) for (let j = 0; j < n; j++) c[i * n + j] += a[i * k + p] * b[p * n + j]
  return c
}

function fro(a: Float64Array): number {
  let s = 0
  for (const x of a) s += x * x
  return Math.sqrt(s)
}

function relErr(a: Float64Array, b: Float64Array): number {
  let d = 0
  for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2
  return Math.sqrt(d) / (fro(b) || 1)
}

interface Spec {
  key: string
  dtype?: 'BF16' | 'F16' | 'F32'
  shape: number[]
  values: Float64Array
}

function makeFile(specs: Spec[], metadata: Record<string, string> = {}): { header: SafetensorsHeader; data: Buffer } {
  const tensors: SafetensorsHeader['tensors'] = {}
  const parts: Buffer[] = []
  let off = 0
  for (const s of specs) {
    const dtype = s.dtype ?? 'F32'
    const buf = encodeValues(s.values, dtype)
    tensors[s.key] = { dtype, shape: s.shape, offsets: [off, off + buf.length] }
    parts.push(buf)
    off += buf.length
  }
  const data = Buffer.concat(parts)
  return { header: { tensors, metadata, headerBytes: 100, fileBytes: data.length + 108, headerSha256: 'h' }, data }
}

interface LayerSpec {
  base: string
  out: number
  inn: number
  rank: number
  alpha?: number | null
  conv?: [number, number]
  seed: number
}

function loraSpecs(layers: LayerSpec[], style: { down: string; up: string } = { down: '.lora_down.weight', up: '.lora_up.weight' }): Spec[] {
  const specs: Spec[] = []
  for (const l of layers) {
    const down = randMat(l.rank, l.inn * (l.conv ? l.conv[0] * l.conv[1] : 1), l.seed)
    const up = randMat(l.out, l.rank, l.seed + 1000)
    if (l.alpha !== null && l.alpha !== undefined) specs.push({ key: `${l.base}.alpha`, shape: [], values: Float64Array.of(l.alpha) })
    specs.push({ key: l.base + style.down, shape: l.conv ? [l.rank, l.inn, l.conv[0], l.conv[1]] : [l.rank, l.inn], values: down })
    specs.push({ key: l.base + style.up, shape: l.conv ? [l.out, l.rank, 1, 1] : [l.out, l.rank], values: up })
  }
  return specs
}

const recipe = (over: Partial<LoraMergeRecipe> = {}): LoraMergeRecipe => ({
  inputs: [],
  mode: 'concat',
  rank: 64,
  dtype: 'fp32',
  outputName: 'm',
  ...over
})

/** The dense delta of one layer of a file, times weight and alpha/rank. */
function delta(f: { header: SafetensorsHeader; data: Buffer }, base: string, weight: number): Float64Array {
  const p = parseLoraLayers(f.header)
  const l = [...p.layers.values()].find((x) => x.base === base)!
  const up = tensorValues(f.header, f.data, l.up)
  const down = tensorValues(f.header, f.data, l.down)
  const scale = l.alpha === null ? weight : (weight * tensorValues(f.header, f.data, l.alpha)[0]) / l.rank
  const d = matmul(up, down, l.out, l.rank, l.inFlat)
  return d.map((x) => x * scale)
}

const opts = (n: number) => ({ appVersion: '9.9.9', inputHeaderSha256: Array.from({ length: n }, () => 'sha'), now: () => new Date('2026-01-02T03:04:05Z') })

// ---------- linalg ----------

describe('symmetricEigen', () => {
  it.each([1, 2, 5, 40, 150])('reconstructs a random symmetric %i x %i matrix', (n) => {
    const b = randMat(n, n, n)
    const a = new Float64Array(n * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[i * n + j] = b[i * n + j] + b[j * n + i]
    const { values, vectors } = symmetricEigen(a, n)
    for (let i = 1; i < n; i++) expect(values[i - 1]).toBeGreaterThanOrEqual(values[i])
    // A = V diag V^t
    const rec = new Float64Array(n * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += vectors[i * n + k] * values[k] * vectors[j * n + k]
      rec[i * n + j] = s
    }
    expect(relErr(rec, a)).toBeLessThan(1e-11)
    // V^t V = I
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += vectors[k * n + i] * vectors[k * n + j]
      expect(Math.abs(s - (i === j ? 1 : 0))).toBeLessThan(1e-11)
    }
  })
  it('handles a diagonal and a zero matrix', () => {
    const d = Float64Array.from([2, 0, 0, 0, 5, 0, 0, 0, -1])
    expect([...symmetricEigen(d, 3).values]).toEqual([5, 2, -1])
    expect([...symmetricEigen(new Float64Array(9), 3).values]).toEqual([0, 0, 0])
  })
})

describe('svdSmall', () => {
  it.each([
    [6, 4],
    [4, 6],
    [30, 30],
    [90, 70],
    [70, 90],
    [100, 3]
  ])('reconstructs a random %i x %i matrix', (m, n) => {
    const a = randMat(m, n, m * 31 + n)
    const { u, s, vt } = svdSmall(a, m, n)
    const r = Math.min(m, n)
    expect(s.length).toBe(r)
    for (let i = 1; i < r; i++) expect(s[i - 1]).toBeGreaterThanOrEqual(s[i])
    const us = new Float64Array(m * r)
    for (let i = 0; i < m; i++) for (let j = 0; j < r; j++) us[i * r + j] = u[i * r + j] * s[j]
    expect(relErr(matmul(us, vt, m, r, n), a)).toBeLessThan(1e-10)
    // vt rows orthonormal
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) {
      let d = 0
      for (let k = 0; k < n; k++) d += vt[i * n + k] * vt[j * n + k]
      expect(Math.abs(d - (i === j ? 1 : 0))).toBeLessThan(1e-9)
    }
  })
  it('finds the singular values of a known diagonal', () => {
    const a = new Float64Array(9)
    a[0] = 3
    a[4] = -7
    a[8] = 1
    const { s } = svdSmall(a, 3, 3)
    expect([...s].map((x) => +x.toFixed(10))).toEqual([7, 3, 1])
  })
})

describe('shrinkLayer', () => {
  it('is exact when k is at least the true rank', () => {
    const out = 20
    const inn = 30
    const rank = 6
    const up = randMat(out, rank, 1)
    const down = randMat(rank, inn, 2)
    const r = shrinkLayer(up, down, out, inn, rank, 8)
    expect(r.k).toBe(6)
    expect(r.energyKept).toBe(1)
    expect(r.up.length).toBe(out * 6)
    expect(r.down.length).toBe(6 * inn)
    const want = matmul(up, down, out, rank, inn)
    expect(relErr(matmul(r.up, r.down, out, r.k, inn), want)).toBeLessThan(1e-9)
  })
  it('drops the dead rank of a rank-deficient layer', () => {
    // up has 6 columns but only 3 independent ones.
    const out = 16
    const inn = 24
    const base = randMat(out, 3, 3)
    const up = new Float64Array(out * 6)
    for (let o = 0; o < out; o++) for (let c = 0; c < 6; c++) up[o * 6 + c] = base[o * 3 + (c % 3)] * (c < 3 ? 1 : -0.5)
    const down = randMat(6, inn, 4)
    const r = shrinkLayer(up, down, out, inn, 6, 6)
    expect(r.k).toBe(3)
    expect(r.energyKept).toBe(1)
    expect(relErr(matmul(r.up, r.down, out, r.k, inn), matmul(up, down, out, 6, inn))).toBeLessThan(1e-9)
  })
  it('reports the kept energy for a known spectrum and splits the singular values evenly', () => {
    // up = diag-ish orthonormal columns scaled, down = orthonormal rows -> singular values 4, 3, 2, 1.
    const out = 8
    const inn = 8
    const sv = [4, 3, 2, 1]
    const up = new Float64Array(out * 4)
    const down = new Float64Array(4 * inn)
    for (let i = 0; i < 4; i++) {
      up[i * 4 + i] = sv[i]
      down[i * inn + i + 1] = 1
    }
    const r = shrinkLayer(up, down, out, inn, 4, 2)
    expect(r.k).toBe(2)
    expect(r.energyKept).toBeCloseTo((16 + 9) / 30, 12)
    // Balanced: column norms of up' and row norms of down' are sqrt(s).
    const col = (c: number): number => Math.sqrt(Array.from({ length: out }, (_, o) => r.up[o * 2 + c] ** 2).reduce((x, y) => x + y, 0))
    const row = (c: number): number => Math.sqrt(Array.from({ length: inn }, (_, i) => r.down[c * inn + i] ** 2).reduce((x, y) => x + y, 0))
    expect(col(0)).toBeCloseTo(2, 9)
    expect(col(1)).toBeCloseTo(Math.sqrt(3), 9)
    expect(row(0)).toBeCloseTo(2, 9)
    expect(row(1)).toBeCloseTo(Math.sqrt(3), 9)
    // Best rank-2 approximation: the error equals the dropped singular values.
    const full = matmul(up, down, out, 4, inn)
    const err = fro(matmul(r.up, r.down, out, 2, inn).map((x, i) => x - full[i]))
    expect(err).toBeCloseTo(Math.sqrt(4 + 1), 9)
  })
  it('returns one zero term for a zero layer', () => {
    const r = shrinkLayer(new Float64Array(5 * 3), randMat(3, 7, 5), 5, 7, 3, 2)
    expect(r.k).toBe(1)
    expect(r.energyKept).toBe(1)
    expect(fro(r.up) + fro(r.down)).toBe(0)
  })
  it('works on a larger layer (rank 40 -> 16)', () => {
    const out = 60
    const inn = 50
    const rank = 40
    const up = randMat(out, rank, 7)
    const down = randMat(rank, inn, 8)
    const full = matmul(up, down, out, rank, inn)
    const r = shrinkLayer(up, down, out, inn, rank, 16)
    expect(r.k).toBe(16)
    const approx = matmul(r.up, r.down, out, 16, inn)
    // Eckart-Young: its error is the energy of the dropped singular values.
    const { s } = svdSmall(full, out, inn)
    let drop = 0
    let tot = 0
    s.forEach((x, i) => {
      tot += x * x
      if (i >= 16) drop += x * x
    })
    expect(fro(approx.map((x, i) => x - full[i]))).toBeCloseTo(Math.sqrt(drop), 6)
    expect(r.energyKept).toBeCloseTo(1 - drop / tot, 8)
  })
})

// ---------- parseLoraLayers / describe ----------

describe('parseLoraLayers', () => {
  it('reads the lora_up / lora_down family with alpha and kohya names', () => {
    const f = makeFile(loraSpecs([{ base: 'lora_unet_a_b', out: 4, inn: 6, rank: 2, alpha: 1, seed: 1 }]))
    const p = parseLoraLayers(f.header)
    expect(p.family).toBe('updown')
    expect(p.keyStyle).toBe('kohya')
    expect(p.unsupported).toBeNull()
    const l = p.layers.get('lora_unet_a_b')!
    expect(l).toMatchObject({ base: 'lora_unet_a_b', down: 'lora_unet_a_b.lora_down.weight', up: 'lora_unet_a_b.lora_up.weight', alpha: 'lora_unet_a_b.alpha', rank: 2, out: 4, inFlat: 6, textEncoder: false })
  })
  it('reads lora_A / lora_B, PEFT .default and the dot family', () => {
    const ab = parseLoraLayers(makeFile(loraSpecs([{ base: 'transformer.x.to_q', out: 4, inn: 6, rank: 2, seed: 1 }], { down: '.lora_A.weight', up: '.lora_B.weight' })).header)
    expect(ab.family).toBe('AB')
    expect(ab.keyStyle).toBe('diffusers')
    expect(ab.layers.get('transformer.x.to_q')!.alpha).toBeNull()
    const peft = parseLoraLayers(
      makeFile(loraSpecs([{ base: 'base_model.model.transformer.x.to_q', out: 4, inn: 6, rank: 2, seed: 1 }], { down: '.lora_A.default.weight', up: '.lora_B.default.weight' })).header
    )
    expect(peft.family).toBe('AB')
    const l = peft.layers.get('transformer.x.to_q')!
    expect(l.base).toBe('base_model.model.transformer.x.to_q')
    expect(l.down).toBe('base_model.model.transformer.x.to_q.lora_A.default.weight')
    const dot = parseLoraLayers(makeFile(loraSpecs([{ base: 'diffusion_model.blocks.0.attn', out: 4, inn: 6, rank: 2, seed: 1 }], { down: '.lora.down.weight', up: '.lora.up.weight' })).header)
    expect(dot.family).toBe('dot')
    expect(dot.keyStyle).toBe('comfy')
  })
  it('normalizes model.diffusion_model. so comfy files match', () => {
    const p = parseLoraLayers(makeFile(loraSpecs([{ base: 'model.diffusion_model.blocks.0.attn', out: 4, inn: 6, rank: 2, seed: 1 }])).header)
    expect([...p.layers.keys()]).toEqual(['diffusion_model.blocks.0.attn'])
    expect(p.layers.get('diffusion_model.blocks.0.attn')!.base).toBe('model.diffusion_model.blocks.0.attn')
  })
  it('flags text-encoder layers and conv layers', () => {
    const p = parseLoraLayers(
      makeFile(
        loraSpecs([
          { base: 'lora_te_text_model_q', out: 4, inn: 6, rank: 2, seed: 1 },
          { base: 'lora_unet_conv', out: 4, inn: 3, rank: 2, conv: [3, 3], seed: 2 }
        ])
      ).header
    )
    expect(p.layers.get('lora_te_text_model_q')!.textEncoder).toBe(true)
    const c = p.layers.get('lora_unet_conv')!
    expect(c).toMatchObject({ rank: 2, out: 4, inFlat: 27, downShape: [2, 3, 3, 3], upShape: [4, 2, 1, 1] })
  })
  it.each([
    ['LoKr', [{ key: 'x.lokr_w1', shape: [2, 2] }], /LoKr/],
    ['DoRA', [{ key: 'x.dora_scale', shape: [4] }], /DoRA/],
    ['lora_mid', [{ key: 'x.lora_mid.weight', shape: [2, 2, 3, 3] }], /lora_mid/],
    ['diff', [{ key: 'x.diff', shape: [4, 4] }], /\.diff/]
  ])('refuses %s files', (_n, extra, re) => {
    const specs = [...loraSpecs([{ base: 'lora_unet_x', out: 4, inn: 6, rank: 2, seed: 1 }]), ...extra.map((e) => ({ ...e, values: new Float64Array(e.shape.reduce((a, b) => a * b, 1)) }))]
    const p = parseLoraLayers(makeFile(specs).header)
    expect(p.unsupported).toMatch(re)
  })
  it('refuses a big up kernel, a rank mismatch, no layers and other dtypes', () => {
    const k = makeFile([
      { key: 'a.lora_down.weight', shape: [2, 3, 3, 3], values: new Float64Array(54) },
      { key: 'a.lora_up.weight', shape: [4, 2, 3, 3], values: new Float64Array(72) }
    ])
    expect(parseLoraLayers(k.header).unsupported).toMatch(/kernel/)
    const r = makeFile([
      { key: 'a.lora_down.weight', shape: [2, 3], values: new Float64Array(6) },
      { key: 'a.lora_up.weight', shape: [4, 3], values: new Float64Array(12) }
    ])
    expect(parseLoraLayers(r.header).unsupported).toMatch(/different ranks/)
    expect(parseLoraLayers(makeFile([{ key: 'w', shape: [2], values: new Float64Array(2) }]).header).unsupported).toMatch(/no LoRA layers/)
    const q = makeFile(loraSpecs([{ base: 'a', out: 4, inn: 6, rank: 2, seed: 1 }]))
    q.header.tensors['a.lora_up.weight'].dtype = 'I8'
    expect(parseLoraLayers(q.header).unsupported).toMatch(/I8/)
  })
})

describe('describeMergeInput', () => {
  it('summarizes a usable file', () => {
    const f = makeFile(
      loraSpecs([
        { base: 'lora_unet_a', out: 4, inn: 6, rank: 2, seed: 1 },
        { base: 'lora_unet_b', out: 4, inn: 6, rank: 8, seed: 2 },
        { base: 'lora_te_c', out: 4, inn: 6, rank: 4, seed: 3 }
      ]).map((s) => (s.key.endsWith('lora_up.weight') ? { ...s, dtype: 'BF16' as const } : s)),
      { ss_base_model: 'flux-dev' }
    )
    const i = describeMergeInput('x.safetensors', f.header, null)
    expect(i).toMatchObject({ name: 'x.safetensors', readable: true, supported: true, modules: 3, textEncoderModules: 1, rank: { min: 2, max: 8 }, dtype: 'BF16', keyStyle: 'kohya', format: 'LoRA', baseModelHint: 'flux-dev', unsupportedReason: null })
  })
  it('explains unreadable and unsupported files', () => {
    expect(describeMergeInput('n', null, 'boom')).toMatchObject({ readable: false, supported: false, error: 'boom', unsupportedReason: 'boom' })
    const lokr = makeFile([{ key: 'x.lokr_w1', shape: [2, 2], values: new Float64Array(4) }])
    const i = describeMergeInput('n', lokr.header, null)
    expect(i.supported).toBe(false)
    expect(i.unsupportedReason).toMatch(/^It is a LoKr/)
  })
})

// ---------- checkLoraMerge ----------

function infoOf(name: string, specs: Spec[], meta: Record<string, string> = {}): { info: ReturnType<typeof describeMergeInput>; header: SafetensorsHeader } {
  const f = makeFile(specs, meta)
  return { info: describeMergeInput(name, f.header, null), header: f.header }
}

describe('checkLoraMerge', () => {
  const L = (base: string, rank: number, seed: number, out = 4, inn = 6): LayerSpec => ({ base, out, inn, rank, seed })
  const run = (items: Array<ReturnType<typeof infoOf>>, req: Partial<Parameters<typeof checkLoraMerge>[2]> = {}) =>
    checkLoraMerge(items.map((i) => i.info), items.map((i) => i.header), { mode: 'concat', rank: 8, dtype: 'source', ...req })

  it('accepts two compatible LoRAs and reports modules, rank and size', () => {
    const a = infoOf('a', loraSpecs([L('lora_unet_x', 2, 1), L('lora_unet_y', 4, 2)]))
    const b = infoOf('b', loraSpecs([L('lora_unet_x', 3, 3), L('lora_unet_z', 4, 4)]))
    const c = run([a, b])
    expect(c.problems).toEqual([])
    expect(c.totalModules).toBe(3)
    expect(c.sharedModules).toBe(1)
    expect(c.outputRank).toEqual({ min: 4, max: 5 })
    // (4+6)*5*4 + (4+6)*4*4 + (4+6)*4*4 plus alphas and a header estimate
    expect(c.estimatedBytes).toBeGreaterThan(200 + 160 + 160)
    expect(c.estimatedBytes).toBeLessThan(5000)
    expect(run([a, b], { mode: 'shrink', rank: 4 }).outputRank).toEqual({ min: 4, max: 4 })
    expect(run([a, b], { mode: 'shrink', rank: 3 }).outputRank).toEqual({ min: 3, max: 3 })
  })
  it('blocks too few or too many inputs and unreadable or unsupported ones', () => {
    const a = infoOf('a', loraSpecs([L('lora_unet_x', 2, 1)]))
    expect(run([a]).problems).toContain('Choose at least two LoRAs to merge.')
    const many = Array.from({ length: 11 }, (_, i) => infoOf(`n${i}`, loraSpecs([L('lora_unet_x', 2, i)])))
    expect(run(many).problems.join(' ')).toMatch(/at most 10/)
    const bad = describeMergeInput('bad', null, 'could not open it')
    const lokr = infoOf('lk', [{ key: 'x.lokr_w1', shape: [2, 2], values: new Float64Array(4) }])
    const p = checkLoraMerge([a.info, bad, lokr.info], [a.header, null, lokr.header], { mode: 'concat', rank: 8, dtype: 'source' }).problems
    expect(p).toContain('bad: could not open it')
    expect(p.some((x) => x.startsWith('lk: It is a LoKr'))).toBe(true)
  })
  it('blocks different key styles, no shared layers and shape mismatches', () => {
    const a = infoOf('a', loraSpecs([L('lora_unet_x', 2, 1)]))
    const d = infoOf('d', loraSpecs([L('transformer.x', 2, 2)], { down: '.lora_A.weight', up: '.lora_B.weight' }))
    const styles = run([a, d]).problems
    expect(styles.some((x) => /use different layer names \(kohya \/ diffusers\); they were probably made by different trainers/.test(x))).toBe(true)
    const b = infoOf('b', loraSpecs([L('lora_unet_other', 2, 3)]))
    expect(run([a, b]).problems.join(' ')).toMatch(/No layer is shared/)
    const c = infoOf('c', loraSpecs([L('lora_unet_x', 2, 4, 5, 6)]))
    expect(run([a, c]).problems.join(' ')).toMatch(/different shape/)
    const k = infoOf('k', loraSpecs([{ base: 'lora_unet_x', out: 4, inn: 2, rank: 2, conv: [3, 3], seed: 9 }]))
    const k2 = infoOf('k2', loraSpecs([{ base: 'lora_unet_x', out: 4, inn: 3, rank: 2, conv: [1, 1], seed: 9 }]))
    expect(run([k, k2]).problems.join(' ')).toMatch(/different shape/)
  })
  it('blocks a shrink rank outside 1..1024', () => {
    const a = infoOf('a', loraSpecs([L('lora_unet_x', 2, 1)]))
    const b = infoOf('b', loraSpecs([L('lora_unet_x', 2, 2)]))
    expect(run([a, b], { mode: 'shrink', rank: 0 }).problems.join(' ')).toMatch(/1 to 1024/)
    expect(run([a, b], { mode: 'shrink', rank: 2000 }).problems.join(' ')).toMatch(/1 to 1024/)
    expect(run([a, b], { mode: 'concat', rank: 0 }).problems).toEqual([])
  })
  it('warns about few shared layers, different base models and weight 0', () => {
    const many = Array.from({ length: 10 }, (_, i) => L(`lora_unet_l${i}`, 2, i))
    const a = infoOf('a', loraSpecs(many), { ss_base_model: 'flux' })
    const b = infoOf('b', loraSpecs([many[0], L('lora_unet_only', 2, 99), L('lora_unet_only2', 2, 98)]), { ss_base_model: 'sdxl' })
    const c = run([a, b], { weights: [1, 0] })
    expect(c.problems).toEqual([])
    expect(c.warnings.join(' ')).toMatch(/33%/)
    expect(c.warnings.join(' ')).toMatch(/different base models/)
    expect(c.warnings.join(' ')).toMatch(/b has weight 0/)
  })
})

// ---------- mergeLoras ----------

describe('mergeLoras', () => {
  const L = (base: string, rank: number, seed: number, extra: Partial<LayerSpec> = {}): LayerSpec => ({ base, out: 5, inn: 7, rank, seed, ...extra })
  const input = (name: string, layers: LayerSpec[], weight: number, style?: { down: string; up: string }, meta?: Record<string, string>) => {
    const f = makeFile(loraSpecs(layers, style), meta)
    return { name, ...f, weight }
  }
  const asFile = (r: Awaited<ReturnType<typeof mergeLoras>>) => {
    const tensors: SafetensorsHeader['tensors'] = {}
    const parts: Buffer[] = []
    let off = 0
    for (const t of r.tensors) {
      tensors[t.key] = { dtype: t.dtype, shape: t.shape, offsets: [off, off + t.data.length] }
      parts.push(t.data)
      off += t.data.length
    }
    return { header: { tensors, metadata: r.metadata, headerBytes: 1, fileBytes: off, headerSha256: 'm' } as SafetensorsHeader, data: Buffer.concat(parts) }
  }

  it('concat equals the weighted sum of the deltas (alpha scaling, negative weights, layers in one input)', async () => {
    const a = input('a', [L('lora_unet_x', 2, 1, { alpha: 1 }), L('lora_unet_only_a', 3, 2)], 0.7)
    const b = input('b', [L('lora_unet_x', 4, 3, { alpha: 4 }), L('lora_unet_only_b', 2, 4, { alpha: 2 })], -1.5)
    const c = input('c', [L('lora_unet_x', 1, 5)], 2)
    const r = await mergeLoras([a, b, c], recipe(), opts(3))
    expect(r.modules).toBe(3)
    expect(r.rank).toEqual({ min: 2, max: 7 })
    expect(r.energyKept).toBeNull()
    const m = asFile(r)
    expect(r.tensors.map((t) => t.key)).toContain('lora_unet_x.alpha')
    const want = new Float64Array(5 * 7)
    for (const [f, w] of [[a, 0.7], [b, -1.5], [c, 2]] as const) {
      const d = delta(f, 'lora_unet_x', w)
      for (let i = 0; i < want.length; i++) want[i] += d[i]
    }
    expect(relErr(delta(m, 'lora_unet_x', 1), want)).toBeLessThan(1e-6)
    expect(relErr(delta(m, 'lora_unet_only_a', 1), delta(a, 'lora_unet_only_a', 0.7))).toBeLessThan(1e-6)
    expect(relErr(delta(m, 'lora_unet_only_b', 1), delta(b, 'lora_unet_only_b', -1.5))).toBeLessThan(1e-6)
    // alpha is the rank
    expect(tensorValues(m.header, m.data, 'lora_unet_x.alpha')[0]).toBe(7)
    expect(m.header.tensors['lora_unet_x.alpha'].shape).toEqual([])
    expect(m.header.tensors['lora_unet_x.lora_up.weight'].shape).toEqual([5, 7])
    expect(m.header.tensors['lora_unet_x.lora_down.weight'].shape).toEqual([7, 7])
    expect(r.metadata).toMatchObject({ format: 'pt', ss_network_dim: '7', ss_network_alpha: '7' })
    const rec = JSON.parse(r.metadata.rulz_model_merger)
    expect(rec).toMatchObject({ app: 'RulZ Model Merger', appVersion: '9.9.9', kind: 'lora-merge', createdAt: '2026-01-02T03:04:05.000Z' })
    expect(rec.inputHeaderSha256).toEqual(['sha', 'sha', 'sha'])
  })
  it('output order: first input first, then new layers; keys use the first input family', async () => {
    const a = input('a', [L('transformer.p', 2, 1), L('transformer.q', 2, 2)], 1, { down: '.lora_A.weight', up: '.lora_B.weight' })
    const b = input('b', [L('transformer.r', 2, 3), L('transformer.p', 2, 4)], 1, { down: '.lora_down.weight', up: '.lora_up.weight' })
    const r = await mergeLoras([a, b], recipe(), opts(2))
    expect(r.tensors.filter((t) => t.key.endsWith('.alpha')).map((t) => t.key)).toEqual(['transformer.p.alpha', 'transformer.q.alpha', 'transformer.r.alpha'])
    expect(r.tensors.some((t) => t.key === 'transformer.r.lora_A.weight')).toBe(true)
    expect(r.tensors.some((t) => t.key.endsWith('lora_up.weight'))).toBe(false)
  })
  it('restores conv shapes', async () => {
    const a = input('a', [L('lora_unet_c', 2, 1, { inn: 3, conv: [3, 3] })], 1)
    const b = input('b', [L('lora_unet_c', 3, 2, { inn: 3, conv: [3, 3] })], 0.5)
    const r = await mergeLoras([a, b], recipe(), opts(2))
    const up = r.tensors.find((t) => t.key.endsWith('lora_up.weight'))!
    const down = r.tensors.find((t) => t.key.endsWith('lora_down.weight'))!
    expect(up.shape).toEqual([5, 5, 1, 1])
    expect(down.shape).toEqual([5, 3, 3, 3])
    const m = asFile(r)
    const want = new Float64Array(5 * 27)
    for (const [f, w] of [[a, 1], [b, 0.5]] as const) delta(f, 'lora_unet_c', w).forEach((v, i) => (want[i] += v))
    expect(relErr(delta(m, 'lora_unet_c', 1), want)).toBeLessThan(1e-6)
  })
  it('shrink reaches the target rank, is exact when it is not needed, and reports energy', async () => {
    const a = input('a', [L('lora_unet_x', 4, 1)], 1)
    const b = input('b', [L('lora_unet_x', 4, 2)], 1)
    const big = await mergeLoras([a, b], recipe({ mode: 'shrink', rank: 16 }), opts(2))
    expect(big.rank).toEqual({ min: 8, max: 8 })
    expect(big.energyKept).toEqual({ mean: 1, min: 1 })
    const exact = asFile(big)
    const want = new Float64Array(35)
    for (const f of [a, b]) delta(f, 'lora_unet_x', 1).forEach((v, i) => (want[i] += v))
    expect(relErr(delta(exact, 'lora_unet_x', 1), want)).toBeLessThan(1e-6)

    const small = await mergeLoras([a, b], recipe({ mode: 'shrink', rank: 3 }), opts(2))
    expect(small.rank).toEqual({ min: 3, max: 3 })
    expect(small.energyKept!.min).toBeGreaterThan(0.3)
    expect(small.energyKept!.min).toBeLessThan(1)
    expect(small.metadata.ss_network_dim).toBe('3')
    const s = asFile(small)
    expect(s.header.tensors['lora_unet_x.lora_down.weight'].shape).toEqual([3, 7])
    expect(tensorValues(s.header, s.data, 'lora_unet_x.alpha')[0]).toBe(3)
    const err = relErr(delta(s, 'lora_unet_x', 1), want)
    expect(err).toBeLessThan(Math.sqrt(1 - small.energyKept!.min) + 1e-5)
    expect(err).toBeGreaterThan(0)
  })
  it('picks the output dtype and copies agreeing base-model metadata', async () => {
    const a = input('a', [L('lora_unet_x', 2, 1)], 1, undefined, { ss_base_model: 'flux', 'modelspec.architecture': 'x/y' })
    const b = input('b', [L('lora_unet_x', 2, 2)], 1, undefined, { ss_base_model: 'flux' })
    const src = await mergeLoras([a, b], recipe({ dtype: 'source' }), opts(2))
    expect(src.tensors.find((t) => t.key.endsWith('lora_up.weight'))!.dtype).toBe('F32')
    expect(src.metadata.ss_base_model).toBe('flux')
    expect(src.metadata['modelspec.architecture']).toBe('x/y')
    for (const [d, want] of [['bf16', 'BF16'], ['fp16', 'F16'], ['fp32', 'F32']] as const) {
      const r = await mergeLoras([a, b], recipe({ dtype: d }), opts(2))
      expect(r.tensors.find((t) => t.key.endsWith('lora_down.weight'))!.dtype).toBe(want)
    }
    const c = input('c', [L('lora_unet_x', 2, 3)], 1, undefined, { ss_base_model: 'sdxl' })
    expect((await mergeLoras([a, c], recipe(), opts(2))).metadata.ss_base_model).toBeUndefined()
    // 'source' follows the first input's up dtype
    const bf = makeFile(loraSpecs([L('lora_unet_x', 2, 5)]).map((s) => (s.key.endsWith('lora_up.weight') ? { ...s, dtype: 'BF16' as const } : s)))
    const r = await mergeLoras([{ name: 'bf', ...bf, weight: 1 }, b], recipe({ dtype: 'source' }), opts(2))
    expect(r.tensors.find((t) => t.key.endsWith('lora_up.weight'))!.dtype).toBe('BF16')
  })
  it('refuses a merge the check blocks, and reports progress and abort', async () => {
    const a = input('a', [L('lora_unet_x', 2, 1)], 1)
    const b = input('b', [L('lora_unet_zzz', 2, 2)], 1)
    await expect(mergeLoras([a, b], recipe(), opts(2))).rejects.toThrow(/No layer is shared/)
    await expect(mergeLoras([a], recipe(), opts(1))).rejects.toThrow(/at least two/)
    const c = input('c', [L('lora_unet_x', 2, 3), L('lora_unet_y', 2, 4), L('lora_unet_z', 2, 5)], 1)
    const d = input('d', [L('lora_unet_x', 2, 6), L('lora_unet_y', 2, 7), L('lora_unet_z', 2, 8)], 1)
    const seen: Array<[number, number]> = []
    await mergeLoras([c, d], recipe(), { ...opts(2), onProgress: (x, y) => seen.push([x, y]) })
    expect(seen).toEqual([[1, 3], [2, 3], [3, 3]])
    const ac = new AbortController()
    let err: Error | null = null
    try {
      await mergeLoras([c, d], recipe(), { ...opts(2), signal: ac.signal, onProgress: (x) => x === 1 && ac.abort() })
    } catch (e) {
      err = e as Error
    }
    expect(err?.name).toBe('AbortError')
    const pre = new AbortController()
    pre.abort()
    await expect(mergeLoras([c, d], recipe(), { ...opts(2), signal: pre.signal })).rejects.toMatchObject({ name: 'AbortError' })
  })
})

// ---------- verifyMergedLora ----------

describe('verifyMergedLora', () => {
  const L = (base: string, rank: number, seed: number, extra: Partial<LayerSpec> = {}): LayerSpec => ({ base, out: 6, inn: 9, rank, seed, ...extra })
  const mk = (layers: LayerSpec[], weight: number) => ({ ...makeFile(loraSpecs(layers)), weight })
  const asFile = (r: Awaited<ReturnType<typeof mergeLoras>>) => {
    const tensors: SafetensorsHeader['tensors'] = {}
    const parts: Buffer[] = []
    let off = 0
    for (const t of r.tensors) {
      tensors[t.key] = { dtype: t.dtype, shape: t.shape, offsets: [off, off + t.data.length] }
      parts.push(t.data)
      off += t.data.length
    }
    return { header: { tensors, metadata: r.metadata, headerBytes: 1, fileBytes: off, headerSha256: 'm' } as SafetensorsHeader, data: Buffer.concat(parts) }
  }
  const layers = (n: number, seed: number): LayerSpec[] => Array.from({ length: n }, (_, i) => L(`lora_unet_l${i}`, 3, seed + i * 7, { alpha: 2 }))

  it('accepts a correct concat merge (bf16) and checks at most 16 layers', async () => {
    const a = mk(layers(20, 1), 0.8)
    const b = mk(layers(20, 500), 1.2)
    const rcp = recipe({ dtype: 'bf16' })
    const m = asFile(await mergeLoras([{ name: 'a', ...a }, { name: 'b', ...b }], rcp, opts(2)))
    const v = verifyMergedLora(m, [a, b], rcp)
    expect(v.ok).toBe(true)
    expect(v.checkedModules).toBe(16)
    expect(v.maxRelError).toBeLessThan(0.02)
    expect(v.message).toMatch(/^Checked 16 layers: the merged LoRA equals the weighted sum \(largest difference \d+\.\d%\)\.$/)
  })
  it('fails a tampered file', async () => {
    const a = mk(layers(3, 1), 1)
    const b = mk(layers(3, 50), 1)
    const rcp = recipe()
    const merged = await mergeLoras([{ name: 'a', ...a }, { name: 'b', ...b }], rcp, opts(2))
    const good = asFile(merged)
    expect(verifyMergedLora(good, [a, b], rcp).ok).toBe(true)
    // Flip the sign of a block of values in an up tensor.
    const t = good.header.tensors['lora_unet_l1.lora_up.weight']
    const bad = { header: good.header, data: Buffer.from(good.data) }
    for (let o = t.offsets[0]; o < t.offsets[1]; o += 4) bad.data.writeFloatLE(-bad.data.readFloatLE(o), o)
    const vb = verifyMergedLora(bad, [a, b], rcp)
    expect(vb.ok).toBe(false)
    expect(vb.message).toMatch(/differs from the weighted sum/)
    // Wrong alpha
    const wrongAlpha = { header: good.header, data: Buffer.from(good.data) }
    wrongAlpha.data.writeFloatLE(99, good.header.tensors['lora_unet_l0.alpha'].offsets[0])
    expect(verifyMergedLora(wrongAlpha, [a, b], rcp).message).toMatch(/alpha 99 but rank 6/)
    // A non-finite value
    const nan = { header: good.header, data: Buffer.from(good.data) }
    nan.data.writeFloatLE(NaN, good.header.tensors['lora_unet_l2.lora_down.weight'].offsets[0])
    expect(verifyMergedLora(nan, [a, b], rcp).message).toMatch(/not a finite number/)
    // A missing layer
    const fewer = asFile({ ...merged, tensors: merged.tensors.filter((x) => !x.key.startsWith('lora_unet_l2')) })
    expect(verifyMergedLora(fewer, [a, b], rcp).message).toMatch(/layers but the inputs have/)
  })
  it('does not fail a shrink merge on its approximation error', async () => {
    const a = mk(layers(4, 1), 1)
    const b = mk(layers(4, 50), 1)
    const rcp = recipe({ mode: 'shrink', rank: 2 })
    const m = asFile(await mergeLoras([{ name: 'a', ...a }, { name: 'b', ...b }], rcp, opts(2)))
    const v = verifyMergedLora(m, [a, b], rcp)
    expect(v.ok).toBe(true)
    expect(v.maxRelError).toBeGreaterThan(0.001)
    expect(v.message).toMatch(/shrunk LoRA differs/)
  })
})
