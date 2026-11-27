// Fixes from the M5 review: lopsided factors in shrink, ambiguous layer names, memory warning.
import { describe, expect, it } from 'vitest'
import { checkLoraMerge, describeMergeInput, mergeLoras, parseLoraLayers } from '../src/main/loras/merge'
import { encodeValues, tensorValues } from '../src/main/models/tensors'
import type { SafetensorsHeader } from '../src/main/models/safetensors'
import type { LoraMergeRecipe } from '../src/shared/loraMerge'

function rng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5
  }
}
const mat = (n: number, seed: number, scale = 1): Float64Array => {
  const r = rng(seed)
  return Float64Array.from({ length: n }, () => r() * scale)
}

function file(specs: Array<{ key: string; shape: number[]; values: Float64Array }>): { header: SafetensorsHeader; data: Buffer } {
  const tensors: SafetensorsHeader['tensors'] = {}
  const parts: Buffer[] = []
  let off = 0
  for (const s of specs) {
    const buf = encodeValues(s.values, 'F32')
    tensors[s.key] = { dtype: 'F32', shape: s.shape, offsets: [off, off + buf.length] }
    parts.push(buf)
    off += buf.length
  }
  return { header: { tensors, metadata: {}, headerBytes: 1, fileBytes: off + 9, headerSha256: 'h' }, data: Buffer.concat(parts) }
}

function dense(up: Float64Array, down: Float64Array, out: number, rank: number, inn: number, scale = 1): Float64Array {
  const d = new Float64Array(out * inn)
  for (let o = 0; o < out; o++) for (let r = 0; r < rank; r++) for (let c = 0; c < inn; c++) d[o * inn + c] += scale * up[o * rank + r] * down[r * inn + c]
  return d
}
const relErr = (a: Float64Array, b: Float64Array): number => {
  let n = 0
  let m = 0
  for (let i = 0; i < a.length; i++) {
    n += (a[i] - b[i]) ** 2
    m += b[i] ** 2
  }
  return Math.sqrt(n / m)
}

const recipe = (over: Partial<LoraMergeRecipe>): LoraMergeRecipe => ({ inputs: [], mode: 'shrink', rank: 4, dtype: 'fp32', outputName: 'm', ...over })
const opts = { appVersion: 't', inputHeaderSha256: ['a', 'b'] }

describe('shrink keeps inputs whose up and down weights differ hugely in size', () => {
  it('reproduces the weighted sum exactly when the rank is not reduced below the true rank', async () => {
    const out = 6
    const inn = 8
    // Input a: up ~1e-7, down ~1e7, so its delta is as large as b's but its up Gram eigenvalues are ~1e-14 of b's.
    const ua = mat(out * 2, 1, 1e-7)
    const da = mat(2 * inn, 2, 1e7)
    const ub = mat(out * 2, 3)
    const db = mat(2 * inn, 4)
    const a = file([
      { key: 'diffusion_model.x.lora_down.weight', shape: [2, inn], values: da },
      { key: 'diffusion_model.x.lora_up.weight', shape: [out, 2], values: ua }
    ])
    const b = file([
      { key: 'diffusion_model.x.lora_down.weight', shape: [2, inn], values: db },
      { key: 'diffusion_model.x.lora_up.weight', shape: [out, 2], values: ub }
    ])
    const r = await mergeLoras(
      [
        { name: 'a', ...a, weight: 1 },
        { name: 'b', ...b, weight: 1 }
      ],
      recipe({ rank: 4 }),
      opts
    )
    // R = 4 = k, so shrink is skipped; force a real shrink path by asking for rank 3 of a rank-4 sum and compare
    // against the rank-4 sum's best rank-3 approximation energy instead: here just check rank 4 is exact.
    const t = (k: string) => r.tensors.find((x) => x.key === k)!
    const up = tensorValues({ tensors: { u: { dtype: 'F32', shape: t('diffusion_model.x.lora_up.weight').shape, offsets: [0, t('diffusion_model.x.lora_up.weight').data.length] } }, metadata: {}, headerBytes: 0, fileBytes: 0, headerSha256: '' }, t('diffusion_model.x.lora_up.weight').data, 'u')
    const down = tensorValues({ tensors: { d: { dtype: 'F32', shape: t('diffusion_model.x.lora_down.weight').shape, offsets: [0, t('diffusion_model.x.lora_down.weight').data.length] } }, metadata: {}, headerBytes: 0, fileBytes: 0, headerSha256: '' }, t('diffusion_model.x.lora_down.weight').data, 'd')
    const want = dense(ua, da, out, 2, inn)
    const wb = dense(ub, db, out, 2, inn)
    for (let i = 0; i < want.length; i++) want[i] += wb[i]
    const k = r.rank.max
    expect(relErr(dense(up, down, out, k, inn), want)).toBeLessThan(1e-5)
  })

  it('does not drop the lopsided input when the shrink path runs', async () => {
    const out = 6
    const inn = 8
    const spec = (u: Float64Array, d: Float64Array, r: number) =>
      file([
        { key: 'diffusion_model.x.lora_down.weight', shape: [r, inn], values: d },
        { key: 'diffusion_model.x.lora_up.weight', shape: [out, r], values: u }
      ])
    const ua = mat(out * 2, 11, 1e-7)
    const da = mat(2 * inn, 12, 1e7)
    const ub = mat(out * 2, 13)
    const db = mat(2 * inn, 14)
    // c has weight 0: R = 6 > k = 4 so the shrink runs, while the true rank of the sum is 4 and must come back exactly.
    // Without balancing, a's up directions fell under the eigenvalue threshold and its whole change was dropped.
    const r = await mergeLoras(
      [
        { name: 'a', ...spec(ua, da, 2), weight: 1 },
        { name: 'b', ...spec(ub, db, 2), weight: 1 },
        { name: 'c', ...spec(mat(out * 2, 15), mat(2 * inn, 16), 2), weight: 0 }
      ],
      recipe({ rank: 4 }),
      { appVersion: 't', inputHeaderSha256: ['a', 'b', 'c'] }
    )
    expect(r.energyKept).not.toBeNull()
    expect(r.energyKept!.min).toBeGreaterThan(0.999)
    const t = (k: string) => r.tensors.find((x) => x.key === k)!
    const hdr = (k: string) => ({ tensors: { v: { dtype: 'F32', shape: t(k).shape, offsets: [0, t(k).data.length] as [number, number] } }, metadata: {}, headerBytes: 0, fileBytes: 0, headerSha256: '' })
    const up = tensorValues(hdr('diffusion_model.x.lora_up.weight'), t('diffusion_model.x.lora_up.weight').data, 'v')
    const down = tensorValues(hdr('diffusion_model.x.lora_down.weight'), t('diffusion_model.x.lora_down.weight').data, 'v')
    const want = dense(ua, da, out, 2, inn)
    const wb = dense(ub, db, out, 2, inn)
    for (let i = 0; i < want.length; i++) want[i] += wb[i]
    expect(relErr(dense(up, down, out, r.rank.max, inn), want)).toBeLessThan(1e-4)
  })
})

describe('layer names that collide after normalizing', () => {
  it('refuses the file instead of dropping one pair', () => {
    const f = file([
      { key: 'diffusion_model.x.lora_down.weight', shape: [2, 4], values: mat(8, 1) },
      { key: 'diffusion_model.x.lora_up.weight', shape: [3, 2], values: mat(6, 2) },
      { key: 'model.diffusion_model.x.lora_down.weight', shape: [2, 4], values: mat(8, 3) },
      { key: 'model.diffusion_model.x.lora_up.weight', shape: [3, 2], values: mat(6, 4) }
    ])
    const p = parseLoraLayers(f.header)
    expect(p.unsupported).toMatch(/same name/)
    expect(describeMergeInput('dup.safetensors', f.header, null).supported).toBe(false)
  })
})

describe('memory warning', () => {
  it('warns when the inputs add up to more than 8 GiB', () => {
    const f = file([
      { key: 'diffusion_model.x.lora_down.weight', shape: [2, 4], values: mat(8, 1) },
      { key: 'diffusion_model.x.lora_up.weight', shape: [3, 2], values: mat(6, 2) }
    ])
    const big = { ...describeMergeInput('a', f.header, null), sizeBytes: 5 * 1024 ** 3 }
    const big2 = { ...describeMergeInput('b', f.header, null), sizeBytes: 5 * 1024 ** 3 }
    const c = checkLoraMerge([big, big2], [f.header, f.header], { mode: 'concat', rank: 4, dtype: 'source' })
    expect(c.problems).toEqual([])
    expect(c.warnings.some((w) => /10\.0 GiB/.test(w))).toBe(true)
    const small = checkLoraMerge([describeMergeInput('a', f.header, null), describeMergeInput('b', f.header, null)], [f.header, f.header], { mode: 'concat', rank: 4, dtype: 'source' })
    expect(small.warnings.some((w) => /GiB/.test(w))).toBe(false)
  })
})
