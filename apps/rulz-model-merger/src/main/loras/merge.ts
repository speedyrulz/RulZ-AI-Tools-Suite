// LoRA + LoRA merge math (milestone 5). Pure: parsing LoRA layers from a header, checking a merge, computing the merged
// tensors and verifying a saved result. File I/O lives in models/tensors.ts; the worker and service call this.
import type {
  LoraMergeCheck,
  LoraMergeDtype,
  LoraMergeInputInfo,
  LoraMergeMode,
  LoraMergeRecipe,
  LoraMergeRecord
} from '@shared/loraMerge'
import { MAX_LORA_MERGE_INPUTS } from '@shared/loraMerge'
import { describeLoraFile, isTextEncoderLoraKey } from '../merge/inspect'
import { encodeValues, tensorValues } from '../models/tensors'
import type { SafetensorsHeader } from '../models/safetensors'
import { shrinkLayer } from './linalg'

/** Above this total input size the check warns about memory use. */
const LARGE_INPUT_BYTES = 8 * 1024 ** 3

export interface LoraLayer {
  /** Key of the module as normalized for matching between files. */
  match: string
  /** Module name as in the file (without the .lora_up.weight etc. suffix). */
  base: string
  down: string
  up: string
  alpha: string | null
  rank: number
  out: number
  inFlat: number
  downShape: number[]
  upShape: number[]
  textEncoder: boolean
}

type Family = 'updown' | 'AB' | 'dot'
type KeyStyle = 'comfy' | 'kohya' | 'diffusers' | 'unknown'

const FAMILY_SUFFIXES: Array<{ family: Family; down: string; up: string }> = [
  { family: 'updown', down: '.lora_down.weight', up: '.lora_up.weight' },
  { family: 'AB', down: '.lora_A.default.weight', up: '.lora_B.default.weight' },
  { family: 'AB', down: '.lora_A.weight', up: '.lora_B.weight' },
  { family: 'dot', down: '.lora.down.weight', up: '.lora.up.weight' }
]
const FLOAT_DTYPES = new Set(['BF16', 'F16', 'F32', 'F64'])
const BASE_MODEL_PREFIX = 'base_model.model.'

export function normalizeLoraMatch(base: string): string {
  let m = base.startsWith(BASE_MODEL_PREFIX) ? base.slice(BASE_MODEL_PREFIX.length) : base
  if (m.startsWith('model.diffusion_model.')) m = m.slice('model.'.length)
  return m
}

function numel(shape: number[]): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

/** Why a file cannot be merged, from its key names alone; null when the keys look like a plain LoRA. */
function unsupportedFromKeys(keys: string[]): string | null {
  if (keys.some((k) => /\.(lokr_|hada_)/.test(k))) {
    return 'it is a LoKr or LoHa file; only plain LoRA layers (up and down weights) can be merged'
  }
  if (keys.some((k) => k.endsWith('.dora_scale'))) return 'it is a DoRA file (it has dora_scale weights), which cannot be merged'
  if (keys.some((k) => k.endsWith('.lora_mid.weight'))) return 'it has lora_mid weights (a LoCon tucker layout), which cannot be merged'
  if (keys.some((k) => k.endsWith('.diff') || k.endsWith('.diff_b'))) {
    return 'it holds full weight differences (.diff), which cannot be merged as a LoRA'
  }
  return null
}

export function parseLoraLayers(header: SafetensorsHeader): {
  layers: Map<string, LoraLayer>
  family: Family
  keyStyle: KeyStyle
  unsupported: string | null
} {
  const keys = Object.keys(header.tensors)
  const keySet = new Set(keys)
  const layers = new Map<string, LoraLayer>()
  const familyCount = new Map<Family, number>()
  let firstFamily: Family | null = null
  let unsupported = unsupportedFromKeys(keys)
  const note = (msg: string): void => {
    if (unsupported === null) unsupported = msg
  }

  for (const entry of FAMILY_SUFFIXES) {
    for (const key of keys) {
      if (!key.endsWith(entry.down)) continue
      // '.lora_A.weight' must not swallow keys that belong to '.lora_A.default.weight' (different suffix, no overlap),
      // so a plain endsWith is enough; but skip a module already taken by an earlier family.
      const base = key.slice(0, key.length - entry.down.length)
      if (base === '') continue
      const upKey = base + entry.up
      if (!keySet.has(upKey)) continue
      const match = normalizeLoraMatch(base)
      const existing = layers.get(match)
      if (existing) {
        // Two pairs that mean the same layer (e.g. 'diffusion_model.x' and 'model.diffusion_model.x'): merging only
        // one would silently drop the other, so the file is refused.
        if (existing.base !== base) note(`two of its layers have the same name after normalizing (${existing.base}, ${base})`)
        continue
      }
      const dt = header.tensors[key]
      const ut = header.tensors[upKey]
      const downOk = dt.shape.length === 2 || dt.shape.length === 4
      const upOk = ut.shape.length === 2 || (ut.shape.length === 4 && ut.shape[2] === 1 && ut.shape[3] === 1)
      if (ut.shape.length === 4 && (ut.shape[2] !== 1 || ut.shape[3] !== 1)) {
        note(`its up weights use a kernel larger than 1x1 (${upKey})`)
        continue
      }
      if (!downOk || !upOk) {
        note(`the layer ${base} has weights of an unexpected shape`)
        continue
      }
      if (!FLOAT_DTYPES.has(dt.dtype) || !FLOAT_DTYPES.has(ut.dtype)) {
        note(`it stores its weights as ${!FLOAT_DTYPES.has(dt.dtype) ? dt.dtype : ut.dtype}; only BF16, F16, F32 and F64 can be merged`)
        continue
      }
      if (dt.shape[0] !== ut.shape[1]) {
        note(`the layer ${base} has different ranks in its up (${ut.shape[1]}) and down (${dt.shape[0]}) weights`)
        continue
      }
      const alphaKey = `${base}.alpha`
      let alpha: string | null = null
      if (keySet.has(alphaKey)) {
        const at = header.tensors[alphaKey]
        if (!FLOAT_DTYPES.has(at.dtype) || numel(at.shape) !== 1) {
          note(`the alpha value of ${base} is not a single number`)
          continue
        }
        alpha = alphaKey
      }
      let inFlat = 1
      for (const d of dt.shape.slice(1)) inFlat *= d
      layers.set(match, {
        match,
        base,
        down: key,
        up: upKey,
        alpha,
        rank: dt.shape[0],
        out: ut.shape[0],
        inFlat,
        downShape: dt.shape,
        upShape: ut.shape,
        textEncoder: isTextEncoderLoraKey(base)
      })
      if (firstFamily === null) firstFamily = entry.family
      familyCount.set(entry.family, (familyCount.get(entry.family) ?? 0) + 1)
    }
  }
  if (layers.size === 0) note('it has no LoRA layers (no matching up and down weights)')
  let family: Family = firstFamily ?? 'updown'
  let best = 0
  for (const [f, n] of familyCount) {
    if (n > best) {
      best = n
      family = f
    }
  }
  const keyStyle = describeLoraFile('', header, null, null).keyStyle
  return { layers, family, keyStyle, unsupported }
}

function describeFormat(header: SafetensorsHeader): { format: string; hint: string | null } {
  const d = describeLoraFile('', header, null, null)
  return { format: d.format, hint: d.baseModelHint }
}

export function describeMergeInput(name: string, header: SafetensorsHeader | null, error: string | null): LoraMergeInputInfo {
  if (!header) {
    const why = error ?? 'The file was not read.'
    return {
      name,
      readable: false,
      error: why,
      sizeBytes: null,
      format: 'unknown',
      keyStyle: 'unknown',
      modules: 0,
      textEncoderModules: 0,
      rank: null,
      dtype: null,
      baseModelHint: null,
      supported: false,
      unsupportedReason: why
    }
  }
  const parsed = parseLoraLayers(header)
  const { format, hint } = describeFormat(header)
  let min = Infinity
  let max = 0
  let te = 0
  const dtypes = new Set<string>()
  for (const l of parsed.layers.values()) {
    min = Math.min(min, l.rank)
    max = Math.max(max, l.rank)
    if (l.textEncoder) te++
    dtypes.add(header.tensors[l.up].dtype)
  }
  const reason = parsed.unsupported === null ? null : sentence(parsed.unsupported)
  return {
    name,
    readable: true,
    error: null,
    sizeBytes: header.fileBytes,
    format,
    keyStyle: parsed.keyStyle,
    modules: parsed.layers.size,
    textEncoderModules: te,
    rank: parsed.layers.size > 0 ? { min, max } : null,
    dtype: dtypes.size === 0 ? null : dtypes.size === 1 ? [...dtypes][0] : 'mixed',
    baseModelHint: hint,
    supported: parsed.unsupported === null,
    unsupportedReason: reason
  }
}

function sentence(s: string): string {
  const t = s.charAt(0).toUpperCase() + s.slice(1)
  return t.endsWith('.') ? t : `${t}.`
}

function dtypeSize(d: LoraMergeDtype, sourceDtype: string | null): number | null {
  if (d === 'bf16' || d === 'fp16') return 2
  if (d === 'fp32') return 4
  if (sourceDtype === 'BF16' || sourceDtype === 'F16') return 2
  if (sourceDtype === 'F32' || sourceDtype === 'F64') return 4
  return null
}

function sameShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i])
}

export function checkLoraMerge(
  infos: LoraMergeInputInfo[],
  headers: Array<SafetensorsHeader | null>,
  req: { mode: LoraMergeMode; rank: number; dtype: LoraMergeDtype; weights?: number[] }
): LoraMergeCheck {
  const problems: string[] = []
  const warnings: string[] = []
  if (infos.length < 2) problems.push('Choose at least two LoRAs to merge.')
  if (infos.length > MAX_LORA_MERGE_INPUTS) problems.push(`A merge combines at most ${MAX_LORA_MERGE_INPUTS} LoRAs.`)

  const usable: Array<{ index: number; info: LoraMergeInputInfo; layers: Map<string, LoraLayer>; header: SafetensorsHeader }> = []
  infos.forEach((info, index) => {
    if (!info.readable || !info.supported) {
      const why = info.unsupportedReason ?? info.error ?? 'it cannot be merged.'
      problems.push(`${info.name}: ${why}`)
      return
    }
    const header = headers[index]
    if (!header) {
      problems.push(`${info.name}: the file was not read.`)
      return
    }
    usable.push({ index, info, layers: parseLoraLayers(header).layers, header })
  })

  let styleProblem = false
  if (usable.length > 1) {
    const first = usable[0]
    for (const u of usable.slice(1)) {
      if (u.info.keyStyle !== first.info.keyStyle) {
        styleProblem = true
        problems.push(
          `${first.info.name} and ${u.info.name} use different layer names (${first.info.keyStyle} / ${u.info.keyStyle}); ` +
            'they were probably made by different trainers or for different models.'
        )
      }
    }
  }

  // Layers by match key.
  const byMatch = new Map<string, Array<{ input: number; layer: LoraLayer }>>()
  for (const u of usable) {
    for (const [match, layer] of u.layers) {
      const list = byMatch.get(match)
      if (list) list.push({ input: u.index, layer })
      else byMatch.set(match, [{ input: u.index, layer }])
    }
  }
  let shared = 0
  const badShape: string[] = []
  let badCount = 0
  for (const [match, list] of byMatch) {
    if (list.length < 2) continue
    shared++
    const a = list[0].layer
    const bad = list.some(
      (x) => x.layer.out !== a.out || x.layer.inFlat !== a.inFlat || !sameShape(x.layer.downShape.slice(2), a.downShape.slice(2))
    )
    if (bad) {
      badCount++
      if (badShape.length < 3) badShape.push(match)
    }
  }
  if (usable.length > 1 && shared === 0 && !styleProblem) {
    problems.push(
      'No layer is shared by two of the LoRAs, so they are for different model types and cannot be merged into one.'
    )
  }
  if (badCount > 0) {
    problems.push(
      `${badCount} shared layer${badCount === 1 ? ' has' : 's have'} a different shape between the LoRAs (for example ${badShape.join(', ')}); they are for different models.`
    )
  }
  if (req.mode === 'shrink' && (!Number.isInteger(req.rank) || req.rank < 1 || req.rank > 1024)) {
    problems.push('The target rank must be a whole number from 1 to 1024.')
  }

  if (usable.length > 1 && shared > 0) {
    const smaller = Math.min(...usable.map((u) => u.layers.size))
    if (smaller > 0 && shared / smaller < 0.5) {
      warnings.push(
        `Only ${Math.round((shared / smaller) * 100)}% of the smaller LoRA's layers are also in another LoRA; the rest are copied over unchanged by their weight.`
      )
    }
  }
  const hints = new Set(usable.map((u) => u.info.baseModelHint).filter((h): h is string => !!h))
  const totalInputBytes = infos.reduce((n, i) => n + (i.sizeBytes ?? 0), 0)
  if (totalInputBytes > LARGE_INPUT_BYTES) {
    warnings.push(
      `The LoRAs add up to ${(totalInputBytes / 1024 ** 3).toFixed(1)} GiB. The app holds all of them and the result in memory while merging; close other programs if memory is short.`
    )
  }
  if (hints.size > 1) warnings.push(`The LoRAs name different base models (${[...hints].join(' / ')}).`)
  if (req.weights) {
    infos.forEach((info, i) => {
      if (req.weights![i] === 0) warnings.push(`${info.name} has weight 0, so it adds nothing to the result.`)
    })
  }

  let outputRank: { min: number; max: number } | null = null
  let estimatedBytes: number | null = null
  const firstUsable = usable[0]
  if (byMatch.size > 0 && firstUsable) {
    let min = Infinity
    let max = 0
    let bytes = 0
    const sourceDtype = firstUsable.info.dtype === 'mixed' ? null : firstUsable.info.dtype
    const per = dtypeSize(req.dtype, sourceDtype)
    for (const list of byMatch.values()) {
      let sum = 0
      for (const x of list) sum += x.layer.rank
      const r = req.mode === 'shrink' ? Math.max(1, Math.min(Math.floor(req.rank) || 1, sum)) : sum
      min = Math.min(min, r)
      max = Math.max(max, r)
      const l = list[0].layer
      if (per !== null) bytes += (l.out + l.inFlat) * r * per + 4 + 3 * (l.base.length + 90)
    }
    outputRank = { min, max }
    estimatedBytes = per === null ? null : bytes + 1024
  }
  return {
    inputs: infos,
    problems,
    warnings,
    totalModules: byMatch.size,
    sharedModules: shared,
    outputRank,
    estimatedBytes
  }
}

export interface MergeInput {
  name: string
  header: SafetensorsHeader
  data: Buffer
  weight: number
}

export interface MergeOptions {
  appVersion: string
  inputHeaderSha256: Array<string | null>
  onProgress?: (done: number, total: number) => void
  signal?: AbortSignal
  now?: () => Date
}

export interface MergeResult {
  tensors: Array<{ key: string; dtype: string; shape: number[]; data: Buffer }>
  metadata: Record<string, string>
  modules: number
  rank: { min: number; max: number }
  energyKept: { mean: number; min: number } | null
}

function abortError(): Error {
  const e = new Error('The merge was cancelled.')
  e.name = 'AbortError'
  return e
}

function layerScale(input: MergeInput, layer: LoraLayer): number {
  if (layer.alpha === null) return input.weight
  const alpha = tensorValues(input.header, input.data, layer.alpha)[0]
  return (input.weight * alpha) / layer.rank
}

function outputDtype(recipeDtype: LoraMergeDtype, firstUpDtype: string | null): 'BF16' | 'F16' | 'F32' {
  if (recipeDtype === 'bf16') return 'BF16'
  if (recipeDtype === 'fp16') return 'F16'
  if (recipeDtype === 'fp32') return 'F32'
  if (firstUpDtype === 'BF16') return 'BF16'
  if (firstUpDtype === 'F16') return 'F16'
  return 'F32'
}

export async function mergeLoras(inputs: MergeInput[], recipe: LoraMergeRecipe, opts: MergeOptions): Promise<MergeResult> {
  const infos = inputs.map((i) => describeMergeInput(i.name, i.header, null))
  const check = checkLoraMerge(
    infos,
    inputs.map((i) => i.header),
    { mode: recipe.mode, rank: recipe.rank, dtype: recipe.dtype, weights: inputs.map((i) => i.weight) }
  )
  if (check.problems.length > 0) throw new Error(check.problems.join(' '))
  if (inputs.some((i) => !Number.isFinite(i.weight))) throw new Error('Every weight must be a number.')

  const parsed = inputs.map((i) => parseLoraLayers(i.header))
  const order: string[] = []
  const sources = new Map<string, Array<{ input: number; layer: LoraLayer }>>()
  parsed.forEach((p, input) => {
    for (const [match, layer] of p.layers) {
      const list = sources.get(match)
      if (list) list.push({ input, layer })
      else {
        sources.set(match, [{ input, layer }])
        order.push(match)
      }
    }
  })

  // Suffixes of the first input; dtype from its first layer's up weights.
  const firstLayer = parsed[0].layers.values().next().value as LoraLayer
  const downSuffix = firstLayer.down.slice(firstLayer.base.length)
  const upSuffix = firstLayer.up.slice(firstLayer.base.length)
  const dtype = outputDtype(recipe.dtype, inputs[0].header.tensors[firstLayer.up].dtype)

  const tensors: MergeResult['tensors'] = []
  let rankMin = Infinity
  let rankMax = 0
  let energySum = 0
  let energyMin = 1
  const shrink = recipe.mode === 'shrink'
  const total = order.length
  let done = 0
  for (const match of order) {
    if (opts.signal?.aborted) throw abortError()
    const list = sources.get(match)!
    const ref = list[0].layer
    const { out, inFlat } = ref
    let R = 0
    for (const x of list) R += x.layer.rank
    let up: Float64Array = new Float64Array(out * R)
    let down: Float64Array = new Float64Array(R * inFlat)
    let col = 0
    for (const x of list) {
      const inp = inputs[x.input]
      const l = x.layer
      const scale = layerScale(inp, l)
      const u = tensorValues(inp.header, inp.data, l.up)
      const d = tensorValues(inp.header, inp.data, l.down)
      // Shrink: balance this input's factors (same Frobenius norm for scale*up and down, delta unchanged) so no
      // input's directions fall under the rank-drop threshold of the Gram eigen-decompositions just because its
      // up and down weights have very different magnitudes. Concat keeps the factors as they are.
      let cu = scale
      let cd = 1
      if (shrink) {
        let nu = 0
        let nd = 0
        for (let i = 0; i < u.length; i++) nu += u[i] * u[i]
        for (let i = 0; i < d.length; i++) nd += d[i] * d[i]
        nu = Math.sqrt(nu) * Math.abs(scale)
        nd = Math.sqrt(nd)
        if (nu > 0 && nd > 0 && Number.isFinite(nu) && Number.isFinite(nd)) {
          const c = Math.sqrt(nd / nu)
          cu = scale * c
          cd = 1 / c
        }
      }
      for (let o = 0; o < out; o++) {
        const src = o * l.rank
        const dst = o * R + col
        for (let c = 0; c < l.rank; c++) up[dst + c] = cu * u[src + c]
      }
      if (cd === 1) down.set(d, col * inFlat)
      else for (let i = 0; i < d.length; i++) down[col * inFlat + i] = cd * d[i]
      col += l.rank
    }
    let outRank = R
    if (shrink) {
      if (R > recipe.rank) {
        const s = shrinkLayer(up, down, out, inFlat, R, recipe.rank)
        up = s.up
        down = s.down
        outRank = s.k
        energySum += s.energyKept
        energyMin = Math.min(energyMin, s.energyKept)
      } else {
        energySum += 1
      }
    }
    rankMin = Math.min(rankMin, outRank)
    rankMax = Math.max(rankMax, outRank)
    const downShape = [outRank, ...(ref.downShape.length === 4 ? ref.downShape.slice(1) : [inFlat])]
    const upShape = ref.upShape.length === 4 ? [out, outRank, 1, 1] : [out, outRank]
    const alpha = Buffer.alloc(4)
    alpha.writeFloatLE(outRank, 0)
    tensors.push({ key: `${ref.base}.alpha`, dtype: 'F32', shape: [], data: alpha })
    tensors.push({ key: ref.base + downSuffix, dtype, shape: downShape, data: encodeValues(down, dtype) })
    tensors.push({ key: ref.base + upSuffix, dtype, shape: upShape, data: encodeValues(up, dtype) })
    done++
    opts.onProgress?.(done, total)
    await new Promise((r) => setImmediate(r))
    if (opts.signal?.aborted) throw abortError()
  }

  const record: LoraMergeRecord = {
    app: 'RulZ Model Merger',
    appVersion: opts.appVersion,
    kind: 'lora-merge',
    createdAt: (opts.now ? opts.now() : new Date()).toISOString(),
    recipe,
    inputHeaderSha256: opts.inputHeaderSha256
  }
  const metadata: Record<string, string> = {
    format: 'pt',
    rulz_model_merger: JSON.stringify(record),
    ss_network_dim: String(rankMax),
    ss_network_alpha: String(rankMax)
  }
  for (const key of ['ss_base_model', 'modelspec.architecture']) {
    const values = new Set(inputs.map((i) => i.header.metadata[key]).filter((v): v is string => typeof v === 'string' && v !== ''))
    if (values.size === 1) metadata[key] = [...values][0]
  }
  return {
    tensors,
    metadata,
    modules: total,
    rank: { min: rankMin, max: rankMax },
    energyKept: shrink ? { mean: total > 0 ? energySum / total : 1, min: energyMin } : null
  }
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** y = up (out x r) @ (down (r x n) @ v), times scale. */
function applyLayer(up: Float64Array, down: Float64Array, out: number, r: number, n: number, v: Float64Array, scale: number): Float64Array {
  const t = new Float64Array(r)
  for (let i = 0; i < r; i++) {
    let sum = 0
    const row = i * n
    for (let c = 0; c < n; c++) sum += down[row + c] * v[c]
    t[i] = sum
  }
  const y = new Float64Array(out)
  for (let o = 0; o < out; o++) {
    let sum = 0
    const row = o * r
    for (let i = 0; i < r; i++) sum += up[row + i] * t[i]
    y[o] = sum * scale
  }
  return y
}

export function verifyMergedLora(
  merged: { header: SafetensorsHeader; data: Buffer },
  inputs: Array<{ header: SafetensorsHeader; data: Buffer; weight: number }>,
  recipe: LoraMergeRecipe
): { ok: boolean; message: string; checkedModules: number; maxRelError: number } {
  const fail = (message: string, checked = 0, err = 0): { ok: boolean; message: string; checkedModules: number; maxRelError: number } => ({
    ok: false,
    message,
    checkedModules: checked,
    maxRelError: err
  })
  const out = parseLoraLayers(merged.header)
  if (out.unsupported !== null) return fail(`The saved file is not a plain LoRA: ${out.unsupported}.`)
  const parsed = inputs.map((i) => parseLoraLayers(i.header))
  const expected = new Map<string, Array<{ input: number; layer: LoraLayer }>>()
  parsed.forEach((p, input) => {
    for (const [match, layer] of p.layers) {
      const list = expected.get(match)
      if (list) list.push({ input, layer })
      else expected.set(match, [{ input, layer }])
    }
  })
  if (out.layers.size !== expected.size) {
    return fail(`The saved file has ${out.layers.size} layers but the inputs have ${expected.size} between them.`)
  }
  const matches: string[] = []
  for (const [match, layer] of out.layers) {
    const src = expected.get(match)
    if (!src) return fail(`The saved file has a layer (${layer.base}) that no input has.`)
    const ref = src[0].layer
    if (layer.out !== ref.out || layer.inFlat !== ref.inFlat) return fail(`The layer ${layer.base} has the wrong shape in the saved file.`)
    const sum = src.reduce((n, x) => n + x.layer.rank, 0)
    if (recipe.mode === 'concat' && layer.rank !== sum) {
      return fail(`The layer ${layer.base} has rank ${layer.rank} in the saved file but should have ${sum}.`)
    }
    if (recipe.mode === 'shrink' && (layer.rank < 1 || layer.rank > Math.max(sum, 1) || layer.rank > Math.max(recipe.rank, 1))) {
      return fail(`The layer ${layer.base} has rank ${layer.rank} in the saved file, outside what the recipe allows.`)
    }
    if (layer.alpha === null) return fail(`The layer ${layer.base} has no alpha value in the saved file.`)
    const alpha = tensorValues(merged.header, merged.data, layer.alpha)[0]
    if (alpha !== layer.rank) return fail(`The layer ${layer.base} has alpha ${alpha} but rank ${layer.rank}.`)
    matches.push(match)
  }
  // Finite values everywhere.
  for (const layer of out.layers.values()) {
    for (const key of [layer.up, layer.down]) {
      const vals = tensorValues(merged.header, merged.data, key)
      for (let i = 0; i < vals.length; i++) {
        if (!Number.isFinite(vals[i])) return fail(`The saved file has a value that is not a finite number (${key}).`)
      }
    }
  }
  const picks = Math.min(16, matches.length)
  const rand = mulberry32(0x5eed1234)
  let maxRel = 0
  for (let p = 0; p < picks; p++) {
    const match = matches[picks === 1 ? 0 : Math.round((p * (matches.length - 1)) / (picks - 1))]
    const layer = out.layers.get(match)!
    const src = expected.get(match)!
    const v = new Float64Array(layer.inFlat)
    for (let i = 0; i < v.length; i++) v[i] = rand() * 2 - 1
    const got = applyLayer(
      tensorValues(merged.header, merged.data, layer.up),
      tensorValues(merged.header, merged.data, layer.down),
      layer.out,
      layer.rank,
      layer.inFlat,
      v,
      1
    )
    const want = new Float64Array(layer.out)
    for (const x of src) {
      const inp = inputs[x.input]
      const alphaVal = x.layer.alpha === null ? null : tensorValues(inp.header, inp.data, x.layer.alpha)[0]
      const scale = alphaVal === null ? inp.weight : (inp.weight * alphaVal) / x.layer.rank
      const y = applyLayer(
        tensorValues(inp.header, inp.data, x.layer.up),
        tensorValues(inp.header, inp.data, x.layer.down),
        x.layer.out,
        x.layer.rank,
        x.layer.inFlat,
        v,
        scale
      )
      for (let o = 0; o < want.length; o++) want[o] += y[o]
    }
    let diff = 0
    let norm = 0
    for (let o = 0; o < want.length; o++) {
      diff += (got[o] - want[o]) ** 2
      norm += want[o] ** 2
    }
    const rel = norm > 0 ? Math.sqrt(diff / norm) : Math.sqrt(diff)
    maxRel = Math.max(maxRel, rel)
  }
  const pct = `${(maxRel * 100).toFixed(1)}%`
  const layersWord = `${picks} layer${picks === 1 ? '' : 's'}`
  if (recipe.mode === 'concat') {
    if (maxRel > 0.02) {
      return {
        ok: false,
        message: `Checked ${layersWord}: the merged LoRA differs from the weighted sum by up to ${pct}, more than rounding explains.`,
        checkedModules: picks,
        maxRelError: maxRel
      }
    }
    return {
      ok: true,
      message: `Checked ${layersWord}: the merged LoRA equals the weighted sum (largest difference ${pct}).`,
      checkedModules: picks,
      maxRelError: maxRel
    }
  }
  return {
    ok: true,
    message: `Checked ${layersWord}: the shrunk LoRA differs from the weighted sum by at most ${pct} (the part dropped by the lower rank).`,
    checkedModules: picks,
    maxRelError: maxRel
  }
}
