// Pure checks for a merge: what a safetensors header says about a model, and whether two or three files can be merged.
import type { BlockNodeOption, LoraFileInfo, MergeCheck, MergeMode, MergeSourceKind, ModelFileInfo } from '@shared/merge'
import type { SafetensorsHeader, SafetensorsTensor } from '../models/safetensors'

const CHECKPOINT_PREFIX = 'model.diffusion_model.'
const UNET_PREFIX = 'diffusion_model.'
const QUANT_WEIGHT_DTYPES = new Set(['I8', 'U8', 'F8_E4M3', 'F8_E5M2', 'F4', 'U4'])
const EXCLUDED_BLOCK_NODES = new Set(['ModelMergeSimple', 'ModelMergeAdd', 'ModelMergeSubtract'])

function asRecord(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}

function detectPrefix(keys: string[]): string {
  if (keys.some((k) => k.startsWith(CHECKPOINT_PREFIX))) return CHECKPOINT_PREFIX
  if (keys.some((k) => k.startsWith(UNET_PREFIX))) return UNET_PREFIX
  return ''
}

/** The diffusion model's tensors with the prefix removed. For a full checkpoint only keys under the prefix count. */
export function normalizeKeys(header: SafetensorsHeader): { prefix: string; keys: Map<string, SafetensorsTensor> } {
  const names = Object.keys(header.tensors)
  const prefix = detectPrefix(names)
  const keys = new Map<string, SafetensorsTensor>()
  for (const name of names) {
    if (prefix === '') keys.set(name, header.tensors[name])
    else if (name.startsWith(prefix)) keys.set(name.slice(prefix.length), header.tensors[name])
  }
  return { prefix, keys }
}

function labelFromText(text: string): string | null {
  const t = text.toLowerCase()
  if (t.includes('nvfp4')) return 'nvfp4'
  if (t.includes('int8')) return 'int8'
  if (t.includes('fp8') || t.includes('float8')) return 'fp8'
  if (t.includes('fp4') || t.includes('float4')) return 'fp4'
  return null
}

function detectQuantization(
  header: SafetensorsHeader,
  keys: Map<string, SafetensorsTensor>,
  name: string
): { quantized: boolean; label: string | null } {
  const hasScales = [...keys.keys()].some((k) => /scale|comfy_quant/.test(k))
  const hasScale2 = [...keys.keys()].some((k) => k.endsWith('weight_scale_2'))
  const formats = new Set<string>()
  if (hasScales) {
    for (const [k, t] of keys) {
      if (!k.endsWith('.weight') || !QUANT_WEIGHT_DTYPES.has(t.dtype)) continue
      if (t.dtype === 'I8') formats.add('int8')
      else if (t.dtype === 'F8_E4M3' || t.dtype === 'F8_E5M2') formats.add('fp8')
      else if (t.dtype === 'F4') formats.add('fp4')
      else formats.add(hasScale2 ? 'nvfp4' : 'fp4') // U8 / U4 hold packed 4-bit values
    }
  }
  const metaKeys = Object.keys(header.metadata).filter((k) => /quant/i.test(k))
  const quantized = formats.size > 0 || metaKeys.length > 0
  if (!quantized) return { quantized: false, label: null }
  const hint = labelFromText(name)
  let label: string | null
  if (formats.size > 1) label = 'mixed'
  else if (formats.size === 1) {
    label = [...formats][0]
    if (label === 'fp4' && hint === 'nvfp4') label = 'nvfp4' // weak hint: the name settles an ambiguous 4-bit format
  } else {
    label = labelFromText(metaKeys.map((k) => header.metadata[k]).join(' ')) ?? hint
  }
  return { quantized, label }
}

export function describeModelFile(name: string, header: SafetensorsHeader | null, error?: string): ModelFileInfo {
  if (!header) {
    return {
      name,
      readable: false,
      error: error ?? 'The file was not read.',
      sizeBytes: null,
      tensorCount: 0,
      dtypes: {},
      quantized: false,
      quantLabel: null,
      keyPrefix: ''
    }
  }
  const dtypes: Record<string, number> = {}
  for (const t of Object.values(header.tensors)) dtypes[t.dtype] = (dtypes[t.dtype] ?? 0) + 1
  const { prefix, keys } = normalizeKeys(header)
  const q = detectQuantization(header, keys, name)
  return {
    name,
    readable: true,
    error: null,
    sizeBytes: header.fileBytes,
    tensorCount: Object.keys(header.tensors).length,
    dtypes,
    quantized: q.quantized,
    quantLabel: q.label,
    keyPrefix: prefix
  }
}

function floatInputs(node: Record<string, unknown>): { required: Record<string, unknown>; floats: string[] } {
  const input = asRecord(node.input)
  const required = asRecord(input.required)
  const optional = asRecord(input.optional)
  const floats: string[] = []
  for (const group of [required, optional]) {
    for (const [k, spec] of Object.entries(group)) {
      if (Array.isArray(spec) && spec[0] === 'FLOAT') floats.push(k)
    }
  }
  return { required, floats }
}

/** Block merge nodes on the server (two MODEL inputs plus FLOAT block inputs), scored against model A's keys. */
export function blockNodeOptions(objectInfo: Record<string, unknown>, aHeader: SafetensorsHeader | null): BlockNodeOption[] {
  const keyList = aHeader ? [...normalizeKeys(aHeader).keys.keys()] : null
  const out: BlockNodeOption[] = []
  for (const [name, raw] of Object.entries(objectInfo)) {
    if (!name.startsWith('ModelMerge') || EXCLUDED_BLOCK_NODES.has(name)) continue
    const { required, floats } = floatInputs(asRecord(raw))
    const isModel = (k: string): boolean => {
      const spec = required[k]
      return Array.isArray(spec) && spec[0] === 'MODEL'
    }
    if (!isModel('model1') || !isModel('model2') || floats.length === 0) continue
    let match: number | null = null
    let activeInputs = floats
    if (keyList) {
      activeInputs = floats.filter((f) => keyList.some((k) => k.startsWith(f)))
      // Share of A's layers that some input controls. Counting matched inputs instead would rank a generic DiT node
      // above the model's own node when that node has inputs for more blocks than this model has.
      const covered = keyList.filter((k) => floats.some((f) => k.startsWith(f))).length
      match = keyList.length > 0 ? covered / keyList.length : 0
    }
    out.push({ node: name, inputs: floats, match, activeInputs })
  }
  out.sort((x, y) => {
    if (x.match === null && y.match !== null) return 1
    if (x.match !== null && y.match === null) return -1
    if (x.match !== null && y.match !== null && x.match !== y.match) return y.match - x.match
    return x.node.localeCompare(y.node)
  })
  return out
}

const BASE_MODEL_PREFIX = 'base_model.model.'
const TE_PREFIXES = ['lora_te', 'text_encoder.', 'text_encoders.', 'text_model.', 'clip_l.', 'clip_g.', 't5xxl.', 'conditioner.']
const TE_PATTERN = /^(te\d?\.|text_encoders?_\d+\.)/

/** True for a LoRA key that belongs to a text encoder (lora_te*, text_encoder.*, te1.*, clip_l.*, t5xxl.*, ...). */
export function isTextEncoderLoraKey(key: string): boolean {
  const k = key.startsWith(BASE_MODEL_PREFIX) ? key.slice(BASE_MODEL_PREFIX.length) : key
  return TE_PREFIXES.some((p) => k.startsWith(p)) || TE_PATTERN.test(k)
}

type LoraFormat = 'LoRA' | 'LoKr' | 'LoHa'
// Longest suffixes first. rank: the suffix names a down / A matrix (its smaller dimension is the rank).
const LORA_SUFFIXES: Array<{ suffix: string; fmt: LoraFormat | 'DoRA' | null; rank?: boolean }> = [
  { suffix: '.lora_down.weight', fmt: 'LoRA', rank: true },
  { suffix: '.lora_up.weight', fmt: 'LoRA' },
  { suffix: '.lora_A.default.weight', fmt: 'LoRA', rank: true },
  { suffix: '.lora_B.default.weight', fmt: 'LoRA' },
  { suffix: '.lora_A.weight', fmt: 'LoRA', rank: true },
  { suffix: '.lora_B.weight', fmt: 'LoRA' },
  { suffix: '.lora.down.weight', fmt: 'LoRA', rank: true },
  { suffix: '.lora.up.weight', fmt: 'LoRA' },
  { suffix: '.lora_mid.weight', fmt: 'LoRA' },
  { suffix: '.lokr_w1_a', fmt: 'LoKr' },
  { suffix: '.lokr_w1_b', fmt: 'LoKr' },
  { suffix: '.lokr_w2_a', fmt: 'LoKr' },
  { suffix: '.lokr_w2_b', fmt: 'LoKr' },
  { suffix: '.lokr_w1', fmt: 'LoKr' },
  { suffix: '.lokr_w2', fmt: 'LoKr' },
  { suffix: '.lokr_t2', fmt: 'LoKr' },
  { suffix: '.hada_w1_a', fmt: 'LoHa' },
  { suffix: '.hada_w1_b', fmt: 'LoHa' },
  { suffix: '.hada_w2_a', fmt: 'LoHa' },
  { suffix: '.hada_w2_b', fmt: 'LoHa' },
  { suffix: '.hada_t1', fmt: 'LoHa' },
  { suffix: '.hada_t2', fmt: 'LoHa' },
  { suffix: '.dora_scale', fmt: 'DoRA' },
  { suffix: '.alpha', fmt: null },
  { suffix: '.diff_b', fmt: null },
  { suffix: '.diff', fmt: null },
  { suffix: '.w_norm', fmt: null },
  { suffix: '.b_norm', fmt: null }
]

const COMFY_PREFIXES = ['model.diffusion_model.', 'diffusion_model.']
const DIFFUSERS_PREFIXES = ['transformer.', 'unet.', BASE_MODEL_PREFIX]

function loraKeyStyle(module: string): LoraFileInfo['keyStyle'] {
  if (COMFY_PREFIXES.some((p) => module.startsWith(p))) return 'comfy'
  if (module.startsWith('lora_unet_')) return 'kohya'
  if (DIFFUSERS_PREFIXES.some((p) => module.startsWith(p))) return 'diffusers'
  return 'unknown'
}

/** True when a LoRA module ("layers.0.proj.k") targets one part of a weight A stores fused ("layers.0.proj.qkv",
 *  "attn.to_qkv", "ffn.up_gate"). ComfyUI maps such LoRA keys onto the fused weight when it loads them. */
export function isFusedInto(module: string, children: Map<string, string[]>): boolean {
  const dot = module.lastIndexOf('.')
  if (dot < 0) return false
  const parent = module.slice(0, dot)
  const part = bare(module.slice(dot + 1))
  if (!part) return false
  for (const sibling of children.get(parent) ?? []) {
    const pieces = bare(sibling).split('_').filter(Boolean)
    if (pieces.length > 1 && pieces.includes(part)) return true
    // "qkv" holds q, k and v.
    if (pieces.length === 1 && part.length === 1 && /^[qkv]{2,3}$/.test(pieces[0]) && pieces[0].includes(part)) return true
  }
  return false
}

/** Module names of the parts a LoRA may train for each fused weight of A: "x.qkv" gives x.q, x.to_q, x.q_proj ... */
export function fusedPartNames(children: Map<string, string[]>): string[] {
  const out: string[] = []
  for (const [parent, kids] of children) {
    for (const kid of kids) {
      const b = bare(kid)
      const pieces = b.split('_').filter(Boolean)
      let parts: string[] = []
      if (pieces.length > 1) parts = pieces
      else if (/^[qkv]{2,3}$/.test(b)) parts = b.split('')
      for (const p of parts) out.push(`${parent}.${p}`, `${parent}.to_${p}`, `${parent}.${p}_proj`)
    }
  }
  return out
}

function bare(name: string): string {
  return name.replace(/^to_/, '').replace(/_proj$/, '')
}

export function describeLoraFile(
  name: string,
  header: SafetensorsHeader | null,
  aHeader: SafetensorsHeader | null,
  error: string | null
): LoraFileInfo {
  if (!header) {
    return {
      name,
      readable: false,
      error: error ?? 'The file was not read.',
      sizeBytes: null,
      tensorCount: 0,
      format: 'unknown',
      rank: null,
      keyStyle: 'unknown',
      modules: 0,
      match: null,
      textEncoderModules: 0,
      baseModelHint: null
    }
  }
  const formats = new Set<LoraFormat>()
  let dora = false
  let rank: number | null = null
  const diffusionModules = new Set<string>()
  const teModules = new Set<string>()
  for (const [key, t] of Object.entries(header.tensors)) {
    const hit = LORA_SUFFIXES.find((x) => key.endsWith(x.suffix))
    if (!hit) continue
    const module = key.slice(0, key.length - hit.suffix.length)
    if (module === '') continue
    if (hit.fmt === 'DoRA') dora = true
    else if (hit.fmt) formats.add(hit.fmt)
    if (hit.rank && rank === null && t.shape.length >= 2) rank = Math.min(t.shape[0], t.shape[1])
    if (isTextEncoderLoraKey(module)) teModules.add(module)
    else diffusionModules.add(module)
  }
  let format: string
  if (formats.size === 0) format = 'unknown'
  else if (formats.size > 1) format = 'mixed'
  else format = [...formats][0]
  if (dora && formats.size === 1 && formats.has('LoRA')) format = 'DoRA'

  const styleCount = new Map<LoraFileInfo['keyStyle'], number>()
  for (const m of diffusionModules) {
    const st = loraKeyStyle(m)
    styleCount.set(st, (styleCount.get(st) ?? 0) + 1)
  }
  let keyStyle: LoraFileInfo['keyStyle'] = 'unknown'
  let best = 0
  for (const st of ['comfy', 'kohya', 'diffusers', 'unknown'] as const) {
    const n = styleCount.get(st) ?? 0
    if (n > best) {
      best = n
      keyStyle = st
    }
  }

  let match: number | null = null
  if (aHeader && (keyStyle === 'comfy' || keyStyle === 'kohya') && diffusionModules.size > 0) {
    const aKeys = [...normalizeKeys(aHeader).keys.keys()]
    let hits = 0
    // Weights of A per parent module, to find fused layers (q/k/v trained apart, stored as one qkv weight).
    const children = new Map<string, string[]>()
    for (const k of aKeys) {
      if (!k.endsWith('.weight')) continue
      const mod = k.slice(0, -'.weight'.length)
      const dot = mod.lastIndexOf('.')
      if (dot < 0) continue
      const parent = mod.slice(0, dot)
      children.set(parent, [...(children.get(parent) ?? []), mod.slice(dot + 1)])
    }
    if (keyStyle === 'comfy') {
      const aSet = new Set(aKeys)
      for (const m of diffusionModules) {
        const prefix = COMFY_PREFIXES.find((p) => m.startsWith(p)) ?? ''
        const x = m.slice(prefix.length)
        if (aSet.has(`${x}.weight`) || aSet.has(x)) hits++
        else if (isFusedInto(x, children)) hits++
      }
    } else {
      const flatten = (s: string): string => s.replace(/\./g, '_')
      const flat = new Set(aKeys.map((k) => flatten(k.endsWith('.weight') ? k.slice(0, -'.weight'.length) : k)))
      // kohya names lose the dots, so the parts of fused weights are listed under their flattened names too.
      for (const v of fusedPartNames(children)) flat.add(flatten(v))
      for (const m of diffusionModules) if (flat.has(m.slice('lora_unet_'.length))) hits++
    }
    match = hits / diffusionModules.size
  }

  let baseModelHint: string | null = null
  for (const k of ['ss_base_model', 'modelspec.architecture', 'ss_base_model_version', 'base_model']) {
    const v = (header.metadata[k] ?? '').trim()
    if (v !== '') {
      baseModelHint = v.slice(0, 80)
      break
    }
  }

  return {
    name,
    readable: true,
    error: null,
    sizeBytes: header.fileBytes,
    tensorCount: Object.keys(header.tensors).length,
    format,
    rank,
    keyStyle,
    modules: diffusionModules.size,
    match,
    textEncoderModules: teModules.size,
    baseModelHint
  }
}

interface Source {
  info: ModelFileInfo
  header: SafetensorsHeader | null
}

// Rough RAM estimate: the source files' sizes times 1.2, or 1.5 when any source is quantized (dequantized while merging).
const RAM_FACTOR_PLAIN = 1.2
const RAM_FACTOR_QUANT = 1.5

function formatName(info: ModelFileInfo): string {
  return info.quantized ? (info.quantLabel ?? 'quantized') : 'unquantized'
}

function comparePair(la: string, a: SafetensorsHeader, lb: string, b: SafetensorsHeader, problems: string[]): void {
  const ka = normalizeKeys(a).keys
  const kb = normalizeKeys(b).keys
  const smaller = Math.min(ka.size, kb.size)
  if (smaller === 0) {
    problems.push(`${la} and ${lb} look like different model types (no diffusion-model layers found in one of them).`)
    return
  }
  let shared = 0
  const badShape: string[] = []
  let badCount = 0
  for (const [k, ta] of ka) {
    const tb = kb.get(k)
    if (!tb) continue
    shared++
    if (ta.shape.length !== tb.shape.length || ta.shape.some((d, i) => d !== tb.shape[i])) {
      badCount++
      if (badShape.length < 3) badShape.push(k)
    }
  }
  const pct = Math.round((shared / smaller) * 100)
  if (shared / smaller < 0.9) {
    problems.push(`${la} and ${lb} look like different model types (only ${pct}% of their layers match).`)
  }
  if (badCount > 0) {
    problems.push(
      `${la} and ${lb} have ${badCount} layer${badCount === 1 ? '' : 's'} with a different shape (for example ${badShape.join(', ')}).`
    )
  }
}

export function checkMerge(input: {
  kind: MergeSourceKind
  mode: MergeMode
  a: Source | null
  b: Source | null
  c: Source | null
  objectInfo: Record<string, unknown>
  ramFreeBytes: number | null
  loras: LoraFileInfo[]
}): MergeCheck {
  const problems: string[] = []
  const warnings: string[] = []
  const { a, b, c, mode } = input
  const bake = mode === 'bake'
  const needC = mode === 'addDifference'
  const named = (s: Source | null): boolean => !!s && s.info.name.trim() !== ''

  if (!named(a)) problems.push('Choose model A.')
  if (!bake && !named(b)) problems.push('Choose model B.')
  if (needC && !named(c)) problems.push('Difference mode needs a model C (the model subtracted from B).')
  if (!bake && named(a) && named(b) && a!.info.name === b!.info.name) {
    problems.push('A and B are the same file. Choose two different models.')
  }
  if (needC && named(c) && named(a) && c!.info.name === a!.info.name) {
    problems.push('C is the same file as A. Choose a different model C.')
  }

  const labelled: Array<[string, Source | null, boolean]> = [
    ['A', a, true],
    ['B', b, !bake],
    ['C', c, needC]
  ]
  const needed = labelled.filter(([, s, need]) => need && named(s))
  const readable = (s: Source | null): s is Source => !!s && s.info.readable && s.header !== null
  let anyUnreadable = needed.length < (bake ? 1 : needC ? 3 : 2)
  for (const [label, s] of needed) {
    if (!readable(s)) {
      anyUnreadable = true
      const why = s!.info.error ? ` (${s!.info.error})` : ''
      warnings.push(`${label} could not be read${why}, so it was not checked. ComfyUI will report a mismatch itself.`)
    }
  }

  if (!bake && readable(a) && readable(b)) comparePair('A', a.header!, 'B', b.header!, problems)
  if (needC && readable(b) && readable(c)) comparePair('B', b.header!, 'C', c.header!, problems)

  // One warning per quantization label ("A and B are int8"), not one per source.
  const byLabel = new Map<string, string[]>()
  for (const [label, s] of needed) {
    if (readable(s) && s.info.quantized) {
      const q = s.info.quantLabel ?? 'quantized'
      byLabel.set(q, [...(byLabel.get(q) ?? []), label])
    }
  }
  for (const [q, labels] of byLabel) {
    const who = labels.length === 1 ? `${labels[0]} is` : `${labels.slice(0, -1).join(', ')} and ${labels[labels.length - 1]} are`
    warnings.push(`${who} ${q}: weights are merged, then saved as ${q} again, which adds a little loss. Merging the bf16 originals is best.`)
  }
  if (!bake && readable(a) && readable(b) && formatName(a.info) !== formatName(b.info)) {
    warnings.push(
      `A and B are stored differently (A: ${formatName(a.info)}, B: ${formatName(b.info)}); the result is saved in A's format.`
    )
  }

  if (bake && input.loras.length === 0) problems.push('Add at least one LoRA to bake.')
  for (const l of input.loras) {
    if (!l.readable) {
      anyUnreadable = true
      const why = l.error ? ` (${l.error})` : ''
      warnings.push(`${l.name} could not be read${why}, so it was not checked.`)
      continue
    }
    if (l.modules === 0) {
      // A checkpoint's LoraLoader can apply a text-encoder-only LoRA; a diffusion-model file has nothing for it.
      if (input.kind === 'checkpoint' && l.textEncoderModules > 0) {
        warnings.push(`${l.name} only changes the text encoder (${l.textEncoderModules} layers).`)
      } else {
        problems.push(`${l.name} has no layers for the diffusion model.`)
      }
      continue
    }
    if (l.match === null && (l.keyStyle === 'diffusers' || l.keyStyle === 'unknown')) {
      warnings.push(
        `${l.name} uses ${l.keyStyle === 'diffusers' ? 'diffusers' : 'unrecognised'} layer names, so the app cannot check it before baking. After the bake it reports any weights ComfyUI skipped.`
      )
    }
    if (l.match === 0) {
      problems.push(`${l.name} does not fit model A: none of its ${l.modules} layers are in A. It is probably for another model type.`)
    } else if (l.match !== null && l.match < 0.9) {
      warnings.push(`${l.name} fits only ${Math.round(l.match * 100)}% of model A's layers; the rest is skipped.`)
    }
    if (l.textEncoderModules > 0 && input.kind === 'diffusion') {
      warnings.push(
        `${l.name} also changes the text encoder (${l.textEncoderModules} layers); a diffusion-model file has none, so that part is left out.`
      )
    }
  }

  const readableSources = needed.map(([, s]) => s).filter(readable)
  const loraBytes = input.loras.filter((l) => l.readable).reduce((n, l) => n + (l.sizeBytes ?? 0), 0)
  let ramNeeded: number | null = null
  if (readableSources.length > 0) {
    const factor = readableSources.some((s) => s.info.quantized) ? RAM_FACTOR_QUANT : RAM_FACTOR_PLAIN
    ramNeeded = Math.round(
      (readableSources.reduce((n, s) => n + (s.info.sizeBytes ?? s.header!.fileBytes), 0) + loraBytes) * factor
    )
    if (input.ramFreeBytes !== null && ramNeeded > input.ramFreeBytes * 0.8) {
      const gib = (n: number): string => (n / 1024 ** 3).toFixed(1)
      warnings.push(
        `The merge may need about ${gib(ramNeeded)} GiB of memory and only ${gib(input.ramFreeBytes)} GiB is free. It may run slowly or fail.`
      )
    }
  }

  const blockNodes = blockNodeOptions(input.objectInfo, a?.header ?? null)
  const best = blockNodes.find((o) => o.match !== null && o.match >= 0.9)
  return {
    a: a?.info ?? null,
    b: b?.info ?? null,
    c: c?.info ?? null,
    compatible: problems.length > 0 ? false : anyUnreadable ? null : true,
    problems,
    warnings,
    blockNodes,
    suggestedBlockNode: best ? best.node : null,
    loras: input.loras,
    ramNeededBytes: ramNeeded,
    ramFreeBytes: input.ramFreeBytes
  }
}
