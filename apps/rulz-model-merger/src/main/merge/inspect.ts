// Pure checks for a merge: what a safetensors header says about a model, and whether two or three files can be merged.
import type { BlockNodeOption, MergeCheck, MergeMode, MergeSourceKind, ModelFileInfo } from '@shared/merge'
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
}): MergeCheck {
  const problems: string[] = []
  const warnings: string[] = []
  const { a, b, c, mode } = input
  const needC = mode === 'addDifference'
  const named = (s: Source | null): boolean => !!s && s.info.name.trim() !== ''

  if (!named(a)) problems.push('Choose model A.')
  if (!named(b)) problems.push('Choose model B.')
  if (needC && !named(c)) problems.push('Difference mode needs a model C (the model subtracted from B).')
  if (named(a) && named(b) && a!.info.name === b!.info.name) {
    problems.push('A and B are the same file. Choose two different models.')
  }
  if (needC && named(c) && named(a) && c!.info.name === a!.info.name) {
    problems.push('C is the same file as A. Choose a different model C.')
  }

  const labelled: Array<[string, Source | null, boolean]> = [
    ['A', a, true],
    ['B', b, true],
    ['C', c, needC]
  ]
  const needed = labelled.filter(([, s, need]) => need && named(s))
  const readable = (s: Source | null): s is Source => !!s && s.info.readable && s.header !== null
  let anyUnreadable = needed.length < (needC ? 3 : 2)
  for (const [label, s] of needed) {
    if (!readable(s)) {
      anyUnreadable = true
      const why = s!.info.error ? ` (${s!.info.error})` : ''
      warnings.push(`${label} could not be read${why}, so it was not checked. ComfyUI will report a mismatch itself.`)
    }
  }

  if (readable(a) && readable(b)) comparePair('A', a.header!, 'B', b.header!, problems)
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
  if (readable(a) && readable(b) && formatName(a.info) !== formatName(b.info)) {
    warnings.push(
      `A and B are stored differently (A: ${formatName(a.info)}, B: ${formatName(b.info)}); the result is saved in A's format.`
    )
  }

  const readableSources = needed.map(([, s]) => s).filter(readable)
  let ramNeeded: number | null = null
  if (readableSources.length > 0) {
    const factor = readableSources.some((s) => s.info.quantized) ? RAM_FACTOR_QUANT : RAM_FACTOR_PLAIN
    ramNeeded = Math.round(readableSources.reduce((n, s) => n + (s.info.sizeBytes ?? s.header!.fileBytes), 0) * factor)
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
    ramNeededBytes: ramNeeded,
    ramFreeBytes: input.ramFreeBytes
  }
}
