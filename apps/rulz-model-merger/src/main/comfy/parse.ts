// Pure parsing helpers for ComfyUI responses. No I/O here.
import { KNOWN_NODES, type GpuInfo, type ServerCapabilities } from '@shared/types'

/**
 * Turn what a person typed ("localhost:8188", "http://10.0.0.5:8188/") into "scheme://host[:port]".
 * Drops a trailing slash, any path, query and fragment. Throws an Error with a plain message when the value cannot be
 * used.
 */
export function normalizeServerUrl(input: string): string {
  const trimmed = (input ?? '').trim()
  if (!trimmed) throw new Error('Enter the address of a ComfyUI server, for example http://127.0.0.1:8188.')
  const schemeMatch = /^([a-z][a-z0-9+.-]*):\/\//i.exec(trimmed)
  if (schemeMatch) {
    const scheme = schemeMatch[1].toLowerCase()
    if (scheme !== 'http' && scheme !== 'https') {
      throw new Error(`Only http:// and https:// addresses work (got ${scheme}://).`)
    }
  }
  const withScheme = schemeMatch ? trimmed : `http://${trimmed}`
  let url: URL
  try {
    url = new URL(withScheme)
  } catch {
    throw new Error(`"${trimmed}" is not a valid server address.`)
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Only http:// and https:// addresses work (got ${url.protocol}//).`)
  }
  if (!url.hostname) throw new Error(`"${trimmed}" is not a valid server address.`)
  return `${url.protocol}//${url.host}`
}

export interface SystemStatsInfo {
  version: string | null
  devices: string[]
  argv: string[]
  cudaDevice: string | null
  ramTotalBytes: number | null
  ramFreeBytes: number | null
}

function asRecord(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/** Read GET /system_stats. Tolerates missing fields. */
export function parseSystemStats(json: unknown): SystemStatsInfo {
  const root = asRecord(json)
  const system = asRecord(root.system)
  const argv = Array.isArray(system.argv) ? system.argv.filter((a): a is string => typeof a === 'string') : []
  const devices = Array.isArray(root.devices)
    ? root.devices.map((d) => asRecord(d).name).filter((n): n is string => typeof n === 'string')
    : []
  let cudaDevice: string | null = null // the last occurrence wins, as with argparse
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--cuda-device') {
      cudaDevice = i + 1 < argv.length ? argv[i + 1] : null
    }
    if (a.startsWith('--cuda-device=')) {
      cudaDevice = a.slice('--cuda-device='.length) || null
    }
  }
  return {
    version: typeof system.comfyui_version === 'string' ? system.comfyui_version : null,
    devices,
    argv,
    cudaDevice,
    ramTotalBytes: num(system.ram_total),
    ramFreeBytes: num(system.ram_free)
  }
}

/**
 * Options of a combo input. ComfyUI has two encodings: the old `[["a","b"], {...}]` and the newer
 * `["COMBO", {"options": [...]}]`.
 */
function comboOptions(spec: unknown): string[] {
  if (!Array.isArray(spec) || spec.length === 0) return []
  const first = spec[0]
  if (Array.isArray(first)) return first.filter((x): x is string => typeof x === 'string')
  if (first === 'COMBO') {
    const opts = asRecord(spec[1]).options
    if (Array.isArray(opts)) return opts.filter((x): x is string => typeof x === 'string')
  }
  return []
}

function nodeCombo(info: Record<string, unknown>, node: string, input: string): string[] {
  const inputs = asRecord(asRecord(info[node]).input)
  const spec = asRecord(inputs.required)[input] ?? asRecord(inputs.optional)[input]
  return comboOptions(spec)
}

/** Build ServerCapabilities from GET /object_info. */
export function parseCapabilities(objectInfo: unknown): ServerCapabilities {
  const info = asRecord(objectInfo)
  const nodes: Record<string, boolean> = {}
  for (const n of KNOWN_NODES) nodes[n] = Object.prototype.hasOwnProperty.call(info, n)
  const known = new Set<string>(KNOWN_NODES)
  const blockMergeNodes = Object.keys(info)
    .filter((n) => n.startsWith('ModelMerge') && !known.has(n))
    .sort()
  return {
    checkpoints: nodeCombo(info, 'CheckpointLoaderSimple', 'ckpt_name'),
    diffusionModels: nodeCombo(info, 'UNETLoader', 'unet_name'),
    loras: nodeCombo(info, 'LoraLoader', 'lora_name'),
    textEncoders: nodeCombo(info, 'CLIPLoader', 'clip_name'),
    vaes: nodeCombo(info, 'VAELoader', 'vae_name'),
    nodes,
    blockMergeNodes
  }
}

/** "cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync" -> "nvidia geforce rtx 5060 ti" */
function cardName(device: string): string {
  return device
    .trim()
    .replace(/^[a-z]+:\d+\s+/i, '')
    .replace(/\s+:\s+\S+$/, '')
    .trim()
    .toLowerCase()
}

/**
 * GpuInfo.index of every card whose name appears in one of the server's device strings. Cards that share a name are all
 * returned. A CPU device matches nothing.
 */
export function matchGpuIndexes(devices: string[], gpus: GpuInfo[]): number[] {
  const out = new Set<number>()
  for (const dev of devices) {
    const lower = dev.trim().toLowerCase()
    if (lower.startsWith('cpu')) continue
    const card = cardName(dev)
    const exact = gpus.filter((g) => g.name.trim() && g.name.trim().toLowerCase() === card)
    if (exact.length > 0) {
      for (const g of exact) out.add(g.index)
      continue
    }
    // No exact name: containment, keeping only the longest matching names so "RTX 4060" never beats "RTX 4060 Ti".
    const hits = gpus.filter((g) => g.name.trim() && lower.includes(g.name.trim().toLowerCase()))
    const longest = Math.max(0, ...hits.map((g) => g.name.trim().length))
    for (const g of hits) if (g.name.trim().length === longest) out.add(g.index)
  }
  return [...out].sort((a, b) => a - b)
}
