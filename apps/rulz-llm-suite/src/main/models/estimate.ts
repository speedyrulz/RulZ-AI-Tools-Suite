import { effectiveKvTypes } from '@shared/kv'
import { parseTensorSplit } from '@shared/split'
import type { KvCacheType, LoadConfig, MemoryEstimate, ModelEntry } from '@shared/types'

const MiB = 1024 * 1024

/** Bytes per element for each KV cache type (block-quantized types include their scales). */
export const KV_BYTES: Record<KvCacheType, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q5_1: 24 / 32,
  q5_0: 22 / 32,
  q4_1: 20 / 32,
  q4_0: 18 / 32,
  iq4_nl: 18 / 32
}

export function kvCacheBytes(model: ModelEntry, cfg: LoadConfig, ctx: number): number | null {
  const i = model.info
  if (!i?.blockCount || !i.headCountKv || !i.keyLength || !i.valueLength) return null
  const kv = effectiveKvTypes(cfg)
  const perToken = i.headCountKv * (i.keyLength * KV_BYTES[kv.k] + i.valueLength * KV_BYTES[kv.v])
  const swa = i.slidingWindow && i.slidingWindowFraction ? i.slidingWindowFraction : 0
  const swaCtx = Math.min(ctx, (i.slidingWindow ?? ctx) + (cfg.ubatchSize ?? 512))
  const tokenLayers = i.blockCount * ((1 - swa) * ctx + swa * swaCtx)
  return perToken * tokenLayers
}

export interface GpuFree {
  index: number
  name: string
  freeBytes: number
  totalBytes: number
}

export interface FreeMemory {
  vramBytes: number | null
  ramBytes: number
  /** Every GPU in llama.cpp's device order, for the per-GPU breakdown. */
  gpus?: GpuFree[]
}

/** How much of the split model each GPU gets, as llama.cpp does it: by free memory unless split by hand. */
export function gpuShares(cfg: LoadConfig, gpus: GpuFree[]): number[] {
  if (cfg.splitMode === 'single') return gpus.map((g) => (g.index === cfg.mainGpu ? 1 : 0))
  const manual = cfg.splitMode === 'custom' ? parseTensorSplit(cfg.tensorSplit, gpus.length) : null
  if (manual) return manual
  const free = gpus.reduce((sum, g) => sum + g.freeBytes, 0)
  return gpus.map((g) => (free > 0 ? g.freeBytes / free : 1 / gpus.length))
}

/** Each GPU in use pays this much for its CUDA context and compute buffers. */
const PER_GPU_BYTES = 600 * MiB

/** Rough memory estimate for loading `model` with `cfg`. Errs on the high side. */
export function estimateMemory(model: ModelEntry, cfg: LoadConfig, free: FreeMemory): MemoryEstimate {
  const mmprojBytes = cfg.useMmproj && model.mmproj ? model.mmproj.sizeBytes : 0
  // A projector kept on the CPU (--no-mmproj-offload) costs RAM, not VRAM.
  const mmprojGpuBytes = cfg.mmprojOnCpu ? 0 : mmprojBytes
  const draftBytes = cfg.useDraft && model.draft?.kind ? model.draft.sizeBytes : 0
  const weightsBytes = model.sizeBytes + mmprojBytes + draftBytes
  const ctx = cfg.contextLength
  const kvBytes = ctx ? kvCacheBytes(model, cfg, ctx) : null
  const scaledOverhead = model.sizeBytes * 0.03

  const layers = model.info?.blockCount
  const fraction =
    typeof cfg.gpuLayers === 'number' ? (layers ? Math.min(1, cfg.gpuLayers / (layers + 1)) : 1) : 1
  // The part that is divided between GPUs; the vision projector stays whole on the first one.
  const splitBytes = (model.sizeBytes + (kvBytes ?? 0)) * fraction + scaledOverhead + (cfg.draftGpu === 'cpu' ? 0 : draftBytes)

  const gpus = free.gpus ?? []
  const shares = gpuShares(cfg, gpus)
  const used = gpus.map((g, i) => ({ g, share: shares[i] })).filter((u) => u.share > 0)
  const perGpu = used.map(({ g, share }, i) => ({
    index: g.index,
    name: g.name,
    share,
    bytes: splitBytes * share + PER_GPU_BYTES + (i === 0 ? mmprojGpuBytes : 0),
    freeBytes: g.freeBytes,
    totalBytes: g.totalBytes
  }))

  const overheadBytes = PER_GPU_BYTES * Math.max(1, perGpu.length) + scaledOverhead
  const gpuBytes = perGpu.length ? perGpu.reduce((sum, g) => sum + g.bytes, 0) : splitBytes + PER_GPU_BYTES + mmprojGpuBytes
  const totalBytes = weightsBytes + (kvBytes ?? 0) + overheadBytes
  const freeVramBytes = perGpu.length ? perGpu.reduce((sum, g) => sum + g.freeBytes, 0) : free.vramBytes

  let verdict: MemoryEstimate['verdict'] = 'unknown'
  if (freeVramBytes !== null) {
    const fits = perGpu.length ? perGpu.every((g) => g.bytes <= g.freeBytes * 0.95) : gpuBytes <= freeVramBytes * 0.95
    if (fits) verdict = 'fits'
    else if (cfg.gpuLayers === 'auto' && totalBytes <= freeVramBytes + free.ramBytes * 0.8) verdict = 'partial'
    else verdict = 'tight'
  }

  return {
    weightsBytes,
    kvBytes,
    overheadBytes,
    gpuBytes,
    totalBytes,
    contextLength: ctx,
    freeVramBytes,
    freeRamBytes: free.ramBytes,
    verdict,
    perGpu
  }
}
