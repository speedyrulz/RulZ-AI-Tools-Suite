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
  const perToken = i.headCountKv * (i.keyLength * KV_BYTES[cfg.cacheTypeK] + i.valueLength * KV_BYTES[cfg.cacheTypeV])
  const swa = i.slidingWindow && i.slidingWindowFraction ? i.slidingWindowFraction : 0
  const swaCtx = Math.min(ctx, (i.slidingWindow ?? ctx) + (cfg.ubatchSize ?? 512))
  const tokenLayers = i.blockCount * ((1 - swa) * ctx + swa * swaCtx)
  return perToken * tokenLayers
}

export interface FreeMemory {
  vramBytes: number | null
  ramBytes: number
}

/** Rough memory estimate for loading `model` with `cfg`. Errs on the high side. */
export function estimateMemory(model: ModelEntry, cfg: LoadConfig, free: FreeMemory): MemoryEstimate {
  const projector = cfg.useMmproj && model.mmproj ? model.mmproj.sizeBytes : 0
  const weightsBytes = model.sizeBytes + projector
  const ctx = cfg.contextLength
  const kvBytes = ctx ? kvCacheBytes(model, cfg, ctx) : null
  const overheadBytes = 600 * MiB + model.sizeBytes * 0.03

  const layers = model.info?.blockCount
  const fraction =
    typeof cfg.gpuLayers === 'number' ? (layers ? Math.min(1, cfg.gpuLayers / (layers + 1)) : 1) : 1
  const gpuBytes = model.sizeBytes * fraction + projector + (kvBytes ?? 0) * fraction + overheadBytes
  const totalBytes = weightsBytes + (kvBytes ?? 0) + overheadBytes

  let verdict: MemoryEstimate['verdict'] = 'unknown'
  if (free.vramBytes !== null) {
    if (gpuBytes <= free.vramBytes * 0.95) verdict = 'fits'
    else if (cfg.gpuLayers === 'auto' && totalBytes <= free.vramBytes + free.ramBytes * 0.8) verdict = 'partial'
    else verdict = 'tight'
  }

  return {
    weightsBytes,
    kvBytes,
    overheadBytes,
    gpuBytes,
    totalBytes,
    contextLength: ctx,
    freeVramBytes: free.vramBytes,
    freeRamBytes: free.ramBytes,
    verdict
  }
}
