import type { KvCacheType, LoadConfig } from './types'

export interface KvTypes {
  k: KvCacheType
  v: KvCacheType
  draftK: KvCacheType
  draftV: KvCacheType
}

/** True for the block-quantized KV cache types (everything but the float types). */
export function isQuantizedKv(t: KvCacheType): boolean {
  return t !== 'f16' && t !== 'bf16' && t !== 'f32'
}

/**
 * The KV cache types llama-server actually gets. A quantized V cache needs flash attention
 * (llama.cpp refuses to load otherwise), so with flash attention off V falls back to f16.
 * K may stay quantized without flash attention.
 */
export function effectiveKvTypes(c: LoadConfig): KvTypes {
  const faOff = c.flashAttention === 'off'
  const v = (t: KvCacheType): KvCacheType => (faOff && isQuantizedKv(t) ? 'f16' : t)
  return { k: c.cacheTypeK, v: v(c.cacheTypeV), draftK: c.draftCacheTypeK, draftV: v(c.draftCacheTypeV) }
}
