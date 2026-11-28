import { effectiveKvTypes } from '@shared/kv'
import { layerPhys, loopShape, placeLoop, placePositions, planLoopSplit } from '@shared/loopSplit'
import { parseTensorSplit } from '@shared/split'
import type { GgufInfo, KvCacheType, LoadConfig, MemoryEstimate, ModelEntry, ScratchEstimate } from '@shared/types'

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
  const shape = loopShape(i)
  if (shape) {
    // A loop model keeps one cache per slot (every pass of the loop), not per physical layer.
    let tokens = 0
    for (let s = 0; s < shape.nSlot; s++) tokens += layerContext(i, cfg, ctx, layerPhys(shape, s))
    return perToken * tokens
  }
  const swa = i.slidingWindow && i.slidingWindowFraction ? i.slidingWindowFraction : 0
  const swaCtx = Math.min(ctx, (i.slidingWindow ?? ctx) + (cfg.ubatchSize ?? 512))
  const tokenLayers = i.blockCount * ((1 - swa) * ctx + swa * swaCtx)
  return perToken * tokenLayers
}

/** Tokens the KV cache of physical layer `phys` holds: the window for sliding-window layers, else the context. */
function layerContext(i: GgufInfo, cfg: LoadConfig, ctx: number, phys: number): number {
  const swaCtx = Math.min(ctx, (i.slidingWindow ?? ctx) + (cfg.ubatchSize ?? 512))
  if (!i.slidingWindow) return ctx
  if (i.slidingWindowPattern) return i.slidingWindowPattern[phys] ? swaCtx : ctx
  const f = i.slidingWindowFraction ?? 0
  return (1 - f) * ctx + f * swaCtx
}

/** Micro-batch sizes offered when the prompt scratch buffer does not fit, largest first. */
const UBATCH_STEPS = [512, 256, 128, 64, 32]

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

/** Expert weights that --cpu-moe / --n-cpu-moe keep in system RAM; 0 when unknown or off. */
export function cpuExpertBytes(model: ModelEntry, cfg: LoadConfig): number {
  const layers = model.info?.expertBytesByLayer
  if (!layers || cfg.cpuMoe === 'off') return 0
  const upTo = cfg.cpuMoe === 'all' ? layers.length : Math.min(cfg.cpuMoe, layers.length)
  return layers.slice(0, upTo).reduce((sum, b) => sum + b, 0)
}

/**
 * Inkling adds a relative-position bias to the attention scores: an n_head x n_kv x n_tokens f32 matrix, twice (the bias
 * and the sum), per micro-batch (src/models/inkling2.cpp). Full-attention layers have n_kv = the context. The logits
 * (vocabulary x micro-batch f32) sit on the GPU of the output layer.
 */
function scratchBytes(nHead: number, ctx: number, vocab: number, ubatch: number, full: boolean, output: boolean): number {
  return (full ? 2 * nHead * ctx * ubatch * 4 : 0) + (output ? vocab * ubatch * 4 : 0)
}

/** Each GPU in use pays this much for its CUDA context and compute buffers. */
const PER_GPU_BYTES = 600 * MiB

/**
 * Rough memory estimate for loading `model` with `cfg`. Errs on the high side. `adapterBytes` is the
 * size of the LoRA adapter files the load would add; they follow the model's weights onto the GPU.
 */
export function estimateMemory(model: ModelEntry, cfg: LoadConfig, free: FreeMemory, adapterBytes = 0): MemoryEstimate {
  const mmprojBytes = cfg.useMmproj && model.mmproj ? model.mmproj.sizeBytes : 0
  // A projector kept on the CPU (--no-mmproj-offload) costs RAM, not VRAM.
  const mmprojGpuBytes = cfg.mmprojOnCpu ? 0 : mmprojBytes
  const draftBytes = cfg.useDraft && model.draft?.kind ? model.draft.sizeBytes : 0
  const weightsBytes = model.sizeBytes + mmprojBytes + draftBytes
  // Inkling loop models get their split converted (shared/loopSplit.ts). Passing a tensor split turns llama.cpp's --fit
  // off, so an unset context then means the model's full trained context, for the KV cache and the scratch buffer alike.
  const plan = planLoopSplit(model.info, cfg, free.gpus)
  const ctx = cfg.contextLength ?? (plan ? (model.info?.contextLength ?? null) : null)
  const kvBytes = ctx ? kvCacheBytes(model, cfg, ctx) : null
  const scaledOverhead = model.sizeBytes * 0.03
  const expertCpuBytes = Math.min(cpuExpertBytes(model, cfg), model.sizeBytes)

  const layers = model.info?.blockCount
  const fraction =
    typeof cfg.gpuLayers === 'number' ? (layers ? Math.min(1, cfg.gpuLayers / (layers + 1)) : 1) : 1
  // The part that is divided between GPUs; the vision projector stays whole on the first one.
  // Experts kept on the CPU leave the GPU share; the rest of each layer still follows gpuLayers.
  const splitBytes = (model.sizeBytes - expertCpuBytes + adapterBytes + (kvBytes ?? 0)) * fraction + scaledOverhead + (cfg.draftGpu === 'cpu' ? 0 : draftBytes)

  const gpus = free.gpus ?? []
  const shares = gpuShares(cfg, gpus)
  const used = gpus.map((g, i) => ({ g, share: shares[i] })).filter((u) => u.share > 0)

  // Inkling: where llama.cpp puts the layers, cache slots and output layer for the split that will really be passed.
  const info = model.info
  const shape = loopShape(info)
  const inkling = info?.architecture === 'inkling2' || info?.architecture === 'inkling2_loop'
  // llama.cpp sums the raw split values in float32 (llama-model.cpp load_tensors), so the placement gets those, not shares.
  const customRaw = cfg.splitMode === 'custom' && parseTensorSplit(cfg.tensorSplit, gpus.length) ? cfg.tensorSplit.split(',').map((p) => Number(p.trim())) : null
  const rawSplit = plan
    ? gpus.map((_, i) => Number(plan.split.split(',')[i] ?? 0))
    : customRaw
      ? gpus.map((_, i) => customRaw[i] ?? 0)
      : cfg.splitMode === 'single'
        ? shares
        : gpus.map((g) => g.freeBytes)
  const place = shape && gpus.length ? placeLoop(shape, rawSplit, cfg.gpuLayers) : null
  let slotGpu: (number | null)[] | null = place?.slotGpu ?? null
  let outputGpu: number | null = place?.outputGpu ?? null
  if (!shape && inkling && gpus.length && info?.blockCount) {
    const at = placePositions(info.blockCount, rawSplit, cfg.gpuLayers)
    slotGpu = at.slice(0, info.blockCount)
    outputGpu = at[info.blockCount]
  }
  const slotFull = (s: number): boolean => {
    const phys = shape ? layerPhys(shape, s) : s
    return !(info?.slidingWindow && info.slidingWindowPattern?.[phys])
  }

  // Bytes of the model's weights, KV cache and extras per GPU for a loop model, from the placement above.
  let loopBytes: number[] | null = null
  if (shape && info && place) {
    const w = gpus.map(() => 0)
    const kv = gpus.map(() => 0)
    const unit = model.sizeBytes / (shape.nPhys + 1)
    const expertLayers = info.expertBytesByLayer
    const upTo = !expertLayers || cfg.cpuMoe === 'off' ? 0 : cfg.cpuMoe === 'all' ? expertLayers.length : Math.min(cfg.cpuMoe, expertLayers.length)
    place.physGpu.forEach((g, p) => {
      if (g !== null) w[g] += Math.max(0, unit - (p < upTo ? (expertLayers?.[p] ?? 0) : 0))
    })
    if (place.outputGpu !== null) w[place.outputGpu] += unit
    if (ctx && info.headCountKv && info.keyLength && info.valueLength) {
      const k = effectiveKvTypes(cfg)
      const perToken = info.headCountKv * (info.keyLength * KV_BYTES[k.k] + info.valueLength * KV_BYTES[k.v])
      place.slotGpu.forEach((g, s) => {
        if (g !== null) kv[g] += perToken * layerContext(info, cfg, ctx, layerPhys(shape, s))
      })
    }
    const extras = adapterBytes + scaledOverhead + (cfg.draftGpu === 'cpu' ? 0 : draftBytes)
    const wTotal = w.reduce((a, b) => a + b, 0)
    const first = Math.max(0, shares.findIndex((x) => x > 0))
    loopBytes = gpus.map((_, g) => w[g] + kv[g] + (wTotal > 0 ? extras * (w[g] / wTotal) : g === first ? extras : 0))
  }

  // The prompt scratch buffer of Inkling models, per GPU (one lump when the GPUs are unknown).
  const nHead = info?.headCount
  // Unset context: llama.cpp's --fit may still shrink it, unless a tensor split is passed (then fit is off and the
  // model's full context is used), so only then is the model's own context length counted.
  const scratchCtx = ctx
  const ubatch = cfg.ubatchSize ?? 512
  const vocab = info?.vocabSize ?? 0
  const canScratch = inkling && !!nHead && !!scratchCtx
  const scratchOn = (g: number, u: number): number => {
    if (!canScratch || !slotGpu) return 0
    // A repeated loop slot reads the weights of its physical layer p, but slot p itself (same weights, same GPU) is
    // counted here too, so every GPU that runs a full-attention layer is covered.
    const full = slotGpu.some((x, s) => x === g && slotFull(s))
    return scratchBytes(nHead, scratchCtx, vocab, u, full, outputGpu === g)
  }
  const scratchWhole = canScratch && !gpus.length ? scratchBytes(nHead, scratchCtx, vocab, ubatch, true, true) : 0

  const perGpu = used.map(({ g, share }, i) => {
    const at = gpus.indexOf(g)
    return {
      index: g.index,
      name: g.name,
      share,
      bytes: (loopBytes ? loopBytes[at] : splitBytes * share) + PER_GPU_BYTES + (i === 0 ? mmprojGpuBytes : 0) + scratchOn(at, ubatch),
      freeBytes: g.freeBytes,
      totalBytes: g.totalBytes
    }
  })

  const overheadBytes = PER_GPU_BYTES * Math.max(1, perGpu.length) + scaledOverhead
  const gpuBytes = perGpu.length ? perGpu.reduce((sum, g) => sum + g.bytes, 0) : splitBytes + PER_GPU_BYTES + mmprojGpuBytes + scratchWhole
  const scratchTotal = perGpu.length ? used.reduce((sum, u) => sum + scratchOn(gpus.indexOf(u.g), ubatch), 0) : scratchWhole
  const totalBytes = weightsBytes + adapterBytes + (kvBytes ?? 0) + overheadBytes + scratchTotal
  const freeVramBytes = perGpu.length ? perGpu.reduce((sum, g) => sum + g.freeBytes, 0) : free.vramBytes

  let verdict: MemoryEstimate['verdict'] = 'unknown'
  if (freeVramBytes !== null) {
    const fits = perGpu.length ? perGpu.every((g) => g.bytes <= g.freeBytes * 0.95) : gpuBytes <= freeVramBytes * 0.95
    // CPU-held experts must fit in RAM too (mmap can page them from disk, but slowly).
    const ramOk = expertCpuBytes <= free.ramBytes * 0.8
    if (!ramOk) verdict = 'tight'
    else if (fits) verdict = 'fits'
    else if (cfg.gpuLayers === 'auto' && totalBytes <= freeVramBytes + free.ramBytes * 0.8) verdict = 'partial'
    else verdict = 'tight'
  }

  // Which GPU's scratch buffer is what does not fit, and the micro-batch that would make everything fit.
  let scratch: ScratchEstimate | undefined
  if (canScratch) {
    const limit = (g: { freeBytes: number }): number => g.freeBytes * 0.95
    const per = used.map((u) => scratchOn(gpus.indexOf(u.g), ubatch))
    const over = perGpu.findIndex((g, i) => per[i] > 0 && g.bytes > limit(g) && g.bytes - per[i] <= limit(g))
    const fits = (u: number): boolean => perGpu.every((g, i) => g.bytes - per[i] + (per[i] * u) / ubatch <= limit(g))
    const suggested = over >= 0 ? (UBATCH_STEPS.find((u) => u < ubatch && fits(u)) ?? null) : null
    scratch = {
      bytes: over >= 0 ? per[over] : perGpu.length ? Math.max(0, ...per) : scratchWhole,
      ubatch,
      overflowGpu: over >= 0 ? perGpu[over].index : null,
      suggestedUbatch: suggested
    }
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
    perGpu,
    cpuExpertBytes: expertCpuBytes,
    adapterBytes,
    ...(scratch ? { scratch } : {}),
    ...(plan ? { loopSplit: { passed: plan.split, percents: plan.percents } } : {})
  }
}
