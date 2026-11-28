// Inkling 2 loop models (architecture "inkling2_loop"). llama.cpp runs layers [loop.start, loop.end) `loop.count` times and
// keeps one KV cache slot per pass: n_slot = n_phys + (count - 1) * (end - start) (src/models/inkling2-loop.cpp). The GPU
// split divides n_slot layers plus the output layer over the GPUs (src/llama-model.cpp, load_tensors), so a --tensor-split
// that looks like "55,45" puts all 24 physical layers on GPU 0: the slots after them carry no weights, only KV cache.
// This file mirrors that rule, and turns "share of the weights" into the split llama.cpp needs.
import { toPercents } from './split'
import type { GgufInfo, LoadConfig } from './types'

export interface LoopShape {
  /** Physical layers (block_count). */
  nPhys: number
  start: number
  end: number
  count: number
  /** Cache slots: physical layers plus the repeated passes. */
  nSlot: number
}

/** The loop layout of an inkling2_loop model; null for every other model. */
export function loopShape(info: GgufInfo | null | undefined): LoopShape | null {
  if (!info || info.architecture !== 'inkling2_loop' || !info.loop || !info.blockCount) return null
  const { start, end, count } = info.loop
  const nPhys = info.blockCount
  if (!(end > start) || start < 0 || end > nPhys || count < 1) return null
  return { nPhys, start, end, count, nSlot: nPhys + (count - 1) * (end - start) }
}

/** The physical layer whose weights cache slot `il` uses (layer_phys in inkling2-loop.cpp). */
export function layerPhys(shape: LoopShape, il: number): number {
  return il < shape.nPhys ? il : shape.start + ((il - shape.nPhys) % (shape.end - shape.start))
}

/**
 * Where llama.cpp puts each position: 0..nLayers-1 are the layers (cache slots), nLayers is the output layer.
 * Values are indexes into `split` (the GPU list), null = CPU. Mirrors load_tensors: float32 arithmetic, a position goes
 * to the first GPU whose cumulative share is strictly above (il - start) / act.
 */
export function placePositions(nLayers: number, split: number[], gpuLayers: LoadConfig['gpuLayers']): (number | null)[] {
  const f = Math.fround
  const n = split.length
  let sum = 0
  const cumulative = split.map((s) => (sum = f(sum + f(s)))) // eslint-disable-line no-return-assign
  const splits = cumulative.map((c, i) => (sum !== 0 ? f(c / sum) : f((i + 1) / n)))
  const total = nLayers + 1
  const ngl = typeof gpuLayers === 'number' ? gpuLayers : total
  const first = Math.max(total - ngl, 0)
  const act = n === 0 ? 0 : Math.min(ngl, total)
  const out: (number | null)[] = []
  for (let il = 0; il < total; il++) {
    if (il < first || il - first >= act) {
      out.push(null)
      continue
    }
    const x = f(f(il - first) / act)
    const at = splits.findIndex((s) => s > x)
    out.push(at < 0 ? n - 1 : at)
  }
  return out
}

export interface LoopPlacement {
  /** GPU index (position in the split list) holding each physical layer's weights; null = CPU. */
  physGpu: (number | null)[]
  /** GPU holding each cache slot's KV cache (slots 0..nPhys-1 are the physical layers themselves). */
  slotGpu: (number | null)[]
  /** GPU holding the output layer. */
  outputGpu: number | null
}

/** Placement of a loop model for a tensor split (per GPU, any scale) and an n-gpu-layers setting. */
export function placeLoop(shape: LoopShape, split: number[], gpuLayers: LoadConfig['gpuLayers']): LoopPlacement {
  const at = placePositions(shape.nSlot, split, gpuLayers)
  return { physGpu: at.slice(0, shape.nPhys), slotGpu: at.slice(0, shape.nSlot), outputGpu: at[shape.nSlot] }
}

/** Whole-model offload: n-gpu-layers auto/all, or a number that covers every slot and the output layer. */
export function offloadsAll(shape: LoopShape, gpuLayers: LoadConfig['gpuLayers']): boolean {
  return typeof gpuLayers !== 'number' || gpuLayers >= shape.nSlot + 1
}

export interface LoopConversion {
  /** The --tensor-split value. */
  split: string
  /** Physical layers each GPU ends up with. */
  counts: number[]
}

/**
 * The integer tensor split that gives each GPU its share of the physical layers (largest remainder), with every
 * cache slot and the output layer after the last GPU that has a share. Needs n-gpu-layers to cover the whole model.
 * Null when fewer than two GPUs or no positive share.
 */
export function convertLoopSplit(shape: LoopShape, shares: number[]): LoopConversion | null {
  const sum = shares.reduce((a, b) => a + (b > 0 ? b : 0), 0)
  if (shares.length < 2 || !(sum > 0)) return null
  const ideal = shares.map((s) => (s > 0 ? (s / sum) * shape.nPhys : 0))
  const counts = ideal.map((v) => Math.floor(v + 1e-9))
  let left = shape.nPhys - counts.reduce((a, b) => a + b, 0)
  const order = ideal
    .map((v, i) => ({ i, r: v - counts[i] }))
    .filter((e) => shares[e.i] > 0)
    .sort((a, b) => b.r - a.r || a.i - b.i)
  for (let k = 0; left > 0 && order.length; k = (k + 1) % order.length, left--) counts[order[k].i]++
  let last = shares.length - 1
  while (last > 0 && !(shares[last] > 0)) last--
  const total = shape.nSlot + 1
  const entries = counts.map((c, g) => (g < last ? c : 0))
  entries[last] = total - entries.reduce((a, b) => a + b, 0)
  return { split: entries.join(','), counts }
}

export interface LoopSplitPlan extends LoopConversion {
  /** The shares the split stands for, in whole percents per GPU. */
  percents: number[]
}

/**
 * The split to pass for a loop model, or null when the settings are passed as they are (not a loop model, one GPU,
 * a single-GPU setting, a partial n-gpu-layers, or GPUs unknown for an automatic split). A custom split keeps its meaning
 * of "share of the weights"; an automatic one divides by free memory, as the estimate does for other models.
 */
export function planLoopSplit(
  info: GgufInfo | null | undefined,
  cfg: Pick<LoadConfig, 'splitMode' | 'tensorSplit' | 'gpuLayers'>,
  gpus?: { freeBytes: number }[]
): LoopSplitPlan | null {
  const shape = loopShape(info)
  if (!shape || cfg.splitMode === 'single' || !offloadsAll(shape, cfg.gpuLayers)) return null
  let shares: number[] | null = null
  if (cfg.splitMode === 'custom' && cfg.tensorSplit.trim()) {
    const nums = cfg.tensorSplit.split(',').map((p) => Number(p.trim()))
    if (nums.some((n) => !Number.isFinite(n) || n < 0) || (gpus && nums.length > gpus.length)) return null
    shares = nums
  } else if (gpus && gpus.length >= 2) {
    shares = gpus.map((g) => Math.max(0, g.freeBytes))
  }
  if (!shares) return null
  const conv = convertLoopSplit(shape, shares)
  if (!conv) return null
  const sum = shares.reduce((a, b) => a + b, 0)
  return { ...conv, percents: toPercents(shares.map((s) => s / sum)) }
}
