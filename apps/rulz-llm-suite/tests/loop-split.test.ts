import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { convertLoopSplit, layerPhys, loopShape, placeLoop, placePositions, planLoopSplit } from '@shared/loopSplit'
import type { GgufInfo, LoadConfig, ModelEntry } from '@shared/types'
import { buildServerArgs } from '../src/main/engine/args'
import { estimateMemory, kvCacheBytes } from '../src/main/models/estimate'
import { toGgufInfo } from '../src/main/models/gguf'
import { ModelLibrary } from '../src/main/models/library'
import { buildRemoteArgs } from '../src/main/remote/remoteArgs'

const GiB = 2 ** 30

/** Inkling-2-Mini-26B Q5_K_M as read from the file (measured 2026-11-28). */
const kv: Record<string, unknown> = {
  'general.architecture': 'inkling2_loop',
  'inkling2_loop.block_count': 24,
  'inkling2_loop.context_length': 1048576,
  'inkling2_loop.embedding_length': 3072,
  'inkling2_loop.attention.head_count': Array(24).fill(24),
  'inkling2_loop.attention.head_count_kv': Array(24).fill(8),
  'inkling2_loop.attention.key_length': 128,
  'inkling2_loop.attention.value_length': 128,
  'inkling2_loop.attention.sliding_window': 1024,
  'inkling2_loop.attention.sliding_window_pattern': Array.from({ length: 24 }, (_, i) => i % 6 !== 5),
  'inkling2_loop.loop.start': 6,
  'inkling2_loop.loop.end': 18,
  'inkling2_loop.loop.count': 3,
  'inkling2_loop.loop.gate_rank': 64,
  'inkling2_loop.loop.varying': false,
  'inkling2_loop.vocab_size': 201024
}
const info: GgufInfo = toGgufInfo(kv, 'Inkling-2-Mini-26B-Heretic-Q5_K_M.gguf')
const shape = loopShape(info)!

const loopModel: ModelEntry = {
  id: 'ink/ink.gguf',
  name: 'ink',
  folder: 'ink',
  path: 'C:\\LLM\\ink\\ink.gguf',
  files: ['C:\\LLM\\ink\\ink.gguf'],
  sizeBytes: 18.5 * GiB,
  info,
  modifiedAt: 0
}
const llama: ModelEntry = { ...loopModel, id: 'q', info: { architecture: 'llama', blockCount: 32, headCountKv: 8, keyLength: 128, valueLength: 128, hasChatTemplate: true } }

const cfg = (over: Partial<LoadConfig> = {}): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, ...over })
const gpus2 = [
  { index: 0, name: '5060 Ti', freeBytes: 15.6 * GiB, totalBytes: 16 * GiB },
  { index: 1, name: '4060 Ti', freeBytes: 13.4 * GiB, totalBytes: 16 * GiB }
]
const splitOf = (args: string[]): string | undefined => args[args.indexOf('--tensor-split') + 1]

describe('GGUF loop keys', () => {
  it('reads the loop layout, vocabulary and per-layer sliding-window flags', () => {
    expect(info.loop).toEqual({ start: 6, end: 18, count: 3, varying: false })
    expect(info.vocabSize).toBe(201024)
    expect(info.slidingWindowPattern).toHaveLength(24)
    expect(info.slidingWindowPattern?.filter((x) => !x)).toHaveLength(4)
    expect(shape).toEqual({ nPhys: 24, start: 6, end: 18, count: 3, nSlot: 48 })
    expect(info.headCount).toBe(24)
  })

  it('leaves other architectures without a loop', () => {
    expect(toGgufInfo({ 'general.architecture': 'llama', 'llama.block_count': 32 }, 'x.gguf').loop).toBeUndefined()
    expect(loopShape({ ...info, architecture: 'inkling2' })).toBeNull()
    expect(loopShape({ ...info, loop: undefined })).toBeNull()
  })

  it('maps slots onto the physical layers of the loop core', () => {
    expect(layerPhys(shape, 3)).toBe(3)
    expect(layerPhys(shape, 24)).toBe(6)
    expect(layerPhys(shape, 35)).toBe(17)
    expect(layerPhys(shape, 36)).toBe(6)
    expect(layerPhys(shape, 47)).toBe(17)
  })
})

describe('placement mirror of llama.cpp', () => {
  it('55,45 puts every physical layer on GPU 0 (the measured out-of-memory)', () => {
    const p = placeLoop(shape, [55, 45], 'auto')
    expect(p.physGpu.every((g) => g === 0)).toBe(true)
    expect(p.slotGpu.slice(24).filter((g) => g === 0)).toHaveLength(3)
    expect(p.outputGpu).toBe(1)
  })

  it('27,73 gives 14 physical layers to GPU 0 and 10 to GPU 1, output on the last GPU', () => {
    const p = placeLoop(shape, [27, 73], 'all')
    expect(p.physGpu.filter((g) => g === 0)).toHaveLength(14)
    expect(p.physGpu.filter((g) => g === 1)).toHaveLength(10)
    expect(p.outputGpu).toBe(1)
  })

  it('keeps the first layers on the CPU when n-gpu-layers is below the layer count', () => {
    const at = placePositions(48, [1], 10)
    expect(at.slice(0, 39).every((g) => g === null)).toBe(true)
    expect(at.slice(39).every((g) => g === 0)).toBe(true)
  })

  it('an all-zero split falls back to equal shares and no GPUs means CPU', () => {
    expect(placePositions(3, [0, 0], 'auto')).toEqual([0, 0, 1, 1])
    expect(placePositions(3, [], 'auto')).toEqual([null, null, null, null])
  })
})

describe('convertLoopSplit', () => {
  it('55/45 on the real model is 13,36', () => {
    expect(convertLoopSplit(shape, [55, 45])).toEqual({ split: '13,36', counts: [13, 11] })
  })

  it('round-trips through the placement for many shares and GPU counts', () => {
    for (const shares of [[55, 45], [1, 1], [1, 2, 1], [5, 3, 2], [1, 0, 1], [0, 1], [1, 0], [3, 1, 1, 1], [0.2, 0.3, 0.5]]) {
      const conv = convertLoopSplit(shape, shares)!
      const place = placeLoop(shape, conv.split.split(',').map(Number), 'all')
      const got = shares.map((_, g) => place.physGpu.filter((x) => x === g).length)
      expect(got, shares.join('/')).toEqual(conv.counts)
      expect(conv.counts.reduce((a, b) => a + b, 0)).toBe(24)
      shares.forEach((s, g) => s === 0 && expect(got[g]).toBe(0))
      // Every GPU gets a number of layers within one of its exact share.
      const sum = shares.reduce((a, b) => a + b, 0)
      shares.forEach((s, g) => expect(Math.abs(got[g] - (s / sum) * 24)).toBeLessThan(1))
    }
  })

  it('a zero share gets nothing and the cache slots go to the last GPU with a share', () => {
    const conv = convertLoopSplit(shape, [1, 0, 1])!
    expect(conv.split).toBe('12,0,37')
    expect(convertLoopSplit(shape, [1, 0])!.split).toBe('49,0')
    const p = placeLoop(shape, [49, 0], 'all')
    expect(p.slotGpu.every((g) => g === 0)).toBe(true)
  })

  it('needs two GPUs and a positive share', () => {
    expect(convertLoopSplit(shape, [1])).toBeNull()
    expect(convertLoopSplit(shape, [0, 0])).toBeNull()
  })
})

describe('planLoopSplit', () => {
  const free = [{ freeBytes: 15.6 * GiB }, { freeBytes: 13.4 * GiB }]
  it('converts a custom split and an automatic one from free memory', () => {
    expect(planLoopSplit(info, cfg({ splitMode: 'custom', tensorSplit: '55,45' }))?.split).toBe('13,36')
    expect(planLoopSplit(info, cfg({ splitMode: 'custom', tensorSplit: '55,45' }), free)?.percents).toEqual([55, 45])
    expect(planLoopSplit(info, cfg(), free)?.split).toBe('13,36')
  })

  it('does nothing without two known GPUs, for single GPU, other models and partial offload', () => {
    expect(planLoopSplit(info, cfg())).toBeNull()
    expect(planLoopSplit(info, cfg(), [free[0]])).toBeNull()
    expect(planLoopSplit(info, cfg({ splitMode: 'single' }), free)).toBeNull()
    expect(planLoopSplit(llama.info, cfg({ splitMode: 'custom', tensorSplit: '1,1' }), free)).toBeNull()
    expect(planLoopSplit(info, cfg({ gpuLayers: 30 }), free)).toBeNull()
    expect(planLoopSplit(info, cfg({ splitMode: 'custom', tensorSplit: '55,45', gpuLayers: 30 }))).toBeNull()
    expect(planLoopSplit(info, cfg({ splitMode: 'custom', tensorSplit: '1,1,1' }), free)).toBeNull()
  })

  it('a numeric n-gpu-layers that covers every slot and the output still converts', () => {
    expect(planLoopSplit(info, cfg({ gpuLayers: 49, splitMode: 'custom', tensorSplit: '55,45' }))?.split).toBe('13,36')
    expect(planLoopSplit(info, cfg({ gpuLayers: 48, splitMode: 'custom', tensorSplit: '55,45' }))).toBeNull()
  })
})

describe('buildServerArgs for loop models', () => {
  const base = { model: loopModel, port: 1, alias: 'a' }
  const gpus = gpus2.map(({ index, freeBytes }) => ({ index, freeBytes }))

  it('converts a custom split', () => {
    const args = buildServerArgs({ ...base, config: cfg({ splitMode: 'custom', tensorSplit: '55, 45' }) })
    expect(args.slice(args.indexOf('--split-mode'), args.indexOf('--split-mode') + 4)).toEqual(['--split-mode', 'layer', '--tensor-split', '13,36'])
  })

  it('passes a split worked out from free memory for an automatic one on two GPUs', () => {
    const args = buildServerArgs({ ...base, config: cfg(), gpus })
    expect(args).toContain('--split-mode')
    expect(splitOf(args)).toBe('13,36')
  })

  it('passes nothing for an automatic split with unknown or one GPU, as before', () => {
    expect(buildServerArgs({ ...base, config: cfg() })).not.toContain('--tensor-split')
    expect(buildServerArgs({ ...base, config: cfg(), gpus: [gpus[0]] })).not.toContain('--tensor-split')
  })

  it('leaves a partial n-gpu-layers split, single GPU and other models as they were', () => {
    expect(splitOf(buildServerArgs({ ...base, config: cfg({ splitMode: 'custom', tensorSplit: '55,45', gpuLayers: 30 }) }))).toBe('55,45')
    const single = buildServerArgs({ ...base, config: cfg({ splitMode: 'single', mainGpu: 1 }), gpus })
    expect(single).toContain('--main-gpu')
    expect(single).not.toContain('--tensor-split')
    expect(splitOf(buildServerArgs({ model: llama, port: 1, alias: 'a', config: cfg({ splitMode: 'custom', tensorSplit: '55,45' }), gpus }))).toBe('55,45')
    expect(buildServerArgs({ model: llama, port: 1, alias: 'a', config: cfg(), gpus })).not.toContain('--tensor-split')
  })

  it('a rental converts too, from the GPUs it reports', () => {
    const paths = { main: '/w/ink.gguf', adapters: [] }
    expect(splitOf(buildRemoteArgs({ model: loopModel, config: cfg(), port: 'P', alias: 'a', paths, gpus }))).toBe('13,36')
    expect(splitOf(buildRemoteArgs({ model: loopModel, config: cfg({ splitMode: 'custom', tensorSplit: '55,45' }), port: 'P', alias: 'a', paths }))).toBe('13,36')
    expect(buildRemoteArgs({ model: loopModel, config: cfg(), port: 'P', alias: 'a', paths })).not.toContain('--tensor-split')
  })
})

describe('estimateMemory for loop models', () => {
  const free = { vramBytes: 29 * GiB, ramBytes: 32 * GiB, gpus: gpus2 }
  const c = cfg({ contextLength: 65536, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' })

  it('counts a KV cache for every slot, not only the physical layers', () => {
    const slots = kvCacheBytes(loopModel, c, 65536)!
    const flat = kvCacheBytes({ ...loopModel, info: { ...info, loop: undefined } }, c, 65536)!
    expect(slots / flat).toBeGreaterThan(1.8)
    expect(slots / flat).toBeLessThan(2.2)
  })

  it('shows the converted split and puts the weights where llama.cpp puts them', () => {
    const e = estimateMemory(loopModel, { ...c, splitMode: 'custom', tensorSplit: '55,45', ubatchSize: 32 }, free)
    expect(e.loopSplit).toEqual({ passed: '13,36', percents: [55, 45] })
    // 13 of the 24 physical layers on GPU0, 11 plus the output layer on GPU1 (one unit = size / 25).
    const unit = (18.5 * GiB) / 25
    expect(e.perGpu[0].bytes).toBeGreaterThan(13 * unit)
    expect(e.perGpu[0].bytes).toBeLessThan(13 * unit + 2.5 * GiB)
    expect(e.perGpu[1].bytes).toBeGreaterThan(12 * unit)
    // A partial offload is passed as it is; llama.cpp then puts all 15 offloaded physical layers on GPU0.
    const raw = estimateMemory(loopModel, { ...c, splitMode: 'custom', tensorSplit: '55,45', ubatchSize: 32, gpuLayers: 40 }, free)
    expect(raw.loopSplit).toBeUndefined()
    expect(raw.perGpu[0].bytes).toBeGreaterThan(raw.perGpu[1].bytes + 8 * GiB)
  })

  it('adds the prompt scratch buffer (about 6.4 GB at 65536 x 512) and suggests a smaller micro-batch', () => {
    const e = estimateMemory(loopModel, c, free)
    expect(e.scratch).toBeDefined()
    expect(e.scratch!.ubatch).toBe(512)
    // 2 * 24 heads * 65536 * 512 * 4 = 6 GiB on every GPU with a full-attention slot, plus the logits on the last one.
    const per = e.perGpu.map((g) => g.bytes)
    const small = estimateMemory(loopModel, { ...c, ubatchSize: 32 }, free)
    // (the sliding-window layers' caches also grow a little with the micro-batch, hence the tolerance)
    const attn = 2 * 24 * 65536 * (512 - 32) * 4
    expect(Math.abs(per[0] - small.perGpu[0].bytes - attn)).toBeLessThan(0.01 * attn)
    expect(Math.abs(per[1] - small.perGpu[1].bytes - attn - 201024 * (512 - 32) * 4)).toBeLessThan(0.01 * attn)
    expect(e.scratch!.overflowGpu).not.toBeNull()
    expect(e.scratch!.suggestedUbatch).toBe(128)
    expect(e.verdict).not.toBe('fits')
    const withSuggested = estimateMemory(loopModel, { ...c, ubatchSize: 128 }, free)
    expect(withSuggested.verdict).toBe('fits')
    expect(withSuggested.scratch!.overflowGpu).toBeNull()
    expect(withSuggested.scratch!.suggestedUbatch).toBeNull()
  })

  it('also adds the scratch buffer for the plain inkling2 architecture, and not for other models', () => {
    const plain: ModelEntry = { ...loopModel, info: { ...info, architecture: 'inkling2', loop: undefined } }
    const e = estimateMemory(plain, c, free)
    expect(e.scratch?.bytes).toBeGreaterThan(5 * GiB)
    expect(estimateMemory(llama, c, free).scratch).toBeUndefined()
  })

  it('does not change the numbers of other models', () => {
    const e = estimateMemory(llama, cfg({ contextLength: 4096 }), free)
    expect(e.scratch).toBeUndefined()
    expect(e.loopSplit).toBeUndefined()
  })

  it('an unset context counts the full trained context only when a converted split turns --fit off', () => {
    // Converted split: llama.cpp gets --tensor-split, --fit is off, so the model's 1,048,576 tokens are used for both.
    const forced = estimateMemory(loopModel, cfg({ contextLength: null, splitMode: 'custom', tensorSplit: '55,45' }), free)
    expect(forced.contextLength).toBe(1048576)
    expect(forced.kvBytes).toBeGreaterThan(0)
    expect(forced.scratch?.bytes).toBeGreaterThan(50 * GiB)
    expect(forced.verdict).not.toBe('fits')
    // Single GPU: nothing is passed, --fit may still pick the context, so neither is guessed.
    const fit = estimateMemory(loopModel, cfg({ contextLength: null, splitMode: 'single', mainGpu: 0 }), free)
    expect(fit.contextLength).toBeNull()
    expect(fit.kvBytes).toBeNull()
    expect(fit.scratch).toBeUndefined()
  })

  it('places a partial offload with the raw custom split values, as llama.cpp sums them', () => {
    const e = estimateMemory(loopModel, { ...c, splitMode: 'custom', tensorSplit: '1,2', ubatchSize: 32, gpuLayers: 40 }, free)
    const f = estimateMemory(loopModel, { ...c, splitMode: 'custom', tensorSplit: '10,20', ubatchSize: 32, gpuLayers: 40 }, free)
    expect(e.loopSplit).toBeUndefined()
    expect(e.perGpu.map((g) => Math.round(g.bytes))).toEqual(f.perGpu.map((g) => Math.round(g.bytes)))
  })

  it('without GPU information the scratch buffer is added to the one lump', () => {
    const a = estimateMemory(loopModel, { ...c, ubatchSize: 32 }, { vramBytes: null, ramBytes: 32 * GiB })
    const b = estimateMemory(loopModel, c, { vramBytes: null, ramBytes: 32 * GiB })
    expect(b.gpuBytes - a.gpuBytes).toBeGreaterThan(5.5 * GiB)
    expect(b.perGpu).toEqual([])
  })
})

describe('gguf cache', () => {
  it('re-reads entries cached before the loop fields existed', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-loop-'))
    const root = join(dir, 'models')
    const file = join(root, 'a.gguf')
    const { mkdirSync } = await import('node:fs')
    mkdirSync(root)
    // A GGUF header with one key: the architecture.
    const str = (s: string): Buffer => {
      const b = Buffer.from(s)
      const len = Buffer.alloc(8)
      len.writeBigUInt64LE(BigInt(b.length))
      return Buffer.concat([len, b])
    }
    const u32 = (n: number): Buffer => {
      const b = Buffer.alloc(4)
      b.writeUInt32LE(n)
      return b
    }
    const u64 = (n: number): Buffer => {
      const b = Buffer.alloc(8)
      b.writeBigUInt64LE(BigInt(n))
      return b
    }
    const keys: [string, number, Buffer][] = [
      ['general.architecture', 8, str('inkling2_loop')],
      ['inkling2_loop.block_count', 4, u32(24)],
      ['inkling2_loop.loop.start', 4, u32(6)],
      ['inkling2_loop.loop.end', 4, u32(18)],
      ['inkling2_loop.loop.count', 4, u32(3)]
    ]
    const header = Buffer.concat([u32(0x46554747), u32(3), u64(0), u64(keys.length), ...keys.flatMap(([k, t, v]) => [str(k), u32(t), v])])
    writeFileSync(file, header)
    const { statSync } = await import('node:fs')
    const st = statSync(file)
    const cache = join(dir, 'cache.json')
    // The entry an older version wrote: same file, no loop fields.
    writeFileSync(cache, JSON.stringify({ [file]: { v: 5, size: st.size, mtimeMs: st.mtimeMs, info: { architecture: 'inkling2_loop', blockCount: 24, hasChatTemplate: false } } }))
    const lib = new ModelLibrary(cache, () => root, () => undefined)
    await lib.scan()
    const got = lib.list().find((m) => m.path === file)
    expect(got?.info?.loop).toEqual({ start: 6, end: 18, count: 3, varying: false })
  })
})
