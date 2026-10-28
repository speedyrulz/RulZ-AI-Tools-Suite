import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { ModelEntry } from '@shared/types'
import { buildServerArgs } from '../src/main/engine/args'
import { parseDevices, parseVersion } from '../src/main/engine/engine'
import { parseNvidiaSmi } from '../src/main/engine/hardware'
import { describeExitCode, splitArgs } from '../src/main/engine/process'
import { LogLineSplitter, nextProgress, progressFromDots } from '../src/main/engine/progress'
import { estimateMemory, gpuShares, kvCacheBytes } from '../src/main/models/estimate'
import { parseTensorSplit, rebalanceSplit } from '../src/shared/split'

const model: ModelEntry = {
  id: 'Qwen/q.gguf',
  name: 'q',
  folder: 'Qwen',
  path: 'C:\\LLM\\Qwen\\q.gguf',
  files: ['C:\\LLM\\Qwen\\q.gguf'],
  sizeBytes: 9e9,
  mmproj: { path: 'C:\\LLM\\Qwen\\mmproj-BF16.gguf', sizeBytes: 0.9e9 },
  info: { architecture: 'llama', blockCount: 32, headCountKv: 8, keyLength: 128, valueLength: 128, hasChatTemplate: true },
  modifiedAt: 0
}

describe('buildServerArgs', () => {
  it('leaves auto values to llama.cpp and always binds to localhost', () => {
    const args = buildServerArgs({ model, config: { ...DEFAULT_LOAD_CONFIG, contextLength: null }, port: 5555, alias: 'q' })
    expect(args).toContain('--no-webui')
    expect(args.slice(args.indexOf('--host'), args.indexOf('--host') + 2)).toEqual(['--host', '127.0.0.1'])
    expect(args).not.toContain('--ctx-size')
    expect(args).not.toContain('--n-gpu-layers')
    expect(args).toContain('--mmproj')
  })

  it('maps explicit settings to flags', () => {
    const args = buildServerArgs({
      model,
      config: {
        ...DEFAULT_LOAD_CONFIG,
        contextLength: 16384,
        gpuLayers: 20,
        splitMode: 'single',
        mainGpu: 1,
        flashAttention: 'on',
        cacheTypeK: 'q8_0',
        useMmap: false,
        useMmproj: false,
        extraArgs: '--override-tensor "exps=CPU" -v'
      },
      port: 1,
      alias: 'q'
    }).join(' ')
    expect(args).toContain('--ctx-size 16384')
    expect(args).toContain('--n-gpu-layers 20')
    expect(args).toContain('--split-mode none --main-gpu 1')
    expect(args).toContain('--flash-attn on')
    expect(args).toContain('--cache-type-k q8_0')
    expect(args).not.toContain('--cache-type-v')
    expect(args).toContain('--no-mmap')
    expect(args).not.toContain('--mmproj')
    expect(args.endsWith('--override-tensor exps=CPU -v')).toBe(true)
  })
})

describe('describeExitCode', () => {
  it('explains a missing DLL whether the code is signed or unsigned', () => {
    expect(describeExitCode(3221225781)).toMatch(/DLL .* could not be found/)
    expect(describeExitCode(-1073741515)).toMatch(/0xC0000135/)
    expect(describeExitCode(1)).toBeUndefined()
  })
})

describe('draft head arguments', () => {
  const withDraft: ModelEntry = { ...model, draft: { path: 'C:\\LLM\\Qwen\\mtp-q.gguf', sizeBytes: 2e8, kind: 'mtp' } }
  it('passes the sidecar as an MTP draft when enabled', () => {
    const args = buildServerArgs({ model: withDraft, config: { ...DEFAULT_LOAD_CONFIG, draftTokens: 4 }, port: 1, alias: 'q' }).join(' ')
    expect(args).toContain('--spec-type draft-mtp --spec-draft-model C:\\LLM\\Qwen\\mtp-q.gguf')
    expect(args).toContain('--spec-draft-n-max 4')
  })
  it('maps the MTP tuning settings to flags and leaves defaults out', () => {
    const tuned = buildServerArgs({
      model: withDraft,
      config: { ...DEFAULT_LOAD_CONFIG, draftMinTokens: 1, draftPMin: 0.6, draftSampling: 'greedy', draftGpu: 'cpu', draftCacheTypeK: 'q8_0' },
      port: 1,
      alias: 'q'
    }).join(' ')
    expect(tuned).toContain('--spec-draft-n-min 1')
    expect(tuned).toContain('--spec-draft-p-min 0.6')
    expect(tuned).toContain('--spec-draft-sampling greedy')
    expect(tuned).toContain('--spec-draft-ngl 0')
    expect(tuned).toContain('--spec-draft-type-k q8_0')
    expect(tuned).not.toContain('--spec-draft-type-v')
    const plain = buildServerArgs({ model: withDraft, config: DEFAULT_LOAD_CONFIG, port: 1, alias: 'q' }).join(' ')
    expect(plain).not.toMatch(/--spec-draft-(n-min|p-min|sampling|ngl|type-k)/)
  })
  it('skips it when disabled or when the sidecar kind is unknown', () => {
    expect(buildServerArgs({ model: withDraft, config: { ...DEFAULT_LOAD_CONFIG, useDraft: false }, port: 1, alias: 'q' }).join(' ')).not.toContain('--spec-type')
    const unknown = { ...withDraft, draft: { path: 'x', sizeBytes: 1 } }
    expect(buildServerArgs({ model: unknown, config: DEFAULT_LOAD_CONFIG, port: 1, alias: 'q' }).join(' ')).not.toContain('--spec-type')
  })
})

describe('splitArgs', () => {
  it('honours quotes', () => {
    expect(splitArgs('-a "b c" d')).toEqual(['-a', 'b c', 'd'])
  })
})

describe('engine output parsing', () => {
  it('reads version and devices', () => {
    const out = 'version: 0.7.0-dev (build 11776, commit ead9f3205)\nAvailable devices:\n  CUDA0: NVIDIA GeForce RTX 5060 Ti (16283 MiB, 15174 MiB free)\n  CUDA1: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 15235 MiB free)\n'
    expect(parseVersion(out)).toBe('0.7.0-dev (build 11776, commit ead9f3205)')
    expect(parseDevices(out)).toEqual([
      { id: 'CUDA0', name: 'NVIDIA GeForce RTX 5060 Ti', totalMiB: 16283, freeMiB: 15174 },
      { id: 'CUDA1', name: 'NVIDIA GeForce RTX 4060 Ti', totalMiB: 16379, freeMiB: 15235 }
    ])
  })
  it('reads nvidia-smi csv', () => {
    expect(parseNvidiaSmi('0, NVIDIA GeForce RTX 5060 Ti, 1200, 16311, 7\n')).toEqual([
      { index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', memUsedMiB: 1200, memTotalMiB: 16311, utilPct: 7 }
    ])
  })
})

describe('load progress', () => {
  it('tracks the dot line and stages', () => {
    const s = new LogLineSplitter()
    expect(s.push('load_tensors: offloaded 33/33 layers\n.....').dots).toBe(5)
    expect(s.push('.....').dots).toBe(10)
    const done = s.push('\nllama_context: constructing\n')
    expect(done.dots).toBe(10)
    expect(done.lines).toEqual(['llama_context: constructing'])
    expect(progressFromDots(100)).toBeCloseTo(0.85)
    expect(nextProgress({ progress: 0.5, stage: 'x' }, 'llama_context: n_ctx = 8192')).toEqual({ progress: 0.86, stage: 'Allocating context' })
    expect(nextProgress({ progress: 0.9, stage: 'x' }, 'load_tensors: foo').progress).toBe(0.9)
  })
})

describe('estimateMemory', () => {
  it('computes KV cache from layers, heads and cache type', () => {
    // 32 layers * 8 heads * (128 + 128) * 2 bytes * 8192 tokens = 1 GiB
    expect(kvCacheBytes(model, { ...DEFAULT_LOAD_CONFIG, contextLength: 8192 }, 8192)).toBe(1024 ** 3)
    const q8 = kvCacheBytes(model, { ...DEFAULT_LOAD_CONFIG, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' }, 8192)!
    expect(q8).toBeCloseTo(1024 ** 3 * (34 / 32 / 2))
  })
  it('gives a verdict against free memory', () => {
    const cfg = { ...DEFAULT_LOAD_CONFIG, contextLength: 8192 }
    expect(estimateMemory(model, cfg, { vramBytes: 30e9, ramBytes: 60e9 }).verdict).toBe('fits')
    expect(estimateMemory(model, cfg, { vramBytes: 4e9, ramBytes: 60e9 }).verdict).toBe('partial')
    expect(estimateMemory(model, { ...cfg, gpuLayers: 'all' }, { vramBytes: 4e9, ramBytes: 60e9 }).verdict).toBe('tight')
    expect(estimateMemory(model, cfg, { vramBytes: null, ramBytes: 60e9 }).verdict).toBe('unknown')
  })

  const gpus = [
    { index: 0, name: 'A', freeBytes: 12e9, totalBytes: 16e9 },
    { index: 1, name: 'B', freeBytes: 4e9, totalBytes: 16e9 }
  ]
  const cfg = { ...DEFAULT_LOAD_CONFIG, contextLength: 8192, useMmproj: false }
  it('splits automatically by free memory and by hand by the sliders', () => {
    expect(gpuShares(cfg, gpus)).toEqual([0.75, 0.25])
    expect(gpuShares({ ...cfg, splitMode: 'custom', tensorSplit: '40,60' }, gpus)).toEqual([0.4, 0.6])
    expect(gpuShares({ ...cfg, splitMode: 'single', mainGpu: 1 }, gpus)).toEqual([0, 1])
  })
  it('breaks the GPU estimate down per GPU and judges each one', () => {
    const even = estimateMemory(model, { ...cfg, splitMode: 'custom', tensorSplit: '50,50' }, { vramBytes: 16e9, ramBytes: 60e9, gpus })
    expect(even.perGpu.map((g) => g.index)).toEqual([0, 1])
    expect(even.perGpu[0].bytes).toBeCloseTo(even.perGpu[1].bytes)
    expect(even.gpuBytes).toBeCloseTo(even.perGpu[0].bytes + even.perGpu[1].bytes)
    // About 5 GB each overflows the 4 GB GPU even though 16 GB is free in total.
    expect(even.verdict).not.toBe('fits')
    const weighted = estimateMemory(model, { ...cfg, splitMode: 'custom', tensorSplit: '85,15' }, { vramBytes: 16e9, ramBytes: 60e9, gpus })
    expect(weighted.verdict).toBe('fits')
    const single = estimateMemory(model, { ...cfg, splitMode: 'single', mainGpu: 0 }, { vramBytes: 12e9, ramBytes: 60e9, gpus })
    expect(single.perGpu.map((g) => g.index)).toEqual([0])
  })
})

describe('GPU split sliders', () => {
  it('reads tensor-split strings', () => {
    expect(parseTensorSplit('3,1', 2)).toEqual([0.75, 0.25])
    expect(parseTensorSplit('1', 2)).toEqual([1, 0])
    expect(parseTensorSplit('', 2)).toBeNull()
    expect(parseTensorSplit('1,x', 2)).toBeNull()
    expect(parseTensorSplit('1,1,1', 2)).toBeNull()
  })
  it('keeps the total at 100 when one slider moves', () => {
    expect(rebalanceSplit([50, 50], 0, 70)).toEqual([70, 30])
    expect(rebalanceSplit([50, 25, 25], 0, 20)).toEqual([20, 40, 40])
    expect(rebalanceSplit([100, 0, 0], 0, 40)).toEqual([40, 30, 30])
    const odd = rebalanceSplit([34, 33, 33], 1, 51)
    expect(odd[1]).toBe(51)
    expect(odd.reduce((a, b) => a + b, 0)).toBe(100)
  })
})
