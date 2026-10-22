import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { ModelEntry } from '@shared/types'
import { buildServerArgs } from '../src/main/engine/args'
import { parseDevices, parseVersion } from '../src/main/engine/engine'
import { parseNvidiaSmi } from '../src/main/engine/hardware'
import { splitArgs } from '../src/main/engine/process'
import { LogLineSplitter, nextProgress, progressFromDots } from '../src/main/engine/progress'
import { estimateMemory, kvCacheBytes } from '../src/main/models/estimate'

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
})
