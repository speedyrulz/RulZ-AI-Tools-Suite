import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { GgufInfo } from '@shared/types'
import { classifyFile, draftKind, groupModels, type ScannedFile } from '../src/main/models/library'

const root = 'C:\\LLM'
const info = (architecture: string, extra: Partial<GgufInfo> = {}): GgufInfo => ({
  architecture,
  hasChatTemplate: true,
  ...extra
})
const file = (rel: string, sizeGb: number, i: GgufInfo | null = info('llama')): ScannedFile => ({
  path: join(root, rel),
  sizeBytes: sizeGb * 1e9,
  mtimeMs: 1,
  info: i
})

describe('classifyFile', () => {
  it('detects projectors by architecture or name', () => {
    expect(classifyFile('whatever.gguf', info('clip'))).toBe('mmproj')
    expect(classifyFile('Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf', info('qwen2vl'))).toBe('mmproj')
    expect(classifyFile('gemma-4-26B-A4B-it-mmproj-BF16.gguf', null)).toBe('mmproj')
  })
  it('treats draft-head files as draft sidecars but not models with MTP in their name', () => {
    expect(classifyFile('mtp-gemma-4-26B-A4B-it (1).gguf', info('gemma4'))).toBe('draft')
    expect(classifyFile('eagle3-llama.gguf', null)).toBe('draft')
    expect(classifyFile('llama-mtp.gguf', info('llama-assistant'))).toBe('draft')
    expect(classifyFile('Qwen3.6-27B-Fable-NM-DAU-MTP.i1-Q4_K_M.gguf', info('qwen3'))).toBe('model')
    expect(draftKind('mtp-gemma-4-26B-A4B-it (1).gguf')).toBe('mtp')
    expect(draftKind('model-dflash.gguf')).toBe('dflash')
    expect(draftKind('plain.gguf')).toBeUndefined()
  })
  it('marks non-first shards and adapters', () => {
    expect(classifyFile('big-00002-of-00003.gguf', null)).toBe('shard')
    expect(classifyFile('big-00001-of-00003.gguf', info('llama'))).toBe('model')
    expect(classifyFile('lora.gguf', info('llama', { type: 'adapter' }))).toBe('adapter')
  })
})

describe('groupModels', () => {
  it('builds models with sidecars from the same folder and joins shards', () => {
    const models = groupModels(root, [
      file('Qwen3.5\\Q8\\Qwen3.5-9B-Q8_0.gguf', 8.9),
      file('Qwen3.5\\Q8\\mmproj-BF16.gguf', 0.9, info('clip')),
      file('Gemma\\Q5\\gemma-Q5_K_M.gguf', 17.8),
      file('Gemma\\Q5\\mtp-gemma (1).gguf', 0.2),
      file('Gemma\\Q5\\gemma-mmproj-BF16.gguf', 1.1, info('clip')),
      file('Qwen3.6\\x.Q4_K_M.gguf\\x.Q4_K_M.gguf', 15.7),
      file('Kimi\\kimi-00001-of-00002.gguf', 40),
      file('Kimi\\kimi-00002-of-00002.gguf', 30, null)
    ])
    expect(models.map((m) => m.id)).toEqual([
      'Gemma/Q5/gemma-Q5_K_M.gguf',
      'Kimi/kimi-00001-of-00002.gguf',
      'Qwen3.5/Q8/Qwen3.5-9B-Q8_0.gguf',
      'Qwen3.6/x.Q4_K_M.gguf/x.Q4_K_M.gguf'
    ])
    const gemma = models[0]
    expect(gemma.mmproj?.path).toMatch(/gemma-mmproj-BF16\.gguf$/)
    expect(gemma.draft?.path).toMatch(/mtp-gemma \(1\)\.gguf$/)
    expect(gemma.draft?.kind).toBe('mtp')
    const kimi = models[1]
    expect(kimi.name).toBe('kimi')
    expect(kimi.files).toHaveLength(2)
    expect(kimi.sizeBytes).toBe(70e9)
    expect(models[2].mmproj?.sizeBytes).toBe(0.9e9)
    expect(models[3].folder).toBe('Qwen3.6/x.Q4_K_M.gguf')
    expect(models[3].mmproj).toBeUndefined()
  })
})
