import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { GgufInfo, ModelEntry } from '@shared/types'
import {
  classifyFile,
  draftKind,
  groupModels,
  ModelLibrary,
  modelTokens,
  otherShards,
  type ScannedFile
} from '../src/main/models/library'

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

  it('prefers built-in MTP layers and keeps a separate head as a choice', () => {
    const [m] = groupModels(root, [file('Q\\q-MTP.gguf', 17, { ...info('qwen35'), mtpLayers: 1 }), file('Q\\mtp-q.gguf', 0.2)])
    expect(m.draft).toMatchObject({ kind: 'mtp', builtIn: true, path: m.path })
    expect(m.draftSidecar?.path).toMatch(/mtp-q\.gguf$/)
  })
})

describe('sidecars in shared folders', () => {
  it('reduces file names to the model they belong to', () => {
    expect(modelTokens('gemma-3-12b-it-Q4_K_M.gguf')).toEqual(['gemma', '3', '12b', 'it'])
    expect(modelTokens('mmproj-gemma-3-12b-it-f16.gguf')).toEqual(['gemma', '3', '12b', 'it'])
    expect(modelTokens('mtp-gemma-4-26B-A4B-it (1).gguf')).toEqual(['gemma', '4', '26b', 'a4b', 'it'])
    expect(modelTokens('Qwen3-8B.i1-Q4_K_M.gguf')).toEqual(['qwen3', '8b'])
    expect(modelTokens('big-UD-Q4_K_XL-00001-of-00003.gguf')).toEqual(['big'])
    expect(modelTokens('mmproj-BF16.gguf')).toEqual([])
    expect(modelTokens('mmproj-model-f16.gguf')).toEqual([])
  })

  it("gives a flat folder's projector only to the model it was made for", () => {
    const models = groupModels(root, [
      file('gemma-3-12b-it-Q4_K_M.gguf', 7),
      file('gemma-3-27b-it-Q4_K_M.gguf', 16),
      file('Qwen3-8B-Q8_0.gguf', 8),
      file('mmproj-gemma-3-12b-it-f16.gguf', 0.8, info('clip')),
      file('mtp-Qwen3-8B.gguf', 0.2)
    ])
    const by = Object.fromEntries(models.map((m) => [m.name, m]))
    expect(by['gemma-3-12b-it-Q4_K_M'].mmproj?.path).toMatch(/mmproj-gemma-3-12b-it-f16\.gguf$/)
    expect(by['gemma-3-27b-it-Q4_K_M'].mmproj).toBeUndefined()
    expect(by['Qwen3-8B-Q8_0'].mmproj).toBeUndefined()
    expect(by['Qwen3-8B-Q8_0'].draft?.path).toMatch(/mtp-Qwen3-8B\.gguf$/)
    expect(by['gemma-3-12b-it-Q4_K_M'].draft).toBeUndefined()
    expect(by['gemma-3-12b-it-Q4_K_M'].draftSidecar).toBeUndefined()
  })

  it('leaves a generic projector alone when the folder holds different models', () => {
    const models = groupModels(root, [
      file('llama-3-8b-Q4_K_M.gguf', 5),
      file('qwen3-vl-8b-Q4_K_M.gguf', 5),
      file('mmproj-F16.gguf', 0.6, info('clip'))
    ])
    expect(models.every((m) => !m.mmproj)).toBe(true)
  })

  it('matches a generic projector by the model name inside the file', () => {
    const models = groupModels(root, [
      file('llama-3-8b-Q4_K_M.gguf', 5, info('llama', { name: 'Llama 3 8B' })),
      file('qwen3-vl-8b-Q4_K_M.gguf', 5, info('qwen3vl', { name: 'Qwen3 VL 8B' })),
      file('mmproj-F16.gguf', 0.6, info('clip', { name: 'Qwen3 VL 8B' }))
    ])
    expect(models.find((m) => m.name.startsWith('qwen'))?.mmproj?.path).toMatch(/mmproj-F16\.gguf$/)
    expect(models.find((m) => m.name.startsWith('llama'))?.mmproj).toBeUndefined()
  })

  it('shares generic sidecars between quants of one model in its own folder', () => {
    const models = groupModels(root, [
      file('Qwen3.5\\Qwen3.5-9B-Q4_K_M.gguf', 5),
      file('Qwen3.5\\Qwen3.5-9B-Q8_0.gguf', 9),
      file('Qwen3.5\\Qwen3.5-9B-Instruct-Q6_K.gguf', 7),
      file('Qwen3.5\\mmproj-BF16.gguf', 0.9, info('clip')),
      file('Qwen3.5\\mmproj-F16.gguf', 0.9, info('clip'))
    ])
    expect(models).toHaveLength(3)
    expect(models.every((m) => m.folder === 'Qwen3.5')).toBe(true)
    for (const m of models) expect(m.mmproj?.path).toMatch(/mmproj-BF16\.gguf$/)
  })

  it('finds the other shards of a split model from its first shard only', () => {
    const found = ['big-00001-of-00003.gguf', 'big-00003-of-00003.gguf', 'big-00002-of-00003.gguf', 'big-00002-of-00004.gguf', 'other-00002-of-00003.gguf']
      .map((n) => ({ path: join(root, n) }))
    expect(otherShards(join(root, 'big-00001-of-00003.gguf'), found).map((f) => f.path)).toEqual([
      join(root, 'big-00002-of-00003.gguf'),
      join(root, 'big-00003-of-00003.gguf')
    ])
    expect(otherShards(join(root, 'big-00002-of-00003.gguf'), found)).toEqual([])
    expect(otherShards(join(root, 'single.gguf'), found)).toEqual([])
  })
})

describe('ModelLibrary.scan', () => {
  const setup = (): { a: string; b: string; cache: string } => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-lib-'))
    const a = join(dir, 'A')
    const b = join(dir, 'B')
    mkdirSync(a)
    mkdirSync(b)
    // Not real GGUF files: they show up as models with a read error, which is all this needs.
    writeFileSync(join(a, 'a.gguf'), 'x')
    writeFileSync(join(b, 'b.gguf'), 'x')
    return { a, b, cache: join(dir, 'cache.json') }
  }

  it('scans the new folder when the folder changes during a scan', async () => {
    const { a, b, cache } = setup()
    let root = a
    const published: ModelEntry[][] = []
    const lib = new ModelLibrary(cache, () => root, (m) => published.push(m))
    const first = lib.scan()
    root = b
    const second = lib.scan()
    expect(second).not.toBe(first)
    expect((await second).map((m) => m.id)).toEqual(['b.gguf'])
    await first
    expect(published.at(-1)?.map((m) => m.id)).toEqual(['b.gguf'])
    expect(lib.list().map((m) => m.id)).toEqual(['b.gguf'])
    lib.dispose()
  })

  it('rescans by itself when the folder changed and nobody asked', async () => {
    const { a, b, cache } = setup()
    let root = a
    const published: string[][] = []
    const lib = new ModelLibrary(cache, () => root, (m) => published.push(m.map((x) => x.id)))
    const first = lib.scan()
    root = b
    await first
    await expect.poll(() => published.at(-1)).toEqual(['b.gguf'])
    expect(published).not.toContainEqual(['a.gguf'])
    lib.dispose()
  })
})
