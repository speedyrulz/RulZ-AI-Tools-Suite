import { existsSync, mkdtempSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { ModelEntry } from '@shared/types'
import { kvCacheBytes } from '../src/main/models/estimate'
import { quantFromName, readGgufHeader, readGgufHeaderAsync, toGgufInfo } from '../src/main/models/gguf'
import { ModelLibrary } from '../src/main/models/library'

type Val =
  | { t: 'u32'; v: number }
  | { t: 'f32'; v: number }
  | { t: 'str'; v: string }
  | { t: 'bool'; v: boolean }
  | { t: 'arr_i32'; v: number[] }
  | { t: 'arr_str'; v: string[] }

function str(s: string): Buffer {
  const b = Buffer.from(s, 'utf8')
  const len = Buffer.alloc(8)
  len.writeBigUInt64LE(BigInt(b.length))
  return Buffer.concat([len, b])
}
function u32(n: number): Buffer {
  const b = Buffer.alloc(4)
  b.writeUInt32LE(n)
  return b
}
function u64(n: number): Buffer {
  const b = Buffer.alloc(8)
  b.writeBigUInt64LE(BigInt(n))
  return b
}

/** Tensors are names (offset 0) or [name, offset]; dataBytes adds an aligned data section of that size. */
function buildGguf(kv: Record<string, Val>, tensors: (string | [string, number])[] = [], dataBytes = 0): Buffer {
  const parts: Buffer[] = [u32(0x46554747), u32(3), u64(tensors.length), u64(Object.keys(kv).length)]
  const tensorInfos = tensors.map((t) => {
    const [name, offset] = typeof t === 'string' ? [t, 0] : t
    return Buffer.concat([str(name), u32(2), u64(4096), u64(4096), u32(12), u64(offset)])
  })
  for (const [k, val] of Object.entries(kv)) {
    parts.push(str(k))
    switch (val.t) {
      case 'u32':
        parts.push(u32(4), u32(val.v))
        break
      case 'f32': {
        const b = Buffer.alloc(4)
        b.writeFloatLE(val.v)
        parts.push(u32(6), b)
        break
      }
      case 'str':
        parts.push(u32(8), str(val.v))
        break
      case 'bool':
        parts.push(u32(7), Buffer.from([val.v ? 1 : 0]))
        break
      case 'arr_i32':
        parts.push(u32(9), u32(5), u64(val.v.length), ...val.v.map((n) => {
          const b = Buffer.alloc(4)
          b.writeInt32LE(n)
          return b
        }))
        break
      case 'arr_str':
        parts.push(u32(9), u32(8), u64(val.v.length), ...val.v.map(str))
        break
    }
  }
  const header = Buffer.concat([...parts, ...tensorInfos])
  if (!dataBytes) return header
  return Buffer.concat([header, Buffer.alloc((32 - (header.length % 32)) % 32), Buffer.alloc(dataBytes)])
}

describe('readGgufHeader', () => {
  it('reads scalars, keeps small numeric arrays and skips string arrays', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-gguf-'))
    const file = join(dir, 'Test-7B-Q4_K_M.gguf')
    const tokens = Array.from({ length: 50000 }, (_, i) => `tok${i}`)
    writeFileSync(
      file,
      buildGguf({
        'general.architecture': { t: 'str', v: 'llama' },
        'general.name': { t: 'str', v: 'Test Model' },
        'general.file_type': { t: 'u32', v: 15 },
        'llama.context_length': { t: 'u32', v: 32768 },
        'llama.block_count': { t: 'u32', v: 4 },
        'llama.embedding_length': { t: 'u32', v: 4096 },
        'llama.attention.head_count': { t: 'u32', v: 32 },
        'llama.attention.head_count_kv': { t: 'arr_i32', v: [8, 0, 8, 0] },
        'llama.rope.freq_base': { t: 'f32', v: 10000 },
        'tokenizer.ggml.tokens': { t: 'arr_str', v: tokens },
        'tokenizer.chat_template': { t: 'str', v: '{{ messages }}' },
        'general.flag': { t: 'bool', v: true }
      })
    )
    const header = readGgufHeader(file)
    expect(header.version).toBe(3)
    expect(header.kv['tokenizer.ggml.tokens']).toBeNull()
    expect(header.kv['general.flag']).toBe(true)

    const info = toGgufInfo(header.kv, 'Test-7B-Q4_K_M.gguf')
    expect(info.architecture).toBe('llama')
    expect(info.contextLength).toBe(32768)
    expect(info.blockCount).toBe(4)
    expect(info.headCountKv).toBe(4) // mean of per-layer [8,0,8,0]
    expect(info.keyLength).toBe(128)
    expect(info.quantization).toBe('Q4_K_M')
    expect(info.hasChatTemplate).toBe(true)
    expect(header.hasNextn).toBe(false)
    expect(info.mtpLayers).toBeUndefined()
  })

  it('finds built-in MTP layers only when the nextn tensors are really there', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-gguf-'))
    const kv = { 'general.architecture': { t: 'str', v: 'qwen35' }, 'qwen35.nextn_predict_layers': { t: 'u32', v: 1 } } as const
    const withLayers = join(dir, 'with.gguf')
    writeFileSync(withLayers, buildGguf(kv, ['blk.0.attn_q.weight', 'blk.64.nextn.eh_proj.weight']))
    const h = readGgufHeader(withLayers)
    expect(h.hasNextn).toBe(true)
    expect(toGgufInfo(h.kv, 'with.gguf', h.hasNextn).mtpLayers).toBe(1)
    // Some conversions keep the key but drop the layers.
    const stripped = join(dir, 'stripped.gguf')
    writeFileSync(stripped, buildGguf(kv, ['blk.0.attn_q.weight']))
    const s = readGgufHeader(stripped)
    expect(toGgufInfo(s.kv, 'stripped.gguf', s.hasNextn).mtpLayers).toBeUndefined()
  })

  it('reads the same header asynchronously, also when it is larger than the first read', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-gguf-'))
    // About 12 MB of string arrays, more than the 8 MiB the async reader starts with.
    const tokens = Array.from({ length: 60000 }, (_, i) => `token-${i}-${'x'.repeat(32)}`)
    const file = join(dir, 'big.gguf')
    writeFileSync(
      file,
      buildGguf(
        {
          'general.architecture': { t: 'str', v: 'qwen35' },
          'tokenizer.ggml.tokens': { t: 'arr_str', v: tokens },
          'tokenizer.ggml.merges': { t: 'arr_str', v: tokens },
          'tokenizer.ggml.extra1': { t: 'arr_str', v: tokens },
          'tokenizer.ggml.extra2': { t: 'arr_str', v: tokens },
          'qwen35.block_count': { t: 'u32', v: 48 }
        },
        ['blk.0.attn_q.weight', 'blk.47.nextn.eh_proj.weight']
      )
    )
    const sync = readGgufHeader(file)
    const async = await readGgufHeaderAsync(file)
    expect(async).toEqual(sync)
    expect(async.kv['qwen35.block_count']).toBe(48)
    expect(async.hasNextn).toBe(true)
    await expect(readGgufHeaderAsync(join(dir, 'missing.gguf'))).rejects.toThrow()
  })

  it('rejects files that are not GGUF', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-gguf-'))
    const file = join(dir, 'bad.gguf')
    writeFileSync(file, Buffer.from('not a gguf file at all'))
    expect(() => readGgufHeader(file)).toThrow(/Not a GGUF/)
  })
})

describe('sliding-window layouts llama.cpp hard-codes', () => {
  const kv = (arch: string, extra: Record<string, unknown> = {}): Record<string, unknown> => ({
    'general.architecture': arch,
    [`${arch}.block_count`]: 48,
    [`${arch}.attention.head_count`]: 16,
    [`${arch}.attention.head_count_kv`]: 8,
    [`${arch}.attention.key_length`]: 256,
    [`${arch}.attention.value_length`]: 256,
    ...extra
  })

  it('fills in the pattern when the file has no pattern key', () => {
    const g3 = toGgufInfo(kv('gemma3', { 'gemma3.attention.sliding_window': 1024 }), 'g3.gguf')
    expect(g3.slidingWindowFraction).toBeCloseTo(5 / 6)
    expect(g3.slidingWindow).toBe(1024)
    expect(toGgufInfo(kv('gpt-oss', { 'gpt-oss.attention.sliding_window': 128 }), 'o.gguf').slidingWindowFraction).toBe(0.5)
    expect(toGgufInfo(kv('cohere2', { 'cohere2.attention.sliding_window': 4096 }), 'c.gguf').slidingWindowFraction).toBe(0.75)
    expect(toGgufInfo(kv('gemma3n', { 'gemma3n.attention.sliding_window': 512 }), 'n.gguf').slidingWindowFraction).toBe(0.8)
    // A pattern stored in the file still wins.
    const own = toGgufInfo(kv('gemma3', { 'gemma3.attention.sliding_window': 1024, 'gemma3.attention.sliding_window_pattern': 4 }), 'g.gguf')
    expect(own.slidingWindowFraction).toBe(0.75)
    expect(toGgufInfo(kv('llama'), 'l.gguf').slidingWindowFraction).toBeUndefined()
  })

  it('uses the window llama.cpp uses', () => {
    expect(toGgufInfo(kv('gemma2'), 'g2.gguf').slidingWindow).toBe(4096) // its default when the key is missing
    expect(toGgufInfo(kv('llama4', { 'llama4.attention.sliding_window': 1234 }), 'l4.gguf').slidingWindow).toBe(8192)
    expect(toGgufInfo(kv('llama4', { 'llama4.attention.sliding_window': 0 }), 'l4.gguf').slidingWindow).toBe(0)
    // Only the 64-layer EXAONE 4 has sliding layers.
    expect(toGgufInfo(kv('exaone4', { 'exaone4.attention.sliding_window': 4096 }), 'e.gguf').slidingWindow).toBeUndefined()
    // Gemma 3 without a window has no sliding layers at all.
    expect(toGgufInfo(kv('gemma3'), 'g3.gguf').slidingWindow).toBeUndefined()
  })

  it('shrinks the KV estimate for long contexts accordingly', () => {
    const model = (info: ReturnType<typeof toGgufInfo>): ModelEntry => ({
      id: 'm',
      name: 'm',
      folder: '',
      path: 'm.gguf',
      files: ['m.gguf'],
      sizeBytes: 1e9,
      info,
      modifiedAt: 0
    })
    const cfg = { ...DEFAULT_LOAD_CONFIG, cacheTypeK: 'f16' as const, cacheTypeV: 'f16' as const, ubatchSize: 512 }
    const ctx = 131072
    const full = kvCacheBytes(model(toGgufInfo(kv('llama'), 'l.gguf')), cfg, ctx)!
    const gemma3 = kvCacheBytes(model(toGgufInfo(kv('gemma3', { 'gemma3.attention.sliding_window': 1024 }), 'g.gguf')), cfg, ctx)!
    // 1 in 6 layers keeps the whole context; the rest keep window + ubatch.
    expect(gemma3 / full).toBeCloseTo(1 / 6 + (5 / 6) * ((1024 + 512) / ctx), 5)
    expect(full / gemma3).toBeGreaterThan(5)
  })
})

describe('built-in MTP layers in split models', () => {
  it('finds nextn tensors that sit in the last shard', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-split-'))
    const kv = {
      'general.architecture': { t: 'str', v: 'qwen35' },
      'qwen35.block_count': { t: 'u32', v: 64 },
      'qwen35.nextn_predict_layers': { t: 'u32', v: 1 }
    } as const
    const split = (n: number): Record<string, Val> => ({ 'split.no': { t: 'u32', v: n } })
    writeFileSync(join(dir, 'q-MTP-00001-of-00003.gguf'), buildGguf(kv, ['blk.0.attn_q.weight']))
    writeFileSync(join(dir, 'q-MTP-00002-of-00003.gguf'), buildGguf(split(1), ['blk.30.attn_q.weight']))
    writeFileSync(join(dir, 'q-MTP-00003-of-00003.gguf'), buildGguf(split(2), ['blk.63.attn_q.weight', 'blk.64.nextn.eh_proj.weight']))
    // A split model without the key stays plain.
    writeFileSync(join(dir, 'plain-00001-of-00002.gguf'), buildGguf({ 'general.architecture': { t: 'str', v: 'llama' } }, ['blk.0.attn_q.weight']))
    writeFileSync(join(dir, 'plain-00002-of-00002.gguf'), buildGguf(split(1), ['blk.1.attn_q.weight']))

    const lib = new ModelLibrary(join(dir, 'cache.json'), () => dir, () => undefined)
    const models = await lib.scan()
    lib.dispose()
    expect(models.map((m) => m.name)).toEqual(['plain', 'q-MTP'])
    expect(models[1].info?.mtpLayers).toBe(1)
    expect(models[1].draft).toMatchObject({ kind: 'mtp', builtIn: true })
    expect(models[0].info?.mtpLayers).toBeUndefined()
  })
})

describe('quantFromName', () => {
  it.each([
    ['gemma-4-26b-a4b-heretic-styletune-v2-head.i1-Q5_K_M.gguf', 'Q5_K_M'],
    ['Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf', 'Q8_0'],
    ['Qwen3.8-27B-TURBO-NM-DAU.i1-IQ4_XS.gguf', 'IQ4_XS'],
    ['mmproj-BF16.gguf', 'BF16'],
    ['Kimi-K2-UD-Q4_K_XL-00001-of-00006.gguf', 'UD-Q4_K_XL'],
    ['Qwen-Image-2.1-PE-I2I.Q5_K_M.gguf', 'Q5_K_M'],
    ['model.gguf', undefined]
  ])('%s → %s', (name, expected) => {
    expect(quantFromName(name)).toBe(expected)
  })
})

// Smoke test against the real model library when it exists on this machine.
const LIB = 'C:\\LLM'
describe.runIf(existsSync(LIB))('real GGUF files', () => {
  it('parses every top-level model header in C:\\LLM', () => {
    const files: string[] = []
    const walk = (d: string): void => {
      for (const n of readdirSync(d)) {
        const p = join(d, n)
        const st = statSync(p)
        if (st.isDirectory()) walk(p)
        else if (/\.gguf$/i.test(n)) files.push(p)
      }
    }
    walk(LIB)
    expect(files.length).toBeGreaterThan(0)
    for (const f of files) {
      const h = readGgufHeader(f)
      expect(typeof h.kv['general.architecture']).toBe('string')
    }
  })
})

describe('MoE expert sizes', () => {
  const moeKv = {
    'general.architecture': { t: 'str', v: 'qwen3moe' },
    'qwen3moe.block_count': { t: 'u32', v: 2 },
    'qwen3moe.expert_count': { t: 'u32', v: 8 }
  } as const

  it('adds up the routed-expert tensors of each layer from the tensor offsets', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-moe-'))
    const path = join(dir, 'moe.gguf')
    // Listed out of order on purpose: sizes come from the next tensor by offset, the last runs to the end.
    writeFileSync(
      path,
      buildGguf(
        moeKv,
        [
          ['blk.1.ffn_down_exps.weight', 1600],
          ['blk.0.attn_q.weight', 0],
          ['blk.0.ffn_up_exps.weight', 100],
          ['blk.0.ffn_gate_exps.weight', 400],
          ['blk.0.ffn_down_shexp.weight', 700],
          ['blk.1.ffn_gate_up_exps.weight', 800]
        ],
        2000
      )
    )
    const h = readGgufHeader(path)
    // Layer 0: up 300 + gate 300; the shared expert stays on the GPU. Layer 1: fused gate_up 800 +
    // down 400 (Gemma 4 style), which llama.cpp's --cpu-moe pattern covers too.
    expect(h.expertBytes).toEqual([600, 1200])
    expect(toGgufInfo(h.kv, 'moe.gguf', h.hasNextn, h.expertBytes).expertBytesByLayer).toEqual([600, 1200])
  })

  it('sums experts over every shard of a split model', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-moe-split-'))
    const split = (n: number): Record<string, Val> => ({ 'split.no': { t: 'u32', v: n } })
    writeFileSync(join(dir, 'm-00001-of-00002.gguf'), buildGguf(moeKv, [['blk.0.ffn_up_exps.weight', 0]], 500))
    writeFileSync(join(dir, 'm-00002-of-00002.gguf'), buildGguf(split(1), [['blk.1.ffn_up_exps.weight', 0]], 700))
    const lib = new ModelLibrary(join(dir, 'cache.json'), () => dir, () => undefined)
    const models = await lib.scan()
    lib.dispose()
    expect(models[0].info?.expertBytesByLayer).toEqual([500, 700])
  })

  it('leaves dense models without expert sizes', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-dense-'))
    const path = join(dir, 'dense.gguf')
    writeFileSync(path, buildGguf({ 'general.architecture': { t: 'str', v: 'llama' } }, [['blk.0.ffn_up.weight', 0]], 64))
    expect(readGgufHeader(path).expertBytes).toBeUndefined()
  })
})
