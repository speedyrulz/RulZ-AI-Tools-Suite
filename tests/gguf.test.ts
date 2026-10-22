import { existsSync, mkdtempSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { quantFromName, readGgufHeader, toGgufInfo } from '../src/main/models/gguf'

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

function buildGguf(kv: Record<string, Val>): Buffer {
  const parts: Buffer[] = [u32(0x46554747), u32(3), u64(0), u64(Object.keys(kv).length)]
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
  return Buffer.concat(parts)
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
  })

  it('rejects files that are not GGUF', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-gguf-'))
    const file = join(dir, 'bad.gguf')
    writeFileSync(file, Buffer.from('not a gguf file at all'))
    expect(() => readGgufHeader(file)).toThrow(/Not a GGUF/)
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
