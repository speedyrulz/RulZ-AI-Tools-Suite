import { closeSync, fstatSync, openSync, readSync } from 'node:fs'
import type { GgufInfo } from '@shared/types'

// GGUF value types (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
const U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7, STR = 8, ARR = 9, U64 = 10, I64 = 11, F64 = 12 // prettier-ignore
const SCALAR_SIZE: Record<number, number> = { [U8]: 1, [I8]: 1, [U16]: 2, [I16]: 2, [U32]: 4, [I32]: 4, [F32]: 4, [BOOL]: 1, [U64]: 8, [I64]: 8, [F64]: 8 } // prettier-ignore
const CHUNK = 1 << 20
/** Numeric arrays up to this length are kept (per-layer head counts and similar). */
const MAX_KEPT_ARRAY = 1024

/** Sequential reader over a file descriptor with a 1 MiB window. */
class Reader {
  private buf = Buffer.alloc(0)
  private start = 0
  pos = 0

  constructor(
    private readonly fd: number,
    private readonly size: number
  ) {}

  private view(n: number): number {
    if (this.pos < this.start || this.pos + n > this.start + this.buf.length) {
      if (this.pos + n > this.size) throw new Error('Unexpected end of GGUF header')
      const len = Math.min(Math.max(n, CHUNK), this.size - this.pos)
      this.buf = Buffer.allocUnsafe(len)
      readSync(this.fd, this.buf, 0, len, this.pos)
      this.start = this.pos
    }
    const off = this.pos - this.start
    this.pos += n
    return off
  }

  // view() may swap this.buf, so the offset must be taken before touching the buffer.
  private at<T>(n: number, read: (buf: Buffer, off: number) => T): T {
    const off = this.view(n)
    return read(this.buf, off)
  }

  u8 = (): number => this.at(1, (b, o) => b.readUInt8(o))
  i8 = (): number => this.at(1, (b, o) => b.readInt8(o))
  u16 = (): number => this.at(2, (b, o) => b.readUInt16LE(o))
  i16 = (): number => this.at(2, (b, o) => b.readInt16LE(o))
  u32 = (): number => this.at(4, (b, o) => b.readUInt32LE(o))
  i32 = (): number => this.at(4, (b, o) => b.readInt32LE(o))
  f32 = (): number => this.at(4, (b, o) => b.readFloatLE(o))
  f64 = (): number => this.at(8, (b, o) => b.readDoubleLE(o))
  u64 = (): number => this.at(8, (b, o) => Number(b.readBigUInt64LE(o)))
  i64 = (): number => this.at(8, (b, o) => Number(b.readBigInt64LE(o)))

  str(): string {
    const len = this.u64()
    if (len > 64 * 1024 * 1024) throw new Error('GGUF string too long')
    const off = this.view(len)
    return this.buf.toString('utf8', off, off + len)
  }

  skip(n: number): void {
    if (this.pos + n > this.size) throw new Error('Unexpected end of GGUF header')
    this.pos += n
  }
}

function readScalar(r: Reader, type: number): number | boolean {
  switch (type) {
    case U8: return r.u8() // prettier-ignore
    case I8: return r.i8() // prettier-ignore
    case U16: return r.u16() // prettier-ignore
    case I16: return r.i16() // prettier-ignore
    case U32: return r.u32() // prettier-ignore
    case I32: return r.i32() // prettier-ignore
    case F32: return r.f32() // prettier-ignore
    case BOOL: return r.u8() !== 0 // prettier-ignore
    case U64: return r.u64() // prettier-ignore
    case I64: return r.i64() // prettier-ignore
    case F64: return r.f64() // prettier-ignore
    default: throw new Error(`Unknown GGUF value type ${type}`) // prettier-ignore
  }
}

/** Reads a value; large arrays and arrays of strings are skipped and returned as undefined. */
function readValue(r: Reader, type: number): unknown {
  if (type === STR) return r.str()
  if (type !== ARR) return readScalar(r, type)
  const elemType = r.u32()
  const count = r.u64()
  const size = SCALAR_SIZE[elemType]
  if (size !== undefined) {
    if (count > MAX_KEPT_ARRAY) {
      r.skip(count * size)
      return undefined
    }
    const out: (number | boolean)[] = []
    for (let i = 0; i < count; i++) out.push(readScalar(r, elemType))
    return out
  }
  for (let i = 0; i < count; i++) readValue(r, elemType)
  return undefined
}

export interface GgufHeader {
  version: number
  tensorCount: number
  kv: Record<string, unknown>
}

export function readGgufHeader(path: string): GgufHeader {
  const fd = openSync(path, 'r')
  try {
    const r = new Reader(fd, fstatSync(fd).size)
    if (r.u32() !== 0x46554747) throw new Error('Not a GGUF file')
    const version = r.u32()
    if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version ${version}`)
    const tensorCount = r.u64()
    const kvCount = r.u64()
    const kv: Record<string, unknown> = {}
    for (let i = 0; i < kvCount; i++) {
      const key = r.str()
      const type = r.u32()
      const value = readValue(r, type)
      kv[key] = value === undefined ? null : value
    }
    return { version, tensorCount, kv }
  } finally {
    closeSync(fd)
  }
}

// llama_ftype → quantization name
const FILE_TYPES: Record<number, string> = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S',
  12: 'Q3_K_M', 13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K',
  19: 'IQ2_XXS', 20: 'IQ2_XS', 21: 'Q2_K_S', 22: 'IQ3_XS', 23: 'IQ3_XXS', 24: 'IQ1_S', 25: 'IQ4_NL',
  26: 'IQ3_S', 27: 'IQ3_M', 28: 'IQ2_S', 29: 'IQ2_M', 30: 'IQ4_XS', 31: 'IQ1_M', 32: 'BF16',
  36: 'TQ1_0', 37: 'TQ2_0', 38: 'MXFP4_MOE'
} // prettier-ignore

const QUANT_IN_NAME = /(?:^|[-_.])((?:UD-)?(?:I?Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?=$|[-_.])/i

export function quantFromName(fileName: string): string | undefined {
  const base = fileName.replace(/\.gguf$/i, '').replace(/-\d{5}-of-\d{5}$/i, '')
  const parts = base.match(new RegExp(QUANT_IN_NAME.source, 'gi'))
  const last = parts?.at(-1)?.replace(/^[-_.]/, '')
  return last?.toUpperCase()
}

function num(v: unknown, mode: 'mean' | 'max' = 'max'): number | undefined {
  if (typeof v === 'number') return v
  if (Array.isArray(v)) {
    const nums = v.filter((x): x is number => typeof x === 'number')
    if (!nums.length) return undefined
    return mode === 'max' ? Math.max(...nums) : nums.reduce((a, b) => a + b, 0) / nums.length
  }
  return undefined
}

export function toGgufInfo(kv: Record<string, unknown>, fileName: string): GgufInfo {
  const arch = typeof kv['general.architecture'] === 'string' ? (kv['general.architecture'] as string) : 'unknown'
  const a = (k: string): unknown => kv[`${arch}.${k}`]
  const fileType = num(kv['general.file_type'])
  const str = (k: string): string | undefined => (typeof kv[k] === 'string' ? (kv[k] as string) : undefined)
  const headCount = num(a('attention.head_count'))
  const embeddingLength = num(a('embedding_length'))
  const pattern = a('attention.sliding_window_pattern')
  let slidingWindowFraction: number | undefined
  if (Array.isArray(pattern) && pattern.length) slidingWindowFraction = pattern.filter(Boolean).length / pattern.length
  else if (typeof pattern === 'number' && pattern > 1) slidingWindowFraction = (pattern - 1) / pattern
  return {
    architecture: arch,
    type: str('general.type'),
    name: str('general.name'),
    sizeLabel: str('general.size_label'),
    fileType,
    quantization: quantFromName(fileName) ?? (fileType !== undefined ? FILE_TYPES[fileType] : undefined),
    contextLength: num(a('context_length')),
    blockCount: num(a('block_count')),
    embeddingLength,
    headCount,
    // Per-layer arrays: hybrid models have 0 KV heads on recurrent layers, so the mean gives the right total.
    headCountKv: num(a('attention.head_count_kv'), 'mean') ?? headCount,
    keyLength: num(a('attention.key_length')) ?? (headCount && embeddingLength ? embeddingLength / headCount : undefined),
    valueLength: num(a('attention.value_length')) ?? (headCount && embeddingLength ? embeddingLength / headCount : undefined),
    expertCount: num(a('expert_count')),
    expertUsedCount: num(a('expert_used_count')),
    slidingWindow: num(a('attention.sliding_window')),
    slidingWindowFraction,
    hasChatTemplate: typeof kv['tokenizer.chat_template'] === 'string'
  }
}
