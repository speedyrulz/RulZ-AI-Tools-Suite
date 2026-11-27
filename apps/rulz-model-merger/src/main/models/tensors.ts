// Reading and writing tensor data of .safetensors files (the header reader lives in safetensors.ts).
// Only small files are loaded whole (LoRAs); writes go through a file handle in chunks.
import { createHash } from 'node:crypto'
import { open } from 'node:fs/promises'
import { readSafetensorsHeader, type SafetensorsHeader } from './safetensors'

const READ_CHUNK = 64 * 1024 * 1024
const WRITE_CHUNK = 16 * 1024 * 1024

const DTYPE_BYTES: Record<string, number> = {
  BOOL: 1,
  U8: 1,
  I8: 1,
  F8_E4M3: 1,
  F8_E5M2: 1,
  I16: 2,
  U16: 2,
  F16: 2,
  BF16: 2,
  I32: 4,
  U32: 4,
  F32: 4,
  I64: 8,
  U64: 8,
  F64: 8
}

export function dtypeBytes(dtype: string): number | null {
  return DTYPE_BYTES[dtype] ?? null
}

function numel(shape: number[]): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

/** Reads a whole file: its header and the bytes after it (tensor offsets are relative to `data`). */
export async function readSafetensorsFile(path: string): Promise<{ header: SafetensorsHeader; data: Buffer }> {
  const header = await readSafetensorsHeader(path)
  const start = 8 + header.headerBytes
  const size = header.fileBytes - start
  const data = Buffer.allocUnsafe(size)
  const fh = await open(path, 'r')
  try {
    let got = 0
    while (got < size) {
      const want = Math.min(READ_CHUNK, size - got)
      const r = await fh.read(data, got, want, start + got)
      if (r.bytesRead === 0) throw new Error('The file ended before all of its tensor data was read.')
      got += r.bytesRead
    }
  } finally {
    await fh.close()
  }
  for (const [key, t] of Object.entries(header.tensors)) {
    const [a, b] = t.offsets
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < a || b > size) {
      throw new Error(`Tensor "${key}" points outside the file's data.`)
    }
  }
  return { header, data }
}

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

/** The tensor's values as doubles. Handles BF16, F16, F32 and F64 (little-endian). */
export function tensorValues(header: SafetensorsHeader, data: Buffer, key: string): Float64Array {
  const t = header.tensors[key]
  if (!t) throw new Error(`The file has no tensor "${key}".`)
  const bytes = dtypeBytes(t.dtype)
  if (t.dtype !== 'BF16' && t.dtype !== 'F16' && t.dtype !== 'F32' && t.dtype !== 'F64') {
    throw new Error(`Tensor "${key}" is stored as ${t.dtype}; only BF16, F16, F32 and F64 can be read.`)
  }
  const n = numel(t.shape)
  const [a, b] = t.offsets
  if (a < 0 || b < a || b > data.length) throw new Error(`Tensor "${key}" points outside the file's data.`)
  if (b - a !== n * bytes!) {
    throw new Error(`Tensor "${key}" has ${b - a} bytes but its shape and ${t.dtype} type need ${n * bytes!}.`)
  }
  const out = new Float64Array(n)
  if (t.dtype === 'BF16') {
    for (let i = 0; i < n; i++) {
      u32[0] = data.readUInt16LE(a + 2 * i) << 16
      out[i] = f32[0]
    }
  } else if (t.dtype === 'F16') {
    for (let i = 0; i < n; i++) out[i] = halfToDouble(data.readUInt16LE(a + 2 * i))
  } else if (t.dtype === 'F32') {
    for (let i = 0; i < n; i++) out[i] = data.readFloatLE(a + 4 * i)
  } else {
    for (let i = 0; i < n; i++) out[i] = data.readDoubleLE(a + 8 * i)
  }
  return out
}

function halfToDouble(h: number): number {
  const sign = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const m = h & 0x3ff
  if (e === 0) return sign * m * 2 ** -24
  if (e === 31) return m === 0 ? sign * Infinity : NaN
  return sign * (1 + m / 1024) * 2 ** (e - 15)
}

/** Round half to even for non-negative x. */
function roundEven(x: number): number {
  const r = Math.floor(x)
  const d = x - r
  if (d > 0.5 || (d === 0.5 && r % 2 === 1)) return r + 1
  return r
}

function bf16Bits(v: number): number {
  if (Number.isNaN(v)) return 0x7fc0
  f32[0] = v
  const bits = u32[0]
  return ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff
}

function f16Bits(v: number): number {
  if (Number.isNaN(v)) return 0x7e00
  const sign = v < 0 || Object.is(v, -0) ? 0x8000 : 0
  const abs = Math.abs(v)
  if (abs >= 65520) return sign | 0x7c00
  if (abs < 2 ** -14) return sign | roundEven(abs * 2 ** 24) // subnormal; 1024 carries into the smallest normal
  // Normal range: exponent and 10-bit mantissa from the double directly.
  let e = Math.floor(Math.log2(abs))
  if (2 ** e > abs) e--
  else if (2 ** (e + 1) <= abs) e++
  const m = roundEven((abs / 2 ** e - 1) * 1024) // 0..1024
  return sign | (((e + 15) << 10) + m)
}

/** Encodes doubles as BF16 (round to nearest even), F16 or F32, little-endian. */
export function encodeValues(values: Float64Array, dtype: 'BF16' | 'F16' | 'F32'): Buffer {
  const n = values.length
  if (dtype === 'F32') {
    const out = Buffer.allocUnsafe(n * 4)
    for (let i = 0; i < n; i++) out.writeFloatLE(values[i], 4 * i)
    return out
  }
  const out = Buffer.allocUnsafe(n * 2)
  if (dtype === 'BF16') for (let i = 0; i < n; i++) out.writeUInt16LE(bf16Bits(values[i]), 2 * i)
  else if (dtype === 'F16') for (let i = 0; i < n; i++) out.writeUInt16LE(f16Bits(values[i]), 2 * i)
  else throw new Error(`Cannot encode values as ${String(dtype)}.`)
  return out
}

export interface TensorToWrite {
  key: string
  dtype: string
  shape: number[]
  data: Buffer
}

/** Writes a valid .safetensors file. Returns its size and the SHA-256 of its JSON header. */
export async function writeSafetensors(
  path: string,
  tensors: TensorToWrite[],
  metadata: Record<string, string>
): Promise<{ bytes: number; headerSha256: string }> {
  const seen = new Set<string>()
  const header: Record<string, unknown> = { __metadata__: metadata }
  let offset = 0
  for (const t of tensors) {
    if (seen.has(t.key)) throw new Error(`The tensor "${t.key}" appears twice.`)
    seen.add(t.key)
    if (t.key === '__metadata__') throw new Error('A tensor cannot be called "__metadata__".')
    const bytes = dtypeBytes(t.dtype)
    if (bytes === null) throw new Error(`Tensor "${t.key}" has the unknown type ${t.dtype}.`)
    const want = numel(t.shape) * bytes
    if (t.data.length !== want) {
      throw new Error(`Tensor "${t.key}" has ${t.data.length} bytes but its shape and ${t.dtype} type need ${want}.`)
    }
    header[t.key] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + t.data.length] }
    offset += t.data.length
  }
  let json = JSON.stringify(header)
  const jsonBytes = Buffer.byteLength(json, 'utf8')
  json += ' '.repeat((8 - (jsonBytes % 8)) % 8)
  const headerBuf = Buffer.from(json, 'utf8')
  const lenBuf = Buffer.alloc(8)
  lenBuf.writeBigUInt64LE(BigInt(headerBuf.length), 0)

  const fh = await open(path, 'w')
  try {
    await writeAll(fh, lenBuf)
    await writeAll(fh, headerBuf)
    for (const t of tensors) {
      for (let p = 0; p < t.data.length; p += WRITE_CHUNK) {
        await writeAll(fh, t.data.subarray(p, Math.min(t.data.length, p + WRITE_CHUNK)))
      }
    }
  } finally {
    await fh.close()
  }
  return { bytes: 8 + headerBuf.length + offset, headerSha256: createHash('sha256').update(headerBuf).digest('hex') }
}

async function writeAll(fh: Awaited<ReturnType<typeof open>>, buf: Buffer): Promise<void> {
  let done = 0
  while (done < buf.length) {
    const r = await fh.write(buf, done, buf.length - done)
    if (r.bytesWritten === 0) throw new Error('Writing the file made no progress.')
    done += r.bytesWritten
  }
}
