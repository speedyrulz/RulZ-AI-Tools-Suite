import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { readSafetensorsHeader } from '../src/main/models/safetensors'
import { encodeValues, readSafetensorsFile, tensorValues, writeSafetensors } from '../src/main/models/tensors'
import type { SafetensorsHeader } from '../src/main/models/safetensors'

let dir = ''
beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-tensors-'))
})
afterAll(async () => {
  await rm(dir, { recursive: true, force: true })
})

function oneTensor(dtype: string, data: Buffer, shape: number[]): { header: SafetensorsHeader; data: Buffer } {
  return {
    header: { tensors: { t: { dtype, shape, offsets: [0, data.length] } }, metadata: {}, headerBytes: 0, fileBytes: 0, headerSha256: '' },
    data
  }
}

function roundTrip(values: number[], dtype: 'BF16' | 'F16' | 'F32'): number[] {
  const buf = encodeValues(Float64Array.from(values), dtype)
  const f = oneTensor(dtype, buf, [values.length])
  return [...tensorValues(f.header, f.data, 't')]
}

describe('BF16', () => {
  it('keeps values that fit and rounds others to nearest even', () => {
    expect(roundTrip([0, 1, -2, 0.5, 256, 3.140625], 'BF16')).toEqual([0, 1, -2, 0.5, 256, 3.140625])
    // 1 + 2^-8 is exactly between 1 and 1 + 2^-7: ties to even -> 1. 1 + 3*2^-8 ties between 1+2^-7 (odd) and 1+2^-6 (even).
    expect(roundTrip([1 + 2 ** -8, 1 + 3 * 2 ** -8], 'BF16')).toEqual([1, 1 + 2 ** -6])
    // Just above the tie rounds up.
    expect(roundTrip([1 + 2 ** -8 + 2 ** -20], 'BF16')).toEqual([1 + 2 ** -7])
    expect(roundTrip([-(1 + 2 ** -8 + 2 ** -20)], 'BF16')).toEqual([-(1 + 2 ** -7)])
  })
  it('keeps NaN and infinities', () => {
    const r = roundTrip([NaN, Infinity, -Infinity], 'BF16')
    expect(Number.isNaN(r[0])).toBe(true)
    expect(r[1]).toBe(Infinity)
    expect(r[2]).toBe(-Infinity)
  })
  it('has a relative error of at most 2^-8 for random values', () => {
    const vals = Array.from({ length: 500 }, (_, i) => Math.sin(i * 12.9898) * 43758.5453)
    const r = roundTrip(vals, 'BF16')
    vals.forEach((v, i) => expect(Math.abs(r[i] - v)).toBeLessThanOrEqual(Math.abs(v) * 2 ** -8))
  })
})

describe('F16', () => {
  it('encodes exact values, subnormals and overflow', () => {
    expect(roundTrip([0, 1, -1, 0.5, 65504, 2 ** -14, 2 ** -24, -(2 ** -24), 1000.5], 'F16')).toEqual([
      0, 1, -1, 0.5, 65504, 2 ** -14, 2 ** -24, -(2 ** -24), 1000.5
    ])
    expect(roundTrip([3 * 2 ** -24], 'F16')).toEqual([3 * 2 ** -24])
    // Half of the smallest subnormal ties to even (0); a bit more rounds up.
    expect(roundTrip([2 ** -25, 2 ** -25 * 1.01], 'F16')).toEqual([0, 2 ** -24])
    expect(roundTrip([1e-9], 'F16')).toEqual([0])
    // 65519 rounds down to the max, 65520 and above go to infinity.
    expect(roundTrip([65519, 65520, 1e9, -1e9], 'F16')).toEqual([65504, Infinity, Infinity, -Infinity])
  })
  it('rounds to nearest even in the normal range and carries into the exponent', () => {
    expect(roundTrip([1 + 2 ** -11, 1 + 3 * 2 ** -11], 'F16')).toEqual([1, 1 + 2 ** -9])
    expect(roundTrip([2 - 2 ** -12], 'F16')).toEqual([2])
    const r = roundTrip([NaN, Infinity], 'F16')
    expect(Number.isNaN(r[0])).toBe(true)
    expect(r[1]).toBe(Infinity)
  })
  it('round trips every half-precision bit pattern', () => {
    const all = Buffer.alloc(65536 * 2)
    for (let i = 0; i < 65536; i++) all.writeUInt16LE(i, 2 * i)
    const f = oneTensor('F16', all, [65536])
    const vals = tensorValues(f.header, f.data, 't')
    const back = encodeValues(vals, 'F16')
    for (let i = 0; i < 65536; i++) {
      const isNaNPattern = (i & 0x7c00) === 0x7c00 && (i & 0x3ff) !== 0
      if (isNaNPattern) expect(Number.isNaN(vals[i])).toBe(true)
      else expect(back.readUInt16LE(2 * i)).toBe(i)
    }
  })
})

describe('tensorValues', () => {
  it('reads F32 and F64', () => {
    const b32 = Buffer.alloc(8)
    b32.writeFloatLE(1.5, 0)
    b32.writeFloatLE(-2.25, 4)
    const f = oneTensor('F32', b32, [2])
    expect([...tensorValues(f.header, f.data, 't')]).toEqual([1.5, -2.25])
    const b64 = Buffer.alloc(16)
    b64.writeDoubleLE(Math.PI, 0)
    b64.writeDoubleLE(-1e300, 8)
    const g = oneTensor('F64', b64, [2])
    expect([...tensorValues(g.header, g.data, 't')]).toEqual([Math.PI, -1e300])
  })
  it('refuses other types, size mismatches and missing keys', () => {
    expect(() => {
      const f = oneTensor('I8', Buffer.alloc(4), [4])
      tensorValues(f.header, f.data, 't')
    }).toThrow(/I8/)
    const f = oneTensor('F32', Buffer.alloc(8), [3])
    expect(() => tensorValues(f.header, f.data, 't')).toThrow(/needs? 12|need 12/)
    expect(() => tensorValues(f.header, f.data, 'nope')).toThrow(/no tensor/)
    const g = oneTensor('F32', Buffer.alloc(4), [1])
    g.header.tensors.t.offsets = [0, 8]
    expect(() => tensorValues(g.header, g.data, 't')).toThrow(/outside/)
  })
})

describe('writeSafetensors', () => {
  it('writes a file the header reader and file reader accept, aligned to 8 bytes', async () => {
    const path = join(dir, 'a.safetensors')
    const a = encodeValues(Float64Array.from([1, 2, 3, 4, 5, 6]), 'BF16')
    const b = encodeValues(Float64Array.from([0.25]), 'F32')
    const res = await writeSafetensors(
      path,
      [
        { key: 'a.weight', dtype: 'BF16', shape: [2, 3], data: a },
        { key: 'b.alpha', dtype: 'F32', shape: [], data: b }
      ],
      { format: 'pt', note: 'héllo' }
    )
    const raw = await readFile(path)
    expect(raw.length).toBe(res.bytes)
    const headerLen = Number(raw.readBigUInt64LE(0))
    expect(headerLen % 8).toBe(0)
    expect(res.headerSha256).toBe(createHash('sha256').update(raw.subarray(8, 8 + headerLen)).digest('hex'))
    const h = await readSafetensorsHeader(path)
    expect(h.metadata).toEqual({ format: 'pt', note: 'héllo' })
    expect(h.headerSha256).toBe(res.headerSha256)
    expect(h.tensors['a.weight']).toEqual({ dtype: 'BF16', shape: [2, 3], offsets: [0, 12] })
    expect(h.tensors['b.alpha'].offsets).toEqual([12, 16])
    const f = await readSafetensorsFile(path)
    expect([...tensorValues(f.header, f.data, 'a.weight')]).toEqual([1, 2, 3, 4, 5, 6])
    expect([...tensorValues(f.header, f.data, 'b.alpha')]).toEqual([0.25])
    expect(f.data.length).toBe(16)
  })
  it('writes an empty tensor list', async () => {
    const path = join(dir, 'empty.safetensors')
    const res = await writeSafetensors(path, [], {})
    const f = await readSafetensorsFile(path)
    expect(Object.keys(f.header.tensors)).toEqual([])
    expect(res.bytes).toBe(f.header.fileBytes)
  })
  it('refuses duplicate keys and wrong data lengths', async () => {
    const path = join(dir, 'bad.safetensors')
    const d = Buffer.alloc(4)
    await expect(
      writeSafetensors(path, [{ key: 'x', dtype: 'F32', shape: [1], data: d }, { key: 'x', dtype: 'F32', shape: [1], data: d }], {})
    ).rejects.toThrow(/twice/)
    await expect(writeSafetensors(path, [{ key: 'x', dtype: 'F32', shape: [2], data: d }], {})).rejects.toThrow(/needs? 8|need 8/)
    await expect(writeSafetensors(path, [{ key: 'x', dtype: 'Q9', shape: [1], data: d }], {})).rejects.toThrow(/unknown/)
  })
  it('readSafetensorsFile rejects offsets that run past the data', async () => {
    const path = join(dir, 'trunc.safetensors')
    await writeSafetensors(path, [{ key: 'x', dtype: 'F32', shape: [4], data: Buffer.alloc(16) }], {})
    const raw = await readFile(path)
    const { writeFile } = await import('node:fs/promises')
    const cut = join(dir, 'cut.safetensors')
    await writeFile(cut, raw.subarray(0, raw.length - 4))
    await expect(readSafetensorsFile(cut)).rejects.toThrow(/outside/)
  })
})
