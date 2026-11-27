import { createHash } from 'node:crypto'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { readSafetensorsHeader } from '../src/main/models/safetensors'

const dir = mkdtempSync(join(tmpdir(), 'merge-st-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

function writeRaw(name: string, headerJson: string | Buffer, tail = 16, declaredLen?: bigint): string {
  const body = Buffer.isBuffer(headerJson) ? headerJson : Buffer.from(headerJson, 'utf8')
  const len = Buffer.alloc(8)
  len.writeBigUInt64LE(declaredLen ?? BigInt(body.length))
  const p = join(dir, name)
  writeFileSync(p, Buffer.concat([len, body, Buffer.alloc(tail)]))
  return p
}

describe('readSafetensorsHeader', () => {
  it('parses tensors, metadata, sizes and the header hash', async () => {
    const json = JSON.stringify({
      __metadata__: { format: 'pt', note: 'hello', count: 3 },
      'a.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [0, 8] },
      'b.bias': { dtype: 'F32', shape: [4], data_offsets: [8, 24] }
    })
    const p = writeRaw('ok.safetensors', json, 24)
    const h = await readSafetensorsHeader(p)
    expect(Object.keys(h.tensors)).toEqual(['a.weight', 'b.bias'])
    expect(h.tensors['a.weight']).toEqual({ dtype: 'BF16', shape: [2, 2], offsets: [0, 8] })
    expect(h.metadata).toEqual({ format: 'pt', note: 'hello' }) // non-string value dropped
    expect(h.headerBytes).toBe(Buffer.byteLength(json))
    expect(h.fileBytes).toBe(8 + Buffer.byteLength(json) + 24)
    expect(h.headerSha256).toBe(createHash('sha256').update(json).digest('hex'))
  })

  it('accepts a header with no metadata and no tensors', async () => {
    const h = await readSafetensorsHeader(writeRaw('empty.safetensors', '{}', 0))
    expect(h.tensors).toEqual({})
    expect(h.metadata).toEqual({})
  })

  it('rejects a file shorter than 8 bytes', async () => {
    const p = join(dir, 'tiny.safetensors')
    writeFileSync(p, Buffer.from([1, 2, 3]))
    await expect(readSafetensorsHeader(p)).rejects.toThrow(/shorter than 8 bytes/)
  })

  it('rejects a header longer than the limit', async () => {
    const p = writeRaw('big.safetensors', '{}', 0, 5000n)
    await expect(readSafetensorsHeader(p, { maxHeaderBytes: 1000 })).rejects.toThrow(/limit 1000/)
    const q = writeRaw('huge.safetensors', '{}', 0, 2n ** 40n)
    await expect(readSafetensorsHeader(q)).rejects.toThrow(/limit/)
  })

  it('rejects a header that runs past the end of the file', async () => {
    const p = writeRaw('short.safetensors', '{}', 0, 500n)
    await expect(readSafetensorsHeader(p)).rejects.toThrow(/past the end/)
  })

  it('rejects JSON that does not parse or is not an object', async () => {
    await expect(readSafetensorsHeader(writeRaw('bad.safetensors', '{not json', 0))).rejects.toThrow(/not valid JSON/)
    await expect(readSafetensorsHeader(writeRaw('arr.safetensors', '[1,2]', 0))).rejects.toThrow(/not a JSON object/)
    await expect(readSafetensorsHeader(writeRaw('null.safetensors', 'null', 0))).rejects.toThrow(/not a JSON object/)
  })

  it('rejects tensor entries without dtype, shape or data_offsets', async () => {
    const bad = [
      { x: { shape: [1], data_offsets: [0, 1] } },
      { x: { dtype: 'F32', data_offsets: [0, 1] } },
      { x: { dtype: 'F32', shape: [1] } },
      { x: { dtype: 'F32', shape: [1], data_offsets: [0] } },
      { x: 'nope' }
    ]
    for (const [i, entry] of bad.entries()) {
      await expect(readSafetensorsHeader(writeRaw(`t${i}.safetensors`, JSON.stringify(entry), 4))).rejects.toThrow(/tensor "x"/)
    }
  })

  it('rejects a missing file and leaves no open handle behind', async () => {
    await expect(readSafetensorsHeader(join(dir, 'missing.safetensors'))).rejects.toThrow()
    // A rejected parse must still close the handle: the file can be deleted on Windows right away.
    const p = writeRaw('closeme.safetensors', '{bad', 0)
    await expect(readSafetensorsHeader(p)).rejects.toThrow()
    rmSync(p)
  })
})
