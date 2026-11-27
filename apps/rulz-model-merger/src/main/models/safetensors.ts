// Reads only the header of a .safetensors file (8 bytes length + JSON). Never the tensor data: files are 5-30 GB and may
// sit behind a \\wsl.localhost\ UNC path.
import { createHash } from 'node:crypto'
import { open } from 'node:fs/promises'

export interface SafetensorsTensor {
  dtype: string
  shape: number[]
  offsets: [number, number]
}

export interface SafetensorsHeader {
  tensors: Record<string, SafetensorsTensor>
  /** The "__metadata__" entry (string values only). */
  metadata: Record<string, string>
  /** Length of the JSON header in bytes (without the 8-byte length field). */
  headerBytes: number
  fileBytes: number
  /** SHA-256 hex of the JSON header bytes. */
  headerSha256: string
}

const DEFAULT_MAX_HEADER = 100 * 1024 * 1024

export async function readSafetensorsHeader(path: string, opts?: { maxHeaderBytes?: number }): Promise<SafetensorsHeader> {
  const maxHeader = opts?.maxHeaderBytes ?? DEFAULT_MAX_HEADER
  const fh = await open(path, 'r')
  try {
    const fileBytes = (await fh.stat()).size
    if (fileBytes < 8) throw new Error('Not a safetensors file: it is shorter than 8 bytes.')
    const lenBuf = Buffer.alloc(8)
    const first = await fh.read(lenBuf, 0, 8, 0)
    if (first.bytesRead < 8) throw new Error('Not a safetensors file: it is shorter than 8 bytes.')
    const n = lenBuf.readBigUInt64LE(0)
    if (n > BigInt(maxHeader)) {
      throw new Error(`Not a safetensors file: the header claims ${n} bytes (limit ${maxHeader}).`)
    }
    const headerBytes = Number(n)
    if (8 + headerBytes > fileBytes) throw new Error('Not a safetensors file: the header runs past the end of the file.')
    const buf = Buffer.alloc(headerBytes)
    let got = 0
    while (got < headerBytes) {
      const r = await fh.read(buf, got, headerBytes - got, 8 + got)
      if (r.bytesRead === 0) throw new Error('The file ended while reading the header.')
      got += r.bytesRead
    }
    let json: unknown
    try {
      json = JSON.parse(buf.toString('utf8'))
    } catch {
      throw new Error('Not a safetensors file: the header is not valid JSON.')
    }
    if (json === null || typeof json !== 'object' || Array.isArray(json)) {
      throw new Error('Not a safetensors file: the header is not a JSON object.')
    }
    const tensors: Record<string, SafetensorsTensor> = {}
    const metadata: Record<string, string> = {}
    for (const [key, value] of Object.entries(json as Record<string, unknown>)) {
      if (key === '__metadata__') {
        if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
          for (const [mk, mv] of Object.entries(value as Record<string, unknown>)) {
            if (typeof mv === 'string') metadata[mk] = mv
          }
        }
        continue
      }
      const t = value as Record<string, unknown> | null
      const shape = t?.shape
      const offs = t?.data_offsets
      if (
        t === null ||
        typeof t !== 'object' ||
        typeof t.dtype !== 'string' ||
        !Array.isArray(shape) ||
        !shape.every((d) => typeof d === 'number') ||
        !Array.isArray(offs) ||
        offs.length !== 2 ||
        !offs.every((d) => typeof d === 'number')
      ) {
        throw new Error(`Bad safetensors header: tensor "${key}" has no dtype, shape and data_offsets.`)
      }
      tensors[key] = { dtype: t.dtype, shape: shape as number[], offsets: [offs[0] as number, offs[1] as number] }
    }
    const headerSha256 = createHash('sha256').update(buf).digest('hex')
    return { tensors, metadata, headerBytes, fileBytes, headerSha256 }
  } finally {
    await fh.close()
  }
}
