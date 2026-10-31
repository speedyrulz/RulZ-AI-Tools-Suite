import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { ChecksumError, download, parseSha256Sums, publishedSha256 } from '../src/main/download'
import { getFetch, setFetch } from '../src/main/github'

const original = getFetch()
let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-dl-'))
})
afterEach(async () => {
  setFetch(original)
  await rm(dir, { recursive: true, force: true })
})

const DATA = Buffer.from('0123456789abcdefghijklmnopqrstuvwxyz'.repeat(100))
const sha = (b: Buffer): string => createHash('sha256').update(b).digest('hex')

/** A file server that honours Range + If-Range against `etag`, recording the request headers. */
function serve(data: Buffer, etag = '"v1"'): { requests: Record<string, string>[] } {
  const requests: Record<string, string>[] = []
  setFetch(async (_url, init) => {
    const h = Object.fromEntries(Object.entries((init?.headers ?? {}) as Record<string, string>))
    requests.push(h)
    const range = /^bytes=(\d+)-$/.exec(h.Range ?? '')
    if (range && (!h['If-Range'] || h['If-Range'] === etag)) {
      const start = Number(range[1])
      if (start >= data.length) return new Response(null, { status: 416 })
      return new Response(data.subarray(start), {
        status: 206,
        headers: { etag, 'content-length': String(data.length - start), 'content-range': `bytes ${start}-${data.length - 1}/${data.length}` }
      })
    }
    return new Response(data, { status: 200, headers: { etag, 'content-length': String(data.length) } })
  })
  return { requests }
}

describe('download', () => {
  it('verifies the SHA-256 while it streams', async () => {
    serve(DATA)
    const dest = join(dir, 'a.zip')
    await download('https://x/a.zip', dest, { sha256: sha(DATA).toUpperCase() })
    expect(await readFile(dest)).toEqual(DATA)
    expect(existsSync(`${dest}.part`)).toBe(false)
  })

  it('rejects and deletes a download whose checksum is wrong', async () => {
    serve(DATA)
    const dest = join(dir, 'a.zip')
    await expect(download('https://x/a.zip', dest, { sha256: '0'.repeat(64) })).rejects.toBeInstanceOf(ChecksumError)
    expect(existsSync(dest)).toBe(false)
    expect(existsSync(`${dest}.part`)).toBe(false)
  })

  it('resumes a partial download with Range and checks the whole file', async () => {
    const { requests } = serve(DATA)
    const dest = join(dir, 'a.zip')
    await writeFile(`${dest}.part`, DATA.subarray(0, 1000))
    await writeFile(`${dest}.part.json`, JSON.stringify({ url: 'https://x/a.zip', validator: '"v1"' }))
    const seen: number[] = []
    await download('https://x/a.zip', dest, { sha256: sha(DATA), onProgress: (p) => seen.push(p.bytes) })
    expect(requests[0].Range).toBe('bytes=1000-')
    expect(requests[0]['If-Range']).toBe('"v1"')
    expect(await readFile(dest)).toEqual(DATA)
    expect(seen.at(-1)).toBe(DATA.length)
    expect(existsSync(`${dest}.part.json`)).toBe(false)
  })

  it('starts over when the file changed since the partial download', async () => {
    serve(DATA, '"v2"')
    const dest = join(dir, 'a.zip')
    await writeFile(`${dest}.part`, Buffer.from('stale bytes from an older file'))
    await writeFile(`${dest}.part.json`, JSON.stringify({ url: 'https://x/a.zip', validator: '"v1"' }))
    await download('https://x/a.zip', dest, { sha256: sha(DATA) })
    expect(await readFile(dest)).toEqual(DATA)
  })

  it("doesn't resume a part from a different address", async () => {
    const { requests } = serve(DATA)
    const dest = join(dir, 'a.zip')
    await writeFile(`${dest}.part`, DATA.subarray(0, 10))
    await writeFile(`${dest}.part.json`, JSON.stringify({ url: 'https://x/other.zip', validator: '"v1"' }))
    await download('https://x/a.zip', dest)
    expect(requests[0].Range).toBeUndefined()
    expect(await readFile(dest)).toEqual(DATA)
  })

  it('keeps the part file of a download that ended early, for the next try', async () => {
    setFetch(async () => new Response(DATA.subarray(0, 500), { status: 200, headers: { etag: '"v1"' } }))
    const dest = join(dir, 'a.zip')
    await expect(download('https://x/a.zip', dest, { expectedSize: DATA.length })).rejects.toThrow(/ended early/)
    expect((await readFile(`${dest}.part`)).length).toBe(500)
    serve(DATA)
    await download('https://x/a.zip', dest, { expectedSize: DATA.length, sha256: sha(DATA) })
    expect(await readFile(dest)).toEqual(DATA)
  })
})

describe('SHA256SUMS', () => {
  it('parses sha256sum output, CRLF and binary markers included', () => {
    const a = 'a'.repeat(64)
    const b = 'B'.repeat(64)
    const sums = parseSha256Sums(`${a}  RulZ-App-v1.0.0-win64.zip\r\n${b} *Setup 1.0.exe\r\n\r\nnot a line\n`)
    expect(sums.get('RulZ-App-v1.0.0-win64.zip')).toBe(a)
    expect(sums.get('Setup 1.0.exe')).toBe('b'.repeat(64))
    expect(sums.size).toBe(2)
  })

  it('is optional: releases without SHA256SUMS give no hash', async () => {
    expect(await publishedSha256(null, 'x.zip')).toBeNull()
  })

  it('looks the asset up in the published file', async () => {
    const hash = sha(DATA)
    setFetch(async () => new Response(`${hash}  a.zip\n`, { status: 200 }))
    expect(await publishedSha256('https://x/SHA256SUMS', 'a.zip')).toBe(hash)
    expect(await publishedSha256('https://x/SHA256SUMS', 'b.zip')).toBeNull()
  })
})
