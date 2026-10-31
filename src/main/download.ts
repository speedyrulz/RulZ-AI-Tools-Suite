import { createHash, type Hash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { getFetch } from './github'

export interface DownloadProgress {
  bytes: number
  total: number | null
  /** Bytes per second, averaged over the last few seconds. */
  speed: number
}

export interface DownloadOptions {
  signal?: AbortSignal
  onProgress?: (p: DownloadProgress) => void
  expectedSize?: number | null
  /** Hex SHA-256 the file must have; checked while it streams, and the file is deleted when it doesn't match. */
  sha256?: string | null
}

/** What a .part file was downloaded from, so a later attempt can ask the server for the rest of the same file. */
interface PartMeta {
  url: string
  /** ETag or Last-Modified of the response, sent back as If-Range: a changed file is sent whole again. */
  validator: string
}

export class ChecksumError extends Error {}

async function readMeta(path: string): Promise<PartMeta | null> {
  try {
    return JSON.parse(await readFile(path, 'utf8')) as PartMeta
  } catch {
    return null
  }
}

/** Feeds the bytes already in a .part file into the hash, so a resumed download is checked whole. */
async function hashFile(path: string, hash: Hash): Promise<void> {
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer)
}

/**
 * Downloads `url` to `dest` via a .part file, so a cancelled download never looks complete. A download that stopped
 * part way is resumed from where it got to (HTTP Range), when the server says the file hasn't changed since.
 */
export async function download(url: string, dest: string, opts: DownloadOptions = {}): Promise<void> {
  await mkdir(dirname(dest), { recursive: true })
  const part = `${dest}.part`
  const metaPath = `${part}.json`
  const meta = await readMeta(metaPath)
  const have = meta?.url === url && meta.validator ? ((await stat(part).catch(() => null))?.size ?? 0) : 0

  const headers: Record<string, string> = { 'User-Agent': 'RulZ-AI-Tools-Suite' }
  if (have > 0 && meta) {
    headers.Range = `bytes=${have}-`
    headers['If-Range'] = meta.validator
  }
  let res = await getFetch()(url, { headers, signal: opts.signal, redirect: 'follow' })
  // 416: the part is no use (already the whole file, or bigger than it now); start over.
  if (res.status === 416) {
    await res.body?.cancel().catch(() => undefined)
    await rm(part, { force: true })
    res = await getFetch()(url, { headers: { 'User-Agent': 'RulZ-AI-Tools-Suite' }, signal: opts.signal, redirect: 'follow' })
  }
  if (!res.ok || !res.body) throw new Error(`Download failed: ${res.status} ${res.statusText}`)

  // 206 continues the part file; anything else is the whole file from the start.
  const resumeAt = res.status === 206 && have > 0 && new RegExp(`^bytes ${have}-`).test(res.headers.get('content-range') ?? '') ? have : 0
  if (res.status === 206 && !resumeAt) {
    await res.body.cancel().catch(() => undefined)
    await rm(part, { force: true })
    await rm(metaPath, { force: true })
    return download(url, dest, opts)
  }
  const length = Number(res.headers.get('content-length')) || 0
  const total = (length ? resumeAt + length : 0) || opts.expectedSize || null
  const validator = res.headers.get('etag') ?? res.headers.get('last-modified') ?? ''
  if (validator) await writeFile(metaPath, JSON.stringify({ url, validator } satisfies PartMeta), 'utf8')
  else await rm(metaPath, { force: true })

  const hash = opts.sha256 ? createHash('sha256') : null
  if (hash && resumeAt) await hashFile(part, hash)

  let bytes = resumeAt
  let lastEmit = 0
  const samples: { t: number; bytes: number }[] = [{ t: Date.now(), bytes }]
  const counter = new Transform({
    transform(chunk: Buffer, _enc, done) {
      bytes += chunk.length
      hash?.update(chunk)
      const now = Date.now()
      if (now - lastEmit >= 200) {
        lastEmit = now
        samples.push({ t: now, bytes })
        while (samples.length > 2 && now - samples[0].t > 4000) samples.shift()
        const first = samples[0]
        const speed = now > first.t ? ((bytes - first.bytes) * 1000) / (now - first.t) : 0
        opts.onProgress?.({ bytes, total, speed })
      }
      done(null, chunk)
    }
  })

  // The part file is kept on failure (when the server named a version of the file), so a retry can resume it.
  const keepPart = !!validator
  try {
    await pipeline(Readable.fromWeb(res.body as never), counter, createWriteStream(part, { flags: resumeAt ? 'a' : 'w' }), {
      signal: opts.signal
    })
  } catch (e) {
    if (!keepPart) await rm(part, { force: true })
    throw e
  }
  if (total && bytes < total) {
    if (!keepPart) await rm(part, { force: true })
    throw new Error(`Download ended early (${bytes} of ${total} bytes).`)
  }
  await rm(metaPath, { force: true })
  if (hash && opts.sha256) {
    const got = hash.digest('hex')
    if (got.toLowerCase() !== opts.sha256.toLowerCase()) {
      await rm(part, { force: true })
      throw new ChecksumError(
        "The download is damaged or isn't the published file (its SHA-256 checksum doesn't match). Try again."
      )
    }
  }
  opts.onProgress?.({ bytes, total: total ?? bytes, speed: 0 })
  await rm(dest, { force: true })
  await rename(part, dest)
}

/** Parses a SHA256SUMS file ("<hex>  <name>" lines, as sha256sum writes them) into name → lower-case hash. */
export function parseSha256Sums(text: string): Map<string, string> {
  const sums = new Map<string, string>()
  for (const line of text.split(/\r?\n/)) {
    const m = /^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$/.exec(line.trim())
    if (m) sums.set(m[2], m[1].toLowerCase())
  }
  return sums
}

/**
 * The published SHA-256 of `assetName`, from the release's SHA256SUMS asset. Null when the release has no
 * SHA256SUMS (releases made before it existed) or doesn't list the file.
 */
export async function publishedSha256(
  sumsUrl: string | null | undefined,
  assetName: string,
  signal?: AbortSignal
): Promise<string | null> {
  if (!sumsUrl) return null
  const res = await getFetch()(sumsUrl, { headers: { 'User-Agent': 'RulZ-AI-Tools-Suite' }, signal, redirect: 'follow' })
  if (!res.ok) throw new Error(`Couldn't fetch the release's checksums (${res.status}).`)
  return parseSha256Sums(await res.text()).get(assetName) ?? null
}
