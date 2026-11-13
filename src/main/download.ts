import { createHash, type Hash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { renameRetrying } from './archive'
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
  /** Stops the download when no data arrives for this long (default STALL_MS). */
  stallMs?: number
}

/**
 * A download that receives nothing for this long is stopped. Nothing else would: a connection that dies without
 * closing (Wi-Fi dropping, say) leaves the job waiting forever, its speed frozen at the last value.
 */
export const STALL_MS = 60_000
/** The SHA256SUMS file is a few lines: it gets this long in all. */
const SUMS_TIMEOUT_MS = 30_000

function either(a: AbortSignal | undefined, b: AbortSignal): AbortSignal {
  return a ? AbortSignal.any([a, b]) : b
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

/**
 * Feeds the bytes already in a .part file into the hash, so a resumed download is checked whole. `alive` is called as
 * it reads: reading back gigabytes from a slow disk can take longer than the stall limit, and isn't a stall.
 */
async function hashFile(path: string, hash: Hash, alive: () => void, signal: AbortSignal): Promise<void> {
  for await (const chunk of createReadStream(path)) {
    signal.throwIfAborted()
    hash.update(chunk as Buffer)
    alive()
  }
}

/**
 * Downloads `url` to `dest` via a .part file, so a cancelled download never looks complete. A download that stopped
 * part way is resumed from where it got to (HTTP Range), when the server says the file hasn't changed since.
 */
export async function download(url: string, dest: string, opts: DownloadOptions = {}): Promise<void> {
  const stallMs = opts.stallMs ?? STALL_MS
  const stalled = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  const alive = (): void => {
    clearTimeout(timer)
    timer = setTimeout(() => stalled.abort(), stallMs)
  }
  alive()
  try {
    await fetchTo(url, dest, opts, either(opts.signal, stalled.signal), alive)
  } catch (e) {
    if (stalled.signal.aborted && !opts.signal?.aborted) {
      // The part file is kept (when the server named the file's version), so trying again carries on from here.
      throw new Error(`The download stopped: nothing arrived for ${Math.round(stallMs / 1000)} seconds. Check the connection and try again.`)
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
}

/** The download itself; `alive` is called whenever data arrives, and `signal` also fires when it stalls. */
async function fetchTo(
  url: string,
  dest: string,
  opts: DownloadOptions,
  signal: AbortSignal,
  alive: () => void
): Promise<void> {
  await mkdir(dirname(dest), { recursive: true })
  const part = `${dest}.part`
  const metaPath = `${part}.json`
  const meta = await readMeta(metaPath)
  const have = meta?.url === url && meta.validator ? ((await stat(part).catch(() => null))?.size ?? 0) : 0

  // The part is read into the hash before asking for the rest: reading gigabytes back while the answer waits unread
  // could let the server drop the connection. Wasted (and redone) only when the server sends the whole file again.
  let hash = opts.sha256 ? createHash('sha256') : null
  if (hash && have > 0) await hashFile(part, hash, alive, signal)

  const headers: Record<string, string> = { 'User-Agent': 'RulZ-AI-Tools-Suite' }
  if (have > 0 && meta) {
    headers.Range = `bytes=${have}-`
    headers['If-Range'] = meta.validator
  }
  let res = await getFetch()(url, { headers, signal, redirect: 'follow' })
  // 416: the part is no use (already the whole file, or bigger than it now); start over.
  if (res.status === 416) {
    await res.body?.cancel().catch(() => undefined)
    await rm(part, { force: true })
    res = await getFetch()(url, { headers: { 'User-Agent': 'RulZ-AI-Tools-Suite' }, signal, redirect: 'follow' })
  }
  if (!res.ok || !res.body) throw new Error(`Download failed: ${res.status} ${res.statusText}`)

  // 206 continues the part file; anything else is the whole file from the start.
  const resumeAt = res.status === 206 && have > 0 && new RegExp(`^bytes ${have}-`).test(res.headers.get('content-range') ?? '') ? have : 0
  if (res.status === 206 && !resumeAt) {
    await res.body.cancel().catch(() => undefined)
    await rm(part, { force: true })
    await rm(metaPath, { force: true })
    return fetchTo(url, dest, opts, signal, alive)
  }
  const length = Number(res.headers.get('content-length')) || 0
  const total = (length ? resumeAt + length : 0) || opts.expectedSize || null
  const validator = res.headers.get('etag') ?? res.headers.get('last-modified') ?? ''
  if (validator) await writeFile(metaPath, JSON.stringify({ url, validator } satisfies PartMeta), 'utf8')
  else await rm(metaPath, { force: true })

  if (hash && have > 0 && !resumeAt) hash = createHash('sha256')
  alive()

  let bytes = resumeAt
  let lastEmit = 0
  const samples: { t: number; bytes: number }[] = [{ t: Date.now(), bytes }]
  const counter = new Transform({
    transform(chunk: Buffer, _enc, done) {
      alive()
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
      signal
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
  // The size GitHub lists for the file wins over the response's own: a short file sent with a matching
  // Content-Length would pass the check above, and older releases have no checksum to catch it.
  if (opts.expectedSize && bytes !== opts.expectedSize) {
    await rm(part, { force: true })
    throw new Error(`The download isn't the size of the published file (${bytes} of ${opts.expectedSize} bytes). Try again.`)
  }
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
  // An antivirus scan of the file just written can hold it for a moment.
  await renameRetrying(part, dest)
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
  const res = await getFetch()(sumsUrl, {
    headers: { 'User-Agent': 'RulZ-AI-Tools-Suite' },
    signal: either(signal, AbortSignal.timeout(SUMS_TIMEOUT_MS)),
    redirect: 'follow'
  })
  if (!res.ok) throw new Error(`Couldn't fetch the release's checksums (${res.status}).`)
  return parseSha256Sums(await res.text()).get(assetName) ?? null
}
