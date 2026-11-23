// Uploads of local model files to a rental over ssh (docs/vast-design.md, section 6): resumable, checked, abortable.
import { createHash } from 'node:crypto'
import { createReadStream, promises as fs, statSync } from 'node:fs'
import { Transform, type Readable } from 'node:stream'
import type { TransferProgress } from './contract'
import { marker, SshError, type ScriptRunner } from './ssh'
import { uploadAppendScript, uploadDiscardScript, uploadFinishScript, uploadStateScript } from './scripts'

export interface HashCache {
  get(key: string): string | undefined
  set(key: string, sha256: string): void
}

/** Local hashes by path, size and modification time (hashing a 40 GB file takes minutes; it is done once). */
export function memoryHashCache(): HashCache {
  const m = new Map<string, string>()
  return { get: (k) => m.get(k), set: (k, v) => void m.set(k, v) }
}
/** The cache the upload and the load check share by default. */
export const sharedHashCache: HashCache = memoryHashCache()
const sharedCache = sharedHashCache

/** The cached sha256 of a local file if its path, size and modification time are unchanged since it was hashed; never hashes. */
export function cachedHash(path: string, cache: HashCache = sharedCache): string | undefined {
  try {
    const st = statSync(path)
    return cache.get(`${path}|${st.size}|${Math.round(st.mtimeMs)}`)
  } catch {
    return undefined
  }
}

const abortError = (): SshError => new SshError('aborted', 'Cancelled.')

/** The `source` recorded for an uploaded file: its modification time on this PC, so a file replaced by another of the same size is noticed. */
export function localSource(path: string): string | undefined {
  try {
    return `pc:${Math.max(0, Math.round(statSync(path).mtimeMs))}`
  } catch {
    return undefined
  }
}

/** sha256 of a file, from the cache when its path, size and modification time are unchanged. */
export async function hashFile(
  path: string,
  opts: { cache?: HashCache; signal?: AbortSignal; onProgress?: (done: number, total: number) => void } = {}
): Promise<{ sha256: string; size: number }> {
  const st = await fs.stat(path)
  const key = `${path}|${st.size}|${Math.round(st.mtimeMs)}`
  const cache = opts.cache ?? sharedCache
  const hit = cache.get(key)
  if (hit) return { sha256: hit, size: st.size }
  const hash = createHash('sha256')
  let done = 0
  const stream = createReadStream(path, { highWaterMark: 4 << 20 })
  const onAbort = (): void => void stream.destroy(abortError())
  opts.signal?.addEventListener('abort', onAbort, { once: true })
  try {
    if (opts.signal?.aborted) throw abortError()
    for await (const chunk of stream as AsyncIterable<Buffer>) {
      hash.update(chunk)
      done += chunk.length
      opts.onProgress?.(done, st.size)
    }
  } finally {
    opts.signal?.removeEventListener('abort', onAbort)
  }
  const sha256 = hash.digest('hex')
  cache.set(key, sha256)
  return { sha256, size: st.size }
}

export interface UploadOptions {
  ssh: ScriptRunner
  rentalId: number
  slug: string
  /** Name inside the model folder on the instance. */
  remoteName: string
  localPath: string
  /** Bytes of this file. */
  onProgress: (p: TransferProgress) => void
  signal?: AbortSignal
  cache?: HashCache
  /** Tries after a lost connection (each resumes from what the instance has). Default 5. */
  retries?: number
  /** For tests. */
  openStream?: (path: string, start: number) => Readable
  sleep?: (ms: number) => Promise<void>
}

const realSleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

/**
 * Sends one file. A part file already on the instance is the start (only its size is trusted, so the end result is
 * what is checked): when size and sha256 of the finished file match, it is moved into place; when they do not and the
 * upload was a resume, the part is thrown away and the file is sent once more from the start.
 * Returns the size and sha256 of the file on the instance.
 */
export async function uploadFile(o: UploadOptions): Promise<{ size: number; sha256: string }> {
  const name = o.remoteName
  const total = (await fs.stat(o.localPath)).size
  const hashed = await hashFile(o.localPath, {
    cache: o.cache,
    signal: o.signal,
    onProgress: (d) => o.onProgress({ stage: `Reading ${name}`, bytesDone: d, bytesTotal: total })
  })
  const sha = hashed.sha256
  const open = o.openStream ?? ((p, start) => createReadStream(p, { start, highWaterMark: 1 << 20 }))
  const sleep = o.sleep ?? realSleep
  const retries = o.retries ?? 5
  const check = (): void => {
    if (o.signal?.aborted) throw abortError()
  }

  let restarted = false
  let failures = 0
  for (;;) {
    check()
    o.onProgress({ stage: `Checking ${name} on the instance`, bytesDone: 0, bytesTotal: total })
    const state = await o.ssh.run(o.rentalId, uploadStateScript(o.slug, name), { signal: o.signal })
    const final = Number(marker(state.stdout, 'FINAL') ?? -1)
    let part = Number(marker(state.stdout, 'PART') ?? -1)
    if (final === total) {
      const done = await o.ssh.run(o.rentalId, uploadFinishScript(o.slug, name, total, sha, 'final'), { timeoutMs: 30 * 60_000, signal: o.signal })
      if (marker(done.stdout, 'RESULT') === 'ok') {
        o.onProgress({ stage: `${name} is on the instance`, bytesDone: total, bytesTotal: total })
        return { size: total, sha256: sha }
      }
      await o.ssh.run(o.rentalId, uploadDiscardScript(o.slug, name), { signal: o.signal })
      part = -1
    } else if (final >= 0) {
      await o.ssh.run(o.rentalId, uploadDiscardScript(o.slug, name), { signal: o.signal })
      part = -1
    }
    if (part > total) {
      await o.ssh.run(o.rentalId, uploadDiscardScript(o.slug, name), { signal: o.signal })
      part = -1
    }
    const offset = Math.max(0, part)
    if (offset < total) {
      let sent = 0
      const counter = new Transform({
        transform(chunk: Buffer, _enc, cb) {
          sent += chunk.length
          o.onProgress({ stage: `Uploading ${name}`, bytesDone: offset + sent, bytesTotal: total })
          cb(null, chunk)
        }
      })
      const src = open(o.localPath, offset)
      src.on('error', (err: Error) => counter.destroy(err))
      src.pipe(counter)
      o.onProgress({ stage: `Uploading ${name}`, bytesDone: offset, bytesTotal: total })
      try {
        await o.ssh.run(o.rentalId, uploadAppendScript(o.slug, name), { stdin: counter, timeoutMs: 0, signal: o.signal })
      } catch (err) {
        src.destroy()
        if (err instanceof SshError && (err.kind === 'unreachable' || err.kind === 'transport') && ++failures <= retries) {
          await sleep(Math.min(30_000, 1000 * 2 ** failures))
          continue
        }
        throw err
      }
    }
    o.onProgress({ stage: `Checking ${name} on the instance`, bytesDone: total, bytesTotal: total })
    const fin = await o.ssh.run(o.rentalId, uploadFinishScript(o.slug, name, total, sha, 'part'), { timeoutMs: 30 * 60_000, signal: o.signal })
    const result = marker(fin.stdout, 'RESULT')
    if (result === 'ok') {
      o.onProgress({ stage: `${name} is on the instance`, bytesDone: total, bytesTotal: total })
      return { size: total, sha256: sha }
    }
    if (result === 'short' && ++failures <= retries) continue
    // A resumed prefix is only trusted when the whole file matches: otherwise start over, once.
    if (!restarted && offset > 0) {
      restarted = true
      await o.ssh.run(o.rentalId, uploadDiscardScript(o.slug, name), { signal: o.signal })
      continue
    }
    await o.ssh.run(o.rentalId, uploadDiscardScript(o.slug, name), { signal: o.signal }).catch(() => undefined)
    throw new Error(
      result === 'mismatch'
        ? `${name} arrived damaged: its checksum on the instance differs from the file on this PC. Try the setup again.`
        : `${name} did not arrive completely. Try the setup again.`
    )
  }
}
