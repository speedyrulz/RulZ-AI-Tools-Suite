import { createWriteStream } from 'node:fs'
import { mkdir, rename, rm } from 'node:fs/promises'
import { dirname } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { getFetch } from './github'

export interface DownloadProgress {
  bytes: number
  total: number | null
  /** Bytes per second, averaged over the last few seconds. */
  speed: number
}

/** Downloads `url` to `dest` (via a .part file, so a cancelled download never looks complete). */
export async function download(
  url: string,
  dest: string,
  opts: { signal?: AbortSignal; onProgress?: (p: DownloadProgress) => void; expectedSize?: number | null } = {}
): Promise<void> {
  await mkdir(dirname(dest), { recursive: true })
  const res = await getFetch()(url, {
    headers: { 'User-Agent': 'RulZ-AI-Tools-Suite' },
    signal: opts.signal,
    redirect: 'follow'
  })
  if (!res.ok || !res.body) throw new Error(`Download failed: ${res.status} ${res.statusText}`)

  const total = Number(res.headers.get('content-length')) || opts.expectedSize || null
  const part = `${dest}.part`
  let bytes = 0
  let lastEmit = 0
  const samples: { t: number; bytes: number }[] = []

  const body = Readable.fromWeb(res.body as never)
  body.on('data', (chunk: Buffer) => {
    bytes += chunk.length
    const now = Date.now()
    if (now - lastEmit < 200) return
    lastEmit = now
    samples.push({ t: now, bytes })
    while (samples.length > 2 && now - samples[0].t > 4000) samples.shift()
    const first = samples[0]
    const speed = now > first.t ? ((bytes - first.bytes) * 1000) / (now - first.t) : 0
    opts.onProgress?.({ bytes, total, speed })
  })

  try {
    await pipeline(body, createWriteStream(part), { signal: opts.signal })
  } catch (e) {
    await rm(part, { force: true })
    throw e
  }
  if (total && bytes < total) {
    await rm(part, { force: true })
    throw new Error(`Download ended early (${bytes} of ${total} bytes).`)
  }
  opts.onProgress?.({ bytes, total: total ?? bytes, speed: 0 })
  await rm(dest, { force: true })
  await rename(part, dest)
}
