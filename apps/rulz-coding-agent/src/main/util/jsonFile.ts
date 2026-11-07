import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'

/** Reading failed for a reason other than the file not being there (it is kept as it is). */
export class JsonReadError extends Error {}

const LOCKED = new Set(['EBUSY', 'EPERM', 'EACCES', 'EAGAIN'])

/**
 * Reads a JSON file. `fallback` is only for a file that isn't there. A file that can't be parsed
 * (cut short by a crash, say) is moved aside as `<name>.corrupt-<time>` before the fallback is used,
 * so the next save can't destroy what was in it; a file that stays locked (antivirus, indexer)
 * throws instead. A byte-order mark (PowerShell's `Set-Content -Encoding UTF8` writes one) is fine.
 */
export async function readJson<T>(path: string, fallback: T): Promise<T> {
  let text: string | null = null
  for (let attempt = 0; text === null; attempt++) {
    try {
      text = await fs.readFile(path, 'utf8')
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (code === 'ENOENT') return fallback
      if (!LOCKED.has(code) || attempt >= 5) throw new JsonReadError(`Couldn't read ${path}: ${(err as Error).message}`)
      await new Promise((r) => setTimeout(r, 100 * (attempt + 1)))
    }
  }
  try {
    return JSON.parse(text.replace(/^﻿/, '')) as T
  } catch {
    await fs.rename(path, `${path}.corrupt-${Date.now()}`).catch(() => undefined)
    return fallback
  }
}

const writeQueues = new Map<string, Promise<void>>()

/**
 * Writes JSON via a temp file + rename so a crash never leaves a half-written file.
 * Writes to the same path are serialized. `compact` leaves out the indentation (for big files
 * written often, like a project's state).
 */
export function writeJsonAtomic(path: string, data: unknown, opts: { compact?: boolean } = {}): Promise<void> {
  const previous = writeQueues.get(path) ?? Promise.resolve()
  const next = previous.catch(() => undefined).then(() => doWrite(path, data, opts.compact ?? false))
  writeQueues.set(path, next)
  void next.finally(() => {
    if (writeQueues.get(path) === next) writeQueues.delete(path)
  })
  return next
}

async function doWrite(path: string, data: unknown, compact: boolean): Promise<void> {
  await fs.mkdir(dirname(path), { recursive: true })
  const tmp = `${path}.${process.pid}.tmp`
  // On disk before the rename, so a power cut can't leave the new name pointing at empty data.
  const file = await fs.open(tmp, 'w')
  try {
    await file.writeFile(compact ? JSON.stringify(data) : JSON.stringify(data, null, 2), 'utf8')
    await file.sync()
  } finally {
    await file.close()
  }
  // Windows can briefly lock the target (antivirus, indexer); retry a few times.
  for (let attempt = 0; ; attempt++) {
    try {
      await fs.rename(tmp, path)
      return
    } catch (err) {
      if (attempt >= 5) throw err
      await new Promise((r) => setTimeout(r, 50 * (attempt + 1)))
    }
  }
}
