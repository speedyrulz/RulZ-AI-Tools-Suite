import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

/** Windows briefly locks files (antivirus, indexer, backup); these errors are worth a retry. */
const TRANSIENT = new Set(['EBUSY', 'EPERM', 'EACCES', 'EAGAIN', 'EMFILE', 'ENFILE'])

/** Suffix for a file that no longer parses, e.g. "settings.json.corrupt-20261107T161900". */
export function corruptName(path: string, now = new Date()): string {
  return `${path}.corrupt-${now.toISOString().replace(/[-:]/g, '').replace(/\..*$/, '')}`
}

/**
 * Reads a JSON file. A missing file gives `fallback`. A file that doesn't parse is moved aside
 * (see corruptName) so the next write can't destroy it, and also gives `fallback`. A file that
 * exists but can't be read (still locked after a few retries, or no permission) throws: returning
 * the fallback there would let the caller overwrite the real data with defaults.
 */
export async function readJson<T>(path: string, fallback: T): Promise<T> {
  let text: string
  for (let attempt = 0; ; attempt++) {
    try {
      text = await fs.readFile(path, 'utf8')
      break
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code
      if (code === 'ENOENT' || code === 'ENOTDIR') return fallback
      if (!TRANSIENT.has(code ?? '') || attempt >= 5) throw err
      await sleep(50 * (attempt + 1))
    }
  }
  try {
    return JSON.parse(text.replace(/^﻿/, '')) as T
  } catch {
    await setAside(path)
    return fallback
  }
}

async function setAside(path: string): Promise<void> {
  const target = corruptName(path)
  try {
    await fs.rename(path, target)
  } catch {
    // Renaming fails while another process holds the file open; a copy still keeps the data.
    await fs.copyFile(path, target).catch((err) => console.error(`Could not keep a copy of unreadable ${path}:`, err))
  }
  console.warn(`${path} was not valid JSON; kept it as ${target} and used defaults.`)
}

const writeQueues = new Map<string, Promise<void>>()

/**
 * Writes JSON via a temp file + rename so a crash never leaves a half-written file.
 * Writes to the same path are serialized.
 */
export function writeJsonAtomic(path: string, data: unknown): Promise<void> {
  const previous = writeQueues.get(path) ?? Promise.resolve()
  const next = previous.catch(() => undefined).then(() => doWrite(path, data))
  writeQueues.set(path, next)
  // Handles the rejection on this branch only; the caller still gets it through `next`.
  const cleanup = (): void => {
    if (writeQueues.get(path) === next) writeQueues.delete(path)
  }
  void next.then(cleanup, cleanup)
  return next
}

async function doWrite(path: string, data: unknown): Promise<void> {
  await fs.mkdir(dirname(path), { recursive: true })
  const tmp = `${path}.${process.pid}.tmp`
  const fh = await fs.open(tmp, 'w')
  try {
    await fh.writeFile(JSON.stringify(data, null, 2), 'utf8')
    // Without this a power cut can leave the renamed file empty: the rename reaches the disk
    // before the data does.
    await fh.sync()
  } finally {
    await fh.close()
  }
  // Windows can briefly lock the target (antivirus, indexer); retry a few times.
  for (let attempt = 0; ; attempt++) {
    try {
      await fs.rename(tmp, path)
      return
    } catch (err) {
      if (attempt >= 5) {
        await fs.rm(tmp, { force: true }).catch(() => undefined)
        throw err
      }
      await sleep(50 * (attempt + 1))
    }
  }
}
