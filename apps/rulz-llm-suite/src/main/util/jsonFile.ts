import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'

export async function readJson<T>(path: string, fallback: T): Promise<T> {
  try {
    return JSON.parse(await fs.readFile(path, 'utf8')) as T
  } catch {
    return fallback
  }
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
  void next.finally(() => {
    if (writeQueues.get(path) === next) writeQueues.delete(path)
  })
  return next
}

async function doWrite(path: string, data: unknown): Promise<void> {
  await fs.mkdir(dirname(path), { recursive: true })
  const tmp = `${path}.${process.pid}.tmp`
  await fs.writeFile(tmp, JSON.stringify(data, null, 2), 'utf8')
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
