import { cp, mkdir, readdir, rename, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'
import extract from 'extract-zip'

/** Extracts a zip, reporting progress as a 0..1 fraction of entries. */
export async function extractZip(zip: string, dest: string, onProgress?: (fraction: number) => void): Promise<void> {
  await mkdir(dest, { recursive: true })
  let done = 0
  await extract(zip, {
    dir: dest,
    onEntry: (_entry, zipfile) => {
      done++
      if (done % 25 === 0) onProgress?.(done / zipfile.entryCount)
    }
  })
  onProgress?.(1)
}

/** If `dir` holds nothing but one folder (how most zips are packed), moves that folder's contents up. */
export async function flattenSingleFolder(dir: string): Promise<void> {
  const entries = await readdir(dir, { withFileTypes: true })
  if (entries.length !== 1 || !entries[0].isDirectory()) return
  const inner = join(dir, entries[0].name)
  for (const name of await readdir(inner)) await rename(join(inner, name), join(dir, name))
  await rm(inner, { recursive: true, force: true })
}

/** Finds a file by name up to `depth` folders below `dir` (breadth first). */
export async function findFile(dir: string, name: string, depth = 3): Promise<string | null> {
  let level = [dir]
  for (let d = 0; d <= depth && level.length; d++) {
    const next: string[] = []
    for (const folder of level) {
      const entries = await readdir(folder, { withFileTypes: true }).catch(() => [])
      for (const e of entries) {
        if (e.isFile() && e.name.toLowerCase() === name.toLowerCase()) return join(folder, e.name)
        if (e.isDirectory()) next.push(join(folder, e.name))
      }
    }
    level = next
  }
  return null
}

export async function exists(path: string): Promise<boolean> {
  return stat(path).then(
    () => true,
    () => false
  )
}

/**
 * Puts a freshly extracted folder in place of `dir`. The old folder is renamed aside first, so a locked file
 * (the app is still running) fails cleanly and leaves the old version untouched.
 */
export async function swapInto(staged: string, dir: string): Promise<void> {
  const old = `${dir}.old`
  await rm(old, { recursive: true, force: true })
  const hadOld = await exists(dir)
  if (hadOld) {
    try {
      await rename(dir, old)
    } catch {
      throw new Error('Some of its files are in use. Close the app and try again.')
    }
  }
  try {
    await rename(staged, dir)
  } catch (e) {
    if (hadOld) await rename(old, dir).catch(() => undefined)
    throw e
  }
  await rm(old, { recursive: true, force: true }).catch(() => undefined)
}

/** Copies `src` over `dir`, replacing files that exist in both and keeping everything else in `dir`. */
export async function overlay(src: string, dir: string): Promise<void> {
  await cp(src, dir, { recursive: true, force: true })
}
