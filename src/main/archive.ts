import { cp, mkdir, readdir, rename, rm, stat } from 'node:fs/promises'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import extract from 'extract-zip'

/**
 * The folder to install into, tidied up ("D:/Apps/X\" and Explorer's quoted "Copy as path" both give D:\Apps\X). It
 * must be a full path below a drive, and either not exist yet or be an empty folder: a failed install deletes the
 * folder again, so it must not hold anything of the user's. A file there is refused, not replaced.
 */
export async function installFolder(raw: string): Promise<string> {
  const input = raw.trim().replace(/^"(.*)"$/, '$1')
  if (!isAbsolute(input) || !/^[a-z]:[\\/]/i.test(input)) throw new Error('Choose a full folder path, such as D:\\Apps.')
  // Also drops a trailing backslash, which would put the staging folder (<dir>.new) inside the install folder.
  const dir = resolve(input)
  if (dirname(dir) === dir) throw new Error('Choose a folder, not a whole drive, such as D:\\Apps.')
  let entries: string[]
  try {
    entries = await readdir(dir)
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code
    if (code === 'ENOENT') return dir
    if (code === 'ENOTDIR') throw new Error(`${dir} is a file. Choose a folder.`)
    throw new Error(`Can't use ${dir}: ${e instanceof Error ? e.message : String(e)}`)
  }
  if (entries.length) throw new Error(`Choose an empty folder: ${dir} already has files in it.`)
  return dir
}

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
