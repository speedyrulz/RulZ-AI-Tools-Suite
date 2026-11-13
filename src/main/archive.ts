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
  // Also drops a trailing backslash, which would put the staging folder (stagingDir) inside the install folder.
  const dir = resolve(input)
  if (dirname(dir) === dir) throw new Error('Choose a folder, not a whole drive, such as D:\\Apps.')
  // cmd.exe, which runs the apps' .bat setup and start files, replaces %NAME% with a Windows setting even in quotes.
  const named = /%[^%\\]+%/.exec(dir)
  if (named) throw new Error(`Choose a folder without "${named[0]}" in its path: Windows reads it as the name of a setting.`)
  // The launcher's own working folders next to an app's folder: an update deletes and replaces them.
  if (/\.rulz-(new|old)(\\|$)/i.test(dir)) throw new Error('Choose another folder: names ending in .rulz-new or .rulz-old are used by the launcher.')
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

/**
 * Windows refuses a rename while another program (an antivirus scan of files just written, say) has a file open;
 * that passes quickly, so a few tries are made before giving up.
 */
export async function renameRetrying(from: string, to: string): Promise<void> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await rename(from, to)
    } catch (e) {
      const code = (e as NodeJS.ErrnoException).code
      if (attempt >= 5 || (code !== 'EPERM' && code !== 'EBUSY' && code !== 'EACCES')) throw e
      await new Promise((r) => setTimeout(r, 50 * attempt))
    }
  }
}

/**
 * Where a new version is unpacked before it replaces `dir`, and where the old one waits until the swap is done. The
 * launcher deletes both whenever it finds them, so their names are its own: a "<dir>.old" could be the user's backup.
 */
export function stagingDir(dir: string): string {
  return `${dir}.rulz-new`
}

export function backupDir(dir: string): string {
  return `${dir}.rulz-old`
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
  // Renamed out of the way first: the folder may hold something with its own name (App\App\...).
  const inner = join(dir, `.rulz-flatten-${process.pid}`)
  await renameRetrying(join(dir, entries[0].name), inner)
  for (const name of await readdir(inner)) await renameRetrying(join(inner, name), join(dir, name))
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
  const old = backupDir(dir)
  await rm(old, { recursive: true, force: true })
  const hadOld = await exists(dir)
  if (hadOld) {
    try {
      await renameRetrying(dir, old)
    } catch {
      throw new Error('Some of its files are in use. Close the app and try again.')
    }
  }
  try {
    await renameRetrying(staged, dir)
  } catch (e) {
    if (hadOld) await renameRetrying(old, dir).catch(() => undefined)
    throw e
  }
  await rm(old, { recursive: true, force: true }).catch(() => undefined)
}

/** Copies `src` over `dir`, replacing files that exist in both and keeping everything else in `dir`. */
export async function overlay(src: string, dir: string): Promise<void> {
  await cp(src, dir, { recursive: true, force: true })
}
