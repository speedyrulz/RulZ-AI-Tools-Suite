// In-project trash: deleted scenes, chapters and entries are moved to .rulz/trash/<stamp>/ with a manifest
// (data safety rule 9). Files keep their relative path inside the stamp folder.
import { promises as fs } from 'node:fs'
import { dirname, join } from 'node:path'
import type { NovelChapter, NovelScene, NovelTrashItem } from '@shared/novel'
import { cleanRelPath, retryTransient, USER_PATH, writeBytesAtomic } from './files'
import { stampOf } from './history'

export type TrashOutline =
  | { kind: 'scene'; chapterId: string; index: number; scene: NovelScene }
  | { kind: 'chapter'; index: number; chapter: NovelChapter }

export interface TrashManifest {
  id: string
  deletedAt: string
  label: string
  files: { from: string; to: string }[]
  outline?: TrashOutline
  /** 'moving' while the files are being moved (a crash leaves it so); absent once complete. */
  status?: 'moving'
}

export const trashRoot = (folder: string): string => join(folder, '.rulz', 'trash')

async function exists(path: string): Promise<boolean> {
  return fs.access(path).then(
    () => true,
    () => false
  )
}

/** rename, or copy + delete across volumes (a project folder can be a junction onto another disk). */
async function moveFile(from: string, to: string): Promise<void> {
  await fs.mkdir(dirname(to), { recursive: true })
  try {
    await retryTransient(() => fs.rename(from, to))
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'EXDEV') throw err
    await retryTransient(() => fs.copyFile(from, to))
    await retryTransient(() => fs.rm(from))
  }
}

/** Moves pairs back in reverse order; returns what could not be moved so the caller can say so. */
async function rollback(moved: { src: string; dst: string }[]): Promise<string[]> {
  const failed: string[] = []
  for (const m of [...moved].reverse()) {
    try {
      await moveFile(m.dst, m.src)
    } catch (err) {
      failed.push(`${m.dst} (${err instanceof Error ? err.message : String(err)})`)
    }
  }
  return failed
}

function withRollbackNote(err: unknown, failed: string[]): Error {
  const base = err instanceof Error ? err : new Error(String(err))
  if (!failed.length) return base
  return new Error(`${base.message} Some files could not be put back and stay where they are: ${failed.join('; ')}`)
}

/**
 * Moves the existing files among `rels` into a new trash folder and writes its manifest. If a move fails,
 * the ones already moved are put back and the error is thrown, so a delete is all or nothing. Files that do
 * not exist are skipped (a missing scene file can still be deleted from the outline).
 */
export async function moveToTrash(folder: string, rels: string[], label: string, outline?: TrashOutline, now = new Date()): Promise<TrashManifest> {
  const base = stampOf(now)
  let id = base
  for (let n = 2; await exists(join(trashRoot(folder), id)); n++) id = `${base}-${n}`
  const dir = join(trashRoot(folder), id)
  const manifestPath = join(dir, 'manifest.json')
  const tmpDir = join(folder, '.rulz', 'tmp')
  const writeManifest = (m: TrashManifest): Promise<boolean> => writeBytesAtomic(manifestPath, tmpDir, Buffer.from(JSON.stringify(m, null, 2), 'utf8'))
  const files: { from: string; to: string }[] = []
  const moved: { src: string; dst: string }[] = []
  try {
    const plan: { rel: string; src: string; to: string; dst: string }[] = []
    for (const rel of rels) {
      const src = join(folder, ...rel.split('/'))
      if (!(await exists(src))) continue
      const to = `.rulz/trash/${id}/${rel}`
      plan.push({ rel, src, to, dst: join(folder, ...to.split('/')) })
    }
    const manifest: TrashManifest = { id, deletedAt: now.toISOString(), label, files: plan.map((x) => ({ from: x.rel, to: x.to })), ...(outline ? { outline } : {}) }
    // The manifest goes first, so a crash in the middle of the moves leaves an item that can be listed and restored.
    await writeManifest({ ...manifest, status: 'moving' })
    for (const x of plan) {
      await moveFile(x.src, x.dst)
      moved.push({ src: x.src, dst: x.dst })
      files.push({ from: x.rel, to: x.to })
    }
    await writeManifest(manifest)
    return manifest
  } catch (err) {
    const failed = await rollback(moved)
    // Only a trash folder that holds no files any more loses its manifest.
    if (!failed.length) await fs.rm(join(dir, 'manifest.json'), { force: true }).catch(() => undefined)
    throw withRollbackNote(err, failed)
  }
}

/** Puts the moved files of a trashed set back; used when the outline write after a delete failed. */
export async function undoTrash(folder: string, manifest: TrashManifest): Promise<void> {
  const failed: string[] = []
  for (const f of manifest.files) {
    try {
      await moveFile(join(folder, ...f.to.split('/')), join(folder, ...f.from.split('/')))
    } catch (err) {
      failed.push(`${f.from} (${err instanceof Error ? err.message : String(err)})`)
    }
  }
  if (failed.length) throw new Error(`Some deleted files could not be put back and stay in the trash: ${failed.join('; ')}`)
  await removeTrashFolder(folder, manifest.id)
}

/** After a restore whose outline write failed: moves the restored files back into their trash item. */
export async function returnToTrash(folder: string, manifest: TrashManifest): Promise<void> {
  const moved = manifest.files.map((f) => ({ src: join(folder, ...f.to.split('/')), dst: join(folder, ...f.from.split('/')) }))
  const failed = await rollback(moved)
  if (failed.length) throw new Error(`Some restored files could not be moved back into the trash: ${failed.join('; ')}`)
}

export async function readManifest(folder: string, id: string): Promise<TrashManifest> {
  if (!/^[\w-]+$/.test(id)) throw new Error('That trash item does not exist.')
  try {
    const m = JSON.parse(await fs.readFile(join(trashRoot(folder), id, 'manifest.json'), 'utf8')) as TrashManifest
    if (!m || !Array.isArray(m.files)) throw new Error('bad manifest')
    // The folder name is the id; a manifest cannot claim another one.
    m.id = id
    return m
  } catch {
    throw new Error('That trash item does not exist or its manifest is unreadable.')
  }
}

/** Trash items, newest first. A folder with an unreadable manifest is skipped (its files stay). */
export async function listTrash(folder: string): Promise<NovelTrashItem[]> {
  let dirs: string[]
  try {
    dirs = (await fs.readdir(trashRoot(folder), { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name)
  } catch {
    return []
  }
  const items: NovelTrashItem[] = []
  for (const id of dirs) {
    try {
      const m = await readManifest(folder, id)
      items.push({ id, deletedAt: m.deletedAt, label: m.status === 'moving' ? `${m.label} (interrupted)` : m.label, files: m.files.map((f) => f.from) })
    } catch {
      // not ours or damaged
    }
  }
  return items.sort((a, b) => (a.deletedAt < b.deletedAt ? 1 : a.deletedAt > b.deletedAt ? -1 : 0))
}

/** Moves the files of a trash item back. Refuses (moving nothing) if any target exists. */
export async function restoreFiles(folder: string, manifest: TrashManifest): Promise<void> {
  // A manifest is plain JSON on disk: its paths are untrusted. "to" must lie inside this item's own
  // folder; "from" must be a file the Writing page edits, or a journal under .rulz/unsaved.
  const pairs = manifest.files.flatMap((f) => {
    if (!f || typeof f.from !== 'string' || typeof f.to !== 'string') throw new Error('The trash item is damaged (bad file list).')
    const prefix = `.rulz/trash/${manifest.id}/`.toLowerCase()
    const to = cleanRelPath(folder, f.to, '.rulz/trash/')
    if (!to.toLowerCase().startsWith(prefix)) throw new Error('The trash item is damaged (a file points outside it).')
    const from = cleanRelPath(folder, f.from, '.rulz/unsaved/')
    if (!USER_PATH.test(from) && !/^\.rulz\/unsaved\/[^/]+\.json$/i.test(from)) throw new Error(`The trash item is damaged ("${f.from}" is not a restorable file).`)
    return [{ from, to, src: join(folder, ...to.split('/')), dst: join(folder, ...from.split('/')) }]
  })
  // An item left by a crash during the moves lists every planned file; only those that arrived can come back.
  if (manifest.status === 'moving') {
    for (let i = pairs.length - 1; i >= 0; i--) if (!(await exists(pairs[i].src))) pairs.splice(i, 1)
  }
  for (const p of pairs) {
    if (await exists(p.dst)) throw new Error(`Cannot restore: "${p.from}" already exists. Rename or remove it first.`)
  }
  const moved: { src: string; dst: string }[] = []
  try {
    for (const p of pairs) {
      await moveFile(p.src, p.dst)
      moved.push({ src: p.src, dst: p.dst })
    }
  } catch (err) {
    // rollback moves dst back to src, so the pairs are swapped here.
    const failed = await rollback(moved.map((m) => ({ src: m.dst, dst: m.src })))
    throw withRollbackNote(err, failed)
  }
}

async function holdsFiles(d: string): Promise<boolean> {
  let entries
  try {
    entries = await fs.readdir(d, { withFileTypes: true })
  } catch {
    return false
  }
  for (const e of entries) {
    if (e.isDirectory() ? await holdsFiles(join(d, e.name)) : true) return true
  }
  return false
}

/**
 * Deletes the manifest and then every empty folder below the stamp folder; never removes a file. If any
 * file is still in the item (apart from the manifest) nothing is touched, so the item stays restorable.
 */
export async function removeTrashFolder(folder: string, id: string): Promise<void> {
  const dir = join(trashRoot(folder), id)
  const manifest = join(dir, 'manifest.json')
  const rest = (await fs.readdir(dir, { withFileTypes: true }).catch(() => [])).filter((e) => !(e.isFile() && e.name === 'manifest.json'))
  for (const e of rest) {
    if (!e.isDirectory() || (await holdsFiles(join(dir, e.name)))) return
  }
  await fs.rm(manifest, { force: true }).catch(() => undefined)
  const prune = async (d: string): Promise<void> => {
    let entries
    try {
      entries = await fs.readdir(d, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) if (e.isDirectory()) await prune(join(d, e.name))
    await fs.rmdir(d).catch(() => undefined)
  }
  await prune(dir)
}
