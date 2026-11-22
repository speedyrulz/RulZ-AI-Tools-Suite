// Bytes, encoding, hashes and atomic writes for the Writing mode (data safety rules 1 and 2). No Electron here.
import { createHash, randomBytes } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { basename, dirname, isAbsolute, join, resolve, sep } from 'node:path'
import type { NovelFileMeta } from '@shared/novel'

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

/** Windows briefly locks files (antivirus, indexer, sync); these errors are worth a retry. */
const TRANSIENT = new Set(['EBUSY', 'EPERM', 'EACCES', 'EAGAIN', 'EMFILE', 'ENFILE'])
/** Waits between attempts: about 750 ms in all before giving up. */
const RETRY_WAITS = [50, 100, 150, 200, 250]

export const sha256 = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex')

export interface DecodedFile {
  /** LF line ends, no BOM. Invalid UTF-8 is decoded with replacement characters (display only). */
  text: string
  hash: string
  meta: NovelFileMeta
  readOnly: boolean
}

export const NEW_FILE_META: NovelFileMeta = { bom: false, eol: 'lf' }

/** The text a file's bytes mean, plus the facts needed to write it back the way it was. */
export function decodeBytes(buf: Buffer): DecodedFile {
  const bom = buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf
  const body = bom ? buf.subarray(3) : buf
  let readOnly = false
  let raw: string
  try {
    raw = new TextDecoder('utf-8', { fatal: true }).decode(body)
  } catch {
    readOnly = true
    raw = new TextDecoder('utf-8').decode(body)
  }
  const crlf = (raw.match(/\r\n/g) ?? []).length
  const lf = (raw.match(/\n/g) ?? []).length - crlf
  // The majority decides; a file without line breaks (or a tie) is LF.
  const eol = crlf > lf ? 'crlf' : 'lf'
  return { text: raw.replace(/\r\n/g, '\n'), hash: sha256(buf), meta: { bom, eol }, readOnly }
}

/** Text → bytes with the file's own BOM and line endings. */
export function encodeText(text: string, meta: NovelFileMeta): Buffer {
  let t = text.replace(/\r\n/g, '\n')
  if (meta.eol === 'crlf') t = t.replace(/\n/g, '\r\n')
  const body = Buffer.from(t, 'utf8')
  return meta.bom ? Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), body]) : body
}

/**
 * Every path under a project's .rulz folder goes through this before it is read, written, moved or deleted.
 * The .rulz subfolders (history, unsaved, trash, tmp) are app-owned and never need to be links, so any link
 * from .rulz down to the path is refused (even one that points inside the project), and so is a path whose
 * real location is outside the project. Paths without a .rulz part are not checked here (the store guards
 * user files with realInside). Throws; nothing has been touched when it does.
 */
export async function assertAppPath(path: string): Promise<void> {
  const full = resolve(path)
  const parts = full.split(sep)
  let idx = -1
  for (let i = parts.length - 1; i >= 0; i--) {
    if (parts[i].toLowerCase() === '.rulz') {
      idx = i
      break
    }
  }
  if (idx < 0) return
  const project = resolve(parts.slice(0, idx).join(sep) + sep)
  let cur = project
  for (const seg of parts.slice(idx)) {
    cur = join(cur, seg)
    let st
    try {
      st = await fs.lstat(cur)
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (code === 'ENOENT' || code === 'ENOTDIR') break
      throw err
    }
    if (st.isSymbolicLink()) throw new Error('The project\'s .rulz folder contains a link, so nothing was written there.')
  }
  if (!(await realInside(project, full))) throw new Error('The project\'s .rulz folder leads outside the project, so nothing was written there.')
}

/** Reads a file's bytes; null when it does not exist. Retries transient Windows locks. */
export async function readBytes(path: string): Promise<Buffer | null> {
  await assertAppPath(path)
  for (let attempt = 0; ; attempt++) {
    try {
      return await fs.readFile(path)
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (code === 'ENOENT' || code === 'ENOTDIR') return null
      if (!TRANSIENT.has(code) || attempt >= RETRY_WAITS.length) throw err
      await sleep(RETRY_WAITS[attempt])
    }
  }
}

/** Runs an fs step, retrying the transient Windows lock errors (about 750 ms in all). */
export async function retryTransient<T>(fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn()
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (!TRANSIENT.has(code) || attempt >= RETRY_WAITS.length) throw err
      await sleep(RETRY_WAITS[attempt])
    }
  }
}

/** The files the Writing page edits: style.md, bible entries and scenes. */
export const USER_PATH = /^(style\.md|bible\/[^/]+\.md|manuscript\/[^/]+\.md)$/i

const queues = new Map<string, Promise<unknown>>()
const keyOf = (path: string): string => (process.platform === 'win32' ? path.toLowerCase() : path)

/** Runs `fn` after every earlier task for the same path; a failed task does not stop the next one. */
export function serialized<T>(path: string, fn: () => Promise<T>): Promise<T> {
  const key = keyOf(path)
  const previous = queues.get(key) ?? Promise.resolve()
  const next = previous.catch(() => undefined).then(fn)
  queues.set(key, next)
  const cleanup = (): void => {
    if (queues.get(key) === next) queues.delete(key)
  }
  void next.then(cleanup, cleanup)
  return next
}

/**
 * Writes bytes via a temp file in `tmpDir` (inside the project's .rulz, never next to the user's files) and
 * a rename over `target`. Throws after about 750 ms of lock retries; the target is then unchanged.
 */
export async function writeBytesAtomic(target: string, tmpDir: string, bytes: Uint8Array, stillValid?: () => Promise<boolean>): Promise<boolean> {
  await assertAppPath(target)
  await assertAppPath(tmpDir)
  await fs.mkdir(dirname(target), { recursive: true })
  await fs.mkdir(tmpDir, { recursive: true })
  await assertAppPath(tmpDir)
  const tmp = join(tmpDir, `${randomBytes(6).toString('hex')}.tmp`)
  const fh = await fs.open(tmp, 'w')
  try {
    await fh.writeFile(bytes)
    // Without this a power cut can leave the renamed file empty.
    await fh.sync()
  } finally {
    await fh.close()
  }
  // The last look before the rename: if the target changed while the temp file was written, give up.
  // What remains is the few microseconds between this check and the rename, which no file API closes.
  if (stillValid && !(await stillValid().catch(() => false))) {
    await fs.rm(tmp, { force: true }).catch(() => undefined)
    return false
  }
  let source = tmp
  for (let attempt = 0; ; attempt++) {
    try {
      await fs.rename(source, target)
      return true
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (code === 'EXDEV' && source === tmp) {
        // The target is on another disk than .rulz (a folder linked onto another volume): carry the temp file
        // over to a hidden file next to the target, and rename that, which keeps the replace atomic.
        const sibling = join(dirname(target), `.${basename(target)}.rulz-tmp`)
        try {
          await fs.copyFile(tmp, sibling)
          const fh2 = await fs.open(sibling, 'r+')
          await fh2.sync().finally(() => fh2.close())
        } catch (copyErr) {
          await fs.rm(sibling, { force: true }).catch(() => undefined)
          await fs.rm(tmp, { force: true }).catch(() => undefined)
          throw copyErr
        }
        await fs.rm(tmp, { force: true }).catch(() => undefined)
        source = sibling
        continue
      }
      if (!TRANSIENT.has(code) || attempt >= RETRY_WAITS.length) {
        await fs.rm(source, { force: true }).catch(() => undefined)
        if (source !== tmp) await fs.rm(tmp, { force: true }).catch(() => undefined)
        throw err
      }
      await sleep(RETRY_WAITS[attempt])
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Paths coming over IPC (rule: resolved inside the project, never into .rulz\)
// ---------------------------------------------------------------------------------------------

const insensitive = process.platform === 'win32'
const same = (a: string, b: string): boolean => (insensitive ? a.toLowerCase() === b.toLowerCase() : a === b)

/**
 * True when `target` (which may not exist yet) stays inside the real folder after links are followed. The
 * deepest existing part of the path is resolved with realpath; a link that cannot be resolved (dangling) or
 * that leads outside counts as outside. Links that stay inside the folder are fine.
 */
export async function realInside(folder: string, target: string): Promise<boolean> {
  let root: string
  try {
    root = await fs.realpath(folder)
  } catch {
    return false
  }
  let cur = resolve(target)
  for (;;) {
    let exists = true
    try {
      await fs.lstat(cur)
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code ?? ''
      if (code !== 'ENOENT' && code !== 'ENOTDIR') return false
      exists = false
    }
    if (exists) {
      try {
        return isInside(root, await fs.realpath(cur))
      } catch {
        return false
      }
    }
    const up = dirname(cur)
    if (up === cur) return false
    cur = up
  }
}

/** True when `child` is `parent` or inside it (case-insensitive on Windows). */
export function isInside(parent: string, child: string): boolean {
  const p = resolve(parent)
  const c = resolve(child)
  if (same(p, c)) return true
  const prefix = p.endsWith(sep) ? p : p + sep
  return insensitive ? c.toLowerCase().startsWith(prefix.toLowerCase()) : c.startsWith(prefix)
}

/**
 * Checks a project-relative path from the renderer and returns it with forward slashes. Refuses absolute
 * paths, drive letters, "..", empty parts, alternate data streams (":") and anything inside .rulz\ unless
 * `allowRulz` says the path may start with that prefix (history snapshots).
 */
export function cleanRelPath(folder: string, rel: unknown, allowRulz?: string): string {
  if (typeof rel !== 'string' || !rel) throw new Error('The file path is missing.')
  if (rel.includes('\0') || rel.includes(':') || isAbsolute(rel) || /^[\\/]/.test(rel)) throw new Error(`The path "${rel}" is not inside the project.`)
  const parts = rel.replace(/\\/g, '/').split('/')
  if (parts.some((p) => p === '' || p === '.' || p === '..')) throw new Error(`The path "${rel}" is not inside the project.`)
  const clean = parts.join('/')
  if (same(parts[0], '.rulz') && !(allowRulz && clean.toLowerCase().startsWith(allowRulz.toLowerCase()))) {
    throw new Error('The .rulz folder is managed by the app.')
  }
  if (!isInside(folder, join(folder, ...parts))) throw new Error(`The path "${rel}" is not inside the project.`)
  return clean
}
