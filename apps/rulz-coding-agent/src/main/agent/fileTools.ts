// File tools for the model. Every path goes through the project path guard.
import { promises as fs, type Dirent } from 'node:fs'
import { dirname, join } from 'node:path'
import type { ChangedFile } from '@shared/types'
import { PathError, resolveInProject, toRel } from './pathGuard'

export interface ToolOutcome {
  text: string
  isError: boolean
  /** Files this call created, changed or deleted. */
  changed?: ChangedFile[]
}

/** Folders that are generated or huge: listed but never walked into. */
export const SKIP_DIRS = new Set([
  'node_modules',
  '.git',
  '.venv',
  'venv',
  '__pycache__',
  'dist',
  'out',
  '.next',
  '.cache',
  '.parcel-cache',
  'coverage',
  '.pytest_cache',
  '.mypy_cache'
])

const MAX_LIST_ENTRIES = 400
const MAX_READ_BYTES = 2 * 1024 * 1024
const MAX_READ_LINES = 2000
const MAX_SEARCH_FILE_BYTES = 1024 * 1024
const MAX_SEARCH_HITS = 200
const MAX_SCANNED = 50_000

const ok = (text: string, changed?: ChangedFile[]): ToolOutcome => ({ text, isError: false, changed })
const fail = (text: string): ToolOutcome => ({ text, isError: true })

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let v = bytes / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

/** Wildcard (* and ?) or plain-text name matcher, case-insensitive. */
export function nameMatcher(pattern: string): (name: string) => boolean {
  const p = pattern.trim().toLowerCase()
  if (!p) return () => true
  if (!/[*?]/.test(p)) return (n) => n.toLowerCase().includes(p)
  const re = new RegExp(`^${p.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.')}$`, 'i')
  return (n) => re.test(n)
}

/** Glob for search_text: "*.js", "src/**\/*.ts", or "*.{js,ts}". Matched against the relative path or the name. */
export function globMatcher(glob: string | undefined): (rel: string) => boolean {
  if (!glob?.trim()) return () => true
  const alts = glob
    .split(/,(?![^{]*\})/)
    .map((g) => g.trim())
    .filter(Boolean)
    .flatMap((g) => {
      const brace = g.match(/^(.*)\{([^}]+)\}(.*)$/)
      return brace ? brace[2].split(',').map((b) => brace[1] + b.trim() + brace[3]) : [g]
    })
  const res = alts.map((g) => {
    const body = g
      .replace(/\\/g, '/')
      .replace(/[.+^${}()|[\]]/g, '\\$&')
      .replace(/\*\*\//g, '\u0000')
      .replace(/\*\*/g, '.*')
      .replace(/\*/g, '[^/]*')
      .replace(/\?/g, '[^/]')
      .replace(/\u0000/g, '(?:.*/)?')
    return new RegExp(`^${body}$`, 'i')
  })
  return (rel) => {
    const name = rel.split('/').pop() ?? rel
    return res.some((re) => re.test(rel) || re.test(name))
  }
}

const clampInt = (v: unknown, def: number, min: number, max: number): number => {
  const n = Math.round(Number(v))
  return v !== undefined && v !== null && v !== '' && Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : def
}

function explain(err: unknown, rel: string): string {
  if (err instanceof PathError) return err.message
  const code = (err as NodeJS.ErrnoException).code
  const shown = rel || 'the project folder'
  if (code === 'ENOENT') return `"${shown}" does not exist.`
  if (code === 'ENOTDIR') return `"${shown}" is a file, not a folder.`
  if (code === 'EISDIR') return `"${shown}" is a folder, not a file.`
  if (code === 'EPERM' || code === 'EACCES' || code === 'EBUSY') return `"${shown}" is locked or access is denied (is it open in another program?).`
  return `Couldn't use "${shown}": ${(err as Error).message}`
}

async function readDir(dir: string): Promise<Dirent[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  return entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
}

/** Walks the project breadth-first, skipping generated folders. */
async function* walkFiles(start: string): AsyncGenerator<{ abs: string; entry: Dirent }> {
  let level = [start]
  let scanned = 0
  while (level.length) {
    const next: string[] = []
    for (const dir of level) {
      let entries: Dirent[]
      try {
        entries = await readDir(dir)
      } catch {
        continue
      }
      for (const entry of entries) {
        if (++scanned > MAX_SCANNED) return
        const abs = join(dir, entry.name)
        yield { abs, entry }
        if (entry.isDirectory() && !entry.isSymbolicLink() && !SKIP_DIRS.has(entry.name.toLowerCase())) next.push(abs)
      }
    }
    level = next
  }
}

export async function listFiles(root: string, args: { path?: unknown; depth?: unknown }): Promise<ToolOutcome> {
  let target
  try {
    target = await resolveInProject(root, args.path)
  } catch (e) {
    return fail(explain(e, String(args.path ?? '')))
  }
  const depth = clampInt(args.depth, 2, 1, 4)
  const lines: string[] = []
  let truncated = false

  const walk = async (dir: string, level: number): Promise<void> => {
    const entries = await readDir(dir)
    // The folder's file sizes are read together rather than one after another.
    const files = entries.filter((e) => !e.isDirectory()).slice(0, MAX_LIST_ENTRIES)
    const sizes = new Map(
      await Promise.all(files.map(async (e) => [e.name, await fs.stat(join(dir, e.name)).then((s) => formatSize(s.size), () => '?')] as const))
    )
    for (const e of entries) {
      if (lines.length >= MAX_LIST_ENTRIES) {
        truncated = true
        return
      }
      const indent = '  '.repeat(level - 1)
      const full = join(dir, e.name)
      if (e.isDirectory()) {
        const skipped = SKIP_DIRS.has(e.name.toLowerCase())
        lines.push(`${indent}${e.name}/${skipped ? '  (generated or installed files, not listed)' : ''}`)
        if (!skipped && level < depth) {
          try {
            await walk(full, level + 1)
          } catch (err) {
            lines.push(`${indent}  (${explain(err, toRel(root, full))})`)
          }
        }
      } else {
        lines.push(`${indent}${e.name}  (${sizes.get(e.name) ?? '?'})`)
      }
    }
  }

  try {
    const st = await fs.stat(target.abs)
    if (!st.isDirectory()) return fail(`"${target.rel}" is a file, not a folder. Use read_file to read it.`)
    await walk(target.abs, 1)
  } catch (e) {
    return fail(explain(e, target.rel))
  }
  const title = target.rel ? `${target.rel}/` : 'Project folder'
  if (!lines.length) return ok(`${title} is empty.`)
  const note = truncated ? `\n(Stopped at ${MAX_LIST_ENTRIES} entries; list a subfolder for more.)` : ''
  return ok(`${title}\n${lines.join('\n')}${note}`)
}

export async function findFiles(root: string, args: { pattern?: unknown; path?: unknown }): Promise<ToolOutcome> {
  if (typeof args.pattern !== 'string' || !args.pattern.trim()) return fail('A pattern is required, like "*.js" or "config".')
  let target
  try {
    target = await resolveInProject(root, args.path)
  } catch (e) {
    return fail(explain(e, String(args.path ?? '')))
  }
  const match = nameMatcher(args.pattern)
  const hits: string[] = []
  for await (const { abs, entry } of walkFiles(target.abs)) {
    if (!match(entry.name)) continue
    hits.push(toRel(root, abs) + (entry.isDirectory() ? '/' : ''))
    if (hits.length >= MAX_SEARCH_HITS) break
  }
  if (!hits.length) return ok(`No files or folders named like "${args.pattern}" were found.`)
  const more = hits.length >= MAX_SEARCH_HITS ? `\n(Showing the first ${MAX_SEARCH_HITS}.)` : ''
  return ok(hits.join('\n') + more)
}

function looksBinary(buf: Buffer): boolean {
  const n = Math.min(buf.length, 8000)
  for (let i = 0; i < n; i++) if (buf[i] === 0) return true
  return false
}

export async function searchText(
  root: string,
  args: { query?: unknown; path?: unknown; glob?: unknown }
): Promise<ToolOutcome> {
  if (typeof args.query !== 'string' || !args.query) return fail('Give the text to search for.')
  let target
  try {
    target = await resolveInProject(root, args.path)
  } catch (e) {
    return fail(explain(e, String(args.path ?? '')))
  }
  const needle = args.query.toLowerCase()
  const inGlob = globMatcher(typeof args.glob === 'string' ? args.glob : undefined)
  const hits: string[] = []

  /** One file's matching lines (at most MAX_SEARCH_HITS). */
  const searchFile = async (abs: string): Promise<string[]> => {
    const rel = toRel(root, abs)
    if (!inGlob(rel)) return []
    let buf: Buffer
    try {
      const st = await fs.stat(abs)
      if (st.size > MAX_SEARCH_FILE_BYTES) return []
      buf = await fs.readFile(abs)
    } catch {
      return []
    }
    if (looksBinary(buf)) return []
    const found: string[] = []
    const lines = buf.toString('utf8').split(/\r?\n/)
    for (let i = 0; i < lines.length && found.length < MAX_SEARCH_HITS; i++) {
      if (lines[i].toLowerCase().includes(needle)) {
        const line = lines[i].length > 300 ? `${lines[i].slice(0, 300)}…` : lines[i]
        found.push(`${rel}:${i + 1}: ${line.trim()}`)
      }
    }
    return found
  }
  // Files are read a few at a time; results keep the walk's order.
  const SEARCH_BATCH = 16
  const searchBatch = async (batch: string[]): Promise<void> => {
    for (const found of await Promise.all(batch.map(searchFile))) {
      hits.push(...found.slice(0, MAX_SEARCH_HITS - hits.length))
      if (hits.length >= MAX_SEARCH_HITS) return
    }
  }

  const st = await fs.stat(target.abs).catch(() => null)
  if (!st) return fail(`"${target.rel}" does not exist.`)
  if (st.isFile()) await searchBatch([target.abs])
  else {
    let batch: string[] = []
    for await (const { abs, entry } of walkFiles(target.abs)) {
      if (!entry.isFile()) continue
      batch.push(abs)
      if (batch.length < SEARCH_BATCH) continue
      await searchBatch(batch)
      batch = []
      if (hits.length >= MAX_SEARCH_HITS) break
    }
    if (batch.length && hits.length < MAX_SEARCH_HITS) await searchBatch(batch)
  }
  if (!hits.length) return ok(`"${args.query}" was not found.`)
  const more = hits.length >= MAX_SEARCH_HITS ? `\n(Showing the first ${MAX_SEARCH_HITS} matches.)` : ''
  return ok(hits.join('\n') + more)
}

export async function readFile(
  root: string,
  args: { path?: unknown; start_line?: unknown; end_line?: unknown }
): Promise<ToolOutcome> {
  let target
  try {
    target = await resolveInProject(root, args.path)
    const st = await fs.stat(target.abs)
    if (st.isDirectory()) return fail(`"${target.rel}" is a folder. Use list_files to see what is in it.`)
    if (st.size > MAX_READ_BYTES) return fail(`"${target.rel}" is too large to read (${formatSize(st.size)}).`)
    const buf = await fs.readFile(target.abs)
    if (looksBinary(buf)) return fail(`"${target.rel}" is not a text file (${formatSize(st.size)}), so it can't be read as text.`)
    const lines = buf.toString('utf8').replace(/^﻿/, '').split(/\r?\n/)
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop()
    const total = lines.length
    const start = clampInt(args.start_line, 1, 1, Math.max(1, total))
    const end = clampInt(args.end_line, Math.min(total, start + MAX_READ_LINES - 1), start, Math.min(total, start + MAX_READ_LINES - 1))
    if (!buf.length) return ok(`${target.rel} is empty.`)
    const width = String(end).length
    const body = lines
      .slice(start - 1, end)
      .map((l, i) => `${String(start + i).padStart(width)}| ${l}`)
      .join('\n')
    const range = start === 1 && end === total ? `${total} lines` : `lines ${start}-${end} of ${total}`
    const more = end < total ? `\n(More below: read again with start_line ${end + 1}.)` : ''
    return ok(`${target.rel} (${range}; the "N| " line numbers are not part of the file)\n${body}${more}`)
  } catch (e) {
    return fail(explain(e, target?.rel ?? String(args.path ?? '')))
  }
}

/** Models sometimes pass a JSON file's content as an object instead of text. */
function contentText(content: unknown): string | null {
  if (typeof content === 'string') return content
  if (content && typeof content === 'object') return JSON.stringify(content, null, 2) + '\n'
  return null
}

export async function writeFile(root: string, args: { path?: unknown; content?: unknown }): Promise<ToolOutcome> {
  let target
  try {
    target = await resolveInProject(root, args.path, { write: true })
    const content = contentText(args.content)
    if (content === null) return fail('write_file needs "content": the full text of the file.')
    const existed = await fs.stat(target.abs).then((s) => {
      if (s.isDirectory()) throw Object.assign(new Error('folder'), { code: 'EISDIR' })
      return true
    }, () => false)
    await fs.mkdir(dirname(target.abs), { recursive: true })
    await fs.writeFile(target.abs, content, 'utf8')
    const lines = content ? content.split('\n').length - (content.endsWith('\n') ? 1 : 0) : 0
    return ok(`${existed ? 'Updated' : 'Created'} ${target.rel} (${lines} lines).`, [{ path: target.rel, change: existed ? 'changed' : 'created' }])
  } catch (e) {
    return fail(explain(e, target?.rel ?? String(args.path ?? '')))
  }
}

function countOccurrences(text: string, needle: string): number[] {
  const at: number[] = []
  let i = text.indexOf(needle)
  while (i >= 0) {
    at.push(i)
    i = text.indexOf(needle, i + needle.length)
  }
  return at
}

const lineOf = (text: string, index: number): number => text.slice(0, index).split('\n').length

export async function editFile(
  root: string,
  args: { path?: unknown; old_text?: unknown; new_text?: unknown }
): Promise<ToolOutcome> {
  let target
  try {
    target = await resolveInProject(root, args.path, { write: true })
    if (typeof args.old_text !== 'string' || !args.old_text) {
      return fail('edit_file needs "old_text": an exact piece of the file to replace. To create or overwrite a whole file, use write_file.')
    }
    if (typeof args.new_text !== 'string') return fail('edit_file needs "new_text": the text to put in place of old_text (it can be empty).')
    const text = await fs.readFile(target.abs, 'utf8')
    let oldText = args.old_text
    let newText = args.new_text
    // Match the file's line endings.
    if (text.includes('\r\n') && !oldText.includes('\r\n')) {
      oldText = oldText.replace(/\n/g, '\r\n')
      newText = newText.replace(/\r?\n/g, '\r\n')
    }
    const found = countOccurrences(text, oldText)
    if (found.length === 0) {
      const first = args.old_text.split('\n').find((l) => l.trim())?.trim()
      const near = first
        ? text
            .split(/\r?\n/)
            .map((l, i) => (l.includes(first) ? i + 1 : 0))
            .filter(Boolean)
            .slice(0, 5)
        : []
      const hint = near.length ? ` Its first line appears at line ${near.join(', ')}, so the rest must differ (spaces, tabs or other lines).` : ''
      return fail(`old_text was not found in ${target.rel}.${hint} Read the file again with read_file and copy the text exactly, without the line numbers.`)
    }
    if (found.length > 1) {
      const lines = found.map((i) => lineOf(text, i)).slice(0, 8)
      return fail(`old_text appears ${found.length} times in ${target.rel} (lines ${lines.join(', ')}). Include more of the surrounding lines so it matches only one place.`)
    }
    const updated = text.slice(0, found[0]) + newText + text.slice(found[0] + oldText.length)
    await fs.writeFile(target.abs, updated, 'utf8')
    return ok(`Edited ${target.rel} at line ${lineOf(text, found[0])}.`, [{ path: target.rel, change: 'changed' }])
  } catch (e) {
    return fail(explain(e, target?.rel ?? String(args.path ?? '')))
  }
}

/** Moves a file or folder to the Recycle Bin (never deletes permanently). */
export async function deletePath(
  root: string,
  args: { path?: unknown },
  trash: (abs: string) => Promise<void>
): Promise<ToolOutcome> {
  let target
  try {
    target = await resolveInProject(root, args.path, { write: true })
    const st = await fs.stat(target.abs)
    await trash(target.abs)
    return ok(`Moved ${target.rel}${st.isDirectory() ? '/' : ''} to the Recycle Bin.`, [{ path: target.rel, change: 'deleted' }])
  } catch (e) {
    return fail(explain(e, target?.rel ?? String(args.path ?? '')))
  }
}
