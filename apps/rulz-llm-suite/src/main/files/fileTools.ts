// Built-in file tools: let a model see what is in a folder and find files by name, so it can hand
// exact paths to tools such as REA instead of guessing them. Clone sessions also get write_file and
// read_file, which only work inside the session's output folder (and, for reading, its target).
import { promises as fs, type Dirent } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { unquotePath } from '@shared/re'
import type { FileToolScope } from '@shared/types'

export const FILE_TOOLS_SERVER = 'rulz'

export interface FileToolDef {
  name: string
  description: string
  inputSchema: unknown
}

export const FILE_TOOL_DEFS: FileToolDef[] = [
  {
    name: 'list_directory',
    description:
      'List the files and folders in a folder on this computer (read-only). Folders end with "\\". Shows file sizes. ' +
      "Use it whenever you have a folder but need a file path (for example a program's .exe) instead of guessing names.",
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Absolute folder path, e.g. "C:\\Program Files\\Some App". %VARIABLES% are expanded.' },
        depth: { type: 'integer', description: 'How many folder levels to show: 1 (default) to 3.' }
      },
      required: ['path']
    }
  },
  {
    name: 'find_files',
    description:
      'Find files or folders by name under a folder on this computer (read-only). The pattern is a wildcard such as "*.exe" ' +
      'or "*model*", or plain text matched anywhere in the name. Case-insensitive.',
    inputSchema: {
      type: 'object',
      properties: {
        root: { type: 'string', description: 'Absolute folder to search in. %VARIABLES% are expanded.' },
        pattern: { type: 'string', description: 'Name pattern, e.g. "*.exe", "*.qml", "config".' },
        max_depth: { type: 'integer', description: 'How deep to search (default 8, at most 20).' },
        limit: { type: 'integer', description: 'Maximum results (default 100, at most 500).' }
      },
      required: ['root', 'pattern']
    }
  },
  {
    name: 'write_file',
    description:
      "Write a text file in this clone session's output folder (and only there). Folders are created as needed. " +
      'Use a path relative to the output folder, e.g. "src/parser.c". An existing file is replaced, unless append is true.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path relative to the output folder (an absolute path inside it also works).' },
        content: { type: 'string', description: 'The text to write.' },
        append: { type: 'boolean', description: 'Add to the end of the file instead of replacing it (for writing a large file in parts).' }
      },
      required: ['path', 'content']
    }
  },
  {
    name: 'read_file',
    description:
      "Read a text file in this clone session's output folder, or inside the analysed target's folder. " +
      'Relative paths are relative to the output folder. Long files are shown in parts: pass offset (first line, from 1) and limit.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path, relative to the output folder or absolute.' },
        offset: { type: 'integer', description: 'First line to show, from 1 (default 1).' },
        limit: { type: 'integer', description: 'Lines to show (default 400, at most 2000).' }
      },
      required: ['path']
    }
  }
]

/** Tools that only exist in clone sessions (they need a FileToolScope). */
export const SCOPED_FILE_TOOLS = new Set(['write_file', 'read_file'])

const MAX_LIST_ENTRIES = 500
const MAX_SCANNED = 200_000
const SEARCH_TIME_MS = 15_000

/**
 * Network shares (\\host\share, //host) and device paths (\\?\, \\.\). Opening a share makes
 * Windows sign in to that host with the user's NTLM credentials, so a model told to read one
 * could leak them; these tools are for this computer's own files.
 */
const NETWORK_OR_DEVICE = /^[\\/]{2}/

/** Expands %VAR% and checks the path is absolute and on this computer. */
export function expandPath(input: unknown, env: NodeJS.ProcessEnv = process.env): string {
  if (typeof input !== 'string' || !input.trim()) throw new Error('A path is required.')
  const expanded = input.trim().replace(/^["']|["']$/g, '').replace(/%([^%]+)%/g, (m, name: string) => {
    const key = Object.keys(env).find((k) => k.toLowerCase() === name.toLowerCase())
    return key ? (env[key] ?? m) : m
  })
  const local = `Network and device paths such as "\\\\server\\share" can't be used; give a folder on this computer, not "${input}".`
  if (NETWORK_OR_DEVICE.test(expanded)) throw new Error(local)
  if (!isAbsolute(expanded)) throw new Error(`Use an absolute path (for example "C:\\Program Files\\..."), not "${input}".`)
  const full = resolve(expanded)
  if (NETWORK_OR_DEVICE.test(full)) throw new Error(local)
  return full
}

/** Wildcard (* and ?) or plain-text name matcher, case-insensitive. */
export function nameMatcher(pattern: string): (name: string) => boolean {
  const p = pattern.trim().toLowerCase()
  if (!p) return () => true
  if (!/[*?]/.test(p)) return (n) => n.toLowerCase().includes(p)
  const re = new RegExp(`^${p.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.')}$`, 'i')
  return (n) => re.test(n)
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = bytes / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

const clampInt = (v: unknown, def: number, min: number, max: number): number => {
  const n = Math.round(Number(v))
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : def
}

/** One shared collator: localeCompare per comparison is far slower on big folders. "file2" sorts before "file10". */
const collator = new Intl.Collator(undefined, { numeric: true })
/** Above this many entries the names are left in the order Windows returns them (alphabetical on NTFS). */
const MAX_SORTED = 5000

/** Folders first, then by name. */
async function readDir(dir: string): Promise<Dirent[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const dirs = entries.filter((e) => e.isDirectory())
  const files = entries.filter((e) => !e.isDirectory())
  if (entries.length <= MAX_SORTED) {
    const byName = (a: Dirent, b: Dirent): number => collator.compare(a.name, b.name)
    dirs.sort(byName)
    files.sort(byName)
  }
  return [...dirs, ...files]
}

async function sizeOf(path: string): Promise<string> {
  try {
    return formatSize((await fs.stat(path)).size)
  } catch {
    return '?'
  }
}

function explain(err: unknown, path: string): string {
  const code = (err as NodeJS.ErrnoException).code
  if (code === 'ENOENT') return `"${path}" does not exist.`
  if (code === 'ENOTDIR') return `"${path}" is a file, not a folder.`
  if (code === 'EPERM' || code === 'EACCES') return `Access to "${path}" is denied.`
  return `Couldn't read "${path}": ${(err as Error).message}`
}

export async function listDirectory(args: { path?: unknown; depth?: unknown }): Promise<{ text: string; isError: boolean }> {
  let root: string
  try {
    root = expandPath(args.path)
  } catch (e) {
    return { text: (e as Error).message, isError: true }
  }
  const depth = clampInt(args.depth, 1, 1, 3)
  const lines: string[] = []
  let truncated = false

  const walk = async (dir: string, level: number): Promise<void> => {
    let entries: Dirent[]
    try {
      entries = await readDir(dir)
    } catch (e) {
      if (level === 1) throw e
      lines.push(`${'  '.repeat(level - 1)}(${explain(e, dir)})`)
      return
    }
    for (const e of entries) {
      if (lines.length >= MAX_LIST_ENTRIES) {
        truncated = true
        return
      }
      const indent = '  '.repeat(level - 1)
      const full = join(dir, e.name)
      if (e.isDirectory()) {
        lines.push(`${indent}${e.name}\\`)
        if (level < depth) await walk(full, level + 1)
      } else {
        lines.push(`${indent}${e.name}  (${await sizeOf(full)})`)
      }
    }
  }

  try {
    await walk(root, 1)
  } catch (e) {
    return { text: explain(e, root), isError: true }
  }
  if (!lines.length) return { text: `${root} is empty.`, isError: false }
  const note = truncated ? `\n(Stopped at ${MAX_LIST_ENTRIES} entries; list a subfolder or use find_files.)` : ''
  return { text: `${root}\n${lines.join('\n')}${note}`, isError: false }
}

export async function findFiles(
  args: {
    root?: unknown
    pattern?: unknown
    max_depth?: unknown
    limit?: unknown
  },
  signal?: AbortSignal
): Promise<{ text: string; isError: boolean }> {
  let root: string
  try {
    root = expandPath(args.root)
  } catch (e) {
    return { text: (e as Error).message, isError: true }
  }
  if (typeof args.pattern !== 'string' || !args.pattern.trim()) return { text: 'A pattern is required, e.g. "*.exe".', isError: true }
  const match = nameMatcher(args.pattern)
  const maxDepth = clampInt(args.max_depth, 8, 1, 20)
  const limit = clampInt(args.limit, 100, 1, 500)
  const deadline = Date.now() + SEARCH_TIME_MS
  const hits: string[] = []
  let scanned = 0
  let skipped = 0
  let stopped = ''

  try {
    const st = await fs.stat(root)
    if (!st.isDirectory()) return { text: `"${root}" is a file, not a folder.`, isError: true }
  } catch (e) {
    return { text: explain(e, root), isError: true }
  }

  // Breadth-first, so shallow matches (usually the interesting ones) come first.
  let level: string[] = [root]
  for (let d = 1; d <= maxDepth && level.length && !stopped; d++) {
    const next: string[] = []
    for (const dir of level) {
      if (signal?.aborted) stopped = 'the call was stopped'
      if (stopped) break
      let entries: Dirent[]
      try {
        entries = await readDir(dir)
      } catch {
        skipped++
        continue
      }
      for (const e of entries) {
        if (++scanned > MAX_SCANNED) {
          stopped = `scanned ${MAX_SCANNED.toLocaleString('en-US')} entries`
          break
        }
        const full = join(dir, e.name)
        if (match(e.name)) {
          hits.push(e.isDirectory() ? `${relative(root, full)}\\` : `${relative(root, full)}  (${await sizeOf(full)})`)
          if (hits.length >= limit) {
            stopped = `reached the limit of ${limit} results`
            break
          }
        }
        if (e.isDirectory() && !e.isSymbolicLink()) next.push(full)
      }
      if (!stopped && Date.now() > deadline) stopped = `searched for ${SEARCH_TIME_MS / 1000} seconds`
    }
    level = next
  }

  const notes: string[] = []
  if (stopped) notes.push(`Stopped early: ${stopped}. Narrow the folder or pattern for more.`)
  if (skipped) notes.push(`${skipped} folder${skipped === 1 ? '' : 's'} could not be read.`)
  const tail = notes.length ? `\n(${notes.join(' ')})` : ''
  if (!hits.length) return { text: `No names matching "${args.pattern}" under ${root}.${tail}`, isError: false }
  return { text: `Under ${root}:\n${hits.join('\n')}${tail}`, isError: false }
}

const MAX_WRITE_CHARS = 1_000_000
const MAX_READ_BYTES = 4 * 1024 * 1024

const isInside = (root: string, path: string): boolean => {
  const rel = relative(root, path)
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))
}

/** The real path of `path`, or of its nearest existing parent with the rest added (links resolved). */
async function realish(path: string): Promise<string> {
  let cur = path
  const rest: string[] = []
  for (;;) {
    try {
      return join(await fs.realpath(cur), ...rest.reverse())
    } catch {
      const up = dirname(cur)
      if (up === cur) return path
      rest.push(relative(up, cur))
      cur = up
    }
  }
}

/**
 * Resolves a path for the scoped tools: relative paths are relative to the output folder. Returns
 * the full path, or an error when it is outside every allowed root (links included).
 */
export async function resolveScoped(input: unknown, roots: string[], base: string): Promise<string> {
  if (typeof input !== 'string' || !input.trim()) throw new Error('A path is required.')
  const raw = input.trim().replace(/^["']|["']$/g, '')
  const full = isAbsolute(raw) ? expandPath(raw) : resolve(base, raw)
  const real = await realish(full)
  for (const root of roots) {
    const r = resolve(root)
    if (isInside(r, full) && isInside(await realish(r), real)) return full
  }
  throw new Error(`"${input}" is outside the output folder (${base}). Use a path inside it, such as "src/main.c".`)
}

export async function writeFile(args: { path?: unknown; content?: unknown; append?: unknown }, scope: FileToolScope): Promise<{ text: string; isError: boolean }> {
  let full: string
  try {
    full = await resolveScoped(args.path, [scope.writeRoot], resolve(scope.writeRoot))
  } catch (e) {
    return { text: (e as Error).message, isError: true }
  }
  if (typeof args.content !== 'string') return { text: 'content must be text.', isError: true }
  if (args.content.length > MAX_WRITE_CHARS) return { text: `That is ${args.content.length.toLocaleString('en-US')} characters; write at most ${MAX_WRITE_CHARS.toLocaleString('en-US')} per call (use append for the rest).`, isError: true }
  if (full === resolve(scope.writeRoot)) return { text: 'Give a file name, not the output folder itself.', isError: true }
  // The output folder may sit next to (or around) the target; never overwrite what is being analysed,
  // also not through another name for it (a junction or link, or a short name like TARGET~1).
  if (scope.target) {
    const target = resolve(scope.target)
    if (isInside(target, full) || isInside(await realish(target), await realish(full))) {
      return { text: 'That path is part of the target being analysed; write the clone somewhere else in the output folder.', isError: true }
    }
  }
  try {
    await fs.mkdir(dirname(full), { recursive: true })
    const st = await fs.stat(full).catch(() => null)
    if (st?.isDirectory()) return { text: `"${full}" is a folder.`, isError: true }
    if (args.append === true) await fs.appendFile(full, args.content, 'utf8')
    else await fs.writeFile(full, args.content, 'utf8')
    const size = (await fs.stat(full)).size
    const lines = args.content.split('\n').length
    return { text: `${args.append === true ? 'Added' : 'Wrote'} ${lines} line${lines === 1 ? '' : 's'} ${args.append === true ? 'to' : 'in'} ${relative(resolve(scope.writeRoot), full)} (now ${formatSize(size)}).`, isError: false }
  } catch (e) {
    return { text: explain(e, full), isError: true }
  }
}

export async function readFile(args: { path?: unknown; offset?: unknown; limit?: unknown }, scope: FileToolScope): Promise<{ text: string; isError: boolean }> {
  let full: string
  try {
    const target = resolve(scope.target)
    const st = await fs.stat(target).catch(() => null)
    // A target that is a file: its own folder is readable (config files and resources next to it).
    const targetRoot = st && !st.isDirectory() ? dirname(target) : target
    full = await resolveScoped(args.path, [scope.writeRoot, ...(scope.target ? [targetRoot] : [])], resolve(scope.writeRoot))
  } catch (e) {
    return { text: (e as Error).message, isError: true }
  }
  try {
    const st = await fs.stat(full)
    if (st.isDirectory()) return { text: `"${full}" is a folder; use list_directory.`, isError: true }
    if (st.size > MAX_READ_BYTES) return { text: `"${full}" is ${formatSize(st.size)}, too big to read as text; use the REA tools.`, isError: true }
    const buf = await fs.readFile(full)
    if (buf.subarray(0, 8000).includes(0)) return { text: `"${full}" is a binary file; use the REA tools for it.`, isError: true }
    const lines = buf.toString('utf8').split(/\r?\n/)
    const offset = clampInt(args.offset, 1, 1, Math.max(1, lines.length))
    const limit = clampInt(args.limit, 400, 1, 2000)
    const part = lines.slice(offset - 1, offset - 1 + limit)
    const end = offset - 1 + part.length
    const head = lines.length > part.length ? `${full} (lines ${offset}-${end} of ${lines.length})\n` : `${full}\n`
    return { text: head + part.map((l, i) => `${String(offset + i).padStart(5)}  ${l}`).join('\n'), isError: false }
  } catch (e) {
    return { text: explain(e, full), isError: true }
  }
}

export function runFileTool(
  name: string,
  args: Record<string, unknown>,
  signal?: AbortSignal,
  scope?: FileToolScope
): Promise<{ text: string; isError: boolean }> {
  if (name === 'list_directory') return listDirectory(args)
  if (name === 'find_files') return findFiles(args, signal)
  if (SCOPED_FILE_TOOLS.has(name) && !scope?.writeRoot) return Promise.resolve({ text: `${name} only works in a clone session.`, isError: true })
  // A pasted path may keep its quotes; "C:\app" in quotes resolves as a relative path, and the
  // guard against writing to the target would then miss it.
  const clean = scope && { ...scope, writeRoot: unquotePath(scope.writeRoot), ...(scope.target ? { target: unquotePath(scope.target) } : {}) }
  if (name === 'write_file') return writeFile(args, clean!)
  if (name === 'read_file') return readFile(args, clean!)
  return Promise.resolve({ text: `Unknown tool "${name}".`, isError: true })
}
