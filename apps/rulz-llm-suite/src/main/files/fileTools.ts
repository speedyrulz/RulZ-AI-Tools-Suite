// Built-in, read-only file tools: let a model see what is in a folder and find files by name,
// so it can hand exact paths to tools such as REA instead of guessing them.
import { promises as fs, type Dirent } from 'node:fs'
import { isAbsolute, join, relative, resolve } from 'node:path'

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
  }
]

const MAX_LIST_ENTRIES = 500
const MAX_SCANNED = 200_000
const SEARCH_TIME_MS = 15_000

/** Expands %VAR% and checks the path is absolute. */
export function expandPath(input: unknown, env: NodeJS.ProcessEnv = process.env): string {
  if (typeof input !== 'string' || !input.trim()) throw new Error('A path is required.')
  const expanded = input.trim().replace(/^["']|["']$/g, '').replace(/%([^%]+)%/g, (m, name: string) => {
    const key = Object.keys(env).find((k) => k.toLowerCase() === name.toLowerCase())
    return key ? (env[key] ?? m) : m
  })
  if (!isAbsolute(expanded)) throw new Error(`Use an absolute path (for example "C:\\Program Files\\..."), not "${input}".`)
  return resolve(expanded)
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

async function readDir(dir: string): Promise<Dirent[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  return entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
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

export async function findFiles(args: {
  root?: unknown
  pattern?: unknown
  max_depth?: unknown
  limit?: unknown
}): Promise<{ text: string; isError: boolean }> {
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

export function runFileTool(name: string, args: Record<string, unknown>): Promise<{ text: string; isError: boolean }> {
  if (name === 'list_directory') return listDirectory(args)
  if (name === 'find_files') return findFiles(args)
  return Promise.resolve({ text: `Unknown tool "${name}".`, isError: true })
}
