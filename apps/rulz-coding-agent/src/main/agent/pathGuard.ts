// Keeps every file tool inside the project folder: no "..", no absolute paths elsewhere, and no
// links (symlinks, junctions) that lead out of it.
import { promises as fs } from 'node:fs'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'

export class PathError extends Error {}

export interface ResolvedPath {
  /** Absolute path on disk. */
  abs: string
  /** Path relative to the project, with forward slashes ("" for the project folder itself). */
  rel: string
}

const isWindows = process.platform === 'win32'
const norm = (p: string): string => (isWindows ? p.toLowerCase() : p)

/** True when `child` is `root` or inside it. Both must be absolute and resolved. */
export function isInside(root: string, child: string): boolean {
  const r = norm(resolve(root))
  const c = norm(resolve(child))
  if (c === r) return true
  return c.startsWith(r.endsWith(sep) ? r : r + sep)
}

export function toRel(root: string, abs: string): string {
  return relative(root, abs).split(sep).join('/')
}

/** Nearest part of `abs` that exists, resolved through links. */
async function realExisting(abs: string): Promise<string> {
  let cur = abs
  const rest: string[] = []
  for (;;) {
    try {
      const real = await fs.realpath(cur)
      return rest.length ? resolve(real, ...rest.reverse()) : real
    } catch (e) {
      const code = (e as NodeJS.ErrnoException).code
      if (code !== 'ENOENT' && code !== 'ENOTDIR') throw e
      const parent = dirname(cur)
      if (parent === cur) return abs
      rest.push(cur.slice(parent.length).replace(/^[\\/]+/, ''))
      cur = parent
    }
  }
}

/**
 * Resolves a path the model gave against the project folder and checks it stays inside.
 * An empty path or "." means the project folder itself.
 */
export async function resolveInProject(
  root: string,
  input: unknown,
  opts: { write?: boolean } = {}
): Promise<ResolvedPath> {
  if (input !== undefined && input !== null && typeof input !== 'string') throw new PathError('The path must be text, like "src/main.js".')
  const raw = (input ?? '').trim().replace(/^["']|["']$/g, '')
  const abs = raw === '' || raw === '.' || raw === './' ? resolve(root) : isAbsolute(raw) ? resolve(raw) : resolve(root, raw)
  const outside = `"${raw}" is outside the project folder. Only paths inside the project can be used, like "src/main.js".`
  if (!isInside(root, abs)) throw new PathError(outside)

  const realRoot = await fs.realpath(root).catch(() => resolve(root))
  const real = await realExisting(abs)
  if (!isInside(realRoot, real)) throw new PathError(`"${raw}" leads outside the project folder through a link, so it can't be used.`)

  const rel = toRel(root, abs)
  if (opts.write) {
    const parts = rel.toLowerCase().split('/')
    if (parts.includes('.git')) throw new PathError('Files inside .git folders are managed by git and must not be edited directly.')
    if (rel === '') throw new PathError('Give a file path inside the project, not the project folder itself.')
  }
  return { abs, rel }
}
