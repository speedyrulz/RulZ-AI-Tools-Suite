import { promises as fs, watch, type FSWatcher } from 'node:fs'
import { basename, dirname, join, relative, sep } from 'node:path'
import type { DraftKind, GgufInfo, ModelEntry, SidecarFile } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { readGgufHeader, toGgufInfo } from './gguf'

export interface ScannedFile {
  path: string
  sizeBytes: number
  mtimeMs: number
  info: GgufInfo | null
  error?: string
}

export type FileKind = 'model' | 'mmproj' | 'draft' | 'adapter' | 'shard'

const SHARD_RE = /-(\d{5})-of-(\d{5})\.gguf$/i

/** Draft heads are named after their format, e.g. "mtp-<model>.gguf" or "<model>-eagle3.gguf". */
export function draftKind(fileName: string): DraftKind | undefined {
  const base = fileName.replace(/\.gguf$/i, '').toLowerCase()
  for (const kind of ['mtp', 'eagle3', 'dflash', 'dspark'] as const) {
    if (new RegExp(`(^|[-_. ])${kind}([-_. ]|$)`).test(base)) return kind
  }
  return undefined
}

export function classifyFile(fileName: string, info: GgufInfo | null): FileKind {
  const shard = fileName.match(SHARD_RE)
  if (shard && Number(shard[1]) !== 1) return 'shard'
  if (info?.architecture === 'clip' || info?.type === 'mmproj' || /mmproj/i.test(fileName)) return 'mmproj'
  // A draft head is tiny next to its model; a full model with "MTP" in its name stays a model.
  if (draftKind(fileName) && (/^(mtp|eagle3|dflash|dspark)[-_. ]/i.test(fileName) || info?.architecture?.endsWith('-assistant'))) return 'draft'
  if (info?.type === 'adapter') return 'adapter'
  return 'model'
}

export function displayName(fileName: string): string {
  return fileName.replace(SHARD_RE, '').replace(/\.gguf$/i, '')
}

function toId(root: string, path: string): string {
  return relative(root, path).split(sep).join('/')
}

/** Groups scanned files into models with their shards and same-folder sidecars. Pure; unit-tested. */
export function groupModels(root: string, files: ScannedFile[]): ModelEntry[] {
  const byDir = new Map<string, ScannedFile[]>()
  for (const f of files) {
    const d = dirname(f.path)
    byDir.set(d, [...(byDir.get(d) ?? []), f])
  }

  const models: ModelEntry[] = []
  for (const [dir, dirFiles] of byDir) {
    const kinds = dirFiles.map((f) => ({ f, kind: classifyFile(basename(f.path), f.info) }))
    const sidecar = (kind: FileKind): SidecarFile | undefined => {
      const hit = kinds
        .filter((k) => k.kind === kind)
        .sort((a, b) => basename(a.f.path).localeCompare(basename(b.f.path)))[0]
      if (!hit) return undefined
      const out: SidecarFile = { path: hit.f.path, sizeBytes: hit.f.sizeBytes }
      if (kind === 'draft') out.kind = draftKind(basename(hit.f.path))
      return out
    }
    const mmproj = sidecar('mmproj')
    const draft = sidecar('draft')

    for (const { f, kind } of kinds) {
      if (kind !== 'model') continue
      const name = basename(f.path)
      const shardMatch = name.match(SHARD_RE)
      let shardFiles = [f]
      if (shardMatch) {
        const prefix = name.slice(0, shardMatch.index)
        shardFiles = dirFiles
          .filter((o) => {
            const m = basename(o.path).match(SHARD_RE)
            return m && basename(o.path).slice(0, m.index) === prefix && m[2] === shardMatch[2]
          })
          .sort((a, b) => a.path.localeCompare(b.path))
      }
      const folder = toId(root, dir)
      models.push({
        id: toId(root, f.path),
        name: displayName(name),
        folder: folder === '.' ? '' : folder,
        path: f.path,
        files: shardFiles.map((s) => s.path),
        sizeBytes: shardFiles.reduce((sum, s) => sum + s.sizeBytes, 0),
        mmproj,
        draft,
        info: f.info,
        error: f.error,
        modifiedAt: Math.max(...shardFiles.map((s) => s.mtimeMs))
      })
    }
  }
  return models.sort((a, b) => a.id.localeCompare(b.id))
}

async function walk(dir: string, depth: number, out: { path: string; size: number; mtimeMs: number }[]): Promise<void> {
  if (depth > 8) return
  let entries
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const e of entries) {
    if (e.name.startsWith('.')) continue
    const full = join(dir, e.name)
    let isDir = e.isDirectory()
    let isFile = e.isFile()
    if (e.isSymbolicLink()) {
      const st = await fs.stat(full).catch(() => null)
      isDir = !!st?.isDirectory()
      isFile = !!st?.isFile()
    }
    // Some folders are themselves named "*.gguf", so the isFile check matters.
    if (isDir) await walk(full, depth + 1, out)
    else if (isFile && /\.gguf$/i.test(e.name)) {
      const st = await fs.stat(full).catch(() => null)
      if (st) out.push({ path: full, size: st.size, mtimeMs: st.mtimeMs })
    }
  }
}

/** Bump when GgufInfo gains fields so cached metadata is re-read. */
const CACHE_VERSION = 1

interface CacheEntry {
  v?: number
  size: number
  mtimeMs: number
  info: GgufInfo | null
  error?: string
}

/** Scans the models folder, caches GGUF metadata, and watches for changes. */
export class ModelLibrary {
  private models: ModelEntry[] = []
  private cache: Record<string, CacheEntry> = {}
  private cacheLoaded = false
  private watcher: FSWatcher | null = null
  private watchedDir = ''
  private debounce: NodeJS.Timeout | null = null
  private scanning: Promise<ModelEntry[]> | null = null

  constructor(
    private readonly cachePath: string,
    private readonly getRoot: () => string,
    private readonly onChange: (models: ModelEntry[]) => void
  ) {}

  list(): ModelEntry[] {
    return this.models
  }

  get(id: string): ModelEntry | undefined {
    return this.models.find((m) => m.id === id)
  }

  scan(): Promise<ModelEntry[]> {
    this.scanning ??= this.doScan().finally(() => (this.scanning = null))
    return this.scanning
  }

  private async doScan(): Promise<ModelEntry[]> {
    if (!this.cacheLoaded) {
      this.cache = await readJson(this.cachePath, {})
      this.cacheLoaded = true
    }
    const root = this.getRoot()
    const found: { path: string; size: number; mtimeMs: number }[] = []
    await walk(root, 0, found)

    let dirty = false
    const seen = new Set<string>()
    const files: ScannedFile[] = []
    for (const f of found) {
      seen.add(f.path)
      let entry = this.cache[f.path]
      if (!entry || entry.v !== CACHE_VERSION || entry.size !== f.size || entry.mtimeMs !== f.mtimeMs) {
        const shard = basename(f.path).match(SHARD_RE)
        if (shard && Number(shard[1]) !== 1) {
          entry = { v: CACHE_VERSION, size: f.size, mtimeMs: f.mtimeMs, info: null }
        } else {
          try {
            entry = { v: CACHE_VERSION, size: f.size, mtimeMs: f.mtimeMs, info: toGgufInfo(readGgufHeader(f.path).kv, basename(f.path)) }
          } catch (err) {
            entry = { v: CACHE_VERSION, size: f.size, mtimeMs: f.mtimeMs, info: null, error: (err as Error).message }
          }
        }
        this.cache[f.path] = entry
        dirty = true
        await new Promise((r) => setImmediate(r))
      }
      files.push({ path: f.path, sizeBytes: f.size, mtimeMs: f.mtimeMs, info: entry.info, error: entry.error })
    }
    for (const p of Object.keys(this.cache)) {
      if (!seen.has(p) && p.startsWith(root)) {
        delete this.cache[p]
        dirty = true
      }
    }
    if (dirty) await writeJsonAtomic(this.cachePath, this.cache)

    this.models = groupModels(root, files)
    this.ensureWatcher(root)
    this.onChange(this.models)
    return this.models
  }

  private ensureWatcher(root: string): void {
    if (this.watcher && this.watchedDir === root) return
    this.watcher?.close()
    this.watcher = null
    this.watchedDir = root
    try {
      this.watcher = watch(root, { recursive: true }, (_event, file) => {
        // Ignore non-GGUF files (partial downloads, notes); folders have no extension.
        const name = String(file ?? '')
        if (name && !/\.gguf$/i.test(name) && /\.[a-z0-9]{1,10}$/i.test(name)) return
        if (this.debounce) clearTimeout(this.debounce)
        this.debounce = setTimeout(() => void this.scan(), 2000)
      })
      this.watcher.on('error', () => {
        this.watcher?.close()
        this.watcher = null
      })
    } catch {
      this.watcher = null
    }
  }

  dispose(): void {
    this.watcher?.close()
    if (this.debounce) clearTimeout(this.debounce)
  }
}
