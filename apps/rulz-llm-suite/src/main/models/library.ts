import { promises as fs, watch, type FSWatcher } from 'node:fs'
import { basename, dirname, join, relative, sep } from 'node:path'
import type { GgufInfo, ModelEntry, SidecarFile } from '@shared/types'
import { draftKind } from '@shared/draft'
export { draftKind }
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { QUANT_TAGS, readGgufHeaderAsync, toGgufInfo } from './gguf'

export interface ScannedFile {
  path: string
  sizeBytes: number
  mtimeMs: number
  info: GgufInfo | null
  error?: string
}

export type FileKind = 'model' | 'mmproj' | 'draft' | 'adapter' | 'shard'

const SHARD_RE = /-(\d{5})-of-(\d{5})\.gguf$/i

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

/** Words that name a file's role or precision rather than the model it belongs to. */
const NOT_MODEL_WORDS = new Set(['mmproj', 'mtp', 'eagle3', 'dflash', 'dspark', 'draft', 'model', 'i1', 'f16', 'f32', 'bf16', 'fp16', 'fp32', 'fp8'])

/**
 * The model a file belongs to, as name tokens: "mmproj-gemma-3-12b-it-f16" and
 * "gemma-3-12b-it-Q4_K_M" both give gemma/3/12b/it. Empty for generic names like "mmproj-BF16".
 */
export function modelTokens(fileName: string): string[] {
  const base = displayName(fileName)
    .replace(/\s*\(\d+\)$/, '') // "(1)" left by a repeated download
    .replace(QUANT_TAGS, '')
    .toLowerCase()
  return base.split(/[^a-z0-9]+/).filter((t) => t && !NOT_MODEL_WORDS.has(t))
}

const isPrefix = (a: string[], b: string[]): boolean => a.length <= b.length && a.every((t, i) => t === b[i])

/** How well a sidecar's name fits a model: 2 = same model, 0 = says nothing, -1 = another model. */
function nameFit(sidecar: ScannedFile, model: ScannedFile): number {
  const s = modelTokens(basename(sidecar.path))
  const m = modelTokens(basename(model.path))
  // Projectors converted with their model usually carry its general.name, even when the file is just "mmproj-F16".
  const sameName = !!sidecar.info?.name && sidecar.info.name.toLowerCase() === model.info?.name?.toLowerCase()
  if ((s.length && (isPrefix(s, m) || isPrefix(m, s))) || sameName) return 2
  return s.length ? -1 : 0
}

/** True when every model base name extends another, as with several quants of one model in its own folder. */
function oneModelFamily(models: ScannedFile[]): boolean {
  const bases = models.map((f) => modelTokens(basename(f.path))).sort((a, b) => a.length - b.length)
  return bases.every((b, i) => i === 0 || isPrefix(bases[i - 1], b))
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
    const kinds = dirFiles
      .map((f) => ({ f, kind: classifyFile(basename(f.path), f.info) }))
      .sort((a, b) => basename(a.f.path).localeCompare(basename(b.f.path)))
    const modelFiles = kinds.filter((k) => k.kind === 'model').map((k) => k.f)
    // Generic sidecars ("mmproj-BF16") only go with a folder that holds one model. In a shared
    // folder such as a flat C:\LLM they could belong to any model, and a wrong projector fails the load.
    const anyFits = oneModelFamily(modelFiles)
    const sidecar = (kind: FileKind, model: ScannedFile): SidecarFile | undefined => {
      let best: { f: ScannedFile; fit: number } | undefined
      for (const k of kinds) {
        if (k.kind !== kind) continue
        const fit = nameFit(k.f, model)
        if (!best || fit > best.fit) best = { f: k.f, fit }
      }
      if (!best || (best.fit < 2 && !anyFits)) return undefined
      const out: SidecarFile = { path: best.f.path, sizeBytes: best.f.sizeBytes }
      if (kind === 'draft') out.kind = draftKind(basename(best.f.path))
      return out
    }

    for (const { f, kind } of kinds) {
      if (kind !== 'model') continue
      const mmproj = sidecar('mmproj', f)
      const draftFile = sidecar('draft', f)
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
        // Built-in MTP layers were trained with the model, so they win over a separate head.
        draft: f.info?.mtpLayers ? { path: f.path, sizeBytes: 0, kind: 'mtp', builtIn: true } : draftFile,
        draftSidecar: draftFile,
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

/** Bump when GgufInfo gains fields or its derivation changes, so cached metadata is re-read. */
const CACHE_VERSION = 3

interface CacheEntry {
  v?: number
  size: number
  mtimeMs: number
  info: GgufInfo | null
  error?: string
  /** For a split model's first shard: sizes and dates of the other shards when it was read. */
  shards?: string
}

interface FoundFile {
  path: string
  size: number
  mtimeMs: number
}

const SHARD_TAIL = /^-(\d{5})-of-(\d{5})\.gguf$/i

/** The other files of a split model, for its first shard (the one with the metadata). */
export function otherShards<T extends { path: string }>(path: string, found: T[]): T[] {
  const m = basename(path).match(SHARD_RE)
  if (!m || Number(m[1]) !== 1) return []
  const prefix = path.slice(0, path.length - m[0].length)
  return found
    .filter((o) => o.path !== path && o.path.startsWith(prefix) && SHARD_TAIL.exec(o.path.slice(prefix.length))?.[2] === m[2])
    .sort((a, b) => a.path.localeCompare(b.path))
}

/** Split shards only hold a short header and their tensor list, so a small first read covers them. */
const SHARD_FIRST_READ = 1 << 20

async function ggufInfo(path: string, shards: string[]): Promise<GgufInfo> {
  const header = await readGgufHeaderAsync(path)
  let hasNextn = header.hasNextn
  // Split models keep their MTP (nextn) layers at the end, in the last shard. The other tensor
  // lists are only read when the metadata says the model has such layers.
  if (!hasNextn && shards.length && toGgufInfo(header.kv, basename(path), true).mtpLayers) {
    for (const s of [...shards].reverse()) {
      hasNextn = await readGgufHeaderAsync(s, SHARD_FIRST_READ).then(
        (h) => h.hasNextn,
        () => false // a shard still downloading only costs the MTP check
      )
      if (hasNextn) break
    }
  }
  return toGgufInfo(header.kv, basename(path), hasNextn)
}

/** Files that downloaders and editors write before or while saving the real one. */
const TEMP_FILE = /(\.(part|partial|tmp|temp|crdownload|download|incomplete|aria2)|~)$/i

/** Scans the models folder, caches GGUF metadata, and watches for changes. */
export class ModelLibrary {
  private models: ModelEntry[] = []
  private cache: Record<string, CacheEntry> = {}
  private cacheLoaded = false
  private watcher: FSWatcher | null = null
  private watchedDir = ''
  private debounce: NodeJS.Timeout | null = null
  private scanning: { root: string; promise: Promise<ModelEntry[]> } | null = null

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
    const root = this.getRoot()
    if (this.scanning?.root === root) return this.scanning.promise
    // A scan of the previous folder may still be running; this one starts after it, so the
    // new folder's models are the ones that end up shown.
    const before = this.scanning?.promise.catch(() => undefined) ?? Promise.resolve()
    const run = { root, promise: before.then(() => this.doScan(root)) }
    this.scanning = run
    const done = (): void => {
      if (this.scanning !== run) return
      this.scanning = null
      // The folder changed during the scan without anyone asking for a new one.
      if (this.getRoot() !== root) void this.scan().catch(() => undefined)
    }
    void run.promise.then(done, done)
    return run.promise
  }

  private async doScan(root: string): Promise<ModelEntry[]> {
    if (!this.cacheLoaded) {
      // Only a cache: if it can't be read, the headers are read again and the file rewritten.
      this.cache = await readJson<Record<string, CacheEntry>>(this.cachePath, {}).catch(() => ({}))
      this.cacheLoaded = true
    }
    const found: FoundFile[] = []
    await walk(root, 0, found)

    let dirty = false
    const seen = new Set<string>()
    const files: ScannedFile[] = []
    for (const f of found) {
      seen.add(f.path)
      let entry = this.cache[f.path]
      const shards = otherShards(f.path, found)
      const shardSig = shards.length ? shards.map((s) => `${s.size}:${s.mtimeMs}`).join(',') : undefined
      if (
        !entry ||
        entry.v !== CACHE_VERSION ||
        entry.size !== f.size ||
        entry.mtimeMs !== f.mtimeMs ||
        entry.shards !== shardSig
      ) {
        const shard = basename(f.path).match(SHARD_RE)
        if (shard && Number(shard[1]) !== 1) {
          entry = { v: CACHE_VERSION, size: f.size, mtimeMs: f.mtimeMs, info: null }
        } else {
          const base = { v: CACHE_VERSION, size: f.size, mtimeMs: f.mtimeMs, shards: shardSig }
          try {
            entry = { ...base, info: await ggufInfo(f.path, shards.map((s) => s.path)) }
          } catch (err) {
            entry = { ...base, info: null, error: (err as Error).message }
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

    const models = groupModels(root, files)
    // The models folder changed while this scan ran; the scan of the new folder publishes instead.
    if (this.getRoot() !== root) return models
    this.models = models
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
        // Partial downloads change constantly; the finished file's rename triggers the scan.
        // Anything else counts, including folders with dots in their names ("Qwen3.5").
        const name = String(file ?? '')
        if (name && TEMP_FILE.test(name)) return
        if (this.debounce) clearTimeout(this.debounce)
        this.debounce = setTimeout(() => void this.scan().catch(() => undefined), 2000)
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
