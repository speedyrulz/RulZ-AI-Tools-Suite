// Remote model definitions (docs/vast-design.md, section 6): saved in remote-models.json, shown as ModelEntry, and
// turned into the list of files a rental needs.
import { statSync } from 'node:fs'
import { basename } from 'node:path'
import { isRemoteModelId, REMOTE_MODEL_PREFIX, REMOTE_SLUG, type RemoteModelDef, type RemoteModelSource } from '@shared/remote'
import type { AdapterEntry, LoadConfig, ModelEntry } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { FILE_NAME_RE } from './ssh'

export const REMOTE_FOLDER = 'vast.ai'

// ---------- Validation ----------

const REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/
const REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$/

/** A path inside a repo: no empty, `.` or `..` parts, no backslash or control characters. Returns it without a leading slash. */
function cleanRepoPath(raw: unknown, what: string, required: boolean): string {
  const s = typeof raw === 'string' ? raw.trim().replace(/^\/+/, '') : ''
  if (!s) {
    if (required) throw new Error(`${what} is empty.`)
    return ''
  }
  // eslint-disable-next-line no-control-regex
  if (/[\\\u0000-\u001f]/.test(s)) throw new Error(`${what} has characters that are not allowed.`)
  const parts = s.split('/')
  if (parts.some((p) => !p || p === '.' || p === '..')) throw new Error(`${what} must not contain empty parts or "..".`)
  if (!/\.gguf$/i.test(s)) throw new Error(`${what} must be a .gguf file.`)
  return s
}

export function validateSource(raw: unknown): RemoteModelSource {
  const r = (raw ?? {}) as Record<string, unknown>
  if (r.kind === 'local') {
    const id = typeof r.modelId === 'string' ? r.modelId.trim() : ''
    if (!id) throw new Error('Choose a model from the library.')
    if (isRemoteModelId(id)) throw new Error('A remote model cannot be made from another remote model.')
    return { kind: 'local', modelId: id }
  }
  if (r.kind === 'hf') {
    const repo = typeof r.repo === 'string' ? r.repo.trim() : ''
    if (!REPO_RE.test(repo) || repo.includes('..')) throw new Error('The repository must look like owner/name.')
    const revision = typeof r.revision === 'string' && r.revision.trim() ? r.revision.trim() : 'main'
    if (!REVISION_RE.test(revision) || revision.includes('..')) throw new Error('The revision is not valid.')
    return {
      kind: 'hf',
      repo,
      bucket: r.bucket === true,
      revision,
      file: cleanRepoPath(r.file, 'The file', true),
      mmproj: cleanRepoPath(r.mmproj, 'The projector file', false)
    }
  }
  throw new Error('The model source must be a library model or a Hugging Face file.')
}

/** A folder name from a model name: lower case, [a-z0-9._-], at most 60 characters, unique among `taken`. */
export function makeSlug(name: string, taken: ReadonlySet<string>): string {
  let base = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^[^a-z0-9]+/, '')
    .replace(/[-._]+$/, '')
    .slice(0, 52)
    .replace(/[-._]+$/, '')
  if (!base) base = 'model'
  let slug = base
  for (let n = 2; taken.has(REMOTE_MODEL_PREFIX + slug); n++) slug = `${base}-${n}`
  if (!REMOTE_SLUG.test(slug)) throw new Error('Could not make a folder name for this model.')
  return slug
}

export const slugOf = (modelId: string): string => modelId.slice(REMOTE_MODEL_PREFIX.length)

export type RemoteModelInput = Omit<RemoteModelDef, 'id' | 'createdAt'> & { id?: string }

// ---------- Store ----------

interface StoreFile {
  version: 1
  models: RemoteModelDef[]
}

export class RemoteModelStore {
  private defs: RemoteModelDef[] = []
  private loading: Promise<void> | null = null

  constructor(
    private readonly path: string,
    private readonly now: () => number = Date.now
  ) {}

  /** Reads the file once; call it at start. Entries that do not validate are dropped. */
  load(): Promise<void> {
    this.loading ??= (async () => {
      const file = await readJson<Partial<StoreFile>>(this.path, {})
      const kept: RemoteModelDef[] = []
      for (const d of Array.isArray(file.models) ? file.models : []) {
        try {
          kept.push(this.normalize(d, new Set(kept.map((k) => k.id))))
        } catch (err) {
          console.warn(`Dropped a remote model definition from ${this.path}: ${(err as Error).message}`)
        }
      }
      this.defs = kept
    })()
    return this.loading
  }

  list(): RemoteModelDef[] {
    return this.defs.map((d) => structuredClone(d))
  }

  get(id: string): RemoteModelDef | undefined {
    const d = this.defs.find((x) => x.id === id)
    return d ? structuredClone(d) : undefined
  }

  private normalize(raw: unknown, taken: ReadonlySet<string>, createdAt = this.now()): RemoteModelDef {
    const r = (raw ?? {}) as Record<string, unknown>
    const name = typeof r.name === 'string' ? r.name.trim() : ''
    if (!name) throw new Error('Give the model a name.')
    if (name.length > 120) throw new Error('The name is too long.')
    const source = validateSource(r.source)
    let id = typeof r.id === 'string' && r.id ? r.id : ''
    if (id) {
      if (!id.startsWith(REMOTE_MODEL_PREFIX) || !REMOTE_SLUG.test(slugOf(id))) throw new Error('The model id is not valid.')
    } else {
      id = REMOTE_MODEL_PREFIX + makeSlug(name, taken)
    }
    const when = typeof r.createdAt === 'number' && Number.isFinite(r.createdAt) ? r.createdAt : createdAt
    return { id, name, source, createdAt: when }
  }

  async save(input: RemoteModelInput): Promise<RemoteModelDef> {
    await this.load()
    const existing = input.id ? this.defs.find((d) => d.id === input.id) : undefined
    const others = new Set(this.defs.filter((d) => d !== existing).map((d) => d.id))
    const def = this.normalize({ ...input, createdAt: existing?.createdAt }, others)
    if (!existing && this.defs.some((d) => d.id === def.id)) throw new Error('A remote model with this id exists already.')
    const next = existing ? this.defs.map((d) => (d === existing ? def : d)) : [...this.defs, def]
    await this.persist(next)
    this.defs = next
    return structuredClone(def)
  }

  async delete(id: string): Promise<boolean> {
    await this.load()
    if (!this.defs.some((d) => d.id === id)) return false
    const next = this.defs.filter((d) => d.id !== id)
    await this.persist(next)
    this.defs = next
    return true
  }

  private persist(models: RemoteModelDef[]): Promise<void> {
    const file: StoreFile = { version: 1, models }
    return writeJsonAtomic(this.path, file)
  }
}

// ---------- As a model in the lists ----------

/** The model as the lists show it. A library source copies its information; one that left the library says so. */
export function toModelEntry(def: RemoteModelDef, getModel: (id: string) => ModelEntry | undefined): ModelEntry {
  const base: ModelEntry = {
    id: def.id,
    name: def.name,
    folder: REMOTE_FOLDER,
    path: '',
    files: [],
    sizeBytes: 0,
    info: null,
    modifiedAt: def.createdAt,
    remote: true
  }
  if (def.source.kind === 'hf') {
    // The projector is listed (size unknown) so that --mmproj is passed and the model shows as a vision model. Nothing on
    // this PC opens the path: it only names the file.
    const mmproj = def.source.mmproj ? { path: `hf:${def.source.repo}/${def.source.mmproj}`, sizeBytes: 0 } : undefined
    return { ...base, path: `hf:${def.source.repo}/${def.source.file}`, mmproj }
  }
  const src = getModel(def.source.modelId)
  if (!src) {
    return { ...base, error: `The library model "${def.source.modelId}" this was made from is no longer in the library.` }
  }
  return {
    ...base,
    path: src.path,
    files: [...src.files],
    sizeBytes: src.sizeBytes,
    mmproj: src.mmproj,
    draft: src.draft,
    draftSidecar: src.draftSidecar,
    info: src.info,
    modifiedAt: src.modifiedAt
  }
}

// ---------- Files ----------

export interface RemoteFile {
  role: 'main' | 'mmproj' | 'draft' | 'adapter'
  /** Name inside models/<slug>/ on the instance. */
  name: string
  /** A file on this PC to upload; null for a download. */
  localPath: string | null
  /** A download address; null for an upload. */
  url: string | null
  /** Bytes; null = not known yet (a download is asked, an upload is read from disk). */
  size: number | null
  sha256: string | null
}

const enc = (s: string): string => s.split('/').map(encodeURIComponent).join('/')

/** The download address of a file of a model repo or bucket. */
export function hfUrl(src: Extract<RemoteModelSource, { kind: 'hf' }>, file: string): string {
  if (src.bucket) return `https://huggingface.co/buckets/${enc(src.repo)}/resolve/${enc(file)}`
  return `https://huggingface.co/${enc(src.repo)}/resolve/${encodeURIComponent(src.revision)}/${enc(file)}`
}

const SHARD_NAME = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i

/** The files of a split GGUF named by any of its parts (the repo path of each), or just the file. */
export function expandShards(file: string): string[] {
  const dir = file.includes('/') ? file.slice(0, file.lastIndexOf('/') + 1) : ''
  const m = SHARD_NAME.exec(file.slice(dir.length))
  if (!m) return [file]
  const total = Number(m[3])
  if (!Number.isInteger(total) || total < 1 || total > 999) return [file]
  const out: string[] = []
  for (let i = 1; i <= total; i++) out.push(`${dir}${m[1]}-${String(i).padStart(5, '0')}-of-${m[3]}.gguf`)
  return out
}

/** A name that passes FILE_NAME_RE: unusual characters become `_`. */
function cleanName(raw: string): string {
  let n = raw.replace(/[^A-Za-z0-9._+-]+/g, '_').replace(/^[^A-Za-z0-9]+/, '')
  if (n.endsWith('.part')) n = `${n}.gguf`
  n = n.replace(/\.\.+/g, '.')
  if (!n) n = 'file.gguf'
  return n.slice(-200).replace(/^[^A-Za-z0-9]+/, '') || 'file.gguf'
}

/**
 * Whether the instance's record of a file still holds: it is on disk with the recorded size, the size is the expected one
 * when known, a Hugging Face file comes from the same address as now (a line without a source is accepted), and an
 * uploaded file is still the one that was sent: its cached hash is the recorded one or, when it was not hashed since the
 * app started, its modification time is (`local.source`). A file that only got a new date is checked again by Set up.
 */
export function recordedComplete(f: RemoteFile, c: { size: number; sha256: string; source: string } | undefined, onDisk: number | undefined, local?: { sha256?: string; source?: string }): boolean {
  if (!c || onDisk === undefined || onDisk !== c.size) return false
  if (f.size !== null && f.size !== c.size) return false
  if (f.url && c.source !== '-' && c.source !== f.url) return false
  if (f.localPath) {
    if (local?.sha256 && c.sha256 !== '-') return c.sha256 === local.sha256
    if (local?.source && c.source.startsWith('pc:') && c.source !== local.source) return false
  }
  return true
}

export interface FileListDeps {
  getModel: (id: string) => ModelEntry | undefined
  getAdapter?: (id: string) => AdapterEntry | undefined
  /** The load config whose draft and adapters go with the model; none = the model's own draft file only. */
  config?: LoadConfig
  /** Size of a local file; default reads the disk. */
  localSize?: (path: string) => number | null
}

const diskSize = (path: string): number | null => {
  try {
    return statSync(path).size
  } catch {
    return null
  }
}

/**
 * The files a rental needs for a model, with their roles and names on the instance (unique within the folder).
 * Throws a sentence when a library source is gone. Main parts keep their names so split files still find each other.
 */
export function fileList(def: RemoteModelDef, deps: FileListDeps): RemoteFile[] {
  const size = deps.localSize ?? diskSize
  const used = new Set<string>()
  const files: RemoteFile[] = []
  const uniqueName = (raw: string, role: RemoteFile['role']): string => {
    let name = cleanName(raw)
    if (used.has(name)) name = cleanName(`${role}-${name}`)
    for (let n = 2; used.has(name); n++) name = cleanName(`${role}-${n}-${raw}`)
    if (!FILE_NAME_RE.test(name)) throw new Error(`Cannot name ${raw} on the instance.`)
    used.add(name)
    return name
  }
  const local = (role: RemoteFile['role'], path: string, knownSize: number | null = null): void => {
    files.push({ role, name: uniqueName(basename(path.replace(/\\/g, '/')), role), localPath: path, url: null, size: knownSize ?? size(path), sha256: null })
  }

  const src = def.source
  if (src.kind === 'hf') {
    for (const f of expandShards(src.file)) files.push({ role: 'main', name: uniqueName(f.slice(f.lastIndexOf('/') + 1), 'main'), localPath: null, url: hfUrl(src, f), size: null, sha256: null })
    if (src.mmproj) files.push({ role: 'mmproj', name: uniqueName(src.mmproj.slice(src.mmproj.lastIndexOf('/') + 1), 'mmproj'), localPath: null, url: hfUrl(src, src.mmproj), size: null, sha256: null })
  } else {
    const model = deps.getModel(src.modelId)
    if (!model) throw new Error(`The library model "${src.modelId}" is no longer in the library, so there is nothing to upload.`)
    const parts = model.files.length ? model.files : [model.path]
    for (const p of parts) local('main', p)
    if (model.mmproj) local('mmproj', model.mmproj.path, model.mmproj.sizeBytes || null)
    const c = deps.config
    const wantsDraft = !c || c.useDraft
    if (wantsDraft) {
      if (c?.draftFile && c.draftFile !== 'builtin') local('draft', c.draftFile)
      else if (model.draftSidecar && c?.draftFile !== 'builtin') local('draft', model.draftSidecar.path, model.draftSidecar.sizeBytes || null)
    }
  }
  for (const a of deps.config?.adapters ?? []) {
    const entry = deps.getAdapter?.(a.id)
    if (!entry) continue
    local('adapter', entry.path, entry.sizeBytes || null)
  }
  return files
}
