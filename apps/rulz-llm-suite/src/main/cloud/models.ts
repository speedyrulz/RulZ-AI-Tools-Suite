// The user's cloud models and options: cloud.json.
import {
  cloudModelId,
  DEFAULT_CONTEXT_LIMIT,
  DEFAULT_DAILY_LIMIT_USD,
  DEFAULT_REPLY_LIMIT,
  MIN_CONTEXT_LIMIT,
  MIN_DAILY_LIMIT_USD,
  MIN_REPLY_LIMIT,
  type CloudModelDef,
  type CloudModelInfo,
  type CloudModelPatch,
  type CloudOptions
} from '@shared/cloud'
import type { ModelEntry } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { parseCachedInfo } from './catalog'

const clamp = (n: number, lo: number, hi: number): number => Math.min(Math.max(n, lo), Math.max(lo, hi))

/** A limit in range for the model: whole tokens between the smaller of the minimum and the model's own size, and its size. */
export function clampContext(n: unknown, info: CloudModelInfo): number {
  const v = typeof n === 'number' && Number.isFinite(n) ? Math.floor(n) : Math.min(info.contextLength, DEFAULT_CONTEXT_LIMIT)
  return clamp(v, Math.min(MIN_CONTEXT_LIMIT, info.contextLength), info.contextLength)
}

export function clampReply(n: unknown, info: CloudModelInfo): number {
  const top = Math.min(info.maxReply ?? info.contextLength, info.contextLength)
  const v = typeof n === 'number' && Number.isFinite(n) ? Math.floor(n) : Math.min(DEFAULT_REPLY_LIMIT, top)
  return clamp(v, Math.min(MIN_REPLY_LIMIT, top), top)
}

/** null (no limit) or a number >= the minimum; anything else is not a limit. */
export function validLimit(v: unknown): number | null | undefined {
  if (v === null) return null
  if (typeof v === 'number' && Number.isFinite(v) && v >= MIN_DAILY_LIMIT_USD) return v
  return undefined
}

export const DEFAULT_OPTIONS: CloudOptions = { dailyLimitUsd: DEFAULT_DAILY_LIMIT_USD, allowTraining: false }

export function parseOptions(raw: unknown): CloudOptions {
  const o = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  // A limit that is missing or unreadable falls back to the $1 default, never to "no limit".
  const limit = 'dailyLimitUsd' in o ? validLimit(o.dailyLimitUsd) : undefined
  return { dailyLimitUsd: limit === undefined ? DEFAULT_OPTIONS.dailyLimitUsd : limit, allowTraining: o.allowTraining === true }
}

export function parseDef(raw: unknown): CloudModelDef | null {
  if (!raw || typeof raw !== 'object') return null
  const o = raw as Record<string, unknown>
  const info = parseCachedInfo(o.info)
  if (!info || o.slug !== info.slug) return null
  return {
    slug: info.slug,
    contextLimit: clampContext(o.contextLimit, info),
    replyLimit: clampReply(o.replyLimit, info),
    allowApps: o.allowApps === true,
    info,
    addedAt: typeof o.addedAt === 'number' && Number.isFinite(o.addedAt) ? o.addedAt : 0
  }
}

export function toModelEntry(def: CloudModelDef): ModelEntry {
  return {
    id: cloudModelId(def.slug),
    name: def.info.name,
    folder: '',
    path: '',
    files: [],
    sizeBytes: 0,
    info: null,
    modifiedAt: def.addedAt,
    cloud: { ...def, provider: 'openrouter' }
  }
}

export interface ModelStoreDeps {
  path: string
  now: () => number
  write?: (path: string, data: unknown) => Promise<void>
}

export class CloudModelStore {
  private opts: CloudOptions = { ...DEFAULT_OPTIONS }
  private defs: CloudModelDef[] = []

  constructor(private readonly deps: ModelStoreDeps) {}

  async load(): Promise<void> {
    const raw = await readJson<{ options?: unknown; models?: unknown }>(this.deps.path, {})
    this.opts = parseOptions(raw.options)
    this.defs = []
    if (Array.isArray(raw.models)) {
      const seen = new Set<string>()
      for (const m of raw.models) {
        const d = parseDef(m)
        if (d && !seen.has(d.slug)) {
          seen.add(d.slug)
          this.defs.push(d)
        }
      }
    }
  }

  private save(): Promise<void> {
    return (this.deps.write ?? writeJsonAtomic)(this.deps.path, { options: this.opts, models: this.defs })
  }

  options(): CloudOptions {
    return { ...this.opts }
  }
  list(): CloudModelDef[] {
    return this.defs.map((d) => ({ ...d }))
  }
  get(slug: string): CloudModelDef | undefined {
    return this.defs.find((d) => d.slug === slug)
  }

  async setOptions(patch: Partial<CloudOptions>): Promise<CloudOptions> {
    const next = { ...this.opts }
    if ('dailyLimitUsd' in patch) {
      const v = validLimit(patch.dailyLimitUsd)
      if (v === undefined) throw new Error(`The daily limit must be off or at least $${MIN_DAILY_LIMIT_USD.toFixed(2)}.`)
      next.dailyLimitUsd = v
    }
    if (typeof patch.allowTraining === 'boolean') next.allowTraining = patch.allowTraining
    const before = this.opts
    this.opts = next
    try {
      await this.save()
    } catch (err) {
      this.opts = before
      throw err
    }
    return this.options()
  }

  async add(info: CloudModelInfo): Promise<CloudModelDef> {
    if (this.get(info.slug)) throw new Error(`${info.name} is already in your models.`)
    const def: CloudModelDef = {
      slug: info.slug,
      contextLimit: clampContext(undefined, info),
      replyLimit: clampReply(undefined, info),
      allowApps: false,
      info,
      addedAt: this.deps.now()
    }
    this.defs.push(def)
    try {
      await this.save()
    } catch (err) {
      this.defs.pop()
      throw err
    }
    return { ...def }
  }

  /** Limits are clamped into range for the model. */
  async update(slug: string, patch: CloudModelPatch): Promise<CloudModelDef> {
    const d = this.get(slug)
    if (!d) throw new Error('That model is not in your list.')
    const before = { ...d }
    if (patch.contextLimit !== undefined) d.contextLimit = clampContext(patch.contextLimit, d.info)
    if (patch.replyLimit !== undefined) d.replyLimit = clampReply(patch.replyLimit, d.info)
    if (typeof patch.allowApps === 'boolean') d.allowApps = patch.allowApps
    try {
      await this.save()
    } catch (err) {
      Object.assign(d, before)
      throw err
    }
    return { ...d }
  }

  async remove(slug: string): Promise<void> {
    const i = this.defs.findIndex((d) => d.slug === slug)
    if (i < 0) return
    const [gone] = this.defs.splice(i, 1)
    try {
      await this.save()
    } catch (err) {
      this.defs.splice(i, 0, gone)
      throw err
    }
  }

  /** Takes the newest info of every listed model that is still in the catalog (limits are clamped again). True when anything changed. */
  async refreshInfo(catalog: CloudModelInfo[]): Promise<boolean> {
    const bySlug = new Map(catalog.map((m) => [m.slug, m]))
    let changed = false
    for (const d of this.defs) {
      const info = bySlug.get(d.slug)
      if (!info || JSON.stringify(info) === JSON.stringify(d.info)) continue
      d.info = info
      d.contextLimit = clampContext(d.contextLimit, info)
      d.replyLimit = clampReply(d.replyLimit, info)
      changed = true
    }
    if (changed) await this.save()
    return changed
  }
}
