// OpenRouter's model list: parsed into usable CloudModelInfo entries, cached on disk for 24 hours.
import { promises as fs } from 'node:fs'
import type { CloudCatalog, CloudModelInfo, CloudReasoning } from '@shared/cloud'
import { writeJsonAtomic } from '../util/jsonFile'
import { isFreePrices, parsePrices } from './prices'

export const OPENROUTER_ORIGIN = 'https://openrouter.ai'
export const OPENROUTER_BASE = `${OPENROUTER_ORIGIN}/api/v1`
export const CATALOG_TTL_MS = 24 * 3600_000

const str = (v: unknown): string => (typeof v === 'string' ? v : '')
const stringList = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [])

function parseReasoning(m: Record<string, unknown>, supported: string[]): CloudReasoning | null {
  const r = m.reasoning
  if (r && typeof r === 'object' && !Array.isArray(r)) {
    const o = r as Record<string, unknown>
    return { mandatory: o.mandatory === true, efforts: stringList(o.supported_efforts), supportsMaxTokens: o.supports_max_tokens === true }
  }
  if (supported.includes('reasoning') || supported.includes('reasoning_effort')) return { mandatory: false, efforts: [], supportsMaxTokens: false }
  return null
}

/**
 * One entry of GET /models, or null when it cannot be used: not text in and text out, not chat completions, a router, an
 * alias (~) or batch (:batch) entry, a price that is not a number >= 0 (routers use -1), no context, or neither
 * max_tokens nor max_completion_tokens (then no cost ceiling can be set).
 */
export function parseModel(raw: unknown): CloudModelInfo | null {
  if (!raw || typeof raw !== 'object') return null
  const m = raw as Record<string, unknown>
  const id = str(m.id)
  if (!id || id.startsWith('~') || id.endsWith(':batch')) return null
  const arch = (m.architecture ?? {}) as Record<string, unknown>
  if (str(arch.tokenizer) === 'Router') return null
  const output = stringList(arch.output_modalities)
  if (output.length !== 1 || output[0] !== 'text') return null
  const input = stringList(arch.input_modalities)
  if (input.length && !input.includes('text')) return null
  if (Array.isArray(m.endpoints) && !m.endpoints.includes('/api/v1/chat/completions')) return null
  const prices = parsePrices(m.pricing)
  if (!prices) return null
  const contextLength = typeof m.context_length === 'number' ? m.context_length : 0
  if (!Number.isFinite(contextLength) || contextLength <= 0) return null
  const supported = stringList(m.supported_parameters)
  if (!supported.includes('max_tokens') && !supported.includes('max_completion_tokens')) return null
  const top = (m.top_provider ?? {}) as Record<string, unknown>
  const maxReply = typeof top.max_completion_tokens === 'number' && top.max_completion_tokens > 0 ? Math.floor(top.max_completion_tokens) : null
  return {
    slug: id,
    name: str(m.name) || id,
    description: str(m.description),
    contextLength: Math.floor(contextLength),
    maxReply,
    prices,
    free: isFreePrices(prices),
    vision: input.includes('image'),
    tools: supported.includes('tools'),
    reasoning: parseReasoning(m, supported),
    supported,
    created: typeof m.created === 'number' && Number.isFinite(m.created) ? m.created : 0,
    expires: typeof m.expiration_date === 'string' && m.expiration_date ? m.expiration_date : null
  }
}

/** The usable models of a GET /models answer, by name. */
export function parseCatalog(json: unknown): CloudModelInfo[] {
  const data = (json as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) throw new Error('OpenRouter sent a model list in a form this app does not know.')
  const out: CloudModelInfo[] = []
  const seen = new Set<string>()
  for (const raw of data) {
    const info = parseModel(raw)
    if (info && !seen.has(info.slug)) {
      seen.add(info.slug)
      out.push(info)
    }
  }
  return out.sort((a, b) => a.name.localeCompare(b.name))
}

export interface CatalogDeps {
  path: string
  fetch: typeof fetch
  now: () => number
  scrub: (text: string) => string
  write?: (path: string, data: unknown) => Promise<void>
  ttlMs?: number
}

export interface CatalogResult {
  catalog: CloudCatalog
  /** Set when OpenRouter could not be read and the cached list (if any) is returned. */
  error: string | null
}

/** Fetches (without a key) and caches the list. Offline: the cached list, with the error. */
export class CatalogStore {
  private cache: CloudCatalog | null = null
  private loaded = false
  private inflight: Promise<CatalogResult> | null = null
  private inflightRefresh = false

  constructor(private readonly deps: CatalogDeps) {}

  private async readDisk(): Promise<void> {
    if (this.loaded) return
    this.loaded = true
    try {
      const j = JSON.parse((await fs.readFile(this.deps.path, 'utf8')).replace(/^﻿/, '')) as { at?: unknown; models?: unknown }
      if (typeof j.at === 'number' && Array.isArray(j.models)) {
        // Entries are read again through the parser, so a damaged or older file cannot hand out bad models.
        const models: CloudModelInfo[] = []
        for (const m of j.models) {
          const info = parseCachedInfo(m)
          if (info) models.push(info)
        }
        this.cache = { models, at: j.at }
      }
    } catch {
      // No cache, or unreadable: the next read from OpenRouter makes a new one.
    }
  }

  /** The cached list without asking OpenRouter; null when there is none. */
  async cached(): Promise<CloudCatalog | null> {
    await this.readDisk()
    return this.cache
  }

  get(refresh: boolean): Promise<CatalogResult> {
    // A read that may be answered from the cache does not stand for a refresh: that one goes after it.
    if (this.inflight && (!refresh || this.inflightRefresh)) return this.inflight
    const before = this.inflight
    const p: Promise<CatalogResult> = (before ? before.then(() => this.run(refresh), () => this.run(refresh)) : this.run(refresh)).finally(() => {
      if (this.inflight === p) this.inflight = null
    })
    this.inflight = p
    this.inflightRefresh = refresh
    return p
  }

  private async run(refresh: boolean): Promise<CatalogResult> {
    await this.readDisk()
    const ttl = this.deps.ttlMs ?? CATALOG_TTL_MS
    if (!refresh && this.cache && this.deps.now() - this.cache.at < ttl && this.cache.at <= this.deps.now() + 60_000) return { catalog: this.cache, error: null }
    try {
      const res = await this.deps.fetch(`${OPENROUTER_BASE}/models`, { redirect: 'error', signal: AbortSignal.timeout(30_000), headers: { Accept: 'application/json' } })
      if (!res.ok) throw new Error(`OpenRouter answered ${res.status} to the model list.`)
      const models = parseCatalog(await res.json())
      const catalog: CloudCatalog = { models, at: this.deps.now() }
      this.cache = catalog
      await (this.deps.write ?? writeJsonAtomic)(this.deps.path, catalog).catch(() => undefined)
      return { catalog, error: null }
    } catch (err) {
      const msg = this.deps.scrub(err instanceof Error ? err.message : String(err))
      return { catalog: this.cache ?? { models: [], at: 0 }, error: `Could not read OpenRouter's model list: ${msg}` }
    }
  }
}

/** A CloudModelInfo read back from a file (ours: already parsed once). Checks the fields everything else relies on. */
export function parseCachedInfo(raw: unknown): CloudModelInfo | null {
  if (!raw || typeof raw !== 'object') return null
  const m = raw as Record<string, unknown>
  const p = m.prices as Record<string, unknown> | undefined
  const nums = p ? (['promptMax', 'completionMax', 'request', 'image', 'prompt', 'completion'] as const).map((k) => p[k]) : []
  if (typeof m.slug !== 'string' || !m.slug || !p || !nums.every((n) => typeof n === 'number' && Number.isFinite(n) && n >= 0)) return null
  if (typeof m.contextLength !== 'number' || !(m.contextLength > 0) || !Array.isArray(m.supported)) return null
  const r = m.reasoning as Record<string, unknown> | null | undefined
  return {
    slug: m.slug,
    name: str(m.name) || m.slug,
    description: str(m.description),
    contextLength: Math.floor(m.contextLength),
    maxReply: typeof m.maxReply === 'number' && m.maxReply > 0 ? Math.floor(m.maxReply) : null,
    prices: {
      promptMax: p.promptMax as number,
      completionMax: p.completionMax as number,
      request: p.request as number,
      image: p.image as number,
      prompt: p.prompt as number,
      completion: p.completion as number
    },
    free: m.free === true,
    vision: m.vision === true,
    tools: m.tools === true,
    reasoning: r && typeof r === 'object' ? { mandatory: r.mandatory === true, efforts: stringList(r.efforts), supportsMaxTokens: r.supportsMaxTokens === true } : null,
    supported: stringList(m.supported),
    created: typeof m.created === 'number' ? m.created : 0,
    expires: typeof m.expires === 'string' && m.expires ? m.expires : null
  }
}
