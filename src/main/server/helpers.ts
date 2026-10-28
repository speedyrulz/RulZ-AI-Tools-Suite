// Pure helpers for the local API server (unit-tested separately from the HTTP code).
import { z } from 'zod'
import { KV_CACHE_TYPES, type LoadConfig, type ModelEntry } from '@shared/types'

/** The id a model is listed under in GET /v1/models: its name, or its path when names collide. */
export function publicModelId(model: ModelEntry, all: ModelEntry[]): string {
  const dupes = all.filter((m) => m.name.toLowerCase() === model.name.toLowerCase())
  return dupes.length > 1 ? model.id : model.name
}

/**
 * Finds the model a request asked for. Accepts the public id, the library path, the file name
 * (with or without .gguf), any of those case-insensitively, and finally a unique substring match.
 */
export function resolveModel(models: ModelEntry[], requested: string | undefined | null): ModelEntry | undefined {
  const q = (requested ?? '').trim()
  if (!q) return undefined
  const lower = q.toLowerCase()
  const strip = (s: string): string => s.toLowerCase().replace(/\.gguf$/, '')
  const unique = (hits: ModelEntry[]): ModelEntry | undefined => (hits.length === 1 ? hits[0] : undefined)
  // The library path is unique; a name is only accepted when exactly one model carries it.
  return (
    models.find((m) => m.id === q) ??
    unique(models.filter((m) => m.id.toLowerCase() === lower)) ??
    unique(models.filter((m) => strip(m.name) === strip(lower))) ??
    unique(models.filter((m) => strip(m.path.split(/[\\/]/).pop() ?? '') === strip(lower))) ??
    unique(models.filter((m) => m.name.toLowerCase().includes(lower) || m.id.toLowerCase().includes(lower)))
  )
}

export interface UsageInfo {
  promptTokens?: number
  completionTokens?: number
  tokensPerSecond?: number
}

/** Reads token usage from a completion response body, or from the tail of an SSE stream. */
export function parseUsage(text: string, stream: boolean): UsageInfo {
  const out: UsageInfo = {}
  const apply = (j: unknown): void => {
    const o = j as {
      usage?: { prompt_tokens?: number; completion_tokens?: number; input_tokens?: number; output_tokens?: number }
      timings?: { prompt_n?: number; predicted_n?: number; predicted_per_second?: number; cache_n?: number }
    }
    if (o?.usage) {
      out.promptTokens = o.usage.prompt_tokens ?? o.usage.input_tokens ?? out.promptTokens
      out.completionTokens = o.usage.completion_tokens ?? o.usage.output_tokens ?? out.completionTokens
    }
    if (o?.timings) {
      if (o.timings.prompt_n !== undefined) out.promptTokens = o.timings.prompt_n + (o.timings.cache_n ?? 0)
      if (o.timings.predicted_n !== undefined) out.completionTokens = o.timings.predicted_n
      if (o.timings.predicted_per_second !== undefined) out.tokensPerSecond = o.timings.predicted_per_second
    }
  }
  if (!stream) {
    try {
      apply(JSON.parse(text))
    } catch {
      // not JSON
    }
    return out
  }
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith('data:')) continue
    const payload = line.slice(5).trim()
    if (!payload || payload === '[DONE]') continue
    try {
      apply(JSON.parse(payload))
    } catch {
      // partial line at the start of the tail buffer
    }
  }
  return out
}

/** True when the request's credentials match the configured key (any key passes when none is set). */
export function authorized(apiKey: string, headers: { authorization?: string; 'x-api-key'?: string }): boolean {
  if (!apiKey) return true
  const bearer = headers.authorization?.match(/^Bearer\s+(.+)$/i)?.[1]?.trim()
  return bearer === apiKey || headers['x-api-key']?.trim() === apiKey
}

/** OpenAI-style error body. */
export function errorBody(message: string, type = 'invalid_request_error'): string {
  return JSON.stringify({ error: { message, type } })
}

/**
 * The load settings other apps may change over the API. File paths and free-form arguments are
 * left out on purpose: the server can listen on the LAN, and those would let a client point
 * llama-server at anything on disk.
 */
const remoteLoadPatchSchema = z
  .object({
    contextLength: z.number().int().min(512).max(4_194_304).nullable(),
    gpuLayers: z.union([z.literal('auto'), z.literal('all'), z.number().int().min(0).max(1000)]),
    flashAttention: z.enum(['auto', 'on', 'off']),
    cacheTypeK: z.enum(KV_CACHE_TYPES as [string, ...string[]]),
    cacheTypeV: z.enum(KV_CACHE_TYPES as [string, ...string[]]),
    parallel: z.number().int().min(1).max(64),
    useDraft: z.boolean()
  })
  .partial()
  .strict()

export type RemoteLoadPatch = Partial<
  Pick<LoadConfig, 'contextLength' | 'gpuLayers' | 'flashAttention' | 'cacheTypeK' | 'cacheTypeV' | 'parallel' | 'useDraft'>
>

/** Checks a load-settings change sent to the API; unlike saved settings, bad values are refused, not replaced. */
export function parseRemoteLoadPatch(raw: unknown): { ok: true; patch: RemoteLoadPatch } | { ok: false; error: string } {
  const r = remoteLoadPatchSchema.safeParse(raw)
  if (r.success) return { ok: true, patch: r.data as RemoteLoadPatch }
  const issue = r.error.issues[0]
  const where = issue.path.join('.') || 'config'
  return { ok: false, error: `${where}: ${issue.message}` }
}

/** The part of a load config the API shows and accepts. */
export function remoteLoadView(config: LoadConfig): Required<RemoteLoadPatch> {
  const { contextLength, gpuLayers, flashAttention, cacheTypeK, cacheTypeV, parallel, useDraft } = config
  return { contextLength, gpuLayers, flashAttention, cacheTypeK, cacheTypeV, parallel, useDraft }
}
