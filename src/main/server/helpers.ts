// Pure helpers for the local API server (unit-tested separately from the HTTP code).
import type { ModelEntry } from '@shared/types'

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
