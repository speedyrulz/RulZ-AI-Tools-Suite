import type { GenerationStats, WireToolCall } from '@shared/types'

/** Incremental parser for text/event-stream bodies; returns each complete `data:` payload. */
export class SseParser {
  private buf = ''

  push(chunk: string): string[] {
    this.buf += chunk
    const out: string[] = []
    let idx: number
    while ((idx = this.buf.indexOf('\n')) >= 0) {
      let line = this.buf.slice(0, idx)
      this.buf = this.buf.slice(idx + 1)
      if (line.endsWith('\r')) line = line.slice(0, -1)
      if (line.startsWith('data:')) out.push(line.slice(5).trimStart())
      else if (line.startsWith('error:')) out.push(JSON.stringify({ error: safeJson(line.slice(6).trim()) }))
    }
    return out
  }
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return { message: text }
  }
}

interface Timings {
  prompt_n?: number
  prompt_ms?: number
  predicted_n?: number
  predicted_ms?: number
  predicted_per_second?: number
  cache_n?: number
}

/** The `usage` object of an OpenAI-style final chunk (OpenRouter fills cost too). */
export interface ChunkUsage {
  prompt_tokens?: number
  completion_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number }
  cost?: number
}

export interface ToolCallDelta {
  index: number
  id?: string
  name?: string
  arguments?: string
}

export interface ChunkInfo {
  content?: string
  reasoning?: string
  toolCalls?: ToolCallDelta[]
  finishReason?: string
  timings?: Timings
  progress?: { processed: number; total: number }
  error?: string
  /** The code of an error chunk's body (a CloudErrorKind when it came from the OpenRouter side). */
  errorCode?: string
  usage?: ChunkUsage
}

/** Extracts what we need from one /v1/chat/completions stream chunk. */
export function interpretChunk(json: unknown): ChunkInfo {
  const j = json as {
    choices?: {
      delta?: {
        content?: string | null
        reasoning_content?: string | null
        tool_calls?: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[]
      }
      finish_reason?: string | null
    }[]
    timings?: Timings
    prompt_progress?: { total?: number; processed?: number; cache?: number }
    error?: { message?: string; code?: unknown } | string
    usage?: ChunkUsage
  }
  const out: ChunkInfo = {}
  if (j.error) {
    out.error = typeof j.error === 'string' ? j.error : (j.error.message ?? JSON.stringify(j.error))
    if (typeof j.error === 'object' && typeof j.error.code === 'string') out.errorCode = j.error.code
    return out
  }
  if (j.usage && typeof j.usage === 'object') out.usage = j.usage
  const choice = j.choices?.[0]
  if (choice?.delta?.content) out.content = choice.delta.content
  if (choice?.delta?.reasoning_content) out.reasoning = choice.delta.reasoning_content
  if (choice?.delta?.tool_calls?.length) {
    out.toolCalls = choice.delta.tool_calls.map((t, i) => ({
      index: t.index ?? i,
      id: t.id,
      name: t.function?.name,
      arguments: t.function?.arguments
    }))
  }
  if (choice?.finish_reason) out.finishReason = choice.finish_reason
  if (j.timings) out.timings = j.timings
  if (j.prompt_progress?.total) {
    const cache = j.prompt_progress.cache ?? 0
    out.progress = {
      processed: Math.max(0, (j.prompt_progress.processed ?? 0) - cache),
      total: Math.max(1, j.prompt_progress.total - cache)
    }
  }
  return out
}

export function statsFromTimings(t: Timings | undefined, extra: GenerationStats): GenerationStats {
  return {
    ...extra,
    tokensPerSecond: t?.predicted_per_second ?? extra.tokensPerSecond,
    predictedTokens: t?.predicted_n ?? extra.predictedTokens,
    promptTokens: t?.prompt_n !== undefined ? t.prompt_n + (t.cache_n ?? 0) : extra.promptTokens,
    cachedTokens: t?.cache_n ?? extra.cachedTokens,
    promptTokensPerSecond: t?.prompt_n && t.prompt_ms && t.prompt_n > 0 && t.prompt_ms > 0 ? (t.prompt_n / t.prompt_ms) * 1000 : extra.promptTokensPerSecond
  }
}

/**
 * Stats from OpenRouter's usage, for what llama-server's timings did not fill: tokens, cache hits, the cost (rounded up to a
 * whole micro-dollar; the gateway has already added the provider's part of a BYOK request to it) and tokens/s.
 */
export function statsFromUsage(u: ChunkUsage, extra: GenerationStats, genSeconds: number): GenerationStats {
  const out: GenerationStats = { ...extra }
  if (typeof u.prompt_tokens === 'number') out.promptTokens = u.prompt_tokens
  if (typeof u.completion_tokens === 'number') {
    out.predictedTokens = u.completion_tokens
    if (out.tokensPerSecond === undefined && genSeconds > 0) out.tokensPerSecond = u.completion_tokens / genSeconds
  }
  const cached = u.prompt_tokens_details?.cached_tokens
  if (typeof cached === 'number') out.cachedTokens = cached
  if (typeof u.cost === 'number' && u.cost >= 0) {
    // toPrecision first: 0.000003 * 1e6 is 3.0000000000000004 and must not become 4.
    out.costMicro = Math.ceil(Number((u.cost * 1e6).toPrecision(12)))
  }
  return out
}

/** Merges streamed tool-call fragments (keyed by index) into complete calls. */
export class ToolCallAccumulator {
  private calls = new Map<number, { id: string; name: string; arguments: string }>()

  push(deltas: ToolCallDelta[]): void {
    for (const d of deltas) {
      const cur = this.calls.get(d.index) ?? { id: '', name: '', arguments: '' }
      if (d.id) cur.id = d.id
      if (d.name) cur.name += d.name
      if (d.arguments) cur.arguments += d.arguments
      this.calls.set(d.index, cur)
    }
  }

  result(): WireToolCall[] {
    return [...this.calls.entries()]
      .sort(([a], [b]) => a - b)
      .filter(([, c]) => c.name)
      .map(([i, c]) => ({ id: c.id || `call_${i}`, type: 'function' as const, function: { name: c.name, arguments: c.arguments } }))
  }
}

/** Pulls a readable message out of an error response body. */
export function errorFromBody(status: number, text: string): string {
  try {
    const j = JSON.parse(text) as { error?: { message?: string } | string; message?: string }
    const msg = typeof j.error === 'string' ? j.error : (j.error?.message ?? j.message)
    if (msg) return msg
  } catch {
    // not JSON
  }
  return text.trim() || `HTTP ${status}`
}

/** The `error.code` of an error response body when it is a string (the cloud gateway's CloudErrorKind). */
export function errorCodeFromBody(text: string): string | undefined {
  try {
    const j = JSON.parse(text) as { error?: { code?: unknown } | string }
    if (j.error && typeof j.error === 'object' && typeof j.error.code === 'string') return j.error.code
  } catch {
    // not JSON
  }
  return undefined
}
