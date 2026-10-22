import type { GenerationStats } from '@shared/types'

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

export interface ChunkInfo {
  content?: string
  reasoning?: string
  finishReason?: string
  timings?: Timings
  progress?: { processed: number; total: number }
  error?: string
}

/** Extracts what we need from one /v1/chat/completions stream chunk. */
export function interpretChunk(json: unknown): ChunkInfo {
  const j = json as {
    choices?: { delta?: { content?: string | null; reasoning_content?: string | null }; finish_reason?: string | null }[]
    timings?: Timings
    prompt_progress?: { total?: number; processed?: number; cache?: number }
    error?: { message?: string } | string
  }
  const out: ChunkInfo = {}
  if (j.error) {
    out.error = typeof j.error === 'string' ? j.error : (j.error.message ?? JSON.stringify(j.error))
    return out
  }
  const choice = j.choices?.[0]
  if (choice?.delta?.content) out.content = choice.delta.content
  if (choice?.delta?.reasoning_content) out.reasoning = choice.delta.reasoning_content
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
    cachedTokens: t?.cache_n ?? extra.cachedTokens
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
