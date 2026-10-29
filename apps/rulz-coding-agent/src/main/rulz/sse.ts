import type { WireToolCall } from '@shared/types'

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
  /** Speculative decoding: tokens the draft (MTP head or draft model) proposed, and how many the model accepted. */
  draft_n?: number
  draft_n_accepted?: number
}

/** One turn's generation numbers, for the draft stats. */
export interface GenTimings {
  generated: number
  genMs: number
  /** Absent when the server isn't drafting. */
  drafted?: number
  accepted?: number
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
  /** Exact token counts, from llama.cpp's timings or an OpenAI-style usage block. */
  usage?: { prompt: number; completion: number }
  gen?: GenTimings
  progress?: { processed: number; total: number }
  error?: string
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
    usage?: { prompt_tokens?: number; completion_tokens?: number } | null
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
  if (j.timings?.prompt_n !== undefined) {
    // prompt_n counts only the part that wasn't cached.
    out.usage = { prompt: (j.timings.cache_n ?? 0) + j.timings.prompt_n, completion: j.timings.predicted_n ?? 0 }
  } else if (typeof j.usage?.prompt_tokens === 'number') {
    out.usage = { prompt: j.usage.prompt_tokens, completion: j.usage.completion_tokens ?? 0 }
  }
  if (j.timings?.predicted_n !== undefined && j.timings.predicted_ms !== undefined) {
    out.gen = { generated: j.timings.predicted_n, genMs: j.timings.predicted_ms }
    if (j.timings.draft_n !== undefined) {
      out.gen.drafted = j.timings.draft_n
      out.gen.accepted = j.timings.draft_n_accepted ?? 0
    }
  }
  if (j.prompt_progress?.total) {
    const cache = j.prompt_progress.cache ?? 0
    out.progress = {
      processed: Math.max(0, (j.prompt_progress.processed ?? 0) - cache),
      total: Math.max(1, j.prompt_progress.total - cache)
    }
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
