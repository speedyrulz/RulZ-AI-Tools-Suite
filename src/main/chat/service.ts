import type { ChatRequest, GenerationStats, LoadedInstance, SamplingParams, StreamEvent, WireMessage } from '@shared/types'
import type { InstanceManager } from '../engine/instances'
import { fitToContext, IMAGE_TOKENS, imageCount, textOf, type ContextApi } from './context'
import { errorFromBody, interpretChunk, SseParser, statsFromTimings } from './sse'

export function buildRequestBody(alias: string, messages: WireMessage[], s: SamplingParams): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: alias,
    messages,
    stream: true,
    return_progress: true,
    temperature: s.temperature,
    top_k: s.topK,
    top_p: s.topP,
    min_p: s.minP,
    repeat_penalty: s.repeatPenalty,
    presence_penalty: s.presencePenalty,
    frequency_penalty: s.frequencyPenalty
  }
  if (s.maxTokens !== null) body.max_tokens = s.maxTokens
  if (s.stop.length) body.stop = s.stop
  if (s.seed !== null) body.seed = s.seed
  if (s.jsonSchema.trim()) {
    body.response_format = { type: 'json_schema', json_schema: { name: 'output', schema: JSON.parse(s.jsonSchema) } }
  }
  return body
}

/** Streams chat completions from a loaded instance and forwards them to the renderer. */
export class ChatService {
  private active = new Map<string, AbortController>()

  constructor(
    private readonly instances: InstanceManager,
    private readonly send: (event: StreamEvent) => void
  ) {}

  abort(requestId: string): void {
    this.active.get(requestId)?.abort()
  }

  contextApi(instance: LoadedInstance): ContextApi {
    const base = this.instances.baseUrl(instance)
    const tokenize = async (content: string): Promise<number | null> => {
      const res = await fetch(`${base}/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, add_special: true }),
        signal: AbortSignal.timeout(15000)
      })
      if (!res.ok) return null
      const j = (await res.json()) as { tokens?: unknown[] }
      return j.tokens?.length ?? null
    }
    return {
      countText: (text) => tokenize(text).catch(() => null),
      countConversation: async (messages) => {
        try {
          // Images are replaced by a placeholder for counting; /apply-template only needs text.
          const textOnly = messages.map((m) => ({ role: m.role, content: textOf(m) }))
          const res = await fetch(`${base}/apply-template`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: textOnly }),
            signal: AbortSignal.timeout(15000)
          })
          if (!res.ok) return null
          const { prompt } = (await res.json()) as { prompt?: string }
          if (typeof prompt !== 'string') return null
          const n = await tokenize(prompt)
          return n === null ? null : n + messages.reduce((a, m) => a + imageCount(m) * IMAGE_TOKENS, 0)
        } catch {
          return null
        }
      }
    }
  }

  async countTokens(modelId: string, messages: WireMessage[]): Promise<number | null> {
    const inst = this.instances.readyFor(modelId)
    if (!inst) return null
    return this.contextApi(inst).countConversation(messages)
  }

  async start(req: ChatRequest): Promise<void> {
    const { requestId } = req
    const ctrl = new AbortController()
    this.active.set(requestId, ctrl)
    const t0 = performance.now()
    let firstTokenAt: number | null = null
    let reasoningStart: number | null = null
    let reasoningEnd: number | null = null
    let finishReason: string | undefined
    let stats: GenerationStats = {}
    let predicted = 0

    try {
      const inst = this.instances.readyFor(req.modelId)
      if (!inst) throw new Error('The model for this chat is not loaded.')
      stats.modelName = inst.modelName

      let messages = req.messages
      if (inst.contextLength) {
        const reserve = req.sampling.maxTokens ?? Math.min(2048, Math.floor(inst.contextLength / 4))
        messages = await fitToContext(messages, inst.contextLength, reserve, req.overflow, this.contextApi(inst))
      }

      const res = await fetch(`${this.instances.baseUrl(inst)}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildRequestBody(inst.modelName, messages, req.sampling)),
        signal: ctrl.signal
      })
      if (!res.ok || !res.body) throw new Error(errorFromBody(res.status, await res.text()))

      const parser = new SseParser()
      const decoder = new TextDecoder()
      const reader = res.body.getReader()
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        for (const data of parser.push(decoder.decode(value, { stream: true }))) {
          if (data === '[DONE]') continue
          let json: unknown
          try {
            json = JSON.parse(data)
          } catch {
            continue
          }
          const c = interpretChunk(json)
          if (c.error) throw new Error(c.error)
          if (c.progress) this.send({ type: 'progress', requestId, ...c.progress })
          if (c.reasoning || c.content) {
            const now = performance.now()
            firstTokenAt ??= now
            if (c.reasoning) reasoningStart ??= now
            if (c.content && reasoningStart !== null) reasoningEnd ??= now
            predicted++
            this.send({ type: 'delta', requestId, content: c.content, reasoning: c.reasoning })
          }
          if (c.finishReason) finishReason = c.finishReason
          if (c.timings) stats = statsFromTimings(c.timings, stats)
        }
      }
      this.send({ type: 'done', requestId, stats: this.finalStats(stats, t0, firstTokenAt, predicted, finishReason), reasoningMs: reasoningMs() })
    } catch (err) {
      const final = this.finalStats(stats, t0, firstTokenAt, predicted, ctrl.signal.aborted ? 'stopped' : 'error')
      if (ctrl.signal.aborted) this.send({ type: 'done', requestId, stats: final, reasoningMs: reasoningMs() })
      else this.send({ type: 'error', requestId, error: (err as Error).message, stats: final })
    } finally {
      this.active.delete(requestId)
    }

    function reasoningMs(): number | undefined {
      if (reasoningStart === null) return undefined
      return Math.round((reasoningEnd ?? performance.now()) - reasoningStart)
    }
  }

  private finalStats(
    stats: GenerationStats,
    t0: number,
    firstTokenAt: number | null,
    predicted: number,
    stopReason: string | undefined
  ): GenerationStats {
    const end = performance.now()
    const genSeconds = firstTokenAt !== null ? (end - firstTokenAt) / 1000 : 0
    return {
      ...stats,
      ttftMs: firstTokenAt !== null ? Math.round(firstTokenAt - t0) : undefined,
      totalMs: Math.round(end - t0),
      predictedTokens: stats.predictedTokens ?? predicted,
      tokensPerSecond: stats.tokensPerSecond ?? (genSeconds > 0 ? predicted / genSeconds : undefined),
      stopReason: stopReason ?? stats.stopReason
    }
  }
}
