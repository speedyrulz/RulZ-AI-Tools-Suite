import type { ChatRequest, GenerationStats, LoadedInstance, SamplingParams, StreamEvent, WireMessage } from '@shared/types'
import type { InstanceManager } from '../engine/instances'
import { fitToContext, IMAGE_TOKENS, imageCount, textOf, type ContextApi } from './context'
import { errorFromBody, interpretChunk, SseParser, statsFromTimings, ToolCallAccumulator } from './sse'

export function buildRequestBody(
  alias: string,
  messages: WireMessage[],
  s: SamplingParams,
  disableThinking = false
): Record<string, unknown> {
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
  if (disableThinking) body.chat_template_kwargs = { enable_thinking: false }
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
    private readonly send: (event: StreamEvent) => void,
    /** Tool definitions to offer when a request asks for tools. */
    private readonly getTools: () => unknown[] = () => []
  ) {}

  abort(requestId: string): void {
    this.active.get(requestId)?.abort()
  }

  /** Token count of the tool definitions sent with each request, cached per model and tool set. */
  private toolTokenCache = new Map<string, number>()

  /** `tools` are the definitions sent with the request; they take context space too. */
  contextApi(instance: LoadedInstance, tools: unknown[] = []): ContextApi {
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
    const toolTokens = async (): Promise<number> => {
      if (!tools.length) return 0
      const json = JSON.stringify(tools)
      const key = `${instance.instanceId}:${json}`
      let n = this.toolTokenCache.get(key)
      if (n === undefined) {
        n = (await tokenize(json).catch(() => null)) ?? Math.ceil(json.length / 3)
        this.toolTokenCache.set(key, n)
      }
      return n
    }
    return {
      countTools: toolTokens,
      countText: (text) => tokenize(text).catch(() => null),
      countConversation: async (messages) => {
        try {
          // Images are replaced by a placeholder for counting; /apply-template only needs text.
          // Tool messages are counted as plain text so templates without tool support still apply.
          const textOnly = messages.map((m) => ({ role: m.role === 'tool' ? 'user' : m.role, content: textOf(m) }))
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
          if (n === null) return null
          return n + messages.reduce((a, m) => a + imageCount(m) * IMAGE_TOKENS, 0) + (await toolTokens())
        } catch {
          return null
        }
      }
    }
  }

  async countTokens(modelId: string, messages: WireMessage[], useTools = false): Promise<number | null> {
    const inst = this.instances.readyFor(modelId)
    if (!inst) return null
    return this.contextApi(inst, useTools ? this.getTools() : []).countConversation(messages)
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
      const tools = req.useTools ? this.getTools() : []
      if (inst.contextLength) {
        const api = this.contextApi(inst, tools)
        // Tool definitions are resent with every message; if they alone eat most of the context,
        // no amount of trimming the chat helps, so say so plainly.
        const toolTokens = await api.countTools!()
        if (toolTokens > inst.contextLength * 0.6) {
          throw new Error(
            `The enabled MCP tools take about ${toolTokens.toLocaleString()} tokens of descriptions, more than this model's ` +
              `${inst.contextLength.toLocaleString()}-token context can hold alongside the chat. Disable some tools on the MCP Tools page, ` +
              'turn tools off for chats, or load the model with a larger context.'
          )
        }
        // Room kept for the reply; capped so a large max-tokens setting can't crowd out the whole chat.
        const reserve = Math.min(
          req.sampling.maxTokens ?? Math.min(2048, Math.floor(inst.contextLength / 4)),
          Math.floor(inst.contextLength / 3)
        )
        messages = await fitToContext(messages, inst.contextLength, reserve, req.overflow, api)
      }

      const body = buildRequestBody(inst.modelName, messages, req.sampling, req.disableThinking)
      if (tools.length) body.tools = tools
      const res = await fetch(`${this.instances.baseUrl(inst)}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: ctrl.signal
      })
      if (!res.ok || !res.body) throw new Error(errorFromBody(res.status, await res.text()))

      const parser = new SseParser()
      const toolCalls = new ToolCallAccumulator()
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
          if (c.toolCalls) {
            firstTokenAt ??= performance.now()
            toolCalls.push(c.toolCalls)
          }
          if (c.finishReason) finishReason = c.finishReason
          if (c.timings) stats = statsFromTimings(c.timings, stats)
        }
      }
      const calls = toolCalls.result()
      this.send({
        type: 'done',
        requestId,
        stats: this.finalStats(stats, t0, firstTokenAt, predicted, finishReason),
        reasoningMs: reasoningMs(),
        toolCalls: calls.length ? calls : undefined
      })
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
