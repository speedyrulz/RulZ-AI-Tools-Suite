import { createHash } from 'node:crypto'
import type {
  ChatRequest,
  GenerationStats,
  LoadedInstance,
  SamplingParams,
  StreamEvent,
  ToolRequest,
  WireMessage
} from '@shared/types'
import type { InstanceManager } from '../engine/instances'
import { fitToContext, IMAGE_TOKENS, imageCount, textOf, type ContextApi } from './context'
import { errorFromBody, interpretChunk, SseParser, statsFromTimings, ToolCallAccumulator } from './sse'
import { jsonifyToolCallArgs, templateStringifiesArgs } from './toolArgs'

/** Streamed text is sent to the renderer at most this often (one IPC message per token is wasteful). */
const DELTA_FLUSH_MS = 30

/**
 * Collects streamed text and sends it in batches. Anything else sent for the same request goes
 * through `before()` first so the renderer still sees events in order.
 */
export class DeltaBatcher {
  private content = ''
  private reasoning = ''
  private timer: ReturnType<typeof setTimeout> | null = null

  constructor(
    private readonly emit: (content: string, reasoning: string) => void,
    private readonly intervalMs = DELTA_FLUSH_MS
  ) {}

  push(content: string | undefined, reasoning: string | undefined): void {
    if (content) this.content += content
    if (reasoning) this.reasoning += reasoning
    this.timer ??= setTimeout(() => this.flush(), this.intervalMs)
  }

  flush(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
    if (!this.content && !this.reasoning) return
    const { content, reasoning } = this
    this.content = ''
    this.reasoning = ''
    this.emit(content, reasoning)
  }
}

function hashOf(text: string): string {
  return createHash('sha1').update(text).digest('hex')
}

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
  const thinking = disableThinking ? 'off' : s.thinking
  if (thinking !== 'auto') body.chat_template_kwargs = { enable_thinking: thinking === 'on' }
  if (thinking !== 'off') {
    if (s.reasoningEffort !== 'default') body.reasoning_effort = s.reasoningEffort
    if (s.thinkingBudget !== null) body.thinking_budget_tokens = s.thinkingBudget
    if (s.thinkingBudget !== null && s.thinkingBudgetMessage.trim()) body.reasoning_budget_message = s.thinkingBudgetMessage
  }
  if (s.jsonSchema.trim()) {
    body.response_format = { type: 'json_schema', json_schema: { name: 'output', schema: JSON.parse(s.jsonSchema) } }
  }
  return body
}

export interface ToolSet {
  defs: unknown[]
  /** Extra system-prompt text (the on-demand tool catalog). */
  note?: string
}

/** Appends `note` to the leading system message, or adds one. */
export function withSystemNote(messages: WireMessage[], note: string | undefined): WireMessage[] {
  if (!note) return messages
  const first = messages[0]
  if (first?.role === 'system' && typeof first.content === 'string') {
    return [{ ...first, content: `${first.content}\n\n${note}` }, ...messages.slice(1)]
  }
  return [{ role: 'system', content: note }, ...messages]
}

/** Streams chat completions from a loaded instance and forwards them to the renderer. */
export class ChatService {
  private active = new Map<string, AbortController>()

  constructor(
    private readonly instances: InstanceManager,
    private readonly send: (event: StreamEvent) => void,
    /** Tool definitions (and, in on-demand mode, a system note) for a request's tool selection. */
    private readonly getTools: (req: ToolRequest) => ToolSet = () => ({ defs: [] })
  ) {}

  /** Per loaded instance: does its chat template print tool-call arguments in Python form? */
  private argQuirk = new Map<string, Promise<boolean>>()

  /** Drops what was cached for an instance that has been unloaded. */
  forget(instanceId: string): void {
    this.argQuirk.delete(instanceId)
    this.toolTokenCache.delete(instanceId)
  }

  private stringifiesArgs(instance: LoadedInstance): Promise<boolean> {
    let p = this.argQuirk.get(instance.instanceId)
    if (!p) {
      p = fetch(`${this.instances.baseUrl(instance)}/props`, { signal: AbortSignal.timeout(10000) })
        .then(async (r) => (r.ok ? templateStringifiesArgs(((await r.json()) as { chat_template?: string }).chat_template) : false))
        .catch(() => false)
      this.argQuirk.set(instance.instanceId, p)
    }
    return p
  }

  private toolSet(req: ToolRequest | null | undefined): ToolSet {
    return req ? this.getTools(req) : { defs: [] }
  }

  abort(requestId: string): void {
    this.active.get(requestId)?.abort()
  }

  /** Token count of the tool definitions sent with each request: per instance, by hash of the tool set. */
  private toolTokenCache = new Map<string, Map<string, number>>()

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
      const key = hashOf(json)
      let perInstance = this.toolTokenCache.get(instance.instanceId)
      if (!perInstance) this.toolTokenCache.set(instance.instanceId, (perInstance = new Map()))
      let n = perInstance.get(key)
      if (n === undefined) {
        n = (await tokenize(json).catch(() => null)) ?? Math.ceil(json.length / 3)
        perInstance.set(key, n)
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

  async countTokens(modelId: string, messages: WireMessage[], tools: ToolRequest | null = null): Promise<number | null> {
    const inst = this.instances.readyFor(modelId)
    if (!inst) return null
    const set = this.toolSet(tools)
    return this.contextApi(inst, set.defs).countConversation(withSystemNote(messages, set.note))
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
    const deltas = new DeltaBatcher((content, reasoning) =>
      this.send({ type: 'delta', requestId, content: content || undefined, reasoning: reasoning || undefined })
    )

    try {
      const inst = this.instances.readyFor(req.modelId)
      if (!inst) throw new Error('The model for this chat is not loaded.')
      stats.modelName = inst.modelName

      const set = this.toolSet(req.tools)
      const tools = set.defs
      let messages = withSystemNote(req.messages, set.note)
      if (inst.contextLength) {
        const api = this.contextApi(inst, tools)
        // Tool definitions are resent with every message; if they alone eat most of the context,
        // no amount of trimming the chat helps, so say so plainly.
        const toolTokens = await api.countTools!()
        if (toolTokens > inst.contextLength * 0.6) {
          throw new Error(
            `The enabled MCP tools take about ${toolTokens.toLocaleString()} tokens of descriptions, more than this model's ` +
              `${inst.contextLength.toLocaleString()}-token context can hold alongside the chat. On the MCP Tools page, set Tool loading ` +
              'to "On demand" or "Auto", or disable some tools; or load the model with a larger context.'
          )
        }
        // Room kept for the reply; capped so a large max-tokens setting can't crowd out the whole chat.
        const reserve = Math.min(
          req.sampling.maxTokens ?? Math.min(2048, Math.floor(inst.contextLength / 4)),
          Math.floor(inst.contextLength / 3)
        )
        messages = await fitToContext(messages, inst.contextLength, reserve, req.overflow, api)
      }

      // Some templates print earlier tool arguments as Python lists; send them as JSON text instead
      // so the model keeps writing JSON the server can parse.
      if (messages.some((m) => m.tool_calls?.length) && (await this.stringifiesArgs(inst))) {
        messages = jsonifyToolCallArgs(messages)
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
          if (c.progress) {
            deltas.flush()
            this.send({ type: 'progress', requestId, ...c.progress })
          }
          if (c.reasoning || c.content) {
            const now = performance.now()
            firstTokenAt ??= now
            if (c.reasoning) reasoningStart ??= now
            if (c.content && reasoningStart !== null) reasoningEnd ??= now
            predicted++
            deltas.push(c.content, c.reasoning)
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
      deltas.flush()
      this.send({
        type: 'done',
        requestId,
        stats: this.finalStats(stats, t0, firstTokenAt, predicted, finishReason),
        reasoningMs: reasoningMs(),
        toolCalls: calls.length ? calls : undefined
      })
    } catch (err) {
      deltas.flush()
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
