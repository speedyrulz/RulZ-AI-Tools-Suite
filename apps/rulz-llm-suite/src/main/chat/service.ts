import { createHash } from 'node:crypto'
import { requestLora } from '@shared/adapters'
import { isFatalCloudError, isLocalInstance } from '@shared/cloud'
import type {
  ChatRequest,
  GenerationStats,
  LoadedInstance,
  SamplingParams,
  StreamEvent,
  ToolRequest,
  WireMessage,
  WireToolCall
} from '@shared/types'
import type { InstanceManager } from '../engine/instances'
import { parseChatAdapters } from '../settings/schema'
import { fitToContext, IMAGE_TOKENS, imageCount, textOf, type ContextApi } from './context'
import { errorCodeFromBody, errorFromBody, interpretChunk, SseParser, statsFromTimings, statsFromUsage, ToolCallAccumulator, type ChunkUsage } from './sse'
import { jsonifyToolCallArgs, templateStringifiesArgs } from './toolArgs'

type UseBy = 'window' | 'api' | 'phone'

/** True while any model running on this PC is answering something (a chat reply, an API request, a tool turn, a Writing job). */
export function localModelInUse(instances: Pick<InstanceManager, 'list' | 'usersOf'>): boolean {
  return instances.list().some((i) => isLocalInstance(i) && instances.usersOf(i.instanceId) > 0)
}

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

/** Message token counts remembered per loaded instance, so a long chat isn't re-tokenized every round. */
const COUNT_CACHE_SIZE = 4000
/** /tokenize requests in flight at once per instance; a long chat would otherwise send hundreds together. */
const TOKENIZE_CONCURRENCY = 4
/** Longest a model stays held between tool rounds without hearing from the renderer. */
const TURN_HOLD_MAX_MS = 30 * 60_000

/** Chats whose instance is remembered (see ChatRequest.sessionKey). */
const SESSION_MEMORY = 500

/** The connection to the server broke; says nothing about the model. */
class TransportError extends Error {}

/** An error OpenRouter's side reported, with the code of its error body (CloudErrorKind); never a transport error. */
class CloudRequestError extends Error {
  constructor(
    message: string,
    readonly code: string | undefined
  ) {
    super(message)
  }
}

const TRANSPORT_CODES = new Set(['ECONNRESET', 'ECONNREFUSED', 'ECONNABORTED', 'EPIPE', 'ETIMEDOUT', 'UND_ERR_SOCKET', 'UND_ERR_CLOSED', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT'])

/** A failed fetch or a stream cut off in the middle (node reports both as a TypeError, with the socket error as its cause). */
export function isTransportError(err: unknown): boolean {
  if (err instanceof TransportError) return true
  if (!(err instanceof Error)) return false
  const code = (err as { code?: string }).code ?? (err.cause as { code?: string } | undefined)?.code
  if (code && TRANSPORT_CODES.has(code)) return true
  return err instanceof TypeError && /fetch failed|terminated|network|socket|closed|reset/i.test(err.message)
}

type Limiter = <T>(task: () => Promise<T>) => Promise<T>

/** Runs at most `n` tasks at once; the rest wait their turn. */
export function limiter(n: number): Limiter {
  let running = 0
  const waiting: (() => void)[] = []
  return async (task) => {
    if (running < n) running++
    else await new Promise<void>((r) => waiting.push(r))
    try {
      return await task()
    } finally {
      // Hand the slot straight to the next task, so a newcomer can't slip in between.
      const next = waiting.shift()
      if (next) next()
      else running--
    }
  }
}

/**
 * logit_bias entries that discourage each word, with and without a leading space (llama-server biases
 * every token of a string key). Empty and duplicate entries are skipped; strength 0 sends nothing.
 */
export function avoidWordsBias(words: string[], strength: number): [string, number][] {
  if (!(strength > 0)) return []
  const out: [string, number][] = []
  const seen = new Set<string>()
  for (const raw of words) {
    const w = raw.trim()
    if (!w) continue
    for (const key of [w, ' ' + w]) {
      if (seen.has(key)) continue
      seen.add(key)
      out.push([key, -strength])
    }
  }
  return out
}

/** The first sampler fields of every request (always sent). */
function baseSamplerFields(s: SamplingParams): Record<string, unknown> {
  return {
    temperature: s.temperature,
    top_k: s.topK,
    top_p: s.topP,
    min_p: s.minP,
    repeat_penalty: s.repeatPenalty,
    presence_penalty: s.presencePenalty,
    frequency_penalty: s.frequencyPenalty
  }
}

/** Fiction samplers and avoided words: only sent when on, so requests of people who never touch them stay as they were. */
function extraSamplerFields(s: SamplingParams): Record<string, unknown> {
  const body: Record<string, unknown> = {}
  if (s.dryMultiplier > 0) {
    body.dry_multiplier = s.dryMultiplier
    body.dry_base = s.dryBase
    body.dry_allowed_length = s.dryAllowedLength
    // -1 (the whole context) is llama-server's default, and some builds refuse it ("must be between 0 <= value"),
    // which failed every request with DRY on: leave it out then.
    if (s.dryPenaltyLastN >= 0) body.dry_penalty_last_n = s.dryPenaltyLastN
    if (s.drySequenceBreakers.length) body.dry_sequence_breakers = s.drySequenceBreakers
  }
  if (s.xtcProbability > 0) {
    body.xtc_probability = s.xtcProbability
    body.xtc_threshold = s.xtcThreshold
  }
  const bias = avoidWordsBias(s.avoidWords, s.avoidStrength)
  if (bias.length) body.logit_bias = bias
  return body
}

/** The thinking fields of a request: whether thinking is on, and how much of it. */
function thinkingFields(s: SamplingParams, disableThinking: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = {}
  const thinking = disableThinking ? 'off' : s.thinking
  if (thinking !== 'auto') body.chat_template_kwargs = { enable_thinking: thinking === 'on' }
  if (thinking !== 'off') {
    if (s.reasoningEffort !== 'default') body.reasoning_effort = s.reasoningEffort
    if (s.thinkingBudget !== null) body.thinking_budget_tokens = s.thinkingBudget
    if (s.thinkingBudget !== null && s.thinkingBudgetMessage.trim()) body.reasoning_budget_message = s.thinkingBudgetMessage
  }
  return body
}

/**
 * The request fields a sampling gives, apart from what belongs to one request or one chat (the model, messages,
 * streaming, JSON schema, reply length, stop words, seed, adapter strengths). The chat request is built from these, and
 * the API server shows them for a mode's own sampling, so the two cannot drift.
 */
export function samplingRequestFields(s: SamplingParams, disableThinking = false): { sampling: Record<string, unknown>; thinking: Record<string, unknown> } {
  return { sampling: { ...baseSamplerFields(s), ...extraSamplerFields(s) }, thinking: thinkingFields(s, disableThinking) }
}

export function buildRequestBody(
  alias: string,
  messages: WireMessage[],
  s: SamplingParams,
  disableThinking = false,
  lora: { id: number; scale: number }[] | null = null
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: alias,
    messages,
    stream: true,
    return_progress: true,
    ...baseSamplerFields(s)
  }
  // Adapter strengths for this request only; llama-server runs the adapters it leaves out at 0.
  if (lora) body.lora = lora
  if (s.maxTokens !== null) body.max_tokens = s.maxTokens
  if (s.stop.length) body.stop = s.stop
  if (s.seed !== null) body.seed = s.seed
  Object.assign(body, extraSamplerFields(s), thinkingFields(s, disableThinking))
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

/** One model round of an RE session, as recorded for training data. */
export interface RecordedRound {
  requestId: string
  at: number
  model: string
  /** Exactly what llama-server received (after the tool note, context fitting and argument fixes). */
  messages: WireMessage[]
  tools: unknown[]
  /** The context was trimmed to fit, so `messages` lacks part of the session. */
  trimmed: boolean
  response: {
    content: string
    reasoning: string
    toolCalls: WireToolCall[]
    finishReason?: string
  }
}

export type RoundRecorder = (conversationId: string, round: RecordedRound) => void

/** Streams chat completions from a loaded instance and forwards them to the renderer. */
export class ChatService {
  private active = new Map<string, AbortController>()

  /** True while any chat request is running (the window's, the API's or a phone job's). */
  get busy(): boolean {
    return this.active.size > 0
  }
  /** Releases for models kept in use between tool rounds, by the request that asked for the tools. */
  private turnHolds = new Map<string, () => void>()

  constructor(
    private readonly instances: InstanceManager,
    private readonly send: (event: StreamEvent) => void,
    /** Tool definitions (and, in on-demand mode, a system note) for a request's tool selection. */
    private readonly getTools: (req: ToolRequest) => ToolSet = () => ({ defs: [] }),
    /** Receives finished rounds of requests that ask to be recorded (RE sessions). */
    private readonly recorder?: RoundRecorder
  ) {}

  /** Per loaded instance: does its chat template print tool-call arguments in Python form? */
  private argQuirk = new Map<string, Promise<boolean>>()

  /** Drops what was cached for an instance that has been unloaded. */
  forget(instanceId: string): void {
    this.argQuirk.delete(instanceId)
    this.toolTokenCache.delete(instanceId)
    this.countCache.delete(instanceId)
    this.tokenizeLimits.delete(instanceId)
  }

  private countCache = new Map<string, Map<string, Promise<number | null>>>()
  private tokenizeLimits = new Map<string, Limiter>()

  /** Token count of `text` on an instance, remembered (least recently used goes first). */
  private cachedCount(instanceId: string, text: string, count: () => Promise<number | null>): Promise<number | null> {
    let cache = this.countCache.get(instanceId)
    if (!cache) this.countCache.set(instanceId, (cache = new Map()))
    const key = hashOf(text)
    const hit = cache.get(key)
    if (hit) {
      cache.delete(key)
      cache.set(key, hit)
      return hit
    }
    const p: Promise<number | null> = count().then((n) => {
      // Failures aren't remembered; the next round asks again.
      if (n === null && cache.get(key) === p) cache.delete(key)
      return n
    })
    cache.set(key, p)
    if (cache.size > COUNT_CACHE_SIZE) cache.delete(cache.keys().next().value!)
    return p
  }

  private stringifiesArgs(instance: LoadedInstance): Promise<boolean> {
    let p = this.argQuirk.get(instance.instanceId)
    if (!p) {
      // OpenRouter has no chat template to ask about; the adapter takes JSON arguments as they are.
      if (instance.cloud) return Promise.resolve(false)
      const mine: Promise<boolean> = fetch(`${this.instances.baseUrl(instance)}/props`, { signal: AbortSignal.timeout(10000) })
        .then(async (r) => {
          if (r.ok) return templateStringifiesArgs(((await r.json()) as { chat_template?: string }).chat_template)
          // A failed lookup is not remembered, so the next round asks again.
          if (this.argQuirk.get(instance.instanceId) === mine) this.argQuirk.delete(instance.instanceId)
          return false
        })
        .catch(() => {
          if (this.argQuirk.get(instance.instanceId) === mine) this.argQuirk.delete(instance.instanceId)
          return false
        })
      p = mine
      this.argQuirk.set(instance.instanceId, p)
    }
    return p
  }

  private toolSet(req: ToolRequest | null | undefined): ToolSet {
    return req ? this.getTools(req) : { defs: [] }
  }

  /**
   * Keeps a model in use between tool rounds: the renderer runs the tools and may wait on the
   * user's approval, and an idle-unload then would force a full reload for the next round. The
   * renderer calls releaseTurn once the next round has started or the turn ended; the timer
   * covers a renderer that never does (reloaded mid-turn).
   */
  private holdTurn(requestId: string, release: () => void): void {
    let timer: ReturnType<typeof setTimeout> | undefined
    const done = (): void => {
      clearTimeout(timer)
      this.turnHolds.delete(requestId)
      release()
    }
    timer = setTimeout(done, TURN_HOLD_MAX_MS)
    timer.unref?.()
    this.turnHolds.set(requestId, done)
  }

  releaseTurn(requestId: string): void {
    this.turnHolds.get(requestId)?.()
  }

  abort(requestId: string): void {
    this.active.get(requestId)?.abort()
  }

  /** Token count of the tool definitions sent with each request: per instance, by hash of the tool set. */
  private toolTokenCache = new Map<string, Map<string, number>>()

  /** `tools` are the definitions sent with the request; they take context space too. */
  contextApi(instance: LoadedInstance, tools: unknown[] = [], by: UseBy = 'window'): ContextApi {
    const limit = this.tokenizeLimits.get(instance.instanceId) ?? limiter(TOKENIZE_CONCURRENCY)
    this.tokenizeLimits.set(instance.instanceId, limit)
    const tokenize = (content: string): Promise<number | null> =>
      limit(async () => {
        const res = await this.instances.fetch(instance, '/tokenize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, add_special: true }),
          signal: AbortSignal.timeout(15000)
        }, by)
        if (!res.ok) return null
        const j = (await res.json()) as { tokens?: unknown[] }
        return j.tokens?.length ?? null
      })
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
      countText: (text) => this.cachedCount(instance.instanceId, text, () => tokenize(text).catch(() => null)),
      countConversation: async (messages) => {
        try {
          // Images are replaced by a placeholder for counting; /apply-template only needs text.
          // Tool messages are counted as plain text so templates without tool support still apply.
          const textOnly = messages.map((m) => ({ role: m.role === 'tool' ? 'user' : m.role, content: textOf(m) }))
          const res = await this.instances.fetch(instance, '/apply-template', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: textOnly }),
            signal: AbortSignal.timeout(15000)
          }, by)
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

  /**
   * The instance a request runs on: the pinned one (it must be ready), else the one this session used last while it
   * is ready, else the best ready one (docs/vast-design.md, section 7). A model may be loaded on several rentals.
   */
  private instanceFor(modelId: string, instanceId?: string, sessionKey?: string): LoadedInstance | undefined {
    if (instanceId) {
      const pinned = this.instances.get(instanceId)
      return pinned && pinned.status === 'ready' && pinned.modelId === modelId ? pinned : undefined
    }
    const inst = this.instances.readyFor(modelId, sessionKey ? this.sessions.get(sessionKey) : undefined)
    if (inst && sessionKey) {
      this.sessions.delete(sessionKey)
      this.sessions.set(sessionKey, inst.instanceId)
      if (this.sessions.size > SESSION_MEMORY) this.sessions.delete(this.sessions.keys().next().value!)
    }
    return inst
  }

  /** Session key → the instance its last request used (least recently used goes first). */
  private sessions = new Map<string, string>()

  async countTokens(modelId: string, messages: WireMessage[], tools: ToolRequest | null = null, instanceId?: string, by: UseBy = 'window'): Promise<number | null> {
    const inst = instanceId ? this.instanceFor(modelId, instanceId) : this.instances.readyFor(modelId)
    if (!inst) return null
    const set = this.toolSet(tools)
    return this.contextApi(inst, set.defs, by).countConversation(withSystemNote(messages, set.note))
  }

  /** Token count of plain text on the loaded model (cached per instance); null if it can't count. */
  async countText(modelId: string, text: string, instanceId?: string, by: UseBy = 'window'): Promise<number | null> {
    const inst = instanceId ? this.instanceFor(modelId, instanceId) : this.instances.readyFor(modelId)
    if (!inst) return null
    return this.contextApi(inst, [], by).countText(text)
  }

  /** `by` says who asked, so a rental can tell its owner what keeps it running. */
  async start(req: ChatRequest, by: UseBy = 'window'): Promise<void> {
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
    // The whole reply, for recording.
    let fullContent = ''
    let fullReasoning = ''
    const deltas = new DeltaBatcher((content, reasoning) =>
      this.send({ type: 'delta', requestId, content: content || undefined, reasoning: reasoning || undefined })
    )
    let release: (() => void) | undefined
    let inst: LoadedInstance | undefined
    // The instance this chat was pinned to is gone: the connection to it is what failed, not the model.
    let pinnedGone = false

    try {
      inst = this.instanceFor(req.modelId, req.instanceId, req.sessionKey)
      if (!inst) {
        pinnedGone = !!req.instanceId
        throw new Error(req.instanceId ? 'The instance this chat was using is not available any more.' : 'The model for this chat is not loaded.')
      }
      // Marks the model in use, so the API server's idle-unload leaves it alone while this reply runs.
      release = this.instances.acquire(inst.instanceId, by, req.prefitted ? 'writing' : 'chat')
      stats.modelName = inst.modelName

      const set = this.toolSet(req.tools)
      const tools = set.defs
      let messages = withSystemNote(req.messages, set.note)
      let trimmed = false
      if (inst.contextLength) {
        const api = this.contextApi(inst, tools, by)
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
        if (req.prefitted) {
          // The caller assembled the request to fit (Writing mode): never trim it, fail clearly instead.
          const max = req.sampling.maxTokens ?? reserve
          const used = await api.countConversation(messages)
          if (used === null) throw new Error("Could not count the request's tokens, so it was not sent.")
          if (used + max > inst.contextLength) {
            throw new Error(`The request is larger than the model's context (${(used + max).toLocaleString()} of ${inst.contextLength.toLocaleString()} tokens).`)
          }
        } else {
          const fitted = await fitToContext(messages, inst.contextLength, reserve, req.overflow, api)
          trimmed = fitted !== messages
          messages = fitted
        }
      }

      // Some templates print earlier tool arguments as Python lists; send them as JSON text instead
      // so the model keeps writing JSON the server can parse.
      if (messages.some((m) => m.tool_calls?.length) && (await this.stringifiesArgs(inst))) {
        messages = jsonifyToolCallArgs(messages)
      }

      const body = buildRequestBody(inst.modelName, messages, req.sampling, req.disableThinking, requestLora(inst.adapters ?? [], parseChatAdapters(req.adapters)))
      if (tools.length) body.tools = tools
      const res = await this.instances.fetch(inst, '/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: ctrl.signal
      }, by)
      if (!res.ok || !res.body) {
        const text = await res.text()
        if (inst.cloud) throw new CloudRequestError(errorFromBody(res.status, text), errorCodeFromBody(text))
        throw new Error(errorFromBody(res.status, text))
      }

      const parser = new SseParser()
      const toolCalls = new ToolCallAccumulator()
      const decoder = new TextDecoder()
      const reader = res.body.getReader()
      let sawDone = false
      let usage: ChunkUsage | undefined
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        for (const data of parser.push(decoder.decode(value, { stream: true }))) {
          if (data === '[DONE]') {
            sawDone = true
            continue
          }
          let json: unknown
          try {
            json = JSON.parse(data)
          } catch {
            continue
          }
          const c = interpretChunk(json)
          if (c.error) throw inst.cloud ? new CloudRequestError(c.error, c.errorCode) : new Error(c.error)
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
            if (c.content) fullContent += c.content
            if (c.reasoning) fullReasoning += c.reasoning
            deltas.push(c.content, c.reasoning)
          }
          if (c.toolCalls) {
            firstTokenAt ??= performance.now()
            toolCalls.push(c.toolCalls)
          }
          if (c.finishReason) finishReason = c.finishReason
          if (c.timings) stats = statsFromTimings(c.timings, stats)
          // Only OpenRouter's usage is read: a llama-server's own timings are what its stats come from.
          if (c.usage && inst.cloud) usage = c.usage
        }
      }
      // A stream that just stops, with no finish reason and no [DONE], is a tunnel that broke mid-answer. On this PC
      // that has always ended as a done event, and stays so.
      if (inst.host && !finishReason && !sawDone && !ctrl.signal.aborted) {
        throw new TransportError(`The connection to ${inst.host.label} broke while the model was answering.`)
      }
      // A cloud answer that just stops is broken off. Not a transport error (nothing can be retried for free) and not fatal.
      if (inst.cloud && !finishReason && !sawDone && !ctrl.signal.aborted) throw new CloudRequestError('The answer from OpenRouter broke off.', undefined)
      if (usage) {
        const genSeconds = firstTokenAt !== null ? (performance.now() - firstTokenAt) / 1000 : 0
        stats = statsFromUsage(usage, stats, genSeconds)
      }
      const calls = toolCalls.result()
      if (req.record && this.recorder) {
        try {
          this.recorder(req.record.conversationId, {
            requestId,
            at: Date.now(),
            // An OpenRouter model is recorded under its whole id, as the API server does.
            model: inst.cloud ? inst.modelId : inst.modelName,
            messages,
            tools,
            trimmed,
            response: { content: fullContent, reasoning: fullReasoning, toolCalls: calls, finishReason }
          })
        } catch (err) {
          // Recording is a side job; it must never fail the reply.
          console.error('Could not record the RE round:', err)
        }
      }
      if (calls.length && release) {
        // The turn goes on after the tools run, so the model stays in use until then.
        this.holdTurn(requestId, release)
        release = undefined
      }
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
      else {
        // Only for rentals (and a pinned instance that vanished): this PC's server dying is reported as it always was.
        const now = inst ? this.instances.get(inst.instanceId) : undefined
        const transport = pinnedGone || (!!inst?.host && (isTransportError(err) || !now || !!now.reconnecting))
        const message = (err as Error).message
        const error = transport && inst?.host && !(err instanceof TransportError) && !pinnedGone ? `The connection to ${inst.host.label} was lost (${message}). The model is still loaded there.` : message
        const fatal = inst?.cloud && err instanceof CloudRequestError && isFatalCloudError(err.code)
        this.send({ type: 'error', requestId, error, stats: final, ...(transport ? { transport: true } : {}), ...(fatal ? { fatal: true } : {}) })
      }
    } finally {
      release?.()
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
