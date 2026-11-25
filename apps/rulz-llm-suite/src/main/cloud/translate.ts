// llama-server style requests and replies <-> OpenRouter's (docs/cloud-design.md, section 4). The request sent upstream is
// built from scratch from a whitelist: nothing else of what a client sent passes.
import type { CloudErrorKind, CloudModelDef, CloudOptions } from '@shared/cloud'
import { maxPriceFor } from './prices'
import { callsKey, type CallKey, type ReasoningCache } from './reasoningCache'
import { estimateInput, textUnits } from './tokens'

export interface TranslateOk {
  ok: true
  body: Record<string, unknown>
  /** The client asked for a stream (else the stream is collected into one answer). */
  stream: boolean
  /** max_tokens (or max_completion_tokens) as sent. */
  maxTokens: number
  inputTokens: number
  images: number
}

export interface TranslateFail {
  ok: false
  status: number
  code: CloudErrorKind
  message: string
}

type Rec = Record<string, unknown>
const isRec = (v: unknown): v is Rec => !!v && typeof v === 'object' && !Array.isArray(v)
const fail = (message: string): TranslateFail => ({ ok: false, status: 400, code: 'cloud_request', message })
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

const ROLES = new Set(['system', 'user', 'assistant', 'tool'])

interface Converted {
  messages: Rec[]
  images: number
}

/** Arguments as the JSON text of an object; empty or invalid text becomes '{}'. */
export function cleanArguments(a: unknown): string {
  if (a && typeof a === 'object') {
    try {
      return JSON.stringify(a)
    } catch {
      return '{}'
    }
  }
  if (typeof a !== 'string' || !a.trim()) return '{}'
  try {
    const v = JSON.parse(a)
    return v && typeof v === 'object' && !Array.isArray(v) ? a : '{}'
  } catch {
    return '{}'
  }
}

function convertContent(c: unknown, counter: { images: number }): string | Rec[] | null {
  if (c === null || c === undefined) return null
  if (typeof c === 'string') return c
  if (!Array.isArray(c)) return null
  const parts: Rec[] = []
  for (const p of c) {
    if (!isRec(p)) continue
    if (p.type === 'text' && typeof p.text === 'string') parts.push({ type: 'text', text: p.text })
    else if (p.type === 'image_url' || p.type === 'image') {
      const iu = p.image_url
      const url = typeof iu === 'string' ? iu : isRec(iu) && typeof iu.url === 'string' ? iu.url : null
      if (!url) continue
      counter.images++
      const out: Rec = { url }
      if (isRec(iu) && typeof iu.detail === 'string') out.detail = iu.detail
      parts.push({ type: 'image_url', image_url: out })
    }
    // Any other part type (audio, files) is not passed.
  }
  return parts
}

function convertMessages(list: unknown[], def: CloudModelDef, cache: ReasoningCache): Converted | TranslateFail {
  const counter = { images: 0 }
  const out: Rec[] = []
  for (const raw of list) {
    if (!isRec(raw)) return fail('Every message must be an object.')
    let role = typeof raw.role === 'string' ? raw.role : ''
    if (role === 'developer') role = 'system'
    if (!ROLES.has(role)) return fail(`A message has the role "${String(raw.role)}", which this model cannot take.`)
    const content = convertContent(raw.content, counter)
    const m: Rec = { role }
    if (role === 'tool') {
      if (typeof raw.tool_call_id !== 'string' || !raw.tool_call_id) return fail('A tool message needs a tool_call_id.')
      m.tool_call_id = raw.tool_call_id
      m.content = content ?? ''
    } else if (role === 'assistant') {
      const calls: CallKey[] = []
      const wire: Rec[] = []
      if (Array.isArray(raw.tool_calls)) {
        raw.tool_calls.forEach((c, i) => {
          const f = isRec(c) && isRec(c.function) ? c.function : null
          if (!f || typeof f.name !== 'string' || !f.name) return
          const id = isRec(c) && typeof c.id === 'string' && c.id ? c.id : `call_${i}`
          const args = cleanArguments(f.arguments)
          calls.push({ id, name: f.name, arguments: args })
          wire.push({ id, type: 'function', function: { name: f.name, arguments: args } })
        })
      }
      if (wire.length) {
        m.content = content ?? null
        m.tool_calls = wire
      } else m.content = content ?? ''
      const clientDetails = Array.isArray(raw.reasoning_details) && raw.reasoning_details.length ? raw.reasoning_details : undefined
      // Details a client sends itself win over the ones remembered here.
      const details = clientDetails ?? (wire.length ? cache.get(def.slug, calls) : undefined)
      if (details) m.reasoning_details = details
      else if (typeof raw.reasoning_content === 'string' && raw.reasoning_content) m.reasoning_content = raw.reasoning_content
    } else {
      m.content = content ?? ''
    }
    if (typeof raw.name === 'string' && raw.name) m.name = raw.name
    out.push(m)
  }
  if (counter.images > 0 && !def.info.vision) return fail(`${def.info.name} cannot look at images. Remove the images, or pick a model that can.`)
  return { messages: out, images: counter.images }
}

function convertTools(tools: unknown): Rec[] | null {
  if (!Array.isArray(tools)) return null
  const out: Rec[] = []
  for (const t of tools) {
    const f = isRec(t) && isRec(t.function) ? t.function : null
    if (!f || typeof f.name !== 'string' || !f.name) continue
    const fn: Rec = { name: f.name }
    if (typeof f.description === 'string') fn.description = f.description
    if (isRec(f.parameters)) fn.parameters = f.parameters
    out.push({ type: 'function', function: fn })
  }
  return out
}

const KNOWN_EFFORTS = ['max', 'xhigh', 'high', 'medium', 'low', 'minimal']

/**
 * The `reasoning` object from the llama-server fields (chat_template_kwargs.enable_thinking, reasoning_effort,
 * thinking_budget_tokens) or a client's own `reasoning`. Nothing for "auto" or a model without reasoning; off is
 * {effort:'none'}, or nothing when the model's reasoning is mandatory; effort OR max_tokens (max_tokens only when the
 * model supports it, below the request's max_tokens and at least 1024, else left out).
 */
export function mapReasoning(b: Rec, def: CloudModelDef, maxTokens: number): Rec | undefined {
  const r = def.info.reasoning
  if (!r) return undefined
  const kwargs = isRec(b.chat_template_kwargs) ? b.chat_template_kwargs : {}
  const own = isRec(b.reasoning) ? b.reasoning : {}
  let enabled: boolean | undefined = typeof kwargs.enable_thinking === 'boolean' ? kwargs.enable_thinking : undefined
  if (enabled === undefined && typeof own.enabled === 'boolean') enabled = own.enabled
  if (enabled === undefined && own.effort === 'none') enabled = false
  if (enabled === false) return r.mandatory ? undefined : { effort: 'none' }
  const rawEffort = typeof b.reasoning_effort === 'string' ? b.reasoning_effort : typeof own.effort === 'string' ? own.effort : undefined
  const effort = rawEffort && rawEffort !== 'default' && rawEffort !== 'none' && (r.efforts.length ? r.efforts.includes(rawEffort) : KNOWN_EFFORTS.includes(rawEffort)) ? rawEffort : undefined
  const rawBudget = finite(b.thinking_budget_tokens) ? b.thinking_budget_tokens : finite(own.max_tokens) ? own.max_tokens : undefined
  if (rawBudget !== undefined && r.supportsMaxTokens) {
    const budget = Math.min(Math.floor(rawBudget), maxTokens - 1)
    if (budget >= 1024) return { max_tokens: budget }
  }
  if (effort) return { effort }
  return enabled === true && !r.mandatory ? { enabled: true } : undefined
}

function samplers(b: Rec, def: CloudModelDef): Rec {
  const s = new Set(def.info.supported)
  const out: Rec = {}
  if (s.has('temperature') && finite(b.temperature)) out.temperature = b.temperature
  if (s.has('top_p') && finite(b.top_p)) out.top_p = b.top_p
  if (s.has('top_k') && finite(b.top_k) && b.top_k > 0) out.top_k = Math.floor(b.top_k)
  if (s.has('min_p') && finite(b.min_p) && b.min_p > 0) out.min_p = b.min_p
  const rep = finite(b.repeat_penalty) ? b.repeat_penalty : finite(b.repetition_penalty) ? b.repetition_penalty : undefined
  if (s.has('repetition_penalty') && rep !== undefined && rep !== 1 && rep > 0) out.repetition_penalty = rep
  if (s.has('presence_penalty') && finite(b.presence_penalty) && b.presence_penalty !== 0) out.presence_penalty = b.presence_penalty
  if (s.has('frequency_penalty') && finite(b.frequency_penalty) && b.frequency_penalty !== 0) out.frequency_penalty = b.frequency_penalty
  if (s.has('seed') && finite(b.seed) && Number.isInteger(b.seed)) out.seed = b.seed
  if (s.has('stop')) {
    const list = typeof b.stop === 'string' ? [b.stop] : Array.isArray(b.stop) ? b.stop : []
    const stops = list.filter((x): x is string => typeof x === 'string' && x.length > 0).slice(0, 4)
    if (stops.length) out.stop = stops
  }
  return out
}

function responseFormat(b: Rec, def: CloudModelDef): { format?: Rec; schema: boolean } {
  const rf = b.response_format
  if (!isRec(rf)) return { schema: false }
  const s = new Set(def.info.supported)
  if (rf.type === 'json_object' && (s.has('response_format') || s.has('structured_outputs'))) return { format: { type: 'json_object' }, schema: false }
  if (rf.type === 'json_schema' && isRec(rf.json_schema) && (s.has('structured_outputs') || s.has('response_format'))) {
    const js = rf.json_schema
    const out: Rec = { name: typeof js.name === 'string' && js.name ? js.name : 'output' }
    if (isRec(js.schema)) out.schema = js.schema
    if (typeof js.strict === 'boolean') out.strict = js.strict
    return { format: { type: 'json_schema', json_schema: out }, schema: true }
  }
  return { schema: false }
}

/** Builds the OpenRouter request of a llama-server style one, or says why it cannot be sent. */
export function translateRequest(raw: string | undefined, def: CloudModelDef, options: CloudOptions, cache: ReasoningCache): TranslateOk | TranslateFail {
  let b: unknown
  try {
    b = JSON.parse(raw ?? '')
  } catch {
    return fail('The request is not valid JSON.')
  }
  if (!isRec(b)) return fail('The request must be a JSON object.')
  if (!Array.isArray(b.messages) || !b.messages.length) return fail('The request has no messages.')
  const info = def.info
  const conv = convertMessages(b.messages, def, cache)
  if ('ok' in conv) return conv
  const tools = convertTools(b.tools)
  if (tools?.length && !info.tools) return fail(`${info.name} cannot use tools. Switch the tools off, or pick a model that can.`)

  const wanted = [b.max_tokens, b.max_completion_tokens, b.n_predict].find((v) => finite(v) && v > 0) as number | undefined
  const maxTokens = Math.max(1, Math.floor(Math.min(wanted ?? def.replyLimit, def.replyLimit, info.maxReply ?? Infinity)))
  const out: Rec = { model: def.slug, messages: conv.messages, stream: true }
  if (tools?.length) {
    out.tools = tools
    if (typeof b.tool_choice === 'string' || isRec(b.tool_choice)) out.tool_choice = b.tool_choice
    if (typeof b.parallel_tool_calls === 'boolean') out.parallel_tool_calls = b.parallel_tool_calls
  }
  // max_tokens, or max_completion_tokens for a model that takes only that one.
  out[info.supported.includes('max_tokens') ? 'max_tokens' : 'max_completion_tokens'] = maxTokens
  Object.assign(out, samplers(b, def))
  const rf = responseFormat(b, def)
  if (rf.format) out.response_format = rf.format
  const reasoning = mapReasoning(b, def, maxTokens)
  if (reasoning) out.reasoning = reasoning
  const provider: Rec = {}
  if (!options.allowTraining) provider.data_collection = 'deny'
  // Never routed to an endpoint dearer than the dearest price the model lists; see maxPriceFor.
  provider.max_price = maxPriceFor(info.prices)
  if (rf.schema) provider.require_parameters = true
  out.provider = provider

  const est = estimateInput(conv.messages, tools ?? undefined)
  return { ok: true, body: out, stream: b.stream === true, maxTokens, inputTokens: est.tokens, images: conv.images }
}

// ---------- replies ----------

export interface ErrorInfo {
  message: string
  type: string
  code: CloudErrorKind
}

/** The kind of an error from its HTTP status (or the numeric code OpenRouter puts in an error body) and message. */
export function kindForStatus(status: number, message: string): CloudErrorKind {
  const m = message.toLowerCase()
  if (status === 401) return 'cloud_key'
  if (status === 402) return 'cloud_credits'
  if (status === 429) return 'cloud_rate'
  if (status === 403) return /moderat|flagged|content|policy|safety/.test(m) ? 'cloud_policy' : /limit|credit|budget|spend|key/.test(m) ? 'cloud_credits' : 'cloud_policy'
  if (status === 404 || status === 503) return /provider|endpoint|polic|price|privacy|data/.test(m) ? 'cloud_policy' : 'cloud_error'
  if (status === 400 || status === 422) return 'cloud_request'
  return 'cloud_error'
}

export const errorInfo = (message: string, code: CloudErrorKind): ErrorInfo => ({ message, type: code === 'cloud_request' ? 'invalid_request_error' : 'cloud_error', code })

export const errorFrame = (e: ErrorInfo): string => `data: ${JSON.stringify({ error: { message: e.message, type: e.type, code: e.code } })}\n\n`
export const DONE_FRAME = 'data: [DONE]\n\n'

interface CallAcc {
  id: string
  name: string
  arguments: string
}

/** Turns OpenRouter chunks into llama-server style ones, and keeps what the whole reply holds. */
export class ReplyTranslator {
  id: string | undefined
  model: string | undefined
  created: number | undefined
  finishReason: string | null = null
  usage: Rec | undefined
  /** Dollars OpenRouter reports (BYOK: with the upstream cost); null = no usage.cost. */
  costUsd: number | null = null
  error: ErrorInfo | null = null
  content = ''
  reasoning = ''
  details: Rec[] = []
  private calls = new Map<number, CallAcc>()
  /** Estimated tokens of the reply so far (text, reasoning and tool calls). */
  units = 0

  constructor(private readonly slug: string) {}

  callList(): CallKey[] {
    return [...this.calls.entries()]
      .sort(([a], [b]) => a - b)
      .filter(([, c]) => c.name)
      .map(([i, c]) => ({ id: c.id || `call_${i}`, name: c.name, arguments: c.arguments }))
  }

  /** Remembers the reply's details for the next tool round. */
  remember(cache: ReasoningCache): void {
    const calls = this.callList()
    if (calls.length && this.details.length) cache.put(this.slug, calls, this.details)
  }

  /** Same key the next request will compute for this reply's message (test aid). */
  key(): string {
    return callsKey(this.slug, this.callList())
  }

  private addDetails(list: unknown[]): void {
    for (const d of list) {
      if (!isRec(d)) continue
      const last = this.details[this.details.length - 1]
      if (last && last.type === d.type && last.index === d.index && (d.type === 'reasoning.text' || d.type === 'reasoning.summary')) {
        for (const [k, v] of Object.entries(d)) {
          if ((k === 'text' || k === 'summary') && typeof v === 'string') last[k] = (typeof last[k] === 'string' ? (last[k] as string) : '') + v
          else if (v !== undefined && v !== null) last[k] = v
        }
      } else this.details.push({ ...d })
    }
  }

  /** The frames (`data: ...\n\n`) one upstream chunk gives; an error chunk sets `error` and gives none. */
  feed(chunk: unknown): string[] {
    if (!isRec(chunk)) return []
    if (typeof chunk.id === 'string' && !this.id) this.id = chunk.id
    if (typeof chunk.model === 'string') this.model = chunk.model
    if (typeof chunk.created === 'number') this.created = chunk.created
    const choice = Array.isArray(chunk.choices) && isRec(chunk.choices[0]) ? chunk.choices[0] : undefined
    if (chunk.error) {
      const e = chunk.error
      const message = typeof e === 'string' ? e : isRec(e) && typeof e.message === 'string' ? e.message : 'The provider reported an error.'
      const status = isRec(e) && finite(e.code) ? e.code : 0
      this.error = errorInfo(message, kindForStatus(status, message))
      return []
    }
    if (choice?.finish_reason === 'error') {
      this.error = errorInfo('The provider stopped with an error.', 'cloud_error')
      return []
    }
    const delta: Rec = {}
    const d = choice && isRec(choice.delta) ? choice.delta : {}
    if (typeof d.role === 'string') delta.role = d.role
    if (typeof d.content === 'string' && d.content) {
      delta.content = d.content
      this.content += d.content
      this.units += textUnits(d.content)
    }
    let reasoning = typeof d.reasoning === 'string' ? d.reasoning : typeof d.reasoning_content === 'string' ? d.reasoning_content : ''
    const details = Array.isArray(d.reasoning_details) ? d.reasoning_details.filter(isRec) : []
    if (!reasoning && !(typeof d.reasoning === 'string')) {
      for (const x of details) {
        if (x.type === 'reasoning.text' && typeof x.text === 'string') reasoning += x.text
        else if (x.type === 'reasoning.summary' && typeof x.summary === 'string') reasoning += x.summary
      }
    }
    if (reasoning) {
      delta.reasoning_content = reasoning
      this.reasoning += reasoning
      this.units += textUnits(reasoning)
    }
    if (details.length) {
      delta.reasoning_details = details
      this.addDetails(details)
    }
    if (Array.isArray(d.tool_calls) && d.tool_calls.length) {
      delta.tool_calls = d.tool_calls
      d.tool_calls.forEach((t: unknown, i: number) => {
        if (!isRec(t)) return
        const idx = finite(t.index) ? t.index : i
        const cur = this.calls.get(idx) ?? { id: '', name: '', arguments: '' }
        if (typeof t.id === 'string' && t.id) cur.id = t.id
        const f = isRec(t.function) ? t.function : {}
        if (typeof f.name === 'string') cur.name += f.name
        if (typeof f.arguments === 'string') cur.arguments += f.arguments
        this.calls.set(idx, cur)
        this.units += textUnits((typeof f.name === 'string' ? f.name : '') + (typeof f.arguments === 'string' ? f.arguments : ''))
      })
    }
    const finish = typeof choice?.finish_reason === 'string' ? choice.finish_reason : null
    if (finish) this.finishReason = finish
    let usage: Rec | undefined
    if (isRec(chunk.usage)) {
      usage = { ...chunk.usage }
      const base = finite(usage.cost) && usage.cost >= 0 ? usage.cost : null
      if (base !== null) {
        const up = isRec(usage.cost_details) && finite(usage.cost_details.upstream_inference_cost) && usage.cost_details.upstream_inference_cost > 0 && usage.is_byok === true ? usage.cost_details.upstream_inference_cost : 0
        this.costUsd = base + up
        usage.cost = this.costUsd
      } else delete usage.cost
      this.usage = usage
    }
    if (!Object.keys(delta).length && !finish && !usage) return []
    const frame: Rec = {
      id: this.id ?? 'chatcmpl-cloud',
      object: 'chat.completion.chunk',
      created: this.created ?? 0,
      model: this.model ?? this.slug,
      choices: choice || !usage ? [{ index: 0, delta, finish_reason: finish }] : []
    }
    if (usage) frame.usage = usage
    return [`data: ${JSON.stringify(frame)}\n\n`]
  }

  /** The whole reply as one chat.completion. */
  collected(): Rec {
    const calls = this.callList()
    const message: Rec = { role: 'assistant', content: this.content || (calls.length ? null : '') }
    if (this.reasoning) message.reasoning_content = this.reasoning
    if (this.details.length) message.reasoning_details = this.details
    if (calls.length) message.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }))
    const out: Rec = {
      id: this.id ?? 'chatcmpl-cloud',
      object: 'chat.completion',
      created: this.created ?? 0,
      model: this.model ?? this.slug,
      choices: [{ index: 0, message, finish_reason: this.finishReason ?? 'stop' }]
    }
    if (this.usage) out.usage = this.usage
    return out
  }
}
