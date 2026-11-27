// Talks to RulZ LLM Suite's local API server (OpenAI-compatible endpoints).
import { cloudOf, isCloudCode, parseModelModes, pickDefaultModel as pickDefault, rentalOf } from '@shared/modes'
import type { AgentMode, ConnectionInfo, LoadSettings, ModelInfo, ModelLoadInfo, ModelModes, ReKind, ReOutcome, ReOverview, WireMessage, WireToolCall } from '@shared/types'
import { noDraftMsPerToken } from './draft'
import { errorCodeFromBody, errorFromBody, type GenTimings, interpretChunk, SseParser, ToolCallAccumulator } from './sse'
import { jsonifyToolCallArgs, templateStringifiesArgs, validToolCallArgs } from './toolArgs'

export interface ServerConfig {
  url: string
  apiKey: string
}

/** How long the server may stay silent before a turn is given up (milliseconds). Tests shorten these. */
export const timeouts = { startMs: 30 * 60_000, streamIdleMs: 10 * 60_000, completeMs: 15 * 60_000 }

/** RulZ LLM Suite records requests carrying this header as part of the named RE session. */
export const RE_SESSION_HEADER = 'x-rulz-re-session'

/** What RulZ LLM Suite shows for a recorded RE session. */
export interface ReSessionInfo {
  client?: string
  title?: string
  target?: string
  goal?: string
  kind?: ReKind
  outcome?: ReOutcome
  notes?: string
  practice?: { taskId: string; family: string; tier: string; answer?: string; expected?: string }
}

const headers = (apiKey: string, json = false): Record<string, string> => ({
  ...(json ? { 'Content-Type': 'application/json' } : {}),
  ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
})

export const normalizeUrl = (url: string): string => {
  let u = url.trim().replace(/\/+$/, '').replace(/\/v1$/, '')
  if (u && !/^https?:\/\//i.test(u)) u = `http://${u}`
  return u
}

/** Reads a /v1/models body; RulZ entries carry extra fields that LM Studio's don't. */
export function parseModels(json: unknown): { isRulz: boolean; models: ModelInfo[] } {
  const data = (json as { data?: unknown[] })?.data
  if (!Array.isArray(data)) return { isRulz: false, models: [] }
  type Raw = {
    id?: string
    owned_by?: string
    path?: string
    state?: string
    arch?: string
    quantization?: string
    max_context_length?: number
    loaded_context_length?: number
    rental?: unknown
    cloud?: unknown
  }
  const raw = data as Raw[]
  const isRulz = raw.length === 0 ? false : raw.some((m) => m.owned_by === 'rulz' || (m.path !== undefined && m.state !== undefined))
  const models = raw
    // An entry that says it is cloud but is not a valid OpenRouter one is left out: never offered, never taken for a local model.
    .filter((m) => !!m && typeof m === 'object' && typeof m.id === 'string' && (!m.cloud || !!cloudOf(m)))
    .map((m) => ({
      id: m.id!,
      loaded: m.state === 'loaded',
      loadedContext: m.loaded_context_length,
      maxContext: m.max_context_length,
      arch: m.arch,
      quantization: m.quantization,
      ...(rentalOf(m) ? { rental: rentalOf(m) } : {}),
      ...(cloudOf(m) ? { cloud: cloudOf(m) } : {})
    }))
  return { isRulz, models }
}

type Probe = { kind: 'rulz'; models: ModelInfo[] } | { kind: 'other' } | { kind: 'needs-key' } | { kind: 'none' }

export async function probe(url: string, apiKey: string, timeoutMs = 2500): Promise<Probe> {
  try {
    const res = await fetch(`${url}/v1/models`, { headers: headers(apiKey), signal: AbortSignal.timeout(timeoutMs) })
    if (res.status === 401 || res.status === 403) {
      // Only RulZ says so in /health; anything else asking for a key is "another server".
      const health = await fetch(`${url}/health`, { signal: AbortSignal.timeout(timeoutMs) })
        .then((r) => r.json() as Promise<{ app?: string }>)
        .catch(() => null)
      return health?.app === 'RulZ LLM Suite' ? { kind: 'needs-key' } : { kind: 'other' }
    }
    if (!res.ok) return { kind: 'other' }
    const parsed = parseModels(await res.json())
    if (parsed.isRulz) return { kind: 'rulz', models: parsed.models }
    // An empty RulZ library has no entries to tell by; ask /health.
    const health = await fetch(`${url}/health`, { signal: AbortSignal.timeout(timeoutMs) })
      .then((r) => r.json() as Promise<{ app?: string }>)
      .catch(() => null)
    return health?.app === 'RulZ LLM Suite' ? { kind: 'rulz', models: parsed.models } : { kind: 'other' }
  } catch {
    return { kind: 'none' }
  }
}

/** Tries each address in order; the first RulZ server wins. */
export async function findServer(candidates: string[], apiKey: string): Promise<ConnectionInfo> {
  const urls = [...new Set(candidates.map(normalizeUrl).filter(Boolean))]
  const results = await Promise.all(urls.map((u) => probe(u, apiKey)))
  const hit = results.findIndex((r) => r.kind === 'rulz')
  if (hit >= 0) return { status: 'connected', url: urls[hit], models: (results[hit] as { models: ModelInfo[] }).models }
  const key = results.findIndex((r) => r.kind === 'needs-key')
  if (key >= 0) return { status: 'needs-key', url: urls[key], models: [] }
  const other = results.findIndex((r) => r.kind === 'other')
  if (other >= 0) return { status: 'other-server', url: '', models: [], otherUrl: urls[other] }
  return { status: 'not-found', url: '', models: [] }
}

/** The model to use when none is chosen: a loaded coder model, then any loaded one, then a coder model. Never a rental's. */
export const pickDefaultModel = pickDefault

export interface ModelProps {
  chatTemplate?: string
  nCtx?: number
}

/** A context length the server actually reported. */
const known = (n: number | undefined): n is number => typeof n === 'number' && n > 0

export interface TurnResult {
  content: string
  toolCalls: WireToolCall[]
  finishReason: string
  /** Exact token counts when the server reports them. */
  usage?: { prompt: number; completion: number }
  /** What an OpenRouter model's answer cost, in dollars (the last chunk's `usage.cost`). */
  cost?: number
  /** Generation time and, when the server drafts, draft counts. */
  gen?: GenTimings
}

export interface StreamHandlers {
  onFirstByte?: () => void
  onContent?: (text: string) => void
  onReasoning?: (text: string) => void
  onToolDelta?: () => void
  onProgress?: (processed: number, total: number) => void
}

export class ServerError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string
  ) {
    super(message)
  }
}

export class RulzClient {
  private props = new Map<string, ModelProps>()
  /** Per model: ms per token seen in a turn without drafting, for the draft stats' time-saved estimate. */
  private noDraftSpeed = new Map<string, number>()

  constructor(private readonly config: () => ServerConfig) {}

  /** Forgets per-model details, e.g. after the server or the model changes. */
  reset(): void {
    this.props.clear()
    this.noDraftSpeed.clear()
  }

  /** The model's last measured ms per token without drafting, if a turn without drafting has been seen. */
  noDraftBaseline(model: string): number | undefined {
    return this.noDraftSpeed.get(model)
  }

  async listModels(): Promise<ModelInfo[]> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/models`, { headers: headers(apiKey), signal: AbortSignal.timeout(5000) })
    if (!res.ok) throw new ServerError(errorFromBody(res.status, await res.text()), res.status)
    const models = parseModels(await res.json()).models
    // A model reloaded with another context (in RulZ LLM Suite, or by another app) makes the cached
    // props stale; the list reports the context it is loaded with now. Only two known, different
    // numbers mean that: forgetting the props on every poll would switch the Qwen3-Coder argument fix
    // (decided by the chat template in them) off until the next task, changing how the history is
    // written and throwing away the server's prompt cache.
    for (const m of models) {
      const cached = this.props.get(m.id)
      if (cached && known(cached.nCtx) && known(m.loadedContext) && cached.nCtx !== m.loadedContext) this.props.delete(m.id)
    }
    return models
  }

  /** Forgets one model's cached props (chat template, context length). */
  forgetProps(model: string): void {
    this.props.delete(model)
  }

  /** Chat template and context length. Asking for an unloaded model makes RulZ load it. */
  async modelProps(model: string, signal?: AbortSignal, extra: Record<string, string> = {}): Promise<ModelProps> {
    const cached = this.props.get(model)
    if (cached) return cached
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/props?model=${encodeURIComponent(model)}`, { headers: { ...headers(apiKey), ...extra }, signal })
    if (!res.ok) {
      const text = await res.text()
      throw new ServerError(errorFromBody(res.status, text), res.status, errorCodeFromBody(text))
    }
    const j = (await res.json()) as { chat_template?: string; default_generation_settings?: { n_ctx?: number }; n_ctx?: number }
    const nCtx = j.default_generation_settings?.n_ctx ?? j.n_ctx
    const props = { chatTemplate: j.chat_template, nCtx: known(nCtx) ? nCtx : undefined }
    this.props.set(model, props)
    return props
  }

  /** Repairs unreadable tool-call arguments, and applies the Qwen3-Coder argument fix when the model's template needs it. */
  prepareMessages(model: string, messages: WireMessage[]): WireMessage[] {
    const valid = validToolCallArgs(messages)
    return templateStringifiesArgs(this.props.get(model)?.chatTemplate) ? jsonifyToolCallArgs(valid) : valid
  }

  /** One streamed /v1/chat/completions turn. Gives up when the server sends nothing for timeouts.streamIdleMs. */
  async streamTurn(
    body: { model: string; messages: WireMessage[]; tools?: unknown[]; [k: string]: unknown },
    signal: AbortSignal,
    on: StreamHandlers = {},
    /** Extra request headers, e.g. the RE session to record this turn in. */
    extra: Record<string, string> = {}
  ): Promise<TurnResult> {
    const { url, apiKey } = this.config()
    // Our own controller, so an idle timeout can cut the connection without the caller's signal
    // (which means "Stop") being set.
    const link = new AbortController()
    const onStop = (): void => link.abort()
    if (signal.aborted) link.abort()
    else signal.addEventListener('abort', onStop, { once: true })
    try {
      return await this.readTurn(url, apiKey, body, link, on, extra)
    } finally {
      signal.removeEventListener('abort', onStop)
    }
  }

  private async readTurn(
    url: string,
    apiKey: string,
    body: { model: string; messages: WireMessage[]; tools?: unknown[]; [k: string]: unknown },
    link: AbortController,
    on: StreamHandlers,
    extra: Record<string, string>
  ): Promise<TurnResult> {
    // Waiting for the answer to start has its own, longer limit: the Suite may first load the model.
    let noAnswer = false
    const startTimer = setTimeout(() => {
      noAnswer = true
      link.abort()
    }, timeouts.startMs)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { ...headers(apiKey, true), ...extra },
      // return_progress: llama-server then streams prompt_progress chunks while it reads the prompt
      // (RulZ LLM Suite passes the body through unchanged), which onProgress turns into a percentage.
      body: JSON.stringify({ ...body, messages: this.prepareMessages(body.model, body.messages), stream: true, return_progress: true }),
      signal: link.signal
    })
      .catch((e: unknown) => {
        if (noAnswer) throw new Error('The server did not start answering for a long time. Check that RulZ LLM Suite (or the rented GPU) is still running, then try again.')
        throw e
      })
      .finally(() => clearTimeout(startTimer))
    if (!res.ok || !res.body) {
      const text = await res.text()
      throw new ServerError(errorFromBody(res.status, text), res.status, errorCodeFromBody(text))
    }
    on.onFirstByte?.()

    const parser = new SseParser()
    const calls = new ToolCallAccumulator()
    const decoder = new TextDecoder()
    let content = ''
    let finishReason = ''
    let usage: TurnResult['usage']
    let cost: number | undefined
    let gen: GenTimings | undefined
    const reader = res.body.getReader()
    try {
      for (;;) {
        let idle: NodeJS.Timeout | undefined
        const stalled = new Promise<never>((_, reject) => {
          idle = setTimeout(() => {
            reject(new Error('The server stopped answering: nothing came in for a long time. Check that RulZ LLM Suite (or the rented GPU) is still running, then try again.'))
            link.abort() // after the reject, so the race reports the timeout and not the abort
          }, timeouts.streamIdleMs)
        })
        let chunk: Awaited<ReturnType<typeof reader.read>>
        try {
          chunk = await Promise.race([reader.read(), stalled])
        } finally {
          clearTimeout(idle)
          stalled.catch(() => undefined)
        }
        const { done, value } = chunk
        if (done) break
      for (const data of parser.push(decoder.decode(value, { stream: true }))) {
        if (data === '[DONE]') continue
        let json: unknown
        try {
          json = JSON.parse(data)
        } catch {
          continue
        }
        const info = interpretChunk(json)
        if (info.error) throw new ServerError(info.error, 500, info.errorCode)
        if (info.progress) on.onProgress?.(info.progress.processed, info.progress.total)
        if (info.reasoning) on.onReasoning?.(info.reasoning)
        if (info.content) {
          content += info.content
          on.onContent?.(info.content)
        }
        if (info.toolCalls) {
          calls.push(info.toolCalls)
          on.onToolDelta?.()
        }
        if (info.finishReason) finishReason = info.finishReason
        if (info.usage) usage = info.usage
        if (info.cost !== undefined) cost = info.cost
        if (info.gen) gen = info.gen
      }
    }
    } finally {
      reader.cancel().catch(() => undefined)
    }
    const baseline = gen && noDraftMsPerToken(gen)
    if (baseline) this.noDraftSpeed.set(body.model, baseline)
    return { content, toolCalls: calls.result(), finishReason, usage, gen, ...(cost !== undefined ? { cost } : {}) }
  }

  /** The model's load settings in RulZ LLM Suite; with `mode`, the ones that mode loads with (0.35). */
  async loadSettings(model: string, mode?: AgentMode): Promise<ModelLoadInfo> {
    return this.loadConfigRequest(model, 'GET', undefined, mode)
  }

  /** The memory estimate for changed load settings, without saving them. */
  async previewLoadSettings(model: string, config: Partial<LoadSettings>, mode?: AgentMode): Promise<ModelLoadInfo> {
    return this.loadConfigRequest(model, 'POST', { config, dry_run: true }, mode)
  }

  /**
   * Saves new load settings in RulZ LLM Suite; a loaded model restarts with them. With `mode` they are
   * that mode's own settings, and `null` removes them (the mode then uses the settings of all modes).
   * A reload LLM Suite refuses still saves the settings: the answer says why in `reloadBlocked`.
   */
  async setLoadSettings(model: string, config: Partial<LoadSettings> | null, mode?: AgentMode): Promise<ModelLoadInfo> {
    if (config === null && !mode) throw new Error('Pick a mode to remove its own load settings.')
    const info = await this.loadConfigRequest(model, 'POST', { config, reload: true }, mode)
    this.props.delete(model)
    return info
  }

  /** The settings of a model for each mode (GET /v1/models/{id}/modes); null from a RulZ LLM Suite older than 0.35. */
  async modelModes(model: string, signal?: AbortSignal, extra: Record<string, string> = {}): Promise<ModelModes | null> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/models/${encodeURIComponent(model)}/modes`, {
      headers: { ...headers(apiKey), ...extra },
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10_000)]) : AbortSignal.timeout(10_000)
    })
    const text = await res.text()
    if (res.status === 404) return null
    if (!res.ok) throw new ServerError(errorFromBody(res.status, text), res.status, errorCodeFromBody(text))
    return parseModelModes(JSON.parse(text))
  }

  /** Loads the model again with a mode's load settings (POST /v1/models/{id}/reload). Refusals come back as ServerError with LLM Suite's sentence. */
  async reloadMode(model: string, mode: AgentMode): Promise<ModelModes> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/models/${encodeURIComponent(model)}/reload`, {
      method: 'POST',
      headers: headers(apiKey, true),
      body: JSON.stringify({ mode }),
      signal: AbortSignal.timeout(30_000)
    })
    const text = await res.text()
    if (res.status === 404) throw new ServerError('This version of RulZ LLM Suite cannot reload a model from here. Update RulZ LLM Suite, or reload it on its Models page.', 404)
    if (!res.ok) throw new ServerError(errorFromBody(res.status, text), res.status)
    this.props.delete(model)
    return parseModelModes(JSON.parse(text))
  }

  private async loadConfigRequest(model: string, method: 'GET' | 'POST', body?: unknown, mode?: AgentMode): Promise<ModelLoadInfo> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/models/${encodeURIComponent(model)}${mode ? `/modes/${encodeURIComponent(mode)}` : ''}/load-config`, {
      method,
      headers: headers(apiKey, !!body),
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(10_000)
    })
    const text = await res.text()
    if (res.status === 404 && mode && /Unknown endpoint/i.test(text)) {
      throw new ServerError('This version of RulZ LLM Suite has no load settings per mode. Update RulZ LLM Suite, or change them on its Models page.', 404)
    }
    if (res.status === 404 && /Unknown endpoint/i.test(text)) {
      throw new ServerError('This version of RulZ LLM Suite cannot change load settings from here. Update RulZ LLM Suite, or change them on its Models page.', 404)
    }
    if (!res.ok) throw new ServerError(errorFromBody(res.status, text), res.status)
    return parseLoadInfo(JSON.parse(text))
  }

  /**
   * Describes an RE session RulZ LLM Suite records. Resolves whether the Suite records it: false for
   * a Suite that is too old or has recording from other apps turned off. Never throws.
   */
  async describeReSession(id: string, info: ReSessionInfo): Promise<boolean> {
    const { url, apiKey } = this.config()
    try {
      const res = await fetch(`${url}/v1/re/sessions/${encodeURIComponent(id)}`, {
        method: 'POST',
        headers: headers(apiKey, true),
        body: JSON.stringify(info),
        signal: AbortSignal.timeout(5000)
      })
      if (!res.ok) return false
      return ((await res.json()) as { recording?: boolean }).recording === true
    } catch {
      return false
    }
  }

  private async reJson<T>(path: string, body?: unknown, timeoutMs = 15000): Promise<T> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: headers(apiKey, body !== undefined),
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
      signal: AbortSignal.timeout(timeoutMs)
    })
    if (res.status === 404) throw new ServerError('This needs RulZ LLM Suite 0.15.0 or later. Update RulZ LLM Suite and try again.', 404)
    if (!res.ok) throw new ServerError(errorFromBody(res.status, await res.text()), res.status)
    return (await res.json()) as T
  }

  /** RE sessions, practice tasks and prompt templates from RulZ LLM Suite. With `model` the practice counts are for that model (0.35). */
  reOverview(model?: string): Promise<ReOverview> {
    return this.reJson<ReOverview>(`/v1/re/overview${model ? `?model=${encodeURIComponent(model)}` : ''}`)
  }

  /** Changes RulZ LLM Suite's practice kit folder (one folder for both apps) and returns the kit's tasks. */
  async setReKit(kitDir: string): Promise<ReOverview['practice']> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/re/kit`, {
      method: 'POST',
      headers: headers(apiKey, true),
      body: JSON.stringify({ kitDir }),
      signal: AbortSignal.timeout(30000)
    })
    const text = await res.text()
    if (res.status === 404) throw new ServerError('Update RulZ LLM Suite to 0.27.0 or later to change the kit folder from here.', 404)
    if (!res.ok) throw new ServerError(errorFromBody(res.status, text), res.status)
    return JSON.parse(text) as ReOverview['practice']
  }

  /** Has RulZ LLM Suite copy a practice task's program into a fresh work folder. */
  preparePractice(taskId: string): Promise<{ target: string; prompt: string; family: string; tier: string }> {
    return this.reJson(`/v1/re/practice/${encodeURIComponent(taskId)}/prepare`, {})
  }

  /** Checks a practice task's final reply (its ANSWER line) against the task's answer. */
  gradePractice(taskId: string, target: string, text: string): Promise<{ answer: string | null; passed: boolean; expected: string }> {
    return this.reJson(`/v1/re/practice/${encodeURIComponent(taskId)}/grade`, { target, text }, 60000)
  }

  /** Non-streamed completion without tools (used for summaries). */
  async complete(
    model: string,
    messages: WireMessage[],
    maxTokens: number,
    signal?: AbortSignal,
    extra: Record<string, string> = {},
    /** Told what the answer cost, when the server says (OpenRouter models). */
    onCost?: (dollars: number) => void
  ): Promise<string> {
    const { url, apiKey } = this.config()
    // A total limit, so a connection that goes quiet cannot hang a summary forever. The caller's own
    // signal still aborts first and still reads as Stop (it is the caller's signal that is checked).
    const limit = AbortSignal.timeout(timeouts.completeMs)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { ...headers(apiKey, true), ...extra },
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, stream: false, temperature: 0.3 }),
      signal: signal ? AbortSignal.any([signal, limit]) : limit
    }).catch((e: unknown) => {
      if (limit.aborted && !signal?.aborted) throw new Error('The server stopped answering while writing a summary. Check that RulZ LLM Suite (or the rented GPU) is still running, then try again.')
      throw e
    })
    if (!res.ok) {
      const text = await res.text()
      throw new ServerError(errorFromBody(res.status, text), res.status, errorCodeFromBody(text))
    }
    const j = (await res.json()) as { choices?: { message?: { content?: string } }[]; usage?: { cost?: unknown } }
    if (typeof j.usage?.cost === 'number' && Number.isFinite(j.usage.cost) && j.usage.cost > 0) onCost?.(j.usage.cost)
    return j.choices?.[0]?.message?.content ?? ''
  }
}

/** Reads a GET/POST /v1/models/{id}/load-config body. */
export function parseLoadInfo(json: unknown): ModelLoadInfo {
  const j = json as {
    config?: LoadSettings
    custom?: boolean
    max_context_length?: number
    state?: ModelLoadInfo['state']
    loaded_context_length?: number
    has_draft?: boolean
    default_draft?: { name?: string; built_in?: boolean } | null
    draft_choices?: { id?: string; name?: string; folder?: string; kind?: string; built_in?: boolean }[]
    gpus?: ModelLoadInfo['gpus']
    moe_layers?: number | null
    estimate?: ModelLoadInfo['estimate'] | null
    mode?: unknown
    mode_own?: unknown
    reload_blocked?: unknown
  }
  if (!j?.config || typeof j.config !== 'object') throw new Error('RulZ LLM Suite sent load settings I could not read.')
  return {
    config: j.config,
    custom: !!j.custom,
    maxContext: j.max_context_length,
    state: j.state ?? 'not-loaded',
    loadedContext: j.loaded_context_length,
    hasDraft: j.has_draft,
    defaultDraft: j.default_draft === undefined ? undefined : j.default_draft && { name: j.default_draft.name ?? 'draft', builtIn: !!j.default_draft.built_in },
    draftChoices: Array.isArray(j.draft_choices)
      ? j.draft_choices
          .filter((c) => typeof c?.id === 'string')
          .map((c) => ({ id: c.id!, name: c.name ?? c.id!, folder: c.folder ?? '', kind: c.kind ?? 'mtp', builtIn: !!c.built_in }))
      : undefined,
    moeLayers: typeof j.moe_layers === 'number' && j.moe_layers > 0 ? j.moe_layers : j.moe_layers === undefined ? undefined : null,
    gpus: Array.isArray(j.gpus) ? j.gpus : [],
    estimate: j.estimate ?? undefined,
    ...(j.mode === 'coding' || j.mode === 're' ? { mode: j.mode, modeOwn: j.mode_own === true } : {}),
    ...(typeof j.reload_blocked === 'string' && j.reload_blocked ? { reloadBlocked: j.reload_blocked.slice(0, 500) } : {})
  }
}

/** Socket-level error codes (Node's own and undici's) that mean the server couldn't be reached or the connection broke. */
const NETWORK_CODES = new Set(['ECONNREFUSED', 'ECONNRESET', 'ECONNABORTED', 'ENOTFOUND', 'EAI_AGAIN', 'ETIMEDOUT', 'EPIPE', 'EHOSTUNREACH', 'ENETUNREACH', 'ENETDOWN'])
/** undici codes for mistakes in the request itself rather than in the connection. */
const NOT_NETWORK_UNDICI = new Set(['UND_ERR_INVALID_ARG', 'UND_ERR_INVALID_RETURN_VALUE', 'UND_ERR_NOT_SUPPORTED'])

/**
 * True for a failure to reach the server or a connection that broke mid-reply: fetch's
 * `TypeError: fetch failed` / `terminated`, or an error (or its cause) with a network error code.
 * Other TypeErrors are mistakes in the code and keep their own message.
 */
export function isNetworkError(e: unknown): boolean {
  let cur = e as { code?: unknown; cause?: unknown } | null | undefined
  for (let depth = 0; depth < 5 && cur && typeof cur === 'object'; depth++) {
    const code = cur.code
    if (typeof code === 'string' && (NETWORK_CODES.has(code) || (code.startsWith('UND_ERR_') && !NOT_NETWORK_UNDICI.has(code)))) return true
    cur = cur.cause as typeof cur
  }
  return e instanceof TypeError && (e.message === 'fetch failed' || e.message === 'terminated')
}

/** Plain-language explanation of a server error for the owner. */
export function friendlyServerError(e: unknown): string {
  const err = (e ?? {}) as Error & { status?: number }
  if (err.name === 'AbortError') return 'Stopped.'
  if (isNetworkError(e)) {
    return 'I lost the connection to RulZ LLM Suite. Check that it is open and its server is started (Developer page), then try again.'
  }
  if (err instanceof ServerError) {
    // OpenRouter errors: the server's own sentence says what to do.
    if (isCloudCode(err.code)) return err.message
    // A rental refused (409), or lost its connection: LLM Suite's own sentence says what to do.
    if (err.status === 409 || (err.status >= 500 && /connection to the rental/i.test(err.message))) return err.message
    if (err.status === 503) return `The model could not be loaded: ${err.message}`
    if (err.status === 401) return 'RulZ LLM Suite rejected the API key. Check it in Settings.'
    if (err.status === 404) return `RulZ LLM Suite doesn't know that model any more. Pick another one in Settings. (${err.message})`
    if (/context|too long|exceed/i.test(err.message)) return `The conversation got too long for the model: ${err.message}`
    return `The model server reported a problem: ${err.message}`
  }
  return `Something went wrong: ${err.message ?? String(e)}`
}
