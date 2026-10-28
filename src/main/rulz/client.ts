// Talks to RulZ LLM Suite's local API server (OpenAI-compatible endpoints).
import type { ConnectionInfo, LoadSettings, ModelInfo, ModelLoadInfo, WireMessage, WireToolCall } from '@shared/types'
import { errorFromBody, interpretChunk, SseParser, ToolCallAccumulator } from './sse'
import { jsonifyToolCallArgs, templateStringifiesArgs } from './toolArgs'

export interface ServerConfig {
  url: string
  apiKey: string
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
  }
  const raw = data as Raw[]
  const isRulz = raw.length === 0 ? false : raw.some((m) => m.owned_by === 'rulz' || (m.path !== undefined && m.state !== undefined))
  const models = raw
    .filter((m) => typeof m.id === 'string')
    .map((m) => ({
      id: m.id!,
      loaded: m.state === 'loaded',
      loadedContext: m.loaded_context_length,
      maxContext: m.max_context_length,
      arch: m.arch,
      quantization: m.quantization
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

/** The model to use when none is chosen: a loaded coder model, then any loaded one, then a coder model. */
export function pickDefaultModel(models: ModelInfo[]): string {
  const coder = (m: ModelInfo): boolean => /qwen3[-_.]?coder/i.test(m.id)
  return (
    models.find((m) => m.loaded && coder(m)) ??
    models.find((m) => m.loaded) ??
    models.find(coder) ??
    models.find((m) => /coder|code/i.test(m.id)) ??
    models[0]
  )?.id ?? ''
}

export interface ModelProps {
  chatTemplate?: string
  nCtx?: number
}

export interface TurnResult {
  content: string
  toolCalls: WireToolCall[]
  finishReason: string
  /** Exact token counts when the server reports them. */
  usage?: { prompt: number; completion: number }
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
    readonly status: number
  ) {
    super(message)
  }
}

export class RulzClient {
  private props = new Map<string, ModelProps>()

  constructor(private readonly config: () => ServerConfig) {}

  /** Forgets per-model details, e.g. after the server or the model changes. */
  reset(): void {
    this.props.clear()
  }

  async listModels(): Promise<ModelInfo[]> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/models`, { headers: headers(apiKey), signal: AbortSignal.timeout(5000) })
    if (!res.ok) throw new ServerError(errorFromBody(res.status, await res.text()), res.status)
    return parseModels(await res.json()).models
  }

  /** Chat template and context length. Asking for an unloaded model makes RulZ load it. */
  async modelProps(model: string, signal?: AbortSignal): Promise<ModelProps> {
    const cached = this.props.get(model)
    if (cached) return cached
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/props?model=${encodeURIComponent(model)}`, { headers: headers(apiKey), signal })
    if (!res.ok) throw new ServerError(errorFromBody(res.status, await res.text()), res.status)
    const j = (await res.json()) as { chat_template?: string; default_generation_settings?: { n_ctx?: number }; n_ctx?: number }
    const props = { chatTemplate: j.chat_template, nCtx: j.default_generation_settings?.n_ctx ?? j.n_ctx }
    this.props.set(model, props)
    return props
  }

  /** Applies the Qwen3-Coder argument fix when the model's template needs it. */
  prepareMessages(model: string, messages: WireMessage[]): WireMessage[] {
    return templateStringifiesArgs(this.props.get(model)?.chatTemplate) ? jsonifyToolCallArgs(messages) : messages
  }

  /** One streamed /v1/chat/completions turn. */
  async streamTurn(
    body: { model: string; messages: WireMessage[]; tools?: unknown[]; [k: string]: unknown },
    signal: AbortSignal,
    on: StreamHandlers = {}
  ): Promise<TurnResult> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: headers(apiKey, true),
      body: JSON.stringify({ ...body, messages: this.prepareMessages(body.model, body.messages), stream: true }),
      signal
    })
    if (!res.ok || !res.body) throw new ServerError(errorFromBody(res.status, await res.text()), res.status)
    on.onFirstByte?.()

    const parser = new SseParser()
    const calls = new ToolCallAccumulator()
    const decoder = new TextDecoder()
    let content = ''
    let finishReason = ''
    let usage: TurnResult['usage']
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
        const info = interpretChunk(json)
        if (info.error) throw new ServerError(info.error, 500)
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
      }
    }
    return { content, toolCalls: calls.result(), finishReason, usage }
  }

  /** The model's load settings in RulZ LLM Suite. */
  async loadSettings(model: string): Promise<ModelLoadInfo> {
    return this.loadConfigRequest(model, 'GET')
  }

  /** The memory estimate for changed load settings, without saving them. */
  async previewLoadSettings(model: string, config: Partial<LoadSettings>): Promise<ModelLoadInfo> {
    return this.loadConfigRequest(model, 'POST', { config, dry_run: true })
  }

  /** Saves new load settings in RulZ LLM Suite; a loaded model restarts with them. */
  async setLoadSettings(model: string, config: Partial<LoadSettings>): Promise<ModelLoadInfo> {
    const info = await this.loadConfigRequest(model, 'POST', { config, reload: true })
    this.props.delete(model)
    return info
  }

  private async loadConfigRequest(model: string, method: 'GET' | 'POST', body?: unknown): Promise<ModelLoadInfo> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/models/${encodeURIComponent(model)}/load-config`, {
      method,
      headers: headers(apiKey, !!body),
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(10_000)
    })
    const text = await res.text()
    if (res.status === 404 && /Unknown endpoint/i.test(text)) {
      throw new ServerError('This version of RulZ LLM Suite cannot change load settings from here. Update RulZ LLM Suite, or change them on its Models page.', 404)
    }
    if (!res.ok) throw new ServerError(errorFromBody(res.status, text), res.status)
    return parseLoadInfo(JSON.parse(text))
  }

  /** Non-streamed completion without tools (used for summaries). */
  async complete(model: string, messages: WireMessage[], maxTokens: number, signal?: AbortSignal): Promise<string> {
    const { url, apiKey } = this.config()
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: headers(apiKey, true),
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, stream: false, temperature: 0.3 }),
      signal
    })
    if (!res.ok) throw new ServerError(errorFromBody(res.status, await res.text()), res.status)
    const j = (await res.json()) as { choices?: { message?: { content?: string } }[] }
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
    gpus?: ModelLoadInfo['gpus']
    estimate?: ModelLoadInfo['estimate'] | null
  }
  if (!j?.config || typeof j.config !== 'object') throw new Error('RulZ LLM Suite sent load settings I could not read.')
  return {
    config: j.config,
    custom: !!j.custom,
    maxContext: j.max_context_length,
    state: j.state ?? 'not-loaded',
    loadedContext: j.loaded_context_length,
    hasDraft: j.has_draft,
    gpus: Array.isArray(j.gpus) ? j.gpus : [],
    estimate: j.estimate ?? undefined
  }
}

/** Plain-language explanation of a server error for the owner. */
export function friendlyServerError(e: unknown): string {
  const err = e as Error & { status?: number; cause?: { code?: string } }
  if (err.name === 'AbortError') return 'Stopped.'
  if (err instanceof TypeError || err.cause?.code === 'ECONNREFUSED' || err.cause?.code === 'ECONNRESET') {
    return 'I lost the connection to RulZ LLM Suite. Check that it is open and its server is started (Developer page), then try again.'
  }
  if (err instanceof ServerError) {
    if (err.status === 503) return `The model could not be loaded: ${err.message}`
    if (err.status === 401) return 'RulZ LLM Suite rejected the API key. Check it in Settings.'
    if (err.status === 404) return `RulZ LLM Suite doesn't know that model any more. Pick another one in Settings. (${err.message})`
    if (/context|too long|exceed/i.test(err.message)) return `The conversation got too long for the model: ${err.message}`
    return `The model server reported a problem: ${err.message}`
  }
  return `Something went wrong: ${err.message ?? String(e)}`
}
