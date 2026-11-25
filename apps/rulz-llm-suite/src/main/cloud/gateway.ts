// The one seam: answers as a llama-server would and talks to OpenRouter (docs/cloud-design.md, sections 2, 4, 5, 6).
// Everything OpenRouter-specific (auth, translation, cost, limits, retries, who may use it) is behind fetch().
import type { CloudErrorKind, CloudModelDef, CloudOptions } from '@shared/cloud'
import { SseParser, errorFromBody } from '../chat/sse'
import { OPENROUTER_BASE } from './catalog'
import type { CloudCaller, CloudFetchInit } from './contract'
import type { CloudLedger } from './ledger'
import { toMicro, worstCaseMicro } from './prices'
import type { ReasoningCache } from './reasoningCache'
import { estimateText } from './tokens'
import { DONE_FRAME, ReplyTranslator, errorFrame, errorInfo, kindForStatus, translateRequest, type ErrorInfo, type TranslateOk } from './translate'

export const IDLE_MS = 120_000
export const MAX_TRIES = 3
export const RETRY_AFTER_CAP_MS = 30_000
const NO_KEY = 'No OpenRouter key is saved. Add one on the OpenRouter page.'
const DEFAULT_RETRY_MS = [2000, 8000]
export const GENERATION_DELAYS_MS = [1000, 4000, 15000]
/** A reply streamed past this many times max_tokens (estimated) is cut off. */
export const REPLY_GUARD = 1.5

/** Errors that prove the request never left this PC: nothing can have been billed. */
const NOT_SENT = new Set(['ENOTFOUND', 'ECONNREFUSED', 'EAI_AGAIN', 'UND_ERR_CONNECT_TIMEOUT'])

export interface GatewayDeps {
  fetch: typeof fetch
  now: () => number
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>
  /** The saved key, or null. */
  key: () => string | null
  /** The model of a cloud model id (`openrouter/<slug>`); undefined when it is not in the list. */
  model: (modelId: string) => CloudModelDef | undefined
  /** Read at the start of every request, so a change applies to the next one. */
  options: () => CloudOptions
  ledger: CloudLedger
  reasoning: ReasoningCache
  scrub: (text: string) => string
  /** Test hook: no bytes for this long ends a request (IDLE_MS). */
  idleMs?: number
  /** Scrubbed lines about retries and settlements. */
  log?: (line: string) => void
}

export const abortError = (): Error => new DOMException('The request was cancelled.', 'AbortError')

const STATUS: Record<CloudErrorKind, number> = {
  cloud_limit: 402,
  cloud_key: 401,
  cloud_credits: 402,
  cloud_rate: 429,
  cloud_not_allowed: 403,
  cloud_policy: 403,
  cloud_request: 400,
  cloud_error: 502
}

export function errorResponse(code: CloudErrorKind, message: string, status = STATUS[code], headers: Record<string, string> = {}): Response {
  const e = errorInfo(message, code)
  return new Response(JSON.stringify({ error: { message: e.message, type: e.type, code: e.code } }), { status, headers: { 'Content-Type': 'application/json', ...headers } })
}

const json = (v: unknown): Response => new Response(JSON.stringify(v), { status: 200, headers: { 'Content-Type': 'application/json' } })

function errCode(err: unknown): string | undefined {
  const e = err as { code?: string; cause?: { code?: string; errors?: { code?: string }[]; cause?: { code?: string } } } | null
  return e?.code ?? e?.cause?.code ?? e?.cause?.errors?.[0]?.code ?? e?.cause?.cause?.code
}

const isRedirectError = (err: unknown): boolean => err instanceof TypeError && /redirect/i.test(`${err.message} ${(err.cause as Error | undefined)?.message ?? ''}`)

/** Seconds (or an HTTP date) of a Retry-After header, capped; null when absent or unreadable. */
export function retryAfterMs(header: string | null, now: number): number | null {
  if (!header) return null
  const s = Number(header)
  if (Number.isFinite(s) && s >= 0) return Math.min(s * 1000, RETRY_AFTER_CAP_MS)
  const at = Date.parse(header)
  if (Number.isFinite(at)) return Math.min(Math.max(0, at - now), RETRY_AFTER_CAP_MS)
  return null
}

/** Re-armable idle timer. */
class Watchdog {
  timedOut = false
  private timer: ReturnType<typeof setTimeout> | undefined
  constructor(
    private readonly ms: number,
    private readonly onTimeout: () => void
  ) {}
  touch(): void {
    this.stop()
    this.timer = setTimeout(() => {
      this.timedOut = true
      this.onTimeout()
    }, this.ms)
    this.timer.unref?.()
  }
  stop(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = undefined
  }
}

type Outcome = 'none' | 'unknown' | { micro: number }

interface Run {
  resId: string
  ctl: AbortController
  caller: AbortSignal | undefined
  done: boolean
  /** The consumer of the Response body cancelled it. */
  cancelled: boolean
  /** What ended the reply early (scrubbed). */
  error?: ErrorInfo
  cleanup: () => void
}

interface Sink {
  write(frame: string): void
  close(): void
  fail(err: unknown): void
}

type Route = { kind: 'props' | 'tokenize' | 'template' | 'chat' | 'count' } | { kind: 'unknown' }

function routeOf(path: string, method: string): Route {
  let p: string
  try {
    p = new URL(path, 'http://local').pathname
  } catch {
    return { kind: 'unknown' }
  }
  p = p.replace(/^\/v1(?=\/)/, '').replace(/\/+$/, '')
  if (method === 'GET' && p === '/props') return { kind: 'props' }
  if (method === 'POST' && p === '/tokenize') return { kind: 'tokenize' }
  if (method === 'POST' && p === '/apply-template') return { kind: 'template' }
  if (method === 'POST' && p === '/chat/completions') return { kind: 'chat' }
  if (method === 'POST' && p === '/chat/completions/input_tokens') return { kind: 'count' }
  return { kind: 'unknown' }
}

export class CloudGatewayCore {
  private disposed = false
  private readonly disposeCtl = new AbortController()
  private readonly pending = new Set<Promise<unknown>>()
  private readonly runs = new Set<Run>()

  constructor(private readonly deps: GatewayDeps) {}

  private log(line: string): void {
    this.deps.log?.(this.deps.scrub(line))
  }

  private track<T>(p: Promise<T>): Promise<T> {
    this.pending.add(p)
    const drop = (): void => void this.pending.delete(p)
    p.then(drop, drop)
    return p
  }

  /** Resolves when every background settlement started so far is finished. */
  async settled(): Promise<void> {
    while (this.pending.size) await Promise.allSettled([...this.pending])
  }

  /** Aborts what runs. Reservations stay on disk and are settled at the next start. */
  dispose(): void {
    this.disposed = true
    this.disposeCtl.abort()
    for (const r of this.runs) r.ctl.abort()
  }

  // ---------- settling ----------

  /** Settles a run exactly once, whatever path ended it. After dispose nothing is written: the reservation stays on disk. */
  private async finish(run: Run, outcome: Outcome): Promise<void> {
    if (run.done) return
    run.done = true
    run.cleanup()
    this.runs.delete(run)
    if (this.disposed) return
    try {
      if (outcome === 'none') await this.deps.ledger.settle(run.resId, 0)
      else if (outcome === 'unknown') await this.settleUnknown(run.resId)
      else await this.deps.ledger.settle(run.resId, outcome.micro)
    } catch (err) {
      this.log(`Settling a request failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  /** No usage known: without a generation id the whole reservation is charged; with one OpenRouter is asked, in the background. */
  private async settleUnknown(resId: string): Promise<void> {
    const r = this.deps.ledger.get(resId)
    if (!r) return
    if (!r.generationId) {
      await this.deps.ledger.settle(resId, r.amount)
      return
    }
    void this.track(this.lookupAndSettle(resId, r.generationId, r.amount))
  }

  private async lookupAndSettle(resId: string, generationId: string, amount: number): Promise<void> {
    for (const delay of GENERATION_DELAYS_MS) {
      try {
        await this.deps.sleep(delay, this.disposeCtl.signal)
      } catch {
        return
      }
      if (this.disposed) return
      const r = await this.lookupCost(generationId)
      if (this.disposed) return
      if (r.state === 'cost') {
        await this.deps.ledger.settle(resId, r.micro)
        return
      }
      if (r.state === 'stop') break
    }
    if (!this.disposed) await this.deps.ledger.settle(resId, amount)
  }

  private async lookupCost(generationId: string): Promise<{ state: 'cost'; micro: number } | { state: 'retry' | 'stop' }> {
    const key = this.deps.key()
    if (!key) return { state: 'stop' }
    try {
      const res = await this.deps.fetch(`${OPENROUTER_BASE}/generation?id=${encodeURIComponent(generationId)}`, {
        headers: { Authorization: `Bearer ${key}`, 'X-OpenRouter-Title': 'RulZ LLM Suite' },
        redirect: 'error',
        signal: AbortSignal.any([this.disposeCtl.signal, AbortSignal.timeout(15_000)])
      })
      if (res.status === 200) {
        const data = ((await res.json()) as { data?: Record<string, unknown> | null }).data
        const total = data?.total_cost
        if (typeof total === 'number' && Number.isFinite(total) && total >= 0) {
          const up = data?.is_byok === true && typeof data.upstream_inference_cost === 'number' && data.upstream_inference_cost > 0 ? data.upstream_inference_cost : 0
          return { state: 'cost', micro: toMicro(total + up) }
        }
        return { state: 'retry' }
      }
      return res.status === 404 || res.status >= 500 ? { state: 'retry' } : { state: 'stop' }
    } catch {
      return { state: 'retry' }
    }
  }

  /** Settles reservations a crash left: those without a generation id at once, the others in the background. */
  async settleLeftovers(): Promise<void> {
    const live = new Set([...this.runs].map((r) => r.resId))
    for (const r of this.deps.ledger.reservations()) if (!live.has(r.id)) await this.settleUnknown(r.id)
  }

  // ---------- routes ----------

  /** The model when this caller may use it now, else the refusal. */
  private allowed(modelId: string, by: CloudCaller): CloudModelDef | Response {
    const def = this.deps.model(modelId)
    if (by !== 'window' && !def?.allowApps) {
      return errorResponse('cloud_not_allowed', 'This OpenRouter model is not available to other apps. Tick "Allow other apps" for it in LLM Suite.')
    }
    return def ?? errorResponse('cloud_request', 'That OpenRouter model is not in your list any more.', 404)
  }

  async fetch(modelId: string, path: string, init: CloudFetchInit, by: CloudCaller): Promise<Response> {
    // Checked on every call, not only for chat: /props and /tokenize too.
    const def = this.allowed(modelId, by)
    if (def instanceof Response) return def
    if (init.signal?.aborted) throw abortError()
    const route = routeOf(path, (init.method ?? 'GET').toUpperCase())
    switch (route.kind) {
      case 'props':
        return json({
          default_generation_settings: { n_ctx: def.contextLimit },
          n_ctx: def.contextLimit,
          modalities: { vision: def.info.vision },
          chat_template: ''
        })
      case 'tokenize': {
        const content = this.parse(init.body)?.content
        if (typeof content !== 'string') return errorResponse('cloud_request', 'Tokenize needs a "content" text.')
        // An estimate: the model's own tokenizer is not here.
        return json({ tokens: Array.from({ length: estimateText(content) }, (_, i) => i) })
      }
      case 'template': {
        const list = this.parse(init.body)?.messages
        if (!Array.isArray(list)) return errorResponse('cloud_request', 'Apply-template needs "messages".')
        const lines = list.map((m: { role?: unknown; content?: unknown }) => `${String(m?.role ?? 'user')}: ${typeof m?.content === 'string' ? m.content : JSON.stringify(m?.content ?? '')}`)
        return json({ prompt: lines.join('\n') + '\nassistant: ' })
      }
      case 'count': {
        const t = translateRequest(init.body, def, this.deps.options(), this.deps.reasoning)
        return t.ok ? json({ input_tokens: t.inputTokens }) : errorResponse(t.code, t.message, t.status)
      }
      case 'chat':
        return this.chat(modelId, def, init, by)
      default:
        return errorResponse('cloud_request', 'This OpenRouter model answers only /props, /tokenize, /apply-template and /v1/chat/completions.')
    }
  }

  private parse(body: string | undefined): Record<string, unknown> | null {
    try {
      const v = JSON.parse(body ?? '')
      return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null
    } catch {
      return null
    }
  }

  private async chat(modelId: string, def: CloudModelDef, init: CloudFetchInit, by: CloudCaller): Promise<Response> {
    const t = translateRequest(init.body, def, this.deps.options(), this.deps.reasoning)
    if (!t.ok) return errorResponse(t.code, t.message, t.status)
    if (!this.deps.key()) return errorResponse('cloud_key', NO_KEY)
    if (this.disposed) return errorResponse('cloud_error', 'The app is closing.')
    const amount = worstCaseMicro(def.info.prices, t.inputTokens, t.images, t.maxTokens)
    const stop = init.signal ? AbortSignal.any([init.signal, this.disposeCtl.signal]) : this.disposeCtl.signal
    const r = await this.deps.ledger.reserve(def.slug, amount, stop)
    if (!r.ok) {
      if (r.kind === 'aborted') {
        if (init.signal?.aborted) throw abortError()
        return errorResponse('cloud_error', 'The app is closing.')
      }
      return errorResponse('cloud_limit', this.deps.scrub(r.message), r.kind === 'error' ? 503 : 402)
    }
    // It may have waited for minutes: the app may be closing, the key forgotten, the model removed or unticked by now.
    const still = this.allowed(modelId, by)
    const key = this.deps.key()
    if (this.disposed || still instanceof Response || !key) {
      await this.deps.ledger.settle(r.id, 0)
      if (this.disposed) return errorResponse('cloud_error', 'The app is closing.')
      return still instanceof Response ? still : errorResponse('cloud_key', NO_KEY)
    }
    if (!this.deps.options().allowTraining) (t.body.provider as Record<string, unknown>).data_collection = 'deny'
    const run: Run = { resId: r.id, ctl: new AbortController(), caller: init.signal, done: false, cancelled: false, cleanup: () => undefined }
    const onAbort = (): void => run.ctl.abort()
    init.signal?.addEventListener('abort', onAbort, { once: true })
    // Aborted while the reservation was being written: the event has passed already.
    if (init.signal?.aborted) run.ctl.abort()
    run.cleanup = () => init.signal?.removeEventListener('abort', onAbort)
    this.runs.add(run)
    try {
      return await this.execute(def, key, t, run)
    } catch (err) {
      // Nothing may be left booked: a throw before anything was sent costs nothing, after that it is unknown.
      await this.finish(run, 'unknown')
      if (init.signal?.aborted) throw abortError()
      return errorResponse('cloud_error', this.deps.scrub(err instanceof Error ? err.message : String(err)))
    }
  }

  private async wait(ms: number, run: Run): Promise<void> {
    try {
      await this.deps.sleep(ms, run.ctl.signal)
    } catch {
      // Aborted: handled below.
    }
    if (run.ctl.signal.aborted) {
      await this.finish(run, this.disposed ? 'unknown' : 'none')
      throw abortError()
    }
  }

  private async execute(def: CloudModelDef, key: string, t: TranslateOk, run: Run): Promise<Response> {
    const idleMs = this.deps.idleMs ?? IDLE_MS
    const watch = new Watchdog(idleMs, () => run.ctl.abort())
    const headers = {
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      'X-OpenRouter-Title': 'RulZ LLM Suite'
    }
    const body = JSON.stringify(t.body)
    const secs = Math.round(idleMs / 1000)
    for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
      if (run.ctl.signal.aborted) {
        await this.finish(run, this.disposed ? 'unknown' : 'none')
        throw abortError()
      }
      if (attempt > 1 && this.deps.key() !== key) {
        await this.finish(run, 'none')
        return errorResponse('cloud_key', 'The OpenRouter key was changed while the request was waiting to be tried again.')
      }
      const last = attempt === MAX_TRIES
      watch.touch()
      let res: Response
      try {
        res = await this.deps.fetch(`${OPENROUTER_BASE}/chat/completions`, { method: 'POST', headers, body, signal: run.ctl.signal, redirect: 'error' })
      } catch (err) {
        watch.stop()
        if (run.caller?.aborted) {
          // The request may have left already: its cost is not known.
          await this.finish(run, 'unknown')
          throw abortError()
        }
        if (this.disposed) {
          await this.finish(run, 'unknown')
          return errorResponse('cloud_error', 'The app is closing.')
        }
        if (watch.timedOut) {
          await this.finish(run, 'unknown')
          return errorResponse('cloud_error', `OpenRouter did not answer for ${secs} seconds. The request may still be billed.`)
        }
        if (isRedirectError(err)) {
          await this.finish(run, 'none')
          return errorResponse('cloud_error', 'OpenRouter answered with a redirect, which is refused so the key stays with openrouter.ai.')
        }
        const code = errCode(err)
        if (code && NOT_SENT.has(code)) {
          if (!last) {
            this.log(`Could not reach OpenRouter (${code}); trying again.`)
            await this.wait(DEFAULT_RETRY_MS[attempt - 1] ?? 8000, run)
            continue
          }
          await this.finish(run, 'none')
          return errorResponse('cloud_error', `Could not reach OpenRouter (${code}). Nothing was sent, so nothing was charged.`)
        }
        await this.finish(run, 'unknown')
        return errorResponse('cloud_error', this.deps.scrub(`The connection to OpenRouter broke before an answer came (${err instanceof Error ? err.message : String(err)}). The request may still be billed.`))
      }
      watch.touch()
      const generation = res.headers.get('x-generation-id')
      // Not the id of an answer that is tried again (429, 503): it costs nothing, and looking it up later would stand for the next try.
      const again = (res.status === 429 || res.status === 503) && !last
      if (generation && !again) await this.deps.ledger.setGenerationId(run.resId, generation)

      if (!res.ok) {
        let text = ''
        try {
          text = (await res.text()).slice(0, 4000)
        } catch {
          // The message is a nicety.
        }
        watch.stop()
        const message = this.deps.scrub(errorFromBody(res.status, text))
        if (again) {
          const ms = retryAfterMs(res.headers.get('retry-after'), this.deps.now()) ?? DEFAULT_RETRY_MS[attempt - 1] ?? 8000
          this.log(`OpenRouter answered ${res.status}; trying again in ${Math.round(ms / 1000)} s.`)
          await this.wait(ms, run)
          continue
        }
        const code = kindForStatus(res.status, message)
        if (res.status >= 500 && res.status !== 503) {
          // It may have run and be billed: the cost is looked up, or the reservation charged.
          await this.finish(run, 'unknown')
          return errorResponse('cloud_error', `OpenRouter failed (${res.status}): ${message}`, 502)
        }
        // 4xx, 429 and 503: nothing was charged.
        await this.finish(run, 'none')
        const status = res.status === 503 ? 503 : res.status
        const extra: Record<string, string> = {}
        const ra = res.headers.get('retry-after')
        if (ra) extra['Retry-After'] = ra
        return errorResponse(code, message, status, extra)
      }
      return this.answer(def, t, run, res, watch, !!generation)
    }
    /* c8 ignore next */
    throw new Error('unreachable')
  }

  /** A 200 answer: the stream, or the stream collected into one JSON answer. */
  private async answer(def: CloudModelDef, t: TranslateOk, run: Run, res: Response, watch: Watchdog, haveGeneration: boolean): Promise<Response> {
    const tr = new ReplyTranslator(def.slug)
    if (!res.body) {
      watch.stop()
      await this.finish(run, 'unknown')
      return errorResponse('cloud_error', 'OpenRouter sent an empty answer.')
    }
    if (!t.stream) {
      const frames: string[] = []
      let failure: unknown
      const sink: Sink = { write: (f) => void frames.push(f), close: () => undefined, fail: (e) => void (failure = e) }
      await this.pump(res.body, run, tr, t, sink, watch, haveGeneration)
      if (failure) throw failure
      // An error ends a stream as an error chunk; a collected reply must not pass for a whole one.
      const broken = run.error
      if (broken) return errorResponse(broken.code, broken.message)
      return json(tr.collected())
    }
    const enc = new TextEncoder()
    let controller!: ReadableStreamDefaultController<Uint8Array>
    let closed = false
    const stream = new ReadableStream<Uint8Array>({
      start: (c) => {
        controller = c
      },
      cancel: () => {
        // The caller stopped reading: the upstream request ends and its cost is looked up.
        closed = true
        run.cancelled = true
        watch.stop()
        run.ctl.abort()
        void this.track(this.finish(run, 'unknown'))
      }
    })
    const sink: Sink = {
      write: (f) => {
        if (closed) return
        try {
          controller.enqueue(enc.encode(f))
        } catch {
          closed = true
        }
      },
      close: () => {
        if (closed) return
        closed = true
        try {
          controller.close()
        } catch {
          // Already closed.
        }
      },
      fail: (e) => {
        if (closed) return
        closed = true
        try {
          controller.error(e)
        } catch {
          // Already closed.
        }
      }
    }
    // Read whether or not the caller does: the cost must be settled even when nobody reads the body.
    void this.track(this.pump(res.body, run, tr, t, sink, watch, haveGeneration))
    return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' } })
  }

  private async pump(body: ReadableStream<Uint8Array>, run: Run, tr: ReplyTranslator, t: TranslateOk, sink: Sink, watch: Watchdog, haveGeneration: boolean): Promise<void> {
    const reader = body.getReader()
    const dec = new TextDecoder()
    const parser = new SseParser()
    let sawDone = false
    let guard = false
    let thrown: unknown = null
    let hasThrown = false
    let genSaved = haveGeneration
    try {
      outer: for (;;) {
        const { done, value } = await reader.read()
        // Any bytes count, comment lines like ": OPENROUTER PROCESSING" too.
        watch.touch()
        if (done) break
        for (const payload of parser.push(dec.decode(value, { stream: true }))) {
          if (payload === '[DONE]') {
            sawDone = true
            break outer
          }
          let chunk: unknown
          try {
            chunk = JSON.parse(payload)
          } catch {
            continue
          }
          const id = (chunk as { id?: unknown } | null)?.id
          if (!genSaved && typeof id === 'string' && id) {
            genSaved = true
            await this.deps.ledger.setGenerationId(run.resId, id)
          }
          const frames = tr.feed(chunk)
          if (tr.error) break outer
          for (const f of frames) sink.write(f)
          if (tr.units > REPLY_GUARD * t.maxTokens) {
            guard = true
            break outer
          }
        }
      }
    } catch (err) {
      hasThrown = true
      thrown = err
    } finally {
      watch.stop()
      if (hasThrown || tr.error || guard) run.ctl.abort()
      reader.cancel().catch(() => undefined)
    }
    if (run.cancelled) return
    const fail = async (e: ErrorInfo): Promise<void> => {
      run.error = e
      await this.finish(run, 'unknown')
      sink.write(errorFrame(e))
      sink.write(DONE_FRAME)
      sink.close()
    }
    if (hasThrown) {
      if (run.caller?.aborted) {
        await this.finish(run, 'unknown')
        sink.fail(abortError())
      } else if (this.disposed) {
        await this.finish(run, 'unknown')
        sink.fail(abortError())
      } else if (watch.timedOut) await fail(errorInfo(`OpenRouter sent nothing for ${Math.round((this.deps.idleMs ?? IDLE_MS) / 1000)} seconds, so the reply was ended.`, 'cloud_error'))
      else await fail(errorInfo(this.deps.scrub(`The connection to OpenRouter broke during the reply (${thrown instanceof Error ? thrown.message : String(thrown)}).`), 'cloud_error'))
      return
    }
    if (tr.error) return fail({ ...tr.error, message: this.deps.scrub(tr.error.message) })
    if (guard) {
      sink.write(`data: ${JSON.stringify({ id: tr.id ?? 'chatcmpl-cloud', object: 'chat.completion.chunk', created: tr.created ?? 0, model: tr.model ?? '', choices: [{ index: 0, delta: {}, finish_reason: 'length' }] })}\n\n`)
      tr.finishReason = 'length'
      await this.finish(run, 'unknown')
      sink.write(DONE_FRAME)
      sink.close()
      return
    }
    if (!sawDone && !tr.finishReason) return fail(errorInfo('The connection to OpenRouter ended before the reply was complete.', 'cloud_error'))
    // A finished stream: missing usage.cost counts as no usage.
    await this.finish(run, tr.costUsd === null ? 'unknown' : { micro: toMicro(tr.costUsd) })
    tr.remember(this.deps.reasoning)
    sink.write(DONE_FRAME)
    sink.close()
  }
}
