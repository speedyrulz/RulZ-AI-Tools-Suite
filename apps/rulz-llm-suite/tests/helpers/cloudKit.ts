// Helpers for tests/cloud-*.test.ts only (fake network, clock and sleep, a ready-made gateway on a temp folder). Nothing
// in the app imports this file.
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { CloudModelDef, CloudModelInfo } from '@shared/cloud'
import { parseCatalog } from '../../src/main/cloud/catalog'
import { abortError, CloudGatewayCore } from '../../src/main/cloud/gateway'
import { CloudKeyStore } from '../../src/main/cloud/keyStore'
import { CloudLedger } from '../../src/main/cloud/ledger'
import { CloudModelStore } from '../../src/main/cloud/models'
import { ReasoningCache } from '../../src/main/cloud/reasoningCache'

export const TEST_KEY = 'sk-or-v1-0123456789abcdef0123456789abcdef'

export const frame = (o: unknown): string => `data: ${JSON.stringify(o)}\n\n`
export const DONE = 'data: [DONE]\n\n'
const enc = new TextEncoder()

/** A streamed OpenRouter chunk. */
export function chunk(delta: Record<string, unknown>, extra: Record<string, unknown> = {}, finish: string | null = null): string {
  return frame({ id: 'gen-1', object: 'chat.completion.chunk', created: 1, model: 'm', choices: [{ index: 0, delta, finish_reason: finish }], ...extra })
}

export const usageChunk = (cost: number, extra: Record<string, unknown> = {}): string =>
  frame({ id: 'gen-1', object: 'chat.completion.chunk', created: 1, model: 'm', choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, cost, ...extra } })

/** A complete good stream of one text reply that cost `cost` dollars. */
export const goodStream = (text = 'Hello', cost = 0.0005): string[] => [chunk({ role: 'assistant', content: '' }), chunk({ content: text }), chunk({}, {}, 'stop'), usageChunk(cost), DONE]

/** A body that gives the parts in order. `hold`: stays open afterwards (until aborted), like a stalled connection. */
export function makeBody(parts: (string | Uint8Array)[], signal?: AbortSignal | null, hold = false): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(c) {
      const onAbort = (): void => {
        try {
          c.error(new DOMException('aborted', 'AbortError'))
        } catch {
          // Already closed.
        }
      }
      if (signal?.aborted) return onAbort()
      signal?.addEventListener('abort', onAbort, { once: true })
      for (const p of parts) c.enqueue(typeof p === 'string' ? enc.encode(p) : p)
      if (!hold) c.close()
    }
  })
}

/** A body fed by hand. */
export class Pipe {
  private c!: ReadableStreamDefaultController<Uint8Array>
  private closed = false
  body(signal?: AbortSignal | null): ReadableStream<Uint8Array> {
    return new ReadableStream<Uint8Array>({
      start: (c) => {
        this.c = c
        signal?.addEventListener(
          'abort',
          () => {
            this.closed = true
            try {
              c.error(new DOMException('aborted', 'AbortError'))
            } catch {
              // Closed already.
            }
          },
          { once: true }
        )
      }
    })
  }
  push(s: string | Uint8Array): void {
    if (!this.closed) this.c.enqueue(typeof s === 'string' ? enc.encode(s) : s)
  }
  end(): void {
    if (this.closed) return
    this.closed = true
    this.c.close()
  }
  fail(err: unknown): void {
    if (this.closed) return
    this.closed = true
    this.c.error(err)
  }
}

export interface Call {
  url: string
  method: string
  headers: Record<string, string>
  body: Record<string, any> | null
  rawBody: string | undefined
  signal: AbortSignal | undefined
  redirect: string | undefined
}

export type NetHandler = (call: Call, n: number) => Response | Error | Promise<Response | Error>

/** A fake fetch: records every call, answers with the handler. */
export class FakeNet {
  calls: Call[] = []
  constructor(public handler: NetHandler = () => new Response('{}', { status: 404 })) {}
  fetch: typeof fetch = async (input, init) => {
    const headers: Record<string, string> = {}
    new Headers(init?.headers).forEach((v, k) => (headers[k] = v))
    let body: Record<string, any> | null = null
    const rawBody = typeof init?.body === 'string' ? init.body : undefined
    try {
      body = rawBody ? JSON.parse(rawBody) : null
    } catch {
      // Not JSON.
    }
    const call: Call = { url: String(input), method: init?.method ?? 'GET', headers, body, rawBody, signal: init?.signal ?? undefined, redirect: init?.redirect }
    this.calls.push(call)
    const r = await this.handler(call, this.calls.length)
    if (r instanceof Error) throw r
    return r
  }
  /** The calls to OpenRouter's chat endpoint. */
  get chats(): Call[] {
    return this.calls.filter((c) => c.url.endsWith('/chat/completions'))
  }
}

export const sseResponse = (call: Call, parts: (string | Uint8Array)[], headers: Record<string, string> = {}, hold = false): Response =>
  new Response(makeBody(parts, call.signal, hold), { status: 200, headers: { 'Content-Type': 'text/event-stream', ...headers } })

export const jsonResponse = (v: unknown, status = 200, headers: Record<string, string> = {}): Response => new Response(JSON.stringify(v), { status, headers: { 'Content-Type': 'application/json', ...headers } })

export const connectError = (code: string): Error => Object.assign(new TypeError('fetch failed'), { cause: Object.assign(new Error(code), { code }) })

/** A sleep the test controls: it records the wait and, unless `auto`, waits until released or aborted. */
export class Sleeper {
  waits: { ms: number; release: () => void }[] = []
  auto = false
  sleep = (ms: number, signal?: AbortSignal): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      if (signal?.aborted) return reject(abortError())
      const w = { ms, release: () => resolve() }
      this.waits.push(w)
      if (this.auto) return resolve()
      signal?.addEventListener('abort', () => reject(abortError()), { once: true })
    })
}

export class Clock {
  constructor(public t = new Date(2026, 10, 25, 12, 0, 0).getTime()) {}
  now = (): number => this.t
  advance(ms: number): void {
    this.t += ms
  }
}

export const fakeSafeStorage = (available = true): { isEncryptionAvailable(): boolean; encryptString(t: string): Buffer; decryptString(b: Buffer): string } => ({
  isEncryptionAvailable: () => available,
  encryptString: (t) => Buffer.from(`enc:${t}`),
  decryptString: (b) => {
    const s = b.toString()
    if (!s.startsWith('enc:')) throw new Error('cannot decrypt')
    return s.slice(4)
  }
})

export const pickInfo = (fixture: unknown, slug: string, patch: Partial<CloudModelInfo> = {}): CloudModelInfo => {
  const info = parseCatalog(fixture).find((m) => m.slug === slug)
  if (!info) throw new Error(`fixture has no usable ${slug}`)
  return { ...info, ...patch }
}

export interface Rig {
  dir: string
  gw: CloudGatewayCore
  ledger: CloudLedger
  store: CloudModelStore
  keys: CloudKeyStore
  net: FakeNet
  clock: Clock
  sleeper: Sleeper
  cache: ReasoningCache
  logs: string[]
  def: CloudModelDef
  modelId: string
  cleanup(): void
}

export interface RigOptions {
  fixture: unknown
  slug?: string
  patchInfo?: Partial<CloudModelInfo>
  /** USD; null = no limit. Default 1. */
  limitUsd?: number | null
  idleMs?: number
  key?: string | null
  allowApps?: boolean
  contextLimit?: number
  replyLimit?: number
  allowTraining?: boolean
  net?: FakeNet
  dir?: string
}

/** A gateway over a temp folder with a model added, a key saved and the ledger loaded. */
export async function makeRig(o: RigOptions): Promise<Rig> {
  const dir = o.dir ?? mkdtempSync(join(tmpdir(), 'cloud-test-'))
  const clock = new Clock()
  const sleeper = new Sleeper()
  const net = o.net ?? new FakeNet()
  const keys = new CloudKeyStore({ path: join(dir, 'cloud-secrets.json'), safeStorage: fakeSafeStorage() })
  if (o.key !== null) await keys.save(o.key ?? TEST_KEY)
  const store = new CloudModelStore({ path: join(dir, 'cloud.json'), now: clock.now })
  await store.load()
  const slug = o.slug ?? 'anthropic/claude-fable-5.2'
  if (!store.get(slug)) await store.add(pickInfo(o.fixture, slug, o.patchInfo))
  await store.update(slug, { allowApps: o.allowApps ?? false, ...(o.contextLimit ? { contextLimit: o.contextLimit } : {}), ...(o.replyLimit ? { replyLimit: o.replyLimit } : {}) })
  await store.setOptions({ dailyLimitUsd: o.limitUsd === undefined ? 1 : o.limitUsd, allowTraining: o.allowTraining ?? false })
  const ledger = new CloudLedger({
    path: join(dir, 'cloud-spend.json'),
    now: clock.now,
    sleep: sleeper.sleep,
    limitMicro: () => {
      const l = store.options().dailyLimitUsd
      return l === null ? null : Math.ceil(l * 1e6)
    }
  })
  await ledger.load()
  const cache = new ReasoningCache()
  const logs: string[] = []
  const gw = new CloudGatewayCore({
    fetch: net.fetch,
    now: clock.now,
    sleep: sleeper.sleep,
    key: () => keys.get(),
    model: (id) => (id.startsWith('openrouter/') ? store.get(id.slice('openrouter/'.length)) : undefined),
    options: () => store.options(),
    ledger,
    reasoning: cache,
    scrub: (s) => keys.scrub(s),
    idleMs: o.idleMs,
    log: (l) => logs.push(l)
  })
  return {
    dir,
    gw,
    ledger,
    store,
    keys,
    net,
    clock,
    sleeper,
    cache,
    logs,
    def: store.get(slug)!,
    modelId: `openrouter/${slug}`,
    cleanup: () => {
      gw.dispose()
      rmSync(dir, { recursive: true, force: true })
    }
  }
}
