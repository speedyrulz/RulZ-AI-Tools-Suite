// OpenRouter models through the API server (docs/cloud-design.md, section 7) and the helpers that pick models.
import http from 'node:http'
import { afterEach, describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { ApiRequestLog, LoadedInstance, ModelEntry } from '@shared/types'
import type { CloudCaller } from '../src/main/cloud/contract'
import type { InstanceManager } from '../src/main/engine/instances'
import { ApiServer } from '../src/main/server/apiServer'
import { pickModel, publicModelId, resolveModel } from '../src/main/server/helpers'
import { cloudEntry, cloudInstance, localEntry, sse } from './helpers/cloudFakes'

const LOCAL = localEntry('Qwen/qwen-9b.gguf', 'qwen-9b')
const ALLOWED = cloudEntry('anthropic/claude-x', { allowApps: true })
const PRIVATE = cloudEntry('openai/gpt-y', { allowApps: false })
const FREE = cloudEntry('meta/free-z', { allowApps: true, info: { ...cloudEntry('q').cloud!.info, free: true } })

describe('picking models with OpenRouter ones in the list', () => {
  const all: ModelEntry[] = [LOCAL, ALLOWED, PRIVATE]
  const on = { defaultModelId: '', unknownModelFallback: true }

  it('finds a cloud model only by its exact id, never by name, substring or case', () => {
    expect(resolveModel(all, 'openrouter/anthropic/claude-x')?.id).toBe(ALLOWED.id)
    expect(resolveModel(all, 'anthropic/claude-x')).toBeUndefined()
    expect(resolveModel(all, 'claude')).toBeUndefined()
    expect(resolveModel(all, 'OPENROUTER/ANTHROPIC/CLAUDE-X')).toBeUndefined()
    expect(resolveModel(all, 'openrouter/anthropic')).toBeUndefined()
    // A local model whose name only a cloud model shares is not ambiguous.
    const twin = cloudEntry('qwen-9b', { allowApps: true })
    expect(resolveModel([LOCAL, twin], 'qwen-9b')?.id).toBe(LOCAL.id)
    expect(publicModelId(LOCAL, [LOCAL, twin])).toBe('qwen-9b')
    expect(publicModelId(twin, [LOCAL, twin])).toBe('openrouter/qwen-9b')
    expect(publicModelId(ALLOWED, all)).toBe('openrouter/anthropic/claude-x')
  })

  it('never falls back to or defaults to a cloud model', () => {
    expect(pickModel(all, 'nope', [ALLOWED.id], on)).toBeUndefined()
    expect(pickModel(all, undefined, [ALLOWED.id], on)).toBeUndefined()
    expect(pickModel(all, 'nope', [], { ...on, defaultModelId: ALLOWED.id })).toBeUndefined()
    expect(pickModel(all, 'nope', [LOCAL.id], on)?.model.id).toBe(LOCAL.id)
    expect(pickModel(all, 'openrouter/anthropic/claude-x', [], on)).toEqual({ model: ALLOWED, fallback: false })
  })
})

const cleanups: (() => Promise<void> | void)[] = []
afterEach(async () => {
  for (const c of cleanups.splice(0).reverse()) await c()
})

interface Call {
  modelId: string
  path: string
  method?: string
  body?: string
  by: CloudCaller | undefined
  aborted: () => boolean
}

async function setup(opts: { allowed?: string[]; models?: ModelEntry[]; bind?: 'localhost' | 'lan'; apiKey?: string; fallback?: boolean; cors?: boolean; respond?: (call: Call) => Response | Promise<Response> } = {}) {
  const calls: Call[] = []
  const log: ApiRequestLog[] = []
  const events: string[] = []
  const instances: LoadedInstance[] = []
  const allowed = new Set(opts.allowed ?? [ALLOWED.id, FREE.id])
  const probe = http.createServer()
  await new Promise<void>((r) => probe.listen(0, '127.0.0.1', () => r()))
  const port = (probe.address() as { port: number }).port
  await new Promise<void>((r) => probe.close(() => r()))
  const models = opts.models ?? [LOCAL, ALLOWED, PRIVATE, FREE]
  const manager = {
    list: () => instances,
    readyFor: (id: string) => instances.find((i) => i.modelId === id && i.status === 'ready'),
    ensure: async (id: string) => {
      events.push(`ensure ${id}`)
      const e = models.find((m) => m.id === id)!
      if (!e.cloud) throw new Error('not here')
      const inst = cloudInstance(e.cloud.slug, { cloud: { provider: 'openrouter', free: e.cloud.info.free }, config: DEFAULT_LOAD_CONFIG })
      instances.push(inst)
      return { instance: inst, started: true }
    },
    load: async () => void events.push('load'),
    unload: async () => void events.push('unload'),
    acquire: (id: string, by?: string) => (events.push(`acquire ${id} ${by}`), () => events.push('release')),
    idleSince: () => 0,
    fetch: async (_i: LoadedInstance, path: string, init?: RequestInit, by?: CloudCaller) => {
      const call: Call = { modelId: _i.modelId, path, method: init?.method, body: init?.body as string | undefined, by, aborted: () => !!init?.signal?.aborted }
      calls.push(call)
      return (
        opts.respond?.(call) ??
        new Response(JSON.stringify({ id: 'x', choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }], usage: { prompt_tokens: 4, completion_tokens: 2 } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        })
      )
    }
  } as unknown as InstanceManager
  const server = new ApiServer({
    instances: manager,
    getModels: () => models,
    cloudAllowed: (id) => allowed.has(id),
    getSettings: () => ({ autoStart: false, port, bind: opts.bind ?? 'localhost', apiKey: opts.apiKey ?? '', cors: opts.cors ?? false, idleUnloadMinutes: 1, defaultModelId: '', unknownModelFallback: opts.fallback ?? true }),
    getLoadDefaults: () => ({ config: DEFAULT_LOAD_CONFIG, custom: false }),
    setLoadDefaults: async () => void events.push('setLoadDefaults'),
    estimate: () => {
      throw new Error('no estimate for a cloud model')
    },
    gpus: () => {
      throw new Error('no gpus for a cloud model')
    },
    onStatus: () => undefined,
    onRequest: (e) => log.push(e),
    modes: {
      get: () => ({}),
      setLoad: async () => void events.push('setModeLoad'),
      takeGate: () => () => undefined
    }
  })
  await server.start()
  cleanups.push(async () => void (await server.stop()))
  return { server, port, calls, log, events, instances, allowed }
}

interface Res {
  status: number
  headers: http.IncomingHttpHeaders
  text: string
  json: Record<string, any>
}
function call(port: number, method: string, path: string, body?: unknown, headers: Record<string, string> = {}, json = true): Promise<Res> {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? undefined : typeof body === 'string' ? body : JSON.stringify(body)
    const req = http.request({ host: '127.0.0.1', port, method, path, headers: { ...(data !== undefined && json ? { 'Content-Type': 'application/json' } : {}), ...headers } }, (res) => {
      const chunks: Buffer[] = []
      res.on('data', (c: Buffer) => chunks.push(c))
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8')
        let parsed: Record<string, any> = {}
        try {
          parsed = JSON.parse(text)
        } catch {
          // not JSON
        }
        resolve({ status: res.statusCode ?? 0, headers: res.headers, text, json: parsed })
      })
    })
    req.on('error', reject)
    req.end(data)
  })
}
const chatBody = (model: string): object => ({ model, messages: [{ role: 'user', content: 'hi' }] })
const CLOUD = 'openrouter/anthropic/claude-x'

describe('the API server and OpenRouter models', () => {
  it('lists only ticked cloud models, as loaded, with their context limit and the cloud field', async () => {
    const t = await setup()
    const r = await call(t.port, 'GET', '/v1/models')
    const ids = (r.json.data as { id: string }[]).map((m) => m.id)
    expect(ids).toEqual([LOCAL.name, CLOUD, 'openrouter/meta/free-z'])
    const entry = (r.json.data as Record<string, any>[]).find((m) => m.id === CLOUD)!
    expect(entry).toMatchObject({ state: 'loaded', loaded_context_length: 32_768, cloud: { provider: 'openrouter', free: false } })
    const free = (r.json.data as Record<string, any>[]).find((m) => m.id === 'openrouter/meta/free-z')!
    expect(free.cloud).toEqual({ provider: 'openrouter', free: true })
    expect((await call(t.port, 'GET', `/v1/models/${CLOUD}`)).json.id).toBe(CLOUD)
    // Not ticked, or no key (cloudAllowed false): as unknown as a model that does not exist.
    expect((await call(t.port, 'GET', '/v1/models/openrouter/openai/gpt-y')).status).toBe(404)
    t.allowed.clear()
    expect(((await call(t.port, 'GET', '/v1/models')).json.data as { id: string }[]).map((m) => m.id)).toEqual([LOCAL.name])
  })

  it('answers a chat request through the gateway as the api, with the cloud header, usage logged and the model connected first', async () => {
    const t = await setup()
    const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))
    expect(r.status).toBe(200)
    expect(r.headers['x-rulz-cloud']).toBe('1')
    expect(r.headers['x-rulz-model']).toBe(CLOUD)
    expect(r.json.choices[0].message.content).toBe('hi')
    expect(t.events).toEqual(['ensure openrouter/anthropic/claude-x', 'acquire c-anthropic/claude-x api', 'release'])
    expect(t.calls).toHaveLength(1)
    expect(t.calls[0]).toMatchObject({ path: '/v1/chat/completions', method: 'POST', by: 'api' })
    expect(JSON.parse(t.calls[0].body!)).toMatchObject({ model: CLOUD })
    await new Promise((r) => setTimeout(r, 20))
    expect(t.log.at(-1)).toMatchObject({ path: '/v1/chat/completions', model: CLOUD, status: 200, promptTokens: 4, completionTokens: 2 })
    // Connected now: the second request does not connect again, and the server does not track it for idle-unload.
    await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))
    expect(t.events.filter((e) => e.startsWith('ensure'))).toHaveLength(1)
    expect(t.server.getStatus().jitInstances).toEqual([])
  })

  it('streams the gateway answer through, including usage, and stops the request when the client leaves', async () => {
    const stream = sse([{ choices: [{ delta: { content: 'a' }, finish_reason: null }] }, { choices: [{ delta: { content: 'b' }, finish_reason: 'stop' }], usage: { prompt_tokens: 3, completion_tokens: 2 } }])
    const t = await setup({ respond: () => new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }) })
    const r = await call(t.port, 'POST', '/v1/chat/completions', { ...chatBody(CLOUD), stream: true })
    expect(r.status).toBe(200)
    expect(r.headers['content-type']).toMatch(/event-stream/)
    expect(r.text).toContain('"content":"b"')
    expect(r.text).toContain('[DONE]')
    await new Promise((r) => setTimeout(r, 20))
    expect(t.log.at(-1)).toMatchObject({ stream: true, promptTokens: 3, completionTokens: 2 })

    // A reply that never ends: the client disconnects, and the abort reaches the gateway.
    const hang = await setup({
      respond: (c) =>
        new Response(
          new ReadableStream({
            start(ctrl) {
              ctrl.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"x"}}]}\n\n'))
              const t = setInterval(() => c.aborted() && (clearInterval(t), ctrl.error(new Error('aborted'))), 5)
            }
          }),
          { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
        )
    })
    await new Promise<void>((resolve) => {
      const req = http.request({ host: '127.0.0.1', port: hang.port, method: 'POST', path: '/v1/chat/completions', headers: { 'Content-Type': 'application/json' } }, (res) => {
        res.once('data', () => {
          req.destroy()
          resolve()
        })
      })
      req.end(JSON.stringify({ ...chatBody(CLOUD), stream: true }))
    })
    await new Promise((r) => setTimeout(r, 80))
    expect(hang.calls[0].aborted()).toBe(true)
    expect(hang.log.at(-1)?.status).toBe(499)
  })

  it('passes a gateway error on with its status', async () => {
    const t = await setup({ respond: () => new Response(JSON.stringify({ error: { message: 'Daily limit reached', type: 'x', code: 'cloud_limit' } }), { status: 402, headers: { 'Content-Type': 'application/json' } }) })
    const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))
    expect(r.status).toBe(402)
    expect(r.json.error.code).toBe('cloud_limit')
    expect(r.headers['x-rulz-cloud']).toBe('1')
    await new Promise((r) => setTimeout(r, 20))
    expect(t.log.at(-1)).toMatchObject({ status: 402, error: 'Daily limit reached' })
  })

  it('answers a gateway that throws with 502 and does not hang', async () => {
    const t = await setup({
      respond: () => {
        throw new Error('gateway broke')
      }
    })
    const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))
    expect(r.status).toBe(502)
    expect(r.json.error.message).toBe('gateway broke')
  })

  it('refuses a cloud model that is not ticked, with a code the apps know, on every request', async () => {
    const t = await setup()
    expect((await call(t.port, 'POST', '/v1/chat/completions', chatBody('openrouter/openai/gpt-y'))).status).toBe(403)
    // Ticked, then un-ticked between two requests.
    expect((await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))).status).toBe(200)
    t.allowed.delete(ALLOWED.id)
    const after = await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))
    expect(after.status).toBe(403)
    expect(after.json.error.code).toBe('cloud_not_allowed')
    expect(t.calls).toHaveLength(1)
  })

  it('names that are not available never get another model instead, even with one loaded and the fallback on', async () => {
    const t = await setup({ fallback: true })
    t.instances.push({ instanceId: 'l', modelId: LOCAL.id, modelName: LOCAL.name, port: 1, pid: 1, status: 'ready', progress: 1, stage: 'Ready', config: DEFAULT_LOAD_CONFIG, startedAt: 1, contextLength: 4096, hasVision: false, adapters: [], estimate: null })
    for (const name of [PRIVATE.id, 'openrouter/not/added']) {
      const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody(name))
      expect(r.status).toBe(403)
      expect(r.json.error.code).toBe('cloud_not_allowed')
    }
    expect(t.events).toEqual([])
    expect(t.calls).toEqual([])
  })

  it('never answers a name that is not a cloud id with a cloud model, whatever the fallback', async () => {
    const t = await setup({ models: [ALLOWED, FREE] })
    for (const name of ['claude-x', 'anthropic/claude-x', 'free', undefined]) {
      const r = await call(t.port, 'POST', '/v1/chat/completions', name ? chatBody(name) : { messages: [] })
      expect(r.status).toBe(404)
    }
    expect(t.calls).toEqual([])
    expect(t.events).toEqual([])
  })

  it('wants application/json, no Origin and an own Host, even with CORS on', async () => {
    const t = await setup({ cors: true })
    const body = JSON.stringify(chatBody(CLOUD))
    const plain = await call(t.port, 'POST', '/v1/chat/completions', body, { 'Content-Type': 'text/plain' }, false)
    expect(plain.status).toBe(415)
    const fromPage = await call(t.port, 'POST', '/v1/chat/completions', body, { Origin: 'https://evil.example' })
    expect(fromPage.status).toBe(403)
    const rebound = await call(t.port, 'POST', '/v1/chat/completions', body, { Host: 'evil.example.com' })
    expect(rebound.status).toBe(403)
    expect(t.calls).toEqual([])
    expect((await call(t.port, 'POST', '/v1/chat/completions', body)).status).toBe(200)
  })

  it('needs the API key to be set when the server listens on the network', async () => {
    const open = await setup({ bind: 'lan' })
    const refused = await call(open.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))
    expect(refused.status).toBe(403)
    expect(refused.json.error.message).toMatch(/API key/)
    const keyed = await setup({ bind: 'lan', apiKey: 'secret' })
    expect((await call(keyed.port, 'POST', '/v1/chat/completions', chatBody(CLOUD))).status).toBe(401)
    expect((await call(keyed.port, 'POST', '/v1/chat/completions', chatBody(CLOUD), { Authorization: 'Bearer secret' })).status).toBe(200)
  })

  it('proxies only the gateway paths: props, tokenize, apply-template, chat completions and input_tokens', async () => {
    const t = await setup()
    for (const [method, path] of [
      ['POST', '/v1/completions'],
      ['POST', '/v1/embeddings'],
      ['POST', '/v1/messages'],
      ['POST', '/v1/responses'],
      ['POST', '/completion'],
      ['POST', '/detokenize'],
      ['GET', '/slots']
    ] as const) {
      const r = await call(t.port, method, `${path}${method === 'GET' ? `?model=${CLOUD}` : ''}`, method === 'POST' ? chatBody(CLOUD) : undefined)
      expect(r.status, `${method} ${path}`).toBe(400)
    }
    expect(t.calls).toEqual([])
    for (const [method, path] of [
      ['POST', '/tokenize'],
      ['POST', '/apply-template'],
      ['POST', '/v1/chat/completions/input_tokens']
    ] as const) {
      expect((await call(t.port, method, path, chatBody(CLOUD))).status).toBe(200)
    }
    expect((await call(t.port, 'GET', `/props?model=${encodeURIComponent(CLOUD)}`)).status).toBe(200)
    expect(t.calls.map((c) => `${c.method ?? 'GET'} ${c.path}`)).toEqual(['POST /tokenize', 'POST /apply-template', 'POST /v1/chat/completions/input_tokens', 'GET /props'])
    expect(t.calls.every((c) => c.by === 'api')).toBe(true)
  })

  it('refuses a request that says it must not leave this computer for a rental', async () => {
    const t = await setup()
    const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody(CLOUD), { 'x-rulz-rental': 'deny' })
    expect(r.status).toBe(409)
    expect(t.calls).toEqual([])
  })

  it('answers load-config, modes and the model entry with no gpus or estimate, and refuses every change, reload and unload with 409', async () => {
    const t = await setup()
    const base = `/v1/models/${CLOUD}`
    const cfg = await call(t.port, 'GET', `${base}/load-config`)
    expect(cfg.status).toBe(200)
    expect(cfg.json).toMatchObject({ id: CLOUD, gpus: [], estimate: null, draft_choices: [], adapter_choices: [], max_context_length: 200_000 })
    const modes = await call(t.port, 'GET', `${base}/modes`)
    expect(modes.status).toBe(200)
    expect(modes.json.modes.coding.matches).toBe(true)
    expect(modes.json.modes.re.matches).toBe(true)
    expect((await call(t.port, 'GET', `${base}/modes/coding/load-config`)).status).toBe(200)

    for (const [path, body] of [
      [`${base}/load-config`, { config: { contextLength: 8192 } }],
      [`${base}/modes/coding/load-config`, { config: { contextLength: 8192 } }],
      [`${base}/reload`, { mode: 'coding' }],
      [`${base}/unload`, {}]
    ] as const) {
      const r = await call(t.port, 'POST', path, body)
      expect(r.status, path).toBe(409)
      expect(r.json.error.message).toMatch(/OpenRouter/)
    }
    expect(t.events).toEqual([])
    // Not ticked: the same routes are 404, like for an unknown model.
    expect((await call(t.port, 'GET', '/v1/models/openrouter/openai/gpt-y/load-config')).status).toBe(404)
    expect((await call(t.port, 'POST', '/v1/models/openrouter/openai/gpt-y/unload', {})).status).toBe(404)
  })

  it('leaves local models as they were: a local request still goes to its own port path and the cloud gateway is not touched', async () => {
    const t = await setup()
    // The fake manager has no local instance; a request for the local model tries to load it and fails there, not at the gateway.
    const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody('qwen-9b'))
    expect(r.status).toBe(503)
    expect(t.calls).toEqual([])
  })
})
