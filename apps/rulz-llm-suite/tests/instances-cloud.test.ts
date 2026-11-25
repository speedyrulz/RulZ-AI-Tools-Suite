// OpenRouter models in the instance layer and the chat service (docs/cloud-design.md, sections 2, 3 and 6).
import { keyOfModel, modelKey } from '../src/shared/practice'
import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import type { LoadedInstance, ModelEntry, StreamEvent } from '@shared/types'
import { ChatService } from '../src/main/chat/service'
import { errorCodeFromBody, interpretChunk, statsFromUsage } from '../src/main/chat/sse'
import type { CloudCaller, CloudGateway } from '../src/main/cloud/contract'
import type { EngineService } from '../src/main/engine/engine'
import { InstanceManager, type InstanceManagerDeps } from '../src/main/engine/instances'
import type { InstanceManager as InstanceManagerType } from '../src/main/engine/instances'
import { cloudEntry, cloudInstance, localEntry, sse } from './helpers/cloudFakes'

class FakeServer extends EventEmitter {
  stdout = new EventEmitter()
  stderr = new EventEmitter()
  constructor(readonly name: string) {
    super()
  }
}

interface Rig {
  mgr: InstanceManager
  events: string[]
  gateway: CloudGateway & { check: ReturnType<typeof vi.fn>; props: ReturnType<typeof vi.fn>; fetch: ReturnType<typeof vi.fn> }
  gone: string[]
  models: Map<string, ModelEntry>
}

function rig(over: Partial<InstanceManagerDeps> = {}): Rig {
  const events: string[] = []
  const gone: string[] = []
  const models = new Map<string, ModelEntry>([
    ['a', localEntry('a')],
    ['b', localEntry('b')],
    ['openrouter/x/one', cloudEntry('x/one')],
    ['openrouter/y/two', cloudEntry('y/two')]
  ])
  const gateway = {
    check: vi.fn(),
    props: vi.fn(() => ({ contextLength: 32_768, hasVision: true, free: false })),
    fetch: vi.fn(async () => new Response('{}'))
  } as Rig['gateway']
  const mgr = new InstanceManager({
    engine: { resolveExe: async () => 'C:\\llama\\llama-server.exe' } as unknown as EngineService,
    getModel: (id) => models.get(id),
    draftFor: () => {
      throw new Error('a cloud model has no draft')
    },
    getDefaults: () => {
      throw new Error('a cloud model has no saved load settings')
    },
    unloadPreviousOnLoad: () => true,
    estimate: () => {
      throw new Error('a cloud model has no estimate')
    },
    pidFile: join(mkdtempSync(join(tmpdir(), 'rulz-inst-cloud-')), 'pids.json'),
    onChange: () => undefined,
    onError: () => undefined,
    onGone: (id) => void gone.push(id),
    spawnServer: (_exe, args) => {
      const name = args[args.indexOf('--alias') + 1]
      events.push(`spawn ${name}`)
      return new FakeServer(name) as unknown as ChildProcess
    },
    killProcess: (proc) => {
      const p = proc as unknown as FakeServer
      events.push(`kill ${p.name}`)
      setTimeout(() => p.emit('exit', 0), 5)
    },
    cloud: gateway,
    ...over
  })
  // The local model's getDefaults is needed for local loads in these tests only.
  return { mgr, events, gateway, gone, models }
}

describe('InstanceManager with an OpenRouter model', () => {
  it('connects at once: ready, port 0, no process, the slug as its name, props from the gateway', async () => {
    const r = rig()
    const { instance, started } = await r.mgr.ensure('openrouter/x/one')
    expect(started).toBe(true)
    expect(instance).toMatchObject({
      modelId: 'openrouter/x/one',
      modelName: 'x/one',
      port: 0,
      pid: null,
      status: 'ready',
      contextLength: 32_768,
      hasVision: true,
      adapters: [],
      estimate: null,
      cloud: { provider: 'openrouter', free: false }
    })
    expect(instance.host).toBeUndefined()
    expect(instance.config.sampling.override).toBe(false)
    expect(Object.isFrozen(instance.config)).toBe(true)
    expect(r.events).toEqual([])
    expect(r.gateway.check).toHaveBeenCalledWith('openrouter/x/one')
    expect(r.mgr.getLog(instance.instanceId).join('\n')).toMatch(/OpenRouter/)
    expect(r.mgr.readyFor('openrouter/x/one')?.instanceId).toBe(instance.instanceId)
  })

  it('has one instance per model: a second ensure or load returns it', async () => {
    const r = rig()
    const first = await r.mgr.ensure('openrouter/x/one')
    const again = await r.mgr.ensure('openrouter/x/one')
    expect(again.started).toBe(false)
    expect(again.instance.instanceId).toBe(first.instance.instanceId)
    expect((await r.mgr.load('openrouter/x/one', null)).instanceId).toBe(first.instance.instanceId)
    await r.mgr.ensure('openrouter/y/two')
    expect(r.mgr.list()).toHaveLength(2)
  })

  it('fails with the gateway sentence when it cannot be connected (no key, not in the list)', async () => {
    const r = rig()
    r.gateway.check.mockImplementation(() => {
      throw new Error('Save your OpenRouter key first.')
    })
    await expect(r.mgr.ensure('openrouter/x/one')).rejects.toThrow('Save your OpenRouter key first.')
    expect(r.mgr.list()).toEqual([])
  })

  it('refuses a host, and without a gateway says it is not available', async () => {
    const r = rig()
    await expect(r.mgr.load('openrouter/x/one', null, { kind: 'vast', rentalId: 1, label: 'x' })).rejects.toThrow(/OpenRouter, not on a rental/)
    const none = rig({ cloud: undefined })
    await expect(none.mgr.ensure('openrouter/x/one')).rejects.toThrow(/not available in this build/)
  })

  it('waits for nothing on this PC, and a load on this PC unloads nothing at OpenRouter (and the other way round)', async () => {
    // The local queue is stuck for good: the engine never resolves.
    const stuck = rig({ engine: { resolveExe: () => new Promise(() => undefined) } as unknown as EngineService })
    const blocked = stuck.mgr.ensure('a').catch(() => undefined)
    void blocked
    await expect(stuck.mgr.ensure('openrouter/x/one')).resolves.toMatchObject({ started: true })

    const r = rig({ getDefaults: () => DEFAULT_LOAD_CONFIG, estimate: () => null, draftFor: () => undefined })
    await r.mgr.ensure('openrouter/x/one')
    const a = r.mgr.load('a', null)
    a.catch(() => undefined)
    await vi.waitFor(() => expect(r.events).toEqual(['spawn a']))
    const b = r.mgr.load('b', null)
    b.catch(() => undefined)
    await vi.waitFor(() => expect(r.events).toEqual(['spawn a', 'kill a', 'spawn b']))
    expect(r.mgr.list().map((i) => i.modelName).sort()).toEqual(['b', 'x/one'])
    // A cloud connect does not unload the local model either.
    await r.mgr.ensure('openrouter/y/two')
    expect(r.events).toEqual(['spawn a', 'kill a', 'spawn b'])
    // "Unload everything on this PC" leaves OpenRouter alone; eject all takes it too.
    await r.mgr.unloadAllLocal()
    expect(r.mgr.list().map((i) => i.modelName).sort()).toEqual(['x/one', 'y/two'])
    await r.mgr.unloadAll()
    expect(r.mgr.list()).toEqual([])
  })

  it('unloads by just removing the instance and calls onGone, with no error event', async () => {
    const errors: unknown[] = []
    const r = rig({ onError: (e) => void errors.push(e) })
    const { instance } = await r.mgr.ensure('openrouter/x/one')
    await r.mgr.unload(instance.instanceId)
    expect(r.mgr.list()).toEqual([])
    expect(r.gone).toEqual([instance.instanceId])
    expect(errors).toEqual([])
    await r.mgr.unload(instance.instanceId)
  })

  it('counts users and idle time like any instance, and shutdown does not throw', async () => {
    const r = rig()
    const { instance } = await r.mgr.ensure('openrouter/x/one')
    const onUse = vi.fn(() => () => undefined)
    const used = rig({ onUse })
    const u = (await used.mgr.ensure('openrouter/x/one')).instance
    const release = used.mgr.acquire(u.instanceId, 'api', 'api request')
    expect(used.mgr.usersOf(u.instanceId)).toBe(1)
    expect(used.mgr.idleSince(u.instanceId)).toBeNull()
    release()
    expect(used.mgr.usersOf(u.instanceId)).toBe(0)
    expect(used.mgr.idleSince(u.instanceId)).not.toBeNull()
    expect(onUse).toHaveBeenCalledWith(expect.objectContaining({ cloud: expect.anything() }), 'api', 'api request')
    expect(() => r.mgr.killAllSync()).not.toThrow()
    expect(r.mgr.list().every((i) => i.status === 'stopping')).toBe(true)
    expect(instance.status).toBe('ready')
  })

  it('refreshCloud updates the context and vision of connected models and ejects one that is gone', async () => {
    const r = rig()
    const one = (await r.mgr.ensure('openrouter/x/one')).instance
    const two = (await r.mgr.ensure('openrouter/y/two')).instance
    r.gateway.props.mockImplementation((id: string) => ({ contextLength: id.endsWith('one') ? 8192 : 32_768, hasVision: false, free: true }))
    r.gateway.check.mockImplementation((id: string) => {
      if (id.endsWith('two')) throw new Error('gone')
    })
    await r.mgr.refreshCloud()
    expect(r.mgr.get(one.instanceId)).toMatchObject({ contextLength: 8192, hasVision: false, cloud: { free: true } })
    expect(r.mgr.get(one.instanceId)?.config.contextLength).toBe(8192)
    expect(r.mgr.get(two.instanceId)).toBeUndefined()
    expect(r.gone).toEqual([two.instanceId])
  })

  it('fetch: the gateway for a cloud instance (with who asked), the global fetch on the port for others', async () => {
    const r = rig()
    const { instance } = await r.mgr.ensure('openrouter/x/one')
    const ctrl = new AbortController()
    await r.mgr.fetch(instance, '/tokenize', { method: 'POST', body: '{"content":"hi"}', signal: ctrl.signal }, 'api')
    expect(r.gateway.fetch).toHaveBeenCalledWith('openrouter/x/one', '/tokenize', { method: 'POST', body: '{"content":"hi"}', signal: ctrl.signal }, 'api')
    await r.mgr.fetch(instance, '/props')
    expect(r.gateway.fetch).toHaveBeenLastCalledWith('openrouter/x/one', '/props', { method: undefined, body: undefined, signal: undefined }, 'window')
    await expect(r.mgr.fetch(instance, '/x', { method: 'POST', body: new Uint8Array(1) })).rejects.toThrow(/text body/)

    const seen: string[] = []
    vi.stubGlobal('fetch', async (url: string) => (seen.push(url), new Response('{}')))
    try {
      const local = { ...instance, cloud: undefined, port: 4321 } as LoadedInstance
      await r.mgr.fetch(local, '/props', { signal: undefined })
      expect(seen).toEqual(['http://127.0.0.1:4321/props'])
    } finally {
      vi.unstubAllGlobals()
    }
  })
})

describe('sse: OpenRouter usage and error codes', () => {
  it('reads usage and the error code from a chunk', () => {
    const usage = { prompt_tokens: 10, completion_tokens: 5, cost: 0.0000123 }
    expect(interpretChunk({ choices: [], usage }).usage).toEqual(usage)
    expect(interpretChunk({ error: { message: 'No credits', code: 'cloud_credits' } })).toEqual({ error: 'No credits', errorCode: 'cloud_credits' })
    expect(interpretChunk({ error: { message: 'x', code: 402 } }).errorCode).toBeUndefined()
    expect(errorCodeFromBody('{"error":{"message":"m","type":"t","code":"cloud_limit"}}')).toBe('cloud_limit')
    expect(errorCodeFromBody('nope')).toBeUndefined()
  })

  it('statsFromUsage fills tokens, cache and the cost rounded up to a micro-dollar; a BYOK cost is not added twice', () => {
    const s = statsFromUsage({ prompt_tokens: 100, completion_tokens: 50, prompt_tokens_details: { cached_tokens: 40 }, cost: 0.0000123 }, {}, 2)
    expect(s).toEqual({ promptTokens: 100, predictedTokens: 50, cachedTokens: 40, costMicro: 13, tokensPerSecond: 25 })
    expect(statsFromUsage({ cost: 0.000003 }, {}, 0).costMicro).toBe(3)
    // The gateway has already added the provider's part of a BYOK request to `cost`: it is not added twice.
    expect(statsFromUsage({ cost: 0.00041, is_byok: true, cost_details: { upstream_inference_cost: 0.0004 } } as never, {}, 0).costMicro).toBe(410)
    // An earlier tokens/s (from llama-server timings) is not replaced.
    expect(statsFromUsage({ completion_tokens: 5 }, { tokensPerSecond: 99 }, 1).tokensPerSecond).toBe(99)
  })
})

describe('ChatService with an OpenRouter model', () => {
  type FetchCall = { path: string; by: CloudCaller | undefined; body: Record<string, unknown> }

  function service(respond: (call: FetchCall) => Response | Promise<Response>, over: Partial<LoadedInstance> = {}) {
    const events: StreamEvent[] = []
    const calls: FetchCall[] = []
    const inst = cloudInstance('x/one', over)
    const manager = {
      readyFor: () => inst,
      get: () => inst,
      acquire: () => () => undefined,
      baseUrl: () => {
        throw new Error('a cloud instance has no address')
      },
      fetch: async (_i: LoadedInstance, path: string, init?: RequestInit, by?: CloudCaller) => {
        const call = { path, by, body: init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {} }
        calls.push(call)
        return respond(call)
      }
    } as unknown as InstanceManagerType
    return { chat: new ChatService(manager, (e) => events.push(e)), events, calls }
  }
  const request = (extra: Partial<Parameters<ChatService['start']>[0]> = {}): Parameters<ChatService['start']>[0] => ({
    requestId: 'q1',
    modelId: 'openrouter/x/one',
    messages: [{ role: 'user', content: 'hi' }],
    sampling: DEFAULT_SAMPLING,
    overflow: 'stop',
    ...extra
  })
  const chunk = (delta: object, finish?: string): object => ({ choices: [{ delta, finish_reason: finish ?? null }] })
  const reply = (body: string): Response => new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
  // The context counting calls of a cloud model: the gateway answers /tokenize and /apply-template.
  const counting = (call: FetchCall): Response | null =>
    call.path === '/tokenize' ? Response.json({ tokens: [1, 2, 3] }) : call.path === '/apply-template' ? Response.json({ prompt: 'p' }) : null
  const last = (events: StreamEvent[]): StreamEvent => events[events.length - 1]

  it('sends who asked with every call, the counting calls included, and fills tokens, cache and cost from usage', async () => {
    const { chat, events, calls } = service((call) =>
      counting(call) ??
      reply(
        sse([
          chunk({ reasoning_content: 'hm' }),
          chunk({ content: 'Hi' }),
          chunk({}, 'stop'),
          { choices: [], usage: { prompt_tokens: 12, completion_tokens: 7, prompt_tokens_details: { cached_tokens: 4 }, cost: 0.0000042 } }
        ])
      )
    )
    await chat.start(request(), 'phone')
    const done = last(events)
    expect(done.type).toBe('done')
    expect(done.type === 'done' && done.stats).toMatchObject({ modelName: 'x/one', promptTokens: 12, predictedTokens: 7, cachedTokens: 4, costMicro: 5, stopReason: 'stop' })
    expect(calls.length).toBeGreaterThan(0)
    expect(calls.every((c) => c.by === 'phone')).toBe(true)
    expect(calls.some((c) => c.path === '/v1/chat/completions')).toBe(true)
    // Counting directly asks as the caller says, too.
    calls.length = 0
    await chat.countText('openrouter/x/one', 'hello', 'c-x/one', 'api')
    await chat.countTokens('openrouter/x/one', [{ role: 'user', content: 'hi' }], null, 'c-x/one', 'api')
    expect(calls.length).toBeGreaterThan(0)
    expect(calls.every((c) => c.by === 'api')).toBe(true)
  })

  it('never reads the chat template of a cloud model (/props is not asked for tool-call arguments)', async () => {
    const { chat, events, calls } = service((call) => counting(call) ?? reply(sse([chunk({ content: 'ok' }, 'stop')])))
    await chat.start(
      request({
        messages: [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'f', arguments: '{}' } }] },
          { role: 'tool', content: 'r', tool_call_id: 'c1' }
        ]
      })
    )
    expect(last(events).type).toBe('done')
    expect(calls.some((c) => c.path === '/props')).toBe(false)
  })

  it('marks limit, key, credits and rate errors fatal, others not, and never as transport errors', async () => {
    for (const [code, fatal] of [
      ['cloud_limit', true],
      ['cloud_key', true],
      ['cloud_credits', true],
      ['cloud_rate', true],
      ['cloud_not_allowed', true],
      ['cloud_policy', false],
      ['cloud_error', false]
    ] as const) {
      const { chat, events } = service(
        (call) => counting(call) ?? new Response(JSON.stringify({ error: { message: `Sentence for ${code}`, type: 'x', code } }), { status: 402, headers: { 'Content-Type': 'application/json' } })
      )
      await chat.start(request())
      const e = last(events)
      expect(e).toMatchObject({ type: 'error', error: `Sentence for ${code}` })
      expect(e.type === 'error' && e.fatal).toBe(fatal ? true : undefined)
      expect(e.type === 'error' && e.transport).toBeUndefined()
    }
  })

  it('an error chunk in the stream ends it as an error with its code', async () => {
    const { chat, events } = service(
      (call) => counting(call) ?? reply(sse([chunk({ content: 'par' }), { error: { message: 'Daily limit reached', code: 'cloud_limit' } }], false))
    )
    await chat.start(request())
    expect(last(events)).toMatchObject({ type: 'error', error: 'Daily limit reached', fatal: true })
  })

  it('a stream that just ends is an error: broke off, not fatal, not transport', async () => {
    const { chat, events } = service((call) => counting(call) ?? reply(sse([chunk({ content: 'par' })], false)))
    await chat.start(request())
    const e = last(events)
    expect(e).toMatchObject({ type: 'error', error: 'The answer from OpenRouter broke off.' })
    expect(e.type === 'error' && e.fatal).toBeUndefined()
    expect(e.type === 'error' && e.transport).toBeUndefined()
  })

  it('records the round with the whole id as the model when asked to', async () => {
    const recorded: unknown[] = []
    const events: StreamEvent[] = []
    const inst = cloudInstance('x/one')
    const manager = {
      readyFor: () => inst,
      get: () => inst,
      acquire: () => () => undefined,
      fetch: async (_i: LoadedInstance, path: string) =>
        path === '/v1/chat/completions' ? reply(sse([chunk({ content: 'Hi' }, 'stop')])) : path === '/tokenize' ? Response.json({ tokens: [1] }) : Response.json({ prompt: 'p' })
    } as unknown as InstanceManagerType
    const chat = new ChatService(manager, (e) => events.push(e), undefined, (_id, round) => void recorded.push(round))
    await chat.start(request({ record: { conversationId: 'c1' } as never }))
    expect(recorded).toHaveLength(1)
    expect(recorded[0]).toMatchObject({ model: 'openrouter/x/one', response: { content: 'Hi', finishReason: 'stop' } })
  })

  it('leaves the stats of a llama-server alone: usage is only read for cloud instances', async () => {
    const { chat, events } = service(
      (call) => counting(call) ?? reply(sse([chunk({ content: 'Hi' }, 'stop'), { choices: [], usage: { prompt_tokens: 99, completion_tokens: 98, cost: 1 }, timings: { predicted_n: 3, prompt_n: 2, predicted_per_second: 30 } }])),
      { cloud: undefined, port: 5 }
    )
    await chat.start(request())
    const done = last(events)
    expect(done.type === 'done' && done.stats).toMatchObject({ predictedTokens: 3, promptTokens: 2, tokensPerSecond: 30 })
    expect(done.type === 'done' && done.stats.costMicro).toBeUndefined()
  })
})

describe('practice counts of an OpenRouter model', () => {
  it('runs, chats and recorded sessions count as one model, and never as a local model of that name', () => {
    const id = 'openrouter/qwen/qwen3-32b'
    const models = [
      { id, name: 'Qwen: Qwen3 32B' },
      { id: 'x/qwen3-32b.gguf', name: 'qwen3-32b' }
    ]
    expect(new Set([modelKey(id), keyOfModel(id, models), keyOfModel(id, [])]).size).toBe(1)
    expect(keyOfModel('x/qwen3-32b.gguf', models)).not.toBe(modelKey(id))
    expect(modelKey('openrouter/other/qwen3-32b')).not.toBe(modelKey(id))
  })
})
