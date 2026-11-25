// Models on vast.ai rentals: the instance layer, chat transport errors and what the API server refuses.
import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { mkdtempSync } from 'node:fs'
import http from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import type { RemoteHost } from '@shared/remote'
import type { InstanceErrorEvent, LoadedInstance, ModelEntry, StreamEvent } from '@shared/types'
import { ChatService } from '../src/main/chat/service'
import type { EngineService } from '../src/main/engine/engine'
import { InstanceManager, type InstanceManagerDeps } from '../src/main/engine/instances'
import type { RemoteLaunch, ServerExit, ServerHandle } from '../src/main/engine/serverHandle'
import { ApiServer } from '../src/main/server/apiServer'
import { modelsAdapter } from '../src/main/writing/adapters'

const hostA: RemoteHost = { kind: 'vast', rentalId: 1, label: 'rulz-llm-aaaa' }
const hostB: RemoteHost = { kind: 'vast', rentalId: 2, label: 'rulz-llm-bbbb' }

const entry = (id: string, extra: Partial<ModelEntry> = {}): ModelEntry => ({
  id,
  name: id,
  folder: '',
  path: `C:\\LLM\\${id}.gguf`,
  files: [],
  sizeBytes: 1e9,
  info: null,
  modifiedAt: 0,
  ...extra
})
const modelOf = (id: string): ModelEntry => entry(id, id.startsWith('vast/') ? { remote: true } : {})

/** A stand-in for the remote ServerHandle: the tests decide when it exits, reconnects or refuses to stop. */
class FakeHandle implements ServerHandle {
  readonly pid = null
  stopped = 0
  detached = 0
  isReconnecting = false
  stopError: Error | null = null
  private exitFns: ((e: ServerExit) => void)[] = []
  private exited: ServerExit | null = null
  constructor(
    readonly port: number,
    readonly host: RemoteHost
  ) {}
  onOutput(): void {}
  onExit(fn: (e: ServerExit) => void): void {
    if (this.exited) fn(this.exited)
    else this.exitFns.push(fn)
  }
  reconnecting(): boolean {
    return this.isReconnecting
  }
  async stop(): Promise<void> {
    this.stopped++
    if (this.stopError) throw this.stopError
    this.exit({ code: 0 })
  }
  detachSync(): void {
    this.detached++
  }
  exit(e: ServerExit): void {
    if (this.exited) return
    this.exited = e
    for (const fn of this.exitFns) fn(e)
  }
}

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
  errors: InstanceErrorEvent[]
  handles: FakeHandle[]
  starts: { rentalId: number; launch: RemoteLaunch }[]
  uses: { id: string; by: string; what: string; ended: boolean }[]
  /** Called with each start; return a promise to hold the start back. */
  hold: { next: Promise<void> | null }
  failStart: { error: Error | null }
}

function rig(overrides: Partial<InstanceManagerDeps> = {}): Rig {
  const r = { events: [], errors: [], handles: [], starts: [], uses: [], hold: { next: null }, failStart: { error: null } } as unknown as Rig
  r.mgr = new InstanceManager({
    engine: { resolveExe: async () => 'C:\\llama\\llama-server.exe' } as unknown as EngineService,
    getModel: (id) => modelOf(id),
    draftFor: () => undefined,
    getDefaults: () => DEFAULT_LOAD_CONFIG,
    unloadPreviousOnLoad: () => true,
    estimate: () => null,
    pidFile: join(mkdtempSync(join(tmpdir(), 'rulz-inst-remote-')), 'pids.json'),
    onChange: () => undefined,
    onError: (e) => r.errors.push(e),
    spawnServer: (_exe, args) => {
      const name = args[args.indexOf('--alias') + 1]
      r.events.push(`spawn ${name}`)
      return new FakeServer(name) as unknown as ChildProcess
    },
    killProcess: (proc, force) => {
      const p = proc as unknown as FakeServer
      r.events.push(`${force ? 'force-kill' : 'kill'} ${p.name}`)
      setTimeout(() => p.emit('exit', 0), 5)
    },
    launcher: {
      start: async (rentalId, launch) => {
        r.starts.push({ rentalId, launch })
        if (r.hold.next) await r.hold.next
        if (r.failStart.error) throw r.failStart.error
        const h = new FakeHandle(40000 + r.handles.length, rentalId === 1 ? hostA : hostB)
        r.handles.push(h)
        return h
      }
    },
    onUse: (instance, by, what) => {
      const use = { id: instance.instanceId, by, what, ended: false }
      r.uses.push(use)
      return () => void (use.ended = true)
    },
    reconnectPollMs: 10,
    ...overrides
  })
  return r
}

/** Every model answers /health and /props at once, so loads become ready. */
function readyFetch(): void {
  vi.stubGlobal('fetch', async (url: string) => {
    if (String(url).endsWith('/health')) return new Response('{}', { status: 200 })
    return new Response('{"default_generation_settings":{"n_ctx":4096}}', { status: 200 })
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('InstanceManager with rentals', () => {
  it('loads one model on two rentals as two instances, each reused on its own host', async () => {
    readyFetch()
    const r = rig()
    const a = await r.mgr.load('vast/m', null, hostA)
    const b = await r.mgr.load('vast/m', null, hostB)
    expect(a.instanceId).not.toBe(b.instanceId)
    expect(r.mgr.list().map((i) => i.host?.rentalId)).toEqual([1, 2])
    expect(r.mgr.list().map((i) => i.port)).toEqual([40000, 40001])
    expect(r.starts.map((s) => s.rentalId)).toEqual([1, 2])
    // Same config on the same host: the running one, not a third start.
    const again = await r.mgr.load('vast/m', null, hostA)
    expect(again.instanceId).toBe(a.instanceId)
    expect(r.starts).toHaveLength(2)
    // The launcher is told which model, adapters and instance it starts.
    expect(r.starts[0].launch).toMatchObject({ alias: 'vast/m', instanceId: a.instanceId })
  })

  it('"unload the previous model" only concerns the same host', async () => {
    readyFetch()
    const r = rig()
    await r.mgr.load('local-a', null)
    const m = await r.mgr.load('vast/m', null, hostA)
    // A local load does not touch the rental...
    await r.mgr.load('local-b', null)
    expect(r.mgr.list().map((i) => i.modelId).sort()).toEqual(['local-b', 'vast/m'])
    expect(r.handles[0].stopped).toBe(0)
    // ...a load on another rental does not either...
    await r.mgr.load('vast/n', null, hostB)
    expect(r.handles[0].stopped).toBe(0)
    // ...and a load on the same rental replaces the model there, and only there.
    await r.mgr.load('vast/o', null, hostA)
    expect(r.handles[0].stopped).toBe(1)
    expect(r.mgr.get(m.instanceId)).toBeUndefined()
    expect(r.mgr.list().map((i) => i.modelId).sort()).toEqual(['local-b', 'vast/n', 'vast/o'])
    expect(r.events).toContain('kill local-a')
    expect(r.events).not.toContain('kill local-b')
    await r.mgr.unloadAll()
  })

  it('refuses a request that cannot be placed before it unloads anything', async () => {
    readyFetch()
    // The model is unknown when the load is queued and known when it starts (a rescan in between).
    let known = false
    const r = rig({ getModel: (id) => (known ? modelOf(id) : undefined) })
    known = true
    await r.mgr.load('local-a', null)
    const loaded = r.mgr.list().length
    known = false
    const p = r.mgr.load('vast/m', null)
    known = true
    await expect(p).rejects.toThrow(/on a rental first/)
    expect(r.events).not.toContain('kill local-a')
    expect(r.mgr.list()).toHaveLength(loaded)
    // And a local model sent to a rental.
    known = true
    const rented = await r.mgr.load('vast/m', null, hostA)
    known = false
    const q = r.mgr.load('local-b', null, hostA)
    known = true
    await expect(q).rejects.toThrow(/runs on this PC/)
    expect(r.handles[0].stopped).toBe(0)
    expect(r.mgr.get(rented.instanceId)).toBeDefined()
    await r.mgr.unloadAll()
  })

  it('does not start or rent anything for a remote model without an instance', async () => {
    readyFetch()
    const r = rig()
    await expect(r.mgr.ensure('vast/m')).rejects.toThrow(/Load vast\/m on a rental first \(vast\.ai page\)\./)
    await expect(r.mgr.load('vast/m', null)).rejects.toThrow(/on a rental first/)
    await expect(r.mgr.load('local-a', null, hostA)).rejects.toThrow(/runs on this PC/)
    expect(r.starts).toEqual([])
    expect(r.events).toEqual([])
    // Once it runs somewhere, ensure uses it, and a load without a host reloads it on its own host.
    const a = await r.mgr.load('vast/m', null, hostB)
    const ensured = await r.mgr.ensure('vast/m')
    expect(ensured).toMatchObject({ started: false, instance: { instanceId: a.instanceId } })
    const reloaded = await r.mgr.load('vast/m', { ...DEFAULT_LOAD_CONFIG, contextLength: 2048 })
    expect(reloaded.host?.rentalId).toBe(2)
    expect(reloaded.instanceId).not.toBe(a.instanceId)
    expect(r.starts.map((s) => s.rentalId)).toEqual([2, 2])
    await r.mgr.unloadAll()
  })

  it('a remote start does not hold the local load queue, and can be cancelled while it starts', async () => {
    readyFetch()
    const r = rig()
    let release!: () => void
    r.hold.next = new Promise<void>((res) => (release = res))
    const remote = r.mgr.load('vast/m', null, hostA)
    remote.catch(() => undefined)
    await vi.waitFor(() => expect(r.mgr.list()).toHaveLength(1))
    expect(r.mgr.list()[0]).toMatchObject({ status: 'loading', host: hostA })
    // The local queue is free.
    const local = await r.mgr.load('local-a', null)
    expect(local.status).toBe('ready')
    // Unloading the starting instance cancels the load at once, without waiting for the rental.
    await r.mgr.unload(r.mgr.list().find((i) => i.host)!.instanceId)
    await expect(remote).rejects.toThrow(/cancelled/)
    expect(r.mgr.list().map((i) => i.modelId)).toEqual(['local-a'])
    // The server that comes up afterwards is stopped, not left running.
    release()
    await vi.waitFor(() => expect(r.handles[0]?.stopped).toBe(1))
    await r.mgr.unloadAll()
  })

  it('a rental that cannot start the model answers with its sentence and leaves nothing listed', async () => {
    readyFetch()
    const r = rig()
    r.failStart.error = new Error('The rental is stopped. Start it on the vast.ai page.')
    await expect(r.mgr.load('vast/m', null, hostA)).rejects.toThrow('The rental is stopped. Start it on the vast.ai page.')
    expect(r.mgr.list()).toEqual([])
    expect(r.errors).toEqual([])
  })

  it('keeps an instance listed, ready again, when stop() fails, and reports the error', async () => {
    readyFetch()
    const r = rig()
    const a = await r.mgr.load('vast/m', null, hostA)
    r.handles[0].stopError = new Error('The connection to the instance is down; it may still be running.')
    await expect(r.mgr.unload(a.instanceId)).rejects.toThrow(/may still be running/)
    expect(r.mgr.list()).toHaveLength(1)
    expect(r.mgr.list()[0]).toMatchObject({ instanceId: a.instanceId, status: 'ready', stage: 'Ready' })
    expect(r.mgr.readyFor('vast/m')?.instanceId).toBe(a.instanceId)
    // A later unload works once the connection is back.
    r.handles[0].stopError = null
    await r.mgr.unload(a.instanceId)
    expect(r.mgr.list()).toEqual([])
    // Loading something else on that rental fails the same way while the first one cannot be stopped.
    const b = await r.mgr.load('vast/m', null, hostA)
    r.handles[1].stopError = new Error('stop failed')
    await expect(r.mgr.load('vast/n', null, hostA)).rejects.toThrow('stop failed')
    expect(r.mgr.get(b.instanceId)?.status).toBe('ready')
    r.handles[1].stopError = null
    await r.mgr.unloadAll()
  })

  it('shows reconnecting while the tunnel is being re-made and drops the flag when it is back', async () => {
    readyFetch()
    const r = rig()
    await r.mgr.load('vast/m', null, hostA)
    const id = r.mgr.list()[0].instanceId
    expect(r.mgr.get(id)?.reconnecting).toBeUndefined()
    r.handles[0].isReconnecting = true
    await vi.waitFor(() => expect(r.mgr.list()[0].reconnecting).toBe(true))
    r.handles[0].isReconnecting = false
    await vi.waitFor(() => expect(r.mgr.list()[0].reconnecting).toBeUndefined())
    await r.mgr.unloadAll()
  })

  it('readyFor prefers the pinned instance, then one that is not reconnecting, then the fewest users, then the newest', async () => {
    readyFetch()
    const r = rig({ unloadPreviousOnLoad: () => false })
    const a = await r.mgr.load('vast/m', null, hostA)
    await new Promise((res) => setTimeout(res, 3))
    const b = await r.mgr.load('vast/m', null, hostB)
    expect(r.mgr.readyFor('vast/m')?.instanceId).toBe(b.instanceId)
    expect(r.mgr.readyFor('vast/m', a.instanceId)?.instanceId).toBe(a.instanceId)
    expect(r.mgr.readyFor('vast/m', 'gone')?.instanceId).toBe(b.instanceId)
    // The newest has a user: the idle one is chosen.
    const release = r.mgr.acquire(b.instanceId)
    expect(r.mgr.readyFor('vast/m')?.instanceId).toBe(a.instanceId)
    release()
    // A reconnecting one comes last whatever its users.
    r.handles[1].isReconnecting = true
    await vi.waitFor(() => expect(r.mgr.get(b.instanceId)?.reconnecting).toBe(true))
    expect(r.mgr.readyFor('vast/m')?.instanceId).toBe(a.instanceId)
    expect(r.mgr.readyFor('nothing')).toBeUndefined()
    await r.mgr.unloadAll()
  })

  it('tells the rental side who uses an instance, and when the use ends', async () => {
    readyFetch()
    const r = rig()
    const a = await r.mgr.load('vast/m', null, hostA)
    const end = r.mgr.acquire(a.instanceId, 'api', 'api request')
    expect(r.uses).toEqual([{ id: a.instanceId, by: 'api', what: 'api request', ended: false }])
    end()
    end()
    expect(r.uses[0].ended).toBe(true)
    r.mgr.acquire(a.instanceId)()
    expect(r.uses[1]).toMatchObject({ by: 'window', what: 'chat', ended: true })
    await r.mgr.unloadAll()
  })

  it('unloadAllLocal leaves the rentals alone; unloadAll takes everything', async () => {
    readyFetch()
    const r = rig({ unloadPreviousOnLoad: () => false })
    await r.mgr.load('local-a', null)
    await r.mgr.load('vast/m', null, hostA)
    await r.mgr.unloadAllLocal()
    expect(r.mgr.list().map((i) => i.modelId)).toEqual(['vast/m'])
    expect(r.handles[0].stopped).toBe(0)
    await r.mgr.unloadAll()
    expect(r.mgr.list()).toEqual([])
    expect(r.handles[0].stopped).toBe(1)
  })

  it('an exit with a reason puts that sentence in the failure and in the error event', async () => {
    readyFetch()
    const r = rig()
    await r.mgr.load('vast/m', null, hostA)
    r.handles[0].exit({ code: null, reason: 'The connection to the instance was lost and did not come back.' })
    expect(r.errors).toHaveLength(1)
    expect(r.errors[0].error).toBe('The connection to the instance was lost and did not come back.')
    expect(r.mgr.list()).toEqual([])

    // While loading: the load fails with the sentence.
    vi.stubGlobal('fetch', async () => new Response('', { status: 503 }))
    const loading = r.mgr.load('vast/m', null, hostA)
    loading.catch(() => undefined)
    await vi.waitFor(() => expect(r.handles).toHaveLength(2))
    await vi.waitFor(() => expect(r.mgr.list()[0]?.port).toBe(40001))
    r.handles[1].exit({ code: 1, reason: 'The server on the rental stopped while loading.' })
    await expect(loading).rejects.toThrow('The server on the rental stopped while loading.')
    expect(r.errors).toHaveLength(1)
  })

  it('detaches rentals and kills local servers when the app quits, and starts nothing afterwards', async () => {
    readyFetch()
    const r = rig({ unloadPreviousOnLoad: () => false })
    await r.mgr.load('local-a', null)
    await r.mgr.load('vast/m', null, hostA)
    // A start still in progress: its server is stopped when it arrives.
    let release!: () => void
    r.hold.next = new Promise<void>((res) => (release = res))
    const starting = r.mgr.load('vast/n', null, hostB)
    starting.catch(() => undefined)
    await vi.waitFor(() => expect(r.starts).toHaveLength(2))
    r.mgr.killAllSync()
    expect(r.handles[0].detached).toBe(1)
    expect(r.handles[0].stopped).toBe(0)
    expect(r.mgr.list()).toEqual([])
    await expect(starting).rejects.toThrow(/cancelled/)
    await expect(r.mgr.load('vast/m', null, hostA)).rejects.toThrow(/closing/)
    await expect(r.mgr.load('local-b', null)).rejects.toThrow(/closing/)
    release()
    await vi.waitFor(() => expect(r.handles[1]?.stopped).toBe(1))
  })

  it('stops the reconnect check when no rental instance is left', async () => {
    readyFetch()
    const r = rig()
    await r.mgr.load('vast/m', null, hostA)
    const timer = (): unknown => (r.mgr as unknown as { reconnectTimer: unknown }).reconnectTimer
    expect(timer()).not.toBeNull()
    await r.mgr.unloadAll()
    expect(timer()).toBeNull()
  })
})

describe('ChatService on a rental', () => {
  const remote = (extra: Partial<LoadedInstance> = {}): LoadedInstance =>
    ({
      instanceId: 'r1',
      modelId: 'vast/m',
      modelName: 'm',
      port: 9,
      pid: null,
      status: 'ready',
      progress: 1,
      stage: 'Ready',
      config: DEFAULT_LOAD_CONFIG,
      startedAt: 0,
      contextLength: null,
      hasVision: false,
      adapters: [],
      estimate: null,
      host: hostA,
      ...extra
    }) as LoadedInstance

  function service(instances: LoadedInstance[]): { chat: ChatService; events: StreamEvent[]; readyFor: ReturnType<typeof vi.fn>; acquire: ReturnType<typeof vi.fn> } {
    const events: StreamEvent[] = []
    const readyFor = vi.fn((modelId: string, prefer?: string) => {
      const ready = instances.filter((i) => i.modelId === modelId && i.status === 'ready')
      return ready.find((i) => i.instanceId === prefer) ?? ready[0]
    })
    const acquire = vi.fn(() => () => undefined)
    const manager = {
      readyFor,
      get: (id: string) => instances.find((i) => i.instanceId === id),
      acquire,
      baseUrl: () => 'http://127.0.0.1:9',
      // As InstanceManager.fetch does for a rental: the global fetch, which the tests stub.
      fetch: (_i: LoadedInstance, path: string, init?: RequestInit) => fetch(`http://127.0.0.1:9${path}`, init)
    } as unknown as InstanceManager
    return { chat: new ChatService(manager, (e) => events.push(e)), events, readyFor, acquire }
  }

  const request = (extra: Partial<Parameters<ChatService['start']>[0]> = {}): Parameters<ChatService['start']>[0] => ({
    requestId: 'q1',
    modelId: 'vast/m',
    messages: [{ role: 'user', content: 'hi' }],
    sampling: DEFAULT_SAMPLING,
    overflow: 'stop',
    ...extra
  })

  const sse = (lines: string[]): Response => new Response(lines.join(''), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
  const chunk = (delta: object, finish?: string): string => `data: ${JSON.stringify({ choices: [{ delta, finish_reason: finish ?? null }] })}\n\n`
  const last = (events: StreamEvent[]): StreamEvent => events[events.length - 1]

  it('marks a failed fetch as a transport error, with a sentence naming the rental', async () => {
    const { chat, events } = service([remote()])
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('fetch failed', { cause: Object.assign(new Error('read ECONNRESET'), { code: 'ECONNRESET' }) })
    })
    await chat.start(request())
    const e = last(events)
    expect(e).toMatchObject({ type: 'error', transport: true })
    expect((e as { error: string }).error).toContain('rulz-llm-aaaa')
  })

  it('treats a stream that ends with no finish reason and no [DONE] as a transport error, but not on this PC', async () => {
    const cut = (): Response => sse([chunk({ content: 'Hel' })])
    vi.stubGlobal('fetch', async () => cut())
    const rental = service([remote()])
    await rental.chat.start(request())
    expect(last(rental.events)).toMatchObject({ type: 'error', transport: true })

    const local = service([remote({ host: undefined, modelId: 'vast/m' })])
    await local.chat.start(request())
    expect(last(local.events).type).toBe('done')
  })

  it('a finished stream is a done event; a model or HTTP error is not a transport error', async () => {
    const { chat, events } = service([remote()])
    vi.stubGlobal('fetch', async () => sse([chunk({ content: 'Hi' }), chunk({}, 'stop'), 'data: [DONE]\n\n']))
    await chat.start(request())
    expect(last(events).type).toBe('done')

    vi.stubGlobal('fetch', async () => new Response('{"error":{"message":"context too long"}}', { status: 400 }))
    await chat.start(request({ requestId: 'q2' }))
    const e = last(events)
    expect(e).toMatchObject({ type: 'error', requestId: 'q2' })
    expect(e).not.toHaveProperty('transport')
  })

  it('a pinned instance is used, and an error that says it is gone is a transport error', async () => {
    const a = remote({ instanceId: 'a' })
    const b = remote({ instanceId: 'b', host: hostB })
    const { chat, events, acquire } = service([a, b])
    vi.stubGlobal('fetch', async () => sse([chunk({ content: 'x' }), chunk({}, 'stop')]))
    await chat.start(request({ instanceId: 'b' }), 'phone')
    expect(acquire).toHaveBeenCalledWith('b', 'phone', 'chat')
    expect(last(events).type).toBe('done')

    await chat.start(request({ requestId: 'q2', instanceId: 'gone' }))
    expect(last(events)).toMatchObject({ type: 'error', requestId: 'q2', transport: true })
  })

  it('keeps the tool rounds of one session on the instance the first one used', async () => {
    const a = remote({ instanceId: 'a' })
    const b = remote({ instanceId: 'b', host: hostB })
    const { chat, readyFor } = service([a, b])
    vi.stubGlobal('fetch', async () => sse([chunk({ content: 'x' }), chunk({}, 'stop')]))
    await chat.start(request({ sessionKey: 'chat-1' }))
    // The manager would now pick `a` first; make the session's remembered choice differ from it.
    readyFor.mockImplementation((_modelId: string, prefer?: string) => [b, a].find((i) => i.instanceId === prefer) ?? b)
    await chat.start(request({ requestId: 'q2', sessionKey: 'chat-1' }))
    await chat.start(request({ requestId: 'q3', sessionKey: 'chat-1' }))
    const preferred = readyFor.mock.calls.map((c) => c[1])
    expect(preferred[0]).toBeUndefined()
    // The second round asked for `a` (the first one's choice), and the third kept on whatever the second used.
    expect(preferred[1]).toBe('a')
    expect(preferred[2]).toBe('a')
  })

  it('counts tokens on the pinned instance only', async () => {
    const { chat } = service([remote({ instanceId: 'a' })])
    expect(await chat.countText('vast/m', 'hello', 'gone')).toBeNull()
    expect(await chat.countTokens('other', [], null, 'a')).toBeNull()
  })
})

describe('the API server and a model on a rental', () => {
  const freePort = async (): Promise<number> => {
    const s = http.createServer()
    await new Promise<void>((r) => s.listen(0, '127.0.0.1', () => r()))
    const port = (s.address() as { port: number }).port
    await new Promise<void>((r) => s.close(() => r()))
    return port
  }
  const models = [entry('Local-Model'), entry('vast/big', { name: 'Big Remote', remote: true })]

  it('lists a rental model only while it is loaded, and refuses to load, unload or reconfigure it', async () => {
    const calls: string[] = []
    const instances: LoadedInstance[] = []
    const manager = {
      list: () => instances,
      readyFor: (id: string) => instances.find((i) => i.modelId === id && i.status === 'ready'),
      ensure: async (id: string) => void calls.push(`ensure ${id}`),
      load: async (id: string) => void calls.push(`load ${id}`),
      unload: async (id: string) => void calls.push(`unload ${id}`),
      acquire: () => () => undefined,
      idleSince: () => 0
    } as unknown as InstanceManager
    const port = await freePort()
    const server = new ApiServer({
      instances: manager,
      getModels: () => models,
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: false }),
      getLoadDefaults: () => ({ config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async () => void calls.push('setLoadDefaults'),
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: () => undefined
    })
    await server.start()
    const url = (p: string): string => `http://127.0.0.1:${port}${p}`
    const ids = async (): Promise<string[]> => ((await (await fetch(url('/v1/models'))).json()) as { data: { id: string }[] }).data.map((m) => m.id)
    try {
      expect(await ids()).toEqual(['Local-Model'])
      expect((await fetch(url('/v1/models/vast%2Fbig'))).status).toBe(404)

      // Not loaded: a request names it and is refused, with nothing started.
      const chat = (model: string): Promise<Response> => fetch(url('/v1/chat/completions'), { method: 'POST', body: JSON.stringify({ model, messages: [] }) })
      const refused = await chat('Big Remote')
      expect(refused.status).toBe(409)
      expect(((await refused.json()) as { error: { message: string } }).error.message).toMatch(/vast\.ai/)
      expect((await fetch(url('/v1/models/Big%20Remote/unload'), { method: 'POST' })).status).toBe(409)

      instances.push({ instanceId: 'r1', modelId: 'vast/big', modelName: 'Big Remote', status: 'ready', port: 1, config: DEFAULT_LOAD_CONFIG, contextLength: 4096, host: hostA } as LoadedInstance)
      expect(await ids()).toEqual(['Local-Model', 'Big Remote'])
      const unload = await fetch(url('/v1/models/Big%20Remote/unload'), { method: 'POST' })
      expect(unload.status).toBe(409)
      expect(((await unload.json()) as { error: { message: string } }).error.message).toMatch(/cannot unload it/)
      const config = await fetch(url('/v1/models/Big%20Remote/load-config'), { method: 'POST', body: JSON.stringify({ config: { contextLength: 8192 } }) })
      expect(config.status).toBe(409)
      const read = (await (await fetch(url('/v1/models/Big%20Remote/load-config'))).json()) as { estimate: unknown; gpus: unknown[] }
      expect(read.estimate).toBeNull()
      expect(read.gpus).toEqual([])
      expect(calls).toEqual([])
    } finally {
      await server.stop()
    }
  })

  it('uses a loaded rental model as it is (counted as an api use) and answers 502 when the tunnel is down', async () => {
    const calls: string[] = []
    const upstream = http.createServer((_req, res) => res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"ok":true}'))
    await new Promise<void>((r) => upstream.listen(0, '127.0.0.1', () => r()))
    const upPort = (upstream.address() as { port: number }).port
    const inst = { instanceId: 'r1', modelId: 'vast/big', modelName: 'Big Remote', status: 'ready', port: upPort, config: DEFAULT_LOAD_CONFIG, contextLength: 4096, host: hostA } as LoadedInstance
    const manager = {
      list: () => [inst],
      readyFor: (id: string) => (id === 'vast/big' ? inst : undefined),
      ensure: async () => {
        throw new Error('no')
      },
      acquire: (id: string, by?: string, what?: string) => {
        calls.push(`acquire ${id} ${by} ${what}`)
        return () => undefined
      },
      idleSince: () => 0
    } as unknown as InstanceManager
    const port = await freePort()
    const server = new ApiServer({
      instances: manager,
      getModels: () => models,
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: false }),
      getLoadDefaults: () => ({ config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async () => undefined,
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: () => undefined
    })
    await server.start()
    try {
      const ask = (): Promise<Response> => fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: 'POST', body: JSON.stringify({ model: 'Big Remote', messages: [] }) })
      const ok = await ask()
      expect(ok.status).toBe(200)
      expect(calls).toEqual(['acquire r1 api api request'])
      // The tunnel drops: nothing listens on the instance's port any more.
      await new Promise<void>((r) => upstream.close(() => r()))
      const down = await ask()
      expect(down.status).toBe(502)
      expect(((await down.json()) as { error: { message: string } }).error.message).toMatch(/connection to the rental rulz-llm-aaaa was lost/)
    } finally {
      await server.stop()
      upstream.closeAllConnections?.()
    }
  })
})

describe('modelsAdapter on a rental', () => {
  it('only returns a remote model that is already ready, and says so otherwise', async () => {
    const ready = { instanceId: 'r1', modelId: 'vast/big', status: 'ready', contextLength: 4096 } as LoadedInstance
    let loaded: LoadedInstance | undefined
    const ensure = vi.fn(async () => ({ instance: ready, started: true }))
    const instances = { readyFor: () => loaded, list: () => [], ensure } as unknown as InstanceManager
    const adapter = modelsAdapter({ list: () => [entry('vast/big', { name: 'Big Remote', remote: true }), entry('Local')] }, instances)
    await expect(adapter.ensure('vast/big')).rejects.toThrow(/Load Big Remote on a rental first/)
    loaded = ready
    expect(await adapter.ensure('vast/big')).toMatchObject({ modelId: 'vast/big', instanceId: 'r1' })
    expect(ensure).not.toHaveBeenCalled()
    await adapter.ensure('Local')
    expect(ensure).toHaveBeenCalledWith('Local', undefined)
    await adapter.ensure('Local', 'writing')
    expect(ensure).toHaveBeenLastCalledWith('Local', 'writing')
  })
})
