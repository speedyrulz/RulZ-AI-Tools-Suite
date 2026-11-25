// The API's mode routes, the X-RulZ-Mode / X-RulZ-Rental headers, reply headers, leases and the mode reload
// (docs/app-modes-design.md, section 1.2), against a fake instance manager and a fake model server.
import { mkdtempSync, rmSync } from 'node:fs'
import http from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import type { ApiRequestLog, LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import type { InstanceManager } from '../src/main/engine/instances'
import { ModelGate } from '../src/main/modelGate'
import { ApiServer } from '../src/main/server/apiServer'
import { SettingsStore } from '../src/main/settings/store'
import { buildOverview } from '../src/main/re/overview'

const model = (id: string, name: string, extra: Partial<ModelEntry> = {}): ModelEntry => ({
  id,
  name,
  folder: id.split('/').slice(0, -1).join('/'),
  path: `C:\\LLM\\${id.replace(/\//g, '\\')}`,
  files: [],
  sizeBytes: 1,
  info: null,
  modifiedAt: 0,
  ...extra
})
const LOCAL = model('Qwen3.5/Q8/Qwen3.5-9B-Q8_0.gguf', 'Qwen3.5-9B-Q8_0')
const SLASHED = model('folder/model.gguf', 'model')
const RENTED = model('vast/x', 'Rented-70B', { remote: true })
const UMLAUT = model('Mod/modele.gguf', 'Modèle')
const MODELS = [LOCAL, SLASHED, RENTED, UMLAUT]
const HOST = { kind: 'vast' as const, rentalId: 1, label: 'RTX 4090' }

let seq = 0
const inst = (modelId: string, port: number, over: Partial<LoadedInstance> = {}): LoadedInstance => ({
  instanceId: `i-${++seq}`,
  modelId,
  modelName: MODELS.find((m) => m.id === modelId)?.name ?? modelId,
  port,
  pid: 1,
  status: 'ready',
  progress: 1,
  stage: '',
  config: DEFAULT_LOAD_CONFIG,
  startedAt: 1,
  contextLength: 8192,
  hasVision: false,
  estimate: null,
  adapters: [],
  ...over
})

/** The parts of InstanceManager the API server touches. A load waits until the test finishes or fails it. */
class FakeInstances {
  insts: LoadedInstance[] = []
  ensureCalls: { modelId: string; mode?: string }[] = []
  loadCalls: { modelId: string; config: LoadConfig | null }[] = []
  /** What a load asks again when its turn comes. */
  refuse: (() => string | null) | undefined
  held = new Map<string, number>()
  pendingLoads: { finish: (i: LoadedInstance) => void; fail: (e: Error) => void }[] = []
  /** What an instance listed by list() looks like to the freshness check; set per instance id. */
  listOverride = new Map<string, Partial<LoadedInstance>>()
  constructor(private readonly port: number) {}
  list(): LoadedInstance[] {
    return this.insts.map((i) => ({ ...i, ...this.listOverride.get(i.instanceId) }))
  }
  readyFor(modelId: string): LoadedInstance | undefined {
    return this.insts.find((i) => i.modelId === modelId && i.status === 'ready')
  }
  acquire(id: string): () => void {
    const modelId = this.insts.find((i) => i.instanceId === id)?.modelId ?? id
    this.held.set(modelId, (this.held.get(modelId) ?? 0) + 1)
    return () => this.held.set(modelId, (this.held.get(modelId) ?? 1) - 1)
  }
  usersOf(): number {
    return 0
  }
  idleSince(): number | null {
    return null
  }
  async ensure(modelId: string, mode?: string): Promise<{ instance: LoadedInstance; started: boolean }> {
    this.ensureCalls.push({ modelId, mode })
    const existing = this.insts.find((i) => i.modelId === modelId)
    if (existing) return { instance: existing, started: false }
    const created = inst(modelId, this.port)
    this.insts.push(created)
    return { instance: created, started: true }
  }
  load(modelId: string, config: LoadConfig | null, _host?: unknown, refuse?: () => string | null): Promise<LoadedInstance> {
    this.refuse = refuse
    this.loadCalls.push({ modelId, config })
    return new Promise((resolve, reject) => {
      this.pendingLoads.push({
        finish: (i) => {
          this.insts = this.insts.filter((x) => x.modelId !== modelId).concat(i)
          resolve(i)
        },
        fail: reject
      })
    })
  }
}

interface Res {
  status: number
  headers: http.IncomingHttpHeaders
  text: string
  json: Record<string, any>
}

function call(port: number, method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<Res> {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? undefined : typeof body === 'string' ? body : JSON.stringify(body)
    const req = http.request({ host: '127.0.0.1', port, method, path, headers: { ...(data !== undefined ? { 'Content-Type': 'application/json' } : {}), ...headers } }, (res) => {
      const chunks: Buffer[] = []
      res.on('data', (c: Buffer) => chunks.push(c))
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8')
        let json: Record<string, any> = {}
        try {
          json = JSON.parse(text)
        } catch {
          // not JSON
        }
        resolve({ status: res.statusCode ?? 0, headers: res.headers, text, json })
      })
    })
    req.on('error', reject)
    req.end(data)
  })
}

const cleanups: (() => Promise<void> | void)[] = []
afterEach(async () => {
  for (const c of cleanups.splice(0).reverse()) await c()
})

const modeEntry = (over: Partial<{ load: LoadConfig | null; sampling: typeof DEFAULT_SAMPLING | null; systemPrompt: string }> = {}) => ({ load: null, sampling: null, systemPrompt: '', ...over })
const CODING_LOAD: LoadConfig = { ...DEFAULT_LOAD_CONFIG, contextLength: 65536, flashAttention: 'off' }

async function setup(opts: { fallback?: boolean } = {}) {
  const seen: { method: string; path: string; headers: http.IncomingHttpHeaders }[] = []
  const upstream = http.createServer((req, res) => {
    seen.push({ method: req.method ?? '', path: req.url ?? '', headers: req.headers })
    req.resume()
    req.on('end', () => res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"ok":true}'))
  })
  await new Promise<void>((r) => upstream.listen(0, '127.0.0.1', () => r()))
  const upPort = (upstream.address() as { port: number }).port
  const dir = mkdtempSync(join(tmpdir(), 'rulz-api-modes-'))
  const store = new SettingsStore(dir)
  await store.init()
  const fake = new FakeInstances(upPort)
  const gate = new ModelGate()
  const state = { block: null as string | null, changed: 0, now: 1_000_000, overviewModel: [] as (string | undefined)[] }
  const log: ApiRequestLog[] = []
  const probe = http.createServer()
  await new Promise<void>((r) => probe.listen(0, '127.0.0.1', () => r()))
  const port = (probe.address() as { port: number }).port
  await new Promise<void>((r) => probe.close(() => r()))
  const server = new ApiServer({
    instances: fake as unknown as InstanceManager,
    getModels: () => MODELS,
    getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: true, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: opts.fallback ?? true }),
    getLoadDefaults: (id) => store.getModelDefaults(id),
    setLoadDefaults: (id, c) => store.setModelDefaults(id, c),
    estimate: () => null,
    gpus: () => [],
    onStatus: () => undefined,
    onRequest: (e) => log.push(e),
    now: () => state.now,
    modes: {
      get: (id) => store.modeDefaults(id),
      setLoad: (id, mode, load) => store.setModeLoad(id, mode, load),
      changed: () => void state.changed++,
      // The app's own check asks the gate too.
      reloadBlock: (_id, holding) => state.block ?? (gate.heldBy() !== null && gate.heldBy() === holding ? null : gate.busyReason()),
      takeGate: (holder) => gate.take(holder)
    },
    re: {
      overview: async (m?: string) => {
        state.overviewModel.push(m)
        return buildOverview({ chats: [], traces: [], practice: { kitDir: 'k', tasks: [] }, recording: false, model: m })
      }
    } as never
  })
  await server.start()
  cleanups.push(async () => {
    await server.stop()
    upstream.closeAllConnections()
    await new Promise<void>((r) => upstream.close(() => r()))
    rmSync(dir, { recursive: true, force: true })
  })
  return { server, store, fake, gate, state, seen, port, upPort, log }
}

const enc = encodeURIComponent
const chatBody = { model: 'Qwen3.5-9B-Q8_0', messages: [{ role: 'user', content: 'hi' }] }

describe('GET /v1/models/{id}/modes', () => {
  it('shows only coding and re, with the sampling built by the chat service and never another mode\'s prompt', async () => {
    const t = await setup()
    await t.store.setModeDefaults(LOCAL.id, 'coding', modeEntry({ load: CODING_LOAD, sampling: { ...DEFAULT_SAMPLING, temperature: 0.6, topK: 20, thinking: 'on' }, systemPrompt: '  Use tools.  ' }))
    await t.store.setModeDefaults(LOCAL.id, 'chat', modeEntry({ systemPrompt: 'SECRET-CHAT' }))
    await t.store.setModeDefaults(LOCAL.id, 'writing', modeEntry({ systemPrompt: 'SECRET-WRITING' }))
    await t.store.setModeDefaults(LOCAL.id, 'adventure', modeEntry({ systemPrompt: 'SECRET-ADVENTURE' }))
    const r = await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`)
    expect(r.status).toBe(200)
    expect(r.json).toMatchObject({ object: 'model.modes', id: 'Qwen3.5-9B-Q8_0', state: 'not-loaded', instance: null, api_override: null })
    expect(Object.keys(r.json.modes)).toEqual(['coding', 're'])
    expect(r.json.modes.coding).toMatchObject({ own_load: true, system_prompt: 'Use tools.', matches: null })
    expect(r.json.modes.coding.sampling).toMatchObject({ temperature: 0.6, top_k: 20 })
    expect(r.json.modes.coding.sampling).not.toHaveProperty('max_tokens')
    expect(r.json.modes.coding.sampling).not.toHaveProperty('seed')
    expect(r.json.modes.coding.thinking).toEqual({ chat_template_kwargs: { enable_thinking: true } })
    expect(r.json.modes.re).toEqual({ own_load: false, sampling: null, thinking: null, system_prompt: '', matches: null })
    expect(r.text).not.toContain('SECRET')
  })

  it('says whether the running settings are what each mode loads with, and the API-server override', async () => {
    const t = await setup()
    await t.store.setModeDefaults(LOCAL.id, 'coding', modeEntry({ load: CODING_LOAD }))
    await t.store.setModelDefaults(LOCAL.id, { ...DEFAULT_LOAD_CONFIG, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 1, override: true } })
    const loaded = inst(LOCAL.id, t.upPort, { config: { ...DEFAULT_LOAD_CONFIG, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 1, override: true } } })
    t.fake.insts = [loaded]
    const r = await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`)
    expect(r.json).toMatchObject({ state: 'ready', instance: loaded.instanceId, api_override: { temperature: 1 } })
    expect(r.json.modes.coding.matches).toBe(false)
    // A mode without load settings, for a model that runs with the default ones.
    expect(r.json.modes.re.matches).toBe(true)
    t.fake.insts = [inst(LOCAL.id, t.upPort, { config: CODING_LOAD })]
    const again = await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`)
    expect(again.json.modes.coding.matches).toBe(true)
  })

  it('shows loading while the model loads', async () => {
    const t = await setup()
    t.fake.insts = [inst(LOCAL.id, t.upPort, { status: 'loading' })]
    const r = await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`)
    expect(r.json).toMatchObject({ state: 'loading', instance: t.fake.insts[0].instanceId })
  })

  it('works with ids that contain slashes, encoded or not, and for a rental\'s model while it is loaded', async () => {
    const t = await setup()
    t.fake.insts = [inst(RENTED.id, t.upPort, { host: HOST })]
    expect((await call(t.port, 'GET', `/v1/models/${enc('folder/model.gguf')}/modes`)).status).toBe(200)
    expect((await call(t.port, 'GET', '/v1/models/folder/model.gguf/modes')).json.id).toBe('model')
    const rented = await call(t.port, 'GET', `/v1/models/${enc('vast/x')}/modes`)
    expect(rented.status).toBe(200)
    expect(rented.json.id).toBe('Rented-70B')
    expect((await call(t.port, 'GET', `/v1/models/${enc('vast/x')}/modes/coding/load-config`)).status).toBe(200)
    expect((await call(t.port, 'GET', `/v1/models/${enc('nope/none')}/modes`)).status).toBe(404)
  })

  it('is refused from a web page, and a rental\'s model that is not loaded is not found', async () => {
    const t = await setup()
    const origin = { Origin: 'https://example.com' }
    expect((await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`, undefined, { Origin: '' })).status).toBe(403)
    expect((await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`, undefined, origin)).status).toBe(403)
    expect((await call(t.port, 'POST', `/v1/models/${enc(LOCAL.id)}/reload`, { mode: 'coding' }, origin)).status).toBe(403)
    expect((await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes/coding/load-config`, undefined, origin)).status).toBe(403)
    expect((await call(t.port, 'POST', `/v1/models/${enc(LOCAL.id)}/modes/coding/load-config`, { config: {} }, origin)).status).toBe(403)
    expect((await call(t.port, 'GET', `/v1/models/${enc(RENTED.id)}/modes`)).status).toBe(404)
  })
})

describe('GET and POST /v1/models/{id}/modes/{mode}/load-config', () => {
  const path = (id: string, mode: string): string => `/v1/models/${enc(id)}/modes/${mode}/load-config`

  it('reads the effective settings of a mode', async () => {
    const t = await setup()
    await t.store.setModeDefaults(LOCAL.id, 'coding', modeEntry({ load: CODING_LOAD }))
    const coding = await call(t.port, 'GET', path(LOCAL.id, 'coding'))
    expect(coding.json).toMatchObject({ object: 'model.load_config', mode: 'coding', mode_own: true, custom: false })
    expect(coding.json.config).toMatchObject({ contextLength: 65536, flashAttention: 'off' })
    const re = await call(t.port, 'GET', path(LOCAL.id, 're'))
    expect(re.json).toMatchObject({ mode: 're', mode_own: false })
    expect(re.json.config.contextLength).toBe(8192)
  })

  it('stores only that mode\'s load and keeps its sampling and prompt, building on the effective settings', async () => {
    const t = await setup()
    await t.store.setModeDefaults(LOCAL.id, 're', modeEntry({ sampling: { ...DEFAULT_SAMPLING, temperature: 0.2 }, systemPrompt: 'be exact' }))
    await t.store.setModeDefaults(LOCAL.id, 'coding', modeEntry({ load: CODING_LOAD }))
    const before = t.store.getModelDefaults(LOCAL.id)
    const r = await call(t.port, 'POST', path(LOCAL.id, 're'), { config: { contextLength: 32768 } })
    expect(r.status).toBe(200)
    expect(r.json).toMatchObject({ mode: 're', mode_own: true })
    expect(r.json.config.contextLength).toBe(32768)
    const modes = t.store.modeDefaults(LOCAL.id)
    expect(modes.re?.load?.contextLength).toBe(32768)
    expect(modes.re?.sampling?.temperature).toBe(0.2)
    expect(modes.re?.systemPrompt).toBe('be exact')
    expect(modes.coding?.load).toEqual({ ...CODING_LOAD, sampling: DEFAULT_LOAD_CONFIG.sampling })
    expect(t.store.getModelDefaults(LOCAL.id)).toEqual(before)
    expect(t.state.changed).toBe(1)
    // A second change builds on the first one (flashAttention is not reset by a patch that leaves it out).
    await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 4096 } })
    expect(t.store.modeDefaults(LOCAL.id).coding?.load).toMatchObject({ contextLength: 4096, flashAttention: 'off' })
  })

  it('removes the mode\'s own load with config null, dropping an entry that becomes empty', async () => {
    const t = await setup()
    await t.store.setModeDefaults(LOCAL.id, 'coding', modeEntry({ load: CODING_LOAD }))
    await t.store.setModeDefaults(LOCAL.id, 're', modeEntry({ load: CODING_LOAD, systemPrompt: 'keep me' }))
    const r = await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: null })
    expect(r.status).toBe(200)
    expect(r.json).toMatchObject({ mode_own: false })
    expect(r.json.config.contextLength).toBe(8192)
    expect(t.store.modeDefaults(LOCAL.id).coding).toBeUndefined()
    await call(t.port, 'POST', path(LOCAL.id, 're'), { config: null })
    expect(t.store.modeDefaults(LOCAL.id).re).toMatchObject({ load: null, systemPrompt: 'keep me' })
    expect(t.store.allModeDefaults()[LOCAL.id]).toBeDefined()
    // Nothing is left of the model's entry when the last one goes.
    await t.store.setModeDefaults(LOCAL.id, 're', null)
    await t.store.setModeLoad(LOCAL.id, 'coding', CODING_LOAD)
    await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: null })
    expect(t.store.allModeDefaults()[LOCAL.id]).toBeUndefined()
  })

  it('refuses sampling in the patch, bad drafts and adapters, and writes nothing', async () => {
    const t = await setup()
    const sampling = await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { sampling: { temperature: 1 } } })
    expect(sampling.status).toBe(400)
    expect(sampling.json.error.message).toMatch(/sampling/)
    expect((await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { draftFile: 'nope.gguf' } })).json.error.message).toMatch(/draft_choices/)
    expect((await call(t.port, 'POST', path(LOCAL.id, 'coding'), { adapters: [{ id: 'nope.gguf', scale: 1 }] })).json.error.message).toMatch(/adapter_choices/)
    expect((await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 3 } })).status).toBe(400)
    expect(t.store.allModeDefaults()).toEqual({})
    expect(t.state.changed).toBe(0)
  })

  it('previews with dry_run without writing', async () => {
    const t = await setup()
    const r = await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 16384 }, dry_run: true })
    expect(r.status).toBe(200)
    expect(r.json).toMatchObject({ dry_run: true, mode_own: false })
    expect(r.json.config.contextLength).toBe(16384)
    expect(t.store.allModeDefaults()).toEqual({})
  })

  it('answers 404 for another mode or an unknown model, 409 for a rental\'s model, 415 without JSON', async () => {
    const t = await setup()
    for (const mode of ['chat', 'writing', 'adventure', 'nope']) expect((await call(t.port, 'GET', path(LOCAL.id, mode))).status).toBe(404)
    expect((await call(t.port, 'POST', path(LOCAL.id, 'chat'), { config: {} })).status).toBe(404)
    expect((await call(t.port, 'GET', path('missing', 'coding'))).status).toBe(404)
    t.fake.insts = [inst(RENTED.id, t.upPort, { host: HOST })]
    const rented = await call(t.port, 'POST', path(RENTED.id, 'coding'), { config: { contextLength: 4096 } })
    expect(rented.status).toBe(409)
    expect(rented.json.error.message).toMatch(/rented/)
    expect(t.store.allModeDefaults()).toEqual({})
    const plain = await call(t.port, 'POST', path(LOCAL.id, 'coding'), '{"config":{}}', { 'Content-Type': 'text/plain' })
    expect(plain.status).toBe(415)
  })

  it('keeps both of two changes that arrive at once', async () => {
    const t = await setup()
    await Promise.all([
      call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 65536 } }),
      call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { gpuLayers: 7 } })
    ])
    expect(t.store.modeDefaults(LOCAL.id).coding?.load).toMatchObject({ contextLength: 65536, gpuLayers: 7 })
  })

  it('does not reload unless asked (reload defaults to false)', async () => {
    const t = await setup()
    t.fake.insts = [inst(LOCAL.id, t.upPort)]
    const r = await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 65536 } })
    expect(r.status).toBe(200)
    expect(r.json.reload_blocked).toBeUndefined()
    expect(t.fake.loadCalls).toEqual([])
  })

  it('reloads with reload: true when it is allowed, and says why when it is not, with the settings saved either way', async () => {
    const t = await setup()
    t.fake.insts = [inst(LOCAL.id, t.upPort)]
    t.state.block = 'LLM Suite is writing something else on the computer.'
    const blocked = await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 65536 }, reload: true })
    expect(blocked.status).toBe(200)
    expect(blocked.json.reload_blocked).toBe('LLM Suite is writing something else on the computer.')
    expect(t.store.modeDefaults(LOCAL.id).coding?.load?.contextLength).toBe(65536)
    expect(t.fake.loadCalls).toEqual([])
    t.state.block = null
    const ok = await call(t.port, 'POST', path(LOCAL.id, 'coding'), { config: { contextLength: 65536 }, reload: true })
    expect(ok.json.reload_blocked).toBeUndefined()
    expect(t.fake.loadCalls).toHaveLength(1)
    expect(t.fake.loadCalls[0].config?.contextLength).toBe(65536)
  })
})

describe('POST /v1/models/{id}/reload', () => {
  const reload = (port: number, id: string, body: unknown = { mode: 'coding' }): Promise<Res> => call(port, 'POST', `/v1/models/${enc(id)}/reload`, body)

  async function withCoding() {
    const t = await setup()
    await t.store.setModeDefaults(LOCAL.id, 'coding', modeEntry({ load: CODING_LOAD }))
    t.fake.insts = [inst(LOCAL.id, t.upPort)]
    return t
  }

  it('loads the model again with the mode\'s settings, holds the model gate until the load has ended, and returns the modes body', async () => {
    const t = await withCoding()
    const r = await reload(t.port, LOCAL.id)
    expect(r.status).toBe(200)
    expect(r.json).toMatchObject({ object: 'model.modes', state: 'loading', instance: null })
    expect(t.fake.loadCalls).toHaveLength(1)
    expect(t.fake.loadCalls[0].config).toEqual({ ...CODING_LOAD, sampling: DEFAULT_LOAD_CONFIG.sampling })
    expect(t.gate.heldBy()).toBe(`reload:${LOCAL.id}`)
    // The load asks again when its turn comes.
    expect(t.fake.refuse?.()).toBeNull()
    t.state.block = 'The model is in use.'
    expect(t.fake.refuse?.()).toBe('The model is in use.')
    t.state.block = null
    expect(t.gate.busyReason()).toBe('LLM Suite is loading a model.')
    t.fake.pendingLoads[0].finish(inst(LOCAL.id, t.upPort, { config: CODING_LOAD }))
    await new Promise((r) => setTimeout(r, 20))
    expect(t.gate.heldBy()).toBeNull()
    const after = await call(t.port, 'GET', `/v1/models/${enc(LOCAL.id)}/modes`)
    expect(after.json.modes.coding.matches).toBe(true)
  })

  it('gives the gate back when the load fails', async () => {
    const t = await withCoding()
    expect((await reload(t.port, LOCAL.id)).status).toBe(200)
    expect(t.gate.heldBy()).not.toBeNull()
    t.fake.pendingLoads[0].fail(new Error('out of memory'))
    await new Promise((r) => setTimeout(r, 20))
    expect(t.gate.heldBy()).toBeNull()
  })

  it('is refused while it reloads, while the app is busy, and while another job holds the gate', async () => {
    const t = await withCoding()
    t.state.block = 'LLM Suite is writing something else on the computer.'
    const busy = await reload(t.port, LOCAL.id)
    expect(busy.status).toBe(409)
    expect(busy.json.error.message).toBe('LLM Suite is writing something else on the computer.')
    expect(t.fake.loadCalls).toEqual([])
    expect(t.gate.heldBy()).toBeNull()
    t.state.block = null
    const release = t.gate.take('adventure:1') as () => void
    const held = await reload(t.port, LOCAL.id)
    expect(held.status).toBe(409)
    expect(held.json.error.message).toMatch(/adventure/)
    release()
    expect((await reload(t.port, LOCAL.id)).status).toBe(200)
    // A second one while the first still loads: not a second load.
    const twice = await reload(t.port, LOCAL.id)
    expect(twice.status).toBe(409)
    expect(t.fake.loadCalls).toHaveLength(1)
  })

  it('refuses a model that is still loading, and does nothing when it is not loaded or already runs with the settings', async () => {
    const t = await withCoding()
    t.fake.insts = [inst(LOCAL.id, t.upPort, { status: 'loading' })]
    const loading = await reload(t.port, LOCAL.id)
    expect(loading.status).toBe(409)
    expect(loading.json.error.message).toMatch(/loading/)
    t.fake.insts = []
    const none = await reload(t.port, LOCAL.id)
    expect(none.status).toBe(200)
    expect(none.json.state).toBe('not-loaded')
    t.fake.insts = [inst(LOCAL.id, t.upPort, { config: CODING_LOAD })]
    const same = await reload(t.port, LOCAL.id)
    expect(same.status).toBe(200)
    expect(same.json.state).toBe('ready')
    expect(t.fake.loadCalls).toEqual([])
  })

  it('answers 409 for a rental\'s model, 400 for another mode, 404 for an unknown model, 415 without JSON', async () => {
    const t = await withCoding()
    t.fake.insts = [inst(RENTED.id, t.upPort, { host: HOST })]
    const rented = await reload(t.port, RENTED.id)
    expect(rented.status).toBe(409)
    expect(rented.json.error.message).toMatch(/rented vast.ai/)
    expect((await reload(t.port, LOCAL.id, { mode: 'chat' })).status).toBe(400)
    expect((await reload(t.port, LOCAL.id, {})).status).toBe(400)
    expect((await reload(t.port, 'missing')).status).toBe(404)
    expect((await call(t.port, 'POST', `/v1/models/${enc(LOCAL.id)}/reload`, '{"mode":"coding"}', { 'Content-Type': 'text/plain' })).status).toBe(415)
    expect(t.fake.loadCalls).toEqual([])
  })

  it('keeps a model the server loaded itself one it unloads when idle', async () => {
    const t = await withCoding()
    t.fake.insts = []
    // The server loads the model for a request: it is the server's own.
    expect((await call(t.port, 'POST', '/v1/chat/completions', chatBody)).status).toBe(200)
    const first = t.fake.insts[0].instanceId
    expect(t.server.getStatus().jitInstances.map((j) => j.instanceId)).toEqual([first])
    expect((await reload(t.port, LOCAL.id)).status).toBe(200)
    expect(t.server.getStatus().jitInstances).toEqual([])
    const next = inst(LOCAL.id, t.upPort, { config: CODING_LOAD })
    t.fake.pendingLoads[0].finish(next)
    await new Promise((r) => setTimeout(r, 20))
    expect(t.server.getStatus().jitInstances.map((j) => j.instanceId)).toEqual([next.instanceId])
  })
})

describe('X-RulZ-Mode', () => {
  it('loads a model that is not loaded with that mode\'s settings on chat completions and /props only', async () => {
    const t = await setup()
    const mode = (v: string): Record<string, string> => ({ 'X-RulZ-Mode': v })
    expect((await call(t.port, 'POST', '/v1/chat/completions', chatBody, mode('coding'))).status).toBe(200)
    expect(t.fake.ensureCalls).toEqual([{ modelId: LOCAL.id, mode: 'coding' }])
    t.fake.insts = []
    expect((await call(t.port, 'GET', `/props?model=${enc('Qwen3.5-9B-Q8_0')}`, undefined, mode('re'))).status).toBe(200)
    expect(t.fake.ensureCalls.at(-1)).toEqual({ modelId: LOCAL.id, mode: 're' })
    t.fake.insts = []
    await call(t.port, 'POST', '/v1/embeddings', { model: 'Qwen3.5-9B-Q8_0', input: 'x' }, mode('coding'))
    expect(t.fake.ensureCalls.at(-1)).toEqual({ modelId: LOCAL.id, mode: undefined })
    for (const other of ['chat', 'writing', 'Coding', '', 'coding,re']) {
      t.fake.insts = []
      await call(t.port, 'POST', '/v1/chat/completions', chatBody, other ? mode(other) : {})
      expect(t.fake.ensureCalls.at(-1)).toEqual({ modelId: LOCAL.id, mode: undefined })
    }
  })

  it('uses a loaded model as it is, whatever mode is asked for', async () => {
    const t = await setup()
    t.fake.insts = [inst(LOCAL.id, t.upPort)]
    await call(t.port, 'POST', '/v1/chat/completions', chatBody, { 'X-RulZ-Mode': 'coding' })
    expect(t.fake.ensureCalls).toEqual([])
    expect(t.fake.loadCalls).toEqual([])
  })

  it('leaves a lease on the model until 2 minutes after the request, only for a valid mode', async () => {
    const t = await setup()
    t.fake.insts = [inst(LOCAL.id, t.upPort)]
    await call(t.port, 'POST', '/v1/chat/completions', chatBody)
    await call(t.port, 'POST', '/v1/chat/completions', chatBody, { 'X-RulZ-Mode': 'chat' })
    expect(t.server.leasedUntil(LOCAL.id)).toBeNull()
    await call(t.port, 'POST', '/v1/chat/completions', chatBody, { 'X-RulZ-Mode': 're' })
    expect(t.server.leasedUntil(LOCAL.id)).toBe(t.state.now + 120_000)
    t.state.now += 119_000
    expect(t.server.leasedUntil(LOCAL.id)).not.toBeNull()
    t.state.now += 2_000
    expect(t.server.leasedUntil(LOCAL.id)).toBeNull()
  })
})

describe('X-RulZ-Rental: deny and the reply headers', () => {
  const deny = { 'X-RulZ-Rental': 'deny' }

  it('refuses an answer from a rental\'s model on every proxied path, /props included', async () => {
    const t = await setup()
    t.fake.insts = [inst(RENTED.id, t.upPort, { host: HOST })]
    const rented = { model: 'vast/x', messages: [] }
    const post = await call(t.port, 'POST', '/v1/chat/completions', rented, deny)
    expect(post.status).toBe(409)
    expect(post.json.error.message).toMatch(/rental RTX 4090/)
    expect((await call(t.port, 'POST', '/v1/messages', rented, deny)).status).toBe(409)
    expect((await call(t.port, 'POST', '/v1/embeddings', { model: 'vast/x', input: 'x' }, deny)).status).toBe(409)
    expect((await call(t.port, 'GET', `/props?model=${enc('vast/x')}`, undefined, deny)).status).toBe(409)
    expect((await call(t.port, 'GET', `/slots?model=${enc('vast/x')}`, undefined, deny)).status).toBe(409)
    expect(t.seen).toEqual([])
    // Without the header the same request is answered.
    expect((await call(t.port, 'POST', '/v1/chat/completions', rented)).status).toBe(200)
    // Any other value of the header changes nothing.
    expect((await call(t.port, 'POST', '/v1/chat/completions', rented, { 'X-RulZ-Rental': 'allow' })).status).toBe(200)
    // Several values: one deny among them is enough.
    expect((await call(t.port, 'POST', '/v1/chat/completions', rented, { 'X-RulZ-Rental': 'deny, allow' })).status).toBe(409)
    expect((await call(t.port, 'POST', '/v1/chat/completions', rented, { 'X-RulZ-Rental': 'allow,DENY' })).status).toBe(409)
  })

  it('refuses when the fallback for an unknown model name picks a rental\'s model', async () => {
    const t = await setup()
    t.fake.insts = [inst(RENTED.id, t.upPort, { host: HOST })]
    const unknown = { model: 'claude-sonnet-5-5', messages: [] }
    expect((await call(t.port, 'POST', '/v1/chat/completions', unknown, deny)).status).toBe(409)
    expect(t.seen).toEqual([])
    const open = await call(t.port, 'POST', '/v1/chat/completions', unknown)
    expect(open.status).toBe(200)
    expect(open.headers['x-rulz-rental']).toBe('1')
    expect(open.headers['x-rulz-model']).toBe('Rented-70B')
  })

  it('checks the instance that answers, also when it is not the one found first', async () => {
    const t = await setup()
    // The model loads for this request; by the time the request would go out, that instance is on a rental.
    const original = t.fake.ensure.bind(t.fake)
    t.fake.ensure = async (modelId: string, mode?: string) => {
      const r = await original(modelId, mode)
      t.fake.listOverride.set(r.instance.instanceId, { host: HOST })
      return r
    }
    const r = await call(t.port, 'POST', '/v1/chat/completions', chatBody, deny)
    expect(r.status).toBe(409)
    expect(t.seen).toEqual([])
  })

  it('lets a local model answer, and tells who answered in X-RulZ-Model and X-RulZ-Rental', async () => {
    const t = await setup()
    t.fake.insts = [inst(LOCAL.id, t.upPort), inst(UMLAUT.id, t.upPort)]
    const local = await call(t.port, 'POST', '/v1/chat/completions', chatBody, deny)
    expect(local.status).toBe(200)
    expect(local.headers['x-rulz-model']).toBe('Qwen3.5-9B-Q8_0')
    expect(local.headers['x-rulz-rental']).toBeUndefined()
    const prop = await call(t.port, 'GET', `/props?model=${enc('Qwen3.5-9B-Q8_0')}`)
    expect(prop.headers['x-rulz-model']).toBe('Qwen3.5-9B-Q8_0')
    // A name that is not plain ASCII is percent-encoded, so the header is valid.
    const umlaut = await call(t.port, 'POST', '/v1/chat/completions', { model: 'Modèle', messages: [] })
    expect(umlaut.status).toBe(200)
    expect(umlaut.headers['x-rulz-model']).toBe(encodeURIComponent('Modèle'))
    // The request headers are not passed to the model server.
    expect(t.seen.every((s) => !('x-rulz-rental' in s.headers) && !('x-rulz-mode' in s.headers))).toBe(true)
  })

  it('shows a rental on the model list and its entry only while it is loaded there', async () => {
    const t = await setup()
    const idle = await call(t.port, 'GET', '/v1/models')
    expect(idle.json.data.map((m: { id: string }) => m.id)).not.toContain('Rented-70B')
    t.fake.insts = [inst(RENTED.id, t.upPort, { host: HOST, reconnecting: true }), inst(LOCAL.id, t.upPort)]
    const list = await call(t.port, 'GET', '/v1/models')
    const rented = list.json.data.find((m: { id: string }) => m.id === 'Rented-70B')
    expect(rented.rental).toEqual({ label: 'RTX 4090', reconnecting: true })
    expect(list.json.data.find((m: { id: string }) => m.id === 'Qwen3.5-9B-Q8_0')).not.toHaveProperty('rental')
    const one = await call(t.port, 'GET', `/v1/models/${enc('vast/x')}`)
    expect(one.json.rental).toEqual({ label: 'RTX 4090', reconnecting: true })
  })
})

describe('GET /v1/re/overview?model=', () => {
  it('passes the model to the overview and leaves it out without ?model', async () => {
    const t = await setup()
    const plain = await call(t.port, 'GET', '/v1/re/overview')
    expect(plain.status).toBe(200)
    expect(plain.json.practice).not.toHaveProperty('model')
    const one = await call(t.port, 'GET', `/v1/re/overview?model=${enc('Qwen3.5-9B-Q8_0')}`)
    expect(one.json.practice.model).toBe('qwen3.5-9b-q8_0')
    expect(t.state.overviewModel).toEqual([undefined, 'Qwen3.5-9B-Q8_0'])
  })
})
