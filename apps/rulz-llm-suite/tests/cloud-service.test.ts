import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { CloudState } from '@shared/cloud'
import fixture from './fixtures/openrouter-models.json'
import { createCloudService, type CloudServiceHandle } from '../src/main/cloud/service'
import { Clock, FakeNet, Sleeper, TEST_KEY, fakeSafeStorage, goodStream, jsonResponse, sseResponse, type Call } from './helpers/cloudKit'

const SLUG = 'anthropic/claude-fable-5.2'
const ID = `openrouter/${SLUG}`

let dir: string
let svc: CloudServiceHandle | null = null
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'cloud-svc-'))
})
afterEach(() => {
  svc?.dispose()
  svc = null
  rmSync(dir, { recursive: true, force: true })
})

interface Env {
  svc: CloudServiceHandle
  net: FakeNet
  states: CloudState[]
  ejected: string[]
  models: ReturnType<typeof vi.fn>
  forgotten: string[]
  users: Map<string, number>
  sleeper: Sleeper
  clock: Clock
}

function setup(o: { encrypt?: boolean; keyAnswer?: (c: Call) => Response; net?: FakeNet } = {}): Env {
  const clock = new Clock()
  const sleeper = new Sleeper()
  sleeper.auto = true
  const net =
    o.net ??
    new FakeNet((c) => {
      if (c.url.endsWith('/models')) return jsonResponse(fixture)
      if (c.url.endsWith('/key')) return o.keyAnswer ? o.keyAnswer(c) : jsonResponse({ data: { label: 'my key', limit_remaining: 4.5, usage_daily: 0.25, is_free_tier: false } })
      if (c.url.endsWith('/chat/completions')) return sseResponse(c, goodStream('hi', 0.001), { 'x-generation-id': 'g1' })
      return new Response('', { status: 404 })
    })
  const states: CloudState[] = []
  const ejected: string[] = []
  const forgotten: string[] = []
  const users = new Map<string, number>()
  const models = vi.fn()
  const s = createCloudService({
    userData: dir,
    safeStorage: fakeSafeStorage(o.encrypt !== false),
    onChange: (st) => states.push(st),
    onModels: () => models(),
    eject: async (id) => void ejected.push(id),
    usersOf: (id) => users.get(id) ?? 0,
    forgetDefaults: async (id) => void forgotten.push(id),
    appVersion: '0.36.0',
    fetch: net.fetch,
    now: clock.now,
    sleep: sleeper.sleep
  })
  svc = s
  return { svc: s, net, states, ejected, models, forgotten, users, sleeper, clock }
}

const h = (e: Env) => e.svc.handlers

describe('key', () => {
  it('checks GET /key, then saves encrypted, and reports the key info', async () => {
    const e = setup()
    await e.svc.load()
    expect(e.svc.hasKey()).toBe(false)
    const st = await h(e)['cloud:setKey'](`  ${TEST_KEY} `)
    expect(st).toMatchObject({ keySaved: true, key: { label: 'my key', limitRemaining: 4.5, usageToday: 0.25, freeTier: false } })
    expect(e.net.calls[0].url).toBe('https://openrouter.ai/api/v1/key')
    expect(e.net.calls[0].headers.authorization).toBe(`Bearer ${TEST_KEY}`)
    expect(e.net.calls[0].redirect).toBe('error')
    expect(readFileSync(join(dir, 'cloud-secrets.json'), 'utf8')).not.toContain(TEST_KEY)
    expect(e.svc.hasKey()).toBe(true)
    // A new start finds it.
    const again = setup()
    await again.svc.load()
    expect(again.svc.hasKey()).toBe(true)
  })

  it.each([401, 403])('a refused key (%i) is not saved', async (status) => {
    const e = setup({ keyAnswer: () => jsonResponse({ error: { message: `no ${TEST_KEY}` } }, status) })
    await e.svc.load()
    await expect(h(e)['cloud:setKey'](TEST_KEY)).rejects.toThrow(/did not accept/)
    expect(e.svc.hasKey()).toBe(false)
    expect(existsSync(join(dir, 'cloud-secrets.json'))).toBe(false)
  })

  it('an unreachable OpenRouter: an error without the key, nothing saved', async () => {
    const e = setup({ net: new FakeNet(() => new Error(`offline ${TEST_KEY}`)) })
    await e.svc.load()
    const err = await h(e)['cloud:setKey'](TEST_KEY).catch((x: Error) => x)
    expect(err).toBeInstanceOf(Error)
    expect((err as Error).message).not.toContain(TEST_KEY)
    expect(e.svc.hasKey()).toBe(false)
  })

  it('without encryption on this PC the key is refused, not written', async () => {
    const e = setup({ encrypt: false })
    await e.svc.load()
    await expect(h(e)['cloud:setKey'](TEST_KEY)).rejects.toThrow(/encrypt/)
    expect(e.svc.hasKey()).toBe(false)
    expect(existsSync(join(dir, 'cloud-secrets.json'))).toBe(false)
  })

  it("'' forgets the key and ejects every cloud model", async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:setKey'](TEST_KEY)
    await h(e)['cloud:addModel'](SLUG)
    const st = await h(e)['cloud:setKey']('')
    expect(st).toMatchObject({ keySaved: false, key: null })
    expect(e.ejected).toEqual([ID])
    expect(e.svc.hasKey()).toBe(false)
    expect(e.svc.allowedForApps(ID)).toBe(false)
  })

  it('checkKey reads the limits again; a failure becomes the state error', async () => {
    const e = setup()
    await e.svc.load()
    expect((await h(e)['cloud:checkKey']()).key).toBeNull()
    await h(e)['cloud:setKey'](TEST_KEY)
    let answer: Response = jsonResponse({ data: { label: 'k', limit_remaining: null, usage_daily: 1.5, is_free_tier: true } })
    e.net.handler = () => answer
    expect((await h(e)['cloud:checkKey']()).key).toMatchObject({ limitRemaining: null, usageToday: 1.5, freeTier: true })
    answer = jsonResponse({}, 401)
    const st = await h(e)['cloud:checkKey']()
    expect(st.error).toMatch(/did not accept/)
  })

  it('never exposes the key in a state', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:setKey'](TEST_KEY)
    await h(e)['cloud:addModel'](SLUG)
    await e.svc.fetch(ID, '/v1/chat/completions', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'x' }], stream: true }) }, 'window').then((r) => r.text())
    await e.svc.settled()
    expect(JSON.stringify(e.states)).not.toContain(TEST_KEY)
    expect(JSON.stringify(await h(e)['cloud:state']())).not.toContain(TEST_KEY)
    expect(JSON.stringify(e.svc.models())).not.toContain(TEST_KEY)
    expect(e.svc.scrub(new Error(`oops ${TEST_KEY}`))).toBe('oops ***')
    expect(e.svc.scrub(`oops ${TEST_KEY}`)).toBe('oops ***')
  })
})

describe('models', () => {
  it('adds from the catalog, lists as catalog entries, updates with clamping, removes', async () => {
    const e = setup()
    await e.svc.load()
    await expect(h(e)['cloud:addModel']('nope/none')).rejects.toThrow(/not in OpenRouter/)
    const st = await h(e)['cloud:addModel'](SLUG)
    expect(st.models.map((m) => m.slug)).toEqual([SLUG])
    expect(e.models).toHaveBeenCalledTimes(1)
    const entries = e.svc.models()
    expect(entries).toHaveLength(1)
    expect(entries[0]).toMatchObject({ id: ID, name: 'Anthropic: Claude Fable 5.2', path: '', sizeBytes: 0 })
    expect(entries[0].remote).toBeUndefined()
    const up = await h(e)['cloud:updateModel'](SLUG, { contextLimit: 1, replyLimit: 99_999_999, allowApps: true })
    expect(up.models[0]).toMatchObject({ contextLimit: 2048, replyLimit: 128000, allowApps: true })
    expect(e.models).toHaveBeenCalledTimes(2)
    expect(e.svc.props(ID)).toEqual({ contextLength: 2048, hasVision: true, free: false })
    const gone = await h(e)['cloud:removeModel'](SLUG)
    expect(gone.models).toEqual([])
    expect(e.ejected).toEqual([ID])
    expect(e.forgotten).toEqual([ID])
    expect(e.svc.models()).toEqual([])
    expect(() => e.svc.props(ID)).toThrow()
  })

  it('removing is refused while the model answers', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:addModel'](SLUG)
    e.users.set(ID, 1)
    await expect(h(e)['cloud:removeModel'](SLUG)).rejects.toThrow(/answering/)
    expect(e.svc.models()).toHaveLength(1)
    expect(e.ejected).toEqual([])
    e.users.set(ID, 0)
    await h(e)['cloud:removeModel'](SLUG)
    expect(e.svc.models()).toEqual([])
  })

  it('check: needs a key and a listed model', async () => {
    const e = setup()
    await e.svc.load()
    expect(() => e.svc.check(ID)).toThrow(/not in your list/)
    await h(e)['cloud:addModel'](SLUG)
    expect(() => e.svc.check(ID)).toThrow(/No OpenRouter key/)
    await h(e)['cloud:setKey'](TEST_KEY)
    expect(() => e.svc.check(ID)).not.toThrow()
    expect(() => e.svc.check('model.gguf')).toThrow()
  })

  it('allowedForApps needs the tick and a key', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:addModel'](SLUG)
    expect(e.svc.allowedForApps(ID)).toBe(false)
    await h(e)['cloud:updateModel'](SLUG, { allowApps: true })
    expect(e.svc.allowedForApps(ID)).toBe(false)
    await h(e)['cloud:setKey'](TEST_KEY)
    expect(e.svc.allowedForApps(ID)).toBe(true)
    expect(e.svc.allowedForApps('openrouter/other')).toBe(false)
    expect(e.svc.allowedForApps('x.gguf')).toBe(false)
  })

  it('the catalog handler returns the list and refreshes the info of added models', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:addModel'](SLUG)
    const changed = JSON.parse(JSON.stringify(fixture))
    changed.data.find((m: any) => m.id === SLUG).name = 'Fable Renamed'
    e.net.handler = () => jsonResponse(changed)
    const cat = await h(e)['cloud:catalog'](true)
    expect(cat.models).toHaveLength(6)
    expect(e.svc.models()[0].name).toBe('Fable Renamed')
    expect(e.models).toHaveBeenCalledTimes(2)
    // Offline: the cached list and the error in the state.
    e.net.handler = () => new Error('down')
    const again = await h(e)['cloud:catalog'](true)
    expect(again.models).toHaveLength(6)
    expect((await h(e)['cloud:state']()).error).toMatch(/model list/)
  })

  it('models are kept across a restart', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:addModel'](SLUG)
    await h(e)['cloud:updateModel'](SLUG, { allowApps: true, replyLimit: 4000 })
    const again = setup()
    await again.svc.load()
    expect(again.svc.models()[0].cloud).toMatchObject({ slug: SLUG, allowApps: true, replyLimit: 4000, provider: 'openrouter' })
  })
})

describe('options and state', () => {
  it('daily limit: null or at least $0.01; starts at $1; changes are pushed with increasing rev', async () => {
    const e = setup()
    await e.svc.load()
    const s0 = await h(e)['cloud:state']()
    expect(s0.options).toEqual({ dailyLimitUsd: 1, allowTraining: false })
    expect((await h(e)['cloud:setOptions']({ dailyLimitUsd: null })).options.dailyLimitUsd).toBeNull()
    expect((await h(e)['cloud:setOptions']({ dailyLimitUsd: 0.5, allowTraining: true })).options).toEqual({ dailyLimitUsd: 0.5, allowTraining: true })
    await expect(h(e)['cloud:setOptions']({ dailyLimitUsd: 0 })).rejects.toThrow()
    await expect(h(e)['cloud:setOptions']({ dailyLimitUsd: 0.009 })).rejects.toThrow()
    expect((await h(e)['cloud:state']()).options.dailyLimitUsd).toBe(0.5)
    const revs = e.states.map((s) => s.rev)
    expect(revs.length).toBeGreaterThanOrEqual(3)
    expect(revs).toEqual([...revs].sort((a, b) => a - b))
    expect(new Set(revs).size).toBe(revs.length)
    const again = setup()
    await again.svc.load()
    expect((await h(again)['cloud:state']()).options).toEqual({ dailyLimitUsd: 0.5, allowTraining: true })
  })

  it('spending shows in the state (throttled pushes) and a request uses the live limit', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:setKey'](TEST_KEY)
    await h(e)['cloud:addModel'](SLUG)
    const res = await e.svc.fetch(ID, '/v1/chat/completions', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'x' }], stream: true }) }, 'window')
    await res.text()
    await e.svc.settled()
    const st = await h(e)['cloud:state']()
    expect(st.spend.spent).toBe(1000)
    expect(st.spend.models).toEqual({ [SLUG]: 1000 })
    expect(st.spendError).toBeNull()
    await vi.waitFor(() => expect(e.states.at(-1)?.spend.spent).toBe(1000))
  })

  it('a corrupt spending file shows as spendError and refuses requests', async () => {
    writeFileSync(join(dir, 'cloud-spend.json'), 'garbage')
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:setKey'](TEST_KEY)
    await h(e)['cloud:addModel'](SLUG)
    expect((await h(e)['cloud:state']()).spendError).toMatch(/spending record/)
    const res = await e.svc.fetch(ID, '/v1/chat/completions', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'x' }] }) }, 'window')
    expect(res.status).toBe(503)
    expect(readFileSync(join(dir, 'cloud-spend.json'), 'utf8')).toBe('garbage')
  })
})

describe('start and stop', () => {
  it('load settles reservations a crash left; dispose leaves open ones on disk', async () => {
    const a = setup()
    await a.svc.load()
    await h(a)['cloud:setKey'](TEST_KEY)
    await h(a)['cloud:addModel'](SLUG)
    // A request that is still running when the app closes.
    a.net.handler = (c) => {
      if (c.url.endsWith('/chat/completions')) return new Promise<Response>(() => undefined)
      return jsonResponse({}, 404)
    }
    void a.svc.fetch(ID, '/v1/chat/completions', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'x' }], stream: true }) }, 'window').catch(() => undefined)
    await vi.waitFor(() => expect(a.net.chats).toHaveLength(1))
    a.svc.dispose()
    const left = JSON.parse(readFileSync(join(dir, 'cloud-spend.json'), 'utf8')).reservations
    expect(left).toHaveLength(1)
    // The next start settles it: no generation id, so the reservation is charged.
    const b = setup()
    await b.svc.load()
    const st = await h(b)['cloud:state']()
    expect(st.spend.reserved).toBe(0)
    expect(st.spend.spent).toBe(left[0].amount)
    expect(JSON.parse(readFileSync(join(dir, 'cloud-spend.json'), 'utf8')).reservations).toEqual([])
  })

  it('a fetch through the service answers as the gateway does and checks who asks', async () => {
    const e = setup()
    await e.svc.load()
    await h(e)['cloud:setKey'](TEST_KEY)
    await h(e)['cloud:addModel'](SLUG)
    expect((await e.svc.fetch(ID, '/props', {}, 'api')).status).toBe(403)
    await h(e)['cloud:updateModel'](SLUG, { allowApps: true })
    const props = await (await e.svc.fetch(ID, '/props', {}, 'api')).json()
    expect(props.n_ctx).toBe(32768)
    expect((await e.svc.fetch('openrouter/missing', '/props', {}, 'window')).status).toBe(404)
  })
})

describe('what the window sends', () => {
  it('a wrong type is refused before anything is read or sent', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'cloud-args-'))
    const net = new FakeNet()
    const s = createCloudService({ userData: dir, safeStorage: fakeSafeStorage(), appVersion: '0', onChange: () => undefined, onModels: () => undefined, eject: async () => undefined, usersOf: () => 0, forgetDefaults: async () => undefined, fetch: net.fetch })
    await s.load()
    const bad = [
      () => s.handlers['cloud:setKey'](5 as never),
      () => s.handlers['cloud:addModel'](undefined as never),
      () => s.handlers['cloud:addModel']('  '),
      () => s.handlers['cloud:updateModel']('a/b', null as never),
      () => s.handlers['cloud:removeModel']({} as never),
      () => s.handlers['cloud:setOptions']([] as never)
    ]
    for (const f of bad) await expect(f()).rejects.toThrow()
    expect(net.calls).toEqual([])
    s.dispose()
    rmSync(dir, { recursive: true, force: true })
  })
})
