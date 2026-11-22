import { mkdirSync, mkdtempSync, renameSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { WebSocket } from 'ws'
import { DEFAULT_SETTINGS } from '../src/shared/defaults'
import { PHONE_EVENTS, PHONE_INVOKE, PHONE_PATH, PHONE_PROTOCOL } from '../src/shared/phone'
import type { AppSettings } from '../src/shared/types'
import { parseSettings } from '../src/main/settings/schema'
import { NovelStore } from '../src/main/novel/store'
import { PhoneAccess } from '../src/main/phone/access'
import { createPhoneService, type PhoneService } from '../src/main/phone/host'
import { settingsForRenderer, settingsPatchFromRenderer } from '../src/main/phone/settings'
import { newToken, originAllowed, PhoneServer, tokenMatches, type PhoneHost } from '../src/main/phone/server'
import { isTailscaleIPv4, tailscaleAddresses } from '../src/main/phone/tailscale'

type Msg = Record<string, unknown>

class Phone {
  readonly messages: Msg[] = []
  private waiters: ((m: Msg) => void)[] = []
  closed: Promise<number>
  constructor(readonly ws: WebSocket) {
    ws.on('message', (d) => {
      const m = JSON.parse(d.toString()) as Msg
      this.messages.push(m)
      this.waiters.splice(0).forEach((w) => w(m))
    })
    this.closed = new Promise((r) => ws.on('close', (code) => r(code)))
  }
  static async connect(port: number): Promise<Phone> {
    const ws = new WebSocket(`ws://127.0.0.1:${port}${PHONE_PATH}`)
    const phone = new Phone(ws)
    await new Promise((r, j) => ws.once('open', r).once('error', j))
    return phone
  }
  send(m: Msg): void {
    this.ws.send(JSON.stringify(m))
  }
  next(pred: (m: Msg) => boolean = () => true): Promise<Msg> {
    const found = this.messages.find(pred)
    if (found) {
      this.messages.splice(this.messages.indexOf(found), 1)
      return Promise.resolve(found)
    }
    return new Promise((resolve) => {
      const wait = (m: Msg): void => {
        if (pred(m)) {
          this.messages.splice(this.messages.indexOf(m), 1)
          resolve(m)
        } else this.waiters.push(wait)
      }
      this.waiters.push(wait)
    })
  }
  async invoke(channel: string, ...args: unknown[]): Promise<Msg> {
    const id = Math.random().toString(36).slice(2)
    this.send({ t: 'invoke', id, channel, args })
    return this.next((m) => m.t === 'result' && m.id === id)
  }
}

let server: PhoneServer | null = null
const phones: Phone[] = []
afterEach(async () => {
  phones.splice(0).forEach((p) => p.ws.terminate())
  await server?.close()
  server = null
})

function fakeHost() {
  const listeners = new Set<(c: string, p: unknown) => void>()
  const calls: { owner: string; channel: string; args: unknown[] }[] = []
  const released: string[] = []
  const host: PhoneHost = {
    version: '9.9.9',
    name: 'test-pc',
    subscribe: (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    invoke: async (owner, channel, args) => {
      calls.push({ owner, channel, args })
      if (channel === 'boom') throw new Error('It broke.')
      return { echo: args }
    },
    release: (owner) => void released.push(owner)
  }
  const emit = (c: string, p: unknown): void => listeners.forEach((l) => l(c, p))
  return { host, calls, released, emit }
}

async function start(token = 'right-code') {
  const fake = fakeHost()
  const clients: number[] = []
  server = new PhoneServer(fake.host, (c) => clients.push(c.length))
  await server.listen(['127.0.0.1'], 0, token)
  const connect = async (): Promise<Phone> => {
    const p = await Phone.connect(server!.port!)
    phones.push(p)
    return p
  }
  const pair = async (code = token): Promise<Phone> => {
    const p = await connect()
    p.send({ t: 'hello', token: code, device: 'Pixel', protocol: PHONE_PROTOCOL })
    return p
  }
  return { ...fake, clients, connect, pair }
}

describe('PhoneServer', () => {
  it('answers a plain request with only who it is', async () => {
    await start()
    const res = await fetch(`http://127.0.0.1:${server!.port}/`)
    expect(await res.json()).toEqual({ app: 'rulz-llm-suite', protocol: PHONE_PROTOCOL })
  })

  it('welcomes a phone with the right pairing code and lists it', async () => {
    const t = await start()
    const p = await t.pair()
    expect(await p.next()).toEqual({ t: 'welcome', protocol: PHONE_PROTOCOL, version: '9.9.9', name: 'test-pc' })
    expect(server!.clients()).toMatchObject([{ device: 'Pixel', address: '127.0.0.1' }])
    expect(typeof server!.clients()[0].connectedAt).toBe('number')
    expect(t.clients).toEqual([1])
  })

  it('turns away a wrong code without running anything', async () => {
    const t = await start()
    const p = await t.pair('wrong')
    expect(await p.next()).toEqual({ t: 'denied', reason: 'wrong-code' })
    expect(await p.closed).toBe(4001)
    expect(t.calls).toEqual([])
    expect(server!.clients()).toEqual([])
  })

  it('ignores requests sent before pairing', async () => {
    const t = await start()
    const p = await t.connect()
    p.send({ t: 'invoke', id: '1', channel: 'novel:listProjects', args: [] })
    expect(await p.next()).toEqual({ t: 'denied', reason: 'bad-hello' })
    expect(t.calls).toEqual([])
  })

  it('locks an address out after ten wrong codes', async () => {
    const t = await start()
    for (let i = 0; i < 10; i++) await (await t.pair('wrong')).closed
    const p = await t.pair()
    expect(await p.next()).toEqual({ t: 'denied', reason: 'locked' })
  })

  it('gives connections opened side by side no extra guesses past the limit', async () => {
    const t = await start()
    for (let i = 0; i < 6; i++) await (await t.pair('wrong')).closed
    const opened = await Promise.all(Array.from({ length: 6 }, () => t.connect()))
    for (const p of opened.slice(0, 4)) {
      p.send({ t: 'hello', token: 'wrong', device: 'x', protocol: PHONE_PROTOCOL })
      await p.closed
    }
    const late = opened[4]
    late.send({ t: 'hello', token: 'right-code', device: 'x', protocol: PHONE_PROTOCOL })
    expect(await late.next()).toEqual({ t: 'denied', reason: 'locked' })
  })

  it('refuses web pages from other sites, but not the phone app or clients without an origin', async () => {
    await start()
    const refused = new WebSocket(`ws://127.0.0.1:${server!.port}${PHONE_PATH}`, { origin: 'https://evil.example' })
    const status = await new Promise<number>((resolve) => refused.once('unexpected-response', (_req, res) => resolve(res.statusCode ?? 0)).once('error', () => resolve(-1)))
    expect(status).toBe(403)
    expect(originAllowed('http://localhost')).toBe(true)
    expect(originAllowed('https://localhost')).toBe(true)
    expect(originAllowed('capacitor://localhost')).toBe(true)
    expect(originAllowed(undefined)).toBe(true)
    expect(originAllowed('http://localhost.evil.example')).toBe(false)
    expect(originAllowed('http://100.64.0.1:5173')).toBe(false)
  })

  it('answers only on the phone path', async () => {
    await start()
    const other = new WebSocket(`ws://127.0.0.1:${server!.port}/rulz`)
    const status = await new Promise<number>((resolve) => other.once('unexpected-response', (_req, res) => resolve(res.statusCode ?? 0)).once('error', () => resolve(-1)))
    expect(status).toBe(404)
  })

  it('caps connections that have not paired yet, and drops them when phone access stops', async () => {
    await start()
    const waiting = await Promise.all(Array.from({ length: 8 }, () => Phone.connect(server!.port!)))
    phones.push(...waiting)
    const extra = new WebSocket(`ws://127.0.0.1:${server!.port}${PHONE_PATH}`)
    const status = await new Promise<number>((resolve) => extra.once('unexpected-response', (_req, res) => resolve(res.statusCode ?? 0)).once('error', () => resolve(-1)))
    expect(status).toBe(503)
    await server!.close()
    server = null
    expect(await Promise.all(waiting.map((p) => p.closed))).toHaveLength(8)
  })

  it("dispatches each request with the phone's own owner id, and returns results and errors", async () => {
    const t = await start()
    const a = await t.pair()
    await a.next()
    const b = await t.pair()
    await b.next()
    expect(await a.invoke('novel:readFile', 'C:\\x', 'a.md')).toMatchObject({ ok: true, value: { echo: ['C:\\x', 'a.md'] } })
    await b.invoke('novel:readFile', 'C:\\x', 'a.md')
    expect(await a.invoke('boom')).toMatchObject({ ok: false, error: 'It broke.' })
    const owners = t.calls.map((c) => c.owner)
    expect(owners[0]).toMatch(/^phone:[0-9a-f-]{36}$/)
    expect(owners[1]).not.toBe(owners[0])
    expect(owners[2]).toBe(owners[0])
    const ids = server!.clients().map((c) => `phone:${c.id}`)
    expect(ids).toEqual(expect.arrayContaining([owners[0], owners[1]]))
  })

  it('releases a phone\'s projects when its socket closes, and when phone access stops', async () => {
    const t = await start()
    const a = await t.pair()
    await a.next()
    const b = await t.pair()
    await b.next()
    const aOwner = `phone:${server!.clients()[0].id}`
    const bOwner = `phone:${server!.clients()[1].id}`
    a.ws.close()
    await a.closed
    await new Promise((r) => setTimeout(r, 20))
    expect(t.released).toEqual([aOwner])
    await server!.close()
    server = null
    expect(t.released.sort()).toEqual([aOwner, bOwner].sort())
  })

  it('sends only the events a phone may have, and only to paired phones', async () => {
    const t = await start()
    const paired = await t.pair()
    await paired.next()
    const unpaired = await t.connect()
    t.emit('chat:event', { secret: true })
    t.emit('models:changed', [])
    t.emit('novel:changed', { kind: 'deleted', folder: 'f', path: 'p' })
    expect(await paired.next()).toEqual({ t: 'event', channel: 'novel:changed', payload: { kind: 'deleted', folder: 'f', path: 'p' } })
    expect(paired.messages).toEqual([])
    expect(unpaired.messages).toEqual([])
    expect([...PHONE_EVENTS].sort()).toEqual(['novel:changed', 'writing:job'])
  })

  it('disconnects phones when the pairing code changes, and one phone on request', async () => {
    const t = await start()
    const p = await t.pair()
    await p.next()
    server!.setToken('new-code')
    expect(await p.closed).toBe(4001)
    expect(await (await t.pair('right-code')).next()).toEqual({ t: 'denied', reason: 'wrong-code' })
    const fresh = await t.pair('new-code')
    expect(await fresh.next()).toMatchObject({ t: 'welcome' })
    server!.disconnect(server!.clients()[0].id)
    expect(await fresh.closed).toBe(4002)
  })
})

describe('the phone service (allowlist and restrictions)', () => {
  async function service(): Promise<{ svc: PhoneService; store: NovelStore; events: unknown[]; connect: () => Promise<Phone>; folderOf: (title: string) => Promise<string> }> {
    const base = mkdtempSync(join(tmpdir(), 'rulz-phone-svc-'))
    mkdirSync(join(base, 'recycle'))
    const events: unknown[] = []
    let svcRef: PhoneService | null = null
    const store = new NovelStore({
      rootSetting: () => '',
      documents: () => join(base, 'Documents'),
      trashItem: async (p) => renameSync(p, join(base, 'recycle', String(Date.now()))),
      emit: (c) => svcRef?.emit('novel:changed', c),
      watch: false
    })
    let settings: AppSettings = { ...DEFAULT_SETTINGS, phoneEnabled: true, phoneToken: 'tok' }
    const svc = createPhoneService({
      settings: { get: () => settings, update: async (patch) => (settings = parseSettings({ ...settings, ...patch })) },
      novel: store,
      version: '1.0.0',
      emitStatus: (s) => events.push(s),
      addresses: () => ['127.0.0.1'],
      dnsName: async () => undefined,
      name: 'pc'
    })
    svcRef = svc
    server = new PhoneServer(svc.host, () => undefined)
    await server.listen(['127.0.0.1'], 0, 'tok')
    const connect = async (): Promise<Phone> => {
      const p = await Phone.connect(server!.port!)
      phones.push(p)
      p.send({ t: 'hello', token: 'tok', device: 'Pixel', protocol: PHONE_PROTOCOL })
      await p.next()
      return p
    }
    const folderOf = async (title: string): Promise<string> => (await store.createProject(title)).folder
    return { svc, store, events, connect, folderOf }
  }

  it('refuses every channel outside the allowlist, before any handler runs', async () => {
    const t = await service()
    const p = await t.connect()
    for (const ch of ['chat:send', 'chat:start', 'settings:get', 'settings:update', 'dialog:pickFolder', 'shell:openPath', 'phone:pairing', 'phone:newToken', 'phone:setEnabled', 'novel:deleteProject', 'novel:emptyTrash', 'novel:exportRecords', 'novel:deleteRecords', 'novel:record', 'novel:recordsInfo', 'novel:journal', 'novel:discardJournal', 'novel:flushDone', 'novel:root', 'wbench:save', 'nothing:here']) {
      expect(PHONE_INVOKE.has(ch)).toBe(false)
      expect(await p.invoke(ch)).toEqual({ t: 'result', id: expect.any(String), ok: false, error: 'That is not available from the phone.' })
    }
  })

  it('only registers channels that are on the allowlist', async () => {
    const t = await service()
    expect(() => t.svc.handlers.register('chat:send' as never, (() => undefined) as never)).toThrow('not a phone channel')
    // Writing channels are plugged in later: until then they answer "not available yet".
    const p = await t.connect()
    expect(await p.invoke('writing:models')).toMatchObject({ ok: false, error: 'That is not available yet.' })
    t.svc.handlers.registerAll({ 'writing:models': () => [{ id: 'm', name: 'M', loaded: true }] })
    expect(await p.invoke('writing:models')).toMatchObject({ ok: true, value: [{ id: 'm' }] })
    t.svc.handlers.register('writing:job', (owner, jobId) => ({ owner, jobId }) as never)
    expect(await p.invoke('writing:job', 'j1')).toMatchObject({ ok: true, value: { jobId: 'j1', owner: expect.stringMatching(/^phone:/) } })
  })

  it('opens only projects that novel:listProjects returns', async () => {
    const t = await service()
    const mine = await t.folderOf('Mine')
    await t.store.closeProject(mine)
    const p = await t.connect()
    const ok = await p.invoke('novel:openProject', mine)
    expect(ok).toMatchObject({ ok: true })
    const outside = mkdtempSync(join(tmpdir(), 'rulz-outside-'))
    for (const bad of [outside, join(mine, '..'), join(mine, 'bible'), 'relative/path', 42, undefined]) {
      expect(await p.invoke('novel:openProject', bad)).toMatchObject({ ok: false })
    }
    // Case differences count as the same folder on Windows only.
    if (process.platform === 'win32') expect(await p.invoke('novel:openProject', mine.toUpperCase())).toMatchObject({ ok: true })
  })

  it('refuses saveFile without a baseHash or with force, and passes a plain save', async () => {
    const t = await service()
    const folder = await t.folderOf('Book')
    const p = await t.connect()
    expect(await p.invoke('novel:openProject', folder)).toMatchObject({ ok: true })
    const st = await t.store.openProject(folder)
    const path = `manuscript/${st.outline.chapters[0].scenes[0].file}`
    const file = await t.store.readFile(folder, path)
    expect(await p.invoke('novel:saveFile', { folder, path, text: 'x' })).toMatchObject({ ok: false, error: expect.stringContaining('based on') })
    expect(await p.invoke('novel:saveFile', { folder, path, text: 'x', baseHash: file.hash, force: true })).toMatchObject({ ok: false, error: expect.stringContaining('overwrite') })
    expect(await p.invoke('novel:saveFile', null)).toMatchObject({ ok: false })
    expect((await t.store.readFile(folder, path)).text).toBe('')
    const good = await p.invoke('novel:saveFile', { folder, path, text: 'Hello.\n', baseHash: file.hash })
    expect(good).toMatchObject({ ok: true, value: { ok: true } })
    expect((await t.store.readFile(folder, path)).text).toBe('Hello.\n')
    // A stale hash comes back as a conflict, as for the desktop.
    expect(await p.invoke('novel:saveFile', { folder, path, text: 'Other.\n', baseHash: file.hash })).toMatchObject({ ok: true, value: { ok: false, reason: 'conflict' } })
  })

  it('counts a phone as an owner and lets go of its projects when the socket closes', async () => {
    const t = await service()
    const folder = await t.folderOf('Book') // the window's (the store default owner)
    const p = await t.connect()
    await p.invoke('novel:openProject', folder)
    expect(t.store.ownersOf(folder)).toHaveLength(2)
    await p.invoke('novel:closeProject', folder)
    expect(t.store.ownersOf(folder)).toEqual(['window'])
    await p.invoke('novel:openProject', folder)
    p.ws.close()
    await p.closed
    await new Promise((r) => setTimeout(r, 50))
    expect(t.store.ownersOf(folder)).toEqual(['window'])
    await expect(t.store.readFile(folder, 'style.md')).resolves.toBeTruthy()
  })

  it('only answers about projects the calling phone opened itself', async () => {
    const t = await service()
    const folder = await t.folderOf('Book') // open for the window only
    const a = await t.connect()
    const b = await t.connect()
    const refused = { ok: false, error: 'That project is not open on this phone.' }
    const calls: [string, ...unknown[]][] = [
      ['novel:readFile', folder, 'style.md'],
      ['novel:saveFile', { folder, path: 'style.md', text: 'x', baseHash: '' }],
      ['novel:outlinePatch', folder, []],
      ['novel:projectPatch', folder, {}],
      ['novel:createEntry', folder, 'X', 'other'],
      ['novel:deleteScene', folder, 's'],
      ['novel:deleteChapter', folder, 'c'],
      ['novel:deleteEntry', folder, 'e'],
      ['novel:history', folder, 'style.md'],
      ['novel:readSnapshot', folder, 'x'],
      ['novel:trash', folder],
      ['novel:restoreTrash', folder, 'x'],
      ['novel:statePatch', folder, []],
      ['novel:dismissFinding', folder, 's', 'k', true],
      ['writing:jobs', folder],
      ['writing:run', { folder }]
    ]
    for (const [ch, ...args] of calls) expect(await a.invoke(ch, ...args), ch).toMatchObject(refused)
    expect(await a.invoke('novel:readFile', 42)).toMatchObject(refused)
    // A project another phone opened is not this phone's either.
    await b.invoke('novel:openProject', folder)
    expect(await a.invoke('novel:readFile', folder, 'style.md')).toMatchObject(refused)
    expect(await b.invoke('novel:readFile', folder, 'style.md')).toMatchObject({ ok: true })
    // Closing takes it away again.
    await b.invoke('novel:closeProject', folder)
    expect(await b.invoke('novel:readFile', folder, 'style.md')).toMatchObject(refused)
  })

  it('does not keep a project for a phone whose connection ended while its open was still running', async () => {
    const t = await service()
    const folder = await t.folderOf('Book')
    await t.store.closeProject(folder)
    const owner = 'phone:gone'
    const open = t.svc.host.invoke(owner, 'novel:openProject', [folder])
    await t.svc.host.release?.(owner) // the socket closes while the open reads the disk
    await open.catch(() => undefined)
    expect(t.store.ownersOf(folder)).toEqual([])
    expect(t.store.hasOpenProject()).toBe(false)
    await expect(t.svc.host.invoke(owner, 'novel:openProject', [folder])).rejects.toThrow('disconnected')
  })

  it('tells every phone about a save made by one phone', async () => {
    const t = await service()
    const folder = await t.folderOf('Book')
    const a = await t.connect()
    const b = await t.connect()
    await a.invoke('novel:openProject', folder)
    await b.invoke('novel:openProject', folder)
    const st = await t.store.openProject(folder)
    const path = `manuscript/${st.outline.chapters[0].scenes[0].file}`
    const file = await t.store.readFile(folder, path)
    const saved = await a.invoke('novel:saveFile', { folder, path, text: 'Hello.\n', baseHash: file.hash })
    const hash = (saved.value as { hash: string }).hash
    const heard = await b.next((m) => m.t === 'event')
    expect(heard).toEqual({ t: 'event', channel: 'novel:changed', payload: { kind: 'text', folder, path, hash } })
  })
})

describe('phone settings and pairing', () => {
  it('has the defaults', () => {
    expect(DEFAULT_SETTINGS).toMatchObject({ phoneEnabled: false, phonePort: 7421, phoneToken: '' })
    expect(parseSettings({})).toMatchObject({ phoneEnabled: false, phonePort: 7421, phoneToken: '' })
    expect(parseSettings({ phonePort: 70000, phoneEnabled: 'yes' })).toMatchObject({ phonePort: 7421, phoneEnabled: false })
  })

  it('never sends the pairing code to the renderer, and the renderer cannot set it or switch phone access', () => {
    const s = { ...DEFAULT_SETTINGS, phoneToken: 'secret', phoneEnabled: true }
    expect(settingsForRenderer(s).phoneToken).toBe('')
    expect(JSON.stringify(settingsForRenderer(s))).not.toContain('secret')
    expect(settingsPatchFromRenderer({ theme: 'light', phoneToken: '', phoneEnabled: false, phonePort: 7500 })).toEqual({ theme: 'light', phonePort: 7500 })
  })

  it('makes a 32 byte base64url pairing code and compares codes safely', () => {
    const t = newToken()
    expect(t).toMatch(/^[A-Za-z0-9_-]{43}$/)
    expect(newToken()).not.toBe(t)
    expect(tokenMatches(t, t)).toBe(true)
    expect(tokenMatches(t + 'x', t)).toBe(false)
    expect(tokenMatches('', '')).toBe(false)
    expect(tokenMatches(undefined, t)).toBe(false)
    expect(tokenMatches(5, t)).toBe(false)
  })

  it('turns on with a first code, pairs with a rulzwriter link, and a new code disconnects phones', async () => {
    let settings: AppSettings = { ...DEFAULT_SETTINGS, phonePort: 0 }
    const statuses: string[] = []
    const fake = fakeHost()
    const access = new PhoneAccess({
      settings: { get: () => settings, update: async (patch) => (settings = parseSettings({ ...settings, ...patch })) },
      host: fake.host,
      emit: (s) => statuses.push(s.state),
      addresses: () => ['127.0.0.1'],
      dnsName: async () => undefined
    })
    try {
      expect(access.status().state).toBe('off')
      const on = await access.setEnabled(true)
      expect(on).toMatchObject({ enabled: true, state: 'on' })
      expect(settings.phoneToken).toMatch(/^[A-Za-z0-9_-]{43}$/)
      const pairing = await access.pairing()
      expect(pairing.url.startsWith('rulzwriter://pair?')).toBe(true)
      const q = new URL(pairing.url).searchParams
      expect(q.get('host')).toBe('127.0.0.1')
      expect(q.get('token')).toBe(pairing.token)
      expect(pairing.qrSvg).toContain('<svg')
      const old = pairing.token
      const fresh = await access.newToken()
      expect(fresh.token).not.toBe(old)
      expect(settings.phoneToken).toBe(fresh.token)
      expect(statuses).toContain('on')
      expect((await access.setEnabled(false)).state).toBe('off')
    } finally {
      await access.dispose()
    }
  })

  it('says no-tailscale when there is no Tailscale address', async () => {
    let settings: AppSettings = { ...DEFAULT_SETTINGS, phoneEnabled: true, phoneToken: 'x' }
    const access = new PhoneAccess({
      settings: { get: () => settings, update: async (patch) => (settings = parseSettings({ ...settings, ...patch })) },
      host: fakeHost().host,
      emit: () => undefined,
      addresses: () => [],
      dnsName: async () => undefined
    })
    try {
      await access.start()
      expect(access.status()).toMatchObject({ enabled: true, state: 'no-tailscale' })
    } finally {
      await access.dispose()
    }
  })

  it('finds Tailscale addresses', () => {
    expect(isTailscaleIPv4('100.88.114.32')).toBe(true)
    expect(isTailscaleIPv4('100.128.0.1')).toBe(false)
    expect(isTailscaleIPv4('192.168.1.5')).toBe(false)
    const iface = (address: string) => ({ address, family: 'IPv4' as const, internal: false, netmask: '', mac: '', cidr: null })
    expect(tailscaleAddresses({ Ethernet: [iface('192.168.1.5')], Tailscale: [iface('100.88.114.32')], Other: [iface('100.70.0.1')] })).toEqual(['100.88.114.32'])
    expect(tailscaleAddresses({ utun3: [iface('100.70.0.1')], Ethernet: [iface('192.168.1.5')] })).toEqual(['100.70.0.1'])
  })
})

describe('server shutdown and pairing codes', () => {
  it('closes even when a peer holds a plain HTTP connection open', async () => {
    await start()
    const net = await import('node:net')
    const sock = net.connect(server!.port!, '127.0.0.1')
    await new Promise((r) => sock.once('connect', r))
    sock.write('GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n')
    await new Promise((r) => sock.once('data', r))
    const t0 = Date.now()
    await server!.close()
    server = null
    expect(Date.now() - t0).toBeLessThan(2000)
    sock.destroy()
  })

  function access(initial: Partial<AppSettings>) {
    let settings: AppSettings = { ...DEFAULT_SETTINGS, phonePort: 0, ...initial }
    const a = new PhoneAccess({
      settings: { get: () => settings, update: async (patch) => (settings = parseSettings({ ...settings, ...patch })) },
      host: fakeHost().host,
      emit: () => undefined,
      addresses: () => ['127.0.0.1'],
      dnsName: async () => undefined
    })
    return { a, get: () => settings }
  }

  it('makes the first code when it starts enabled without one, and a phone can pair with it', async () => {
    const t = access({ phoneEnabled: true, phoneToken: '' })
    try {
      await t.a.start()
      expect(t.get().phoneToken).toMatch(/^[A-Za-z0-9_-]{43}$/)
      expect(t.a.status().state).toBe('on')
    } finally {
      await t.a.dispose()
    }
  })

  it('gives the server the code pairing() makes, and runs token changes one at a time', async () => {
    const t = access({ phoneEnabled: false, phoneToken: '' })
    try {
      // Enabled with an empty code behind the app's back: pairing must still teach the server the new code.
      await t.a.setEnabled(true)
      const [p1, p2, fresh] = await Promise.all([t.a.pairing(), t.a.pairing(), t.a.newToken()])
      expect(p1.token).toBeTruthy()
      expect(p2.token).toBeTruthy()
      // The last one wins, and settings agree with what the last call returned.
      expect(t.get().phoneToken).toBe(fresh.token)
      expect((await t.a.pairing()).token).toBe(fresh.token)
    } finally {
      await t.a.dispose()
    }
  })
})

