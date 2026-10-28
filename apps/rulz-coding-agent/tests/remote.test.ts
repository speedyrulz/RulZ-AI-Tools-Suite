import { afterEach, describe, expect, it } from 'vitest'
import { WebSocket } from 'ws'
import { DEFAULT_SETTINGS, REMOTE_PROTOCOL } from '../src/shared/defaults'
import type { EventChannel } from '../src/shared/ipc'
import { REMOTE_PATH, RemoteServer, settingsPatchFromPhone, type RemoteHost } from '../src/main/remote/server'
import { isTailscaleIPv4, tailscaleAddresses } from '../src/main/remote/tailscale'

type Msg = Record<string, unknown>

function fakeHost() {
  const listeners = new Set<(c: EventChannel, p: unknown) => void>()
  const calls: { channel: string; args: unknown[] }[] = []
  const host: RemoteHost = {
    version: '9.9.9',
    name: 'test-pc',
    subscribe: (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    invoke: async (channel, args) => {
      calls.push({ channel, args })
      if (channel === 'settings:get' || channel === 'settings:update') return { ...DEFAULT_SETTINGS, remoteToken: 'secret', ...(args[0] as object) }
      if (channel === 'boom') throw new Error('It broke.')
      return { echo: args }
    }
  }
  const emit = (c: EventChannel, p: unknown): void => listeners.forEach((l) => l(c, p))
  return { host, calls, emit }
}

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
    const ws = new WebSocket(`ws://127.0.0.1:${port}${REMOTE_PATH}`)
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

let server: RemoteServer | null = null
const phones: Phone[] = []
afterEach(async () => {
  phones.splice(0).forEach((p) => p.ws.terminate())
  await server?.close()
  server = null
})

async function start(token = 'right-code') {
  const fake = fakeHost()
  const clients: number[] = []
  server = new RemoteServer(fake.host, (c) => clients.push(c.length))
  await server.listen(['127.0.0.1'], 0, token)
  const connect = async (): Promise<Phone> => {
    const p = await Phone.connect(server!.port!)
    phones.push(p)
    return p
  }
  const pair = async (code = token): Promise<Phone> => {
    const p = await connect()
    p.send({ t: 'hello', token: code, device: 'Pixel', protocol: REMOTE_PROTOCOL })
    return p
  }
  return { ...fake, clients, connect, pair }
}

describe('RemoteServer', () => {
  it('answers a plain request with only who it is', async () => {
    await start()
    const res = await fetch(`http://127.0.0.1:${server!.port}/`)
    expect(await res.json()).toEqual({ app: 'rulz-coding-agent', protocol: REMOTE_PROTOCOL })
  })

  it('welcomes a phone with the right pairing code and lists it', async () => {
    const t = await start()
    const p = await t.pair()
    expect(await p.next()).toEqual({ t: 'welcome', protocol: REMOTE_PROTOCOL, version: '9.9.9', name: 'test-pc' })
    expect(server!.clients()).toMatchObject([{ device: 'Pixel', address: '127.0.0.1' }])
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
    p.send({ t: 'invoke', id: '1', channel: 'projects:recent', args: [] })
    expect(await p.next()).toEqual({ t: 'denied', reason: 'bad-hello' })
    expect(t.calls).toEqual([])
  })

  it('locks an address out after ten wrong codes', async () => {
    const t = await start()
    for (let i = 0; i < 10; i++) {
      const p = await t.pair('wrong')
      await p.closed
    }
    const p = await t.pair()
    expect(await p.next()).toEqual({ t: 'denied', reason: 'locked' })
  })

  it('runs requests and returns results and errors', async () => {
    const t = await start()
    const p = await t.pair()
    await p.next()
    expect(await p.invoke('projects:open', 'C:\\x')).toMatchObject({ ok: true, value: { echo: ['C:\\x'] } })
    expect(await p.invoke('boom')).toMatchObject({ ok: false, error: 'It broke.' })
  })

  it("refuses requests that only make sense on the computer, including phone access itself", async () => {
    const t = await start()
    const p = await t.pair()
    await p.next()
    for (const ch of ['dialog:pickFolder', 'shell:showItem', 'shell:openUrl', 'remote:newToken', 'remote:setEnabled']) {
      expect(await p.invoke(ch, true)).toMatchObject({ ok: false, error: 'That can only be done on the computer.' })
    }
    expect(t.calls).toEqual([])
  })

  it('never sends the pairing code and never lets the phone change phone access', async () => {
    const t = await start()
    const p = await t.pair()
    await p.next()
    const got = await p.invoke('settings:get')
    expect((got.value as { remoteToken: string }).remoteToken).toBe('')
    await p.invoke('settings:update', { approval: 'careful', remoteEnabled: false, remoteToken: 'mine', remotePort: 80 })
    expect(t.calls.at(-1)).toEqual({ channel: 'settings:update', args: [{ approval: 'careful' }] })
  })

  it('passes events on to paired phones, except phone-access status', async () => {
    const t = await start()
    const paired = await t.pair()
    await paired.next()
    const unpaired = await t.connect()
    t.emit('remote:changed', {})
    t.emit('model:activity', 'working')
    expect(await paired.next()).toEqual({ t: 'event', channel: 'model:activity', payload: 'working' })
    expect(unpaired.messages).toEqual([])
  })

  it('disconnects phones when the pairing code changes', async () => {
    const t = await start()
    const p = await t.pair()
    await p.next()
    server!.setToken('new-code')
    expect(await p.closed).toBe(4001)
    const again = await t.pair('right-code')
    expect(await again.next()).toEqual({ t: 'denied', reason: 'wrong-code' })
    const fresh = await t.pair('new-code')
    expect(await fresh.next()).toMatchObject({ t: 'welcome' })
  })
})

describe('phone access helpers', () => {
  it('drops phone-access settings from changes the phone sends', () => {
    expect(settingsPatchFromPhone({ theme: 'light', remoteEnabled: true, remoteToken: 'x' })).toEqual({ theme: 'light' })
    expect(settingsPatchFromPhone(null)).toEqual({})
  })

  it('finds Tailscale addresses', () => {
    expect(isTailscaleIPv4('100.88.114.32')).toBe(true)
    expect(isTailscaleIPv4('100.128.0.1')).toBe(false)
    expect(isTailscaleIPv4('192.168.1.5')).toBe(false)
    const iface = (address: string) => ({ address, family: 'IPv4' as const, internal: false, netmask: '', mac: '', cidr: null })
    expect(tailscaleAddresses({ Ethernet: [iface('192.168.1.5')], Tailscale: [iface('100.88.114.32')], Other: [iface('100.70.0.1')] })).toEqual(['100.88.114.32'])
    // An adapter with another name still counts when no adapter is called Tailscale.
    expect(tailscaleAddresses({ utun3: [iface('100.70.0.1')], Ethernet: [iface('192.168.1.5')] })).toEqual(['100.70.0.1'])
  })
})
