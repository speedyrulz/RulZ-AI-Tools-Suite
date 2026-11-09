import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { REMOTE_PROTOCOL } from '../src/shared/defaults'
import { cleanHost, parsePairingLink, socketUrl, urlOnComputer } from '../src/lib/computers'
import { RemoteLink, type LinkState } from '../src/lib/remote'

describe('pairing links', () => {
  it('reads the link from the computer’s QR code', () => {
    expect(parsePairingLink('rulzagent://pair?host=100.88.114.32&port=7420&token=abc_-1&name=jeff-ai&dns=jeff-ai.tail1.ts.net')).toEqual({
      name: 'jeff-ai',
      host: '100.88.114.32',
      dns: 'jeff-ai.tail1.ts.net',
      port: 7420,
      token: 'abc_-1'
    })
  })
  it('falls back to the MagicDNS name when the address is missing', () => {
    expect(parsePairingLink('rulzagent://pair?host=&token=t&dns=pc.tail1.ts.net')).toMatchObject({ host: 'pc.tail1.ts.net', dns: undefined, name: 'pc', port: 7420 })
  })
  it('turns down anything else', () => {
    expect(parsePairingLink('https://example.com/?host=1.2.3.4&token=x')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=100.1.1.1')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=100.1.1.1&token=x&port=99999')).toBeNull()
  })
  it('cleans typed addresses', () => {
    expect(cleanHost(' http://100.88.114.32:7420/rulz ')).toBe('100.88.114.32')
    expect(cleanHost('pc.tail1.ts.net')).toBe('pc.tail1.ts.net')
  })
  it('makes socket and app addresses', () => {
    expect(socketUrl('100.1.2.3', 7420)).toBe('ws://100.1.2.3:7420/rulz')
    expect(socketUrl('fd7a:115c::1', 7420)).toBe('ws://[fd7a:115c::1]:7420/rulz')
    expect(urlOnComputer('http://localhost:5173/', '100.1.2.3')).toBe('http://100.1.2.3:5173/')
    expect(urlOnComputer('http://127.0.0.1:3000/app?x=1', '100.1.2.3')).toBe('http://100.1.2.3:3000/app?x=1')
    expect(urlOnComputer('https://example.com/', '100.1.2.3')).toBe('https://example.com/')
  })
})

/** Stands in for the browser's WebSocket; the test plays the computer. */
class FakeSocket {
  static all: FakeSocket[] = []
  readyState = 0
  sent: Record<string, unknown>[] = []
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  constructor(readonly url: string) {
    FakeSocket.all.push(this)
  }
  send(data: string): void {
    this.sent.push(JSON.parse(data))
  }
  close(): void {
    this.readyState = 3
    this.onclose?.({} as CloseEvent)
  }
  // ---- the computer's side ----
  open(): void {
    this.readyState = 1
    this.onopen?.({} as Event)
  }
  reply(msg: object): void {
    this.onmessage?.({ data: JSON.stringify(msg) } as MessageEvent)
  }
  welcome(protocol = REMOTE_PROTOCOL): void {
    this.open()
    this.reply({ t: 'welcome', protocol, version: '0.6.0', name: 'jeff-ai' })
  }
}

const computer = { id: '1', name: 'jeff-ai', host: '100.88.114.32', dns: 'jeff-ai.tail1.ts.net', port: 7420, token: 'secret' }

describe('RemoteLink', () => {
  let link: RemoteLink
  let states: LinkState[]
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.all = []
    link = new RemoteLink((url) => new FakeSocket(url) as never)
    states = []
    link.onState((s) => states.push(s))
  })
  afterEach(() => {
    link.disconnect()
    vi.useRealTimers()
  })
  const last = (): FakeSocket => FakeSocket.all[FakeSocket.all.length - 1]

  it('says hello with the pairing code and becomes connected on welcome', () => {
    link.connect(computer, 'Pixel 9')
    expect(last().url).toBe('ws://100.88.114.32:7420/rulz')
    last().open()
    expect(last().sent).toEqual([{ t: 'hello', token: 'secret', device: 'Pixel 9', protocol: REMOTE_PROTOCOL }])
    last().reply({ t: 'welcome', protocol: REMOTE_PROTOCOL, version: '0.6.0', name: 'jeff-ai' })
    expect(link.state).toEqual({ kind: 'connected', name: 'jeff-ai', version: '0.6.0' })
  })

  it('matches results to requests, including errors', async () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    const a = link.invoke('projects:recent')
    const b = link.invoke('task:stop', 'C:\\p')
    const [ra, rb] = last().sent.slice(1) as { id: string; channel: string; args: unknown[] }[]
    expect(rb).toMatchObject({ t: 'invoke', channel: 'task:stop', args: ['C:\\p'] })
    last().reply({ t: 'result', id: rb.id, ok: false, error: 'That project is not open.' })
    last().reply({ t: 'result', id: ra.id, ok: true, value: [] })
    await expect(a).resolves.toEqual([])
    await expect(b).rejects.toThrow('That project is not open.')
  })

  it('passes events to listeners', () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    const got: unknown[] = []
    const off = link.on('model:activity', (a) => got.push(a))
    last().reply({ t: 'event', channel: 'model:activity', payload: 'working' })
    off()
    last().reply({ t: 'event', channel: 'model:activity', payload: 'idle' })
    expect(got).toEqual(['working'])
  })

  it('refuses requests while not connected', async () => {
    link.connect(computer, 'Pixel')
    await expect(link.invoke('projects:recent')).rejects.toThrow('Not connected to jeff-ai.')
  })

  it('fails waiting requests and reconnects, alternating with the MagicDNS name, when the connection drops', async () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    const pending = link.invoke('projects:recent')
    last().close()
    await expect(pending).rejects.toThrow('Lost the connection to jeff-ai.')
    expect(link.state).toMatchObject({ kind: 'connecting', attempt: 1 })
    vi.advanceTimersByTime(1000)
    expect(last().url).toBe('ws://jeff-ai.tail1.ts.net:7420/rulz')
    last().close()
    vi.advanceTimersByTime(1999)
    expect(FakeSocket.all).toHaveLength(2)
    vi.advanceTimersByTime(1)
    expect(last().url).toBe('ws://100.88.114.32:7420/rulz')
    last().welcome()
    expect(link.state.kind).toBe('connected')
  })

  it('gives up on a socket that never opens', () => {
    link.connect(computer, 'Pixel')
    vi.advanceTimersByTime(8000)
    expect(link.state).toMatchObject({ kind: 'connecting', attempt: 1, error: "Can't reach jeff-ai." })
  })

  it('stops retrying when the computer turns the phone away', () => {
    link.connect(computer, 'Pixel')
    last().open()
    last().reply({ t: 'denied', reason: 'wrong-code' })
    expect(link.state).toEqual({ kind: 'denied', reason: 'wrong-code' })
    vi.advanceTimersByTime(60_000)
    expect(FakeSocket.all).toHaveLength(1)
  })

  it('stops when the two apps speak different versions', () => {
    link.connect(computer, 'Pixel')
    last().welcome(REMOTE_PROTOCOL + 1)
    expect(link.state).toEqual({ kind: 'outdated', theirs: REMOTE_PROTOCOL + 1, ours: REMOTE_PROTOCOL })
  })

  it('reconnects right away when asked, skipping the wait', () => {
    link.connect(computer, 'Pixel')
    last().close()
    vi.advanceTimersByTime(1000)
    last().close()
    expect(FakeSocket.all).toHaveLength(2)
    link.retryNow()
    expect(FakeSocket.all).toHaveLength(3)
    expect(last().url).toBe('ws://100.88.114.32:7420/rulz')
  })
})
