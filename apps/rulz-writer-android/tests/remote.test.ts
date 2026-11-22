import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PHONE_DEFAULT_PORT, PHONE_PROTOCOL } from '../src/shared/phone'
import { cleanHost, parsePairingLink, sameComputer, socketUrl } from '../src/lib/computers'
import { RemoteLink, type LinkState } from '../src/lib/remote'

describe('pairing links', () => {
  it('reads the link from the computer’s QR code', () => {
    expect(parsePairingLink('rulzwriter://pair?host=100.88.114.32&port=7421&token=abc_-1&name=jeff-ai')).toEqual({
      name: 'jeff-ai',
      host: '100.88.114.32',
      dns: undefined,
      port: 7421,
      token: 'abc_-1'
    })
  })
  it('uses the default port and the address as the name when they are missing', () => {
    expect(parsePairingLink('rulzwriter://pair?host=100.1.2.3&token=t')).toEqual({ name: '100.1.2.3', host: '100.1.2.3', dns: undefined, port: PHONE_DEFAULT_PORT, token: 't' })
  })
  it('reads an optional MagicDNS name, and falls back to it when the address is missing', () => {
    expect(parsePairingLink('rulzwriter://pair?host=100.1.1.1&port=7421&token=t&dns=pc.tail1.ts.net')).toMatchObject({ host: '100.1.1.1', dns: 'pc.tail1.ts.net', name: 'pc' })
    expect(parsePairingLink('rulzwriter://pair?host=&token=t&dns=pc.tail1.ts.net')).toMatchObject({ host: 'pc.tail1.ts.net', dns: undefined, name: 'pc' })
  })
  it('decodes encoded names', () => {
    expect(parsePairingLink('rulzwriter://pair?host=100.1.1.1&token=t&name=Office%20PC')?.name).toBe('Office PC')
  })
  it('turns down anything else', () => {
    expect(parsePairingLink('https://example.com/?host=1.2.3.4&token=x')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=100.1.1.1&token=x')).toBeNull()
    expect(parsePairingLink('rulzwriter://pair?host=100.1.1.1')).toBeNull()
    expect(parsePairingLink('rulzwriter://pair?host=100.1.1.1&token=x&port=99999')).toBeNull()
    expect(parsePairingLink('rulzwriter://open/project')).toBeNull()
  })
  it('cleans typed addresses', () => {
    expect(cleanHost(' http://100.88.114.32:7421/rulz-writer ')).toBe('100.88.114.32')
    expect(cleanHost('pc.tail1.ts.net')).toBe('pc.tail1.ts.net')
  })
  it('makes the socket address with the writer path', () => {
    expect(socketUrl('100.1.2.3', 7421)).toBe('ws://100.1.2.3:7421/rulz-writer')
    expect(socketUrl('fd7a:115c::1', 7421)).toBe('ws://[fd7a:115c::1]:7421/rulz-writer')
  })
  it('knows the same computer paired again', () => {
    expect(sameComputer({ host: 'PC.ts.net', port: 7421 }, { host: 'pc.ts.net', port: 7421 })).toBe(true)
    expect(sameComputer({ host: 'pc.ts.net', port: 7421 }, { host: 'pc.ts.net', port: 7420 })).toBe(false)
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
  welcome(protocol = PHONE_PROTOCOL): void {
    this.open()
    this.reply({ t: 'welcome', protocol, version: '0.26.0', name: 'jeff-ai' })
  }
}

const computer = { id: '1', name: 'jeff-ai', host: '100.88.114.32', dns: 'jeff-ai.tail1.ts.net', port: 7421, token: 'secret' }

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

  it('says hello with the pairing code and the protocol, and becomes connected on welcome', () => {
    link.connect(computer, 'Pixel 9')
    expect(last().url).toBe('ws://100.88.114.32:7421/rulz-writer')
    last().open()
    expect(last().sent).toEqual([{ t: 'hello', token: 'secret', device: 'Pixel 9', protocol: PHONE_PROTOCOL }])
    last().reply({ t: 'welcome', protocol: PHONE_PROTOCOL, version: '0.26.0', name: 'jeff-ai' })
    expect(link.state).toEqual({ kind: 'connected', name: 'jeff-ai', version: '0.26.0' })
  })

  it('sends invokes as { t, id, channel, args } and matches results to requests, including errors', async () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    const a = link.invoke('novel:listProjects')
    const b = link.invoke('writing:abort', 'job-1')
    const [ra, rb] = last().sent.slice(1) as { id: string; channel: string; args: unknown[] }[]
    expect(ra).toMatchObject({ t: 'invoke', channel: 'novel:listProjects', args: [] })
    expect(rb).toMatchObject({ t: 'invoke', channel: 'writing:abort', args: ['job-1'] })
    last().reply({ t: 'result', id: rb.id, ok: false, error: 'That job is not running.' })
    last().reply({ t: 'result', id: ra.id, ok: true, value: [] })
    await expect(a).resolves.toEqual([])
    await expect(b).rejects.toThrow('That job is not running.')
  })

  it('passes events to listeners until they unsubscribe', () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    const got: unknown[] = []
    const off = link.on('novel:changed', (c) => got.push(c))
    const change = { kind: 'text', folder: 'f', path: 'manuscript/a.md', hash: 'h1' }
    last().reply({ t: 'event', channel: 'novel:changed', payload: change })
    off()
    last().reply({ t: 'event', channel: 'novel:changed', payload: { ...change, hash: 'h2' } })
    expect(got).toEqual([change])
  })

  it('refuses requests while not connected', async () => {
    link.connect(computer, 'Pixel')
    await expect(link.invoke('novel:listProjects')).rejects.toThrow('Not connected to jeff-ai.')
  })

  it('fails waiting requests and reconnects, alternating with the MagicDNS name, when the connection drops', async () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    const pending = link.invoke('novel:listProjects')
    last().close()
    await expect(pending).rejects.toThrow('Lost the connection to jeff-ai.')
    expect(link.state).toMatchObject({ kind: 'connecting', attempt: 1 })
    vi.advanceTimersByTime(1000)
    expect(last().url).toBe('ws://jeff-ai.tail1.ts.net:7421/rulz-writer')
    last().close()
    vi.advanceTimersByTime(1999)
    expect(FakeSocket.all).toHaveLength(2)
    vi.advanceTimersByTime(1)
    expect(last().url).toBe('ws://100.88.114.32:7421/rulz-writer')
    last().welcome()
    expect(link.state.kind).toBe('connected')
  })

  it('gives up on a socket that never opens', () => {
    link.connect(computer, 'Pixel')
    vi.advanceTimersByTime(8000)
    expect(link.state).toMatchObject({ kind: 'connecting', attempt: 1, error: "Can't reach jeff-ai." })
  })

  it('stops retrying when the computer turns the phone away (wrong pairing code)', () => {
    link.connect(computer, 'Pixel')
    last().open()
    last().reply({ t: 'denied', reason: 'wrong-code' })
    expect(link.state).toEqual({ kind: 'denied', reason: 'wrong-code' })
    vi.advanceTimersByTime(60_000)
    expect(FakeSocket.all).toHaveLength(1)
  })

  it('does not retry after a denial when the app comes back to the front, but does when asked', async () => {
    link.connect(computer, 'Pixel')
    last().open()
    last().reply({ t: 'denied', reason: 'locked' })
    await link.check()
    expect(FakeSocket.all).toHaveLength(1)
    link.retryNow()
    expect(FakeSocket.all).toHaveLength(2)
  })

  it('stops when the two apps speak different versions', () => {
    link.connect(computer, 'Pixel')
    last().welcome(PHONE_PROTOCOL + 1)
    expect(link.state).toEqual({ kind: 'outdated', theirs: PHONE_PROTOCOL + 1, ours: PHONE_PROTOCOL })
  })

  it('reconnects right away when asked, skipping the wait', () => {
    link.connect(computer, 'Pixel')
    last().close()
    vi.advanceTimersByTime(1000)
    last().close()
    expect(FakeSocket.all).toHaveLength(2)
    link.retryNow()
    expect(FakeSocket.all).toHaveLength(3)
    expect(last().url).toBe('ws://100.88.114.32:7421/rulz-writer')
  })

  it.each(['timeout', 'bad-hello'])('retries transient denial %s', (reason) => {
    link.connect(computer, 'Pixel')
    last().open()
    last().reply({ t: 'denied', reason })
    expect(link.state.kind).toBe('connecting')
    vi.advanceTimersByTime(1000)
    expect(FakeSocket.all).toHaveLength(2)
  })

  it('checks a connected socket and reconnects when the heartbeat times out', async () => {
    link.connect(computer, 'Pixel')
    last().welcome()
    await vi.advanceTimersByTimeAsync(40_000)
    expect(last().sent.at(-1)).toMatchObject({ channel: 'novel:listProjects' })
    await vi.advanceTimersByTimeAsync(10_000)
    expect(FakeSocket.all).toHaveLength(2)
    expect(link.state.kind).toBe('connecting')
  })

  it('backs off up to 15 seconds in the foreground', () => {
    link.connect(computer, 'Pixel')
    for (const delay of [1000, 2000, 4000, 8000, 15000, 15000]) {
      last().close()
      const count = FakeSocket.all.length
      vi.advanceTimersByTime(delay - 1)
      expect(FakeSocket.all).toHaveLength(count)
      vi.advanceTimersByTime(1)
      expect(FakeSocket.all).toHaveLength(count + 1)
    }
  })

  it('uses a longer retry delay while idle in the background', () => {
    link.backgroundIdle = true
    link.connect(computer, 'Pixel')
    for (const delay of [1000, 2000, 4000, 8000, 16000, 32000, 64000]) {
      last().close()
      vi.advanceTimersByTime(delay)
    }
    last().close()
    const count = FakeSocket.all.length
    vi.advanceTimersByTime(89_999)
    expect(FakeSocket.all).toHaveLength(count)
    vi.advanceTimersByTime(1)
    expect(FakeSocket.all).toHaveLength(count + 1)
  })
})
