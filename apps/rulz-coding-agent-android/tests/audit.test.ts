import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { REMOTE_PROTOCOL } from '../src/shared/defaults'
import { parsePairingLink, validHost } from '../src/lib/computers'
import { RemoteLink, requestTimeoutFor } from '../src/lib/remote'
import { ARM_MS, createArming } from '../src/components/ApprovalSheet'
import { draftOf, useComposer } from '../src/stores/composer'

// ---- the store test needs the connection's calls and events under its control ----
type Handler = (payload: unknown) => void
const { handlers, invokeMock } = vi.hoisted(() => ({ handlers: new Map<string, (payload: unknown) => void>(), invokeMock: vi.fn() }))
vi.mock('../src/lib/api', () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
  on: (channel: string, h: Handler) => {
    handlers.set(channel, h)
    return () => undefined
  },
  errorMessage: (e: unknown) => (e instanceof Error ? e.message : String(e))
}))
import { useWorkspace } from '../src/stores/workspace'

class FakeSocket {
  static all: FakeSocket[] = []
  readyState = 0
  sent: Record<string, unknown>[] = []
  throwOnSend = false
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  constructor(readonly url: string) {
    FakeSocket.all.push(this)
  }
  send(data: string): void {
    if (this.throwOnSend) throw new Error('socket is closing')
    const msg = JSON.parse(data) as { t: string; id: string; channel: string }
    this.sent.push(msg)
    // The computer answers the phone's keep-alive check.
    if (msg.channel === 'server:status') this.reply({ t: 'result', id: msg.id, ok: true, value: {} })
  }
  close(): void {
    this.readyState = 3
    this.onclose?.({} as CloseEvent)
  }
  welcome(): void {
    this.readyState = 1
    this.onopen?.({} as Event)
    this.reply({ t: 'welcome', protocol: REMOTE_PROTOCOL, version: '0.6.0', name: 'jeff-ai' })
  }
  reply(msg: object): void {
    this.onmessage?.({ data: JSON.stringify(msg) } as MessageEvent)
  }
}

const computer = { id: '1', name: 'jeff-ai', host: '100.88.114.32', dns: 'jeff-ai.tail1.ts.net', port: 7420, token: 'secret' }

describe('A1: approval buttons arm after 600 ms', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('is not armed right away, and is after ARM_MS', () => {
    expect(ARM_MS).toBe(600)
    const states: boolean[] = []
    const a = createArming((v) => states.push(v))
    a.arm()
    expect(a.isArmed()).toBe(false)
    vi.advanceTimersByTime(599)
    expect(a.isArmed()).toBe(false)
    vi.advanceTimersByTime(1)
    expect(a.isArmed()).toBe(true)
    expect(states).toEqual([false, true])
  })

  it('starts over for the next prompt', () => {
    const a = createArming(() => undefined)
    a.arm()
    vi.advanceTimersByTime(700)
    expect(a.isArmed()).toBe(true)
    a.arm()
    expect(a.isArmed()).toBe(false)
    vi.advanceTimersByTime(600)
    expect(a.isArmed()).toBe(true)
  })

  it('cancel stops a pending arming', () => {
    const a = createArming(() => undefined)
    a.arm()
    a.cancel()
    vi.advanceTimersByTime(1000)
    expect(a.isArmed()).toBe(false)
  })
})

describe('A2, A9: RemoteLink timeouts, failed sends and throwing listeners', () => {
  let link: RemoteLink
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.all = []
    link = new RemoteLink((url) => new FakeSocket(url) as never)
    link.connect(computer, 'Pixel')
    FakeSocket.all[0].welcome()
  })
  afterEach(() => {
    link.disconnect()
    vi.useRealTimers()
  })

  it('picks the timeout per channel', () => {
    expect(requestTimeoutFor('projects:recent')).toBe(120_000)
    for (const c of ['models:reloadMode', 'models:setLoadSettings', 'pdf:uploadEnd']) expect(requestTimeoutFor(c)).toBe(600_000)
  })

  it('gives an ordinary request 120 s', async () => {
    const p = link.invoke('projects:recent')
    const seen = expect(p).rejects.toThrow('took too long')
    vi.advanceTimersByTime(120_000)
    await seen
  })

  it('lets a model reload run for 10 minutes', async () => {
    let settled = false
    const p = link.invoke('models:reloadMode', 'some-model', 'coding')
    p.then(() => (settled = true), () => (settled = true))
    vi.advanceTimersByTime(120_001)
    await Promise.resolve()
    expect(settled).toBe(false)
    const id = (FakeSocket.all[0].sent[1] as { id: string }).id
    FakeSocket.all[0].reply({ t: 'result', id, ok: true, value: { ok: 1 } })
    await expect(p).resolves.toEqual({ ok: 1 })
  })

  it('gives up on a model reload after 10 minutes', async () => {
    const p = link.invoke('models:setLoadSettings', 'm', null)
    const seen = expect(p).rejects.toThrow('took too long')
    vi.advanceTimersByTime(600_000)
    await seen
  })

  it('rejects and cleans up when the send throws', async () => {
    FakeSocket.all[0].throwOnSend = true
    await expect(link.invoke('projects:recent')).rejects.toThrow('socket is closing')
    expect((link as unknown as { pending: Map<string, unknown> }).pending.size).toBe(0)
    // No timer is left to reject it a second time.
    expect(vi.getTimerCount()).toBeLessThanOrEqual(2)
  })

  it('calls every listener even when one throws', () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const got: unknown[] = []
    link.on('model:activity', () => {
      throw new Error('boom')
    })
    link.on('model:activity', (a) => got.push(a))
    FakeSocket.all[0].reply({ t: 'event', channel: 'model:activity', payload: 'working' })
    expect(got).toEqual(['working'])
    expect(err).toHaveBeenCalled()
    err.mockRestore()
  })
})

describe('A13: pairing link hosts', () => {
  it('refuses a host that would change where the socket points', () => {
    expect(parsePairingLink('rulzagent://pair?host=1.2.3.4@evil&token=t')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=evil.com/x&token=t')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=a b&token=t')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=[::1]&token=t')).toBeNull()
    expect(parsePairingLink('rulzagent://pair?host=100.1.1.1&token=t&dns=pc@evil')).toBeNull()
  })
  it('accepts hostnames, IPv4 and IPv6', () => {
    expect(validHost('100.88.114.32')).toBe(true)
    expect(validHost('jeff-ai.tail1.ts.net')).toBe(true)
    expect(validHost('fd7a:115c:a1e0::1')).toBe(true)
    expect(parsePairingLink('rulzagent://pair?host=fd7a:115c:a1e0::1&token=t')).toMatchObject({ host: 'fd7a:115c:a1e0::1' })
  })
})

describe('A3: the composer draft survives', () => {
  it('is kept per project and cleared when emptied', () => {
    useComposer.getState().setDraft('C:\\a', 'hello')
    useComposer.getState().setDraft('C:\\b', 'other')
    expect(draftOf('c:\\A')).toBe('hello')
    expect(draftOf('C:\\b')).toBe('other')
    useComposer.getState().setDraft('C:\\a', '')
    expect(draftOf('C:\\a')).toBe('')
    expect(Object.keys(useComposer.getState().drafts)).toEqual(['c:\\b'])
  })
})

describe('AP1, A7: the workspace store', () => {
  const project = { name: 'p', path: 'C:\\p' }
  const step = { id: 's1', taskId: 't1', output: '' }
  const task = { id: 't1', steps: [step], files: [] }
  beforeEach(() => {
    vi.useFakeTimers()
    invokeMock.mockReset()
    useWorkspace.setState({ project: project as never, tasks: [task] as never, timeline: [], run: null })
  })
  afterEach(() => vi.useRealTimers())

  const output = (chunk: string): void => handlers.get('agent:event')!({ type: 'output', projectPath: 'C:\\p', stepId: 's1', chunk })

  it('applies 50 output chunks in one update, with the same text', () => {
    let updates = 0
    const off = useWorkspace.subscribe(() => updates++)
    for (let i = 0; i < 50; i++) output(`line ${i}\n`)
    expect(updates).toBe(0)
    vi.advanceTimersByTime(50)
    expect(updates).toBe(1)
    off()
    const expected = Array.from({ length: 50 }, (_, i) => `line ${i}\n`).join('')
    expect(useWorkspace.getState().tasks[0].steps[0].output).toBe(expected)
  })

  it('applies pending output before the next other event', () => {
    output('a')
    output('b')
    handlers.get('agent:event')!({ type: 'status', projectPath: 'C:\\p', status: { phase: 'working', text: 'x' } })
    expect(useWorkspace.getState().tasks[0].steps[0].output).toBe('ab')
  })

  it('does not put a run result from an earlier project into the open one', async () => {
    let finish: (v: unknown) => void = () => undefined
    invokeMock.mockReturnValue(new Promise((r) => (finish = r)))
    const started = useWorkspace.getState().startRun()
    useWorkspace.setState({ project: { name: 'q', path: 'C:\\q' } as never, run: null })
    finish({ running: true, output: 'from p' })
    await started
    expect(useWorkspace.getState().run).toBeNull()
  })
})
