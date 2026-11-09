import { REMOTE_PROTOCOL } from '@shared/defaults'
import type { EventChannel, InvokeChannel, IpcEvents, IpcInvoke, RulzApi } from '@shared/ipc'
import { socketUrl, type Computer } from './computers'

// The phone's end of the connection to RulZ Coding Agent on the computer (see the desktop app's
// src/main/remote/server.ts for the other end and the message format).

export type LinkState =
  | { kind: 'idle' }
  | { kind: 'connecting'; attempt: number; error?: string }
  | { kind: 'connected'; name: string; version: string }
  /** The computer turned the phone away; trying again won't help until the owner fixes it. */
  | { kind: 'denied'; reason: 'wrong-code' | 'locked' | 'timeout' | 'bad-hello' | string }
  /** The phone app and the computer app don't speak the same version. */
  | { kind: 'outdated'; theirs: number; ours: number }

const OPEN_TIMEOUT_MS = 8000
const REQUEST_TIMEOUT_MS = 120_000
const MAX_BACKOFF_MS = 15_000

interface Pending {
  resolve: (v: unknown) => void
  reject: (e: Error) => void
  timer: ReturnType<typeof setTimeout>
}

type SocketLike = Pick<WebSocket, 'send' | 'close' | 'readyState'> & {
  onopen: ((ev: Event) => void) | null
  onmessage: ((ev: MessageEvent) => void) | null
  onclose: ((ev: CloseEvent) => void) | null
  onerror: ((ev: Event) => void) | null
}

export class RemoteLink implements RulzApi {
  state: LinkState = { kind: 'idle' }
  private computer: Computer | null = null
  private device = 'Android phone'
  private ws: SocketLike | null = null
  private attempt = 0
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private nextId = 1
  private readonly pending = new Map<string, Pending>()
  private readonly listeners = new Map<string, Set<(payload: unknown) => void>>()
  private readonly stateListeners = new Set<(s: LinkState) => void>()

  constructor(private readonly makeSocket: (url: string) => SocketLike = (url) => new WebSocket(url)) {}

  get connected(): boolean {
    return this.state.kind === 'connected'
  }

  /** Connects to a computer (dropping any other), and keeps reconnecting until told to stop. */
  connect(computer: Computer, device: string): void {
    this.drop()
    this.computer = computer
    this.device = device
    this.attempt = 0
    this.open()
  }

  disconnect(): void {
    this.drop()
    this.computer = null
    this.setState({ kind: 'idle' })
  }

  /** Tries again right away (the app came back to the front, or the owner pressed Try again). */
  retryNow(): void {
    if (!this.computer || this.connected) return
    if (this.ws && this.state.kind === 'connecting') return
    this.attempt = 0
    this.drop()
    this.open()
  }

  /** After the phone slept, the connection can look open but be dead; a quick request tells. */
  async check(): Promise<void> {
    if (!this.connected) return this.retryNow()
    try {
      await this.request('server:status', [], 5000)
    } catch {
      if (!this.computer) return
      this.drop()
      this.open()
    }
  }

  invoke<K extends InvokeChannel>(channel: K, ...args: Parameters<IpcInvoke[K]>): Promise<Awaited<ReturnType<IpcInvoke[K]>>> {
    return this.request(channel, args, REQUEST_TIMEOUT_MS) as Promise<Awaited<ReturnType<IpcInvoke[K]>>>
  }

  on<K extends EventChannel>(channel: K, listener: (payload: IpcEvents[K]) => void): () => void {
    let set = this.listeners.get(channel)
    if (!set) this.listeners.set(channel, (set = new Set()))
    const l = listener as (payload: unknown) => void
    set.add(l)
    return () => set.delete(l)
  }

  onState(listener: (s: LinkState) => void): () => void {
    this.stateListeners.add(listener)
    return () => this.stateListeners.delete(listener)
  }

  private request(channel: string, args: unknown[], timeoutMs: number): Promise<unknown> {
    const ws = this.ws
    if (!ws || !this.connected) return Promise.reject(new Error(`Not connected to ${this.computer?.name ?? 'your computer'}.`))
    const id = String(this.nextId++)
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error('Your computer took too long to answer.'))
      }, timeoutMs)
      this.pending.set(id, { resolve, reject, timer })
      ws.send(JSON.stringify({ t: 'invoke', id, channel, args }))
    })
  }

  private setState(s: LinkState): void {
    this.state = s
    for (const l of this.stateListeners) l(s)
  }

  private hostFor(attempt: number): string {
    const c = this.computer!
    const hosts = [c.host, c.dns].filter((h): h is string => !!h)
    return hosts[attempt % hosts.length]
  }

  private open(): void {
    const c = this.computer
    if (!c) return
    const attempt = this.attempt
    this.setState({ kind: 'connecting', attempt, error: this.state.kind === 'connecting' ? this.state.error : undefined })
    let ws: SocketLike
    try {
      ws = this.makeSocket(socketUrl(this.hostFor(attempt), c.port))
    } catch (e) {
      this.lost(e instanceof Error ? e.message : String(e))
      return
    }
    this.ws = ws
    let settled = false
    const openTimer = setTimeout(() => ws.close(), OPEN_TIMEOUT_MS)

    ws.onopen = () => {
      ws.send(JSON.stringify({ t: 'hello', token: c.token, device: this.device, protocol: REMOTE_PROTOCOL }))
    }
    ws.onmessage = (ev) => {
      if (this.ws !== ws) return
      let msg: Record<string, unknown>
      try {
        msg = JSON.parse(String(ev.data)) as Record<string, unknown>
      } catch {
        return
      }
      switch (msg.t) {
        case 'welcome':
          clearTimeout(openTimer)
          settled = true
          if (msg.protocol !== REMOTE_PROTOCOL) {
            this.computer = null
            this.drop()
            this.setState({ kind: 'outdated', theirs: Number(msg.protocol), ours: REMOTE_PROTOCOL })
            return
          }
          this.attempt = 0
          this.setState({ kind: 'connected', name: String(msg.name ?? c.name), version: String(msg.version ?? '') })
          break
        case 'denied':
          clearTimeout(openTimer)
          settled = true
          this.computer = null
          this.drop()
          this.setState({ kind: 'denied', reason: String(msg.reason) })
          break
        case 'result': {
          const p = this.pending.get(String(msg.id))
          if (!p) return
          this.pending.delete(String(msg.id))
          clearTimeout(p.timer)
          if (msg.ok) p.resolve(msg.value)
          else p.reject(new Error(String(msg.error ?? 'Something went wrong on your computer.')))
          break
        }
        case 'event':
          for (const l of this.listeners.get(String(msg.channel)) ?? []) l(msg.payload)
          break
      }
    }
    ws.onerror = () => undefined
    ws.onclose = () => {
      clearTimeout(openTimer)
      if (this.ws !== ws) return
      this.ws = null
      this.failPending(`Lost the connection to ${c.name}.`)
      this.lost(settled ? 'The connection dropped.' : `Can't reach ${c.name}.`)
    }
  }

  /** The socket closed on its own: try again later, a little slower each time. */
  private lost(error: string): void {
    if (!this.computer) return
    this.attempt++
    const delay = Math.min(1000 * 2 ** Math.min(this.attempt - 1, 4), MAX_BACKOFF_MS)
    this.setState({ kind: 'connecting', attempt: this.attempt, error })
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.open()
    }, delay)
  }

  /** Closes the socket without retrying. */
  private drop(): void {
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    const ws = this.ws
    this.ws = null
    if (ws) {
      ws.onclose = null
      ws.onmessage = null
      ws.close()
    }
    this.failPending('The connection was closed.')
  }

  private failPending(message: string): void {
    for (const p of this.pending.values()) {
      clearTimeout(p.timer)
      p.reject(new Error(message))
    }
    this.pending.clear()
  }
}

export const link = new RemoteLink()
