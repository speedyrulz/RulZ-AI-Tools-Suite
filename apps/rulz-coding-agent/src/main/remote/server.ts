import { createHash, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
import { createServer, type IncomingMessage, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import type { Duplex } from 'node:stream'
import { WebSocket, WebSocketServer, type RawData } from 'ws'
import { REMOTE_PROTOCOL } from '@shared/defaults'
import type { EventChannel } from '@shared/ipc'
import type { AppSettings, RemoteClient } from '@shared/types'

// The phone app talks to the same handlers as this computer's window, over a WebSocket on the
// Tailscale address only. Tailscale encrypts the traffic; the pairing code proves it is the owner's phone.
//
// Phone → computer:  { t: 'hello', token, device, protocol }   then   { t: 'invoke', id, channel, args }
// Computer → phone:  { t: 'welcome', protocol, version, name }  { t: 'result', id, ok, value | error }
//                    { t: 'event', channel, payload }           { t: 'denied', reason }

export const REMOTE_PATH = '/rulz'

/** Requests that only make sense on this computer (dialogs, its own browser and Explorer, phone access itself). */
const COMPUTER_ONLY = new Set(['dialog:pickFolder', 'shell:showItem', 'shell:openUrl'])
const isComputerOnly = (channel: string): boolean => COMPUTER_ONLY.has(channel) || channel.startsWith('remote:')

/** Events the phone doesn't get. */
const NOT_SENT = new Set<EventChannel>(['remote:changed'])

const HELLO_TIMEOUT_MS = 10_000
const PING_MS = 25_000
/** A phone this far behind (slow network) is dropped; it reconnects and reloads. */
const MAX_BUFFERED = 16 * 1024 * 1024
const MAX_FAILURES = 10
const LOCKOUT_MS = 10 * 60_000

export interface RemoteHost {
  invoke(channel: string, args: unknown[]): Promise<unknown>
  subscribe(listener: (channel: EventChannel, payload: unknown) => void): () => void
  version: string
  /** This computer's name, shown on the phone. */
  name: string
}

interface Conn {
  info: RemoteClient
  ws: WebSocket
  alive: boolean
}

export function newToken(): string {
  return randomBytes(24).toString('base64url')
}

const digest = (s: string): Buffer => createHash('sha256').update(s).digest()
export function tokenMatches(given: unknown, expected: string): boolean {
  return typeof given === 'string' && !!expected && timingSafeEqual(digest(given), digest(expected))
}

/** Settings as the phone sees them: no pairing code, and no RulZ LLM Suite API key (the phone never needs it). */
export function settingsForPhone(s: AppSettings): AppSettings {
  return { ...s, remoteToken: '', apiKey: '' }
}

/**
 * A settings change from the phone, without the phone-access settings and the API key (only this
 * computer changes those; the phone sees the key as empty, so it must not be able to send it back).
 */
export function settingsPatchFromPhone(patch: unknown): Partial<AppSettings> {
  if (!patch || typeof patch !== 'object') return {}
  return Object.fromEntries(Object.entries(patch).filter(([k]) => !k.startsWith('remote') && k !== 'apiKey')) as Partial<AppSettings>
}

export class RemoteServer {
  private servers: Server[] = []
  private readonly wss = new WebSocketServer({ noServer: true, maxPayload: 4 * 1024 * 1024 })
  private readonly conns = new Map<string, Conn>()
  private readonly failures = new Map<string, { count: number; until: number }>()
  private token = ''
  private unsubscribe: (() => void) | null = null
  private pinger: NodeJS.Timeout | null = null

  constructor(
    private readonly host: RemoteHost,
    private readonly onClientsChanged: (clients: RemoteClient[]) => void
  ) {}

  get listening(): boolean {
    return this.servers.length > 0
  }

  /** The port actually listened on (differs from the one asked for only when that was 0, in tests). */
  get port(): number | undefined {
    return (this.servers[0]?.address() as AddressInfo | null)?.port
  }

  clients(): RemoteClient[] {
    return [...this.conns.values()].map((c) => c.info)
  }

  /** Listens on each address (all must work), replacing any earlier listening. */
  async listen(addresses: string[], port: number, token: string): Promise<void> {
    await this.close()
    this.token = token
    try {
      for (const address of addresses) this.servers.push(await this.listenOn(address, port))
    } catch (e) {
      await this.close()
      const code = (e as NodeJS.ErrnoException).code
      if (code === 'EADDRINUSE') throw new Error(`Port ${port} is already used by another program. Choose another port.`)
      if (code === 'EADDRNOTAVAIL') throw new Error('The Tailscale address is not available. Is Tailscale connected?')
      throw e
    }
    this.unsubscribe = this.host.subscribe((channel, payload) => this.send(channel, payload))
    this.pinger = setInterval(() => this.ping(), PING_MS)
  }

  /** Phones must use the new code from now on; those connected with the old one are disconnected. */
  setToken(token: string): void {
    this.token = token
    for (const c of this.conns.values()) c.ws.close(4001, 'The pairing code changed.')
  }

  async close(): Promise<void> {
    this.unsubscribe?.()
    this.unsubscribe = null
    if (this.pinger) clearInterval(this.pinger)
    this.pinger = null
    for (const c of this.conns.values()) c.ws.terminate()
    this.conns.clear()
    const servers = this.servers
    this.servers = []
    await Promise.all(servers.map((s) => new Promise<void>((r) => s.close(() => r()))))
  }

  private listenOn(address: string, port: number): Promise<Server> {
    const server = createServer((req, res) => {
      // A plain request answers "is this RulZ Coding Agent?" and nothing else.
      if (req.method === 'GET' && (req.url === '/' || req.url === REMOTE_PATH)) {
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ app: 'rulz-coding-agent', protocol: REMOTE_PROTOCOL }))
      } else {
        res.writeHead(404).end()
      }
    })
    server.on('upgrade', (req, socket, head) => this.upgrade(req, socket, head))
    return new Promise((resolve, reject) => {
      server.once('error', reject)
      server.listen(port, address, () => {
        server.off('error', reject)
        resolve(server)
      })
    })
  }

  private upgrade(req: IncomingMessage, socket: Duplex, head: Buffer): void {
    const path = (req.url ?? '').split('?')[0]
    if (path !== REMOTE_PATH) {
      socket.end('HTTP/1.1 404 Not Found\r\n\r\n')
      return
    }
    this.wss.handleUpgrade(req, socket, head, (ws) => this.accept(ws, req.socket.remoteAddress ?? ''))
  }

  private lockedOut(address: string): boolean {
    const f = this.failures.get(address)
    if (!f) return false
    if (f.until && f.until < Date.now()) {
      this.failures.delete(address)
      return false
    }
    return f.count >= MAX_FAILURES
  }

  private failed(address: string): void {
    const f = this.failures.get(address) ?? { count: 0, until: 0 }
    f.count++
    if (f.count >= MAX_FAILURES) f.until = Date.now() + LOCKOUT_MS
    this.failures.set(address, f)
  }

  private accept(ws: WebSocket, address: string): void {
    const deny = (reason: 'wrong-code' | 'locked' | 'timeout' | 'bad-hello'): void => {
      ws.send(JSON.stringify({ t: 'denied', reason }))
      ws.close(4001, reason)
    }
    if (this.lockedOut(address)) return deny('locked')
    const timer = setTimeout(() => deny('timeout'), HELLO_TIMEOUT_MS)

    ws.once('message', (data) => {
      clearTimeout(timer)
      const msg = parse(data)
      if (msg?.t !== 'hello') return deny('bad-hello')
      if (!tokenMatches(msg.token, this.token)) {
        this.failed(address)
        return deny(this.lockedOut(address) ? 'locked' : 'wrong-code')
      }
      this.failures.delete(address)
      const device = typeof msg.device === 'string' && msg.device.trim() ? msg.device.trim().slice(0, 60) : 'Phone'
      const conn: Conn = { info: { id: randomUUID(), device, address, since: Date.now() }, ws, alive: true }
      this.conns.set(conn.info.id, conn)
      ws.on('pong', () => (conn.alive = true))
      ws.on('message', (d) => void this.request(conn, parse(d)))
      ws.on('close', () => {
        if (this.conns.delete(conn.info.id)) this.onClientsChanged(this.clients())
      })
      ws.send(JSON.stringify({ t: 'welcome', protocol: REMOTE_PROTOCOL, version: this.host.version, name: this.host.name }))
      this.onClientsChanged(this.clients())
    })
    ws.on('error', () => ws.terminate())
  }

  private async request(conn: Conn, msg: Record<string, unknown> | null): Promise<void> {
    if (msg?.t !== 'invoke' || typeof msg.id !== 'string' || typeof msg.channel !== 'string') return
    const { id, channel } = msg
    const args = Array.isArray(msg.args) ? msg.args : []
    let reply: object
    try {
      if (isComputerOnly(channel)) throw new Error('That can only be done on the computer.')
      const value = await this.host.invoke(channel, channel === 'settings:update' ? [settingsPatchFromPhone(args[0])] : args)
      const isSettings = channel === 'settings:get' || channel === 'settings:update'
      reply = { t: 'result', id, ok: true, value: isSettings && value ? settingsForPhone(value as AppSettings) : value }
    } catch (e) {
      reply = { t: 'result', id, ok: false, error: e instanceof Error ? e.message : String(e) }
    }
    if (conn.ws.readyState === WebSocket.OPEN) conn.ws.send(JSON.stringify(reply))
  }

  private send(channel: EventChannel, payload: unknown): void {
    if (NOT_SENT.has(channel) || !this.conns.size) return
    const text = JSON.stringify({ t: 'event', channel, payload })
    for (const c of this.conns.values()) {
      if (c.ws.readyState !== WebSocket.OPEN) continue
      if (c.ws.bufferedAmount > MAX_BUFFERED) c.ws.terminate()
      else c.ws.send(text)
    }
  }

  private ping(): void {
    for (const c of this.conns.values()) {
      if (!c.alive) {
        c.ws.terminate()
        continue
      }
      c.alive = false
      c.ws.ping()
    }
  }
}

function parse(data: RawData): Record<string, unknown> | null {
  try {
    const v: unknown = JSON.parse(data.toString())
    return v && typeof v === 'object' ? (v as Record<string, unknown>) : null
  } catch {
    return null
  }
}
