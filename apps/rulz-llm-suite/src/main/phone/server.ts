import { createHash, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
import { createServer, type IncomingMessage, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import type { Duplex } from 'node:stream'
import { WebSocket, WebSocketServer, type RawData } from 'ws'
import { PHONE_EVENTS, PHONE_PATH, PHONE_PROTOCOL, type PhoneClient, type PhoneEventChannel } from '@shared/phone'

// The phone app talks to this computer over a WebSocket on the Tailscale address only. Tailscale encrypts the
// traffic; the pairing code proves it is the owner's phone. Only the channels in PHONE_INVOKE are answered and only
// the events in PHONE_EVENTS are sent (contract: src/shared/phone.ts).
//
// Phone → computer:  { t: 'hello', token, device, protocol }   then   { t: 'invoke', id, channel, args }
// Computer → phone:  { t: 'welcome', protocol, version, name }  { t: 'result', id, ok, value | error }
//                    { t: 'event', channel, payload }           { t: 'denied', reason }

const HELLO_TIMEOUT_MS = 10_000
const PING_MS = 25_000
/** A phone this far behind (slow network) is dropped; it reconnects and reloads. */
const MAX_BUFFERED = 16 * 1024 * 1024
const MAX_FAILURES = 10
const LOCKOUT_MS = 10 * 60_000
/** Connections that haven't said hello yet, at most (each is dropped after HELLO_TIMEOUT_MS anyway). */
const MAX_PENDING = 16
/** The same from one address (one of your devices), so a single device can't fill every slot and keep your phone out. */
const MAX_PENDING_PER_ADDRESS = 8

/**
 * Pages allowed to connect: the phone app (its WebView serves the app from http://localhost) and
 * clients that aren't browsers (no Origin). A web page open in a browser on a device in the
 * tailnet can't connect, so it can't use up the wrong-code limit and lock the phone out.
 */
export function originAllowed(origin: string | undefined): boolean {
  if (!origin) return true
  return /^(https?|capacitor):\/\/localhost(:\d+)?$/i.test(origin)
}

export interface PhoneHost {
  /** Answers one phone request. `owner` is 'phone:<clientId>'. Refusals and failures throw. */
  invoke(owner: string, channel: string, args: unknown[]): Promise<unknown>
  subscribe(listener: (channel: string, payload: unknown) => void): () => void
  /** A phone's connection ended: release what it held (open projects). Must be safe to call twice. */
  release?(owner: string): void | Promise<void>
  version: string
  /** This computer's name, shown on the phone. */
  name: string
}

interface Conn {
  info: PhoneClient
  ws: WebSocket
  alive: boolean
  owner: string
}

export function newToken(): string {
  return randomBytes(32).toString('base64url')
}

const digest = (s: string): Buffer => createHash('sha256').update(s).digest()
export function tokenMatches(given: unknown, expected: string): boolean {
  return typeof given === 'string' && !!expected && timingSafeEqual(digest(given), digest(expected))
}

export class PhoneServer {
  private servers: Server[] = []
  private readonly wss = new WebSocketServer({ noServer: true, maxPayload: 4 * 1024 * 1024 })
  private readonly conns = new Map<string, Conn>()
  /** Connections waiting for their pairing code, and where each comes from. */
  private readonly pending = new Map<WebSocket, string>()
  private readonly failures = new Map<string, { count: number; until: number }>()
  private token = ''
  private unsubscribe: (() => void) | null = null
  private pinger: NodeJS.Timeout | null = null

  constructor(
    private readonly host: PhoneHost,
    private readonly onClientsChanged: (clients: PhoneClient[]) => void
  ) {}

  get listening(): boolean {
    return this.servers.length > 0
  }

  /** The port actually listened on (differs from the one asked for only when that was 0, in tests). */
  get port(): number | undefined {
    return (this.servers[0]?.address() as AddressInfo | null)?.port
  }

  clients(): PhoneClient[] {
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

  /** Disconnects one phone. */
  disconnect(clientId: string): void {
    this.conns.get(clientId)?.ws.close(4002, 'Disconnected on the computer.')
  }

  async close(): Promise<void> {
    this.unsubscribe?.()
    this.unsubscribe = null
    if (this.pinger) clearInterval(this.pinger)
    this.pinger = null
    const gone = [...this.conns.values()]
    for (const c of gone) c.ws.terminate()
    this.conns.clear()
    for (const c of gone) this.release(c)
    for (const ws of this.pending.keys()) ws.terminate()
    this.pending.clear()
    const servers = this.servers
    this.servers = []
    await Promise.all(
      servers.map(
        (s) =>
          new Promise<void>((r) => {
            s.close(() => r())
            // A peer that holds a plain HTTP connection open must not stall the close (and the queue behind it).
            s.closeAllConnections?.()
          })
      )
    )
  }

  private release(conn: Conn): void {
    try {
      void Promise.resolve(this.host.release?.(conn.owner)).catch(() => undefined)
    } catch {
      // Releasing is best effort; the project is dropped at quit anyway.
    }
  }

  private listenOn(address: string, port: number): Promise<Server> {
    const server = createServer((req, res) => {
      // A plain request answers "is this RulZ LLM Suite?" and nothing else.
      if (req.method === 'GET' && (req.url === '/' || req.url === PHONE_PATH)) {
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ app: 'rulz-llm-suite', protocol: PHONE_PROTOCOL }))
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
    if (path !== PHONE_PATH) {
      socket.end('HTTP/1.1 404 Not Found\r\n\r\n')
      return
    }
    if (!originAllowed(req.headers.origin)) {
      socket.end('HTTP/1.1 403 Forbidden\r\n\r\n')
      return
    }
    const address = req.socket.remoteAddress ?? ''
    let fromHere = 0
    for (const a of this.pending.values()) if (a === address) fromHere++
    if (this.pending.size >= MAX_PENDING || fromHere >= MAX_PENDING_PER_ADDRESS) {
      socket.end('HTTP/1.1 503 Service Unavailable\r\n\r\n')
      return
    }
    this.wss.handleUpgrade(req, socket, head, (ws) => this.accept(ws, address))
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
      // No longer waiting for a code (closing can take a while, and mustn't hold a slot meanwhile).
      this.pending.delete(ws)
      ws.send(JSON.stringify({ t: 'denied', reason }))
      ws.close(4001, reason)
    }
    // First: a socket that errors before anything else is set up must not crash or linger.
    ws.on('error', () => ws.terminate())
    if (this.lockedOut(address)) return deny('locked')
    this.pending.set(ws, address)
    const timer = setTimeout(() => deny('timeout'), HELLO_TIMEOUT_MS)
    ws.once('close', () => {
      clearTimeout(timer)
      this.pending.delete(ws)
    })

    ws.once('message', (data) => {
      clearTimeout(timer)
      this.pending.delete(ws)
      // Already refused (the hello came too late, while the socket was closing).
      if (ws.readyState !== WebSocket.OPEN) return
      const msg = parse(data)
      if (msg?.t !== 'hello') return deny('bad-hello')
      // Checked again here: connections opened side by side would each get a guess past the limit.
      if (this.lockedOut(address)) return deny('locked')
      if (!tokenMatches(msg.token, this.token)) {
        this.failed(address)
        return deny(this.lockedOut(address) ? 'locked' : 'wrong-code')
      }
      this.failures.delete(address)
      const device = typeof msg.device === 'string' && msg.device.trim() ? msg.device.trim().slice(0, 60) : 'Phone'
      const id = randomUUID()
      const conn: Conn = { info: { id, device, address, connectedAt: Date.now() }, ws, alive: true, owner: `phone:${id}` }
      this.conns.set(id, conn)
      ws.on('pong', () => (conn.alive = true))
      ws.on('message', (d) => void this.request(conn, parse(d)))
      ws.on('close', () => {
        if (this.conns.delete(id)) {
          this.release(conn)
          this.onClientsChanged(this.clients())
        }
      })
      ws.send(JSON.stringify({ t: 'welcome', protocol: PHONE_PROTOCOL, version: this.host.version, name: this.host.name }))
      this.onClientsChanged(this.clients())
    })
  }

  private async request(conn: Conn, msg: Record<string, unknown> | null): Promise<void> {
    if (msg?.t !== 'invoke' || typeof msg.id !== 'string' || typeof msg.channel !== 'string') return
    const { id, channel } = msg
    const args = Array.isArray(msg.args) ? msg.args : []
    let reply: object
    try {
      reply = { t: 'result', id, ok: true, value: await this.host.invoke(conn.owner, channel, args) }
    } catch (e) {
      reply = { t: 'result', id, ok: false, error: e instanceof Error ? e.message : String(e) }
    }
    if (conn.ws.readyState === WebSocket.OPEN) conn.ws.send(JSON.stringify(reply))
  }

  private send(channel: string, payload: unknown): void {
    if (!PHONE_EVENTS.has(channel as PhoneEventChannel) || !this.conns.size) return
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
