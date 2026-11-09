/** A computer running RulZ Coding Agent that this phone has paired with. */
export interface Computer {
  id: string
  /** Shown in the app, e.g. "my-pc". */
  name: string
  /** Tailscale address, e.g. 100.88.114.32. */
  host: string
  /** Tailscale (MagicDNS) name, tried when the address doesn't answer. */
  dns?: string
  port: number
  token: string
}

export const DEFAULT_PORT = 7420

/**
 * Reads a pairing link from the computer's QR code:
 * rulzagent://pair?host=100.88.114.32&port=7420&token=…&name=my-pc&dns=my-pc.tail1234.ts.net
 */
export function parsePairingLink(link: string): Omit<Computer, 'id'> | null {
  const m = /^rulzagent:\/\/pair\/?\?(.*)$/i.exec(link.trim())
  if (!m) return null
  const q = new URLSearchParams(m[1])
  const host = q.get('host')?.trim() || q.get('dns')?.trim() || ''
  const token = q.get('token')?.trim() ?? ''
  const port = Number(q.get('port') ?? DEFAULT_PORT)
  if (!host || !token || !validPort(port)) return null
  const dns = q.get('dns')?.trim() || undefined
  return { name: q.get('name')?.trim() || dns?.split('.')[0] || host, host, dns: dns !== host ? dns : undefined, port, token }
}

export function validPort(port: number): boolean {
  return Number.isInteger(port) && port > 0 && port < 65536
}

/** Cleans a typed address: no scheme, no path, no port (the port has its own field). */
export function cleanHost(input: string): string {
  return input
    .trim()
    .replace(/^[a-z]+:\/\//i, '')
    .replace(/\/.*$/, '')
    .replace(/:\d+$/, '')
}

/** The WebSocket address for a host (IPv6 addresses need brackets). */
export function socketUrl(host: string, port: number): string {
  const h = host.includes(':') && !host.startsWith('[') ? `[${host}]` : host
  return `ws://${h}:${port}/rulz`
}

/** A web address the computer printed (e.g. http://localhost:5173), as reachable from the phone. */
export function urlOnComputer(url: string, host: string): string {
  try {
    const u = new URL(url)
    if (['localhost', '127.0.0.1', '0.0.0.0', '[::1]', '::1'].includes(u.hostname)) u.hostname = host
    return u.toString()
  } catch {
    return url
  }
}

/** The same computer paired again (a new QR code) replaces the old entry. */
export function sameComputer(a: Pick<Computer, 'host' | 'port'>, b: Pick<Computer, 'host' | 'port'>): boolean {
  return a.host.toLowerCase() === b.host.toLowerCase() && a.port === b.port
}
