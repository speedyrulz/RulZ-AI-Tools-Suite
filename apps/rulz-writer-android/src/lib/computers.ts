import { PHONE_DEFAULT_PORT, PHONE_LINK_SCHEME, PHONE_PATH } from '@shared/phone'

/** A computer running RulZ LLM Suite that this phone has paired with. */
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

export const DEFAULT_PORT = PHONE_DEFAULT_PORT

/**
 * Reads a pairing link from the computer's QR code:
 * rulzwriter://pair?host=100.88.114.32&port=7421&token=…&name=my-pc
 */
export function parsePairingLink(link: string): Omit<Computer, 'id'> | null {
  const m = new RegExp(`^${PHONE_LINK_SCHEME}:\\/\\/pair\\/?\\?(.*)$`, 'i').exec(link.trim())
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
  return `ws://${h}:${port}${PHONE_PATH}`
}

/** The same computer paired again (a new QR code) replaces the old entry. */
export function sameComputer(a: Pick<Computer, 'host' | 'port'>, b: Pick<Computer, 'host' | 'port'>): boolean {
  return a.host.toLowerCase() === b.host.toLowerCase() && a.port === b.port
}
