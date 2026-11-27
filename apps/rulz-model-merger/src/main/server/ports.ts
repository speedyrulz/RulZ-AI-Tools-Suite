// A port is free only when Node can bind it on 127.0.0.1 AND WSL shows nothing listening on it (WSL uses mirrored
// networking, so a server inside WSL and one on Windows share the port space).
import net from 'node:net'
import type { WslRunner } from './wsl'

export function canBindLocally(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const srv = net.createServer()
    srv.once('error', () => resolve(false))
    srv.listen(port, '127.0.0.1', () => srv.close(() => resolve(true)))
  })
}

/** Ports from `ss -ltnH` output: the local address column ends in ":<port>". */
export function parseSsPorts(text: string): Set<number> {
  const ports = new Set<number>()
  for (const line of text.split(/\r?\n/)) {
    const cols = line.trim().split(/\s+/)
    if (cols.length < 4) continue
    const m = /:(\d+)$/.exec(cols[3])
    if (m) ports.add(Number(m[1]))
  }
  return ports
}

/** Ports WSL listens on, or null when that could not be read (ss failed, was missing or timed out). */
export async function wslListeningPorts(wsl: WslRunner, distro: string): Promise<Set<number> | null> {
  const r = await wsl.run(distro, 'ss -ltnH\n', { timeoutMs: 15000 })
  if (r.code !== 0) return null
  return parseSsPorts(r.stdout)
}

/** A port is free only when both checks passed. When WSL cannot be asked, the port is NOT treated as free. */
export async function isPortFree(port: number, wsl: WslRunner, distro: string): Promise<boolean> {
  if (!Number.isInteger(port) || port < 1 || port > 65535) return false
  if (!(await canBindLocally(port))) return false
  const wslPorts = await wslListeningPorts(wsl, distro)
  return wslPorts !== null && !wslPorts.has(port)
}

export interface PortScan {
  port: number | null
  /** 'ss-failed': WSL could not be asked which ports are in use; 'none-free': every port in the span is taken. */
  reason: 'ss-failed' | 'none-free' | null
  /** Every port WSL listens on (empty when ss failed). */
  listening: Set<number>
}

export async function scanForFreePort(start: number, wsl: WslRunner, distro: string, span = 50): Promise<PortScan> {
  const wslPorts = await wslListeningPorts(wsl, distro) // read once for the whole scan
  if (wslPorts === null) return { port: null, reason: 'ss-failed', listening: new Set() }
  for (let p = Math.max(1, start); p < start + span && p <= 65535; p++) {
    if (wslPorts.has(p)) continue
    if (await canBindLocally(p)) return { port: p, reason: null, listening: wslPorts }
  }
  return { port: null, reason: 'none-free', listening: wslPorts }
}

export async function findFreePort(start: number, wsl: WslRunner, distro: string, span = 50): Promise<number | null> {
  return (await scanForFreePort(start, wsl, distro, span)).port
}
