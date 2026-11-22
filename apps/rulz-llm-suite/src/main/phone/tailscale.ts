import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import { networkInterfaces, type NetworkInterfaceInfo } from 'node:os'
import { join } from 'node:path'

/** Tailscale gives each device an address in 100.64.0.0/10. */
export function isTailscaleIPv4(address: string): boolean {
  const [a, b] = address.split('.').map(Number)
  return a === 100 && b >= 64 && b <= 127
}

/** This computer's Tailscale IPv4 addresses (the Tailscale adapter's, or any in its range if the adapter has another name). */
export function tailscaleAddresses(ifaces: NodeJS.Dict<NetworkInterfaceInfo[]> = networkInterfaces()): string[] {
  const candidates = Object.entries(ifaces).flatMap(([name, list]) =>
    (list ?? []).filter((i) => i.family === 'IPv4' && !i.internal && isTailscaleIPv4(i.address)).map((i) => ({ name, address: i.address }))
  )
  const named = candidates.filter((c) => /tailscale/i.test(c.name))
  return [...new Set((named.length ? named : candidates).map((c) => c.address))]
}

function tailscaleExe(): string {
  const installed = join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Tailscale', 'tailscale.exe')
  return existsSync(installed) ? installed : 'tailscale'
}

/** This computer's MagicDNS name (e.g. "my-pc.tail1234.ts.net"), or undefined when Tailscale can't say. */
export function tailscaleDnsName(): Promise<string | undefined> {
  return new Promise((resolve) => {
    execFile(tailscaleExe(), ['status', '--json', '--peers=false'], { timeout: 5000, windowsHide: true }, (err, stdout) => {
      if (err) return resolve(undefined)
      try {
        const name = (JSON.parse(stdout) as { Self?: { DNSName?: string } }).Self?.DNSName
        resolve(name ? name.replace(/\.$/, '') : undefined)
      } catch {
        resolve(undefined)
      }
    })
  })
}
