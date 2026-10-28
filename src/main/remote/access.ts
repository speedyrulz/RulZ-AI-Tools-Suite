import { hostname } from 'node:os'
import QRCode from 'qrcode'
import type { AppSettings, RemotePairing, RemoteStatus } from '@shared/types'
import { newToken, RemoteServer, type RemoteHost } from './server'
import { tailscaleAddresses, tailscaleDnsName } from './tailscale'

/** How often to look for Tailscale coming up (or its address changing) while phone access is on. */
const WATCH_MS = 15_000

export interface PhoneAccessOptions {
  settings: { get(): AppSettings; update(patch: Partial<AppSettings>): Promise<AppSettings> }
  host: RemoteHost
  emit: (status: RemoteStatus) => void
  /** Addresses to listen on; tests pass 127.0.0.1. */
  addresses?: () => string[]
  dnsName?: () => Promise<string | undefined>
}

/** Turns phone access on and off, keeps it listening on the Tailscale address, and makes pairing codes. */
export class PhoneAccess {
  private readonly server: RemoteServer
  private readonly addresses: () => string[]
  private readonly dnsName: () => Promise<string | undefined>
  private listeningOn: string[] = []
  private watch: NodeJS.Timeout | null = null
  private state: RemoteStatus['state'] = 'off'
  private error: string | undefined
  private dns: string | undefined
  private busy: Promise<void> = Promise.resolve()

  constructor(private readonly opts: PhoneAccessOptions) {
    this.addresses = opts.addresses ?? (() => tailscaleAddresses())
    this.dnsName = opts.dnsName ?? tailscaleDnsName
    this.server = new RemoteServer(opts.host, () => this.changed())
  }

  status(): RemoteStatus {
    const s = this.opts.settings.get()
    return {
      enabled: s.remoteEnabled,
      state: this.state,
      port: s.remotePort,
      addresses: this.listeningOn.length ? this.listeningOn : this.addresses(),
      dnsName: this.dns,
      clients: this.server.clients(),
      error: this.error
    }
  }

  /** Starts listening if phone access was left on. */
  start(): Promise<void> {
    return this.serial(() => this.apply())
  }

  async setEnabled(enabled: boolean): Promise<RemoteStatus> {
    const patch: Partial<AppSettings> = { remoteEnabled: enabled }
    if (enabled && !this.opts.settings.get().remoteToken) patch.remoteToken = newToken()
    await this.opts.settings.update(patch)
    await this.serial(() => this.apply())
    return this.status()
  }

  async pairing(): Promise<RemotePairing> {
    const s = this.opts.settings.get()
    if (!s.remoteToken) await this.opts.settings.update({ remoteToken: newToken() })
    const token = this.opts.settings.get().remoteToken
    const host = this.status().addresses[0] ?? ''
    const params = new URLSearchParams({ host, port: String(s.remotePort), token, name: this.dns?.split('.')[0] || hostname() })
    if (this.dns) params.set('dns', this.dns)
    const url = `rulzagent://pair?${params}`
    const qrSvg = await QRCode.toString(url, { type: 'svg', margin: 1, errorCorrectionLevel: 'M' })
    return { url, qrSvg, host, port: s.remotePort, token }
  }

  async newToken(): Promise<RemotePairing> {
    const token = newToken()
    await this.opts.settings.update({ remoteToken: token })
    this.server.setToken(token)
    return this.pairing()
  }

  async dispose(): Promise<void> {
    this.stopWatch()
    await this.server.close()
  }

  private serial(fn: () => Promise<void>): Promise<void> {
    this.busy = this.busy.then(fn, fn)
    return this.busy
  }

  private changed(): void {
    this.opts.emit(this.status())
  }

  private stopWatch(): void {
    if (this.watch) clearInterval(this.watch)
    this.watch = null
  }

  /** Brings the server in line with the settings and the current Tailscale addresses. */
  private async apply(): Promise<void> {
    const s = this.opts.settings.get()
    if (!s.remoteEnabled) {
      this.stopWatch()
      await this.server.close()
      this.listeningOn = []
      this.state = 'off'
      this.error = undefined
      this.changed()
      return
    }
    this.watch ??= setInterval(() => void this.serial(() => this.apply()), WATCH_MS)
    const addresses = this.addresses()
    if (!addresses.length) {
      await this.server.close()
      this.listeningOn = []
      this.state = 'no-tailscale'
      this.error = undefined
      this.changed()
      return
    }
    const same = this.server.listening && addresses.join() === this.listeningOn.join() && this.state === 'on'
    if (same) return
    this.state = 'starting'
    this.changed()
    try {
      await this.server.listen(addresses, s.remotePort, s.remoteToken)
      this.listeningOn = addresses
      this.state = 'on'
      this.error = undefined
      this.dns = await this.dnsName()
    } catch (e) {
      this.listeningOn = []
      this.state = 'error'
      this.error = e instanceof Error ? e.message : String(e)
    }
    this.changed()
  }
}
