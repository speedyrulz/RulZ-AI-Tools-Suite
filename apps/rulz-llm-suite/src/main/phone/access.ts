import { hostname } from 'node:os'
import QRCode from 'qrcode'
import { PHONE_LINK_SCHEME, type PhonePairing, type PhoneStatus } from '@shared/phone'
import type { AppSettings } from '@shared/types'
import { newToken, PhoneServer, type PhoneHost } from './server'
import { tailscaleAddresses, tailscaleDnsName } from './tailscale'

/** How often to look for Tailscale coming up (or its address changing) while phone access is on. */
const WATCH_MS = 15_000

export interface PhoneAccessOptions {
  settings: { get(): AppSettings; update(patch: Partial<AppSettings>): Promise<AppSettings> }
  host: PhoneHost
  emit: (status: PhoneStatus) => void
  /** Addresses to listen on; tests pass 127.0.0.1. */
  addresses?: () => string[]
  dnsName?: () => Promise<string | undefined>
}

/** Turns phone access on and off, keeps it listening on the Tailscale address, and makes pairing codes. */
export class PhoneAccess {
  private readonly server: PhoneServer
  private readonly addresses: () => string[]
  private readonly dnsName: () => Promise<string | undefined>
  private listeningOn: string[] = []
  private listeningPort = 0
  /** The pairing code the server holds. */
  private serverToken = ''
  private watch: NodeJS.Timeout | null = null
  private state: PhoneStatus['state'] = 'off'
  private error: string | undefined
  private dns: string | undefined
  private busy: Promise<void> = Promise.resolve()

  constructor(private readonly opts: PhoneAccessOptions) {
    this.addresses = opts.addresses ?? (() => tailscaleAddresses())
    this.dnsName = opts.dnsName ?? tailscaleDnsName
    this.server = new PhoneServer(opts.host, () => this.changed())
  }

  status(): PhoneStatus {
    const s = this.opts.settings.get()
    return {
      enabled: s.phoneEnabled,
      state: this.state,
      port: s.phonePort,
      addresses: this.listeningOn.length ? this.listeningOn : this.addresses(),
      dnsName: this.dns,
      clients: this.server.clients(),
      error: this.error
    }
  }

  /** Disconnects one paired phone (it can connect again with the same code). */
  disconnect(clientId: string): void {
    this.server.disconnect(clientId)
  }

  /** Starts listening if phone access was left on. */
  start(): Promise<void> {
    return this.serial(() => this.apply())
  }

  async setEnabled(enabled: boolean): Promise<PhoneStatus> {
    await this.serial(async () => {
      await this.opts.settings.update({ phoneEnabled: enabled })
      await this.apply()
    })
    return this.status()
  }

  pairing(): Promise<PhonePairing> {
    return this.serial(() => this.pairingNow())
  }

  newToken(): Promise<PhonePairing> {
    return this.serial(async () => {
      await this.opts.settings.update({ phoneToken: newToken() })
      return this.pairingNow()
    })
  }

  async dispose(): Promise<void> {
    this.stopWatch()
    await this.server.close()
  }

  /** Runs one step at a time (settings changes, token changes and the listening state never interleave). */
  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.busy.then(fn, fn)
    this.busy = run.then(
      () => undefined,
      () => undefined
    )
    return run
  }

  /** The pairing code: made here if it is missing, and handed to the server whenever it differs from the one it holds. */
  private async currentToken(): Promise<string> {
    if (!this.opts.settings.get().phoneToken) await this.opts.settings.update({ phoneToken: newToken() })
    const token = this.opts.settings.get().phoneToken
    if (token !== this.serverToken) {
      this.serverToken = token
      this.server.setToken(token)
    }
    return token
  }

  private async pairingNow(): Promise<PhonePairing> {
    const token = await this.currentToken()
    const s = this.opts.settings.get()
    const host = this.status().addresses[0] ?? ''
    const params = new URLSearchParams({ host, port: String(s.phonePort), token, name: this.dns?.split('.')[0] || hostname() })
    const url = `${PHONE_LINK_SCHEME}://pair?${params}`
    const qrSvg = await QRCode.toString(url, { type: 'svg', margin: 1, errorCorrectionLevel: 'M' })
    return { url, qrSvg, host, port: s.phonePort, token }
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
    let s = this.opts.settings.get()
    if (!s.phoneEnabled) {
      this.stopWatch()
      await this.server.close()
      this.listeningOn = []
      this.state = 'off'
      this.error = undefined
      this.changed()
      return
    }
    // Turning on makes the first pairing code.
    await this.currentToken()
    s = this.opts.settings.get()
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
    const same = this.server.listening && addresses.join() === this.listeningOn.join() && s.phonePort === this.listeningPort && this.state === 'on'
    if (same) return
    this.state = 'starting'
    this.changed()
    try {
      await this.server.listen(addresses, s.phonePort, s.phoneToken)
      this.serverToken = s.phoneToken
      this.listeningOn = addresses
      this.listeningPort = s.phonePort
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
