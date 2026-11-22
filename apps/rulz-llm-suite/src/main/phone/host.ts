// Phone access, assembled: the handler registry (what the phone may call), the event feed (what it is sent), the
// release hooks (what to let go of when a phone disconnects) and the on/off switch with its pairing code.
// No Electron imports here; src/main/index.ts connects it to the window's IPC.
import { hostname } from 'node:os'
import type { PhoneEventChannel, PhoneEvents, PhoneStatus } from '@shared/phone'
import type { AppSettings } from '@shared/types'
import type { NovelStore } from '../novel/store'
import { PhoneAccess } from './access'
import { novelPhoneHandlers } from './novelHandlers'
import { PhoneHandlers } from './registry'
import type { PhoneHost } from './server'

export interface PhoneSetup {
  settings: { get(): AppSettings; update(patch: Partial<AppSettings>): Promise<AppSettings> }
  novel: NovelStore
  version: string
  /** Sends the status to the Settings page. */
  emitStatus: (status: PhoneStatus) => void
  /** Tests: addresses to listen on instead of the Tailscale ones. */
  addresses?: () => string[]
  dnsName?: () => Promise<string | undefined>
  name?: string
}

export interface PhoneService {
  /** Plug channels in with `handlers.register(...)` / `handlers.registerAll(...)`; see registry.ts. */
  handlers: PhoneHandlers
  access: PhoneAccess
  /** What the phone server talks to (tests build their own server on it). */
  host: PhoneHost
  /** Sends an event to every connected phone (only PHONE_EVENTS channels are delivered). */
  emit<K extends PhoneEventChannel>(channel: K, payload: PhoneEvents[K]): void
  /** `fn(owner)` runs when a phone connection ends (owner = 'phone:<clientId>'); for services holding per-phone state. */
  onRelease(fn: (owner: string) => void | Promise<void>): void
}

export function createPhoneService(setup: PhoneSetup): PhoneService {
  const handlers = new PhoneHandlers({
    projectFolders: async () => (await setup.novel.listProjects()).map((p) => p.folder),
    ownsProject: (owner, folder) => setup.novel.ownersOf(folder).includes(owner)
  })
  handlers.registerAll(novelPhoneHandlers(setup.novel))

  const listeners = new Set<(channel: string, payload: unknown) => void>()
  const releasers: ((owner: string) => void | Promise<void>)[] = [(owner) => setup.novel.releaseOwner(owner)]

  /** Owners whose connection has ended; ids are never reused. */
  const closed = new Set<string>()
  const release = async (owner: string): Promise<void> => {
    for (const fn of releasers) await Promise.resolve(fn(owner)).catch(() => undefined)
  }

  const host: PhoneHost = {
    version: setup.version,
    name: setup.name ?? hostname(),
    invoke: async (owner, channel, args) => {
      if (closed.has(owner)) throw new Error('This phone is disconnected.')
      try {
        return await handlers.invoke(owner, channel, args)
      } finally {
        // The socket closed while this request ran (an open still reading the disk): let go of what it just took.
        if (closed.has(owner)) await release(owner)
      }
    },
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    release: async (owner) => {
      closed.add(owner)
      await release(owner)
    }
  }
  const access = new PhoneAccess({ settings: setup.settings, host, emit: setup.emitStatus, addresses: setup.addresses, dnsName: setup.dnsName })

  return {
    handlers,
    access,
    host,
    emit: (channel, payload) => listeners.forEach((l) => l(channel, payload)),
    onRelease: (fn) => void releasers.push(fn)
  }
}
