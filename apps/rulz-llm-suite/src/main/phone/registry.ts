// The phone's handler registry and dispatch layer: what a phone may call, and the restrictions that apply only to
// phones. The window's own handlers are not touched; the phone is held to these rules here (design:
// docs/phone-access-design.md).
import { isAbsolute, resolve } from 'node:path'
import { PHONE_INVOKE, type PhoneInvoke, type PhoneInvokeChannel } from '@shared/phone'

/**
 * A handler gets the caller's owner id ('phone:<clientId>') first, then the channel's arguments as the window's
 * handler would. Owners let the store count who has a project open.
 */
export type PhoneHandlerFor<K extends PhoneInvokeChannel> = (
  owner: string,
  ...args: Parameters<PhoneInvoke[K]>
) => Awaited<ReturnType<PhoneInvoke[K]>> | Promise<Awaited<ReturnType<PhoneInvoke[K]>>>

/** Channel to handler, for the channels in PHONE_INVOKE (any subset; add the rest later with register). */
export type PhoneHandlerTable = { [K in PhoneInvokeChannel]?: PhoneHandlerFor<K> }

type AnyHandler = (owner: string, ...args: unknown[]) => unknown

export interface PhoneGuards {
  /** The folders novel:listProjects returns (a project inside the projects root). */
  projectFolders: () => Promise<string[]>
  /** Whether this owner has the project open (a phone may only touch projects it opened itself). */
  ownsProject: (owner: string, folder: string) => boolean
}

/** Channels whose first argument is the project folder. */
const FOLDER_ARG = new Set([
  'novel:readFile', 'novel:outlinePatch', 'novel:projectPatch', 'novel:createEntry', 'novel:deleteScene', 'novel:deleteChapter',
  'novel:deleteEntry', 'novel:history', 'novel:readSnapshot', 'novel:trash', 'novel:restoreTrash', 'novel:statePatch',
  'novel:dismissFinding', 'writing:jobs'
])
/** Channels whose first argument is a request object with a `folder`. */
const FOLDER_IN_REQUEST = new Set(['novel:saveFile', 'writing:run'])

export const REFUSED = 'That is not available from the phone.'

const normalise = (folder: string): string => {
  const full = resolve(folder)
  return process.platform === 'win32' ? full.toLowerCase() : full
}

export class PhoneHandlers {
  private readonly map = new Map<string, AnyHandler>()

  constructor(private readonly guards: PhoneGuards) {}

  /** Plugs one channel in. A channel that is not in PHONE_INVOKE cannot be registered. */
  register<K extends PhoneInvokeChannel>(channel: K, fn: PhoneHandlerFor<K>): void {
    if (!PHONE_INVOKE.has(channel)) throw new Error(`${channel} is not a phone channel.`)
    this.map.set(channel, fn as AnyHandler)
  }

  /** Plugs several channels in at once. */
  registerAll(table: PhoneHandlerTable): void {
    for (const [channel, fn] of Object.entries(table)) {
      if (fn) this.register(channel as PhoneInvokeChannel, fn as PhoneHandlerFor<PhoneInvokeChannel>)
    }
  }

  has(channel: string): boolean {
    return this.map.has(channel)
  }

  /** Runs a phone's request: the allowlist first, then the phone-only restrictions, then the handler. */
  async invoke(owner: string, channel: string, args: unknown[]): Promise<unknown> {
    if (!PHONE_INVOKE.has(channel)) throw new Error(REFUSED)
    await this.check(owner, channel, args)
    const fn = this.map.get(channel)
    if (!fn) throw new Error('That is not available yet.')
    return fn(owner, ...args)
  }

  private async check(owner: string, channel: string, args: unknown[]): Promise<void> {
    const folderArg = FOLDER_ARG.has(channel) ? args[0] : FOLDER_IN_REQUEST.has(channel) ? (args[0] as { folder?: unknown } | null | undefined)?.folder : undefined
    if (FOLDER_ARG.has(channel) || FOLDER_IN_REQUEST.has(channel)) {
      if (typeof folderArg !== 'string' || !this.guards.ownsProject(owner, folderArg)) throw new Error('That project is not open on this phone.')
    }
    if (channel === 'novel:openProject') {
      const folder = args[0]
      if (typeof folder !== 'string' || !isAbsolute(folder)) throw new Error('The project folder must be a full path.')
      const known = (await this.guards.projectFolders()).map(normalise)
      if (!known.includes(normalise(folder))) throw new Error('The phone can only open projects in the projects folder.')
    } else if (channel === 'adventure:create') {
      // A Writing project whose world is copied into the adventure: the same folders novel:openProject allows; the
      // phone does not need to have opened it.
      const folder = args[1]
      if (folder !== undefined && folder !== null) {
        if (typeof folder !== 'string' || !isAbsolute(folder)) throw new Error('The project folder must be a full path.')
        const known = (await this.guards.projectFolders()).map(normalise)
        if (!known.includes(normalise(folder))) throw new Error('The phone can only use projects in the projects folder.')
      }
    } else if (channel === 'novel:saveFile') {
      const req = args[0] as { baseHash?: unknown; force?: unknown } | null | undefined
      if (!req || typeof req !== 'object') throw new Error('The save request is missing.')
      // A phone never overwrites: it must say which version its text is based on, and a conflict stays a conflict.
      if (typeof req.baseHash !== 'string') throw new Error('The phone must say which version its text is based on.')
      if (req.force) throw new Error('The phone cannot overwrite a changed file.')
    }
  }
}
