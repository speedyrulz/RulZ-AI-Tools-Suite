// Single source of truth for IPC channels between the renderer and the main process.
import type {
  AppView,
  InstallOptions,
  JobState,
  ReadmeDoc,
  SelfUpdate,
  RequirementId,
  RequirementStatus,
  Settings,
  ShortcutOptions
} from './types'

export interface AppInfo {
  version: string
  userData: string
  platform: string
}

export interface LogLine {
  appId: string
  line: string
  /** Replaces the previous line (a progress counter moving on, e.g. "Receiving objects: 40%"). */
  replace?: boolean
}

/** Request/response channels (ipcRenderer.invoke ↔ ipcMain.handle). */
export interface IpcInvoke {
  'app:info': () => AppInfo
  'settings:get': () => Settings
  'settings:update': (patch: Partial<Settings>) => Settings
  'dialog:pickFolder': (opts: { title?: string; defaultPath?: string }) => string | null
  'shell:openExternal': (url: string) => void

  'apps:list': () => AppView[]
  /** Checks GitHub for the latest versions again. */
  'apps:refresh': () => AppView[]
  'apps:readme': (appId: string) => ReadmeDoc
  'apps:defaultDir': (appId: string) => string
  'apps:install': (appId: string, opts: InstallOptions) => void
  'apps:update': (appId: string) => void
  'apps:repair': (appId: string) => void
  'apps:cancel': (appId: string) => void
  'apps:launch': (appId: string) => void
  /** Removes the app; files go to the Recycle Bin. For installs the launcher didn't make, only forgets them. */
  'apps:uninstall': (appId: string) => void
  /** Asks for the app's program file and registers it as installed. Returns false if cancelled. */
  'apps:locate': (appId: string) => boolean
  'apps:setShortcuts': (appId: string, opts: ShortcutOptions) => void
  'apps:openFolder': (appId: string) => void
  'apps:log': (appId: string) => string[]

  /** `reloadPath` re-reads PATH first, to find programs installed since the launcher started. */
  /** `force` skips the 30-minute cache. */
  'self:check': (force?: boolean) => SelfUpdate | null
  'self:install': (update: SelfUpdate) => void
  'requirements:check': (reloadPath?: boolean) => RequirementStatus[]
  /** Installs a requirement with winget; resolves when winget exits. */
  'requirements:install': (id: RequirementId) => RequirementStatus[]
}

/** Push channels (webContents.send → ipcRenderer.on). */
export interface IpcEvents {
  'self:progress': number
  'apps:changed': AppView[]
  'job:update': JobState
  /** Lines in order, sent in batches. */
  'job:log': LogLine[]
}

export type InvokeChannel = keyof IpcInvoke
export type EventChannel = keyof IpcEvents

export interface RulzApi {
  invoke<K extends InvokeChannel>(
    channel: K,
    ...args: Parameters<IpcInvoke[K]>
  ): Promise<Awaited<ReturnType<IpcInvoke[K]>>>
  on<K extends EventChannel>(channel: K, listener: (payload: IpcEvents[K]) => void): () => void
}
