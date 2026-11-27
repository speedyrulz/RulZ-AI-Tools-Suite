// Single source of truth for IPC channels between the renderer and the main process.
import type {
  AppSettings,
  ConnectionStatus,
  GpuInfo,
  PreflightReport,
  ServerCapabilities,
  ServerLogEvent,
  ServerState
} from './types'

/** Request/response channels (renderer -> main). */
export interface IpcInvoke {
  'app:version': () => string
  'settings:get': () => AppSettings
  /** Shallow-merges top-level keys; `ownServer` is merged one level deeper. Returns the saved settings. */
  'settings:set': (patch: Partial<AppSettings>) => AppSettings
  'gpu:list': () => GpuInfo[]
  'server:preflight': () => PreflightReport
  /** `confirmed` lists the PreflightIssue codes the user accepted. Resolves when the server answers HTTP or fails. */
  'server:start': (confirmed: string[]) => ServerState
  'server:stop': () => ServerState
  'server:state': () => ServerState
  /** The last lines of the own server's log (for a page opened after start). */
  'server:log': () => string
  /** Checks the current connection (own server or URL) and returns its status. */
  'comfy:status': () => ConnectionStatus
  /** The user agreed to share the queue of a server the app did not start, for this session. */
  'comfy:acceptSharedQueue': (url: string) => ConnectionStatus
  'comfy:capabilities': () => ServerCapabilities
  'shell:openExternal': (url: string) => void
}

/** Push channels (main -> renderer). */
export interface IpcEvents {
  'server:state': ServerState
  'server:log': ServerLogEvent
  'comfy:status': ConnectionStatus
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
