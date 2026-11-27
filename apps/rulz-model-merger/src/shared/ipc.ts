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
import type { MergeCheck, MergeInspectRequest, MergeJob, MergeRecipe } from './merge'
import type { LoraMergeCheck, LoraMergeInspectRequest, LoraMergeJob, LoraMergeRecipe } from './loraMerge'

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
  /** Reads the source files' headers (own server) and checks they can be merged. Never loads weights. */
  'merge:inspect': (req: MergeInspectRequest) => MergeCheck
  /** Queues the merge on the current server. Rejects when the recipe is invalid or the queue was not accepted. */
  'merge:start': (recipe: MergeRecipe) => MergeJob
  /** Cancels a queued or running merge of this app (never other prompts). */
  'merge:cancel': (jobId: string) => MergeJob
  /** Merges of this session, newest first. */
  'merge:jobs': () => MergeJob[]
  /** Own server only: moves the saved file and its sidecar into <models>/<checkpoints|diffusion_models>/merged/ without overwriting. */
  'merge:moveToModels': (jobId: string) => MergeJob
  /** Reads the input LoRAs' headers from <models>/loras (WSL, own-server settings) and checks they can be merged. */
  'loraMerge:inspect': (req: LoraMergeInspectRequest) => LoraMergeCheck
  /** Starts a LoRA merge in the app (worker thread). Rejects when the recipe is invalid or a merge is running. */
  'loraMerge:start': (recipe: LoraMergeRecipe) => LoraMergeJob
  'loraMerge:cancel': (jobId: string) => LoraMergeJob
  /** LoRA merges of this session, newest first. */
  'loraMerge:jobs': () => LoraMergeJob[]
  /** LoRA names in <models>/loras (relative, forward slashes), read from disk; works without a running server. */
  'loraMerge:list': () => string[]
}

/** Push channels (main -> renderer). */
export interface IpcEvents {
  'server:state': ServerState
  'server:log': ServerLogEvent
  'comfy:status': ConnectionStatus
  'merge:job': MergeJob
  'loraMerge:job': LoraMergeJob
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
