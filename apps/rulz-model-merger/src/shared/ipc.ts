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
import type { GridCellRef, GridJob, GridRequest, GridTemplateInfo } from './grid'
import type { HistoryEntry } from './history'

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
  /** Reads and checks a template workflow (API-format JSON). With no path it opens a file dialog; null if cancelled. */
  'grid:loadTemplate': (path?: string) => GridTemplateInfo | null
  /** Starts a sample grid on the current server. Rejects when the request is invalid, the queue was not accepted, or a
   *  grid is already running. */
  'grid:start': (req: GridRequest) => GridJob
  /** Stops the running grid: deletes its pending prompts and interrupts its running one (never other prompts). */
  'grid:cancel': (gridId: string) => GridJob
  /** Grids of this session and earlier ones (read from the grid folders), newest first. */
  'grid:list': () => GridJob[]
  /** One cell image as a data: URL (image/png or the type ComfyUI returned). */
  'grid:image': (gridId: string, file: string) => string
  /** Saves a PNG the renderer composed (data: URL) through a save dialog; returns the path, or null if cancelled. */
  'grid:saveImage': (dataUrl: string, suggestedName: string) => string | null
  /** Opens the grid's folder in Explorer. */
  'grid:openFolder': (gridId: string) => void
  /** Deletes a grid folder (moves it to the Recycle Bin). */
  'grid:delete': (gridId: string) => void
  /** Saves the starred cells of a finished (or stopped) grid in its grid.json; returns the updated job. Cells outside
   *  the grid are dropped, duplicates removed. Rejects for a running grid. */
  'grid:setFavourites': (gridId: string, cells: GridCellRef[]) => GridJob
  /** Every finished merge, bake and LoRA merge (history.json) plus every grid (grid folders), newest first by
   *  createdAt. Jobs still running are not listed. */
  'history:list': () => HistoryEntry[]
  /** Removes a merge or LoRA merge entry from history.json (the saved file is not touched). For a grid use
   *  grid:delete. */
  'history:remove': (kind: 'merge' | 'loraMerge', jobId: string) => void
}

/** Push channels (main -> renderer). */
export interface IpcEvents {
  'server:state': ServerState
  'server:log': ServerLogEvent
  'comfy:status': ConnectionStatus
  'merge:job': MergeJob
  'loraMerge:job': LoraMergeJob
  'grid:job': GridJob
  /** A merge or LoRA merge reached a final status and was added to history.json (grids: use grid:job). */
  'history:changed': null
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
