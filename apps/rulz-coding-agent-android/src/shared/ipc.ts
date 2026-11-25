// Single source of truth for IPC channels between the renderer and the main process.
import type { PdfAttachment } from './pdf'
import type {
  AgentEvent,
  AgentMode,
  AppSettings,
  BatchOptions,
  BatchState,
  ConnectionInfo,
  FileDiff,
  FolderListing,
  LoadSettings,
  ModelActivity,
  ModelLoadInfo,
  ModelModes,
  ProjectInfo,
  ProjectState,
  PromptAnswer,
  ReOverview,
  ReStartRequest,
  RemotePairing,
  RemoteStatus,
  RunEvent,
  RunState,
  ToolServersInfo
} from './types'

export interface AppInfo {
  version: string
  userData: string
  platform: string
  gitInstalled: boolean
}

/** Request/response channels (ipcRenderer.invoke ↔ ipcMain.handle). */
export interface IpcInvoke {
  'app:info': () => AppInfo
  'settings:get': () => AppSettings
  'settings:update': (patch: Partial<AppSettings>) => AppSettings
  /** The fixed part of the agent's system prompt (shown read-only in Settings). */
  'settings:builtInPrompt': () => string
  /** Tool servers set up in RulZ LLM Suite, and those running. */
  'tools:status': () => ToolServersInfo
  /** Starts the tool servers now (they otherwise start when the agent first needs one). */
  'tools:start': () => ToolServersInfo
  'dialog:pickFolder': (opts: { title?: string; defaultPath?: string }) => string | null
  'dialog:pickFile': (opts: { title?: string; defaultPath?: string }) => string | null
  /** The computer's file chooser for PDFs (several at once); this computer only. */
  'dialog:pickPdfs': (opts: { title?: string }) => string[]
  'shell:showItem': (path: string) => void
  'shell:openUrl': (url: string) => void
  /** Subfolders of a folder (the drives when `path` is empty), for choosing a folder from the phone. */
  'fs:listFolders': (path: string) => FolderListing

  /** Phone access over Tailscale (only from this computer, never from the phone). */
  'remote:status': () => RemoteStatus
  'remote:setEnabled': (enabled: boolean) => RemoteStatus
  'remote:pairing': () => RemotePairing
  /** Makes a new pairing code; phones using the old one are disconnected. */
  'remote:newToken': () => RemotePairing

  /** Finds RulZ LLM Suite (current address, then 1234 / 1235) and lists its models. */
  'server:connect': () => ConnectionInfo
  'server:status': () => ConnectionInfo
  /** A model's load settings (context length, GPU layers…) in RulZ LLM Suite; with `mode`, the ones that mode loads with (needs 0.35). */
  'models:loadSettings': (model: string, mode?: AgentMode) => ModelLoadInfo
  /** RulZ LLM Suite's memory estimate for changed settings, without saving them. */
  'models:previewLoadSettings': (model: string, config: Partial<LoadSettings>, mode?: AgentMode) => ModelLoadInfo
  /**
   * Changes them; a loaded model restarts in RulZ LLM Suite to use them. With `mode` they are that mode's
   * own settings and `null` removes them. Refused for a rental's model and while a task or batch run goes.
   * `reloadBlocked` in the answer says why a reload LLM Suite refused did not happen (the settings are saved).
   */
  'models:setLoadSettings': (model: string, config: Partial<LoadSettings> | null, mode?: AgentMode) => ModelLoadInfo
  /** A model's settings per mode (sampling, prompt, whether it runs with the mode's load settings); null = a RulZ LLM Suite older than 0.35. */
  'models:modes': (model: string) => ModelModes | null
  /** Loads the model again with a mode's load settings (never automatic). Refused while a task or batch run goes, and for a rental's model. */
  'models:reloadMode': (model: string, mode: AgentMode) => ModelModes

  'projects:recent': () => ProjectInfo[]
  /** The project open right now, if any (the phone picks up where the computer is). */
  'projects:current': () => ProjectState | null
  'projects:create': (name: string, parentDir: string | null) => ProjectState
  'projects:open': (path: string) => ProjectState
  'projects:forget': (path: string) => ProjectInfo[]
  'projects:close': () => void

  /**
   * Starts a task. `attachments` are the ids of PDFs attached to the open project (pdf:attachPath or pdf:upload*),
   * at most 10; the text may be empty when there are some. Attachments only go with the message that starts a task.
   */
  'task:start': (projectPath: string, text: string, attachments?: string[]) => void
  /** Attaches a PDF from a path on this computer (this computer only); the open project only. */
  'pdf:attachPath': (projectPath: string, path: string) => PdfAttachment
  /** Starts an upload of a PDF (the phone, or a dropped file without a path): 512 KB chunks in order, base64, at most one per connection. */
  'pdf:uploadBegin': (projectPath: string, name: string, bytes: number) => { uploadId: string; chunkBytes: number }
  'pdf:uploadChunk': (uploadId: string, index: number, base64: string) => void
  'pdf:uploadEnd': (uploadId: string) => PdfAttachment
  /** Removes an attachment that was not sent yet; one a message uses is refused. */
  'pdf:remove': (projectPath: string, id: string) => void
  'task:stop': (projectPath: string) => void
  'task:answer': (projectPath: string, promptId: string, answer: PromptAnswer) => void
  /** A note for the running task, added before its next step; `interrupt` also cuts the model's current reply short. */
  'task:note': (projectPath: string, text: string, interrupt: boolean) => void
  'task:cancelNote': (projectPath: string, noteId: string) => void
  /** Cuts the model's current reply short so the waiting notes apply now (running commands go on). */
  'task:interrupt': (projectPath: string) => void
  /**
   * Restores the project to how it was before the last task that wasn't undone yet, or, with a
   * task id, to before that task (undoing it and every later task).
   */
  'task:undo': (projectPath: string, taskId?: string) => ProjectState
  /** Deletes a message from the conversation (a request takes its replies with it; see shared/history). */
  'task:deleteItem': (projectPath: string, itemId: string) => ProjectState
  /** Runs the latest request again, edited when `text` is given, after undoing its file changes. */
  'task:redo': (projectPath: string, taskId: string, text?: string) => ProjectState
  'files:diff': (projectPath: string, taskId: string, path: string) => FileDiff

  /** RulZ LLM Suite's RE sessions, practice tasks and templates (needs RulZ LLM Suite 0.15.0). */
  /** With `model`, the practice counts are for that model (`practice.model` is then set); an older RulZ LLM Suite counts all models together. */
  're:overview': (model?: string) => ReOverview
  /** Starts a reverse-engineering task (analyse, practice or clone) in the open project. */
  're:start': (projectPath: string, req: ReStartRequest) => void
  /** Changes RulZ LLM Suite's practice kit folder (needs RulZ LLM Suite 0.27.0) and returns the kit's tasks. */
  're:setKit': (kitDir: string) => ReOverview['practice']

  /** Practice batch runs: the saved runs and the one going now. */
  'batch:state': () => BatchState
  /**
   * Starts a batch run of practice tasks in the open project. Tools are approved automatically in its tasks.
   * `rentalOk`: the owner's yes to use a model on a rented computer; the main process refuses a rental's model without it.
   */
  'batch:start': (projectPath: string, label: string, options: BatchOptions, record: boolean, rentalOk?: boolean) => BatchState
  /** 'after' finishes the task in progress, then stops; 'now' stops that task too. */
  'batch:stop': (mode: 'after' | 'now') => BatchState
  /** Removes a saved run from the scoreboard (not the running one). */
  'batch:delete': (id: string) => BatchState
  /** Saves the owner's "Things to always remember" for the project. */
  'project:setMemory': (projectPath: string, text: string) => void

  'run:state': (projectPath: string) => RunState
  'run:start': (projectPath: string) => RunState
  'run:stop': (projectPath: string) => RunState
  'run:setCommand': (projectPath: string, command: string | null) => RunState
  /** Starts a task that asks the agent how to run the project and saves the answer. */
  'run:askAgent': (projectPath: string) => void
  /** Stops the app if it is still running and starts a task to fix the error it showed. */
  'run:fix': (projectPath: string) => void
  /** Stops the app if it is running and starts a task that packages it to give to other people. */
  'run:share': (projectPath: string) => void
}

/** Push channels (webContents.send → ipcRenderer.on). */
export interface IpcEvents {
  'server:changed': ConnectionInfo
  'model:activity': ModelActivity
  'agent:event': AgentEvent
  'run:event': RunEvent
  'tools:changed': ToolServersInfo
  /** The open project changed (opened, closed or undone, possibly from the phone), so every screen follows. */
  'project:changed': ProjectState | null
  'remote:changed': RemoteStatus
  /** A batch run started, moved on, stopped or was removed. */
  'batch:changed': BatchState
}

export type InvokeChannel = keyof IpcInvoke
export type EventChannel = keyof IpcEvents

export interface RulzApi {
  invoke<K extends InvokeChannel>(
    channel: K,
    ...args: Parameters<IpcInvoke[K]>
  ): Promise<Awaited<ReturnType<IpcInvoke[K]>>>
  on<K extends EventChannel>(channel: K, listener: (payload: IpcEvents[K]) => void): () => void
  /** The path of a file dropped on the window ('' when it has none); only this computer's window has it. */
  pathForFile(file: File): string
}
