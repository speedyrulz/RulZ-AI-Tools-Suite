// Single source of truth for IPC channels between the renderer and the main process.
import type {
  AgentEvent,
  AppSettings,
  ConnectionInfo,
  FileDiff,
  FolderListing,
  LoadSettings,
  ModelActivity,
  ModelLoadInfo,
  ProjectInfo,
  ProjectState,
  PromptAnswer,
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
  /** A model's load settings (context length, GPU layers…) in RulZ LLM Suite. */
  'models:loadSettings': (model: string) => ModelLoadInfo
  /** RulZ LLM Suite's memory estimate for changed settings, without saving them. */
  'models:previewLoadSettings': (model: string, config: Partial<LoadSettings>) => ModelLoadInfo
  /** Changes them; a loaded model restarts in RulZ LLM Suite to use them. */
  'models:setLoadSettings': (model: string, config: Partial<LoadSettings>) => ModelLoadInfo

  'projects:recent': () => ProjectInfo[]
  /** The project open right now, if any (the phone picks up where the computer is). */
  'projects:current': () => ProjectState | null
  'projects:create': (name: string, parentDir: string | null) => ProjectState
  'projects:open': (path: string) => ProjectState
  'projects:forget': (path: string) => ProjectInfo[]
  'projects:close': () => void

  'task:start': (projectPath: string, text: string) => void
  'task:stop': (projectPath: string) => void
  'task:answer': (projectPath: string, promptId: string, answer: PromptAnswer) => void
  /**
   * Restores the project to how it was before the last task that wasn't undone yet, or, with a
   * task id, to before that task (undoing it and every later task).
   */
  'task:undo': (projectPath: string, taskId?: string) => ProjectState
  'files:diff': (projectPath: string, taskId: string, path: string) => FileDiff
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
