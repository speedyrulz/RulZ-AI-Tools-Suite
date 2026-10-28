// Single source of truth for IPC channels between the renderer and the main process.
import type {
  AgentEvent,
  AppSettings,
  ConnectionInfo,
  FileDiff,
  ModelActivity,
  ProjectInfo,
  ProjectState,
  PromptAnswer,
  RunEvent,
  RunState
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
  'dialog:pickFolder': (opts: { title?: string; defaultPath?: string }) => string | null
  'shell:showItem': (path: string) => void
  'shell:openUrl': (url: string) => void

  /** Finds RulZ LLM Suite (current address, then 1234 / 1235) and lists its models. */
  'server:connect': () => ConnectionInfo
  'server:status': () => ConnectionInfo

  'projects:recent': () => ProjectInfo[]
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

  'run:state': (projectPath: string) => RunState
  'run:start': (projectPath: string) => RunState
  'run:stop': (projectPath: string) => RunState
  'run:setCommand': (projectPath: string, command: string | null) => RunState
  /** Starts a task that asks the agent how to run the project and saves the answer. */
  'run:askAgent': (projectPath: string) => void
  /** Stops the app if it is still running and starts a task to fix the error it showed. */
  'run:fix': (projectPath: string) => void
}

/** Push channels (webContents.send → ipcRenderer.on). */
export interface IpcEvents {
  'server:changed': ConnectionInfo
  'model:activity': ModelActivity
  'agent:event': AgentEvent
  'run:event': RunEvent
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
