// Single source of truth for IPC channels between the renderer and the main process.
import type {
  AppSettings,
  ChatRequest,
  Conversation,
  ConversationSummary,
  EngineInfo,
  HardwareSnapshot,
  InstanceErrorEvent,
  LoadConfig,
  LoadedInstance,
  McpServerStatus,
  MemoryEstimate,
  ModelEntry,
  Preset,
  StreamEvent,
  ToolResult,
  WireMessage
} from './types'

export interface FileFilter {
  name: string
  extensions: string[]
}

export interface AppInfo {
  version: string
  userData: string
  platform: string
}

/** Request/response channels (ipcRenderer.invoke ↔ ipcMain.handle). */
export interface IpcInvoke {
  'app:info': () => AppInfo
  'settings:get': () => AppSettings
  'settings:update': (patch: Partial<AppSettings>) => AppSettings
  'dialog:pickFolder': (opts: { title?: string; defaultPath?: string }) => string | null
  'dialog:pickFile': (opts: { title?: string; filters?: FileFilter[] }) => string | null
  'shell:showItem': (path: string) => void
  'shell:openPath': (path: string) => void

  'engine:info': (refresh: boolean) => EngineInfo

  'models:list': () => ModelEntry[]
  'models:rescan': () => ModelEntry[]
  'models:trash': (modelId: string) => void
  'models:getDefaults': (modelId: string) => { config: LoadConfig; custom: boolean }
  'models:setDefaults': (modelId: string, config: LoadConfig | null) => void
  'models:estimate': (modelId: string, config: LoadConfig) => MemoryEstimate

  'instances:list': () => LoadedInstance[]
  'instances:load': (modelId: string, config: LoadConfig | null) => LoadedInstance
  'instances:unload': (instanceId: string) => void
  'instances:unloadAll': () => void
  'instances:log': (instanceId: string) => string[]

  'hardware:snapshot': () => HardwareSnapshot

  'chat:start': (req: ChatRequest) => void
  'chat:abort': (requestId: string) => void
  'chat:countTokens': (modelId: string, messages: WireMessage[], useTools: boolean) => number | null

  'conversations:list': () => ConversationSummary[]
  'conversations:get': (id: string) => Conversation | null
  'conversations:save': (conversation: Conversation) => ConversationSummary
  'conversations:delete': (id: string) => void
  'folders:list': () => string[]
  'folders:save': (folders: string[]) => void

  'mcp:getConfig': () => { text: string; path: string }
  'mcp:saveConfig': (text: string) => McpServerStatus[]
  'mcp:status': () => McpServerStatus[]
  'mcp:restart': (name: string | null) => void
  'mcp:callTool': (name: string, args: string) => ToolResult

  'presets:list': () => Preset[]
  'presets:save': (preset: Preset) => Preset[]
  'presets:delete': (id: string) => Preset[]
}

/** Push channels (webContents.send → ipcRenderer.on). */
export interface IpcEvents {
  'models:changed': ModelEntry[]
  'instances:changed': LoadedInstance[]
  'instances:error': InstanceErrorEvent
  'hardware:update': HardwareSnapshot
  'chat:event': StreamEvent
  'mcp:changed': McpServerStatus[]
}

export type InvokeChannel = keyof IpcInvoke
export type EventChannel = keyof IpcEvents

export interface RulzApi {
  invoke<K extends InvokeChannel>(
    channel: K,
    ...args: Parameters<IpcInvoke[K]>
  ): Promise<Awaited<ReturnType<IpcInvoke[K]>>>
  on<K extends EventChannel>(channel: K, listener: (payload: IpcEvents[K]) => void): () => void
  /** Absolute path of a file dropped or picked in the renderer. */
  pathForFile(file: File): string
}
