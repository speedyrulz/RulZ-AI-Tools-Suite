// Single source of truth for IPC channels between the renderer and the main process.
import type { BatchRun } from './batch'
import type { BenchmarkRequest, BenchmarkRun } from './benchmark'
import type {
  AdapterConvertRequest,
  AdapterConvertResult,
  AdapterConverterStatus,
  AdapterEntry,
  ApiRequestLog,
  ApiServerStatus,
  AppSettings,
  ChatRequest,
  Conversation,
  ConversationSummary,
  EngineInfo,
  HardwareSnapshot,
  InstanceErrorEvent,
  LoadConfig,
  LoadedInstance,
  McpConfigState,
  McpServerStatus,
  MemoryEstimate,
  ModelEntry,
  Preset,
  ReExportOptions,
  ReOutcome,
  FileToolScope,
  ReExportResult,
  RePracticeInfo,
  ReTraceInfo,
  StreamEvent,
  ToolRequest,
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

  /** LoRA adapter GGUFs in the models folder (filled by the same scan as models:list). */
  'adapters:list': () => AdapterEntry[]
  /** Sends an adapter file to the Recycle Bin. */
  'adapters:trash': (adapterId: string) => void
  'adapters:converterStatus': () => AdapterConverterStatus
  /** Runs one conversion (one at a time); output lines arrive as adapters:convertLog events. */
  'adapters:convert': (req: AdapterConvertRequest) => AdapterConvertResult
  'adapters:cancelConvert': () => void

  /** Saved benchmark runs, newest first (a running one included). */
  'bench:list': () => BenchmarkRun[]
  /**
   * Starts a load-settings benchmark (one at a time; refused while one runs). It unloads every loaded
   * model first. Returns the new run; progress arrives as bench:changed events.
   */
  'bench:start': (req: BenchmarkRequest) => BenchmarkRun
  /** Stops the running benchmark: the current combination is ended and its model unloaded. */
  'bench:stop': () => void
  'bench:delete': (runId: string) => void

  'instances:list': () => LoadedInstance[]
  'instances:load': (modelId: string, config: LoadConfig | null) => LoadedInstance
  /** The model's running or loading instance whatever its settings, or else a new one with its defaults. */
  'instances:ensure': (modelId: string) => LoadedInstance
  'instances:unload': (instanceId: string) => void
  'instances:unloadAll': () => void
  'instances:log': (instanceId: string) => string[]

  'hardware:snapshot': () => HardwareSnapshot

  'chat:start': (req: ChatRequest) => void
  'chat:abort': (requestId: string) => void
  'chat:countTokens': (modelId: string, messages: WireMessage[], tools: ToolRequest | null) => number | null

  'conversations:list': () => ConversationSummary[]
  'conversations:get': (id: string) => Conversation | null
  'conversations:save': (conversation: Conversation) => ConversationSummary
  'conversations:delete': (id: string) => void
  'folders:list': () => string[]
  'folders:save': (folders: string[]) => void

  'mcp:getConfig': () => McpConfigState
  'mcp:saveConfig': (text: string) => McpServerStatus[]
  'mcp:status': () => McpServerStatus[]
  'mcp:restart': (name: string | null) => void
  /** `callKey` lets `mcp:cancelTool` stop this call while it runs. */
  /** `files` lets the built-in write/read file tools work (clone sessions). */
  'mcp:callTool': (name: string, args: string, callKey?: string, files?: FileToolScope) => ToolResult
  'mcp:cancelTool': (callKey: string) => void
  /** The tool round asked for by `requestId` is over (next round started, or the turn ended). */
  'chat:releaseTurn': (requestId: string) => void

  'server:status': () => ApiServerStatus
  'server:start': () => ApiServerStatus
  'server:stop': () => ApiServerStatus
  'server:log': () => ApiRequestLog[]
  'server:clearLog': () => void

  /** Recorded training data of every RE session. */
  're:traces': () => ReTraceInfo[]
  're:deleteTrace': (conversationId: string) => void
  /** Grades a session another app ran through the API server (id as listed, with its api- prefix). */
  're:setApiOutcome': (id: string, outcome: ReOutcome) => void
  /** Writes graded RE sessions as train.jsonl. */
  're:export': (opts: ReExportOptions) => ReExportResult
  're:dataFolder': () => string
  /** Whether the built-in calc server can run (it needs Python 3). */
  're:calcStatus': () => { available: boolean; python: string | null }
  're:practiceList': (kitDir: string) => RePracticeInfo
  /** Copies a practice task's binary into a fresh work folder. */
  're:practicePrepare': (kitDir: string, taskId: string) => { target: string; prompt: string; family: string; tier: string }
  /** Checks a practice session's final reply against the task's answer. */
  're:practiceGrade': (
    kitDir: string,
    taskId: string,
    target: string,
    finalText: string
  ) => { answer: string | null; passed: boolean; expected: string }
  /** Saved practice batch runs, newest first. */
  're:batches': () => BatchRun[]
  /** Adds a batch run, or replaces the saved one with its id. */
  're:saveBatch': (run: BatchRun) => void
  're:deleteBatch': (id: string) => void

  'presets:list': () => Preset[]
  'presets:save': (preset: Preset) => Preset[]
  'presets:delete': (id: string) => Preset[]
}

/** Push channels (webContents.send → ipcRenderer.on). */
export interface IpcEvents {
  'models:changed': ModelEntry[]
  'adapters:changed': AdapterEntry[]
  /** One output line of the running adapter conversion. */
  'adapters:convertLog': string
  /** A benchmark run changed (started, a combination finished or started, stopped, done). */
  'bench:changed': BenchmarkRun
  'instances:changed': LoadedInstance[]
  'instances:error': InstanceErrorEvent
  'hardware:update': HardwareSnapshot
  'chat:event': StreamEvent
  'mcp:changed': McpServerStatus[]
  'mcp:config': McpConfigState
  'server:changed': ApiServerStatus
  'server:request': ApiRequestLog
  /** An app recorded or described an RE session through the API server. */
  're:changed': undefined
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
