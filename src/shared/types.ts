// Types shared by the main process and the renderer.

export type Theme = 'dark' | 'light' | 'system'

/** How often the agent stops to ask before acting. */
export type ApprovalLevel = 'careful' | 'balanced' | 'handsoff'

export interface AppSettings {
  /** Server address the owner typed; empty = find it automatically. */
  serverUrl: string
  /** The address that worked last time (tried first when finding the server). */
  lastServerUrl: string
  apiKey: string
  /** Model id from /v1/models; empty = pick one automatically. */
  model: string
  approval: ApprovalLevel
  maxSteps: number
  /** Default time limit for one command, in minutes. */
  commandTimeoutMin: number
  theme: Theme
  /** Where "New project" creates folders. */
  projectsDir: string
  /** The owner's own instructions, added to the system prompt in every project. */
  instructions: string
  firstRunDone: boolean
  /** Lets the RulZ Coding Agent phone app connect over Tailscale. */
  remoteEnabled: boolean
  remotePort: number
  /** Secret the phone sends to connect; made the first time phone access is turned on. */
  remoteToken: string
}

// ---------- Connection to RulZ LLM Suite ----------

export interface ModelInfo {
  id: string
  /** True when the model is loaded in RulZ right now. */
  loaded: boolean
  /** Context length it is loaded with, when loaded. */
  loadedContext?: number
  maxContext?: number
  arch?: string
  quantization?: string
}

export type ConnectionStatus =
  /** Still looking. */
  | 'checking'
  | 'connected'
  /** Nothing answered. */
  | 'not-found'
  /** Something answered, but it isn't RulZ LLM Suite (usually LM Studio). */
  | 'other-server'
  /** RulZ answered but wants an API key (or a different one). */
  | 'needs-key'

export interface ConnectionInfo {
  status: ConnectionStatus
  url: string
  models: ModelInfo[]
  /** Address of the non-RulZ server that answered, for the "other-server" explanation. */
  otherUrl?: string
}

/** What the model is doing right now, for the top bar. */
export type ModelActivity = 'idle' | 'loading' | 'working'

// ---------- Projects ----------

export interface ProjectInfo {
  path: string
  name: string
  lastOpened: number
}

export type StepStatus = 'running' | 'done' | 'error' | 'waiting' | 'refused' | 'skipped'

export interface Step {
  id: string
  taskId: string
  /** Plain-language title, e.g. "Installing the pieces your app needs". */
  title: string
  status: StepStatus
  /** Tool name, for the icon. */
  tool: string
  /** Exact technical detail (command, path) shown under "Show details". */
  detail?: string
  /** Command output or tool result, shown under "Show details". */
  output?: string
  /** Why it was refused or failed, in plain words. */
  note?: string
}

export type FileChange = 'created' | 'changed' | 'deleted'

export interface ChangedFile {
  path: string
  change: FileChange
}

export type TimelineItem =
  | { id: string; kind: 'user'; text: string; taskId: string }
  | { id: string; kind: 'assistant'; text: string; taskId: string }
  | {
      id: string
      kind: 'notice'
      text: string
      tone: 'info' | 'ok' | 'warn' | 'error'
      taskId?: string
      /** A file to show with a "Show in folder" button (e.g. the packaged app). */
      reveal?: string
    }

/** How much of the model's context the conversation uses (the "Memory" meter). */
export interface ContextUsage {
  /** Tokens in the last request plus its reply. */
  used: number
  /** The model's context length. */
  size: number
  /** True when counted by the server; false when estimated. */
  exact: boolean
}

export type TaskOutcome = 'running' | 'done' | 'stopped' | 'error'

export interface TaskRecord {
  id: string
  title: string
  startedAt: number
  endedAt?: number
  outcome: TaskOutcome
  /** Snapshot taken before the task, for Undo. */
  snapshot?: { kind: 'git'; commit: string } | { kind: 'copy'; dir: string }
  undone?: boolean
  steps: Step[]
  files: ChangedFile[]
}

/** Everything the workspace needs when a project opens. */
export interface ProjectState {
  project: ProjectInfo
  timeline: TimelineItem[]
  tasks: TaskRecord[]
  running: boolean
  prompt: PendingPrompt | null
  runCommand: string | null
  /** The owner's "Things to always remember" for this project. */
  memory: string
  context: ContextUsage | null
}

// ---------- Questions the agent asks the owner ----------

export type RiskKind = 'install-global' | 'admin' | 'outside-project' | 'system-settings' | 'internet' | 'delete' | 'stop-programs'

export interface ApprovalRequest {
  id: string
  kind: 'approval'
  /** What will happen, in plain words. */
  action: string
  /** Why the agent wants to do it (its own explanation). */
  why: string
  /** Extra warning for risky actions. */
  risk?: string
  /** The exact command or file under "Show details". */
  detail: string
}

export interface QuestionRequest {
  id: string
  kind: 'question'
  question: string
}

/** The loop guard or the step limit paused the task. */
export interface StuckRequest {
  id: string
  kind: 'stuck'
  message: string
  reason: 'repeating' | 'errors' | 'limit'
}

export type PendingPrompt = ApprovalRequest | QuestionRequest | StuckRequest

export type PromptAnswer =
  | { kind: 'approval'; decision: 'allow' | 'allow-task' | 'deny' }
  | { kind: 'question'; text: string }
  | { kind: 'stuck'; decision: 'continue' | 'change' | 'stop'; text?: string }

// ---------- Events from the agent ----------

export type AgentStatus =
  | { phase: 'idle' }
  | { phase: 'loading'; text: string }
  | { phase: 'thinking'; text: string }
  | { phase: 'working'; text: string }
  | { phase: 'waiting'; text: string }

export type AgentEvent =
  | { type: 'task-start'; projectPath: string; task: TaskRecord; item: TimelineItem }
  | { type: 'status'; projectPath: string; status: AgentStatus }
  | { type: 'item'; projectPath: string; item: TimelineItem }
  | { type: 'delta'; projectPath: string; itemId: string; text: string }
  | { type: 'step'; projectPath: string; step: Step }
  | { type: 'output'; projectPath: string; stepId: string; chunk: string }
  | { type: 'files'; projectPath: string; taskId: string; files: ChangedFile[] }
  | { type: 'prompt'; projectPath: string; prompt: PendingPrompt | null }
  | { type: 'task-end'; projectPath: string; task: TaskRecord }
  | { type: 'tasks'; projectPath: string; tasks: TaskRecord[] }
  | { type: 'context'; projectPath: string; context: ContextUsage }

// ---------- Tool servers (set up in RulZ LLM Suite) ----------

export interface ToolServerStatus {
  name: string
  state: 'starting' | 'connected' | 'error'
  tools: number
  error?: string
  /** What the server printed (its stderr), for "Show details". */
  log: string[]
}

export interface ToolServersInfo {
  /** RulZ LLM Suite's mcp.json. */
  configPath: string
  /** Problem reading it, if any. */
  error: string | null
  /** Servers set up there (not switched off). */
  configured: string[]
  /** Servers started so far (they start the first time the agent needs a tool). */
  servers: ToolServerStatus[]
}

// ---------- Run my app ----------

export interface RunState {
  running: boolean
  command: string | null
  /** How the command was found, for "Show details". */
  source: 'saved' | 'detected' | 'none'
  output: string
  exitCode?: number | null
  /** A local web address seen in the output (dev servers). */
  url?: string
  /** For plain web pages: opened in the browser instead of run. */
  opened?: string
  /** The owner pressed Stop (so a non-zero exit code is not an error). */
  stoppedByOwner?: boolean
  /** "Run my app" was clicked but no way to run the project was found. */
  unknown?: boolean
}

export type RunEvent =
  | { type: 'output'; projectPath: string; chunk: string }
  | { type: 'state'; projectPath: string; state: RunState }

export interface FileDiff {
  path: string
  before: string | null
  after: string | null
  binary: boolean
  tooLarge: boolean
}

// ---------- Wire format (OpenAI-compatible) ----------

export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

export interface WireMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: WireToolCall[]
  tool_call_id?: string
}

/** Omit that works on each member of a union separately. */
export type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never


// ---------- Phone access (over Tailscale) ----------

export interface RemoteClient {
  id: string
  /** What the phone calls itself, e.g. "Pixel 9". */
  device: string
  address: string
  since: number
}

export interface RemoteStatus {
  enabled: boolean
  state: 'off' | 'starting' | 'on' | 'no-tailscale' | 'error'
  port: number
  /** This computer's Tailscale addresses (IPv4). */
  addresses: string[]
  /** This computer's Tailscale (MagicDNS) name, when known. */
  dnsName?: string
  clients: RemoteClient[]
  error?: string
}

/** What the phone needs to connect: a link (also shown as a QR code) and the parts to type by hand. */
export interface RemotePairing {
  url: string
  qrSvg: string
  host: string
  port: number
  token: string
}

/** One folder's subfolders, for choosing a folder from the phone. */
export interface FolderListing {
  /** Empty for the list of drives. */
  path: string
  parent: string | null
  folders: string[]
}