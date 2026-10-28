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
  /** Generation settings the owner changed, per model id; anything left out uses the recommended value. */
  modelSampling: Record<string, SamplingSettings>
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

/** How the model writes its answers. Missing = the recommended value for that model. */
export interface SamplingSettings {
  temperature?: number
  topP?: number
  topK?: number
  repeatPenalty?: number
  /** Longest single answer, in tokens. */
  maxOutputTokens?: number
  /** Turn the template's thinking on or off; missing = the model's default. */
  thinking?: 'on' | 'off'
  /** For templates that take one (reasoning_effort); missing = the template's own. */
  reasoningEffort?: ReasoningEffort
  /** Most tokens to spend thinking before the answer is forced; 0 = skip thinking. */
  thinkingBudget?: number
  /** Text added when the thinking budget runs out, e.g. "Time to answer." */
  thinkingBudgetMessage?: string
}

export type ReasoningEffort = 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

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

export type KvCacheType = 'f16' | 'bf16' | 'f32' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0' | 'iq4_nl'

/** The load settings RulZ LLM Suite lets other apps change (GET/POST /v1/models/{id}/load-config). */
export interface LoadSettings {
  /** null = as much as fits in memory. */
  contextLength: number | null
  gpuLayers: 'auto' | 'all' | number
  flashAttention: 'auto' | 'on' | 'off'
  cacheTypeK: KvCacheType
  cacheTypeV: KvCacheType
  parallel: number
  /** GPU split; missing from RulZ LLM Suite before 0.3.7. */
  splitMode?: 'auto' | 'single' | 'custom'
  mainGpu?: number
  /** Proportions per GPU when splitMode is 'custom', e.g. "60,40". */
  tensorSplit?: string
  useDraft: boolean
  /** Draft (MTP) tuning; missing from RulZ LLM Suite before 0.3.7. null = llama.cpp's default. */
  draftTokens?: number | null
  draftMinTokens?: number | null
  draftPMin?: number | null
  draftSampling?: 'auto' | 'greedy' | 'probabilistic'
  draftGpu?: 'auto' | 'gpu' | 'cpu'
  draftCacheTypeK?: KvCacheType
  draftCacheTypeV?: KvCacheType
  /** Which draft: null = automatic, 'builtin' = the model's own MTP layers, or a DraftChoice id.
   *  Missing from RulZ LLM Suite before 0.3.9. */
  draftFile?: string | null
}

/** A draft RulZ LLM Suite can use for a model: its built-in MTP layers or a draft file in the models folder. */
export interface DraftChoice {
  id: string
  name: string
  folder: string
  kind: string
  builtIn: boolean
}

export interface GpuInfo {
  index: number
  name: string
  memUsedMiB: number
  memTotalMiB: number
}

/** RulZ LLM Suite's memory estimate for a load config. */
export interface LoadEstimate {
  gpuBytes: number
  totalBytes: number
  verdict: 'fits' | 'partial' | 'tight' | 'unknown'
  perGpu: { index: number; name: string; share: number; bytes: number; freeBytes: number; totalBytes: number }[]
}

export interface ModelLoadInfo {
  /** What the model loads with next time. */
  config: LoadSettings
  /** False while the model still uses RulZ LLM Suite's general defaults. */
  custom: boolean
  maxContext?: number
  state: 'not-loaded' | 'loading' | 'ready' | 'stopping' | 'error'
  loadedContext?: number
  /** Whether the model comes with a draft (MTP) head; undefined when RulZ doesn't say. */
  hasDraft?: boolean
  /** What the automatic choice uses; null = nothing found. Undefined before RulZ LLM Suite 0.3.9. */
  defaultDraft?: { name: string; builtIn: boolean } | null
  /** Drafts that can be picked; undefined before RulZ LLM Suite 0.3.9. */
  draftChoices?: DraftChoice[]
  gpus: GpuInfo[]
  estimate?: LoadEstimate
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
  | { id: string; kind: 'assistant'; text: string; taskId: string; /** The model's thinking before this reply, when it sent any. */ reasoning?: string }
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
  | { type: 'reasoning'; projectPath: string; itemId: string; text: string }
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