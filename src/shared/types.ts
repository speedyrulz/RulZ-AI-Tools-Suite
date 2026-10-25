// Types shared by the main process, the preload bridge and the renderer.

export type Theme = 'dark' | 'light' | 'system'

export type KvCacheType = 'f16' | 'bf16' | 'f32' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0' | 'iq4_nl'

export const KV_CACHE_TYPES: KvCacheType[] = ['f16', 'bf16', 'f32', 'q8_0', 'q5_1', 'q5_0', 'q4_1', 'q4_0', 'iq4_nl']

/** Settings used to start one llama-server instance for a model. */
export interface LoadConfig {
  /** null = let llama.cpp pick (fits to memory, up to the trained context). */
  contextLength: number | null
  /** 'auto' lets llama.cpp fit layers to free memory; 'all' forces everything onto the GPU. */
  gpuLayers: 'auto' | 'all' | number
  splitMode: 'auto' | 'single' | 'custom'
  /** GPU index used when splitMode is 'single'. */
  mainGpu: number
  /** Proportions per GPU when splitMode is 'custom', e.g. "3,1". */
  tensorSplit: string
  flashAttention: 'auto' | 'on' | 'off'
  cacheTypeK: KvCacheType
  cacheTypeV: KvCacheType
  threads: number | null
  batchSize: number | null
  ubatchSize: number | null
  useMmap: boolean
  useMlock: boolean
  useMmproj: boolean
  /** Use the model's draft sidecar (MTP etc.) for speculative decoding when one exists. */
  useDraft: boolean
  /** Tokens to draft per step (--spec-draft-n-max); null = llama.cpp's default. */
  draftTokens: number | null
  seed: number | null
  chatTemplateFile: string
  parallel: number
  /** Free-form extra llama-server arguments, appended last. */
  extraArgs: string
}

export interface SamplingParams {
  temperature: number
  topK: number
  topP: number
  minP: number
  repeatPenalty: number
  presencePenalty: number
  frequencyPenalty: number
  /** null = generate until the model stops or the context is full. */
  maxTokens: number | null
  stop: string[]
  seed: number | null
  /** JSON schema text for structured output; empty = disabled. */
  jsonSchema: string
}

export type ContextOverflow = 'stop' | 'truncateMiddle'

export interface AppSettings {
  modelsDir: string
  /** Folder containing llama-server.exe (or the exe itself). Empty = auto-detect. */
  enginePath: string
  theme: Theme
  unloadPreviousOnLoad: boolean
  defaultLoadConfig: LoadConfig
  defaultSampling: SamplingParams
  defaultSystemPrompt: string
  contextOverflow: ContextOverflow
  expandThinking: boolean
  rightPanelOpen: boolean
  /** Offer MCP tools to the model in chats. */
  toolsEnabled: boolean
  /** Ask before each tool call unless the tool is in allowedTools. */
  confirmToolCalls: boolean
  /** Qualified tool names that run without asking. */
  allowedTools: string[]
  /** Qualified tool names hidden from the model. */
  disabledTools: string[]
  /** How tool descriptions reach the model; see ToolMode. */
  toolMode: ToolMode
  /** Token budget for tool descriptions in on-demand mode (and the threshold for auto). */
  toolBudgetTokens: number
  /** Send older tool results shortened to their beginning and end. */
  compactToolResults: boolean
  /** How many of the most recent tool results are always sent in full. */
  keepFullToolResults: number
  apiServer: ApiServerSettings
}

// ---------- Local API server ----------

export interface ApiServerSettings {
  /** Start the server when the app starts. */
  autoStart: boolean
  port: number
  /** 'localhost' = 127.0.0.1 only; 'lan' = all interfaces. */
  bind: 'localhost' | 'lan'
  /** Empty = no authentication. */
  apiKey: string
  cors: boolean
  /** Unload models the server loaded after this many idle minutes; 0 = never. */
  idleUnloadMinutes: number
  /** Model used when a request names none; empty = the loaded model, or an error. */
  defaultModelId: string
}

export interface ApiServerStatus {
  running: boolean
  host: string
  port: number
  error?: string
  startedAt?: number
  requestCount: number
  /** Instances the server loaded itself, with when they were last used. */
  jitInstances: { instanceId: string; lastUsedAt: number }[]
}

export interface ApiRequestLog {
  id: string
  at: number
  method: string
  path: string
  model?: string
  status?: number
  durationMs?: number
  stream?: boolean
  promptTokens?: number
  completionTokens?: number
  tokensPerSecond?: number
  error?: string
  /** 'loading' while the server waits for a model to load. */
  phase?: 'loading' | 'running' | 'done'
}

/**
 * 'all': every enabled tool is described in every request.
 * 'dynamic': the model searches and loads the tools it needs, within toolBudgetTokens.
 * 'auto': 'all' while the enabled tools fit the budget, otherwise 'dynamic'.
 */
export type ToolMode = 'auto' | 'all' | 'dynamic'

/** Which tools a chat request offers the model. */
export interface ToolRequest {
  mode: 'all' | 'dynamic'
  /** Loaded tools (qualified names) in dynamic mode. */
  active: string[]
}

export interface ToolSearchHit {
  name: string
  server: string
  summary: string
}

// ---------- MCP ----------

export interface McpServerConfig {
  command?: string
  args?: string[]
  env?: Record<string, string>
  cwd?: string
  url?: string
  headers?: Record<string, string>
  disabled?: boolean
}

export interface McpToolInfo {
  /** Name the model sees: "<server>__<tool>". */
  qualifiedName: string
  server: string
  name: string
  description?: string
  inputSchema: unknown
}

export interface McpServerStatus {
  name: string
  transport: 'stdio' | 'http'
  state: 'disabled' | 'connecting' | 'connected' | 'error'
  error?: string
  tools: McpToolInfo[]
  log: string[]
}

export interface ToolCall {
  id: string
  /** Qualified tool name as the model called it. */
  name: string
  /** Raw JSON arguments from the model. */
  arguments: string
  status: 'pending' | 'running' | 'done' | 'error' | 'denied'
  result?: string
}

export interface ToolResult {
  text: string
  isError: boolean
}

// ---------- Models ----------

export interface GgufInfo {
  architecture: string
  type?: string
  name?: string
  sizeLabel?: string
  fileType?: number
  quantization?: string
  contextLength?: number
  blockCount?: number
  embeddingLength?: number
  headCount?: number
  headCountKv?: number
  keyLength?: number
  valueLength?: number
  expertCount?: number
  expertUsedCount?: number
  slidingWindow?: number
  /** Share of layers using sliding-window attention (their KV cache is capped at the window). */
  slidingWindowFraction?: number
  hasChatTemplate: boolean
}

/** Speculative-decoding head formats llama.cpp can load from a sidecar file. */
export type DraftKind = 'mtp' | 'eagle3' | 'dflash' | 'dspark'

export interface SidecarFile {
  path: string
  sizeBytes: number
  /** Set on draft sidecars; decides the --spec-type passed to llama-server. */
  kind?: DraftKind
}

export interface ModelEntry {
  /** Path of the primary file relative to the models folder, with forward slashes. */
  id: string
  name: string
  /** Folder relative to the models folder ('' for the root). */
  folder: string
  path: string
  files: string[]
  sizeBytes: number
  mmproj?: SidecarFile
  draft?: SidecarFile
  info: GgufInfo | null
  error?: string
  modifiedAt: number
}

export interface MemoryEstimate {
  weightsBytes: number
  kvBytes: number | null
  overheadBytes: number
  gpuBytes: number
  totalBytes: number
  contextLength: number | null
  /** Free VRAM summed over the GPUs this config would use, if known. */
  freeVramBytes: number | null
  freeRamBytes: number
  verdict: 'fits' | 'partial' | 'tight' | 'unknown'
}

// ---------- Engine and instances ----------

export interface EngineDevice {
  id: string
  name: string
  totalMiB?: number
  freeMiB?: number
}

export interface EngineInfo {
  ok: boolean
  exePath: string | null
  version?: string
  devices: EngineDevice[]
  error?: string
  searched: string[]
}

export type InstanceStatus = 'loading' | 'ready' | 'stopping'

export interface LoadedInstance {
  instanceId: string
  modelId: string
  modelName: string
  port: number
  pid: number | null
  status: InstanceStatus
  /** 0..1 while loading. */
  progress: number
  stage: string
  config: LoadConfig
  startedAt: number
  /** Actual context size reported by the server once ready. */
  contextLength: number | null
  hasVision: boolean
  estimate: MemoryEstimate | null
}

export interface InstanceErrorEvent {
  modelName: string
  error: string
  logTail: string[]
}

export interface GpuStat {
  index: number
  name: string
  memUsedMiB: number
  memTotalMiB: number
  utilPct: number
}

export interface HardwareSnapshot {
  gpus: GpuStat[]
  ramUsedBytes: number
  ramTotalBytes: number
  at: number
}

// ---------- Chat ----------

export type Role = 'system' | 'user' | 'assistant'

export interface Attachment {
  id: string
  kind: 'image' | 'text'
  name: string
  mime: string
  sizeBytes: number
  dataUrl?: string
  text?: string
}

export interface GenerationStats {
  tokensPerSecond?: number
  predictedTokens?: number
  promptTokens?: number
  cachedTokens?: number
  ttftMs?: number
  totalMs?: number
  stopReason?: string
  modelName?: string
}

export interface ChatMessage {
  id: string
  role: Role
  content: string
  reasoning?: string
  reasoningMs?: number
  attachments?: Attachment[]
  stats?: GenerationStats
  error?: string
  /** Tool calls the assistant made in this turn, with their results once run. */
  toolCalls?: ToolCall[]
  createdAt: number
}

export interface Conversation {
  id: string
  title: string
  folder: string | null
  pinned: boolean
  createdAt: number
  updatedAt: number
  modelId: string | null
  systemPrompt: string
  sampling: SamplingParams
  presetId: string | null
  messages: ChatMessage[]
  /** Set once the user has answered (or acted on) the "context nearly full" handoff prompt. */
  handoffPrompted?: boolean
  /** Tools the model has loaded in on-demand mode, least recently used first. */
  activeTools?: string[]
}

export interface ConversationSummary {
  id: string
  title: string
  folder: string | null
  pinned: boolean
  updatedAt: number
  messageCount: number
  modelId: string | null
}

export interface Preset {
  id: string
  name: string
  systemPrompt: string
  sampling: SamplingParams
  updatedAt: number
}

/** OpenAI-style message as sent to llama-server. */
export type WireContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }

export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

export interface WireMessage {
  role: Role | 'tool'
  content: string | WireContentPart[]
  tool_calls?: WireToolCall[]
  tool_call_id?: string
}

export interface ChatRequest {
  requestId: string
  modelId: string
  messages: WireMessage[]
  sampling: SamplingParams
  overflow: ContextOverflow
  /** Ask the chat template to skip reasoning (honoured by templates with an enable_thinking switch). */
  disableThinking?: boolean
  /** Tools to offer the model; omitted = no tools. */
  tools?: ToolRequest
}

export type StreamEvent =
  | { type: 'progress'; requestId: string; processed: number; total: number }
  | { type: 'delta'; requestId: string; content?: string; reasoning?: string }
  | { type: 'done'; requestId: string; stats: GenerationStats; reasoningMs?: number; toolCalls?: WireToolCall[] }
  | { type: 'error'; requestId: string; error: string; stats?: GenerationStats }
