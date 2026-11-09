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
  /**
   * Mixture-of-experts offload: keep expert weights in system RAM while attention and shared
   * weights stay on the GPU. 'all' = --cpu-moe, a number N = --n-cpu-moe N (experts of the first N layers).
   */
  cpuMoe: 'off' | 'all' | number
  /** Custom tensor placement rules (--override-tensor), one "pattern=buffer" per line, e.g. "exps=CPU". */
  overrideTensors: string
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
  /** Keep the vision projector in system RAM (--no-mmproj-offload): frees its VRAM, encodes images more slowly. */
  mmprojOnCpu: boolean
  /** Use the model's draft sidecar (MTP etc.) for speculative decoding when one exists. */
  useDraft: boolean
  /** Which draft to use: null = the one found with the model, 'builtin' = the model's own MTP layers,
   *  or the path of a draft file anywhere in the models folder. */
  draftFile: string | null
  /** Tokens to draft per step (--spec-draft-n-max); null = llama.cpp's default. */
  draftTokens: number | null
  /** Fewest tokens worth drafting (--spec-draft-n-min); null = llama.cpp's default (0). */
  draftMinTokens: number | null
  /** Stop drafting once the draft's confidence falls below this (--spec-draft-p-min); null = default. */
  draftPMin: number | null
  /** How the draft picks its tokens (--spec-draft-sampling); 'auto' = llama.cpp's default. */
  draftSampling: 'auto' | 'greedy' | 'probabilistic'
  /** Where the draft head runs (--spec-draft-ngl): 'gpu' = all layers, 'cpu' = none. */
  draftGpu: 'auto' | 'gpu' | 'cpu'
  draftCacheTypeK: KvCacheType
  draftCacheTypeV: KvCacheType
  seed: number | null
  chatTemplateFile: string
  parallel: number
  /** Free-form extra llama-server arguments, appended last. */
  extraArgs: string
  /** Sampling defaults for requests that reach this model through the API server. */
  sampling: ModelSampling
}

/**
 * A model's own sampling defaults. Each value is passed to llama-server at load time, so a request
 * that leaves it out gets it; null = llama.cpp's default. The app's chat sends all of its own values.
 */
export interface ModelSampling {
  temperature: number | null
  topK: number | null
  topP: number | null
  minP: number | null
  repeatPenalty: number | null
  presencePenalty: number | null
  /** Also replace the values API requests send (e.g. Claude Code's temperature), not only fill in missing ones. */
  override: boolean
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
  /** Turn the template's thinking on or off (enable_thinking); 'auto' = the model's default. */
  thinking: 'auto' | 'on' | 'off'
  /** Effort level for templates that take one (reasoning_effort); 'default' = the template's own. */
  reasoningEffort: ReasoningEffort
  /** Most tokens to spend thinking before the answer is forced; null = unlimited, 0 = skip thinking. */
  thinkingBudget: number | null
  /** Text added when the thinking budget runs out, e.g. "Time to answer."; empty = none. */
  thinkingBudgetMessage: string
}

export type ReasoningEffort = 'default' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export const REASONING_EFFORTS: ReasoningEffort[] = ['default', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']

export type ContextOverflow = 'stop' | 'truncateMiddle'

export interface AppSettings {
  modelsDir: string
  /** Folder containing llama-server.exe (or the exe itself). Empty = auto-detect. */
  enginePath: string
  theme: Theme
  /** False until the user picks a look in the first-run "Choose your look" dialog (or changes the theme). */
  themeChosen: boolean
  unloadPreviousOnLoad: boolean
  defaultLoadConfig: LoadConfig
  defaultSampling: SamplingParams
  defaultSystemPrompt: string
  contextOverflow: ContextOverflow
  expandThinking: boolean
  /** Leave a reasoning block open once the answer starts, instead of folding it away. */
  keepThinkingOpen: boolean
  rightPanelOpen: boolean
  /** Minimizing hides the window to the system tray instead of the taskbar. */
  minimizeToTray: boolean
  /** The close button hides the window to the system tray; Quit from the tray menu exits. */
  closeToTray: boolean
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
  /** Reverse engineering mode. */
  re: ReSettings
}

// ---------- Reverse engineering mode ----------

export interface ReSettings {
  /** Model used for new RE sessions (id relative to the models folder); empty = the loaded model. */
  modelId: string
  /** RE training kit folder (build/tasks.jsonl and build/bin/<task>/) for practice tasks. */
  kitDir: string
  /** Record what the model sees and writes in RE sessions, for fine-tuning. */
  collect: boolean
  /** Run REA's static-analysis tools (and calc) without asking. Tools that run or attach to programs still ask. */
  autoApproveStatic: boolean
  /** Rounds of tool calls an RE session may make before it has to answer. */
  maxToolRounds: number
  /** Token budget for loaded tool descriptions in RE sessions (on-demand tool loading). */
  toolBudgetTokens: number
  /** Record API requests that other apps (RulZ Coding Agent) mark as RE sessions. */
  recordApi: boolean
  /** Rounds of tool calls a clone session may make (rebuilding takes far more than answering). */
  cloneMaxToolRounds: number
}

/** A clone session rebuilds part or all of the target as new source code in `outputDir`. */
export interface ReClone {
  /** 'pieces': chosen parts only; 'full': the whole program. */
  scope: 'pieces' | 'full'
  /** What to rebuild (pieces), or what to keep or leave out (full). */
  details: string
  /** Language for the new code; '' = the model picks (usually the original's). */
  language: string
  /** Folder the new code is written to; the only place the session's write tool may write. */
  outputDir: string
}

export type ReOutcome = 'unrated' | 'solved' | 'partial' | 'failed'

/** Set on a conversation that is a reverse-engineering session. */
export interface ReSession {
  /** File or folder being analysed. */
  target: string
  goal: string
  /** Record this session's requests for training. */
  collect: boolean
  outcome: ReOutcome
  /** The outcome came from a practice task's automatic check, not the user. */
  autoGraded?: boolean
  notes?: string
  /** Practice task from the RE training kit. */
  practice?: { taskId: string; family: string; tier: string; answer?: string; expected?: string }
  /** Set for a clone session. */
  clone?: ReClone
}

/** One RE session's recorded data, as listed on the Reverse Engineering page. */
export interface ReTraceInfo {
  conversationId: string
  /** Model responses recorded. */
  responses: number
  /** Responses that asked for tools. */
  toolRounds: number
  /** Context was trimmed at some point, so the sample lacks part of the session. */
  trimmed: boolean
  bytes: number
  updatedAt: number
  /** Set for a session another app ran through the API server (it has no chat here). */
  api?: ApiReSession
}

/** An RE session recorded from the API server, described by the app that ran it. */
export interface ApiReSession {
  /** The app that ran it, e.g. "RulZ Coding Agent". */
  client: string
  title: string
  target: string
  goal: string
  outcome: ReOutcome
  notes?: string
  createdAt: number
  /** What kind of RE task it was; absent = analyse. */
  kind?: ReKind
  /** A practice task from the training kit, with the answer given and the expected one once graded. */
  practice?: { taskId: string; family: string; tier: string; answer?: string; expected?: string }
}

export type ReKind = 'analyse' | 'practice' | 'clone-pieces' | 'clone-full'

/** One RE session as other apps see it (GET /v1/re/overview). */
export interface ReSessionSummary {
  /** Conversation id here, or api-<id> for a session another app ran. */
  id: string
  title: string
  /** "RulZ LLM Suite", or the app that ran it. */
  source: string
  kind: ReKind
  target: string
  goal: string
  outcome: ReOutcome
  practice?: { taskId: string; family: string; tier: string; answer?: string; expected?: string }
  /** Recorded model responses and tool rounds (0 when nothing was recorded). */
  responses: number
  toolRounds: number
  updatedAt: number
}

/** A practice task with how often it was tried and solved. */
export interface RePracticeTaskStats extends RePracticeTask {
  attempts: number
  solved: number
}

/** Everything other apps need to show and start RE tasks (GET /v1/re/overview). */
export interface ReOverview {
  object: 're.overview'
  /** Whether requests marked as RE sessions are recorded. */
  recording: boolean
  /** Newest first. */
  sessions: ReSessionSummary[]
  practice: { kitDir: string; error?: string; tasks: RePracticeTaskStats[] }
  /** Prompt templates; {target} is replaced by the target path. */
  goals: { label: string; text: string }[]
  clonePieces: { label: string; text: string }[]
  cloneLanguages: string[]
}

export interface RePracticeTask {
  id: string
  family: string
  tier: string
  binary: string
  /** Prompt with {exe} filled in for the task's work folder. */
  prompt: string
}

export interface RePracticeInfo {
  kitDir: string
  error?: string
  tasks: RePracticeTask[]
}

export interface ReExportOptions {
  outDir: string
  /** Outcomes to include; 'solved' only is rejection sampling (train on verified successes). */
  outcomes: ReOutcome[]
  /** Leave out sessions whose context was trimmed. */
  skipTrimmed: boolean
  /** Practice tasks: keep at most this many sessions per task (fewest tool rounds first). */
  maxPerTask: number
}

export interface ReExportResult {
  outDir: string
  kept: number
  dropped: Record<string, number>
  files: string[]
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
  /**
   * When a request names a model the library doesn't have (e.g. Claude Code asking for
   * "claude-sonnet-…"), answer with the loaded model or the default model instead of a 404.
   */
  unknownModelFallback: boolean
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
  /** The model name the request asked for, when the server answered with a different one. */
  requestedModel?: string
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
  /** Clone sessions: offer the write/read file tools, limited to this scope. */
  files?: FileToolScope
}

/** Where the built-in write_file and read_file tools may work (clone sessions only). */
export interface FileToolScope {
  /** The only folder write_file may write in. */
  writeRoot: string
  /** The analysed file or folder; read_file may read it too, and write_file never touches it. */
  target: string
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
  transport: 'stdio' | 'http' | 'builtin'
  state: 'disabled' | 'connecting' | 'connected' | 'error'
  error?: string
  tools: McpToolInfo[]
  log: string[]
}

/** mcp.json as the app last read it, and any problem reading or parsing it. */
export interface McpConfigState {
  text: string
  path: string
  /** Why the file couldn't be read or used; null when it loaded fine. */
  error: string | null
  /** True when the file couldn't be read at all, so `text` may not match what is on disk. */
  readFailed: boolean
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
  /** Bytes of routed-expert weights per layer (all shards), for the MoE offload estimate. */
  expertBytesByLayer?: number[]
  slidingWindow?: number
  /** Share of layers using sliding-window attention (their KV cache is capped at the window). */
  slidingWindowFraction?: number
  hasChatTemplate: boolean
  /** Built-in multi-token-prediction layers (nextn), usable as a draft without a separate file. */
  mtpLayers?: number
}

/** Speculative-decoding head formats llama.cpp can load from a sidecar file. */
export type DraftKind = 'mtp' | 'eagle3' | 'dflash' | 'dspark'

export interface SidecarFile {
  path: string
  sizeBytes: number
  /** Set on draft sidecars; decides the --spec-type passed to llama-server. */
  kind?: DraftKind
  /** A draft made of MTP layers inside the model file itself (path is the model; no --spec-draft-model). */
  builtIn?: boolean
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
  /** The draft used by default: built-in MTP layers, or else the separate draft file. */
  draft?: SidecarFile
  /** A separate draft file in the same folder, even when built-in layers are preferred. */
  draftSidecar?: SidecarFile
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
  /** Per-GPU breakdown of gpuBytes for the GPUs the model would use; empty when GPUs are unknown. */
  perGpu: GpuEstimate[]
  /** Expert weights the MoE offload setting keeps in system RAM (0 when off or unknown). */
  cpuExpertBytes: number
}

export interface GpuEstimate {
  index: number
  name: string
  /** Fraction of the split model this GPU holds (0..1). */
  share: number
  bytes: number
  freeBytes: number
  totalBytes: number
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
  /** The user stopped this reply to add a note (Send now); the next user message carries its thinking. */
  interrupted?: boolean
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
  /** Reverse-engineering session details; absent for ordinary chats. */
  re?: ReSession
}

export interface ConversationSummary {
  id: string
  title: string
  folder: string | null
  pinned: boolean
  updatedAt: number
  messageCount: number
  modelId: string | null
  /** Outcome of an RE session; absent for ordinary chats. */
  reOutcome?: ReOutcome
  reTarget?: string
  rePracticeTask?: string
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
  /** Assistant thinking sent back to the template (RE sessions, within the current tool loop). */
  reasoning_content?: string
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
  /** Record the request and reply for this conversation's RE training data. */
  record?: { conversationId: string }
}

export type StreamEvent =
  | { type: 'progress'; requestId: string; processed: number; total: number }
  | { type: 'delta'; requestId: string; content?: string; reasoning?: string }
  | { type: 'done'; requestId: string; stats: GenerationStats; reasoningMs?: number; toolCalls?: WireToolCall[] }
  | { type: 'error'; requestId: string; error: string; stats?: GenerationStats }
