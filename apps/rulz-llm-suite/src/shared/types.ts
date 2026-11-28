// Types shared by the main process, the preload bridge and the renderer.
import type { CloudModelTag } from './cloud'
import type { RemoteHost } from './remote'

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
  /**
   * LoRA adapters loaded with the model (one --lora each), by AdapterEntry id, in this order. `scale` is
   * the default strength: what API requests and chats without their own adapter settings get.
   */
  adapters: LoadAdapter[]
}

/** An adapter by AdapterEntry id with a strength; 0 = off, 1 = as trained. */
export interface LoadAdapter {
  id: string
  scale: number
}

/** An adapter a running instance loaded. Its position in LoadedInstance.adapters is llama-server's adapter id. */
export interface InstanceAdapter {
  /** AdapterEntry id (path relative to the models folder). */
  id: string
  name: string
  path: string
  /** Default scale, set on the server after loading. */
  scale: number
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
  /** DRY: penalises text that would repeat an earlier passage; multiplier 0 = off. */
  dryMultiplier: number | null
  dryBase: number | null
  dryAllowedLength: number | null
  /** -1 = the whole context, 0 = off. */
  dryPenaltyLastN: number | null
  drySequenceBreakers: string[] | null
  /** XTC: now and then drops the likeliest tokens; probability 0 = off. */
  xtcProbability: number | null
  xtcThreshold: number | null
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
  /** DRY: penalises text that would repeat an earlier passage; 0 = off (typical 0.8). */
  dryMultiplier: number
  dryBase: number
  dryAllowedLength: number
  /** -1 = the whole context, 0 = off. */
  dryPenaltyLastN: number
  /** Tokens that reset DRY's repeat matching, e.g. newline, colon, quote, asterisk. */
  drySequenceBreakers: string[]
  /** XTC: now and then drops the likeliest tokens; 0 = off (typical 0.5). */
  xtcProbability: number
  /** Tokens above this chance are candidates for XTC; above 0.5 disables XTC. */
  xtcThreshold: number
  /** Words to discourage, sent as logit_bias. */
  avoidWords: string[]
  /** Bias applied to each avoided word: 5 means -5, 100 means effectively banned. */
  avoidStrength: number
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
  /** Writing mode: folder holding the novel projects; '' = <Documents>\RulZ Novels. */
  writingRoot: string
  /** Phone access (RulZ Writer for Android): listen on the Tailscale address so the phone can connect. */
  phoneEnabled: boolean
  /** Port the phone connects to. */
  phonePort: number
  /** Pairing code: '' until phone access is first turned on. Never sent to the renderer except by phone:pairing. */
  phoneToken: string
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
  /**
   * Practice task from the RE training kit. `kitDir` is the kit the task was copied from, so it is
   * graded against that kit even if the kit folder setting changes later.
   */
  practice?: { taskId: string; family: string; tier: string; answer?: string; expected?: string; kitDir?: string }
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
  /** Names of the models that replied in the session; empty when nothing was recorded. */
  models: string[]
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
  /** Keys (see modelKey) of the models that answered; empty when not known. */
  models: string[]
  /** Recorded model responses and tool rounds (0 when nothing was recorded). */
  responses: number
  toolRounds: number
  updatedAt: number
}

/** A practice task with how often it was tried and solved. */
export interface RePracticeTaskStats extends RePracticeTask {
  attempts: number
  solved: number
  /** Only in an overview asked for one model. */
  failed?: number
}

/** Everything other apps need to show and start RE tasks (GET /v1/re/overview). */
export interface ReOverview {
  object: 're.overview'
  /** Whether requests marked as RE sessions are recorded. */
  recording: boolean
  /** Newest first. */
  sessions: ReSessionSummary[]
  /** `model`: the model key the counts are for (see modelKey); present only when the overview was asked for one model. */
  practice: { kitDir: string; error?: string; warning?: string; model?: string; sets: RePracticeSet[]; tasks: RePracticeTaskStats[] }
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
  /** Id of the task set (folder of generated tasks) the task belongs to; 'original' is <kit>/build. */
  set: string
}

/** A folder of generated tasks: the original (<kit>/build) or one under <kit>/sets/<id>. */
export interface RePracticeSet {
  id: string
  label: string
  count: number
  created?: string
  families: string[]
  seed?: number
}

export interface RePracticeInfo {
  kitDir: string
  error?: string
  /** Set when something is odd but the list is usable (a task id in two sets). */
  warning?: string
  tasks: RePracticeTask[]
  /** Task sets, the original first. */
  sets: RePracticeSet[]
}

/** What re:generateSet takes (the main process checks every field again). */
export interface ReGenerateRequest {
  label: string
  perFamily: number
  families: string[]
  /** Integer 0..2^31-1; random when left out. */
  seed?: number
}

/** Progress of the running set generation (re:generateProgress). */
export interface ReGenerateProgress {
  done: number
  total: number
  line: string
}

export interface ReExportOptions {
  outDir: string
  /** Outcomes to include; 'solved' only is rejection sampling (train on verified successes). */
  outcomes: ReOutcome[]
  /** Leave out sessions whose context was trimmed. */
  skipTrimmed: boolean
  /** Practice tasks: keep at most this many sessions per task (fewest tool rounds first). */
  maxPerTask: number
  /** Models to leave out: a session in which one of them replied is not exported. */
  skipModels?: string[]
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
  /** Practice sessions: leave out the built-in tools they never get (read_pdf). */
  practice?: boolean
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
  /** Per-layer sliding-window flags (true = sliding window) when the GGUF lists them; the loop layout needs them per layer. */
  slidingWindowPattern?: boolean[]
  /** Output vocabulary size (<arch>.vocab_size); sizes the logits scratch buffer of Inkling models. */
  vocabSize?: number
  /** inkling2_loop: layers [start, end) are run `count` times; llama.cpp keeps a KV cache slot for every pass. */
  loop?: { start: number; end: number; count: number; varying: boolean }
  hasChatTemplate: boolean
  /** Built-in multi-token-prediction layers (nextn), usable as a draft without a separate file. */
  mtpLayers?: number
  /** Adapter files: adapter.type ("lora", "alora"). */
  adapterType?: string
  /** Adapter files: adapter.lora.alpha. */
  loraAlpha?: number
  /** Adapter files: the base model it was trained on (general.base_model.0.name, else general.basename). */
  baseModelName?: string
}

/** A LoRA adapter GGUF (general.type = "adapter") found in the models folder. */
export interface AdapterEntry {
  /** Path relative to the models folder, with forward slashes. */
  id: string
  /** File name without .gguf. */
  name: string
  path: string
  sizeBytes: number
  /** general.architecture; adapters are offered to models with the same one. */
  architecture: string
  info: GgufInfo | null
  error?: string
  modifiedAt: number
}

/** Converting a Hugging Face PEFT LoRA folder to GGUF with llama.cpp's convert_lora_to_gguf.py. */
export interface AdapterConvertRequest {
  /** Folder with adapter_config.json and adapter_model.safetensors (or .bin). */
  peftDir: string
  /** Folder with the base model's config.json (weights not needed); empty = use baseModelId or the adapter config. */
  baseDir: string
  /** Hugging Face id of the base model, read from the Hub when baseDir is empty; empty = from adapter_config.json. */
  baseModelId: string
  outType: 'f32' | 'f16' | 'bf16' | 'q8_0' | 'auto'
  /** File name for the result, written into the models folder (".gguf" added when missing). */
  outName: string
}

export interface AdapterConvertResult {
  ok: boolean
  /** The written file, when ok. */
  outPath?: string
  error?: string
  /** Last lines of the converter's output. */
  logTail: string[]
}

export interface AdapterConverterStatus {
  available: boolean
  python: string | null
  /** Path of convert_lora_to_gguf.py, found next to the llama.cpp build the engine uses. */
  script: string | null
  /** Why it isn't available. */
  error?: string
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
  /** Runs on a vast.ai rental (docs/vast-design.md); path, files and sidecars then describe the source, not files to load here. */
  remote?: boolean
  /** An OpenRouter model (docs/cloud-design.md): not a file, no load settings; never set together with `remote`. */
  cloud?: CloudModelTag
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
  /** LoRA adapter weights included in the totals (0 when none). */
  adapterBytes: number
  /** Inkling models: the prompt scratch buffer (relative-position bias) that is part of the GPU numbers. */
  scratch?: ScratchEstimate
  /** Inkling loop models: the --tensor-split LLM Suite passes so the weights land as the settings say. */
  loopSplit?: LoopSplitInfo
}

export interface ScratchEstimate {
  /** The scratch buffer of the GPU that overflows, else the largest one on a GPU (all of it when GPUs are unknown). */
  bytes: number
  /** Micro-batch size the estimate used. */
  ubatch: number
  /** The GPU whose scratch buffer does not fit although its weights do; null when none. */
  overflowGpu: number | null
  /** Largest micro-batch (512, 256, 128, 64, 32) below the used one with which every GPU fits; null when none does or none is needed. */
  suggestedUbatch: number | null
}

export interface LoopSplitInfo {
  /** The value after --tensor-split. */
  passed: string
  /** The weight split the user asked for (or free memory gives), in whole percents per GPU. */
  percents: number[]
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
  /** Adapters this instance loaded, in llama-server id order (index = id). Empty when none. */
  adapters: InstanceAdapter[]
  /** The vast.ai rental it runs on; absent = this PC. */
  host?: RemoteHost
  /** The connection to the rental is being re-made; the model is still loaded. */
  reconnecting?: boolean
  /** A connected OpenRouter model: no process, no port (0), no GPU; requests go through InstanceManager.fetch. */
  cloud?: { provider: 'openrouter'; free: boolean }
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
  /** Prompt processing speed (tokens actually processed, cached ones excluded, per second). */
  promptTokensPerSecond?: number
  ttftMs?: number
  totalMs?: number
  stopReason?: string
  modelName?: string
  /** What OpenRouter charged for this reply, in micro-dollars (cloud models only). */
  costMicro?: number
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
  /**
   * This chat's adapter strengths, by AdapterEntry id. Absent or null = the loaded model's defaults. An
   * adapter the model loaded but this list leaves out keeps its default; ids the model didn't load are ignored.
   */
  adapters?: LoadAdapter[] | null
  /** Goes up with every save by the practice pool; a window copy with an older number is stale and is not saved. */
  poolRev?: number
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
  /** Use this instance of the model (it may be loaded on several hosts); an error if it is gone. */
  instanceId?: string
  /**
   * Any string the caller keeps the same for one chat (a conversation id, a run id). The requests of one session
   * stay on the instance the first one used while that is ready, so a tool round finds its prompt cache. Ignored
   * when `instanceId` is set.
   */
  sessionKey?: string
  messages: WireMessage[]
  sampling: SamplingParams
  overflow: ContextOverflow
  /** Ask the chat template to skip reasoning (honoured by templates with an enable_thinking switch). */
  disableThinking?: boolean
  /** Tools to offer the model; omitted = no tools. */
  tools?: ToolRequest
  /** Record the request and reply for this conversation's RE training data. */
  record?: { conversationId: string }
  /** The conversation's adapter strengths (Conversation.adapters); sent to llama-server as `lora`. */
  adapters?: LoadAdapter[] | null
  /** The caller already fitted the messages to the context (Writing mode): do not trim; fail if too large. */
  prefitted?: boolean
}

export type StreamEvent =
  | { type: 'progress'; requestId: string; processed: number; total: number }
  | { type: 'delta'; requestId: string; content?: string; reasoning?: string }
  | { type: 'done'; requestId: string; stats: GenerationStats; reasoningMs?: number; toolCalls?: WireToolCall[] }
  /** `transport`: the server could not be reached or the connection broke; says nothing about the model. */
  /** `fatal`: the same request would fail again (spending limit, key, credits), so a batch run stops. */
  | { type: 'error'; requestId: string; error: string; stats?: GenerationStats; transport?: boolean; fatal?: boolean }
