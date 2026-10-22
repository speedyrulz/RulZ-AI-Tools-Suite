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

export interface SidecarFile {
  path: string
  sizeBytes: number
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

export interface WireMessage {
  role: Role
  content: string | WireContentPart[]
}

export interface ChatRequest {
  requestId: string
  modelId: string
  messages: WireMessage[]
  sampling: SamplingParams
  overflow: ContextOverflow
}

export type StreamEvent =
  | { type: 'progress'; requestId: string; processed: number; total: number }
  | { type: 'delta'; requestId: string; content?: string; reasoning?: string }
  | { type: 'done'; requestId: string; stats: GenerationStats; reasoningMs?: number }
  | { type: 'error'; requestId: string; error: string; stats?: GenerationStats }
