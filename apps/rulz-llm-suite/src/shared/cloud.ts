// OpenRouter models: the contract between the main process, the window and the other layers (docs/cloud-design.md).
// Money is kept in whole micro-dollars (1 USD = 1,000,000) wherever it is added up; prices are micro-dollars per token
// and may be fractions.

export const CLOUD_ID_PREFIX = 'openrouter/'
export const DEFAULT_DAILY_LIMIT_USD = 1
export const MIN_DAILY_LIMIT_USD = 0.01
/** What the app tells Chat, Writing and the others the context is, unless the user raises it: they fill what they get. */
export const DEFAULT_CONTEXT_LIMIT = 32_768
export const MIN_CONTEXT_LIMIT = 2048
export const DEFAULT_REPLY_LIMIT = 8192
export const MIN_REPLY_LIMIT = 256
/** Batch practice sessions at once on a cloud model. */
export const MAX_CLOUD_SESSIONS = 4

/** The prices of a model, in micro-dollars. The Max ones are the highest of base and every override. */
export interface CloudPrices {
  /** Per prompt token; the dearest of plain prompt and cache write. */
  promptMax: number
  /** Per reply token; the dearest of completion and internal reasoning. */
  completionMax: number
  /** Per request and per image (0 = not billed apart). */
  request: number
  image: number
  /** The plain base prices, for showing. */
  prompt: number
  completion: number
}

export interface CloudReasoning {
  mandatory: boolean
  /** Efforts the model accepts; empty = not stated. */
  efforts: string[]
  supportsMaxTokens: boolean
}

/** One usable model of OpenRouter's list. */
export interface CloudModelInfo {
  slug: string
  name: string
  description: string
  contextLength: number
  /** Most reply tokens the provider allows; null = not stated. */
  maxReply: number | null
  prices: CloudPrices
  /** Every price is 0. */
  free: boolean
  vision: boolean
  tools: boolean
  /** null = the model has no thinking that can be set. */
  reasoning: CloudReasoning | null
  /** OpenRouter's supported_parameters. */
  supported: string[]
  /** Unix seconds. */
  created: number
  /** ISO date after which OpenRouter drops it; null = none. */
  expires: string | null
}

/** A model the user added. */
export interface CloudModelDef {
  slug: string
  /** Tokens of context the app may use (MIN_CONTEXT_LIMIT .. info.contextLength). */
  contextLimit: number
  /** Most reply tokens of one request. */
  replyLimit: number
  /** Other apps (API server, RulZ Coding Agent, phones) may use it. */
  allowApps: boolean
  info: CloudModelInfo
  addedAt: number
}

export type CloudModelPatch = Partial<Pick<CloudModelDef, 'contextLimit' | 'replyLimit' | 'allowApps'>>

/** What a ModelEntry carries when it is an OpenRouter model. */
export type CloudModelTag = CloudModelDef & { provider: 'openrouter' }

export interface CloudOptions {
  /** Dollars a day (the PC's calendar day); null = no limit. */
  dailyLimitUsd: number | null
  /** Also route to providers that may keep prompts or train on them. */
  allowTraining: boolean
}

export interface CloudKeyInfo {
  label: string
  /** Dollars left on the key's own limit; null = the key has none. */
  limitRemaining: number | null
  usageToday: number
  freeTier: boolean
  /** Requests to free models left today; null = not stated. */
  freeRequestsLeft: number | null
}

export interface CloudSpend {
  /** YYYY-MM-DD, the PC's calendar day. */
  day: string
  /** Settled cost today, micro-dollars. */
  spent: number
  /** Worst-case cost of requests still running, micro-dollars. */
  reserved: number
  /** Micro-dollars today by slug. */
  models: Record<string, number>
  /** Micro-dollars of earlier days, newest first (at most 90). */
  days: { day: string; spent: number }[]
}

export interface CloudState {
  /** Goes up with every change; an older state is not applied. */
  rev: number
  keySaved: boolean
  /** Read from OpenRouter; null = not checked (or no key). */
  key: CloudKeyInfo | null
  /** Last error talking to OpenRouter about the key or the list. */
  error: string | null
  options: CloudOptions
  models: CloudModelDef[]
  spend: CloudSpend
  /** The spending record on disk cannot be read or written: requests are refused until it can. */
  spendError: string | null
}

export interface CloudCatalog {
  models: CloudModelInfo[]
  /** When the list was read from OpenRouter (ms). */
  at: number
}

/** Why a cloud request failed: `code` of the error body a cloud fetch answers with ({error: {message, type, code}}). */
export type CloudErrorKind =
  /** The daily limit, or the spending record cannot be used. */
  | 'cloud_limit'
  /** No key, or OpenRouter refused it. */
  | 'cloud_key'
  /** OpenRouter: no credits, or a budget of the key is used up. */
  | 'cloud_credits'
  /** Still rate-limited after the retries. */
  | 'cloud_rate'
  /** Another app asked for a model that is not ticked for other apps. */
  | 'cloud_not_allowed'
  /** Moderation, or no provider under the privacy and price rules. */
  | 'cloud_policy'
  /** The request cannot be sent to this model (images, tools, an unknown path). */
  | 'cloud_request'
  | 'cloud_error'

/** Errors the same request would run into again: a batch run stops on them. */
export const FATAL_CLOUD_ERRORS: readonly CloudErrorKind[] = ['cloud_limit', 'cloud_key', 'cloud_credits', 'cloud_rate', 'cloud_not_allowed']

export const isFatalCloudError = (code: unknown): boolean => typeof code === 'string' && (FATAL_CLOUD_ERRORS as readonly string[]).includes(code)

/** A cloud model's id. A file of the library can sit in a folder of that name; files end in .gguf. */
export const isCloudModelId = (id: string): boolean => id.startsWith(CLOUD_ID_PREFIX) && !id.toLowerCase().endsWith('.gguf')
export const cloudModelId = (slug: string): string => CLOUD_ID_PREFIX + slug

/** Runs on this PC (not on a rental and not at OpenRouter): uses its GPUs, can be reloaded, and unloaded for room. */
export const isLocalInstance = (i: { host?: unknown; cloud?: unknown }): boolean => !i.host && !i.cloud
export const isLocalModel = (m: { remote?: boolean; cloud?: unknown }): boolean => !m.remote && !m.cloud

export const usd = (micro: number): number => micro / 1_000_000

/** Up to four decimals, at least two: $0.0042, $0.01. */
const smallUsd = (v: number): string => `$${v.toFixed(4).replace(/0{1,2}$/, '')}`

/** $0.0042, $0.31, $12.50: enough digits to see small costs. */
export function formatUsd(micro: number): string {
  const v = usd(micro)
  if (v === 0) return '$0.00'
  if (v < 0.0001) return '<$0.0001'
  if (v < 0.1) return smallUsd(v)
  return `$${v.toFixed(2)}`
}

/** Dollars per million tokens, from micro-dollars per token (the same number). */
export const formatPerMillion = (microPerToken: number): string =>
  microPerToken === 0 ? 'Free' : `$${microPerToken >= 0.1 ? microPerToken.toFixed(2) : microPerToken.toPrecision(2)}`

/** Which sampling settings of the app an OpenRouter model honours. */
export interface SamplerSupport {
  temperature: boolean
  topP: boolean
  topK: boolean
  minP: boolean
  repeatPenalty: boolean
  presencePenalty: boolean
  frequencyPenalty: boolean
  seed: boolean
  stop: boolean
  jsonSchema: boolean
  dry: boolean
  xtc: boolean
  avoidWords: boolean
  /** Thinking can be switched at all. */
  thinking: boolean
  /** Thinking can be switched off. */
  thinkingOff: boolean
  reasoningEffort: boolean
  thinkingBudget: boolean
}

export function samplerSupport(info: Pick<CloudModelInfo, 'supported' | 'reasoning'>): SamplerSupport {
  const has = (p: string): boolean => info.supported.includes(p)
  const r = info.reasoning
  return {
    temperature: has('temperature'),
    topP: has('top_p'),
    topK: has('top_k'),
    minP: has('min_p'),
    repeatPenalty: has('repetition_penalty'),
    presencePenalty: has('presence_penalty'),
    frequencyPenalty: has('frequency_penalty'),
    seed: has('seed'),
    stop: has('stop'),
    jsonSchema: has('structured_outputs'),
    dry: false,
    xtc: false,
    avoidWords: false,
    thinking: !!r,
    thinkingOff: !!r && !r.mandatory,
    reasoningEffort: !!r && (has('reasoning_effort') || r.efforts.length > 0),
    thinkingBudget: !!r && r.supportsMaxTokens
  }
}

// ---------- IPC ----------

export interface CloudIpcInvoke {
  'cloud:state': () => CloudState
  /** Checks the key with OpenRouter, then saves it encrypted; '' forgets it (connected models are ejected). */
  'cloud:setKey': (key: string) => CloudState
  /** Reads the key's limits again. */
  'cloud:checkKey': () => CloudState
  /** OpenRouter's usable models; refresh = read again now. */
  'cloud:catalog': (refresh: boolean) => CloudCatalog
  'cloud:addModel': (slug: string) => CloudState
  'cloud:updateModel': (slug: string, patch: CloudModelPatch) => CloudState
  /** Ejects it first. Refused while it is answering. */
  'cloud:removeModel': (slug: string) => CloudState
  'cloud:setOptions': (patch: Partial<CloudOptions>) => CloudState
}

export interface CloudIpcEvents {
  'cloud:changed': CloudState
}
