// Modes and rentals (RulZ LLM Suite 0.35): which settings a task uses, which model it may pick, and
// the small decisions the screens make from them. Pure functions, no node imports: this folder is
// copied into the Android app.
import { samplingRequest, skipsThinking } from './defaults'
import type { AgentMode, ModeSettings, ModelInfo, ModelModes, ModelRental, RePracticeTaskStats, ReOverview, SamplingSettings } from './types'

export const MODE_LABEL: Record<AgentMode, string> = { coding: 'Coding', re: 'Reverse engineering' }

/** LLM Suite's system prompt for a mode is cut to this many characters (it comes from another program). */
export const MAX_MODE_PROMPT_CHARS = 20_000

export const MODE_HEADER = 'X-RulZ-Mode'
export const RENTAL_HEADER = 'X-RulZ-Rental'

/**
 * Headers every request of a task carries; `deny` makes LLM Suite refuse an answer from a rental's model.
 * A request for an OpenRouter model carries no rental header at all: it is not a rental, and the server only
 * answers it from the exact model id it names.
 */
export function modeHeaders(mode: AgentMode, rentalAllowed: boolean, cloud = false): Record<string, string> {
  return { [MODE_HEADER]: mode, ...(rentalAllowed || cloud ? {} : { [RENTAL_HEADER]: 'deny' }) }
}

// ---------- OpenRouter (LLM Suite 0.36): models in the cloud ----------

/**
 * The codes of an OpenRouter error after which the same request would fail again (our spending limit, no or a
 * refused key, no credits, rate limit, model not allowed for other apps). A task that meets one ends at once with
 * the server's sentence, and a batch run stops with that task left pending. The other `cloud_` codes
 * (cloud_policy, cloud_request, cloud_error) end the task as an ordinary error. Neither is ever retried: a retry costs money.
 */
export const CLOUD_FATAL_CODES: readonly string[] = ['cloud_limit', 'cloud_key', 'cloud_credits', 'cloud_rate', 'cloud_not_allowed']
export const isCloudCode = (code: unknown): boolean => typeof code === 'string' && code.startsWith('cloud_')
export const isCloudFatalCode = (code: unknown): boolean => typeof code === 'string' && CLOUD_FATAL_CODES.includes(code)

/** The OpenRouter entry of a /v1/models item; anything that is not `{provider: 'openrouter'}` is not one. A missing `free` counts as paid. */
export function cloudOf(entry: { cloud?: unknown }): { provider: string; free: boolean } | undefined {
  const c = entry.cloud
  if (!c || typeof c !== 'object' || Array.isArray(c)) return undefined
  const o = c as { provider?: unknown; free?: unknown }
  return o.provider === 'openrouter' ? { provider: 'openrouter', free: o.free === true } : undefined
}

/** "OpenRouter, Free" / "OpenRouter, Paid", for a model list. */
export const cloudText = (m: { cloud?: { free: boolean } }): string => `OpenRouter, ${m.cloud?.free ? 'Free' : 'Paid'}`

export const CLOUD_NOTICE = 'Your code and messages go to OpenRouter.'
export const CLOUD_NOTE =
  'This model runs on OpenRouter. Your messages, the PDFs you attach, the files it reads and what its tools return are sent to OpenRouter, and it may cost money.'
export const CLOUD_NEVER_AUTO = "It is never picked for you: only when you choose it. LLM Suite's daily spending limit applies."
export const CLOUD_CONTEXT_NOTE = "The context limit of this model is set on LLM Suite's OpenRouter page."
export const CLOUD_BATCH_NOTE =
  "A batch run sends your code, the files it reads and what its tools return to OpenRouter for every task, without you watching. It may cost money; LLM Suite's daily spending limit applies and a run stops when it is reached."

/** A cost in dollars, small amounts to four decimals ("$0.0042", "$1.37"). */
export const formatDollars = (n: number): string => `$${n < 0.01 ? n.toFixed(4) : n.toFixed(2)}`

// ---------- Rentals ----------

/** The rental a /v1/models entry says it runs on: `rental`, or (LLM Suite 0.29 to 0.34) a `path` that starts with vast/. */
export function rentalOf(entry: { rental?: unknown; path?: unknown }): ModelRental | undefined {
  const r = entry.rental
  if (r && typeof r === 'object') {
    const o = r as { label?: unknown; reconnecting?: unknown }
    return { label: typeof o.label === 'string' ? o.label.slice(0, 120) : null, reconnecting: o.reconnecting === true }
  }
  if (typeof entry.path === 'string' && /^vast\//i.test(entry.path)) return { label: null, reconnecting: false }
  return undefined
}

/** Readable model name: no folders, no ".gguf". */
export const shortModelName = (id: string): string => (id.split(/[\\/]/).pop() ?? id).replace(/\.gguf$/i, '')

export const rentalGoneMessage = (id: string): string =>
  `${shortModelName(id)} is not loaded on a rental now. Load it on LLM Suite's vast.ai page, or choose another model.`

export const rentalChooseAgainMessage = (id: string): string =>
  `${shortModelName(id)} runs on a rented computer now. Choose it again from the model list to use it there, or choose another model.`

export const cloudGoneMessage = (id: string): string =>
  `${shortModelName(id)} is not available on OpenRouter now. Choose another model, or check LLM Suite's OpenRouter page.`

export const RENTAL_NOTE =
  'This model runs on a rented computer. Your messages, the PDFs you attach, the files it reads and what its tools return are sent to that computer.'

/** The model to use when none is chosen: a loaded coder model, then any loaded one, then a coder model. Never a rental's. */
export function pickDefaultModel(all: ModelInfo[]): string {
  const models = all.filter((m) => !m.rental && !m.cloud)
  const coder = (m: ModelInfo): boolean => /qwen3[-_.]?coder/i.test(m.id)
  return (
    models.find((m) => m.loaded && coder(m)) ??
    models.find((m) => m.loaded) ??
    models.find(coder) ??
    models.find((m) => /coder|code/i.test(m.id)) ??
    models[0]
  )?.id ?? ''
}

/** `rental` is true only for a model on a rented computer; `cloud` is set only for an OpenRouter model. */
export type ModelChoice = { ok: true; model: string; rental: boolean; cloud?: true } | { ok: false; reason: string }

/**
 * The model a task uses. The chosen one when the list has it (`rental` says whether it runs on a
 * rental); when it runs on a rental but was not one when it was chosen, the owner has not agreed to that and
 * chooses it again. A chosen model that is gone from the list is replaced by the default pick only when it was
 * a local one; one that was on a rental is never swapped for another model behind the owner's back.
 * Without a list (RulZ LLM Suite not reachable) the chosen model is used and counts as local.
 */
export function chooseModel(chosen: string, chosenWasRental: boolean, models: ModelInfo[], chosenWasCloud = false): ModelChoice {
  if (chosen) {
    const info = models.find((m) => m.id === chosen)
    if (info?.cloud && !chosenWasCloud) return { ok: false, reason: 'This model runs on OpenRouter now. Choose it again to use it there.' }
    if (info?.rental && !chosenWasRental) return { ok: false, reason: rentalChooseAgainMessage(chosen) }
    if (info?.cloud) return { ok: true, model: chosen, rental: false, cloud: true }
    if (info) return { ok: true, model: chosen, rental: !!info.rental }
    // A chosen OpenRouter model that is no longer listed (key removed, "Allow other apps" off) is never swapped for another model.
    if (chosenWasCloud) return { ok: false, reason: cloudGoneMessage(chosen) }
    if (!models.length) return { ok: true, model: chosen, rental: false }
    if (chosenWasRental) return { ok: false, reason: rentalGoneMessage(chosen) }
  }
  const pick = pickDefaultModel(models)
  if (pick) return { ok: true, model: pick, rental: false }
  if (models.length) {
    if (models.some((m) => m.cloud)) return { ok: false, reason: 'The only models right now are on rentals or OpenRouter. Choose one yourself in Model settings, or load a model on this computer.' }
    return { ok: false, reason: 'The only models right now are on rentals. Choose one yourself in Model settings, or load a model on this computer.' }
  }
  return { ok: false, reason: 'RulZ LLM Suite has no models yet. Add one on its My Models page, then try again.' }
}

/** What a batch run says when it needs the owner's yes to use a rental's model. */
export const RENTAL_NEEDS_YES =
  'This model runs on a rented computer, so the run needs a yes first: its messages, the files it reads and what its tools return are sent there.'
export const RENTAL_OLD_APP =
  "This model runs on a rented computer, and this version of the app can't ask for the yes a run needs. Update the app, or start the run on the computer."
/** The same for an OpenRouter model: a run sends code to OpenRouter and may cost money, so it needs a yes (the app's `rentalOk` flag carries it). */
export const CLOUD_NEEDS_YES =
  "This model runs on OpenRouter, so the run needs a yes first: its messages, the files it reads and what its tools return are sent to OpenRouter and it may cost money. If this app can't ask for it, update the app or start the run on the computer."
export const CLOUD_OLD_APP =
  "This model runs on OpenRouter, and this version of the app can't ask for the yes a run needs. Update the app, or start the run on the computer."

// ---------- Reading LLM Suite's answer (untrusted) ----------

/** Request fields that belong to one request or one chat; a mode's sampling never sets them. */
const FORBIDDEN_FIELDS = new Set([
  'model', 'messages', 'tools', 'tool_choice', 'functions', 'function_call', 'parallel_tool_calls', 'stream', 'stream_options',
  'max_tokens', 'max_completion_tokens', 'n_predict', 'n', 'response_format', 'json_schema', 'grammar', 'seed', 'stop', 'logprobs',
  'top_logprobs', 'return_progress', 'cache_prompt', 'lora', 'adapters', 'adapter', 'user'
])
const FIELD_NAME = /^[a-z][a-z0-9_]{0,63}$/
const MAX_FIELDS = 40
const MAX_STRING = 2000

function cleanValue(v: unknown, depth: number): { ok: true; value: unknown } | { ok: false } {
  if (v === null || typeof v === 'boolean') return { ok: true, value: v }
  if (typeof v === 'number') return Number.isFinite(v) ? { ok: true, value: v } : { ok: false }
  if (typeof v === 'string') return v.length <= MAX_STRING ? { ok: true, value: v } : { ok: false }
  if (depth >= 2) return { ok: false }
  if (Array.isArray(v)) {
    if (v.length > 50) return { ok: false }
    const items: unknown[] = []
    for (const x of v) {
      const c = cleanValue(x, depth + 1)
      if (!c.ok) return { ok: false }
      items.push(c.value)
    }
    return { ok: true, value: items }
  }
  if (typeof v === 'object') {
    const out: Record<string, unknown> = {}
    const entries = Object.entries(v as Record<string, unknown>)
    if (entries.length > 20) return { ok: false }
    for (const [k, x] of entries) {
      if (!FIELD_NAME.test(k)) continue
      const c = cleanValue(x, depth + 1)
      if (c.ok) out[k] = c.value
    }
    return { ok: true, value: out }
  }
  return { ok: false }
}

/** LLM Suite's "words to avoid": at most this many [word or token id, bias or false] pairs. */
const MAX_LOGIT_BIAS = 2000
const MAX_BIAS_WORD = 200

/** logit_bias as pairs of [string up to 200 characters or integer, finite number or false]; anything else in it is dropped. null = not a list. */
function cleanLogitBias(v: unknown): [string | number, number | false][] | null {
  if (!Array.isArray(v)) return null
  const out: [string | number, number | false][] = []
  for (const pair of v.slice(0, MAX_LOGIT_BIAS)) {
    if (!Array.isArray(pair) || pair.length !== 2) continue
    const [key, bias] = pair as [unknown, unknown]
    const keyOk = (typeof key === 'string' && key.length <= MAX_BIAS_WORD) || (typeof key === 'number' && Number.isInteger(key))
    const biasOk = bias === false || (typeof bias === 'number' && Number.isFinite(bias))
    if (keyOk && biasOk) out.push([key as string | number, bias as number | false])
  }
  return out
}

/** Request fields from another program, with everything that must stay this app's (or is not plain data) left out. null = not an object. */
export function cleanRequestFields(raw: unknown): Record<string, unknown> | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (Object.keys(out).length >= MAX_FIELDS) break
    if (!FIELD_NAME.test(k) || FORBIDDEN_FIELDS.has(k)) continue
    if (k === 'logit_bias') {
      const bias = cleanLogitBias(v)
      if (bias) out[k] = bias
      continue
    }
    const c = cleanValue(v, 0)
    if (c.ok) out[k] = c.value
  }
  return out
}

function parseModeSettings(raw: unknown): ModeSettings {
  const o = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const sampling = cleanRequestFields(o.sampling)
  const prompt = typeof o.system_prompt === 'string' ? o.system_prompt : ''
  return {
    ownLoad: o.own_load === true,
    sampling,
    thinking: sampling ? cleanRequestFields(o.thinking) : null,
    systemPrompt: prompt.slice(0, MAX_MODE_PROMPT_CHARS),
    matches: typeof o.matches === 'boolean' ? o.matches : null
  }
}

/** Reads a GET /v1/models/{id}/modes body. Throws when it has no `modes`. */
export function parseModelModes(json: unknown): ModelModes {
  const j = json && typeof json === 'object' ? (json as Record<string, unknown>) : null
  const modes = j?.modes && typeof j.modes === 'object' ? (j.modes as Record<string, unknown>) : null
  if (!j || !modes) throw new Error('RulZ LLM Suite sent mode settings I could not read.')
  const state = j.state === 'loading' || j.state === 'ready' ? j.state : 'not-loaded'
  return {
    id: typeof j.id === 'string' ? j.id.slice(0, 500) : '',
    state,
    instance: typeof j.instance === 'string' && j.instance ? j.instance.slice(0, 200) : null,
    apiOverride: (() => {
      const o = cleanRequestFields(j.api_override)
      return o && Object.keys(o).length ? o : null
    })(),
    modes: { coding: parseModeSettings(modes.coding), re: parseModeSettings(modes.re) }
  }
}

// ---------- Which sampling a task uses ----------

/** Fields that decide whether the model thinks. */
const THINKING_FIELDS = ['chat_template_kwargs', 'reasoning_effort', 'thinking_budget_tokens', 'reasoning_budget_message'] as const

const without = (fields: Record<string, unknown>, drop: readonly string[]): Record<string, unknown> =>
  Object.fromEntries(Object.entries(fields).filter(([k]) => !drop.includes(k)))

/**
 * The sampling fields of a task's requests. Coding: the recommended values, then LLM Suite's Coding
 * sampling (and its thinking fields), then the values set in this app; thinking turned off here removes
 * LLM Suite's thinking fields. Reverse engineering with sampling of its own in LLM Suite: that sampling
 * as it is. Otherwise the recommended values with the owner's own, as before. The longest answer is not
 * part of this: it stays this app's setting.
 */
export function taskSamplingFields(model: string, own: SamplingSettings | undefined, mode: AgentMode, modes: ModelModes | null): Record<string, unknown> {
  const entry = modes?.modes[mode]
  if (!entry?.sampling) return samplingRequest(model, own)
  if (mode === 're') return { ...entry.sampling, ...(entry.thinking ?? {}) }
  const off = skipsThinking(own)
  const suite = off ? without({ ...entry.sampling, ...(entry.thinking ?? {}) }, THINKING_FIELDS) : { ...entry.sampling, ...(entry.thinking ?? {}) }
  return { ...samplingRequest(model, undefined), ...suite, ...samplingRequest('', own) }
}

/** True when the fields a request will carry stop the model from thinking. */
export function requestSkipsThinking(fields: Record<string, unknown>): boolean {
  const kwargs = fields.chat_template_kwargs
  const off = !!kwargs && typeof kwargs === 'object' && (kwargs as { enable_thinking?: unknown }).enable_thinking === false
  return off || fields.thinking_budget_tokens === 0
}

/** The fields the server will really use: the all-modes API server override replaces them. */
export const withApiOverride = (fields: Record<string, unknown>, modes: ModelModes | null): Record<string, unknown> => ({ ...fields, ...(modes?.apiOverride ?? {}) })

/** How many values set in this app override LLM Suite's Coding sampling. */
export function overriddenCount(own: SamplingSettings | undefined, modes: ModelModes | null): number {
  const suite = modes?.modes.coding
  if (!own || !suite?.sampling) return 0
  const keys = new Set([...Object.keys(suite.sampling), ...Object.keys(suite.thinking ?? {})])
  return Object.keys(samplingRequest('', own)).filter((k) => keys.has(k)).length
}

const plural = (n: number, one: string, many: string): string => (n === 1 ? one : many)

/**
 * The one line a task says about where its sampling and prompt come from, or null when LLM Suite's
 * mode settings and API server sampling change nothing for it (then nothing is said).
 */
export function describeModeUse(mode: AgentMode, modes: ModelModes | null, own: SamplingSettings | undefined): string | null {
  if (!modes) return null
  const entry = modes.modes[mode]
  const parts: string[] = []
  const sampling = !!entry.sampling
  const prompt = !!entry.systemPrompt.trim()
  if (sampling || prompt) {
    parts.push(`Uses LLM Suite's ${MODE_LABEL[mode]} ${sampling && prompt ? 'sampling and prompt' : sampling ? 'sampling' : 'prompt'}`)
  }
  if (mode === 'coding') {
    const n = overriddenCount(own, modes)
    if (n) parts.push(`${n} ${plural(n, 'value set here overrides', 'values set here override')} it`)
  }
  const api = Object.keys(modes.apiOverride ?? {}).map((k) => k.replaceAll('_', ' '))
  if (api.length) parts.push(`API server sampling overrides ${api.length > 4 ? `${api.slice(0, 4).join(', ')} and ${api.length - 4} more` : api.join(', ')}`)
  return parts.length ? `${parts.join('. ')}.` : null
}

/** The mode's prompt for the end of the system prompt, or null. */
export function modePrompt(mode: AgentMode, modes: ModelModes | null): { label: string; text: string } | null {
  const text = modes?.modes[mode].systemPrompt.trim().slice(0, MAX_MODE_PROMPT_CHARS)
  return text ? { label: MODE_LABEL[mode], text } : null
}

// ---------- The reload bar ----------

/** What the reload bar offers for a mode, or null when nothing is. `kept` is the instance the owner chose "Keep as is" for. */
export function reloadOffer(modes: ModelModes | null, mode: AgentMode, kept: string | null | undefined): { label: string } | null {
  if (!modes || modes.state === 'not-loaded') return null
  if (modes.modes[mode].matches !== false) return null
  if (kept && modes.instance === kept) return null
  return { label: `Reload with ${MODE_LABEL[mode]} settings` }
}

/** The part of a model's list entry that, when it changes, means its modes should be asked for again. */
export const modesStamp = (m: ModelInfo | undefined): string => (m ? `${m.loaded ? 1 : 0}|${m.loadedContext ?? ''}|${m.rental ? 'r' : ''}${m.cloud ? '|c' : ''}` : '')

// ---------- Practice per model ----------

/** The line above the practice tiles: which model the counts are for. */
export function practiceScope(practice: ReOverview['practice'], label: (id: string) => string = (s) => s): string {
  return practice.model ? `For ${label(practice.model)}` : 'All models together'
}

/** "solved 2/5" with the failed tries when LLM Suite sends them. */
export function triesText(t: Pick<RePracticeTaskStats, 'attempts' | 'solved' | 'failed'>): string {
  const base = t.solved > 0 ? `solved ${t.solved}/${t.attempts}` : `0/${t.attempts}`
  return t.failed ? `${base} · ${t.failed} failed` : base
}

/** The mode a task runs in. */
export const modeOfTask = (task: { re?: unknown }): AgentMode => (task.re ? 're' : 'coding')
