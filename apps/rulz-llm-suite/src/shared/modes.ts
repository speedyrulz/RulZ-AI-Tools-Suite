// Per-model defaults for each mode of the app (Chat, Reverse engineering, Adventure, Writing, Coding): load settings, sampling
// and system prompt. Anything a mode does not set falls back to the model's default load settings and to the sampling
// and prompt that mode always had.
import { DEFAULT_SAMPLING } from './defaults'
import { FICTION_SAMPLING } from './novelPrompts'
import { reSampling } from './re'
import type { LoadConfig, ModelSampling, SamplingParams } from './types'

export type AppMode = 'chat' | 're' | 'adventure' | 'writing' | 'coding'

export const APP_MODES: AppMode[] = ['chat', 're', 'adventure', 'writing', 'coding']

export const MODE_LABELS: Record<AppMode, string> = {
  chat: 'Chat',
  re: 'Reverse engineering',
  adventure: 'Adventure',
  writing: 'Writing',
  // No page of its own: RulZ Coding Agent uses it for its normal tasks.
  coding: 'Coding'
}

/** What one model uses in one mode. null (or an empty prompt) = the fallback. */
export interface ModeDefaults {
  /** null = the model's default load settings. */
  load: LoadConfig | null
  /** null = the mode's own sampling (see modeSampling). */
  sampling: SamplingParams | null
  /** Chat: the system prompt of new chats instead of the app's default. Other modes: added after the built-in prompt. */
  systemPrompt: string
}

/** One model's modes; a mode that is absent has no settings of its own. */
export type ModelModeDefaults = Partial<Record<AppMode, ModeDefaults>>

/** Every model with mode settings, by ModelEntry id. */
export type AllModeDefaults = Record<string, ModelModeDefaults>

/** Whether a mode's entry says anything; one that doesn't is not stored. */
export function modeDefaultsEmpty(d: ModeDefaults | null | undefined): boolean {
  return !d || (!d.load && !d.sampling && !d.systemPrompt.trim())
}

/**
 * The load settings a model gets in a mode: its own for that mode, else the model's default ones. The sampling that
 * only the API server uses always comes from the default ones, so a mode's copy of it is never used.
 */
export function modeLoad(base: LoadConfig, modes: ModelModeDefaults | undefined, mode: AppMode | undefined): LoadConfig {
  const own = mode && modes?.[mode]?.load
  return own ? { ...own, sampling: base.sampling } : base
}

/** The sampling a mode starts from when the model sets none for it. */
export function builtInSampling(mode: AppMode, appDefault: SamplingParams): SamplingParams {
  switch (mode) {
    case 'chat':
    case 'coding':
      return appDefault
    case 're':
      return reSampling(appDefault)
    case 'adventure':
      return DEFAULT_SAMPLING
    case 'writing':
      return FICTION_SAMPLING
  }
}

/** The sampling a model gets in a mode. Always a copy, so changing one chat's lists never touches the saved defaults. */
export function modeSampling(mode: AppMode, modes: ModelModeDefaults | undefined, appDefault: SamplingParams): SamplingParams {
  const own = modes?.[mode]?.sampling
  // A JSON schema belongs to one chat, never to a model's defaults.
  return structuredClone(own ? { ...own, jsonSchema: '' } : builtInSampling(mode, appDefault))
}

/** The system prompt of a new chat with a model. */
export function chatPrompt(modes: ModelModeDefaults | undefined, appDefault: string): string {
  const own = modes?.chat?.systemPrompt
  return own?.trim() ? own : appDefault
}

/** The text a model adds after a mode's built-in prompt; '' = none. Not for Chat, see chatPrompt. */
export function modePromptExtra(modes: ModelModeDefaults | undefined, mode: Exclude<AppMode, 'chat'>): string {
  return modes?.[mode]?.systemPrompt.trim() ?? ''
}

/** A built-in prompt with the model's own text after it. */
export function withPromptExtra(builtIn: string, extra: string | undefined): string {
  return extra?.trim() ? `${builtIn}\n\n${extra.trim()}` : builtIn
}

const stable = (v: unknown): string =>
  JSON.stringify(v, (_k, x: unknown) =>
    x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) : x
  )

/** Same load settings, whatever order their fields were saved in. */
export function sameLoad(a: LoadConfig, b: LoadConfig): boolean {
  return stable(a) === stable(b)
}

/**
 * Whether a running load is what a mode's own load settings give: the effective settings, or the stored ones as they
 * are (a model loaded before the API sampling stopped being a mode's own, with leftover values in it).
 */
export function matchesModeLoad(running: LoadConfig, base: LoadConfig, modes: ModelModeDefaults | undefined, mode: AppMode): boolean {
  const own = modes?.[mode]?.load
  return !!own && (sameLoad(running, modeLoad(base, modes, mode)) || sameLoad(running, own))
}

/**
 * The settings to offer a reload with when a model that runs with `running` is used in `mode`; null = no offer.
 * Only for models with load settings for some mode. A model loaded by hand with settings of its own is left alone in
 * a mode that has none, unless it runs with another mode's settings.
 */
export function reloadFor(running: LoadConfig, base: LoadConfig, modes: ModelModeDefaults | undefined, mode: AppMode): LoadConfig | null {
  const wanted = modeLoad(base, modes, mode)
  if (sameLoad(running, wanted) || matchesModeLoad(running, base, modes, mode)) return null
  if (modes?.[mode]?.load) return wanted
  return APP_MODES.some((m) => matchesModeLoad(running, base, modes, m)) ? wanted : null
}

/** Sampling keys a mode's load (ModelSampling) and a mode's own sampling (SamplingParams) both have. */
export const SHARED_SAMPLING_KEYS = [
  'temperature',
  'topK',
  'topP',
  'minP',
  'repeatPenalty',
  'presencePenalty',
  'dryMultiplier',
  'dryBase',
  'dryAllowedLength',
  'dryPenaltyLastN',
  'drySequenceBreakers',
  'xtcProbability',
  'xtcThreshold'
] as const satisfies readonly (keyof ModelSampling & keyof SamplingParams)[]

export type SharedSamplingKey = (typeof SHARED_SAMPLING_KEYS)[number]

/** The values of an API-server sampling that are set. */
export type LeftoverSampling = Partial<Pick<SamplingParams, SharedSamplingKey>>

/**
 * Sampling values stored in a mode's own load settings that only the API server used: every value that is set, when at
 * least one differs from the model's default load settings. null = nothing to adopt.
 */
export function leftoverSampling(own: LoadConfig | null | undefined, base: LoadConfig): LeftoverSampling | null {
  const s = own?.sampling
  if (!s) return null
  const out: Record<string, unknown> = {}
  let differs = false
  for (const k of SHARED_SAMPLING_KEYS) {
    const v = s[k]
    if (v === null || v === undefined) continue
    out[k] = Array.isArray(v) ? [...v] : v
    if (stable(v) !== stable(base.sampling[k])) differs = true
  }
  return differs ? (out as LeftoverSampling) : null
}

/** A mode's own sampling that starts from its built-in one with these values on top. */
export function adoptSampling(mode: AppMode, appDefault: SamplingParams, leftover: LeftoverSampling): SamplingParams {
  return { ...structuredClone(builtInSampling(mode, appDefault)), ...structuredClone(leftover), jsonSchema: '' }
}
