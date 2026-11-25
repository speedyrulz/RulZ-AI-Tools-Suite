// Pure helpers for the per-model, per-mode defaults on the renderer side: which chats follow the active model's
// defaults, where the picker starts, what "remember" saves, and when the reload bar shows.
import type { AllModeDefaults, AppMode, ModeDefaults, ModelModeDefaults } from '@shared/modes'
import { APP_MODES, chatPrompt, matchesModeLoad, modeDefaultsEmpty, modeLoad, modeSampling, reloadFor } from '@shared/modes'
import type { Conversation, LoadConfig, ModelSampling, SamplingParams } from '@shared/types'

/** A model's settings in every mode; an own-property lookup, so a model id like "constructor" finds nothing. */
export function modesFor(all: AllModeDefaults | undefined, modelId: string | null | undefined): ModelModeDefaults | undefined {
  if (!all || !modelId || !Object.prototype.hasOwnProperty.call(all, modelId)) return undefined
  return all[modelId]
}

const stable = (v: unknown): string =>
  JSON.stringify(v, (_k, x: unknown) =>
    x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) : x
  )

export const deepEqual = (a: unknown, b: unknown): boolean => stable(a) === stable(b)

/** What a new chat starts with for a model. */
export interface ChatDefaults {
  sampling: SamplingParams
  systemPrompt: string
}

export function chatDefaultsFor(modes: ModelModeDefaults | undefined, app: { defaultSampling: SamplingParams; defaultSystemPrompt: string }): ChatDefaults {
  return { sampling: modeSampling('chat', modes, app.defaultSampling), systemPrompt: chatPrompt(modes, app.defaultSystemPrompt) }
}

/** An empty chat nobody has touched or started: the only kind that follows the active model's defaults. */
export function isFollowable(c: Pick<Conversation, 'messages' | 're' | 'presetId'>): boolean {
  return c.messages.length === 0 && !c.re && !c.presetId
}

/**
 * Moves an empty chat to `next` defaults, sampling and prompt separately, where they still equal `base` (the defaults the
 * chat was made with). `followed` says which parts were still untouched (and so now follow `next`); `conv` is the same
 * object when nothing needs to change.
 */
export function followChatDefaults(
  conv: Conversation,
  base: { sampling: SamplingParams; systemPrompt: string },
  next: ChatDefaults
): { conv: Conversation; followed: { sampling: boolean; prompt: boolean } } {
  const none = { sampling: false, prompt: false }
  if (!isFollowable(conv)) return { conv, followed: none }
  const sampling = deepEqual(conv.sampling, base.sampling)
  const prompt = conv.systemPrompt === base.systemPrompt
  const moveSampling = sampling && !deepEqual(conv.sampling, next.sampling)
  const movePrompt = prompt && conv.systemPrompt !== next.systemPrompt
  const followed = { sampling, prompt }
  if (!moveSampling && !movePrompt) return { conv, followed }
  return {
    conv: { ...conv, ...(moveSampling ? { sampling: structuredClone(next.sampling) } : {}), ...(movePrompt ? { systemPrompt: next.systemPrompt } : {}) },
    followed
  }
}

/** The scope of the model picker's "Load for" Select: a mode, or null for all modes (the model's default settings). */
export type PickerScope = AppMode | null

/** Where the user is decides the picker's first choice. `chatIsRe`: the open chat is a reverse-engineering session. */
export function pickerScope(page: string, chatIsRe: boolean): PickerScope {
  switch (page) {
    case 'chat':
      return chatIsRe ? 're' : 'chat'
    case 're':
      return 're'
    case 'adventure':
      return 'adventure'
    case 'writing':
      return 'writing'
    default:
      return null
  }
}

/**
 * The scope the picker starts with for a model: the page's mode when the model has load settings of its own for it, else
 * all modes, so "Remember" keeps meaning the model's default settings for everyone who sets nothing per mode.
 */
export function startScope(mode: PickerScope, modes: ModelModeDefaults | undefined): PickerScope {
  return mode && modes?.[mode]?.load ? mode : null
}

/**
 * The load settings the picker first shows. A loaded model shows what it runs with, except when that would let "Remember"
 * write one scope's settings over another's: a mode shows its own, and all modes never shows a mode's.
 */
export function startLoad(base: LoadConfig, modes: ModelModeDefaults | undefined, scope: PickerScope, running: LoadConfig | undefined): LoadConfig {
  if (scope || !running) return scopeLoad(base, modes, scope)
  return APP_MODES.some((m) => matchesModeLoad(running, base, modes, m)) ? base : running
}

/** The Select value of a scope and back. */
export const scopeValue = (s: PickerScope): string => s ?? 'all'
export const scopeFromValue = (v: string): PickerScope => (APP_MODES as string[]).includes(v) ? (v as AppMode) : null

/** The load settings the picker shows for a scope: that mode's own, else the model's default ones. */
export function scopeLoad(base: LoadConfig, modes: ModelModeDefaults | undefined, scope: PickerScope): LoadConfig {
  return modeLoad(base, modes, scope ?? undefined)
}

/**
 * A mode's load settings as they are stored: the API-server sampling in them is not the mode's to set, so an entry that
 * has one keeps what it has and a new one gets the model's default load settings' (`baseSampling`).
 */
export function storedModeLoad(existing: ModeDefaults | undefined, load: LoadConfig | null, baseSampling: ModelSampling | undefined): LoadConfig | null {
  if (!load) return null
  const sampling = existing?.load?.sampling ?? baseSampling ?? load.sampling
  return { ...load, sampling }
}

/** What "Remember settings" saves for a mode: its load settings replaced, its sampling and prompt kept. null = the entry is gone. */
export function withModeLoad(existing: ModeDefaults | undefined, load: LoadConfig | null, baseSampling?: ModelSampling): ModeDefaults | null {
  const next: ModeDefaults = { load: storedModeLoad(existing, load, baseSampling), sampling: existing?.sampling ?? null, systemPrompt: existing?.systemPrompt ?? '' }
  return modeDefaultsEmpty(next) ? null : next
}

/**
 * What the picker's load button saves for a mode, in one entry: `load` (the settings shown) only when they are
 * remembered, `sampling` only when it was edited (null = off). Everything not given stays as `existing` (read just
 * before saving); the prompt always does. Saving own sampling ON hands the stored load's API sampling back to the
 * model's default one, so leftover values are not offered again. null = the entry is gone.
 */
export function withModeSaves(
  existing: ModeDefaults | undefined,
  baseSampling: ModelSampling | undefined,
  change: { load?: LoadConfig; sampling?: SamplingParams | null }
): ModeDefaults | null {
  const sampling = change.sampling === undefined ? (existing?.sampling ?? null) : change.sampling ? { ...change.sampling, jsonSchema: '' } : null
  let load = change.load ? storedModeLoad(existing, change.load, baseSampling) : (existing?.load ?? null)
  if (load && change.sampling && baseSampling) load = { ...load, sampling: baseSampling }
  const next: ModeDefaults = { load, sampling, systemPrompt: existing?.systemPrompt ?? '' }
  return modeDefaultsEmpty(next) ? null : next
}

/** The reload bar's offer: the settings to reload with, or null. A dismissed instance + mode gets none. */
export function reloadOffer(
  running: LoadConfig | null | undefined,
  base: LoadConfig | null | undefined,
  modes: ModelModeDefaults | undefined,
  mode: AppMode,
  dismissed: boolean
): LoadConfig | null {
  if (!running || !base || dismissed) return null
  return reloadFor(running, base, modes, mode)
}
