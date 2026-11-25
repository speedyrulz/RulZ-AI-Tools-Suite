// The phone's writing:mode and writing:reloadMode (contract: src/shared/phone.ts, docs/app-modes-design.md): what applies
// to a mode with a model, and a reload with the mode's load settings. The checks are the API's reload and the app's own
// reloadBlock, plus the lease a request with X-RulZ-Mode leaves, which only a phone looks at.
import { MODE_LABELS, reloadFor, type ModelModeDefaults } from '@shared/modes'
import { PHONE_MODES, type PhoneMode, type PhoneModeInfo } from '@shared/phone'
import type { LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import type { ModeReloadResult } from '../server/apiServer'

/** The longest model id a phone may send. */
const MAX_MODEL_ID = 1024

export const LEASED = 'RulZ Coding Agent is using the model.'
export const CLOUD_RELOAD = 'An OpenRouter model has no load settings, so there is nothing to reload.'
export const RENTAL_RELOAD = "A rented model can only be reloaded in LLM Suite on the computer."

export interface PhoneModesDeps {
  /** The newest loaded model's id: what a job with no model of its own would use. */
  newestReady: () => string | null
  /** The ready instance of a model (the one that answers). */
  readyFor: (modelId: string) => LoadedInstance | undefined
  modeDefaults: (modelId: string) => ModelModeDefaults
  /** The model's default load settings. */
  loadDefaults: (modelId: string) => LoadConfig
  /** Why an instance cannot be reloaded now (main's reloadBlock). */
  reloadBlock: (instanceId: string) => string | null
  /** Whether a request with X-RulZ-Mode leased the model lately. */
  leased: (modelId: string) => boolean
  /** Starts the reload; takes the model gate until the load has ended (ApiServer.requestModeReload). */
  reload: (model: ModelEntry, mode: PhoneMode, also: () => string | null) => ModeReloadResult
  /** A model of the library (or a rental's, or an OpenRouter one) by id. */
  getEntry: (modelId: string) => ModelEntry | undefined
  /** Whether other apps (so a phone) may use an OpenRouter model; without it none may. */
  cloudAllowed?: (modelId: string) => boolean
}

/** The phone's mode and model arguments, checked: a phone is not trusted to send the right shapes. */
export function parsePhoneMode(v: unknown): PhoneMode {
  if (typeof v !== 'string' || !(PHONE_MODES as readonly string[]).includes(v)) throw new Error('The mode must be "writing" or "adventure".')
  return v as PhoneMode
}

export function parsePhoneModelId(v: unknown, allowNull: boolean): string | null {
  if (v === null && allowNull) return null
  if (typeof v !== 'string' || !v || v.length > MAX_MODEL_ID) throw new Error('The model is missing or not valid.')
  return v
}

export class PhoneModes {
  constructor(private readonly deps: PhoneModesDeps) {}

  info(mode: PhoneMode, modelId: string | null): PhoneModeInfo {
    const d = this.deps
    const id = modelId ?? d.newestReady()
    if (!id) return { mode, modelId: null, name: '', loaded: false, ownLoad: false, ownSampling: false, hasPrompt: false, reload: null }
    // A phone sees an OpenRouter model only when it is ticked for other apps; otherwise it is as good as unknown.
    const found = d.getEntry(id)
    const entry = found?.cloud && !d.cloudAllowed?.(id) ? undefined : found
    const inst = entry?.cloud || !found?.cloud ? d.readyFor(id) : undefined
    const modes = d.modeDefaults(id)
    const own = modes[mode]
    const out: PhoneModeInfo = {
      mode,
      modelId: id,
      name: entry?.name ?? id,
      loaded: !!inst,
      ownLoad: !!own?.load,
      ownSampling: !!own?.sampling,
      hasPrompt: !!own?.systemPrompt.trim(),
      reload: null
    }
    if (entry?.remote) out.rental = { label: inst?.host?.label ?? null, reconnecting: !!inst?.reconnecting }
    // An OpenRouter model has no load settings: never a reload.
    if (entry?.cloud) out.cloud = { free: entry.cloud.info.free }
    else if (inst && reloadFor(inst.config, d.loadDefaults(id), modes, mode) !== null) {
      const blocked = inst.host ? RENTAL_RELOAD : (d.reloadBlock(inst.instanceId) ?? (d.leased(id) ? LEASED : null))
      out.reload = { label: own?.load ? `${MODE_LABELS[mode]} settings` : 'default settings', blocked }
    }
    return out
  }

  /** Reloads with the mode's load settings; throws the sentence when it cannot be done now. */
  reloadMode(mode: PhoneMode, modelId: string): PhoneModeInfo {
    const d = this.deps
    const entry = d.getEntry(modelId)
    if (!entry || (entry.cloud && !d.cloudAllowed?.(modelId))) throw new Error('That model is not in the library.')
    if (entry.remote) throw new Error(RENTAL_RELOAD)
    if (entry.cloud) throw new Error(CLOUD_RELOAD)
    const r = d.reload(entry, mode, () => (d.leased(modelId) ? LEASED : null))
    if (r.status === 'blocked') throw new Error(r.why)
    const info = this.info(mode, modelId)
    // Just started: the old instance may still be listed; the phone is told the model is loading.
    return r.status === 'started' ? { ...info, loaded: false, reload: null } : info
  }
}
