// Pure helpers for the local API server (unit-tested separately from the HTTP code).
import { sep } from 'node:path'
import { z } from 'zod'
import { draftFiles, fileName } from '@shared/draft'
import { KV_CACHE_TYPES, type DraftKind, type LoadConfig, type ModelEntry, type ModelSampling } from '@shared/types'

/** The id a model is listed under in GET /v1/models: its name, or its path when names collide. */
export function publicModelId(model: ModelEntry, all: ModelEntry[]): string {
  const dupes = all.filter((m) => m.name.toLowerCase() === model.name.toLowerCase())
  return dupes.length > 1 ? model.id : model.name
}

/**
 * Finds the model a request asked for. Accepts the public id, the library path, the file name
 * (with or without .gguf), any of those case-insensitively, and finally a unique substring match.
 */
export function resolveModel(models: ModelEntry[], requested: string | undefined | null): ModelEntry | undefined {
  const q = (requested ?? '').trim()
  if (!q) return undefined
  const lower = q.toLowerCase()
  const strip = (s: string): string => s.toLowerCase().replace(/\.gguf$/, '')
  const unique = (hits: ModelEntry[]): ModelEntry | undefined => (hits.length === 1 ? hits[0] : undefined)
  // The library path is unique; a name is only accepted when exactly one model carries it.
  return (
    models.find((m) => m.id === q) ??
    unique(models.filter((m) => m.id.toLowerCase() === lower)) ??
    unique(models.filter((m) => strip(m.name) === strip(lower))) ??
    unique(models.filter((m) => strip(m.path.split(/[\\/]/).pop() ?? '') === strip(lower))) ??
    unique(models.filter((m) => m.name.toLowerCase().includes(lower) || m.id.toLowerCase().includes(lower)))
  )
}

/**
 * The model a request is answered with. A named model is looked up in the library; with no name,
 * or (when `fallback` is on) a name the library doesn't have, the single loaded model is used,
 * then the default model. `fallback` is true when a named model was replaced.
 */
export function pickModel(
  models: ModelEntry[],
  requested: string | undefined,
  readyModelIds: string[],
  settings: { defaultModelId: string; unknownModelFallback: boolean }
): { model: ModelEntry; fallback: boolean } | undefined {
  if (requested) {
    const named = resolveModel(models, requested)
    if (named) return { model: named, fallback: false }
    if (!settings.unknownModelFallback) return undefined
  }
  const fallback = !!requested
  const ready = readyModelIds.length === 1 ? models.find((m) => m.id === readyModelIds[0]) : undefined
  if (ready) return { model: ready, fallback }
  const def = settings.defaultModelId ? models.find((m) => m.id === settings.defaultModelId) : undefined
  return def ? { model: def, fallback } : undefined
}

export interface UsageInfo {
  promptTokens?: number
  completionTokens?: number
  tokensPerSecond?: number
}

/** Reads token usage from a completion response body, or from the tail of an SSE stream. */
export function parseUsage(text: string, stream: boolean): UsageInfo {
  const out: UsageInfo = {}
  const apply = (j: unknown): void => {
    const o = j as {
      usage?: { prompt_tokens?: number; completion_tokens?: number; input_tokens?: number; output_tokens?: number }
      timings?: { prompt_n?: number; predicted_n?: number; predicted_per_second?: number; cache_n?: number }
    }
    if (o?.usage) {
      out.promptTokens = o.usage.prompt_tokens ?? o.usage.input_tokens ?? out.promptTokens
      out.completionTokens = o.usage.completion_tokens ?? o.usage.output_tokens ?? out.completionTokens
    }
    if (o?.timings) {
      if (o.timings.prompt_n !== undefined) out.promptTokens = o.timings.prompt_n + (o.timings.cache_n ?? 0)
      if (o.timings.predicted_n !== undefined) out.completionTokens = o.timings.predicted_n
      if (o.timings.predicted_per_second !== undefined) out.tokensPerSecond = o.timings.predicted_per_second
    }
  }
  if (!stream) {
    try {
      apply(JSON.parse(text))
    } catch {
      // not JSON
    }
    return out
  }
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith('data:')) continue
    const payload = line.slice(5).trim()
    if (!payload || payload === '[DONE]') continue
    try {
      apply(JSON.parse(payload))
    } catch {
      // partial line at the start of the tail buffer
    }
  }
  return out
}

/** True when the request's credentials match the configured key (any key passes when none is set). */
export function authorized(apiKey: string, headers: { authorization?: string; 'x-api-key'?: string }): boolean {
  if (!apiKey) return true
  const bearer = headers.authorization?.match(/^Bearer\s+(.+)$/i)?.[1]?.trim()
  return bearer === apiKey || headers['x-api-key']?.trim() === apiKey
}

/** OpenAI-style error body. */
export function errorBody(message: string, type = 'invalid_request_error'): string {
  return JSON.stringify({ error: { message, type } })
}

/**
 * The load settings other apps may change over the API. File paths and free-form arguments are
 * left out on purpose: the server can listen on the LAN, and those would let a client point
 * llama-server at anything on disk.
 */
const remoteLoadPatchSchema = z
  .object({
    contextLength: z.number().int().min(512).max(4_194_304).nullable(),
    gpuLayers: z.union([z.literal('auto'), z.literal('all'), z.number().int().min(0).max(1000)]),
    flashAttention: z.enum(['auto', 'on', 'off']),
    cacheTypeK: z.enum(KV_CACHE_TYPES as [string, ...string[]]),
    cacheTypeV: z.enum(KV_CACHE_TYPES as [string, ...string[]]),
    parallel: z.number().int().min(1).max(64),
    splitMode: z.enum(['auto', 'single', 'custom']),
    mainGpu: z.number().int().min(0).max(15),
    // Proportions per GPU, e.g. "3,1" — numbers only, so nothing else reaches the command line.
    tensorSplit: z.string().regex(/^\s*(\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?){0,15})?\s*$/, 'must be numbers separated by commas'),
    useDraft: z.boolean(),
    draftTokens: z.number().int().min(1).max(64).nullable(),
    draftMinTokens: z.number().int().min(0).max(64).nullable(),
    draftPMin: z.number().min(0).max(1).nullable(),
    draftSampling: z.enum(['auto', 'greedy', 'probabilistic']),
    draftGpu: z.enum(['auto', 'gpu', 'cpu']),
    draftCacheTypeK: z.enum(KV_CACHE_TYPES as [string, ...string[]]),
    draftCacheTypeV: z.enum(KV_CACHE_TYPES as [string, ...string[]]),
    // null, 'builtin' or the id of one of draft_choices; checked against the library before it is used.
    draftFile: z.string().min(1).max(1024).nullable(),
    // Any subset of the model's sampling defaults; null clears one.
    sampling: z
      .object({
        temperature: z.number().min(0).max(5).nullable(),
        topK: z.number().int().min(0).max(1000).nullable(),
        topP: z.number().min(0).max(1).nullable(),
        minP: z.number().min(0).max(1).nullable(),
        repeatPenalty: z.number().min(0).max(5).nullable(),
        presencePenalty: z.number().min(-2).max(2).nullable(),
        override: z.boolean()
      })
      .partial()
      .strict()
  })
  .partial()
  .strict()

const REMOTE_LOAD_KEYS = [
  'contextLength',
  'gpuLayers',
  'flashAttention',
  'cacheTypeK',
  'cacheTypeV',
  'parallel',
  'splitMode',
  'mainGpu',
  'tensorSplit',
  'useDraft',
  'draftTokens',
  'draftMinTokens',
  'draftPMin',
  'draftSampling',
  'draftGpu',
  'draftCacheTypeK',
  'draftCacheTypeV',
  'draftFile',
  'sampling'
] as const satisfies readonly (keyof LoadConfig)[]

export interface DraftChoice {
  /** 'builtin', or the draft file's path relative to the models folder (never an absolute path). */
  id: string
  name: string
  folder: string
  kind: DraftKind
  built_in: boolean
}

/** The drafts a client may pick for `model`: its built-in MTP layers, then every draft file in the library. */
export function draftChoices(model: ModelEntry, models: ModelEntry[]): DraftChoice[] {
  const root = model.path.slice(0, model.path.length - model.id.length)
  const out: DraftChoice[] = []
  if (model.info?.mtpLayers) out.push({ id: 'builtin', name: 'Built-in MTP layers', folder: '', kind: 'mtp', built_in: true })
  for (const d of draftFiles(models)) {
    if (!d.path.startsWith(root)) continue
    const id = d.path.slice(root.length).split(sep).join('/')
    out.push({ id, name: fileName(d.path), folder: id.split('/').slice(0, -1).join('/'), kind: d.kind ?? 'mtp', built_in: false })
  }
  return out
}

/** A saved draftFile as clients see it: null, 'builtin' or a choice id. */
export function draftFileToRemote(draftFile: string | null, model: ModelEntry): string | null {
  if (draftFile === null || draftFile === 'builtin') return draftFile
  const root = model.path.slice(0, model.path.length - model.id.length)
  return draftFile.startsWith(root) ? draftFile.slice(root.length).split(sep).join('/') : fileName(draftFile)
}

/** A draftFile sent by a client, as saved: undefined when it isn't one of the model's choices. */
export function draftFileFromRemote(value: string | null, model: ModelEntry, models: ModelEntry[]): string | null | undefined {
  if (value === null) return null
  const choice = draftChoices(model, models).find((c) => c.id === value)
  if (!choice) return undefined
  return choice.built_in ? 'builtin' : model.path.slice(0, model.path.length - model.id.length) + choice.id.split('/').join(sep)
}

type RemoteLoadKey = (typeof REMOTE_LOAD_KEYS)[number]
export type RemoteLoadView = Pick<LoadConfig, RemoteLoadKey>
export type RemoteLoadPatch = Partial<Omit<RemoteLoadView, 'sampling'>> & { sampling?: Partial<ModelSampling> }

/** A load config with a client's change applied; sampling is merged value by value. */
export function applyRemoteLoadPatch(config: LoadConfig, patch: RemoteLoadPatch): LoadConfig {
  return { ...config, ...patch, sampling: { ...config.sampling, ...patch.sampling } }
}

/** Checks a load-settings change sent to the API; unlike saved settings, bad values are refused, not replaced. */
export function parseRemoteLoadPatch(raw: unknown): { ok: true; patch: RemoteLoadPatch } | { ok: false; error: string } {
  const r = remoteLoadPatchSchema.safeParse(raw)
  if (r.success) return { ok: true, patch: r.data as RemoteLoadPatch }
  const issue = r.error.issues[0]
  const where = issue.path.join('.') || 'config'
  return { ok: false, error: `${where}: ${issue.message}` }
}

/** The part of a load config the API shows and accepts. */
export function remoteLoadView(config: LoadConfig, model: ModelEntry): RemoteLoadView {
  const view = Object.fromEntries(REMOTE_LOAD_KEYS.map((k) => [k, config[k]])) as RemoteLoadView
  return { ...view, draftFile: draftFileToRemote(config.draftFile, model) }
}

/** Generation endpoints whose request bodies take sampling values. */
export const SAMPLING_PATHS = new Set([
  '/v1/chat/completions',
  '/v1/completions',
  '/v1/messages',
  '/v1/responses',
  '/completion',
  '/completions'
])

/**
 * A request body with the model's set sampling values written over the client's, for models whose
 * sampling defaults have `override` on. Values left empty keep whatever the client sent.
 */
export function overrideSampling(body: Record<string, unknown>, s: ModelSampling): Record<string, unknown> {
  const out = { ...body }
  if (s.temperature !== null) out.temperature = s.temperature
  if (s.topK !== null) out.top_k = s.topK
  if (s.topP !== null) out.top_p = s.topP
  if (s.minP !== null) out.min_p = s.minP
  if (s.repeatPenalty !== null) out.repeat_penalty = s.repeatPenalty
  if (s.presencePenalty !== null) out.presence_penalty = s.presencePenalty
  return out
}
