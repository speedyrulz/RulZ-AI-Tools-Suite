import { z } from 'zod'
import { clampScale } from '@shared/adapters'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING, DEFAULT_SETTINGS } from '@shared/defaults'
import { APP_MODES, modeDefaultsEmpty, type AllModeDefaults, type ModeDefaults, type ModelModeDefaults } from '@shared/modes'
import { KV_CACHE_TYPES, REASONING_EFFORTS, type AppSettings, type LoadAdapter, type LoadConfig, type Preset, type SamplingParams } from '@shared/types'

// Every field falls back to its default on its own, so one bad value never resets the rest.
const L = DEFAULT_LOAD_CONFIG
const kvType = z.enum(KV_CACHE_TYPES as [string, ...string[]])
const posIntOrNull = z.number().int().positive().nullable()

/** Most adapters one model can load. */
export const MAX_ADAPTERS = 16

/** One adapter entry: bad entries are dropped, the strength is clamped, later duplicates of an id are dropped. */
function cleanAdapters(raw: unknown): LoadAdapter[] {
  if (!Array.isArray(raw)) return []
  const out: LoadAdapter[] = []
  for (const e of raw) {
    const id = (e as { id?: unknown } | null)?.id
    if (typeof id !== 'string' || !id || out.some((o) => o.id === id)) continue
    out.push({ id, scale: clampScale((e as { scale?: unknown }).scale) })
    if (out.length >= MAX_ADAPTERS) break
  }
  return out
}

/** A chat's adapter strengths: null (use the model's defaults) or a cleaned list. Never throws. */
export function parseChatAdapters(raw: unknown): LoadAdapter[] | null {
  return Array.isArray(raw) ? cleanAdapters(raw) : null
}

export const loadConfigSchema = z.object({
  contextLength: posIntOrNull.catch(L.contextLength),
  gpuLayers: z.union([z.literal('auto'), z.literal('all'), z.number().int().min(0)]).catch(L.gpuLayers),
  cpuMoe: z.union([z.literal('off'), z.literal('all'), z.number().int().min(1)]).catch(L.cpuMoe),
  overrideTensors: z.string().catch(L.overrideTensors),
  splitMode: z.enum(['auto', 'single', 'custom']).catch(L.splitMode),
  mainGpu: z.number().int().min(0).catch(L.mainGpu),
  tensorSplit: z.string().catch(L.tensorSplit),
  flashAttention: z.enum(['auto', 'on', 'off']).catch(L.flashAttention),
  cacheTypeK: kvType.catch(L.cacheTypeK),
  cacheTypeV: kvType.catch(L.cacheTypeV),
  threads: posIntOrNull.catch(L.threads),
  batchSize: posIntOrNull.catch(L.batchSize),
  ubatchSize: posIntOrNull.catch(L.ubatchSize),
  useMmap: z.boolean().catch(L.useMmap),
  useMlock: z.boolean().catch(L.useMlock),
  useMmproj: z.boolean().catch(L.useMmproj),
  mmprojOnCpu: z.boolean().catch(L.mmprojOnCpu),
  useDraft: z.boolean().catch(L.useDraft),
  draftFile: z.string().min(1).nullable().catch(L.draftFile),
  draftTokens: z.number().int().min(1).max(64).nullable().catch(L.draftTokens),
  draftMinTokens: z.number().int().min(0).max(64).nullable().catch(L.draftMinTokens),
  draftPMin: z.number().min(0).max(1).nullable().catch(L.draftPMin),
  draftSampling: z.enum(['auto', 'greedy', 'probabilistic']).catch(L.draftSampling),
  draftGpu: z.enum(['auto', 'gpu', 'cpu']).catch(L.draftGpu),
  draftCacheTypeK: kvType.catch(L.draftCacheTypeK),
  draftCacheTypeV: kvType.catch(L.draftCacheTypeV),
  seed: z.number().int().nullable().catch(L.seed),
  chatTemplateFile: z.string().catch(L.chatTemplateFile),
  parallel: z.number().int().min(1).max(64).catch(L.parallel),
  extraArgs: z.string().catch(L.extraArgs),
  adapters: z.unknown().transform(cleanAdapters).catch([]),
  sampling: z
    .object({
      temperature: z.number().min(0).max(5).nullable().catch(null),
      topK: z.number().int().min(0).nullable().catch(null),
      topP: z.number().min(0).max(1).nullable().catch(null),
      minP: z.number().min(0).max(1).nullable().catch(null),
      repeatPenalty: z.number().min(0).max(5).nullable().catch(null),
      presencePenalty: z.number().min(-2).max(2).nullable().catch(null),
      dryMultiplier: z.number().min(0).max(5).nullable().catch(null),
      dryBase: z.number().min(1).max(10).nullable().catch(null),
      dryAllowedLength: z.number().int().min(1).max(1000).nullable().catch(null),
      dryPenaltyLastN: z.number().int().min(-1).nullable().catch(null),
      drySequenceBreakers: z.array(z.string()).max(64).nullable().catch(null),
      xtcProbability: z.number().min(0).max(1).nullable().catch(null),
      xtcThreshold: z.number().min(0).max(1).nullable().catch(null),
      override: z.boolean().catch(false)
    })
    .catch(L.sampling)
})

const S = DEFAULT_SAMPLING
export const samplingSchema = z.object({
  temperature: z.number().min(0).max(5).catch(S.temperature),
  topK: z.number().int().min(0).catch(S.topK),
  topP: z.number().min(0).max(1).catch(S.topP),
  minP: z.number().min(0).max(1).catch(S.minP),
  repeatPenalty: z.number().min(0).catch(S.repeatPenalty),
  presencePenalty: z.number().catch(S.presencePenalty),
  frequencyPenalty: z.number().catch(S.frequencyPenalty),
  dryMultiplier: z.number().min(0).max(5).catch(S.dryMultiplier),
  dryBase: z.number().min(1).max(10).catch(S.dryBase),
  dryAllowedLength: z.number().int().min(1).max(1000).catch(S.dryAllowedLength),
  dryPenaltyLastN: z.number().int().min(-1).catch(S.dryPenaltyLastN),
  drySequenceBreakers: z.array(z.string()).max(64).catch(S.drySequenceBreakers),
  xtcProbability: z.number().min(0).max(1).catch(S.xtcProbability),
  xtcThreshold: z.number().min(0).max(1).catch(S.xtcThreshold),
  avoidWords: z.array(z.string()).max(500).catch(S.avoidWords),
  avoidStrength: z.number().min(0).max(100).catch(S.avoidStrength),
  maxTokens: posIntOrNull.catch(S.maxTokens),
  stop: z.array(z.string()).catch(S.stop),
  seed: z.number().int().nullable().catch(S.seed),
  jsonSchema: z.string().catch(S.jsonSchema),
  thinking: z.enum(['auto', 'on', 'off']).catch(S.thinking),
  reasoningEffort: z.enum(REASONING_EFFORTS as [string, ...string[]]).catch(S.reasoningEffort),
  thinkingBudget: z.number().int().min(0).nullable().catch(S.thinkingBudget),
  thinkingBudgetMessage: z.string().catch(S.thinkingBudgetMessage)
})

const D = DEFAULT_SETTINGS
export const settingsSchema = z.object({
  modelsDir: z.string().catch(D.modelsDir),
  enginePath: z.string().catch(D.enginePath),
  theme: z.enum(['dark', 'light', 'system']).catch(D.theme),
  themeChosen: z.boolean().catch(D.themeChosen),
  unloadPreviousOnLoad: z.boolean().catch(D.unloadPreviousOnLoad),
  defaultLoadConfig: loadConfigSchema.catch(D.defaultLoadConfig),
  defaultSampling: samplingSchema.catch(D.defaultSampling),
  defaultSystemPrompt: z.string().catch(D.defaultSystemPrompt),
  contextOverflow: z.enum(['stop', 'truncateMiddle']).catch(D.contextOverflow),
  expandThinking: z.boolean().catch(D.expandThinking),
  keepThinkingOpen: z.boolean().catch(D.keepThinkingOpen),
  rightPanelOpen: z.boolean().catch(D.rightPanelOpen),
  minimizeToTray: z.boolean().catch(D.minimizeToTray),
  closeToTray: z.boolean().catch(D.closeToTray),
  writingRoot: z.string().catch(D.writingRoot),
  phoneEnabled: z.boolean().catch(D.phoneEnabled),
  phonePort: z.number().int().min(1).max(65535).catch(D.phonePort),
  phoneToken: z.string().catch(D.phoneToken),
  toolsEnabled: z.boolean().catch(D.toolsEnabled),
  confirmToolCalls: z.boolean().catch(D.confirmToolCalls),
  allowedTools: z.array(z.string()).catch(D.allowedTools),
  disabledTools: z.array(z.string()).catch(D.disabledTools),
  toolMode: z.enum(['auto', 'all', 'dynamic']).catch(D.toolMode),
  toolBudgetTokens: z.number().int().min(500).max(200000).catch(D.toolBudgetTokens),
  compactToolResults: z.boolean().catch(D.compactToolResults),
  keepFullToolResults: z.number().int().min(0).max(100).catch(D.keepFullToolResults),
  apiServer: z
    .object({
      autoStart: z.boolean().catch(D.apiServer.autoStart),
      port: z.number().int().min(1).max(65535).catch(D.apiServer.port),
      bind: z.enum(['localhost', 'lan']).catch(D.apiServer.bind),
      apiKey: z.string().catch(D.apiServer.apiKey),
      cors: z.boolean().catch(D.apiServer.cors),
      idleUnloadMinutes: z.number().min(0).max(100000).catch(D.apiServer.idleUnloadMinutes),
      defaultModelId: z.string().catch(D.apiServer.defaultModelId),
      unknownModelFallback: z.boolean().catch(D.apiServer.unknownModelFallback)
    })
    .catch(D.apiServer),
  re: z
    .object({
      modelId: z.string().catch(D.re.modelId),
      kitDir: z.string().catch(D.re.kitDir),
      collect: z.boolean().catch(D.re.collect),
      autoApproveStatic: z.boolean().catch(D.re.autoApproveStatic),
      maxToolRounds: z.number().int().min(1).max(500).catch(D.re.maxToolRounds),
      toolBudgetTokens: z.number().int().min(500).max(200000).catch(D.re.toolBudgetTokens),
      recordApi: z.boolean().catch(D.re.recordApi),
      cloneMaxToolRounds: z.number().int().min(1).max(2000).catch(D.re.cloneMaxToolRounds)
    })
    .catch(D.re)
})

export const presetSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  systemPrompt: z.string().catch(''),
  sampling: samplingSchema.catch(S),
  updatedAt: z.number().catch(0)
})

export function parseSettings(raw: unknown): AppSettings {
  return settingsSchema.parse(raw ?? {}) as AppSettings
}

export function parseLoadConfig(raw: unknown): LoadConfig {
  return loadConfigSchema.parse(raw ?? {}) as LoadConfig
}

export function parseSampling(raw: unknown): SamplingParams {
  return samplingSchema.parse(raw ?? {}) as SamplingParams
}

/** One mode's entry: load settings or null, sampling (never with a JSON schema) or null, a prompt text. Never throws. */
export function parseModeEntry(raw: unknown): ModeDefaults {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const load = r.load && typeof r.load === 'object' && !Array.isArray(r.load) ? parseLoadConfig(r.load) : null
  const sampling = r.sampling && typeof r.sampling === 'object' && !Array.isArray(r.sampling) ? { ...parseSampling(r.sampling), jsonSchema: '' } : null
  return { load, sampling, systemPrompt: typeof r.systemPrompt === 'string' ? r.systemPrompt : '' }
}

/** One model's modes: unknown modes and entries that set nothing are dropped. */
export function parseModelModes(raw: unknown): ModelModeDefaults {
  const out: ModelModeDefaults = {}
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out
  const own = raw as Record<string, unknown>
  for (const mode of APP_MODES) {
    if (!Object.prototype.hasOwnProperty.call(own, mode)) continue
    const entry = parseModeEntry(own[mode])
    if (!modeDefaultsEmpty(entry)) out[mode] = entry
  }
  return out
}

/** The whole model-mode-defaults file. A bad entry never costs the others; a model with no modes left is dropped. */
export function parseModeDefaults(raw: unknown): AllModeDefaults {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {}
  const entries: [string, ModelModeDefaults][] = []
  for (const [id, modes] of Object.entries(raw)) {
    const parsed = parseModelModes(modes)
    if (Object.keys(parsed).length) entries.push([id, parsed])
  }
  // fromEntries makes '__proto__' an ordinary key instead of setting a prototype.
  return Object.fromEntries(entries)
}

export function parsePresets(raw: unknown): Preset[] {
  if (!Array.isArray(raw)) return []
  return raw.flatMap((p) => {
    const r = presetSchema.safeParse(p)
    return r.success ? [r.data as Preset] : []
  })
}
