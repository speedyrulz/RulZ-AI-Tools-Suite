import { z } from 'zod'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING, DEFAULT_SETTINGS } from '@shared/defaults'
import { KV_CACHE_TYPES, type AppSettings, type LoadConfig, type Preset, type SamplingParams } from '@shared/types'

// Every field falls back to its default on its own, so one bad value never resets the rest.
const L = DEFAULT_LOAD_CONFIG
const kvType = z.enum(KV_CACHE_TYPES as [string, ...string[]])
const posIntOrNull = z.number().int().positive().nullable()

export const loadConfigSchema = z.object({
  contextLength: posIntOrNull.catch(L.contextLength),
  gpuLayers: z.union([z.literal('auto'), z.literal('all'), z.number().int().min(0)]).catch(L.gpuLayers),
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
  seed: z.number().int().nullable().catch(L.seed),
  chatTemplateFile: z.string().catch(L.chatTemplateFile),
  parallel: z.number().int().min(1).max(64).catch(L.parallel),
  extraArgs: z.string().catch(L.extraArgs)
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
  maxTokens: posIntOrNull.catch(S.maxTokens),
  stop: z.array(z.string()).catch(S.stop),
  seed: z.number().int().nullable().catch(S.seed),
  jsonSchema: z.string().catch(S.jsonSchema)
})

const D = DEFAULT_SETTINGS
export const settingsSchema = z.object({
  modelsDir: z.string().catch(D.modelsDir),
  enginePath: z.string().catch(D.enginePath),
  theme: z.enum(['dark', 'light', 'system']).catch(D.theme),
  unloadPreviousOnLoad: z.boolean().catch(D.unloadPreviousOnLoad),
  defaultLoadConfig: loadConfigSchema.catch(D.defaultLoadConfig),
  defaultSampling: samplingSchema.catch(D.defaultSampling),
  defaultSystemPrompt: z.string().catch(D.defaultSystemPrompt),
  contextOverflow: z.enum(['stop', 'truncateMiddle']).catch(D.contextOverflow),
  expandThinking: z.boolean().catch(D.expandThinking),
  rightPanelOpen: z.boolean().catch(D.rightPanelOpen),
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
      defaultModelId: z.string().catch(D.apiServer.defaultModelId)
    })
    .catch(D.apiServer)
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

export function parsePresets(raw: unknown): Preset[] {
  if (!Array.isArray(raw)) return []
  return raw.flatMap((p) => {
    const r = presetSchema.safeParse(p)
    return r.success ? [r.data as Preset] : []
  })
}
