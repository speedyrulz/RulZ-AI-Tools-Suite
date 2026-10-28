import { join } from 'node:path'
import { z } from 'zod'
import { DEFAULT_SETTINGS, MAX_INSTRUCTION_CHARS, REASONING_EFFORTS } from '@shared/defaults'
import type { AppSettings, ReasoningEffort } from '@shared/types'
import { readJson, writeJsonAtomic } from './util/jsonFile'

const D = DEFAULT_SETTINGS

const schema = z.object({
  serverUrl: z.string().catch(D.serverUrl),
  lastServerUrl: z.string().catch(D.lastServerUrl),
  apiKey: z.string().catch(D.apiKey),
  model: z.string().catch(D.model),
  modelSampling: z
    .record(
      z.string(),
      z.object({
        temperature: z.number().min(0).max(2).optional().catch(undefined),
        topP: z.number().min(0).max(1).optional().catch(undefined),
        topK: z.number().int().min(0).max(1000).optional().catch(undefined),
        repeatPenalty: z.number().min(0.5).max(2).optional().catch(undefined),
        maxOutputTokens: z.number().int().min(256).max(131072).optional().catch(undefined),
        thinking: z.enum(['on', 'off']).optional().catch(undefined),
        reasoningEffort: z.enum(REASONING_EFFORTS as [ReasoningEffort, ...ReasoningEffort[]]).optional().catch(undefined),
        thinkingBudget: z.number().int().min(0).max(131072).optional().catch(undefined),
        thinkingBudgetMessage: z.string().max(500).optional().catch(undefined)
      })
    )
    .catch(D.modelSampling),
  approval: z.enum(['careful', 'balanced', 'handsoff']).catch(D.approval),
  maxSteps: z.number().int().min(5).max(500).catch(D.maxSteps),
  commandTimeoutMin: z.number().min(1).max(120).catch(D.commandTimeoutMin),
  theme: z.enum(['dark', 'light', 'system']).catch(D.theme),
  projectsDir: z.string().min(1).catch(D.projectsDir),
  instructions: z
    .string()
    .transform((s) => s.trim().slice(0, MAX_INSTRUCTION_CHARS))
    .catch(D.instructions),
  firstRunDone: z.boolean().catch(D.firstRunDone),
  remoteEnabled: z.boolean().catch(D.remoteEnabled),
  remotePort: z.number().int().min(1024).max(65535).catch(D.remotePort),
  remoteToken: z.string().catch(D.remoteToken)
})

export function parseSettings(raw: unknown): AppSettings {
  const obj = raw && typeof raw === 'object' ? raw : {}
  return schema.parse({ ...D, ...obj })
}

export class SettingsStore {
  private settings: AppSettings = D
  private readonly path: string

  constructor(dir: string) {
    this.path = join(dir, 'settings.json')
  }

  async init(): Promise<void> {
    this.settings = parseSettings(await readJson(this.path, {}))
  }

  get(): AppSettings {
    return this.settings
  }

  async update(patch: Partial<AppSettings>): Promise<AppSettings> {
    this.settings = parseSettings({ ...this.settings, ...patch })
    await writeJsonAtomic(this.path, this.settings)
    return this.settings
  }
}
