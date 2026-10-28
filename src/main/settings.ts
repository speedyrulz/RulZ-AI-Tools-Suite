import { join } from 'node:path'
import { z } from 'zod'
import { DEFAULT_SETTINGS, MAX_INSTRUCTION_CHARS } from '@shared/defaults'
import type { AppSettings } from '@shared/types'
import { readJson, writeJsonAtomic } from './util/jsonFile'

const D = DEFAULT_SETTINGS

const schema = z.object({
  serverUrl: z.string().catch(D.serverUrl),
  lastServerUrl: z.string().catch(D.lastServerUrl),
  apiKey: z.string().catch(D.apiKey),
  model: z.string().catch(D.model),
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
