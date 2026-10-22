import { join } from 'node:path'
import type { AppSettings, LoadConfig, Preset } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { parseLoadConfig, parsePresets, parseSettings } from './schema'

/** App settings, per-model load defaults and presets, each persisted as JSON in userData. */
export class SettingsStore {
  private settings!: AppSettings
  private modelDefaults: Record<string, LoadConfig> = {}
  private presets: Preset[] = []
  private readonly settingsPath: string
  private readonly modelDefaultsPath: string
  private readonly presetsPath: string

  constructor(dir: string) {
    this.settingsPath = join(dir, 'settings.json')
    this.modelDefaultsPath = join(dir, 'model-defaults.json')
    this.presetsPath = join(dir, 'presets.json')
  }

  async init(): Promise<void> {
    this.settings = parseSettings(await readJson(this.settingsPath, {}))
    const rawDefaults = await readJson<Record<string, unknown>>(this.modelDefaultsPath, {})
    for (const [id, cfg] of Object.entries(rawDefaults)) this.modelDefaults[id] = parseLoadConfig(cfg)
    this.presets = parsePresets(await readJson(this.presetsPath, []))
  }

  get(): AppSettings {
    return this.settings
  }

  async update(patch: Partial<AppSettings>): Promise<AppSettings> {
    this.settings = parseSettings({ ...this.settings, ...patch })
    await writeJsonAtomic(this.settingsPath, this.settings)
    return this.settings
  }

  getModelDefaults(modelId: string): { config: LoadConfig; custom: boolean } {
    const custom = this.modelDefaults[modelId]
    return custom ? { config: custom, custom: true } : { config: this.settings.defaultLoadConfig, custom: false }
  }

  async setModelDefaults(modelId: string, config: LoadConfig | null): Promise<void> {
    if (config) this.modelDefaults[modelId] = parseLoadConfig(config)
    else delete this.modelDefaults[modelId]
    await writeJsonAtomic(this.modelDefaultsPath, this.modelDefaults)
  }

  listPresets(): Preset[] {
    return [...this.presets].sort((a, b) => a.name.localeCompare(b.name))
  }

  async savePreset(preset: Preset): Promise<Preset[]> {
    const [parsed] = parsePresets([{ ...preset, updatedAt: Date.now() }])
    if (!parsed) throw new Error('Invalid preset')
    this.presets = [...this.presets.filter((p) => p.id !== parsed.id), parsed]
    await writeJsonAtomic(this.presetsPath, this.presets)
    return this.listPresets()
  }

  async deletePreset(id: string): Promise<Preset[]> {
    this.presets = this.presets.filter((p) => p.id !== id)
    await writeJsonAtomic(this.presetsPath, this.presets)
    return this.listPresets()
  }
}
