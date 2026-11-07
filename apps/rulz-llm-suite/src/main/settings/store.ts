import { basename, join } from 'node:path'
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
  /** Files that exist but couldn't be read at startup. Writing them would replace the user's data with defaults. */
  private readonly unreadable = new Set<string>()

  constructor(dir: string) {
    this.settingsPath = join(dir, 'settings.json')
    this.modelDefaultsPath = join(dir, 'model-defaults.json')
    this.presetsPath = join(dir, 'presets.json')
  }

  async init(): Promise<void> {
    this.settings = parseSettings(await this.load(this.settingsPath, {}))
    const rawDefaults = await this.load<Record<string, unknown>>(this.modelDefaultsPath, {})
    for (const [id, cfg] of Object.entries(rawDefaults)) this.modelDefaults[id] = parseLoadConfig(cfg)
    this.presets = parsePresets(await this.load(this.presetsPath, []))
  }

  /** The app still starts on defaults when a file is locked; that file is then left alone. */
  private async load<T>(path: string, fallback: T): Promise<T> {
    try {
      return await readJson(path, fallback)
    } catch (err) {
      console.error(`Could not read ${path}:`, err)
      this.unreadable.add(path)
      return fallback
    }
  }

  private async save(path: string, data: unknown): Promise<void> {
    if (this.unreadable.has(path)) {
      throw new Error(`${basename(path)} couldn't be read when RulZ LLM Suite started, so it wasn't overwritten. Restart the app to load it.`)
    }
    await writeJsonAtomic(path, data)
  }

  get(): AppSettings {
    return this.settings
  }

  async update(patch: Partial<AppSettings>): Promise<AppSettings> {
    this.settings = parseSettings({ ...this.settings, ...patch })
    await this.save(this.settingsPath, this.settings)
    return this.settings
  }

  getModelDefaults(modelId: string): { config: LoadConfig; custom: boolean } {
    const custom = this.modelDefaults[modelId]
    return custom ? { config: custom, custom: true } : { config: this.settings.defaultLoadConfig, custom: false }
  }

  async setModelDefaults(modelId: string, config: LoadConfig | null): Promise<void> {
    if (config) this.modelDefaults[modelId] = parseLoadConfig(config)
    else delete this.modelDefaults[modelId]
    await this.save(this.modelDefaultsPath, this.modelDefaults)
  }

  listPresets(): Preset[] {
    return [...this.presets].sort((a, b) => a.name.localeCompare(b.name))
  }

  async savePreset(preset: Preset): Promise<Preset[]> {
    const [parsed] = parsePresets([{ ...preset, updatedAt: Date.now() }])
    if (!parsed) throw new Error('Invalid preset')
    this.presets = [...this.presets.filter((p) => p.id !== parsed.id), parsed]
    await this.save(this.presetsPath, this.presets)
    return this.listPresets()
  }

  async deletePreset(id: string): Promise<Preset[]> {
    this.presets = this.presets.filter((p) => p.id !== id)
    await this.save(this.presetsPath, this.presets)
    return this.listPresets()
  }
}
