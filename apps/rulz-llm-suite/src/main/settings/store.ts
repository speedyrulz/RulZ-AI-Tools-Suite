import { promises as fsp } from 'node:fs'
import { basename, join } from 'node:path'
import { APP_MODES, modeDefaultsEmpty, modeLoad, type AllModeDefaults, type AppMode, type ModeDefaults, type ModelModeDefaults } from '@shared/modes'
import type { AppSettings, LoadConfig, Preset } from '@shared/types'
import { corruptName, readJson, writeJsonAtomic } from '../util/jsonFile'
import { parseLoadConfig, parseModeDefaults, parseModeEntry, parsePresets, parseSettings } from './schema'

/** App settings, per-model load defaults and presets, each persisted as JSON in userData. */
export class SettingsStore {
  private settings!: AppSettings
  private modelDefaults: Record<string, LoadConfig> = {}
  /** Per model and mode; its own file so older versions keep reading model-defaults.json. A Map, so ids like '__proto__' are plain keys. */
  private modeMap = new Map<string, ModelModeDefaults>()
  private modeChain: Promise<unknown> = Promise.resolve()
  private updateChain: Promise<unknown> = Promise.resolve()
  private presets: Preset[] = []
  private readonly settingsPath: string
  private readonly modelDefaultsPath: string
  private readonly presetsPath: string
  private readonly modeDefaultsPath: string
  /** Files that exist but couldn't be read at startup. Writing them would replace the user's data with defaults. */
  private readonly unreadable = new Set<string>()

  constructor(dir: string) {
    this.settingsPath = join(dir, 'settings.json')
    this.modelDefaultsPath = join(dir, 'model-defaults.json')
    this.presetsPath = join(dir, 'presets.json')
    this.modeDefaultsPath = join(dir, 'model-mode-defaults.json')
  }

  async init(): Promise<void> {
    this.settings = parseSettings(await this.load(this.settingsPath, {}, 'object'))
    const rawDefaults = await this.load<Record<string, unknown>>(this.modelDefaultsPath, {}, 'object')
    for (const [id, cfg] of Object.entries(rawDefaults)) this.modelDefaults[id] = parseLoadConfig(cfg)
    this.presets = parsePresets(await this.load(this.presetsPath, [], 'array'))
    this.modeMap = new Map(Object.entries(parseModeDefaults(await this.load(this.modeDefaultsPath, {}, 'object'))))
  }

  /** The app still starts on defaults when a file is locked; that file is then left alone. */
  private async load<T>(path: string, fallback: T, kind: 'object' | 'array'): Promise<T> {
    try {
      const value = await readJson<unknown>(path, fallback)
      const ok = kind === 'array' ? Array.isArray(value) : typeof value === 'object' && value !== null && !Array.isArray(value)
      if (ok) return value as T
      // Valid JSON of the wrong kind (a bare number, null, a list): treated like a file that does not parse.
      const aside = corruptName(path)
      await fsp.rename(path, aside).catch(() => fsp.copyFile(path, aside).catch(() => undefined))
      console.warn(`${path} did not hold the expected kind of data; kept it as ${aside} and used defaults.`)
      return fallback
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
    // One after another, so each patch starts from what the one before it saved.
    const run = this.updateChain.then(async () => {
      const next = parseSettings({ ...this.settings, ...patch })
      // Memory only changes once the file is written.
      await this.save(this.settingsPath, next)
      this.settings = next
      return next
    })
    this.updateChain = run.catch(() => undefined)
    return run
  }

  getModelDefaults(modelId: string): { config: LoadConfig; custom: boolean } {
    const custom = Object.prototype.hasOwnProperty.call(this.modelDefaults, modelId) ? this.modelDefaults[modelId] : undefined
    return custom ? { config: custom, custom: true } : { config: this.settings.defaultLoadConfig, custom: false }
  }

  async setModelDefaults(modelId: string, config: LoadConfig | null): Promise<void> {
    if (config) this.modelDefaults[modelId] = parseLoadConfig(config)
    else delete this.modelDefaults[modelId]
    await this.save(this.modelDefaultsPath, this.modelDefaults)
  }

  /** Every model's settings per mode (a copy). */
  allModeDefaults(): AllModeDefaults {
    return structuredClone(Object.fromEntries(this.modeMap))
  }

  /** One model's settings per mode (a copy); {} when it has none. */
  modeDefaults(modelId: string): ModelModeDefaults {
    return structuredClone(this.modeMap.get(modelId) ?? {})
  }

  /** Saves one mode of one model; null, or an entry that sets nothing, forgets it. Memory only changes when the file was written. */
  async setModeDefaults(modelId: string, mode: AppMode, defaults: ModeDefaults | null): Promise<void> {
    if (!APP_MODES.includes(mode)) throw new Error(`Unknown mode: ${String(mode)}`)
    const entry = defaults ? parseModeEntry(defaults) : null
    await this.changeModes((all) => {
      const next: ModelModeDefaults = { ...all.get(modelId) }
      if (entry && !modeDefaultsEmpty(entry)) next[mode] = entry
      else delete next[mode]
      if (Object.keys(next).length) all.set(modelId, next)
      else all.delete(modelId)
    })
  }

  /**
   * Sets or removes only the load settings of one mode of one model; its sampling and prompt stay as they are, read inside
   * the queue so a save that is still running is not undone. The API-server sampling in a mode's load is not the mode's to
   * set: an entry that has one keeps it, a new one gets the model's default load settings'. null removes the mode's own
   * load, and an entry left with nothing is dropped. A function gets the load the mode has when the save's turn comes.
   */
  async setModeLoad(modelId: string, mode: AppMode, load: LoadConfig | null | ((current: LoadConfig) => LoadConfig)): Promise<void> {
    if (!APP_MODES.includes(mode)) throw new Error(`Unknown mode: ${String(mode)}`)
    await this.changeModes((all) => {
      const wanted = typeof load === 'function' ? load(modeLoad(this.getModelDefaults(modelId).config, all.get(modelId), mode)) : load
      const parsed = wanted ? parseLoadConfig(wanted) : null
      const existing = all.get(modelId)?.[mode]
      const next: ModeDefaults = {
        load: parsed ? { ...parsed, sampling: existing?.load?.sampling ?? this.getModelDefaults(modelId).config.sampling } : null,
        sampling: existing?.sampling ?? null,
        systemPrompt: existing?.systemPrompt ?? ''
      }
      const modes: ModelModeDefaults = { ...all.get(modelId) }
      if (modeDefaultsEmpty(next)) delete modes[mode]
      else modes[mode] = next
      if (Object.keys(modes).length) all.set(modelId, modes)
      else all.delete(modelId)
    })
  }

  /** Changes to the modes file go one after another: each starts from what the one before it saved, so none is lost. */
  private changeModes(change: (all: Map<string, ModelModeDefaults>) => void): Promise<void> {
    const run = this.modeChain.then(async () => {
      const all = new Map(this.modeMap)
      change(all)
      await this.save(this.modeDefaultsPath, Object.fromEntries(all))
      this.modeMap = all
    })
    this.modeChain = run.catch(() => undefined)
    return run
  }

  /** A deleted or forgotten model: its default load settings and all its modes go; only the files that change are written. */
  async forgetModel(modelId: string): Promise<void> {
    if (Object.prototype.hasOwnProperty.call(this.modelDefaults, modelId)) await this.setModelDefaults(modelId, null)
    // After a save still under way, which may be this model's first.
    await this.modeChain
    if (this.modeMap.has(modelId)) await this.changeModes((all) => all.delete(modelId))
  }

  /** The load settings a model gets in a mode (its default ones without a mode, or when the mode sets none). */
  loadFor(modelId: string, mode?: AppMode): LoadConfig {
    return structuredClone(modeLoad(this.getModelDefaults(modelId).config, this.modeMap.get(modelId), mode))
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
