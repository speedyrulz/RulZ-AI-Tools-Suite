import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { InstalledInfo, Settings } from '@shared/types'

export function defaultSettings(): Settings {
  const local = process.env.LOCALAPPDATA ?? join(process.env.USERPROFILE ?? 'C:\\', 'AppData', 'Local')
  return { installRoot: join(local, 'Programs', 'RulZ AI Tools'), checkUpdatesOnStart: true, liveReadme: true, closeOnLaunch: false }
}

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try {
    return JSON.parse(await readFile(path, 'utf8')) as T
  } catch {
    return fallback
  }
}

async function writeJson(path: string, data: unknown): Promise<void> {
  const tmp = `${path}.tmp`
  await writeFile(tmp, JSON.stringify(data, null, 2), 'utf8')
  await rename(tmp, path)
}

/** Settings and the list of installed apps, kept as JSON in the launcher's data folder. */
export class Store {
  private settings: Settings = defaultSettings()
  private installed: Record<string, InstalledInfo> = {}
  /** Apps whose found install the user removed from the launcher. */
  private detectHidden: string[] = []

  constructor(private readonly dir: string) {}

  async init(): Promise<void> {
    await mkdir(this.dir, { recursive: true })
    this.settings = { ...defaultSettings(), ...(await readJson<Partial<Settings>>(join(this.dir, 'settings.json'), {})) }
    this.installed = await readJson(join(this.dir, 'installed.json'), {})
    this.detectHidden = await readJson(join(this.dir, 'detect-hidden.json'), [])
  }

  getSettings(): Settings {
    return this.settings
  }

  async updateSettings(patch: Partial<Settings>): Promise<Settings> {
    this.settings = { ...this.settings, ...patch }
    await writeJson(join(this.dir, 'settings.json'), this.settings)
    return this.settings
  }

  getInstalled(id: string): InstalledInfo | null {
    return this.installed[id] ?? null
  }

  async setInstalled(id: string, info: InstalledInfo | null): Promise<void> {
    if (info) this.installed[id] = info
    else delete this.installed[id]
    await writeJson(join(this.dir, 'installed.json'), this.installed)
  }

  /** Small cached JSON (GitHub answers) kept next to the settings. */
  loadCache<T>(name: string, fallback: T): Promise<T> {
    return readJson(join(this.dir, name), fallback)
  }

  saveCache(name: string, data: unknown): Promise<void> {
    return writeJson(join(this.dir, name), data)
  }

  isDetectHidden(id: string): boolean {
    return this.detectHidden.includes(id)
  }

  async setDetectHidden(id: string, hidden: boolean): Promise<void> {
    if (hidden === this.isDetectHidden(id)) return
    this.detectHidden = hidden ? [...this.detectHidden, id] : this.detectHidden.filter((x) => x !== id)
    await writeJson(join(this.dir, 'detect-hidden.json'), this.detectHidden)
  }
}
