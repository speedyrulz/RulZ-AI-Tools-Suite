import { copyFile, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { InstalledInfo, Settings } from '@shared/types'
import { renameRetrying } from './archive'

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

const writes = new Map<string, Promise<void>>()
let tmpSeq = 0

/**
 * Writes JSON atomically (a temporary file renamed over the old one). Writes to the same file run one after another
 * in call order, each with its own temporary file, so the last call's data is what ends up on disk.
 */
export function writeJson(path: string, data: unknown): Promise<void> {
  const text = JSON.stringify(data, null, 2)
  const next = (writes.get(path) ?? Promise.resolve())
    .catch(() => undefined)
    .then(async () => {
      const tmp = `${path}.${process.pid}.${++tmpSeq}.tmp`
      try {
        await writeFile(tmp, text, 'utf8')
        await renameRetrying(tmp, path)
      } catch (e) {
        await rm(tmp, { force: true }).catch(() => undefined)
        throw e
      }
    })
  writes.set(path, next)
  void next
    .catch(() => undefined)
    .then(() => {
      if (writes.get(path) === next) writes.delete(path)
    })
  return next
}

/**
 * Reads the list of installed apps. Each launch that reads it keeps a copy (.bak). A damaged file (a save cut short by
 * a power cut, say) is moved aside instead of being overwritten by the next save, and that copy is used instead:
 * forgetting the list would leave every app installed but unknown to the launcher.
 */
export async function readInstalled(path: string): Promise<Record<string, InstalledInfo>> {
  const backup = `${path}.bak`
  const text = await readFile(path, 'utf8').catch(() => null)
  if (text === null) return {}
  try {
    const list: unknown = JSON.parse(text)
    if (!list || typeof list !== 'object' || Array.isArray(list)) throw new Error('not a list of apps')
    await copyFile(path, backup).catch(() => undefined)
    return list as Record<string, InstalledInfo>
  } catch {
    await rename(path, `${path}.damaged`).catch(() => undefined)
    return readJson(backup, {})
  }
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
    this.installed = await readInstalled(join(this.dir, 'installed.json'))
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

  /** Turns one launch option of an app on or off. Other apps' options and this app's others are kept. */
  setLaunchOption(appId: string, optionId: string, on: boolean): Promise<Settings> {
    const all = this.settings.launchOptions ?? {}
    const current = Array.isArray(all[appId]) ? all[appId].filter((o) => o !== optionId) : []
    return this.updateSettings({ launchOptions: { ...all, [appId]: on ? [...current, optionId] : current } })
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
