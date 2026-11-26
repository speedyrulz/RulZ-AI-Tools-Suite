import { copyFile, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { cleanOrder } from '@shared/arrange'
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

/**
 * Reads settings.json. One that exists but is not a JSON object is moved to .damaged, so the next save does not erase it.
 * One that cannot be read (locked by an antivirus or sync tool) gives `locked: true`: the caller must not overwrite it.
 */
async function readSettings(path: string): Promise<{ settings: Partial<Settings>; locked: boolean }> {
  let text: string
  try {
    text = await readLocked(path)
  } catch (e) {
    return { settings: {}, locked: (e as NodeJS.ErrnoException).code !== 'ENOENT' }
  }
  try {
    const parsed: unknown = JSON.parse(text)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return { settings: parsed as Partial<Settings>, locked: false }
  } catch {
    // falls through to the damaged-file handling
  }
  await rename(path, `${path}.damaged`).catch(() => undefined)
  return { settings: {}, locked: false }
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

/** Reads a file, trying a few times when another program holds it open; a missing file fails at once. */
async function readLocked(path: string): Promise<string> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await readFile(path, 'utf8')
    } catch (e) {
      if (attempt >= 5 || (e as NodeJS.ErrnoException).code === 'ENOENT') throw e
      await new Promise((r) => setTimeout(r, 50 * attempt))
    }
  }
}

/**
 * Reads the list of installed apps. Each launch that reads it keeps a copy (.bak). A damaged file (a save cut short by
 * a power cut, say) is moved aside instead of being overwritten by the next save, and that copy is used instead:
 * forgetting the list would leave every app installed but unknown to the launcher.
 */
export async function readInstalled(path: string): Promise<Record<string, InstalledInfo>> {
  return (await readInstalledState(path)).list
}

/** As readInstalled, and says whether the list is the stale .bak used because installed.json could not be read. */
async function readInstalledState(path: string): Promise<{ list: Record<string, InstalledInfo>; fromBackup: boolean }> {
  const backup = `${path}.bak`
  let text: string
  try {
    text = await readLocked(path)
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return { list: {}, fromBackup: false }
    // Locked or unreadable (an antivirus or sync tool): not "nothing installed". The backup is only read, never
    // copied over or renamed, so the file that could not be read is left alone for the next launch.
    const saved = await readFile(backup, 'utf8').then(
      (t) => {
        try {
          return JSON.parse(t) as unknown
        } catch {
          return null // a truncated .bak is as unusable as a missing one
        }
      },
      () => null
    )
    if (saved && typeof saved === 'object' && !Array.isArray(saved)) return { list: saved as Record<string, InstalledInfo>, fromBackup: true }
    throw new Error(`installed.json can't be read: ${e instanceof Error ? e.message : String(e)}`)
  }
  try {
    const list: unknown = JSON.parse(text)
    if (!list || typeof list !== 'object' || Array.isArray(list)) throw new Error('not a list of apps')
    await copyFile(path, backup).catch(() => undefined)
    return { list: list as Record<string, InstalledInfo>, fromBackup: false }
  } catch {
    await rename(path, `${path}.damaged`).catch(() => undefined)
    return { list: await readJson(backup, {}), fromBackup: false }
  }
}

/** Settings and the list of installed apps, kept as JSON in the launcher's data folder. */
export class Store {
  private settings: Settings = defaultSettings()
  private installed: Record<string, InstalledInfo> = {}
  /** Apps whose found install the user removed from the launcher. */
  private detectHidden: string[] = []

  /**
   * Set while settings.json could not be read at startup: the defaults in memory are not what the file holds, so the
   * changes made meanwhile are kept here and applied on top of the file once it can be read. Nothing is written before.
   */
  private settingsPending: Partial<Settings> | null = null
  /** Set while the installed list in memory is the stale .bak (installed.json was locked): changes wait here likewise. */
  private installedPending: Map<string, InstalledInfo | null> | null = null

  constructor(private readonly dir: string) {}

  async init(): Promise<void> {
    await mkdir(this.dir, { recursive: true })
    const read = await readSettings(join(this.dir, 'settings.json'))
    this.settings = { ...defaultSettings(), ...read.settings }
    this.settingsPending = read.locked ? {} : null
    this.cleanOrder()
    const state = await readInstalledState(join(this.dir, 'installed.json'))
    this.installed = state.list
    this.installedPending = state.fromBackup ? new Map() : null
    this.detectHidden = await readJson(join(this.dir, 'detect-hidden.json'), [])
  }

  getSettings(): Settings {
    return this.settings
  }

  async updateSettings(patch: Partial<Settings>): Promise<Settings> {
    const file = join(this.dir, 'settings.json')
    if (this.settingsPending) {
      Object.assign(this.settingsPending, patch)
      const read = await readSettings(file)
      if (!read.locked) {
        this.settings = { ...defaultSettings(), ...read.settings, ...this.settingsPending }
        this.settingsPending = null
      }
    }
    this.settings = { ...this.settings, ...patch }
    this.cleanOrder()
    // Still locked: the change stays in memory, and the file is not overwritten with defaults.
    if (!this.settingsPending) await writeJson(file, this.settings)
    return this.settings
  }

  /** installedOrder must be a list of distinct strings; anything else, and an empty list, leaves it unset. */
  private cleanOrder(): void {
    const order = cleanOrder(this.settings.installedOrder)
    if (order) this.settings.installedOrder = order
    else delete this.settings.installedOrder
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
    const file = join(this.dir, 'installed.json')
    const pending = this.installedPending
    if (pending) {
      pending.set(id, info)
      // The list in memory is the stale .bak: the real file is the base again once it can be read.
      const state = await readInstalledState(file).catch(() => null)
      // Calls made at once (the startup search saves several apps) all wait here: only the first to get back rebases.
      if (state && !state.fromBackup && this.installedPending === pending) {
        this.installed = state.list
        for (const [pid, pinfo] of pending) {
          if (pinfo) this.installed[pid] = pinfo
          else delete this.installed[pid]
        }
        this.installedPending = null
      }
    }
    if (info) this.installed[id] = info
    else delete this.installed[id]
    // Still unreadable: the change stays in memory only, so the real file is never overwritten with the stale copy.
    if (!this.installedPending) await writeJson(file, this.installed)
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
