import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
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

const writes = new Map<string, Promise<void>>()
let tmpSeq = 0

/** Windows refuses a rename while another program (an antivirus scan, say) has the file open; that passes quickly. */
async function renameRetrying(from: string, to: string): Promise<void> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await rename(from, to)
    } catch (e) {
      const code = (e as NodeJS.ErrnoException).code
      if (attempt >= 5 || (code !== 'EPERM' && code !== 'EBUSY' && code !== 'EACCES')) throw e
      await new Promise((r) => setTimeout(r, 50 * attempt))
    }
  }
}

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
