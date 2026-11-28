import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'
import { DEFAULT_SETTINGS } from '../shared/types'
import type { AppSettings, Connection, OwnServerSettings, ServerDevice, Theme } from '../shared/types'

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function clampInt(v: unknown, min: number, max: number, fallback: number): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) return fallback
  return Math.min(max, Math.max(min, Math.round(v)))
}

function clampNum(v: unknown, min: number, max: number, fallback: number): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) return fallback
  return Math.min(max, Math.max(min, v))
}

function str(v: unknown, fallback: string): string {
  return typeof v === 'string' ? v : fallback
}

function cloneDefaults(): AppSettings {
  return structuredClone(DEFAULT_SETTINGS)
}

function validateDevice(v: unknown): ServerDevice {
  if (isRecord(v) && v.kind === 'gpu') {
    const idx = v.index
    if (typeof idx === 'number' && Number.isInteger(idx) && idx >= 0 && typeof v.name === 'string') {
      return { kind: 'gpu', index: idx, name: v.name }
    }
  }
  return { kind: 'cpu' }
}

function validateConnection(v: unknown): Connection {
  if (!isRecord(v)) return { mode: 'own', url: '' }
  return { mode: v.mode === 'url' ? 'url' : 'own', url: str(v.url, '') }
}

/** Fills in defaults and clamps every field of an untrusted settings object. */
export function normalizeSettings(raw: unknown): AppSettings {
  const d = DEFAULT_SETTINGS
  const r = isRecord(raw) ? raw : {}
  const o = isRecord(r.ownServer) ? r.ownServer : {}
  const theme: Theme = r.theme === 'light' ? 'light' : 'dark'
  const ownServer: OwnServerSettings = {
    distro: str(o.distro, d.ownServer.distro),
    comfyDir: str(o.comfyDir, d.ownServer.comfyDir),
    python: str(o.python, d.ownServer.python),
    baseDir: str(o.baseDir, d.ownServer.baseDir),
    modelsDir: str(o.modelsDir, d.ownServer.modelsDir),
    portStart: clampInt(o.portStart, 1024, 65535, d.ownServer.portStart),
    device: validateDevice(o.device),
    loadCustomNodes: typeof o.loadCustomNodes === 'boolean' ? o.loadCustomNodes : d.ownServer.loadCustomNodes,
    extraArgs: str(o.extraArgs, d.ownServer.extraArgs),
    minFreeRamGiB: clampNum(o.minFreeRamGiB, 0, 1024, d.ownServer.minFreeRamGiB),
    stopTimeoutSec: clampInt(o.stopTimeoutSec, 1, 600, d.ownServer.stopTimeoutSec)
  }
  return { theme, connection: validateConnection(r.connection), ownServer }
}

/** JSON settings file with validation, atomic writes and serialised updates. */
export class SettingsStore {
  private current: AppSettings = cloneDefaults()
  private queue: Promise<unknown> = Promise.resolve()
  /** The file could not be read (not just missing): defaults are in memory only and the file is never overwritten. */
  private readFailed = false

  constructor(private readonly file: string) {}

  /** Runs a job after every earlier job has finished, whether or not they failed. */
  private enqueue<T>(job: () => Promise<T>): Promise<T> {
    const result = this.queue.then(job, job)
    this.queue = result.catch(() => undefined)
    return result
  }

  load(): Promise<AppSettings> {
    return this.enqueue(() => this.loadNow())
  }

  private async loadNow(): Promise<AppSettings> {
    let text: string | null = null
    // readFailed stays set for the rest of the session once a read failed, even if a later load finds no file.
    try {
      text = await fs.readFile(this.file, 'utf8')
    } catch (e) {
      text = null
      // Only a missing file means "no settings yet". Any other error (locked, no access) keeps the file untouched.
      if ((e as NodeJS.ErrnoException).code !== 'ENOENT') this.readFailed = true
    }
    if (text === null) {
      this.current = cloneDefaults()
      return this.get()
    }
    try {
      this.current = normalizeSettings(JSON.parse(text.replace(/^﻿/, '')))
    } catch {
      try {
        await fs.copyFile(this.file, this.file + '.bak')
      } catch {
        // keep going with defaults
      }
      this.current = cloneDefaults()
    }
    return this.get()
  }

  get(): AppSettings {
    return structuredClone(this.current)
  }

  set(patch: Partial<AppSettings>): Promise<AppSettings> {
    return this.enqueue(async () => {
      const prev = this.current
      const merged: Record<string, unknown> = { ...prev, ...patch }
      if (patch.ownServer !== undefined) merged.ownServer = { ...prev.ownServer, ...patch.ownServer }
      // Readers see the new settings at once; a failed write puts the old ones back.
      this.current = normalizeSettings(merged)
      // The app stays usable; changes last for this session only, so a real settings file is not replaced by defaults.
      if (this.readFailed) return this.get()
      try {
        await this.write(this.current)
      } catch (e) {
        this.current = prev
        throw e
      }
      return this.get()
    })
  }

  private async write(s: AppSettings): Promise<void> {
    await fs.mkdir(dirname(this.file), { recursive: true })
    const tmp = `${this.file}.${process.pid}.tmp`
    try {
      await fs.writeFile(tmp, JSON.stringify(s, null, 2) + '\n', 'utf8')
      await fs.rename(tmp, this.file)
    } catch (e) {
      await fs.rm(tmp, { force: true }).catch(() => undefined)
      throw e
    }
  }
}
