import { mkdtempSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { SettingsStore } from '../src/main/settings'
import { DEFAULT_SETTINGS } from '../src/shared/types'

let dir: string
let file: string
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'merger-settings-'))
  file = join(dir, 'sub', 'settings.json')
})
afterEach(() => rmSync(dir, { recursive: true, force: true }))

describe('SettingsStore', () => {
  it('returns defaults when the file is missing', async () => {
    const s = new SettingsStore(file)
    expect(await s.load()).toEqual(DEFAULT_SETTINGS)
  })

  it('keeps a .bak copy of a corrupt file and uses defaults', async () => {
    const bad = join(dir, 'bad.json')
    writeFileSync(bad, '{ nope')
    const s = new SettingsStore(bad)
    expect(await s.load()).toEqual(DEFAULT_SETTINGS)
    expect(readFileSync(bad + '.bak', 'utf8')).toBe('{ nope')
  })

  it('merges loaded data over defaults, one level deep for ownServer', async () => {
    const f = join(dir, 'a.json')
    writeFileSync(f, JSON.stringify({ theme: 'light', ownServer: { distro: 'Debian' } }))
    const got = await new SettingsStore(f).load()
    expect(got.theme).toBe('light')
    expect(got.ownServer.distro).toBe('Debian')
    expect(got.ownServer.portStart).toBe(DEFAULT_SETTINGS.ownServer.portStart)
  })

  it('clamps and validates values', async () => {
    const s = new SettingsStore(file)
    await s.load()
    const got = await s.set({
      theme: 'purple' as never,
      connection: { mode: 'weird' } as never,
      ownServer: {
        ...DEFAULT_SETTINGS.ownServer,
        portStart: 80,
        minFreeRamGiB: 5000,
        stopTimeoutSec: 0,
        device: { kind: 'gpu', index: -1, name: 'x' }
      }
    })
    expect(got.theme).toBe('dark')
    expect(got.connection).toEqual({ mode: 'own', url: '' })
    expect(got.ownServer.portStart).toBe(1024)
    expect(got.ownServer.minFreeRamGiB).toBe(1024)
    expect(got.ownServer.stopTimeoutSec).toBe(1)
    expect(got.ownServer.device).toEqual({ kind: 'cpu' })
    const ok = await s.set({ ownServer: { device: { kind: 'gpu', index: 1, name: 'RTX' } } as never })
    expect(ok.ownServer.device).toEqual({ kind: 'gpu', index: 1, name: 'RTX' })
    expect(ok.ownServer.portStart).toBe(1024)
  })

  it('persists with an atomic write and no temp file left behind', async () => {
    const s = new SettingsStore(file)
    await s.set({ connection: { mode: 'url', url: 'http://127.0.0.1:8188' } })
    const again = await new SettingsStore(file).load()
    expect(again.connection).toEqual({ mode: 'url', url: 'http://127.0.0.1:8188' })
    expect(existsSync(`${file}.${process.pid}.tmp`)).toBe(false)
  })

  it('serialises concurrent set calls', async () => {
    const s = new SettingsStore(file)
    await s.load()
    await Promise.all([2000, 3000, 4000, 5000].map((p) => s.set({ ownServer: { portStart: p } as never })))
    expect(s.get().ownServer.portStart).toBe(5000)
    expect(JSON.parse(readFileSync(file, 'utf8')).ownServer.portStart).toBe(5000)
  })

  it('shows new settings to readers while the write is still running', async () => {
    const s = new SettingsStore(file)
    await s.load()
    const pending = s.set({ ownServer: { portStart: 9000 } as never })
    // The queued job starts on the next tick; the disk write after it takes longer.
    await new Promise((r) => setImmediate(r))
    expect(s.get().ownServer.portStart).toBe(9000)
    expect(existsSync(file)).toBe(false)
    await pending
  })

  it('rolls back and leaves no temp file when the write fails', async () => {
    const blocker = join(dir, 'blocker')
    writeFileSync(blocker, 'x')
    const s = new SettingsStore(join(blocker, 'settings.json'))
    await s.load()
    await expect(s.set({ ownServer: { portStart: 9000 } as never })).rejects.toThrow()
    expect(s.get().ownServer.portStart).toBe(DEFAULT_SETTINGS.ownServer.portStart)
    // The queue keeps working after a failure.
    await expect(s.set({ theme: 'light' })).rejects.toThrow()
  })

  it('runs load after queued sets', async () => {
    const s = new SettingsStore(file)
    await s.load()
    void s.set({ theme: 'light' })
    const loaded = await s.load()
    expect(loaded.theme).toBe('light')
  })
})
