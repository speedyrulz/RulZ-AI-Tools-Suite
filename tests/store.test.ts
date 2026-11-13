import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Store, writeJson } from '../src/main/store'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-store-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

describe('writeJson', () => {
  it('runs overlapping writes one after another, so the last call wins', async () => {
    const file = join(dir, 'x.json')
    await Promise.all(Array.from({ length: 25 }, (_, i) => writeJson(file, { i })))
    expect(JSON.parse(await readFile(file, 'utf8'))).toEqual({ i: 24 })
    expect((await readdir(dir)).filter((f) => f.endsWith('.tmp'))).toEqual([])
  })

  it('keeps going after a failed write', async () => {
    const file = join(dir, 'missing-folder', 'x.json')
    await expect(writeJson(file, 1)).rejects.toThrow()
    const ok = join(dir, 'y.json')
    await writeJson(ok, 2)
    expect(JSON.parse(await readFile(ok, 'utf8'))).toBe(2)
  })
})

describe('Store', () => {
  it('saves every installed record when installs finish at the same time', async () => {
    const store = new Store(dir)
    await store.init()
    const info = (n: number) => ({
      dir: `C:\\a${n}`,
      launchPath: `C:\\a${n}\\a.exe`,
      launchArgs: [],
      version: null,
      method: 'zip' as const,
      managed: true,
      installedAt: '',
      shortcuts: { desktop: null, startMenu: null },
      components: [],
      needsSetup: false
    })
    await Promise.all([1, 2, 3, 4].map((n) => store.setInstalled(`app${n}`, info(n))))
    const again = new Store(dir)
    await again.init()
    for (const n of [1, 2, 3, 4]) expect(again.getInstalled(`app${n}`)?.dir).toBe(`C:\\a${n}`)
  })

  it('falls back to the last good list when installed.json is damaged, keeping the damaged file', async () => {
    const store = new Store(dir)
    await store.init()
    await store.setInstalled('app1', {
      dir: 'C:\\a1',
      launchPath: 'C:\\a1\\a.exe',
      launchArgs: [],
      version: 'v1.0.0',
      method: 'zip',
      managed: true,
      installedAt: '',
      shortcuts: { desktop: null, startMenu: null },
      components: [],
      needsSetup: false
    })
    // A launch reads it and keeps a copy; then a save is cut short.
    await new Store(dir).init()
    await writeFile(join(dir, 'installed.json'), '')
    const after = new Store(dir)
    await after.init()
    expect(after.getInstalled('app1')?.version).toBe('v1.0.0')
    expect(await readFile(join(dir, 'installed.json.damaged'), 'utf8')).toBe('')
  })

  it('starts empty on the first run', async () => {
    const store = new Store(dir)
    await store.init()
    expect(store.getInstalled('app1')).toBeNull()
  })
})

describe('launch options', () => {
  it('keeps quick toggles of different options and apps, and reads settings saved without the field', async () => {
    await writeFile(join(dir, 'settings.json'), JSON.stringify({ closeOnLaunch: true }))
    const store = new Store(dir)
    await store.init()
    expect(store.getSettings().launchOptions).toBeUndefined()
    await Promise.all([
      store.setLaunchOption('rulz-trainer', 'tailscale', true),
      store.setLaunchOption('rulz-trainer', 'other', true),
      store.setLaunchOption('llm-suite', 'x', true)
    ])
    await store.setLaunchOption('rulz-trainer', 'other', false)
    await store.setLaunchOption('rulz-trainer', 'tailscale', true)
    const again = new Store(dir)
    await again.init()
    expect(again.getSettings().launchOptions).toEqual({ 'rulz-trainer': ['tailscale'], 'llm-suite': ['x'] })
    expect(again.getSettings().closeOnLaunch).toBe(true)
  })

  it('replaces a hand-edited value that is not a list', async () => {
    await writeFile(join(dir, 'settings.json'), JSON.stringify({ launchOptions: { 'rulz-trainer': 'tailscale' } }))
    const store = new Store(dir)
    await store.init()
    expect((await store.setLaunchOption('rulz-trainer', 'tailscale', false)).launchOptions).toEqual({ 'rulz-trainer': [] })
  })
})
