import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Failures are injected through these: a rename whose target is listed fails, a read of a listed file fails.
const fail = vi.hoisted(() => ({
  renameTo: new Set<string>(),
  renameFrom: new Set<string>(),
  readFrom: new Set<string>(),
  statOf: new Set<string>()
}))

vi.mock('node:fs/promises', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs/promises')>()
  const errno = (code: string) => Object.assign(new Error(`${code}: injected`), { code })
  return {
    ...real,
    rename: (from: string, to: string) =>
      fail.renameTo.has(String(to)) || fail.renameFrom.has(String(from)) ? Promise.reject(errno('EIO')) : real.rename(from, to),
    stat: ((path: string, ...rest: unknown[]) =>
      fail.statOf.has(String(path)) ? Promise.reject(errno('EBUSY')) : (real.stat as (...a: unknown[]) => unknown)(path, ...rest)) as typeof real.stat,
    readFile: ((path: string, ...rest: unknown[]) =>
      fail.readFrom.has(String(path)) ? Promise.reject(errno('EBUSY')) : (real.readFile as (...a: unknown[]) => unknown)(path, ...rest)) as typeof real.readFile
  }
})

const { swapInto, backupDir, installFolder } = await import('../src/main/archive')
const { Store, readInstalled } = await import('../src/main/store')

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-audit-store-'))
})
afterEach(async () => {
  fail.renameTo.clear()
  fail.renameFrom.clear()
  fail.readFrom.clear()
  fail.statOf.clear()
  await rm(dir, { recursive: true, force: true })
})

async function folder(path: string, file: string, text: string): Promise<void> {
  await mkdir(path, { recursive: true })
  await writeFile(join(path, file), text)
}

describe('B1 swapInto keeps the old version when it cannot be restored', () => {
  it('names the backup folder and leaves it alone when both the swap and the rollback fail', async () => {
    const app = join(dir, 'App')
    await folder(app, 'v.txt', 'old')
    await folder(`${app}.rulz-new`, 'v.txt', 'new')
    // The staged folder cannot move into place, and neither can the backup move back.
    fail.renameTo.add(app)
    await expect(swapInto(`${app}.rulz-new`, app)).rejects.toThrow(/kept in .*App\.rulz-old/)
    expect(await readFile(join(backupDir(app), 'v.txt'), 'utf8')).toBe('old')
  })

  it('puts a leftover backup back instead of deleting it when the app folder is missing', async () => {
    const app = join(dir, 'App')
    await folder(backupDir(app), 'v.txt', 'old')
    await folder(`${app}.rulz-new`, 'v.txt', 'new')
    fail.renameTo.add(app)
    // The recovery rename fails too: the backup must survive and the error must say where it is.
    await expect(swapInto(`${app}.rulz-new`, app)).rejects.toThrow(/previous version is kept in .*App\.rulz-old/)
    expect(await readFile(join(backupDir(app), 'v.txt'), 'utf8')).toBe('old')
  })

  it('restores the leftover backup, so the old version is back in place when the new one then fails to move in', async () => {
    const app = join(dir, 'App')
    await folder(backupDir(app), 'v.txt', 'old')
    await folder(`${app}.rulz-new`, 'v.txt', 'new')
    fail.renameFrom.add(`${app}.rulz-new`)
    await expect(swapInto(`${app}.rulz-new`, app)).rejects.toThrow()
    expect(await readFile(join(app, 'v.txt'), 'utf8')).toBe('old')
  })

  it('does not delete a backup whose existence could not be checked', async () => {
    const app = join(dir, 'App')
    await folder(backupDir(app), 'v.txt', 'old')
    await folder(`${app}.rulz-new`, 'v.txt', 'new')
    fail.statOf.add(backupDir(app))
    await expect(swapInto(`${app}.rulz-new`, app)).rejects.toThrow()
    fail.statOf.clear()
    expect(await readFile(join(backupDir(app), 'v.txt'), 'utf8')).toBe('old')
  })
})

describe('R6 installFolder and a leftover backup', () => {
  it('refuses a folder whose .rulz-old backup exists', async () => {
    const app = join(dir, 'App')
    await folder(backupDir(app), 'v.txt', 'old')
    await expect(installFolder(app)).rejects.toThrow(/previous version of an app is kept in .*App\.rulz-old/)
  })
})

describe('B3 readInstalled does not mistake a locked file for no file', () => {
  const entry = { dir: 'C:\\a', launchPath: 'C:\\a\\a.exe', launchArgs: [], version: null } as never

  it('uses the .bak without touching anything when installed.json cannot be read', async () => {
    const file = join(dir, 'installed.json')
    await writeFile(file, JSON.stringify({ x: entry }))
    await writeFile(`${file}.bak`, JSON.stringify({ x: entry, y: entry }))
    fail.readFrom.add(file)
    const list = await readInstalled(file)
    expect(Object.keys(list).sort()).toEqual(['x', 'y'])
    expect(JSON.parse(await readFile(`${file}.bak`, 'utf8'))).toHaveProperty('y')
    expect(await readdir(dir)).not.toContain('installed.json.damaged')
  })

  it('throws instead of returning an empty list when there is no usable backup', async () => {
    const file = join(dir, 'installed.json')
    await writeFile(file, JSON.stringify({ x: entry }))
    fail.readFrom.add(file)
    await expect(readInstalled(file)).rejects.toThrow(/installed\.json can't be read/)
  })

  it('still returns an empty list when the file does not exist', async () => {
    expect(await readInstalled(join(dir, 'installed.json'))).toEqual({})
  })

  it('reports "can\'t be read" when the .bak is truncated too', async () => {
    const file = join(dir, 'installed.json')
    await writeFile(file, JSON.stringify({ x: entry }))
    await writeFile(`${file}.bak`, '{ "x": {')
    fail.readFrom.add(file)
    await expect(readInstalled(file)).rejects.toThrow(/installed\.json can't be read/)
  })

  it('does not write the stale .bak over the real file once the lock clears', async () => {
    const file = join(dir, 'installed.json')
    await writeFile(file, JSON.stringify({ x: entry, fresh: entry }))
    await writeFile(`${file}.bak`, JSON.stringify({ x: entry }))
    fail.readFrom.add(file)
    const store = new Store(dir)
    await store.init()
    expect(store.getInstalled('fresh')).toBeNull()
    // Still locked: the change is kept in memory and the file is not written.
    await store.setInstalled('late', entry)
    fail.readFrom.clear()
    expect(Object.keys(JSON.parse(await readFile(file, 'utf8'))).sort()).toEqual(['fresh', 'x'])
    await store.setInstalled('new', entry)
    expect(Object.keys(JSON.parse(await readFile(file, 'utf8'))).sort()).toEqual(['fresh', 'late', 'new', 'x'])
  })

  it('keeps every change when several saves wait on the re-read at once', async () => {
    const file = join(dir, 'installed.json')
    await writeFile(file, JSON.stringify({ x: entry, fresh: entry }))
    await writeFile(`${file}.bak`, JSON.stringify({ x: entry }))
    fail.readFrom.add(file)
    const store = new Store(dir)
    await store.init()
    fail.readFrom.clear()
    // The startup search saves found apps in parallel.
    await Promise.all([store.setInstalled('a', entry), store.setInstalled('b', entry), store.setInstalled('c', entry)])
    expect(Object.keys(JSON.parse(await readFile(file, 'utf8'))).sort()).toEqual(['a', 'b', 'c', 'fresh', 'x'])
  })
})

describe('R2 a locked settings.json is not replaced by defaults', () => {
  it('applies the first update on top of what the file holds once it can be read', async () => {
    const file = join(dir, 'settings.json')
    await writeFile(file, JSON.stringify({ theme: 'dark', closeOnLaunch: true }))
    fail.readFrom.add(file)
    const store = new Store(dir)
    await store.init()
    await store.updateSettings({ liveReadme: false })
    fail.readFrom.clear()
    expect(JSON.parse(await readFile(file, 'utf8'))).toEqual({ theme: 'dark', closeOnLaunch: true })
    await store.updateSettings({ checkUpdatesOnStart: false })
    const saved = JSON.parse(await readFile(file, 'utf8'))
    expect(saved).toMatchObject({ theme: 'dark', closeOnLaunch: true, liveReadme: false, checkUpdatesOnStart: false })
  })
})

describe('B5 Store.init keeps a settings.json it cannot parse', () => {
  it('moves it to settings.json.damaged before using defaults', async () => {
    await writeFile(join(dir, 'settings.json'), '{ "closeOnLaunch": tru')
    const store = new Store(dir)
    await store.init()
    expect(store.getSettings().closeOnLaunch).toBe(false)
    expect(await readFile(join(dir, 'settings.json.damaged'), 'utf8')).toBe('{ "closeOnLaunch": tru')
  })

  it('stays silent when settings.json is missing', async () => {
    await new Store(dir).init()
    expect(await readdir(dir)).not.toContain('settings.json.damaged')
  })
})

describe('R3 locked-file recovery is serialised', () => {
  const info = (version: string) => ({ dir: 'C:\\a', launchPath: 'C:\\a\\a.exe', launchArgs: [], version }) as never
  const json = async (file: string) => JSON.parse(await readFile(file, 'utf8'))

  it('keeps both patches of two parallel updateSettings calls', async () => {
    const file = join(dir, 'settings.json')
    await writeFile(file, JSON.stringify({ theme: 'dark' }))
    fail.readFrom.add(file)
    const store = new Store(dir)
    await store.init()
    fail.readFrom.clear()
    await Promise.all([store.updateSettings({ liveReadme: false }), store.updateSettings({ closeOnLaunch: true })])
    store.dispose()
    expect(await json(file)).toMatchObject({ theme: 'dark', liveReadme: false, closeOnLaunch: true })
  })

  it('ends with the later of two parallel setInstalled calls for one app', async () => {
    const file = join(dir, 'installed.json')
    await writeFile(file, JSON.stringify({ x: info('1') }))
    await writeFile(`${file}.bak`, JSON.stringify({}))
    fail.readFrom.add(file)
    const store = new Store(dir)
    await store.init()
    fail.readFrom.clear()
    await Promise.all([store.setInstalled('a', info('v1')), store.setInstalled('a', info('v2'))])
    store.dispose()
    const saved = await json(file)
    expect(saved.a.version).toBe('v2')
    expect(saved.x.version).toBe('1')
  })

  it('keeps other apps launch options when setLaunchOption is called while settings.json was locked', async () => {
    const file = join(dir, 'settings.json')
    await writeFile(file, JSON.stringify({ launchOptions: { other: ['x'] } }))
    fail.readFrom.add(file)
    const store = new Store(dir)
    await store.init()
    fail.readFrom.clear()
    await store.setLaunchOption('app', 'y', true)
    store.dispose()
    expect((await json(file)).launchOptions).toEqual({ other: ['x'], app: ['y'] })
  })

  it('saves a locked change by itself once the lock clears, without another change', async () => {
    const file = join(dir, 'settings.json')
    await writeFile(file, JSON.stringify({ theme: 'dark' }))
    fail.readFrom.add(file)
    const store = new Store(dir, 20)
    await store.init()
    await store.updateSettings({ liveReadme: false })
    fail.readFrom.clear()
    await vi.waitFor(async () => expect(await json(file)).toMatchObject({ theme: 'dark', liveReadme: false }), { timeout: 3000, interval: 25 })
    store.dispose()
  })

  it('keeps retrying when the file reads again but the first write fails', async () => {
    const file = join(dir, 'settings.json')
    await writeFile(file, JSON.stringify({ theme: 'dark' }))
    fail.readFrom.add(file)
    const store = new Store(dir, 20)
    await store.init()
    await store.updateSettings({ liveReadme: false })
    fail.readFrom.clear()
    // Readable again, but saving it fails for a while (the timer's tries in this window all fail).
    fail.renameTo.add(file)
    await new Promise((r) => setTimeout(r, 150))
    fail.renameTo.clear()
    await vi.waitFor(async () => expect(await json(file)).toMatchObject({ theme: 'dark', liveReadme: false }), { timeout: 3000, interval: 25 })
    store.dispose()
  })

  it('installFolder refuses when the backup folder cannot be checked', async () => {
    const app = join(dir, 'App')
    await folder(backupDir(app), 'v.txt', 'old')
    fail.statOf.add(backupDir(app))
    await expect(installFolder(app)).rejects.toThrow()
  })
})
