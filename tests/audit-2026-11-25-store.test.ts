import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Failures are injected through these: a rename whose target is listed fails, a read of a listed file fails.
const fail = vi.hoisted(() => ({ renameTo: new Set<string>(), readFrom: new Set<string>() }))

vi.mock('node:fs/promises', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs/promises')>()
  const errno = (code: string) => Object.assign(new Error(`${code}: injected`), { code })
  return {
    ...real,
    rename: (from: string, to: string) => (fail.renameTo.has(String(to)) ? Promise.reject(errno('EIO')) : real.rename(from, to)),
    readFile: ((path: string, ...rest: unknown[]) =>
      fail.readFrom.has(String(path)) ? Promise.reject(errno('EBUSY')) : (real.readFile as (...a: unknown[]) => unknown)(path, ...rest)) as typeof real.readFile
  }
})

const { swapInto, backupDir } = await import('../src/main/archive')
const { Store, readInstalled } = await import('../src/main/store')

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-audit-store-'))
})
afterEach(async () => {
  fail.renameTo.clear()
  fail.readFrom.clear()
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
    fail.renameTo.clear()
    await swapInto(`${app}.rulz-new`, app)
    expect(await readFile(join(app, 'v.txt'), 'utf8')).toBe('new')
  })
})

describe('B3 readInstalled does not mistake a locked file for no file', () => {
  const entry = { dir: 'C:\\a', launchPath: 'C:\\a\\a.exe', launchArgs: [], version: null }

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
