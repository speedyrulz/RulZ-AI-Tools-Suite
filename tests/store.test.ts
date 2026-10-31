import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises'
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
})
