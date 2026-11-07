import { existsSync, mkdirSync, mkdtempSync, promises as fsp, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { corruptName, readJson, writeJsonAtomic } from '../src/main/util/jsonFile'
import { SettingsStore } from '../src/main/settings/store'

const tmp = (): string => mkdtempSync(join(tmpdir(), 'rulz-json-'))
const errno = (code: string): NodeJS.ErrnoException => Object.assign(new Error(code), { code })

afterEach(() => {
  vi.restoreAllMocks()
})

describe('readJson', () => {
  it('gives the fallback for a missing file', async () => {
    expect(await readJson(join(tmp(), 'nope.json'), { a: 1 })).toEqual({ a: 1 })
  })

  it('reads a file, also one saved with a byte-order mark', async () => {
    const dir = tmp()
    writeFileSync(join(dir, 'x.json'), '\uFEFF{"a":2}')
    expect(await readJson(join(dir, 'x.json'), {})).toEqual({ a: 2 })
  })

  it('moves a file that does not parse aside instead of letting it be overwritten', async () => {
    const dir = tmp()
    const path = join(dir, 'settings.json')
    writeFileSync(path, '{"a": 1,,')
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(await readJson(path, { d: true })).toEqual({ d: true })
    expect(existsSync(path)).toBe(false)
    const kept = readdirSync(dir).filter((n) => n.startsWith('settings.json.corrupt-'))
    expect(kept).toHaveLength(1)
    expect(readFileSync(join(dir, kept[0]), 'utf8')).toBe('{"a": 1,,')
  })

  it('names the kept copy after the time', () => {
    expect(corruptName('C:\\x\\a.json', new Date('2026-11-07T16:19:00.123Z'))).toBe('C:\\x\\a.json.corrupt-20261107T161900')
  })

  it('throws for a file that exists but cannot be read', async () => {
    const dir = tmp()
    mkdirSync(join(dir, 'locked.json'))
    await expect(readJson(join(dir, 'locked.json'), {})).rejects.toThrow()
  })

  it('retries while Windows has the file locked', async () => {
    const dir = tmp()
    const path = join(dir, 'busy.json')
    writeFileSync(path, '{"ok":true}')
    const real = fsp.readFile.bind(fsp)
    let calls = 0
    vi.spyOn(fsp, 'readFile').mockImplementation(((...args: Parameters<typeof fsp.readFile>) => {
      if (++calls <= 2) return Promise.reject(errno('EBUSY'))
      return real(...args)
    }) as typeof fsp.readFile)
    expect(await readJson(path, {})).toEqual({ ok: true })
    expect(calls).toBe(3)
  })

  it('gives up on a lock that does not go away', async () => {
    vi.spyOn(fsp, 'readFile').mockRejectedValue(errno('EBUSY'))
    await expect(readJson(join(tmp(), 'busy.json'), {})).rejects.toThrow('EBUSY')
  })
})

describe('writeJsonAtomic', () => {
  it('writes in order and leaves no temp files', async () => {
    const dir = tmp()
    const path = join(dir, 'sub', 'a.json')
    await Promise.all([writeJsonAtomic(path, { n: 1 }), writeJsonAtomic(path, { n: 2 }), writeJsonAtomic(path, { n: 3 })])
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ n: 3 })
    expect(readdirSync(join(dir, 'sub'))).toEqual(['a.json'])
  })

  it('rejects only to the caller when a write fails', async () => {
    const dir = tmp()
    writeFileSync(join(dir, 'file'), 'x')
    const unhandled = vi.fn()
    process.on('unhandledRejection', unhandled)
    try {
      // The parent "folder" is a file, so the write must fail.
      const path = join(dir, 'file', 'a.json')
      await expect(writeJsonAtomic(path, {})).rejects.toThrow()
      await new Promise((r) => setTimeout(r, 20))
      expect(unhandled).not.toHaveBeenCalled()
    } finally {
      process.off('unhandledRejection', unhandled)
    }
  })
})

describe('SettingsStore with a file it cannot read', () => {
  it('starts on defaults but never overwrites that file', async () => {
    const dir = tmp()
    // A folder where settings.json should be: it exists, but can't be read as a file.
    mkdirSync(join(dir, 'settings.json'))
    writeFileSync(join(dir, 'presets.json'), '[]')
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const store = new SettingsStore(dir)
    await store.init()
    expect(store.get().modelsDir).toBeTypeOf('string')
    await expect(store.update({ modelsDir: 'D:\\Models' })).rejects.toThrow(/wasn't overwritten/)
    expect(readdirSync(join(dir, 'settings.json'))).toEqual([])
    // Files that were read normally still save.
    await store.deletePreset('none')
    expect(JSON.parse(readFileSync(join(dir, 'presets.json'), 'utf8'))).toEqual([])
  })
})
