import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS } from '@shared/defaults'
import { SettingsStore } from '../src/main/settings'
import { JsonReadError, readJson, writeJsonAtomic } from '../src/main/util/jsonFile'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-json-'))
afterAll(() => rmSync(base, { recursive: true, force: true }))
let n = 0
const dir = (): string => {
  const d = join(base, `d${++n}`)
  mkdirSync(d)
  return d
}

describe('saved JSON files', () => {
  it('reads a file with a byte-order mark', async () => {
    const file = join(dir(), 'a.json')
    writeFileSync(file, '﻿{"apiKey":"k"}', 'utf8')
    expect(await readJson(file, {})).toEqual({ apiKey: 'k' })
  })

  it('uses the fallback only for a missing file', async () => {
    expect(await readJson(join(dir(), 'none.json'), { x: 1 })).toEqual({ x: 1 })
    // A folder where the file should be is not "missing": it must not be treated as empty.
    const d = dir()
    mkdirSync(join(d, 'b.json'))
    await expect(readJson(join(d, 'b.json'), {})).rejects.toBeInstanceOf(JsonReadError)
  })

  it('moves a damaged file aside so the next save cannot destroy it', async () => {
    const d = dir()
    const file = join(d, 'settings.json')
    writeFileSync(file, '{"apiKey":"secret","instructions":"keep me"', 'utf8')
    const settings = new SettingsStore(d)
    await settings.init()
    expect(settings.get().apiKey).toBe(DEFAULT_SETTINGS.apiKey)
    await settings.update({ theme: 'light' })
    const aside = readdirSync(d).filter((f) => f.startsWith('settings.json.corrupt-'))
    expect(aside).toHaveLength(1)
    expect(readFileSync(join(d, aside[0]), 'utf8')).toContain('keep me')
    expect(JSON.parse(readFileSync(file, 'utf8')).theme).toBe('light')
  })

  it('writes atomically, one write after another', async () => {
    const file = join(dir(), 'c.json')
    await Promise.all([1, 2, 3].map((v) => writeJsonAtomic(file, { v })))
    expect(JSON.parse(readFileSync(file, 'utf8'))).toEqual({ v: 3 })
    expect(existsSync(`${file}.${process.pid}.tmp`)).toBe(false)
  })
})
