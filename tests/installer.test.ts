import { mkdtemp, mkdir, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { findFile, flattenSingleFolder, swapInto } from '../src/main/archive'
import { findAsset, isNewer, type Release } from '../src/main/github'
import { quoteArg, spawnArgs } from '../src/main/process'

describe('isNewer', () => {
  it('compares version tags numerically', () => {
    expect(isNewer('v1.10.0', 'v1.9.2')).toBe(true)
    expect(isNewer('v1.1.0', '1.1.0')).toBe(false)
    expect(isNewer('v0.2.0', 'v0.2')).toBe(false)
    expect(isNewer('v1.0.0', 'v1.0.1')).toBe(false)
  })

  it('never reports an update for an unknown installed version', () => {
    expect(isNewer('v2.0.0', null)).toBe(false)
  })
})

describe('findAsset', () => {
  const release: Release = {
    tag: 'v1.1.0',
    name: 'v1.1.0',
    date: null,
    prerelease: false,
    htmlUrl: '',
    assets: [
      { name: 'RulZ-LLM-Suite-Setup-0.2.0.exe.blockmap', size: 1, url: 'a' },
      { name: 'RulZ-LLM-Suite-Setup-0.2.0.exe', size: 2, url: 'b' }
    ]
  }
  it('matches the whole asset name', () => {
    expect(findAsset(release, '^RulZ-LLM-Suite-Setup-[\\d.]+\\.exe$')?.url).toBe('b')
  })
})

describe('command lines', () => {
  it('quotes arguments the way Windows programs parse them', () => {
    expect(quoteArg('plain')).toBe('plain')
    expect(quoteArg('C:\\Program Files\\x')).toBe('"C:\\Program Files\\x"')
    expect(quoteArg('C:\\dir with space\\')).toBe('"C:\\dir with space\\\\"')
    expect(quoteArg('say "hi"')).toBe('"say \\"hi\\""')
    expect(quoteArg('')).toBe('""')
  })

  it('runs batch files through cmd.exe with the whole line quoted', () => {
    const r = spawnArgs('C:\\Apps\\Rulz Trainer\\run_webui.bat', ['7870'])
    expect(r.verbatim).toBe(true)
    expect(r.args.slice(0, 3)).toEqual(['/d', '/s', '/c'])
    expect(r.args[3]).toBe('""C:\\Apps\\Rulz Trainer\\run_webui.bat" 7870"')
  })

  it('starts programs directly', () => {
    expect(spawnArgs('C:\\x\\app.exe', ['a b'])).toEqual({ file: 'C:\\x\\app.exe', args: ['a b'], verbatim: false })
  })
})

describe('archive helpers', () => {
  let dir: string
  afterEach(async () => {
    if (dir) await rm(dir, { recursive: true, force: true })
  })

  it('moves the contents of a single top-level folder up', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    await mkdir(join(dir, 'App-v1.0.0', '_internal'), { recursive: true })
    await writeFile(join(dir, 'App-v1.0.0', 'App.exe'), '')
    await flattenSingleFolder(dir)
    expect((await readdir(dir)).sort()).toEqual(['App.exe', '_internal'])
  })

  it('leaves a folder with several entries alone', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    await mkdir(join(dir, 'a'))
    await writeFile(join(dir, 'b.txt'), '')
    await flattenSingleFolder(dir)
    expect((await readdir(dir)).sort()).toEqual(['a', 'b.txt'])
  })

  it('finds a program in a subfolder', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    await mkdir(join(dir, 'x', 'bin'), { recursive: true })
    await writeFile(join(dir, 'x', 'bin', 'Tool.exe'), '')
    expect(await findFile(dir, 'tool.exe')).toBe(join(dir, 'x', 'bin', 'Tool.exe'))
    expect(await findFile(dir, 'missing.exe')).toBeNull()
  })

  it('swaps a staged folder into place and drops the old one', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    await mkdir(join(dir, 'app'))
    await writeFile(join(dir, 'app', 'old.txt'), '')
    await mkdir(join(dir, 'app.new'))
    await writeFile(join(dir, 'app.new', 'new.txt'), '')
    await swapInto(join(dir, 'app.new'), join(dir, 'app'))
    expect(await readdir(dir)).toEqual(['app'])
    expect(await readdir(join(dir, 'app'))).toEqual(['new.txt'])
  })
})
