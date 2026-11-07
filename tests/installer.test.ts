import { existsSync } from 'node:fs'
import { mkdtemp, mkdir, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { findFile, flattenSingleFolder, installFolder, swapInto } from '../src/main/archive'
import { findAsset, getFetch, isNewer, setFetch, type Release } from '../src/main/github'
import { lineSplitter, quoteArg, resolveProgram, runCommand, spawnArgs } from '../src/main/process'

let dir: string
afterEach(async () => {
  if (dir) await rm(dir, { recursive: true, force: true })
})

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

describe('launcher update asset', () => {
  it('matches the setup installer of a release', async () => {
    const { SETUP_ASSET } = await import('../src/main/selfUpdate')
    const release = { tag: 'v1.2.0', name: '', date: null, prerelease: false, htmlUrl: '', assets: [
      { name: 'RulZ-AI-Tools-Suite-Setup-1.2.0.exe.blockmap', size: 1, url: 'a' },
      { name: 'RulZ-AI-Tools-Suite-Setup-1.2.0.exe', size: 2, url: 'b' }
    ] }
    expect(findAsset(release, SETUP_ASSET)?.url).toBe('b')
  }, 30_000) // importing selfUpdate (and starting PowerShell, below) can take several seconds on a cold CI runner
})

describe('launcher update helper', () => {
  it('is a valid PowerShell script', async () => {
    const { execFileSync } = await import('node:child_process')
    const { HELPER } = await import('../src/main/selfUpdate')
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    const file = join(dir, 'helper.ps1')
    await writeFile(file, HELPER, 'utf8')
    const out = execFileSync(
      'powershell.exe',
      ['-NoProfile', '-Command', `$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile('${file}', [ref]$null, [ref]$e); $e.Count`],
      { encoding: 'utf8' }
    )
    expect(out.trim()).toBe('0')
  }, 30_000)

  it("won't start while an app is being installed (the setup would stop the launcher halfway)", async () => {
    const { busyReason, installSelfUpdate } = await import('../src/main/selfUpdate')
    expect(busyReason([])).toBeNull()
    expect(busyReason(['RulZ LLM Suite', 'RulzTrainer'])).toMatch(/^RulZ LLM Suite, RulzTrainer are still being installed/)
    const original = getFetch()
    let fetched = false
    setFetch(async () => {
      fetched = true
      return new Response('')
    })
    try {
      const u = { version: '9.9.9', tag: 'v9.9.9', size: 1, url: 'https://x/setup.exe', notesUrl: '' }
      await expect(installSelfUpdate(u, () => undefined, () => ['RulZ LLM Suite'])).rejects.toThrow(/is still being installed/)
      expect(fetched).toBe(false)
    } finally {
      setFetch(original)
    }
  })
})

describe('install folder', () => {
  it('tidies the path: trailing backslash, forward slashes and Explorer quotes', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    const target = join(dir, 'Grid')
    expect(await installFolder(`${target}\\`)).toBe(target)
    expect(await installFolder(target.replace(/\\/g, '/'))).toBe(target)
    expect(await installFolder(`"${target}"`)).toBe(target)
    expect(await installFolder(`  ${target}  `)).toBe(target)
  })

  it('accepts an empty folder', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    expect(await installFolder(dir)).toBe(dir)
  })

  it('refuses a folder with files in it', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    await writeFile(join(dir, 'mine.txt'), '')
    await expect(installFolder(dir)).rejects.toThrow(/empty folder/)
  })

  it('refuses a file instead of treating it as an empty folder, and leaves it alone', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    const file = join(dir, 'notes.txt')
    await writeFile(file, 'important')
    await expect(installFolder(file)).rejects.toThrow(/is a file/)
    expect(existsSync(file)).toBe(true)
  })

  it('refuses relative paths and whole drives', async () => {
    await expect(installFolder('Apps\\Grid')).rejects.toThrow(/full folder path/)
    await expect(installFolder('\\Apps')).rejects.toThrow(/full folder path/)
    await expect(installFolder('C:\\')).rejects.toThrow(/whole drive/)
  })
})

describe('program output', () => {
  it('keeps characters whose bytes arrive in two chunks whole', () => {
    const lines: string[] = []
    const s = lineSplitter((l) => lines.push(l))
    s.write(Buffer.from([0x63, 0x61, 0x66, 0xc3])) // "caf" + the first byte of é
    s.write(Buffer.from([0xa9, 0x0a]))
    expect(lines).toEqual(['café'])
  })

  it('passes on the last line even without a line break', () => {
    const lines: string[] = []
    const s = lineSplitter((l) => lines.push(l))
    s.write(Buffer.from('one\r\ntwo\rthree\n\nERROR: no break'))
    expect(lines).toEqual(['one', 'two', 'three'])
    s.end()
    expect(lines).toEqual(['one', 'two', 'three', 'ERROR: no break'])
  })

  it('logs the final line of a program that ends without a line break', async () => {
    const lines: string[] = []
    const script = "process.stdout.write('first\\n'); setTimeout(() => process.stderr.write('ERROR: final message'), 50)"
    const code = await runCommand([process.execPath, '-e', script], { cwd: tmpdir(), onLine: (l) => lines.push(l) })
    expect(code).toBe(0)
    expect(lines).toEqual(['first', 'ERROR: final message'])
  }, 30_000)
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

  it('finds bare commands on PATH even when cwd has a folder of that name', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-test-'))
    await mkdir(join(dir, 'winget'))
    await writeFile(join(dir, 'install.bat'), '')
    expect(resolveProgram('winget', dir)).toBe('winget')
    expect(resolveProgram('install.bat', dir)).toBe(join(dir, 'install.bat'))
    expect(resolveProgram('.venv\\Scripts\\python.exe', dir)).toBe(join(dir, '.venv\\Scripts\\python.exe'))
  })

  it('starts programs directly', () => {
    expect(spawnArgs('C:\\x\\app.exe', ['a b'])).toEqual({ file: 'C:\\x\\app.exe', args: ['a b'], verbatim: false })
  })
})

describe('archive helpers', () => {
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
