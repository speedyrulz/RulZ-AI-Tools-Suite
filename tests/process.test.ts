import { spawn } from 'node:child_process'
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { powershell, programsInside, programsRunningFrom, quoteCmdArg, runCommand, spawnArgs } from '../src/main/process'

describe('programsInside', () => {
  const dir = 'C:\\Users\\me\\AppData\\Local\\Programs\\RulZ AI Tools\\RulZ Huggingface Manager'
  const listing = [
    'C:\\Windows\\System32\\svchost.exe',
    '',
    'c:\\users\\me\\appdata\\local\\programs\\rulz ai tools\\rulz huggingface manager\\.venv\\Scripts\\python.exe',
    `${dir}\\.venv\\Scripts\\python.exe`,
    `${dir} 2\\other.exe`,
    'C:\\Users\\me\\AppData\\Local\\Programs\\RulZ AI Tools\\RulzTrainer\\venv\\Scripts\\python.exe'
  ].join('\r\n')

  it('finds programs inside the folder, ignoring case and similar folder names', () => {
    expect(programsInside(listing, dir)).toEqual(['python.exe'])
    expect(programsInside(listing, `${dir}\\`)).toEqual(['python.exe'])
  })

  it('finds nothing in an empty or unreadable listing', () => {
    expect(programsInside(null, dir)).toEqual([])
    expect(programsInside('', dir)).toEqual([])
  })
})

describe.runIf(process.platform === 'win32')('programsRunningFrom', () => {
  it('sees the test runner itself', async () => {
    const own = process.execPath
    const folder = own.slice(0, own.lastIndexOf('\\'))
    const name = own.slice(own.lastIndexOf('\\') + 1)
    expect(await programsRunningFrom(folder)).toContain(name)
  }, 30000)
})

describe('cmd.exe command lines', () => {
  it("quotes cmd.exe's special characters, which CommandLineToArgvW doesn't need quoted", () => {
    expect(quoteCmdArg('D:\\R&D\\run.bat')).toBe('"D:\\R&D\\run.bat"')
    expect(quoteCmdArg('a|b')).toBe('"a|b"')
    expect(quoteCmdArg('plain')).toBe('plain')
    expect(quoteCmdArg('C:\\with space\\x.bat')).toBe('"C:\\with space\\x.bat"')
    expect(spawnArgs('D:\\R&D\\run.bat', []).args[3]).toBe('""D:\\R&D\\run.bat""')
  })
})

describe.runIf(process.platform === 'win32')('Windows programs and folders with unusual names', () => {
  let dir = ''
  afterEach(async () => {
    if (dir) await rm(dir, { recursive: true, force: true })
  })

  it('runs a batch file from a folder with & , ; or = in its name', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-cmd-'))
    // Unquoted, cmd.exe would run "...\R" and then "D\setup.bat", or stop at the separator.
    for (const name of ['R&D', 'a,b', 'a;b', 'a=b']) {
      const folder = join(dir, name)
      await mkdir(folder)
      await writeFile(join(folder, 'setup.bat'), '@echo setup ran\r\n')
      const lines: string[] = []
      const code = await runCommand(['setup.bat'], { cwd: folder, onLine: (l) => lines.push(l) })
      expect(lines, name).toEqual(['setup ran'])
      expect(code, name).toBe(0)
    }
  }, 30_000)

  it('reads PowerShell output as UTF-8, so "José" stays "José"', async () => {
    expect(await powershell("'C:\\Users\\José Ünï'")).toBe('C:\\Users\\José Ünï')
  }, 30_000)

  it('sees a program running from a folder with non-ASCII letters', async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-enc-'))
    const folder = join(dir, 'Jösé Ünï')
    await mkdir(folder)
    await copyFile(join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'PING.EXE'), join(folder, 'rulzping.exe'))
    const child = spawn(join(folder, 'rulzping.exe'), ['-n', '30', '127.0.0.1'], { stdio: 'ignore', windowsHide: true })
    try {
      await new Promise((r) => setTimeout(r, 500))
      expect(await programsRunningFrom(folder)).toEqual(['rulzping.exe'])
    } finally {
      child.kill()
      await new Promise((r) => child.once('exit', r))
    }
  }, 30_000)
})
