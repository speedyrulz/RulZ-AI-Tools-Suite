import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { afterAll, describe, expect, it } from 'vitest'
import { BackgroundCommands, cleanOutput, commandEnv, commandHints, commandReport, headTail, runCommand } from '../src/main/agent/commands'

const cwd = mkdtempSync(join(tmpdir(), 'rulz-agent-cmd-'))
// Killed processes can hold the folder for a moment.
afterAll(() => rmSync(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 }))

/** True while a process with this marker in its command line exists. */
function processWithMarker(marker: string): boolean {
  const out = execFileSync(
    'powershell.exe',
    ['-NoProfile', '-Command', `@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*${marker}*' -and $_.Name -ne 'powershell.exe' }).Count`],
    { encoding: 'utf8' }
  )
  return Number(out.trim()) > 0
}

describe.runIf(process.platform === 'win32')('running commands', () => {
  it('returns output and the exit code', async () => {
    const r = await runCommand('Write-Output "hello"; cmd /c exit 3', { cwd, timeoutMs: 30_000 })
    expect(r.output).toContain('hello')
    expect(r.exitCode).toBe(3)
    expect(commandReport(r, 30_000)).toMatchObject({ isError: true })
    const ok = await runCommand('Write-Output "fine"', { cwd, timeoutMs: 30_000 })
    expect(ok.exitCode).toBe(0)
    expect(commandReport(ok, 30_000).text).toMatch(/^Exit code: 0\nfine/)
  })

  it('reports PowerShell errors as failures', async () => {
    const r = await runCommand('Get-Item does-not-exist-anywhere', { cwd, timeoutMs: 30_000 })
    expect(r.exitCode).toBe(1)
    expect(r.output).toMatch(/does-not-exist-anywhere/)
  })

  it('handles quotes and non-English text without breaking', async () => {
    const r = await runCommand(`Write-Output 'it''s "quoted" – ünïcødé'`, { cwd, timeoutMs: 30_000 })
    expect(r.output.trim()).toBe(`it's "quoted" – ünïcødé`)
  })

  it('streams output as it arrives', async () => {
    const chunks: string[] = []
    await runCommand('1..3 | ForEach-Object { Write-Output "line $_"; Start-Sleep -Milliseconds 150 }', { cwd, timeoutMs: 30_000, onOutput: (c) => chunks.push(c) })
    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.join('')).toMatch(/line 1[\s\S]*line 3/)
  })

  it('kills the whole process tree at the time limit', async () => {
    const marker = `rulz-timeout-${Date.now()}`
    const started = Date.now()
    const r = await runCommand(`ping -n 60 127.0.0.1 > $null; Write-Output ${marker}`, { cwd, timeoutMs: 1500 })
    expect(r.timedOut).toBe(true)
    expect(Date.now() - started).toBeLessThan(15_000)
    expect(commandReport(r, 1500).text).toMatch(/stopped after 2 seconds|stopped after 1 seconds|start_background_command/)
    await new Promise((res) => setTimeout(res, 500))
    expect(processWithMarker('ping -n 60 127.0.0.1')).toBe(false)
  }, 30_000)

  it('stops when the task is stopped', async () => {
    const ctrl = new AbortController()
    setTimeout(() => ctrl.abort(), 800)
    const r = await runCommand('Start-Sleep -Seconds 60', { cwd, timeoutMs: 60_000, signal: ctrl.signal })
    expect(r.stopped).toBe(true)
    expect(commandReport(r, 60_000).text).toMatch(/owner stopped/)
  }, 30_000)

  it('runs, checks and stops background commands', async () => {
    const bg = new BackgroundCommands()
    const id = bg.run('Write-Output "server ready"; Start-Sleep -Seconds 60', cwd)
    expect(id).toBe('bg1')
    const first = await bg.check(id, 3000)
    expect(first.text).toMatch(/Still running/)
    expect(first.text).toMatch(/server ready/)
    expect((await bg.check(id, 0)).text).toMatch(/nothing new/)
    expect(bg.running()).toEqual([{ id, command: 'Write-Output "server ready"; Start-Sleep -Seconds 60' }])
    expect((await bg.stop(id)).isError).toBe(false)
    expect((await bg.check(id, 0)).isError).toBe(true)
  }, 30_000)
})

describe('hints for known stumbling blocks', () => {
  it('explains skipped npm install scripts', () => {
    const out = 'added 70 packages\nnpm warn allow-scripts 1 package has install scripts not yet covered by allowScripts:\nnpm warn allow-scripts   electron@30.5.1 (postinstall: node install.js)\n'
    expect(commandHints(out)).toMatch(/npm install-scripts approve electron; npm rebuild electron/)
    expect(commandHints('Error: Electron failed to install correctly, please delete node_modules/electron')).toMatch(/npm rebuild electron/)
    expect(commandHints('all good')).toBe('')
  })

  it('points at a running program when files are locked', () => {
    expect(commandHints('npm error EBUSY: resource busy or locked', [{ id: 'bg1', command: 'npm start' }])).toMatch(/bg1: npm start/)
    expect(commandHints('npm error EBUSY: resource busy or locked', [])).toBe('')
  })
})

describe('output helpers', () => {
  it('keeps the head and the tail of long output', () => {
    const text = 'a'.repeat(30_000) + 'END'
    const out = headTail(text, 1000)
    expect(out.length).toBeLessThan(1100)
    expect(out.endsWith('END')).toBe(true)
    expect(out).toMatch(/characters omitted/)
  })

  it('strips colours and collapses progress lines', () => {
    expect(cleanOutput('\x1b[32mgreen\x1b[0m')).toBe('green')
    expect(cleanOutput('10%\r50%\r100%\ndone')).toBe('100%\ndone')
  })

  it('removes Electron variables and adds quiet settings for child commands', () => {
    const env = commandEnv({ PATH: 'C:\\x', ELECTRON_RUN_AS_NODE: '1', ELECTRON_RENDERER_URL: 'http://x', NODE_OPTIONS: '--inspect' })
    expect(env.ELECTRON_RUN_AS_NODE).toBeUndefined()
    expect(env.ELECTRON_RENDERER_URL).toBeUndefined()
    expect(env.NODE_OPTIONS).toBeUndefined()
    expect(env.npm_config_yes).toBe('true')
    expect(env.PATH!.startsWith('C:\\x')).toBe(true)
  })
})
