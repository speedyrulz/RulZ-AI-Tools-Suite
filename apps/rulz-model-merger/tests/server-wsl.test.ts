import { EventEmitter } from 'node:events'
import { existsSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

interface Call {
  file: string
  args: string[]
  opts: { windowsHide?: boolean; env?: Record<string, string> }
  script: string
  child: EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => boolean }
  hostPath: string
}
const calls: Call[] = []

vi.mock('node:child_process', () => ({
  spawn: (file: string, args: string[], opts: Call['opts']) => {
    const hostPath = args[args.length - 1]
    // the script file must exist (with LF endings) while wsl.exe starts
    const winFile = hostPath.replace(/^\/mnt\/([a-z])\//, (_m, d: string) => `${d.toUpperCase()}:/`)
    const child = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), kill: () => true })
    calls.push({ file, args, opts, script: readFileSync(winFile, 'utf8'), child, hostPath: winFile })
    return child
  }
}))

import { cleanupOldTempScripts, cleanWindowsPath, createWslRunner } from '../src/main/server/wsl'

beforeEach(() => {
  calls.length = 0
})

describe('createWslRunner', () => {
  it('runs a script through a temp file with LF endings and strips NUL bytes', async () => {
    const wsl = createWslRunner()
    const p = wsl.run('Ubuntu-24.04', 'echo a\r\necho b\r\n')
    const c = calls[0]
    expect(c.file).toBe('wsl.exe')
    expect(c.args.slice(0, 4)).toEqual(['-d', 'Ubuntu-24.04', '--exec', 'bash'])
    expect(c.args[4]).toMatch(/^\/mnt\/[a-z]\/.*rulz-merger-.*\.sh$/)
    expect(c.script).toBe('echo a\necho b\n')
    expect(c.opts.windowsHide).toBe(true)
    c.child.stdout.emit('data', Buffer.from('h\0i\0\n'))
    c.child.stderr.emit('data', Buffer.from('w\0'))
    c.child.emit('close', 0)
    expect(await p).toEqual({ code: 0, stdout: 'hi\n', stderr: 'w' })
    await new Promise((r) => setTimeout(r, 100))
    expect(existsSync(c.hostPath)).toBe(false)
  })

  it('reports code null when wsl.exe cannot start', async () => {
    const p = createWslRunner().run('x', 'true')
    calls[0].child.emit('error', new Error('spawn wsl.exe ENOENT'))
    const r = await p
    expect(r.code).toBeNull()
    expect(r.stderr).toContain('ENOENT')
  })

  it('reports code null on timeout and kills the child', async () => {
    const p = createWslRunner().run('x', 'sleep 100', { timeoutMs: 20 })
    const kill = vi.spyOn(calls[0].child, 'kill')
    const r = await p
    expect(r.code).toBeNull()
    expect(r.stderr).toContain('Timed out')
    expect(kill).toHaveBeenCalled()
  })

  it('spawn keeps the temp file until the process closes', async () => {
    const child = createWslRunner().spawn('Ubuntu-24.04', 'exec python main.py\n')
    const c = calls[0]
    expect(child).toBe(c.child)
    expect(existsSync(c.hostPath)).toBe(true)
    c.child.emit('close', 0)
    await new Promise((r) => setTimeout(r, 100))
    expect(existsSync(c.hostPath)).toBe(false)
  })
})

describe('cleanWindowsPath', () => {
  it('drops quotes and broken entries', () => {
    expect(cleanWindowsPath('C:\\a;"C:\\b";C: \\;;D:\\c')).toBe('C:\\a;C:\\b;D:\\c')
  })
})

describe('cleanupOldTempScripts', () => {
  it('deletes only old rulz-merger scripts', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-clean-'))
    try {
      const old = join(dir, 'rulz-merger-1-aa.sh')
      const fresh = join(dir, 'rulz-merger-2-bb.sh')
      const other = join(dir, 'other-old.sh')
      for (const f of [old, fresh, other]) writeFileSync(f, 'x')
      const twoDaysAgo = new Date(Date.now() - 2 * 86400000)
      utimesSync(old, twoDaysAgo, twoDaysAgo)
      utimesSync(other, twoDaysAgo, twoDaysAgo)
      expect(cleanupOldTempScripts(dir)).toBe(1)
      await new Promise((r) => setTimeout(r, 100))
      expect(existsSync(old)).toBe(false)
      expect(existsSync(fresh)).toBe(true)
      expect(existsSync(other)).toBe(true)
      expect(cleanupOldTempScripts(join(dir, 'missing'))).toBe(0)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
