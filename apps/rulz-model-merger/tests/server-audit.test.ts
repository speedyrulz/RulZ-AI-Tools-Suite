import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { promises as fsp } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_SETTINGS, type OwnServerSettings } from '@shared/types'
import { OwnServer } from '../src/main/server/ownServer'
import { buildStartScript } from '../src/main/server/scripts'
import type { WslResult, WslRunner } from '../src/main/server/wsl'
import { SettingsStore } from '../src/main/settings'

const PORT = 8288
const MARK = 'RULZ_BASE=/b/comfy\nRULZ_PID=42\nRULZ_START=1000\n'
const OUR_ARGV = ['main.py', '--base-directory', '/b/comfy', '--port', String(PORT)]

class FakeChild extends EventEmitter {
  stdout = new EventEmitter()
  stderr = new EventEmitter()
  kill(): boolean {
    this.emit('exit', null)
    this.emit('close', null)
    return true
  }
  say(text: string): void {
    this.stdout.emit('data', Buffer.from(text))
  }
}

function harness(opts: { aliveCode?: () => number } = {}) {
  const runs: string[] = []
  const children: FakeChild[] = []
  const live = { settings: { ...DEFAULT_SETTINGS.ownServer, stopTimeoutSec: 1 } as OwnServerSettings, alive: false }
  const wsl: WslRunner = {
    run: async (_d, script): Promise<WslResult> => {
      runs.push(script)
      if (script.includes('echo RULZ_ALIVE')) {
        const code = opts.aliveCode?.() ?? 0
        return { code, stdout: code === 0 && live.alive ? 'RULZ_ALIVE\n' : '', stderr: '' }
      }
      if (script.includes('kill -s')) live.alive = false
      return { code: 0, stdout: '', stderr: '' }
    },
    spawn: () => {
      const c = new FakeChild()
      children.push(c)
      return c as unknown as ChildProcess
    }
  }
  const server = new OwnServer({ wsl, settings: () => live.settings, probe: async () => ({ argv: OUR_ARGV }) })
  return { server, runs, children, live }
}

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('B1 stop during preflight', () => {
  it('a stop() after the ticket was taken cancels the start; nothing is spawned', async () => {
    const h = harness()
    const ticket = h.server.beginStart() // what the IPC entry does before the checks
    await vi.advanceTimersByTimeAsync(50) // preflight running
    const sp = h.server.stop() // Stop or quit while it runs
    const st = await h.server.start(PORT, ticket) // preflight ends, start is called
    await sp
    expect(h.children).toHaveLength(0)
    expect(st.status).toBe('stopped')
    expect(h.server.state().status).toBe('stopped')
  })
})

describe('B2 settings changed during preflight', () => {
  it('refuses to launch settings that were not the ones checked', async () => {
    const h = harness()
    const checked = h.live.settings
    h.live.settings = { ...checked, extraArgs: '--something-new' }
    const p = h.server.start(PORT, h.server.beginStart(), checked)
    const done = p.then(
      () => 'ok',
      (e: Error) => e.message
    )
    await vi.advanceTimersByTimeAsync(100)
    // doStart throws inside the queue; start() reports it as an error state or rejection, but never spawns.
    const r = await done
    expect(h.children).toHaveLength(0)
    expect(r === 'ok' ? h.server.state().error : r).toContain('settings changed while they were checked')
  })
})

describe('B3 own yaml name', () => {
  it('does not write ComfyUI\'s extra_model_paths.yaml', () => {
    const s = buildStartScript({ ...DEFAULT_SETTINGS.ownServer, loadCustomNodes: true }, PORT)
    expect(s).toContain('"$RULZ_BASE/rulz_extra_model_paths.yaml"')
    expect(s).not.toMatch(/\$RULZ_BASE\/extra_model_paths\.yaml/)
  })
})

describe('B5 WSL failure is not "gone"', () => {
  it('a failing alive check keeps the pid file and reports an error on stop', async () => {
    let fail = false
    const h = harness({ aliveCode: () => (fail ? 1 : 0) })
    const p = h.server.start(PORT)
    await vi.advanceTimersByTimeAsync(0)
    h.live.alive = true
    h.children[0].say(MARK)
    await vi.advanceTimersByTimeAsync(600)
    expect((await p).status).toBe('running')
    fail = true
    const sp = h.server.stop()
    await vi.advanceTimersByTimeAsync(10000)
    const st = await sp
    expect(st.status).toBe('error')
    expect(h.runs.filter((s) => s.includes('rm -f'))).toHaveLength(0)
  })
})

describe('B6 unreadable settings file is not overwritten', () => {
  it('keeps the file when it could not be read, and still updates memory', async () => {
    vi.useRealTimers()
    const dir = await fsp.mkdtemp(join(tmpdir(), 'rulz-settings-'))
    const file = join(dir, 'settings.json')
    const original = JSON.stringify({ theme: 'light' })
    await fsp.writeFile(file, original, 'utf8')
    const spy = vi.spyOn(fsp, 'readFile').mockRejectedValueOnce(Object.assign(new Error('denied'), { code: 'EACCES' }))
    const store = new SettingsStore(file)
    await store.load()
    spy.mockRestore()
    const next = await store.set({ theme: 'dark' })
    expect(next.theme).toBe('dark')
    expect(await fsp.readFile(file, 'utf8')).toBe(original)
    await fsp.rm(dir, { recursive: true, force: true })
  })

  it('a missing file still gets written', async () => {
    vi.useRealTimers()
    const dir = await fsp.mkdtemp(join(tmpdir(), 'rulz-settings-'))
    const file = join(dir, 'settings.json')
    const store = new SettingsStore(file)
    await store.load()
    await store.set({ theme: 'light' })
    expect(JSON.parse(await fsp.readFile(file, 'utf8')).theme).toBe('light')
    await fsp.rm(dir, { recursive: true, force: true })
  })
})
