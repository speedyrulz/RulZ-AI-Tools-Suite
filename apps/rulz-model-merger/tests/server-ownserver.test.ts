import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_SETTINGS, type OwnServerSettings, type ServerState } from '@shared/types'
import { OwnServer } from '../src/main/server/ownServer'
import type { WslResult, WslRunner } from '../src/main/server/wsl'

const PORT = 8288
const MARK = 'RULZ_BASE=/b/comfy\nRULZ_PID=42\nRULZ_START=1000\n'
const OUR_ARGV = ['main.py', '--base-directory', '/b/comfy', '--port', String(PORT), '--listen', '127.0.0.1']

class FakeChild extends EventEmitter {
  stdout = new EventEmitter()
  stderr = new EventEmitter()
  killed = false
  constructor(private readonly stubborn = false) {
    super()
  }
  kill(): boolean {
    this.killed = true
    if (!this.stubborn) this.exit(null)
    return true
  }
  exit(code: number | null = 0): void {
    this.emit('exit', code)
    this.emit('close', code)
  }
  say(text: string): void {
    this.stdout.emit('data', Buffer.from(text))
  }
}

interface Opts {
  respond?: (script: string) => string | undefined
  probe?: (n: number) => { argv: string[] } | null
  settings?: Partial<OwnServerSettings>
  readyTimeoutMs?: number
  recordWaitMs?: number
  obeysInt?: boolean
  obeysKill?: boolean
  /** The first child ignores kill(), as a wsl.exe that hangs would. */
  stubbornFirst?: boolean
}

function harness(opts: Opts = {}) {
  const runs: Array<{ distro: string; script: string }> = []
  const spawned: string[] = []
  const children: FakeChild[] = []
  const proc = { alive: false }
  const live = { settings: { ...DEFAULT_SETTINGS.ownServer, stopTimeoutSec: 1, ...opts.settings } as OwnServerSettings }
  let n = 0
  const die = (): void => {
    proc.alive = false
    children[children.length - 1]?.exit(0)
  }
  const wsl: WslRunner = {
    run: async (distro, script): Promise<WslResult> => {
      runs.push({ distro, script })
      const custom = opts.respond?.(script)
      const out = ((): string => {
        if (custom !== undefined) return custom
        if (script.includes('echo RULZ_ALIVE')) return proc.alive ? 'RULZ_ALIVE\n' : ''
        if (script.includes('kill -s INT')) {
          if (opts.obeysInt ?? true) die()
          return ''
        }
        if (script.includes('kill -s KILL')) {
          if (opts.obeysKill ?? true) die()
          return ''
        }
        return ''
      })()
      return { code: 0, stdout: out, stderr: '' }
    },
    spawn: (_d, script) => {
      spawned.push(script)
      const c = new FakeChild(opts.stubbornFirst === true && children.length === 0)
      children.push(c)
      return c as unknown as ChildProcess
    }
  }
  const server = new OwnServer({
    wsl,
    settings: () => live.settings,
    probe: async () => opts.probe?.(++n) ?? null,
    readyTimeoutMs: opts.readyTimeoutMs,
    recordWaitMs: opts.recordWaitMs
  })
  const states: ServerState[] = []
  server.on('state', (s: ServerState) => states.push(s))
  const child = (): FakeChild => children[children.length - 1]
  /** The server process prints its markers and is alive. */
  const boot = (text = MARK): void => {
    proc.alive = true
    child().say(text)
  }
  const scriptsWith = (needle: string): string[] => runs.filter((r) => r.script.includes(needle)).map((r) => r.script)
  return { server, runs, spawned, children, child, proc, live, states, boot, scriptsWith }
}

const tick = async (ms: number): Promise<void> => {
  await vi.advanceTimersByTimeAsync(ms)
}

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
})

describe('OwnServer start', () => {
  it('goes stopped -> starting -> running when our own server answers', async () => {
    const h = harness({ probe: (n) => (n >= 3 ? { argv: OUR_ARGV } : null) })
    expect(h.server.state().status).toBe('stopped')
    const p = h.server.start(PORT)
    await tick(0)
    h.boot('starting\n' + MARK)
    await tick(2000)
    const st = await p
    expect(st.status).toBe('running')
    expect(st.pid).toBe(42)
    expect(st.port).toBe(PORT)
    expect(st.url).toBe('http://127.0.0.1:8288')
    expect(st.startedAt).not.toBeNull()
    expect(h.states.map((s) => s.status)).toEqual(['starting', 'starting', 'running'])
    expect(h.spawned[0]).toContain('CUDA_DEVICE_ORDER=PCI_BUS_ID')
    expect(h.server.logTail()).toContain('starting')
    await expect(h.server.start(8289)).rejects.toThrow()
  })

  it('finds the markers when they are split over chunks', async () => {
    const h = harness({ probe: (n) => (n >= 2 ? { argv: OUR_ARGV } : null) })
    const p = h.server.start(PORT)
    await tick(0)
    h.proc.alive = true
    h.child().say('RULZ_BASE=/b/co')
    h.child().say('mfy\nRULZ_P')
    h.child().say('ID=12')
    h.child().say('34\nRULZ_START=99\n')
    await tick(1000)
    expect((await p).pid).toBe(1234)
  })

  it('does not accept a server that is not ours (different base directory)', async () => {
    const h = harness({ probe: () => ({ argv: ['main.py', '--base-directory', '/other', '--port', String(PORT)] }) })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(10000)
    const st = await p
    expect(st.status).toBe('error')
    expect(st.error).toContain('Another program is already answering on port 8288')
    expect(h.scriptsWith('kill -s INT')).toHaveLength(1) // our own process was stopped again
  })

  it('does not take a base directory that only starts like ours as ours', async () => {
    const h = harness({ probe: () => ({ argv: ['main.py', '--base-directory', '/b/comfy-backup', '--port', String(PORT)] }) })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(10000)
    expect((await p).error).toContain('Another program')
  })

  it('does not take a server on another port as ours', async () => {
    const h = harness({ probe: () => ({ argv: ['main.py', '--base-directory', '/b/comfy', '--port', '9999'] }) })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(10000)
    expect((await p).status).toBe('error')
  })

  it('accepts the --flag=value forms', async () => {
    const h = harness({ probe: () => ({ argv: ['main.py', '--base-directory=/b/comfy', `--port=${PORT}`] }) })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(600)
    expect((await p).status).toBe('running')
  })

  it('ends in error with the last log lines when the process exits early, and can start again', async () => {
    const h = harness()
    const p = h.server.start(PORT)
    await tick(0)
    h.child().stderr.emit('data', Buffer.from('Traceback\nImportError: boom\n'))
    h.child().exit(1)
    await tick(600)
    const st = await p
    expect(st.status).toBe('error')
    expect(st.error).toContain('exited before it was ready')
    expect(st.error).toContain('ImportError: boom')
    expect(st.pid).toBeNull()
    const p2 = h.server.start(PORT)
    await tick(0)
    expect(h.server.state().status).toBe('starting')
    h.child().exit(1)
    await tick(600)
    await p2
  })

  it('gives up after the ready timeout, then stops the process with INT and KILL', async () => {
    const h = harness({ readyTimeoutMs: 2000, obeysInt: false })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(10000)
    const st = await p
    expect(st.status).toBe('error')
    expect(st.error).toContain('did not answer within 2 s')
    const sigs = h.runs.map((r) => r.script).filter((s) => s.includes('kill -s'))
    expect(sigs.map((s) => (s.includes('kill -s INT') ? 'INT' : 'KILL'))).toEqual(['INT', 'KILL'])
    expect(sigs[0]).toContain("RULZ_ID_OK 42 '/b/comfy' '1000'")
    expect(h.scriptsWith('rm -f')).toHaveLength(1)
  })

  it('turns an unexpected exit while running into an error', async () => {
    const h = harness({ probe: () => ({ argv: OUR_ARGV }) })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(600)
    await p
    h.child().stderr.emit('data', Buffer.from('CUDA out of memory\n'))
    h.child().exit(1)
    const st = h.server.state()
    expect(st.status).toBe('error')
    expect(st.error).toContain('stopped unexpectedly')
    expect(st.error).toContain('CUDA out of memory')
    h.proc.alive = false
    expect((await h.server.stop()).status).toBe('stopped')
  })
})

describe('OwnServer stop', () => {
  async function running(opts: Opts = {}) {
    const h = harness({ probe: () => ({ argv: OUR_ARGV }), ...opts })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(600)
    expect((await p).status).toBe('running')
    return h
  }

  it('stops with Ctrl+C when the server obeys, and removes the pid file of that launch', async () => {
    const h = await running()
    const sp = h.server.stop()
    expect(h.server.state().status).toBe('stopping')
    await tick(10)
    expect(await sp).toMatchObject({ status: 'stopped', pid: null, port: null, url: null })
    expect(h.scriptsWith('kill -s KILL')).toHaveLength(0)
    const rm = h.scriptsWith('rm -f')[0]
    expect(rm).toContain("F='/b/comfy/server.pid'")
    expect(rm).toContain('"$P" = "42"')
    expect(h.states.map((s) => s.status).slice(-2)).toEqual(['stopping', 'stopped'])
  })

  it('kills after the stop timeout when Ctrl+C is ignored', async () => {
    const h = await running({ obeysInt: false })
    const sp = h.server.stop()
    await tick(400)
    expect(h.scriptsWith('kill -s KILL')).toHaveLength(0)
    await tick(1500)
    expect((await sp).status).toBe('stopped')
    const sigs = h.runs.map((r) => r.script).filter((s) => s.includes('kill -s'))
    expect(sigs.map((s) => (s.includes('kill -s INT') ? 'INT' : 'KILL'))).toEqual(['INT', 'KILL'])
    expect(h.server.logTail()).toContain('killing it')
  })

  it('ends in error and keeps the pid file when the process cannot be confirmed gone', async () => {
    const h = await running({ obeysInt: false, obeysKill: false })
    const sp = h.server.stop()
    await tick(10000)
    const st = await sp
    expect(st.status).toBe('error')
    expect(st.error).toContain('could not be confirmed stopped')
    expect(h.scriptsWith('rm -f')).toHaveLength(0)
  })

  it('uses the launch record, not the current settings, to stop', async () => {
    const h = await running({ settings: { distro: 'Ubuntu-24.04', baseDir: '/b/comfy' } })
    h.live.settings = { ...h.live.settings, distro: 'Other-Distro', baseDir: '/somewhere/else', stopTimeoutSec: 99 }
    const sp = h.server.stop()
    await tick(10)
    await sp
    const afterStart = h.runs.filter((r) => r.script.includes('kill -s') || r.script.includes('rm -f') || r.script.includes('echo RULZ_ALIVE'))
    expect(afterStart.length).toBeGreaterThan(0)
    expect(afterStart.every((r) => r.distro === 'Ubuntu-24.04')).toBe(true)
    expect(afterStart.every((r) => !r.script.includes('/somewhere/else'))).toBe(true)
    expect(h.scriptsWith('rm -f')[0]).toContain("F='/b/comfy/server.pid'")
  })

  it('before the pid is known: waits for the markers, then stops the process properly', async () => {
    const h = harness({ recordWaitMs: 5000 })
    const p = h.server.start(PORT)
    await tick(0)
    const sp = h.server.stop()
    await tick(1000)
    h.boot() // the markers arrive while stop is waiting
    await tick(1000)
    expect((await sp).status).toBe('stopped')
    expect(h.scriptsWith('kill -s INT')[0]).toContain("RULZ_ID_OK 42 '/b/comfy' '1000'")
    expect((await p).status).toBe('stopped')
  })

  it('before the pid is known: reads the pid file, and uses it when it names a live process', async () => {
    let pidfileCalls = 0
    const h = harness({
      recordWaitMs: 1000,
      respond: (s) => {
        if (s.includes('RULZ_STALE=')) return ++pidfileCalls === 1 ? '' : 'RULZ_BASE=/b/comfy\nRULZ_STALE=99 7\n'
        return undefined
      }
    })
    const p = h.server.start(PORT)
    await tick(0)
    h.proc.alive = true
    const sp = h.server.stop()
    await tick(3000)
    expect((await sp).status).toBe('stopped')
    expect(h.scriptsWith('kill -s INT')[0]).toContain("RULZ_ID_OK 99 '/b/comfy' '7'")
    await p
  })

  it('before the pid is known and nothing is found: does not report stopped, keeps the pid file', async () => {
    const h = harness({ recordWaitMs: 1000 })
    const p = h.server.start(PORT)
    await tick(0)
    const sp = h.server.stop()
    await tick(3000)
    const st = await sp
    expect(st.status).toBe('error')
    expect(st.error).toContain('cannot be sure it has ended')
    expect(h.scriptsWith('rm -f')).toHaveLength(0)
    expect(h.scriptsWith('kill -s')).toHaveLength(0)
    expect(h.child().killed).toBe(true)
    expect((await p).status).toBe('error')
    // the next start can go on; its stale check is what cleans a leftover up
    const p2 = h.server.start(PORT)
    await tick(0)
    expect(h.server.state().status).toBe('starting')
    h.child().exit(1)
    await tick(600)
    await p2
  })

  it('stops a start that is still in the stale check, and never spawns', async () => {
    const h = harness({
      respond: (s) => {
        if (s.includes('RULZ_STALE=')) return 'RULZ_BASE=/b/comfy\nRULZ_STALE=777 5555\n'
        if (s.includes('RULZ_ID_OK 777') && s.includes('echo RULZ_ALIVE')) return 'RULZ_ALIVE\n'
        return undefined
      }
    })
    const p = h.server.start(PORT)
    await tick(100)
    const sp = h.server.stop()
    await tick(3000)
    expect((await sp).status).toBe('stopped')
    expect((await p).status).toBe('stopped')
    expect(h.spawned).toHaveLength(0)
  })

  it('stops a start that has not even begun the stale check', async () => {
    const h = harness()
    const p = h.server.start(PORT)
    const sp = h.server.stop()
    await tick(10)
    await sp
    expect((await p).status).toBe('stopped')
    expect(h.spawned).toHaveLength(0)
  })

  it('cancels a start that is polling for readiness and then stops what was launched', async () => {
    const h = harness()
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(1200)
    const sp = h.server.stop()
    await tick(1200)
    expect((await sp).status).toBe('stopped')
    expect((await p).status).toBe('stopped')
    expect(h.scriptsWith('kill -s INT')).toHaveLength(1)
    expect(h.server.state().error).toBeNull()
  })

  it('dispose waits for an in-flight start and ends stopped', async () => {
    const h = harness()
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    let done = false
    const d = h.server.dispose().then(() => {
      done = true
    })
    expect(done).toBe(false)
    await tick(2000)
    await d
    expect(h.server.state().status).toBe('stopped')
    expect((await p).status).toBe('stopped')
  })

  it('dispose stops a running server', async () => {
    const h = await running()
    const d = h.server.dispose()
    await tick(10)
    await d
    expect(h.server.state().status).toBe('stopped')
    expect(h.scriptsWith('kill -s INT')).toHaveLength(1)
  })

  it('late events of an old launch cannot change the state of a newer one', async () => {
    const h = harness({ probe: () => ({ argv: OUR_ARGV }), stubbornFirst: true })
    const p1 = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(600)
    await p1
    const sp = h.server.stop()
    await tick(3000) // the first wsl.exe ignores kill(), so stop waits for it
    expect((await sp).status).toBe('stopped')

    const p2 = h.server.start(PORT)
    await tick(0)
    h.boot('RULZ_BASE=/b/comfy\nRULZ_PID=43\nRULZ_START=2000\n')
    await tick(600)
    expect((await p2).status).toBe('running')
    expect(h.server.state().pid).toBe(43)

    h.children[0].say('RULZ_BASE=/old\nRULZ_PID=1\nRULZ_START=1\n')
    h.children[0].exit(1) // late exit of the first launch
    expect(h.server.state()).toMatchObject({ status: 'running', pid: 43, error: null })
  })
})

describe('stale own process', () => {
  const staleRespond = (alive: () => boolean) => (s: string): string | undefined => {
    if (s.includes('RULZ_STALE=')) return 'RULZ_BASE=/b/comfy\nRULZ_STALE=777 5555\n'
    if (s.includes('RULZ_ID_OK 777') && s.includes('echo RULZ_ALIVE')) return alive() ? 'RULZ_ALIVE\n' : ''
    if (s.includes('RULZ_ID_OK 777') && s.includes('kill -s')) return ''
    return undefined
  }

  it('stops a leftover process of this app with Ctrl+C, identified by pid, base dir and start time, and then starts', async () => {
    let alive = 2
    const h = harness({ probe: () => ({ argv: OUR_ARGV }), respond: staleRespond(() => alive-- > 0) })
    const p = h.server.start(PORT)
    await tick(5000)
    h.boot()
    await tick(600)
    expect((await p).status).toBe('running')
    const sigs = h.runs.map((r) => r.script).filter((s) => s.includes('kill -s'))
    expect(sigs).toHaveLength(1)
    expect(sigs[0]).toContain("RULZ_ID_OK 777 '/b/comfy' '5555'")
    expect(sigs[0]).toContain('kill -s INT 777')
    expect(h.server.logTail()).toContain('leftover server of this app (pid 777)')
    expect(h.spawned).toHaveLength(1)
  })

  it('kills it when it ignores Ctrl+C', async () => {
    let killed = false
    const respond = (s: string): string | undefined => {
      if (s.includes('RULZ_ID_OK 777') && s.includes('kill -s KILL')) {
        killed = true
        return ''
      }
      return staleRespond(() => !killed)(s)
    }
    const h = harness({ probe: () => ({ argv: OUR_ARGV }), respond })
    const p = h.server.start(PORT)
    await tick(10000)
    h.boot()
    await tick(600)
    expect((await p).status).toBe('running')
    const sigs = h.runs.map((r) => r.script).filter((s) => s.includes('kill -s'))
    expect(sigs.map((s) => (s.includes('kill -s INT') ? 'INT' : 'KILL'))).toEqual(['INT', 'KILL'])
  })

  it('refuses to start when the leftover will not die', async () => {
    const h = harness({ respond: staleRespond(() => true) })
    const p = h.server.start(PORT)
    await tick(20000)
    const st = await p
    expect(st.status).toBe('error')
    expect(st.error).toContain('would not stop')
    expect(h.spawned).toHaveLength(0)
  })

  it('leaves everything alone when there is no stale process', async () => {
    const h = harness({ probe: () => ({ argv: OUR_ARGV }) })
    const p = h.server.start(PORT)
    await tick(0)
    h.boot()
    await tick(600)
    await p
    expect(h.scriptsWith('kill -s')).toHaveLength(0)
  })
})

describe('log', () => {
  it('keeps only the last 2000 lines and keeps \\r progress updates intact', async () => {
    const h = harness()
    const logs: string[] = []
    h.server.on('log', (c: string) => logs.push(c))
    const p = h.server.start(PORT)
    await tick(0)
    h.child().say(Array.from({ length: 2500 }, (_, i) => `line ${i}`).join('\n') + '\n')
    h.child().say('progress 10%\rprogress 50%\r')
    const tail = h.server.logTail()
    expect(tail).toContain('line 2499')
    expect(tail).not.toContain('line 100\n')
    expect(tail.split('\n').length).toBeLessThanOrEqual(2001)
    expect(tail.endsWith('progress 10%\rprogress 50%\r')).toBe(true)
    expect(logs.join('')).toContain('progress 10%\rprogress 50%\r')
    h.child().exit(0)
    await tick(600)
    await p
  })

  it('strips NUL bytes', async () => {
    const h = harness()
    const p = h.server.start(PORT)
    await tick(0)
    h.child().say('a\0b\0c\n')
    expect(h.server.logTail()).toContain('abc')
    h.child().exit(0)
    await tick(600)
    await p
  })
})
