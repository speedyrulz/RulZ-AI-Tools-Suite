// The app's own ComfyUI server inside WSL: start, watch until ready, stop (Ctrl+C first, kill after a timeout).
//
// Every start creates one immutable "launch": a snapshot of the settings plus, once the start script has printed
// it, the process record (distro, absolute base directory, pid, start time). Stopping and cleaning up only ever use
// that record, never the current settings. Start, stop and dispose run one at a time; stop cancels a start in flight.
import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { StringDecoder } from 'node:string_decoder'
import type { OwnServerSettings, ServerState } from '@shared/types'
import {
  buildAliveScript,
  buildPidFileScript,
  buildRemovePidScript,
  buildStartScript,
  buildStopScript,
  type ProcessIdentity
} from './scripts'
import type { WslRunner } from './wsl'

export interface OwnServerDeps {
  wsl: WslRunner
  settings: () => OwnServerSettings
  /** GET /system_stats of the server at `url`: its argv, or null when nothing (that looks like ComfyUI) answers. */
  probe: (url: string) => Promise<{ argv: string[] } | null>
  now?: () => number
  /** Test hooks. Defaults: 500 ms, 180 s and 10 s. */
  pollMs?: number
  readyTimeoutMs?: number
  recordWaitMs?: number
}

export interface LaunchRecord {
  readonly distro: string
  /** Absolute base directory, as printed by the start script. */
  readonly baseDir: string
  readonly pid: number
  readonly startTime: string
}

interface Launch {
  readonly settings: OwnServerSettings
  readonly port: number
  readonly url: string
  child: ChildProcess | null
  exited: boolean
  waiters: Array<() => void>
  rec: LaunchRecord | null
  markBuf: string
  cancelled: boolean
}

const MAX_LOG_LINES = 2000
const EMPTY_STATE: ServerState = { status: 'stopped', port: null, url: null, pid: null, device: null, startedAt: null, error: null }

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

function argvHasBase(argv: string[], base: string): boolean {
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--base-directory' && argv[i + 1] === base) return true
    if (argv[i] === `--base-directory=${base}`) return true
  }
  return false
}

function argvHasPort(argv: string[], port: number): boolean {
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--port' && argv[i + 1] === String(port)) return true
    if (argv[i] === `--port=${port}`) return true
  }
  return false
}

export class OwnServer extends EventEmitter {
  private st: ServerState = { ...EMPTY_STATE }
  private lines: string[] = []
  private partial = ''
  private launch: Launch | null = null
  private chain: Promise<unknown> = Promise.resolve()
  /** Counts stop() calls; a start queued before a stop() is cancelled by it even if it has not begun. */
  private stopTicket = 0
  private stopDone: Promise<ServerState> = Promise.resolve({ ...EMPTY_STATE })

  constructor(private readonly d: OwnServerDeps) {
    super()
  }

  state(): ServerState {
    return { ...this.st }
  }

  /** The running (or starting) launch's process record: distro and absolute base directory. Null before the start script printed it. */
  record(): LaunchRecord | null {
    return this.launch?.rec ?? null
  }

  logTail(): string {
    return this.lines.length ? `${this.lines.join('\n')}\n${this.partial}` : this.partial
  }

  private now(): number {
    return (this.d.now ?? Date.now)()
  }

  /** Sets the state. A launch that is no longer the current one cannot change it. */
  private set(launch: Launch | null, patch: Partial<ServerState>): void {
    if (launch !== null && this.launch !== launch) return
    this.st = { ...this.st, ...patch }
    this.emit('state', this.state())
  }

  private appendLog(text: string): void {
    if (!text) return
    const parts = (this.partial + text).split('\n')
    this.partial = parts.pop() ?? ''
    for (const p of parts) this.lines.push(p)
    if (this.lines.length > MAX_LOG_LINES) this.lines.splice(0, this.lines.length - MAX_LOG_LINES)
    this.emit('log', text)
  }

  private note(text: string): void {
    this.appendLog(`[model merger] ${text}\n`)
  }

  private lastLogLines(n = 15): string {
    const all = (this.partial ? [...this.lines, this.partial] : this.lines).map((l) => l.replace(/\r/g, '\n').trim()).filter(Boolean)
    return all.slice(-n).join('\n')
  }

  private onChunk(launch: Launch, chunk: Buffer | string, decoder: StringDecoder): void {
    const text = (typeof chunk === 'string' ? chunk : decoder.write(chunk)).replace(/\0/g, '')
    if (!text) return
    if (!launch.rec) {
      launch.markBuf = (launch.markBuf + text).slice(-8192)
      const base = /RULZ_BASE=([^\r\n]+)\r?\n/.exec(launch.markBuf)
      const pid = /RULZ_PID=(\d+)\r?\n/.exec(launch.markBuf)
      const start = /RULZ_START=(\d+)\r?\n/.exec(launch.markBuf)
      if (base && pid && start) {
        launch.rec = Object.freeze({ distro: launch.settings.distro, baseDir: base[1], pid: Number(pid[1]), startTime: start[1] })
        this.set(launch, { pid: launch.rec.pid })
      }
    }
    this.appendLog(text)
  }

  private waitExit(launch: Launch, ms: number): Promise<boolean> {
    if (launch.exited) return Promise.resolve(true)
    return new Promise((resolve) => {
      const done = (): void => {
        clearTimeout(timer)
        resolve(true)
      }
      const timer = setTimeout(() => {
        launch.waiters = launch.waiters.filter((w) => w !== done)
        resolve(false)
      }, ms)
      launch.waiters.push(done)
    })
  }

  private markExited(launch: Launch): void {
    if (launch.exited) return
    launch.exited = true
    const w = launch.waiters
    launch.waiters = []
    for (const f of w) f()
    if (this.launch === launch && this.st.status === 'running') {
      this.set(launch, { status: 'error', pid: null, error: `The ComfyUI server stopped unexpectedly.\n${this.lastLogLines()}`.trim() })
    }
  }

  private identity(rec: LaunchRecord): ProcessIdentity {
    return { baseDir: rec.baseDir, startTime: rec.startTime }
  }

  private async signal(rec: LaunchRecord, sig: 'INT' | 'KILL'): Promise<void> {
    await this.d.wsl.run(rec.distro, buildStopScript(rec.pid, sig, this.identity(rec)), { timeoutMs: 10000 })
  }

  /** True while the recorded process is running; also true when WSL cannot be asked, because then it is not confirmed gone. */
  private async isAlive(rec: LaunchRecord): Promise<boolean> {
    const r = await this.d.wsl.run(rec.distro, buildAliveScript(rec.pid, this.identity(rec)), { timeoutMs: 10000 })
    if (r.code !== 0) return true // the alive script always ends with `true`, so a failure is WSL/bash, not "gone"
    return r.stdout.includes('RULZ_ALIVE')
  }

  /** Waits until the recorded process is gone (checked in WSL, not only by wsl.exe ending). */
  private async waitGone(rec: LaunchRecord, ms: number, launch: Launch | null, abort?: () => boolean): Promise<boolean> {
    const deadline = this.now() + ms
    for (;;) {
      if (!(await this.isAlive(rec))) return true
      if (abort?.() || this.now() >= deadline) return false
      if (launch) await this.waitExit(launch, Math.min(500, this.d.pollMs ?? 500))
      else await sleep(Math.min(500, this.d.pollMs ?? 500))
    }
  }

  /** Ctrl+C, wait, then kill. Returns whether the process is confirmed gone. */
  private async endProcess(rec: LaunchRecord, timeoutSec: number, launch: Launch | null, abort?: () => boolean): Promise<boolean> {
    await this.signal(rec, 'INT').catch(() => undefined)
    if (await this.waitGone(rec, Math.max(1, timeoutSec) * 1000, launch, abort)) return true
    if (abort?.()) return false
    this.note(`The server (pid ${rec.pid}) did not stop within ${timeoutSec} s; killing it.`)
    await this.signal(rec, 'KILL').catch(() => undefined)
    return this.waitGone(rec, 5000, launch, abort)
  }

  /** Reads the pid file of the snapshot's base directory: the record of a process whose markers we never saw. */
  private async readPidFile(s: OwnServerSettings): Promise<LaunchRecord | null> {
    const r = await this.d.wsl.run(s.distro, buildPidFileScript(s.baseDir), { timeoutMs: 10000 })
    const base = /RULZ_BASE=([^\r\n]+)/.exec(r.stdout)
    const stale = /RULZ_STALE=(\d+) (\d+)/.exec(r.stdout)
    if (!base || !stale) return null
    return Object.freeze({ distro: s.distro, baseDir: base[1], pid: Number(stale[1]), startTime: stale[2] })
  }

  /** A leftover main.py of this app (same absolute base dir, same start time) from an earlier session: stop it. */
  private async killStale(launch: Launch): Promise<void> {
    const rec = await this.readPidFile(launch.settings)
    if (!rec || launch.cancelled) return
    this.note(`Found a leftover server of this app (pid ${rec.pid}); stopping it.`)
    const gone = await this.endProcess(rec, launch.settings.stopTimeoutSec, null, () => launch.cancelled)
    if (launch.cancelled) return
    if (!gone) throw new Error(`A leftover ComfyUI server of this app (pid ${rec.pid}) would not stop. Stop it by hand in WSL and try again.`)
    this.note(`The leftover server (pid ${rec.pid}) has stopped.`)
  }

  /** Ends the launched process. `ok` false means it could not be confirmed gone; the pid file is then kept. */
  private async terminate(launch: Launch): Promise<{ ok: boolean; message: string }> {
    const s = launch.settings
    if (launch.child && !launch.exited && !launch.rec) {
      // Wait for the start script to print its markers, then fall back to the pid file.
      const deadline = this.now() + (this.d.recordWaitMs ?? 10000)
      while (!launch.rec && !launch.exited && this.now() < deadline) await this.waitExit(launch, 100)
      if (!launch.rec && !launch.exited) {
        const fromFile = await this.readPidFile(s).catch(() => null)
        if (fromFile) launch.rec = fromFile
      }
    }
    const rec = launch.rec
    if (launch.child && !launch.exited && !rec) {
      try {
        launch.child.kill()
      } catch {
        // already gone
      }
      await this.waitExit(launch, 2000)
      return {
        ok: false,
        message:
          'ComfyUI was stopped before it said which process it is, so you cannot be sure it has ended. Check WSL for a running ComfyUI of this app; the next start will clean it up if it is still there.'
      }
    }
    if (rec) {
      if (!launch.exited || (await this.isAlive(rec))) {
        const gone = await this.endProcess(rec, s.stopTimeoutSec, launch)
        if (!gone) {
          try {
            launch.child?.kill()
          } catch {
            // already gone
          }
          return {
            ok: false,
            message: `ComfyUI (pid ${rec.pid}) could not be confirmed stopped, so you may still have a server running. Try Stop again, or end it in WSL; the next start will clean it up.`
          }
        }
      }
    }
    if (launch.child && !launch.exited) {
      try {
        launch.child.kill()
      } catch {
        // already gone
      }
      await this.waitExit(launch, 2000)
    }
    if (rec) await this.d.wsl.run(rec.distro, buildRemovePidScript(rec.baseDir, rec.pid), { timeoutMs: 10000 }).catch(() => undefined)
    return { ok: true, message: '' }
  }

  /** Take this before any await that precedes start(): a stop() or dispose() after it cancels that start. */
  beginStart(): number {
    return this.stopTicket
  }

  /** `ticket` from beginStart(); `expected` = the settings that were checked, refused if they changed since. */
  start(port: number, ticket: number = this.stopTicket, expected?: OwnServerSettings): Promise<ServerState> {
    const status = this.st.status
    if (status !== 'stopped' && status !== 'error') {
      return Promise.reject(new Error(`The server cannot be started while it is ${status}.`))
    }
    const run = this.enqueue(() => this.doStart(port, ticket, expected))
    return run.then(async (r) => {
      if (r.cancelled) await this.stopDone
      return r.cancelled ? this.state() : r.state
    })
  }

  stop(): Promise<ServerState> {
    this.stopTicket++
    if (this.launch) this.launch.cancelled = true // cancels a start in flight; it ends at its next checkpoint
    if (this.st.status === 'starting' || this.st.status === 'running') this.set(null, { status: 'stopping' })
    const p = this.enqueue(() => this.doStop())
    this.stopDone = p
    return p
  }

  async dispose(): Promise<void> {
    await this.stop()
    await this.chain
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const p = this.chain.then(fn)
    this.chain = p.catch(() => undefined)
    return p
  }

  private async doStop(): Promise<ServerState> {
    const l = this.launch
    if (!l) {
      if (this.st.status !== 'stopped') this.set(null, { ...EMPTY_STATE })
      return this.state()
    }
    this.set(null, { status: 'stopping' })
    const r = await this.terminate(l)
    this.launch = null
    if (r.ok) this.set(null, { ...EMPTY_STATE })
    else this.set(null, { ...EMPTY_STATE, status: 'error', error: r.message })
    return this.state()
  }

  private async doStart(port: number, ticket: number, expected?: OwnServerSettings): Promise<{ state: ServerState; cancelled: boolean }> {
    if (ticket !== this.stopTicket) return { state: this.state(), cancelled: true }
    if (this.st.status !== 'stopped' && this.st.status !== 'error') {
      throw new Error(`The server cannot be started while it is ${this.st.status}.`)
    }
    const s: OwnServerSettings = JSON.parse(JSON.stringify(this.d.settings())) as OwnServerSettings
    if (expected && JSON.stringify(expected) !== JSON.stringify(s)) {
      throw new Error('The server settings changed while they were checked; start again.')
    }
    const url = `http://127.0.0.1:${port}`
    const launch: Launch = { settings: s, port, url, child: null, exited: true, waiters: [], rec: null, markBuf: '', cancelled: false }
    this.launch = launch
    this.set(launch, { status: 'starting', port, url, pid: null, device: s.device, startedAt: null, error: null })

    const cancelled = (): { state: ServerState; cancelled: boolean } => ({ state: this.state(), cancelled: true })
    const fail = async (message: string): Promise<{ state: ServerState; cancelled: boolean }> => {
      const r = await this.terminate(launch)
      if (this.launch === launch) {
        this.launch = null
        this.set(null, { status: 'error', pid: null, startedAt: null, error: r.ok ? message : `${message}\n${r.message}` })
      }
      return { state: this.state(), cancelled: false }
    }

    try {
      await this.killStale(launch)
      if (launch.cancelled) return cancelled()

      const child = this.d.wsl.spawn(s.distro, buildStartScript(s, port))
      launch.child = child
      launch.exited = false
      const outDec = new StringDecoder('utf8')
      const errDec = new StringDecoder('utf8')
      child.stdout?.on('data', (c: Buffer | string) => this.onChunk(launch, c, outDec))
      child.stderr?.on('data', (c: Buffer | string) => this.onChunk(launch, c, errDec))
      child.on('error', (e) => {
        this.note(`Could not run wsl.exe: ${e.message}`)
        this.markExited(launch)
      })
      child.on('exit', () => this.markExited(launch))
      child.on('close', () => this.markExited(launch))

      const pollMs = this.d.pollMs ?? 500
      const timeoutMs = this.d.readyTimeoutMs ?? 180000
      const t0 = this.now()
      for (;;) {
        if (launch.cancelled) return cancelled()
        if (launch.exited) return await fail(`ComfyUI exited before it was ready.\n${this.lastLogLines()}`.trim())
        let info: { argv: string[] } | null = null
        try {
          info = await this.d.probe(url)
        } catch {
          info = null
        }
        if (launch.cancelled) return cancelled()
        if (info && launch.rec) {
          if (launch.exited) return await fail(`ComfyUI exited before it was ready.\n${this.lastLogLines()}`.trim())
          if (!argvHasBase(info.argv, launch.rec.baseDir) || !argvHasPort(info.argv, port)) {
            return await fail(
              `Another program is already answering on port ${port}, so the ComfyUI server of this app cannot use it. Choose another start port in the settings and try again.`
            )
          }
          this.set(launch, { status: 'running', startedAt: this.now(), error: null })
          return { state: this.state(), cancelled: false }
        }
        if (this.now() - t0 >= timeoutMs) {
          return await fail(`ComfyUI did not answer within ${Math.round(timeoutMs / 1000)} s.\n${this.lastLogLines()}`.trim())
        }
        await sleep(pollMs)
      }
    } catch (e) {
      if (launch.cancelled) return cancelled()
      return fail(`The server could not be started: ${(e as Error).message}`)
    }
  }
}
