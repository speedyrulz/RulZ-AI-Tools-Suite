// The local ServerHandle: a llama-server child process on this PC, plus the pid file and orphan cleanup that only
// make sense for processes of this PC (docs/vast-design.md, section 7).
import { spawn, type ChildProcess } from 'node:child_process'
import { dirname } from 'node:path'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { engineEnv, isPidAlive, killTree, llamaServerPids } from './process'
import type { ServerExit, ServerHandle } from './serverHandle'

/** How long stop() waits for llama-server to exit after the first kill, and again after a forced one. */
export const UNLOAD_WAIT_MS = 8000
export const FORCED_UNLOAD_WAIT_MS = 5000

export interface LocalServerOptions {
  exe: string
  args: string[]
  port: number
  /** Starts llama-server; injectable for tests. */
  spawnServer?: (exe: string, args: string[]) => ChildProcess
  /** Kills a server's process tree (`force` = synchronous); injectable for tests. */
  killProcess?: (proc: ChildProcess, force: boolean) => void
  /** Overrides the stop waits [first, after forcing] in ms; for tests. */
  unloadWaitMs?: [number, number]
}

/** Starts llama-server as a child process of the app. */
export function startLocalServer(opts: LocalServerOptions): ServerHandle {
  const proc = opts.spawnServer
    ? opts.spawnServer(opts.exe, opts.args)
    : spawn(opts.exe, opts.args, {
        cwd: dirname(opts.exe),
        env: engineEnv(),
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe']
      })
  return new LocalServerHandle(proc, opts.port, opts)
}

export class LocalServerHandle implements ServerHandle {
  readonly pid: number | null
  /** Output from before anyone listened; delivered to the first listener. */
  private early: string[] | null = []
  private outputFns: ((chunk: string) => void)[] = []
  private exitFns: ((exit: ServerExit) => void)[] = []
  private exitInfo: ServerExit | null = null
  private waiters: (() => void)[] = []

  constructor(
    private readonly proc: ChildProcess,
    readonly port: number,
    private readonly opts: Pick<LocalServerOptions, 'killProcess' | 'unloadWaitMs'> = {}
  ) {
    this.pid = proc.pid ?? null
    // Decoded as streams, so a character split between two chunks (non-ASCII file names) isn't garbled.
    proc.stdout?.setEncoding?.('utf8')
    proc.stderr?.setEncoding?.('utf8')
    proc.stdout?.on('data', (c: Buffer | string) => this.deliver(String(c)))
    proc.stderr?.on('data', (c: Buffer | string) => this.deliver(String(c)))
    // The leading newline ends whatever line was half written, like flushing the splitter would.
    proc.on('error', (err) => {
      this.deliver(`\n[failed to start] ${err.message}\n`)
      this.finish({ code: null })
    })
    proc.on('exit', (code) => {
      this.deliver(`\n[process exited with code ${code}]\n`)
      this.finish({ code })
    })
  }

  onOutput(fn: (chunk: string) => void): void {
    this.outputFns.push(fn)
    if (this.early) {
      const early = this.early
      this.early = null
      for (const c of early) fn(c)
    }
  }

  onExit(fn: (exit: ServerExit) => void): void {
    if (this.exitInfo) fn(this.exitInfo)
    else this.exitFns.push(fn)
  }

  reconnecting(): boolean {
    return false
  }

  async stop(): Promise<void> {
    if (this.exitInfo) return
    const kill = this.opts.killProcess ?? ((proc: ChildProcess, force: boolean) => killTree(proc.pid, force))
    kill(this.proc, false)
    const [firstWait, forcedWait] = this.opts.unloadWaitMs ?? [UNLOAD_WAIT_MS, FORCED_UNLOAD_WAIT_MS]
    if (!(await this.waitExit(firstWait)) && isPidAlive(this.proc.pid)) {
      // Still holding its memory: force-kill the whole tree and wait again before the caller loads
      // the next model into the same VRAM.
      this.deliver('\n[did not exit in time; forcing]\n')
      kill(this.proc, true)
      await this.waitExit(forcedWait)
    }
    // A process that never confirms its exit is treated as gone, as it always was here.
    if (!this.exitInfo) this.finish({ code: null })
  }

  detachSync(): void {
    killTree(this.proc.pid, true)
  }

  private deliver(chunk: string): void {
    if (this.exitInfo) return
    if (this.early) this.early.push(chunk)
    else for (const fn of this.outputFns) fn(chunk)
  }

  private finish(exit: ServerExit): void {
    if (this.exitInfo) return
    this.exitInfo = exit
    for (const w of this.waiters.splice(0)) w()
    for (const fn of this.exitFns.splice(0)) fn(exit)
  }

  /** Resolves true once the process exits, or false after `ms`. */
  private waitExit(ms: number): Promise<boolean> {
    return new Promise((resolve) => {
      if (this.exitInfo) return resolve(true)
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w !== onExit)
        resolve(false)
      }, ms)
      const onExit = (): void => {
        clearTimeout(timer)
        resolve(true)
      }
      this.waiters.push(onExit)
    })
  }
}

/** Writes the pids of the app's own llama-server processes, so a crash can be cleaned up after. Never throws. */
export function savePidFile(path: string, pids: number[]): Promise<void> {
  return writeJsonAtomic(path, pids).catch(() => undefined)
}

/** Kills llama-server processes left behind if the app crashed last time. */
export async function killOrphanProcesses(path: string): Promise<void> {
  // A locked PID file mustn't stop the app from starting; orphans then wait for the next start.
  const pids = await readJson<number[]>(path, []).catch(() => [] as number[])
  if (!pids.length) return
  // One process listing for all of them; only PIDs still belonging to llama-server are killed.
  const live = llamaServerPids()
  for (const pid of pids) if (live.has(pid)) killTree(pid, true)
}
