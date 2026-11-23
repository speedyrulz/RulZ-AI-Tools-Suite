// MCP scopes: private copies of the servers that keep state per process (REA keeps its open target), so several
// unattended sessions can each work on their own binary at once (docs/vast-design.md, section 8). Everything else
// stays on the shared connection. The tools a scope offers are the shared ones: same names, same definitions.
import { execFile, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { FileToolScope, ToolResult, WireToolCall } from '@shared/types'
import { contentToText } from './config'

/** Servers that get a private process per scope. REA is the one that keeps an open target. */
export const SCOPED_SERVERS: readonly string[] = ['rea']

/** Calls in a scope get longer than shared ones: three REA processes analysing at once take minutes on a first binary. */
export const SCOPE_CALL_TIMEOUT_MS = 300_000
export const MAX_RESULT_CHARS = 24000

/** The sentence config changes and restarts are refused with while a run holds scopes. */
export const SCOPES_OPEN_MESSAGE = 'A practice run is using the MCP servers; stop it first.'

const taskkillArgs = (pids: number[]): string[] => ['/F', '/T', ...pids.flatMap((p) => ['/PID', String(p)])]

/**
 * On Windows, npx/uvx-style launchers run the real server as a child process, and closing the
 * launcher (all the SDK does) leaves that child running. taskkill /T ends the whole tree.
 */
export function killTreesSync(pids: number[]): void {
  if (process.platform !== 'win32' || !pids.length) return
  spawnSync('taskkill', taskkillArgs(pids), { windowsHide: true, stdio: 'ignore', timeout: 5000 })
}

export function killTree(pid: number): Promise<void> {
  return new Promise((resolve) => execFile('taskkill', taskkillArgs([pid]), { windowsHide: true, timeout: 5000 }, () => resolve()))
}

/**
 * Folders where npx/uvx-style launchers usually live but that are often missing from the PATH
 * an app started from Explorer receives.
 */
export function launcherDirs(): string[] {
  const home = homedir()
  return [
    'C:\\Program Files\\nodejs',
    join(process.env.APPDATA ?? join(home, 'AppData', 'Roaming'), 'npm'),
    join(home, '.local', 'bin'),
    join(home, '.cargo', 'bin')
  ].filter((d) => existsSync(d))
}

/** A started private process, as the scope uses it. */
export interface ScopedProcess {
  callTool(name: string, args: Record<string, unknown>, opts: { timeout: number; signal?: AbortSignal }): Promise<{ content?: unknown; structuredContent?: unknown; isError?: boolean }>
  /** Process id while it runs. */
  pid(): number | null
  /** False once the process has exited or the connection closed. */
  alive(): boolean
  /** Last lines it wrote to stderr, for an error message. */
  stderrTail(): string[]
  /** Polite close, then the whole process tree if it lingers. */
  close(): Promise<void>
}

export interface ScopeDeps {
  /** Servers that run privately, by name. */
  scoped: ReadonlySet<string>
  /** Where a tool lives: its server and its own (unqualified) name. */
  locate(qualifiedName: string): { server: string; tool: string } | undefined
  /** Every other tool. */
  shared(call: Pick<WireToolCall['function'], 'name' | 'arguments'>, signal?: AbortSignal, files?: FileToolScope): Promise<ToolResult>
  /** Starts a private process of a server from the current config. */
  start(server: string): Promise<ScopedProcess>
  /** Called once when the scope is closed. */
  onClose(): void
  timeoutMs?: number
}

/**
 * One lane's private servers. Processes start at the first call that needs them. `beginTask()` ends them, so
 * the next task starts from a clean state: REA keeps every binary it opened until `close_binary`, and the
 * model cannot be trusted to close what it opened; a fresh process is the only reliable clean state. What it
 * costs is a few seconds of start-up; the first analysis of a binary (15-55 s) is paid per binary anyway.
 */
export class McpScope {
  private procs = new Map<string, ScopedProcess>()
  private starting = new Map<string, Promise<ScopedProcess>>()
  private closed = false

  constructor(private readonly deps: ScopeDeps) {}

  /** The scope has been closed; its calls fail. */
  get isClosed(): boolean {
    return this.closed
  }

  /** A new task starts: the next call that needs a private server starts a fresh one. */
  beginTask(): void {
    const old = [...this.procs.values()]
    this.procs.clear()
    for (const p of old) void p.close().catch(() => undefined)
  }

  private async process(server: string): Promise<ScopedProcess> {
    const have = this.procs.get(server)
    if (have?.alive()) return have
    if (have) this.procs.delete(server)
    let pending = this.starting.get(server)
    if (!pending) {
      pending = this.deps.start(server).finally(() => this.starting.delete(server))
      this.starting.set(server, pending)
    }
    const p = await pending
    if (this.closed) {
      void p.close().catch(() => undefined)
      throw new Error('The practice run has ended.')
    }
    this.procs.set(server, p)
    return p
  }

  /** Runs a tool. Aborting `signal` stops the call. Tools of private servers run in this scope's process. */
  async callTool(call: Pick<WireToolCall['function'], 'name' | 'arguments'>, signal?: AbortSignal, files?: FileToolScope): Promise<ToolResult> {
    const stopped: ToolResult = { text: 'The tool call was stopped.', isError: true }
    if (signal?.aborted) return stopped
    if (this.closed) return { text: 'The practice run has ended.', isError: true }
    const where = this.deps.locate(call.name)
    if (!where || !this.deps.scoped.has(where.server)) return this.deps.shared(call, signal, files)
    let args: Record<string, unknown> = {}
    if (call.arguments.trim()) {
      try {
        args = JSON.parse(call.arguments)
      } catch {
        return { text: `The tool arguments were not valid JSON: ${call.arguments}`, isError: true }
      }
    }
    let proc: ScopedProcess
    try {
      proc = await this.process(where.server)
    } catch (err) {
      if (signal?.aborted) return stopped
      return { text: `Tool call failed: could not start ${where.server} for this task: ${(err as Error).message}`, isError: true }
    }
    try {
      const res = await proc.callTool(where.tool, args, { timeout: this.deps.timeoutMs ?? SCOPE_CALL_TIMEOUT_MS, signal })
      let text = contentToText(res.content)
      if (!text && res.structuredContent) text = JSON.stringify(res.structuredContent)
      if (text.length > MAX_RESULT_CHARS) text = `${text.slice(0, MAX_RESULT_CHARS)}\n…[truncated ${text.length - MAX_RESULT_CHARS} characters]`
      return { text, isError: !!res.isError }
    } catch (err) {
      if (signal?.aborted) return stopped
      if (!proc.alive()) {
        // The process died under the call: what it had open is gone. The next call starts a fresh one.
        if (this.procs.get(where.server) === proc) this.procs.delete(where.server)
        const tail = proc.stderrTail().slice(-3).join(' | ')
        return {
          text: `Tool call failed: the ${where.server} process stopped (${(err as Error).message})${tail ? `: ${tail}` : ''}. It starts again on the next call, with nothing open in it; open the target again.`,
          isError: true
        }
      }
      return { text: `Tool call failed: ${(err as Error).message}`, isError: true }
    }
  }

  /** Ends the private processes (and their trees) and releases the scope. */
  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    const all = [...this.procs.values()]
    this.procs.clear()
    this.deps.onClose()
    await Promise.all(all.map((p) => p.close().catch(() => undefined)))
  }

  /** App shutdown: ends the process trees now, without waiting. */
  closeSync(): void {
    this.closed = true
    const pids = [...this.procs.values()].flatMap((p) => {
      const pid = p.pid()
      return pid ? [pid] : []
    })
    this.procs.clear()
    killTreesSync(pids)
  }
}
