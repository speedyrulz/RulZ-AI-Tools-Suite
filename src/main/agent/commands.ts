// Runs PowerShell commands in the project folder: streamed output, time limits, and killing the
// whole process tree on Stop.
import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { delimiter, join } from 'node:path'

/** Cap for one tool result sent to the model. */
export const MAX_RESULT_CHARS = 20_000

/** Keeps the start and the end of a long text. */
export function headTail(text: string, max = MAX_RESULT_CHARS): string {
  if (text.length <= max) return text
  const head = Math.floor(max * 0.4)
  const tail = max - head
  const omitted = text.length - head - tail
  return `${text.slice(0, head)}\n[… ${omitted.toLocaleString('en-US')} characters omitted …]\n${text.slice(-tail)}`
}

// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g

/** Removes colour codes and collapses carriage-return progress lines to their final state. */
export function cleanOutput(text: string): string {
  return text
    .replace(ANSI, '')
    .split('\n')
    .map((line) => {
      const parts = line.replace(/\r$/, '').split('\r')
      return parts[parts.length - 1]
    })
    .join('\n')
}

/** Extra folders where Node.js, npm and Python usually live, so fresh installs are found. */
export function extraPathDirs(env: NodeJS.ProcessEnv = process.env): string[] {
  const dirs: string[] = []
  const pf = env.ProgramFiles ?? 'C:\\Program Files'
  const local = env.LOCALAPPDATA
  const roaming = env.APPDATA
  dirs.push(join(pf, 'nodejs'), join(pf, 'Git', 'cmd'))
  if (roaming) dirs.push(join(roaming, 'npm'))
  if (local) {
    const pyRoot = join(local, 'Programs', 'Python')
    try {
      for (const d of readdirSync(pyRoot).filter((n) => /^Python3\d+/i.test(n)).sort().reverse()) {
        dirs.push(join(pyRoot, d), join(pyRoot, d, 'Scripts'))
      }
    } catch {
      // no per-user Python
    }
    dirs.push(join(local, 'Microsoft', 'WindowsApps'))
  }
  return dirs.filter((d) => existsSync(d))
}

/** Environment for commands: the app's own Electron settings removed, quiet tools, PATH extended. */
export function commandEnv(base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const [k, v] of Object.entries(base)) if (!/^ELECTRON_/i.test(k) && k.toUpperCase() !== 'NODE_OPTIONS') env[k] = v
  const pathKey = Object.keys(env).find((k) => k.toUpperCase() === 'PATH') ?? 'Path'
  const current = (env[pathKey] ?? '').split(delimiter).filter(Boolean)
  const have = new Set(current.map((p) => p.toLowerCase()))
  env[pathKey] = [...current, ...extraPathDirs(base).filter((d) => !have.has(d.toLowerCase()))].join(delimiter)
  Object.assign(env, {
    npm_config_yes: 'true',
    npm_config_fund: 'false',
    npm_config_audit: 'false',
    npm_config_update_notifier: 'false',
    PIP_DISABLE_PIP_VERSION_CHECK: '1',
    PYTHONIOENCODING: 'utf-8',
    PYTHONUNBUFFERED: '1',
    NO_COLOR: '1',
    FORCE_COLOR: '0'
  })
  return env
}

/** PowerShell arguments for a command, passed base64-encoded so no quoting can break it. */
export function powershellArgs(command: string): string[] {
  const script = [
    "$ProgressPreference = 'SilentlyContinue'",
    '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8',
    '$OutputEncoding = [System.Text.Encoding]::UTF8',
    command,
    '$__ok = $?',
    'if ($LASTEXITCODE) { exit $LASTEXITCODE }',
    'if (-not $__ok) { exit 1 }'
  ].join('\n')
  return ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')]
}

/** Kills a process and everything it started; resolves once it has exited (or after 5 s). */
export function killTree(child: ChildProcess): Promise<void> {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) return Promise.resolve()
  return new Promise((resolve) => {
    const done = (): void => {
      clearTimeout(timer)
      resolve()
    }
    const timer = setTimeout(done, 5000)
    child.once('close', done)
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }).on('error', () => child.kill())
    } else {
      child.kill('SIGKILL')
    }
  })
}

export function startShell(command: string, cwd: string): ChildProcess {
  const shell = process.platform === 'win32' ? 'powershell.exe' : 'pwsh'
  return spawn(shell, powershellArgs(command), {
    cwd,
    env: commandEnv(),
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe']
  })
}

export interface CommandResult {
  exitCode: number | null
  output: string
  timedOut: boolean
  stopped: boolean
}

/** Runs a command to completion (or until the time limit or `signal`), streaming its output. */
export function runCommand(
  command: string,
  opts: { cwd: string; timeoutMs: number; signal?: AbortSignal; onOutput?: (chunk: string) => void; start?: typeof startShell }
): Promise<CommandResult> {
  return new Promise((resolve) => {
    let output = ''
    let timedOut = false
    let stopped = false
    let child: ChildProcess
    try {
      child = (opts.start ?? startShell)(command, opts.cwd)
    } catch (e) {
      resolve({ exitCode: null, output: `Could not start PowerShell: ${(e as Error).message}`, timedOut: false, stopped: false })
      return
    }
    const take = (decoder: InstanceType<typeof TextDecoder>) => (buf: Buffer) => {
      const chunk = decoder.decode(buf, { stream: true })
      if (!chunk) return
      output += chunk
      // Keep memory bounded for chatty commands; the head and tail are what get reported.
      if (output.length > MAX_RESULT_CHARS * 20) output = headTail(output, MAX_RESULT_CHARS * 10)
      opts.onOutput?.(chunk)
    }
    child.stdout?.on('data', take(new TextDecoder()))
    child.stderr?.on('data', take(new TextDecoder()))

    const timer = setTimeout(() => {
      timedOut = true
      void killTree(child)
    }, opts.timeoutMs)
    const onAbort = (): void => {
      stopped = true
      void killTree(child)
    }
    if (opts.signal?.aborted) onAbort()
    else opts.signal?.addEventListener('abort', onAbort, { once: true })

    const finish = (exitCode: number | null, extra = ''): void => {
      clearTimeout(timer)
      opts.signal?.removeEventListener('abort', onAbort)
      resolve({ exitCode, output: cleanOutput(output + extra), timedOut, stopped })
    }
    child.on('error', (e) => finish(null, `\n${e.message}`))
    child.on('close', (code) => finish(code))
  })
}

/**
 * Advice for known stumbling blocks in command output: npm 11 skipping install scripts until they
 * are approved (Electron then never downloads itself), and files locked by a program still running.
 */
export function commandHints(output: string, background: { id: string; command: string }[] = []): string {
  const hints: string[] = []
  const skipped = [...new Set([...output.matchAll(/allow-scripts\s+(@?[^\s@]+)@\S+\s+\((?:pre|post)?install\b/g)].map((m) => m[1]))]
  if (skipped.length) {
    hints.push(
      `npm skipped the install scripts of: ${skipped.join(', ')} (npm on this computer runs them only once approved). ` +
        `If the app needs them (Electron does: its script downloads Electron itself), run: ` +
        `npm install-scripts approve ${skipped.join(' ')}; npm rebuild ${skipped.join(' ')}`
    )
  }
  if (!skipped.length && /Electron failed to install correctly/i.test(output)) {
    hints.push(
      "Electron's own download never ran (npm on this computer skips install scripts until they are approved). " +
        'Fix it with: npm install-scripts approve electron; npm rebuild electron'
    )
  }
  if (/\b(EBUSY|EPERM)\b|being used by another process|resource busy or locked/i.test(output) && background.length) {
    hints.push(
      `Some files are locked, probably by a program you started that is still running (${background.map((b) => `${b.id}: ${b.command}`).join('; ')}). ` +
        'Stop it with stop_background_command, then try again.'
    )
  }
  return hints.length ? `\n\nNote: ${hints.join('\n\nNote: ')}` : ''
}

/** Plain report of a finished command for the model. */
export function commandReport(r: CommandResult, timeoutMs: number): { text: string; isError: boolean } {
  const out = headTail(r.output.trim()) || '(no output)'
  if (r.stopped) return { text: `The owner stopped this command.\nOutput so far:\n${out}`, isError: true }
  if (r.timedOut) {
    const mins = Math.round(timeoutMs / 60000)
    const limit = mins >= 1 ? `${mins} minute${mins === 1 ? '' : 's'}` : `${Math.round(timeoutMs / 1000)} seconds`
    return {
      text:
        `The command was stopped after ${limit} because it was still running. If it starts an app or a server that keeps running, ` +
        `use start_background_command instead.\nOutput so far:\n${out}`,
      isError: true
    }
  }
  return { text: `Exit code: ${r.exitCode ?? 'unknown'}\n${out}`, isError: r.exitCode !== 0 }
}

interface BackgroundProc {
  id: string
  command: string
  child: ChildProcess
  output: string
  /** How much of `output` the model has already seen. */
  seen: number
  exitCode: number | null
  running: boolean
}

/** Long-running commands (apps, dev servers) the agent starts and checks on later. */
export class BackgroundCommands {
  private procs = new Map<string, BackgroundProc>()
  private counter = 0

  constructor(private readonly start: typeof startShell = startShell) {}

  run(command: string, cwd: string, onOutput?: (chunk: string) => void): string {
    const id = `bg${++this.counter}`
    const child = this.start(command, cwd)
    const proc: BackgroundProc = { id, command, child, output: '', seen: 0, exitCode: null, running: true }
    const take = (decoder: InstanceType<typeof TextDecoder>) => (buf: Buffer) => {
      const chunk = decoder.decode(buf, { stream: true })
      proc.output += chunk
      if (proc.output.length > 200_000) {
        const cut = proc.output.length - 100_000
        proc.output = proc.output.slice(cut)
        proc.seen = Math.max(0, proc.seen - cut)
      }
      onOutput?.(chunk)
    }
    child.stdout?.on('data', take(new TextDecoder()))
    child.stderr?.on('data', take(new TextDecoder()))
    child.on('error', (e) => {
      proc.output += `\n${e.message}`
      proc.running = false
    })
    child.on('close', (code) => {
      proc.exitCode = code
      proc.running = false
    })
    this.procs.set(id, proc)
    return id
  }

  has(id: string): boolean {
    return this.procs.has(id)
  }

  /** Waits up to `waitMs` (less if it exits), then reports what is new since the last check. */
  async check(id: string, waitMs: number, signal?: AbortSignal): Promise<{ text: string; isError: boolean }> {
    const proc = this.procs.get(id)
    if (!proc) return { text: `There is no background command "${id}". Ids look like "bg1".`, isError: true }
    const until = Date.now() + waitMs
    while (proc.running && Date.now() < until && !signal?.aborted) await new Promise((r) => setTimeout(r, 200))
    const fresh = cleanOutput(proc.output.slice(proc.seen)).trim()
    proc.seen = proc.output.length
    const state = proc.running ? 'Still running.' : `It has stopped (exit code ${proc.exitCode ?? 'unknown'}).`
    return { text: `${state}\nNew output:\n${headTail(fresh) || '(nothing new)'}`, isError: !proc.running && proc.exitCode !== 0 }
  }

  async stop(id: string): Promise<{ text: string; isError: boolean }> {
    const proc = this.procs.get(id)
    if (!proc) return { text: `There is no background command "${id}".`, isError: true }
    this.procs.delete(id)
    if (proc.running) await killTree(proc.child)
    return { text: `Stopped ${id} (${proc.command}).`, isError: false }
  }

  async stopAll(): Promise<void> {
    await Promise.all([...this.procs.keys()].map((id) => this.stop(id)))
  }

  running(): { id: string; command: string }[] {
    return [...this.procs.values()].filter((p) => p.running).map((p) => ({ id: p.id, command: p.command }))
  }
}
