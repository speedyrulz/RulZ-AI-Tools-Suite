import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { existsSync } from 'node:fs'
import { isAbsolute, join } from 'node:path'

/** Quotes one argument the way CommandLineToArgvW (and so most Windows programs) reads it back. */
export function quoteArg(arg: string): string {
  if (arg && !/[\s"]/.test(arg)) return arg
  return `"${arg.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\+)$/, '$1$1')}"`
}

function isBatch(path: string): boolean {
  return /\.(bat|cmd)$/i.test(path)
}

/** A program path from the catalog: relative paths (or bare names of files in `cwd`) resolve in `cwd`. */
export function resolveProgram(program: string, cwd: string): string {
  if (isAbsolute(program)) return program
  if (/[\\/]/.test(program) || existsSync(join(cwd, program))) return join(cwd, program)
  return program
}

/** spawn() arguments that run `program` (a .bat/.cmd goes through cmd.exe, with its quoting rules). */
export function spawnArgs(program: string, args: string[]): { file: string; args: string[]; verbatim: boolean } {
  if (!isBatch(program)) return { file: program, args, verbatim: false }
  // cmd /s /c "<line>" strips the outer quotes and runs <line> as typed.
  const line = [program, ...args].map(quoteArg).join(' ')
  return { file: process.env.ComSpec || 'cmd.exe', args: ['/d', '/s', '/c', `"${line}"`], verbatim: true }
}

export interface RunOptions {
  cwd: string
  env?: NodeJS.ProcessEnv
  signal?: AbortSignal
  onLine?: (line: string) => void
  /**
   * A setup program or uninstaller: arguments are passed exactly as given (NSIS reads /D="C: b" itself) and
   * output isn't captured (electron-builder's NSIS setup crashes when its output is redirected).
   */
  gui?: boolean
}

/** Runs a command to completion, streaming its output line by line. Resolves with the exit code. */
export function runCommand(run: string[], opts: RunOptions): Promise<number> {
  const program = resolveProgram(run[0], opts.cwd)
  const { file, args, verbatim } = opts.gui
    ? { file: program, args: run.slice(1), verbatim: true }
    : spawnArgs(program, run.slice(1))
  return new Promise((resolve, reject) => {
    if (opts.signal?.aborted) return reject(new Error('Cancelled'))
    let child: ChildProcess
    try {
      child = spawn(file, args, {
        cwd: opts.cwd,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', ...opts.env },
        stdio: opts.gui ? 'ignore' : ['ignore', 'pipe', 'pipe'],
        windowsHide: !opts.gui,
        windowsVerbatimArguments: verbatim,
        // Verbatim arguments include argv[0], so the program path needs its quotes here.
        ...(opts.gui ? { argv0: quoteArg(file) } : {})
      })
    } catch (e) {
      return reject(e)
    }
    const onAbort = (): void => killTree(child.pid)
    opts.signal?.addEventListener('abort', onAbort, { once: true })

    const splitter = (): ((chunk: Buffer) => void) => {
      let rest = ''
      return (chunk) => {
        const parts = (rest + chunk.toString('utf8')).split(/\r\n|\r|\n/)
        rest = parts.pop() ?? ''
        for (const p of parts) if (p.trim()) opts.onLine?.(p)
      }
    }
    child.stdout?.on('data', splitter())
    child.stderr?.on('data', splitter())
    child.on('error', (e) => {
      opts.signal?.removeEventListener('abort', onAbort)
      reject(new Error(`Couldn't run ${run[0]}: ${e.message}`))
    })
    child.on('close', (code) => {
      opts.signal?.removeEventListener('abort', onAbort)
      if (opts.signal?.aborted) reject(new Error('Cancelled'))
      else resolve(code ?? 1)
    })
  })
}

/** Runs a short command and returns its trimmed stdout, or null if it fails or isn't found. */
export function capture(file: string, args: string[], timeoutMs = 8000): Promise<string | null> {
  return new Promise((resolve) => {
    execFile(file, args, { timeout: timeoutMs, windowsHide: true }, (err, stdout, stderr) => {
      if (err) resolve(null)
      else resolve((stdout || stderr).toString().trim())
    })
  })
}

export function killTree(pid: number | undefined): void {
  if (!pid) return
  execFile('taskkill', ['/pid', String(pid), '/t', '/f'], { windowsHide: true }, () => undefined)
}

/** Starts an app on its own (it keeps running after the launcher closes). Batch files get a console window. */
export function launchDetached(target: string, args: string[], cwd: string): ChildProcess {
  const { file, args: argv, verbatim } = spawnArgs(target, args)
  const child = spawn(file, argv, {
    cwd,
    detached: true,
    stdio: 'ignore',
    windowsHide: false,
    windowsVerbatimArguments: verbatim
  })
  child.unref()
  return child
}

async function regPath(key: string): Promise<string> {
  const out = await capture('reg', ['query', key, '/v', 'Path'])
  const m = out && /Path\s+REG_(?:EXPAND_)?SZ\s+(.*)$/im.exec(out)
  if (!m) return ''
  return m[1].replace(/%([^%]+)%/g, (whole, name: string) => process.env[name] ?? whole)
}

/**
 * Reloads PATH from the registry, so tools installed while the launcher runs (winget, Python's installer)
 * are found without a restart. Entries only present in the current PATH are kept.
 */
export async function refreshPath(): Promise<void> {
  const machine = await regPath('HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment')
  const user = await regPath('HKCU\\Environment')
  const seen = new Set<string>()
  const merged: string[] = []
  for (const entry of [machine, user, process.env.PATH ?? ''].join(';').split(';')) {
    const e = entry.trim()
    if (!e || seen.has(e.toLowerCase())) continue
    seen.add(e.toLowerCase())
    merged.push(e)
  }
  process.env.PATH = merged.join(';')
}
