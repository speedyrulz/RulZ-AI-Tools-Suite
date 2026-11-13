import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { existsSync } from 'node:fs'
import { isAbsolute, join } from 'node:path'
import { StringDecoder } from 'node:string_decoder'

/** Quotes one argument the way CommandLineToArgvW (and so most Windows programs) reads it back. */
export function quoteArg(arg: string): string {
  if (arg && !/[\s"]/.test(arg)) return arg
  return `"${arg.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\+)$/, '$1$1')}"`
}

/**
 * Quotes one argument for a command line that cmd.exe reads first (a .bat or .cmd): besides spaces, its own special
 * characters and separators need quotes, or "D:\R&D\run.bat" runs "D:\R" and then "D\run.bat", and "D:\a,b\run.bat"
 * runs "D:\a". (A %NAME% is replaced even inside quotes; installFolder refuses folders with one.)
 */
export function quoteCmdArg(arg: string): string {
  return /[&|<>^(),;=]/.test(arg) && !/[\s"]/.test(arg) ? `"${arg.replace(/(\\+)$/, '$1$1')}"` : quoteArg(arg)
}

function isBatch(path: string): boolean {
  return /\.(bat|cmd)$/i.test(path)
}

/** A program path from the catalog: relative paths (or bare names of files in `cwd`) resolve in `cwd`. */
export function resolveProgram(program: string, cwd: string): string {
  if (isAbsolute(program)) return program
  // Only a program file counts: a bare "winget" must not match a folder of that name in cwd.
  if (/[\\/]/.test(program) || (/\.(exe|bat|cmd)$/i.test(program) && existsSync(join(cwd, program)))) {
    return join(cwd, program)
  }
  return program
}

/** spawn() arguments that run `program` (a .bat/.cmd goes through cmd.exe, with its quoting rules). */
export function spawnArgs(program: string, args: string[]): { file: string; args: string[]; verbatim: boolean } {
  if (!isBatch(program)) return { file: program, args, verbatim: false }
  // cmd /s /c "<line>" strips the outer quotes and runs <line> as typed.
  const line = [program, ...args].map(quoteCmdArg).join(' ')
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

/**
 * Turns a stream's chunks into lines (\n, \r\n, or a lone \r as progress bars use), blank ones skipped. The decoder
 * keeps a character whose bytes are split across two chunks whole; `end` passes on the line left without a break.
 */
export function lineSplitter(onLine?: (line: string) => void): { write: (chunk: Buffer) => void; end: () => void } {
  const decoder = new StringDecoder('utf8')
  let rest = ''
  const take = (text: string, last: boolean): void => {
    const parts = (rest + text).split(/\r\n|\r|\n/)
    rest = last ? '' : (parts.pop() ?? '')
    for (const p of parts) if (p.trim()) onLine?.(p)
  }
  return { write: (chunk) => take(decoder.write(chunk), false), end: () => take(decoder.end(), true) }
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

    const out = lineSplitter(opts.onLine)
    const err = lineSplitter(opts.onLine)
    child.stdout?.on('data', out.write)
    child.stderr?.on('data', err.write)
    child.on('error', (e) => {
      opts.signal?.removeEventListener('abort', onAbort)
      reject(new Error(`Couldn't run ${run[0]}: ${e.message}`))
    })
    child.on('close', (code) => {
      // The last line often has no line break (a final error message); it counts too.
      out.end()
      err.end()
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

/**
 * Starts an app on its own (it keeps running after the launcher closes). Batch files get a console window. `env` is
 * added to the launcher's environment.
 */
export function launchDetached(target: string, args: string[], cwd: string, env?: Record<string, string>): ChildProcess {
  const { file, args: argv, verbatim } = spawnArgs(target, args)
  const child = spawn(file, argv, {
    cwd,
    env: env && Object.keys(env).length ? { ...process.env, ...env } : process.env,
    detached: true,
    stdio: 'ignore',
    windowsHide: false,
    windowsVerbatimArguments: verbatim
  })
  child.unref()
  return child
}

/**
 * Runs a PowerShell script and returns its trimmed output (null when it fails). The output is UTF-8: by default
 * PowerShell writes the console's code page, which turns "C:\Users\José" into "C:\Users\Jos?" (reg.exe does too).
 */
export function powershell(script: string, timeoutMs = 20000): Promise<string | null> {
  return capture(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-Command', `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; ${script}`],
    timeoutMs
  )
}

/** A single-quoted PowerShell string. */
export function psString(text: string): string {
  return `'${text.replace(/'/g, "''")}'`
}

/** Separates the machine and user PATH in refreshPath's PowerShell output. */
const PATH_SPLIT = '<<rulz-user-path>>'

/**
 * Reloads PATH from the registry, so tools installed while the launcher runs (winget, Python's installer)
 * are found without a restart. Entries only present in the current PATH are kept.
 */
export async function refreshPath(): Promise<void> {
  // GetValue expands %VARIABLES% in REG_EXPAND_SZ values, with this process's environment.
  const out = await powershell(
    "$m = (Get-Item 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment').GetValue('Path'); " +
      "$k = Get-Item 'HKCU:\\Environment' -ErrorAction SilentlyContinue; " +
      "$u = if ($k) { $k.GetValue('Path') } else { '' }; " +
      `"$m${PATH_SPLIT}$u"`
  )
  if (!out) return
  const [machine = '', user = ''] = out.split(PATH_SPLIT)
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

/**
 * Names of the programs in `listing` (one executable path per line) that live inside `dir`. Pure, for tests.
 */
export function programsInside(listing: string | null, dir: string): string[] {
  if (!listing) return []
  const root = dir.replace(/[\\/]+$/, '').toLowerCase() + '\\'
  const names = new Set<string>()
  for (const raw of listing.split(/\r?\n/)) {
    const path = raw.trim().replace(/\//g, '\\')
    if (path && path.toLowerCase().startsWith(root)) names.add(path.slice(path.lastIndexOf('\\') + 1))
  }
  return [...names].sort()
}

/**
 * Programs running from inside an app's folder, whoever started them: the launcher only knows about the apps it
 * started since it opened, and "close on launch" means it often didn't. Empty when the list can't be read.
 */
export async function programsRunningFrom(dir: string): Promise<string[]> {
  if (process.platform !== 'win32') return []
  const listing = await powershell('Get-CimInstance Win32_Process | ForEach-Object { $_.ExecutablePath }')
  return programsInside(listing, dir)
}
