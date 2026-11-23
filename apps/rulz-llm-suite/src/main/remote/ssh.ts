// ssh to a rental (docs/vast-design.md, section 4): Windows OpenSSH, scripts as `bash -s` on stdin, one known_hosts
// file per rental, routes tried in order. Nothing here uses a shell, and no value of ours is ever on a command line
// except host, port and fixed options (all validated).
import { spawn as nodeSpawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
import { existsSync, readFileSync, mkdirSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import type { Readable } from 'node:stream'
import { REMOTE_SLUG } from '@shared/remote'
import type { RentalDirectory, RentalInfo, SshRoute } from './contract'

// ---------- Quoting and validation ----------

/** POSIX single-quote: the result is one shell word whatever the value holds. Every interpolated value goes through this. */
export function shq(value: string | number): string {
  const s = typeof value === 'number' ? numberText(value) : value
  if (s.includes('\0')) throw new Error('A value with a NUL character cannot be sent to the instance.')
  return `'${s.replace(/'/g, `'\\''`)}'`
}

function numberText(n: number): string {
  if (!Number.isSafeInteger(n)) throw new Error(`Not a whole number: ${n}`)
  return String(n)
}

export const TAG_RE = /^b\d{3,7}$/
/** Instance ids of InstanceManager are uuids; the run folder is named by them. */
export const INSTANCE_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/
/** A file name inside a model folder: flat, no leading dot, no `.part` ending (that is the resume file). */
export const FILE_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}$/
export const SHA256_RE = /^[0-9a-f]{64}$/

export function assertSlug(s: string): string {
  if (!REMOTE_SLUG.test(s)) throw new Error(`Not a valid model folder name: ${s.slice(0, 60)}`)
  return s
}
export function assertTag(s: string): string {
  if (!TAG_RE.test(s)) throw new Error(`Not a valid llama.cpp build tag: ${s.slice(0, 40)}`)
  return s
}
export function assertInstanceId(s: string): string {
  if (!INSTANCE_ID_RE.test(s)) throw new Error('Not a valid instance id.')
  return s
}
export function assertFileName(s: string): string {
  if (!FILE_NAME_RE.test(s) || s.endsWith('.part') || s.includes('..')) throw new Error(`Not a valid file name: ${s.slice(0, 80)}`)
  return s
}
export function assertSha256(s: string): string {
  if (!SHA256_RE.test(s)) throw new Error('Not a sha256 digest.')
  return s
}

// ---------- ssh program and command line ----------

export class SshError extends Error {
  constructor(
    readonly kind: 'auth' | 'hostkey' | 'unreachable' | 'transport' | 'timeout' | 'aborted' | 'missing-ssh' | 'not-running',
    message: string,
    readonly detail = ''
  ) {
    super(message)
    this.name = 'SshError'
  }
}

export interface FindOptions {
  env?: NodeJS.ProcessEnv
  exists?: (path: string) => boolean
  platform?: NodeJS.Platform
}

/** The ssh program: `ssh` on PATH, else Windows' own OpenSSH folder. */
export function findSsh({ env = process.env, exists = existsSync, platform = process.platform }: FindOptions = {}): string {
  const win = platform === 'win32'
  const sep = win ? ';' : ':'
  const pathKey = Object.keys(env).find((k) => k.toLowerCase() === 'path') ?? 'PATH'
  // Not ssh.cmd / ssh.bat: cmd.exe would not pass the bytes on stdin through, and a kill would only end the shim.
  const names = win ? ['ssh.exe'] : ['ssh']
  for (const dir of (env[pathKey] ?? '').split(sep).map((d) => d.trim().replace(/^"|"$/g, '')).filter(Boolean)) {
    for (const n of names) {
      const full = win ? `${dir.replace(/[\\/]+$/, '')}\\${n}` : `${dir.replace(/\/+$/, '')}/${n}`
      if (exists(full)) return full
    }
  }
  if (win) {
    const fallback = `${env.SystemRoot ?? env.SYSTEMROOT ?? 'C:\\Windows'}\\System32\\OpenSSH\\ssh.exe`
    if (exists(fallback)) return fallback
  }
  throw new SshError('missing-ssh', 'No ssh program was found. On Windows: Settings, System, Optional features, add OpenSSH Client.')
}

const toForward = (p: string): string => p.replace(/\\/g, '/')

/** An ssh_config value: quoted when it holds a space (ssh's parser strips the quotes). */
function optionValue(v: string): string {
  if (v.includes('"')) throw new Error('A quote in an ssh option value is not supported.')
  return /\s/.test(v) ? `"${v}"` : v
}

const HOST_RE = /^[A-Za-z0-9][A-Za-z0-9.:-]{0,252}$/

export interface SshArgOptions {
  route: SshRoute
  /** The rental's own known_hosts file. */
  knownHostsFile: string
  /** `ssh -N -L`: a tunnel to a port on the instance. */
  tunnel?: { localPort: number; remotePort: number }
  /** The remote command; without one and without a tunnel, ssh would start a login shell, which is never wanted. */
  command?: string
}

/** Arguments for ssh (not including the program). Host and port are checked so a route can never inject an option. */
export function buildSshArgs({ route, knownHostsFile, tunnel, command }: SshArgOptions): string[] {
  if (!HOST_RE.test(route.host)) throw new Error('The ssh host of this rental does not look like a host name.')
  if (!Number.isInteger(route.port) || route.port < 1 || route.port > 65535) throw new Error('The ssh port of this rental is not valid.')
  const args = [
    '-p', String(route.port),
    '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3',
    '-o', 'LogLevel=ERROR',
    // A rental is new to this PC the first time; a changed key later is reported, not accepted.
    '-o', 'StrictHostKeyChecking=accept-new',
    // Never ask anything: a passphrase-protected key works only through the agent.
    '-o', 'BatchMode=yes',
    '-o', `UserKnownHostsFile=${optionValue(toForward(knownHostsFile))}`
  ] // prettier-ignore
  if (tunnel) {
    if (!Number.isInteger(tunnel.localPort) || !Number.isInteger(tunnel.remotePort)) throw new Error('Tunnel ports must be whole numbers.')
    args.push('-N', '-T', '-o', 'ExitOnForwardFailure=yes', '-L', `127.0.0.1:${tunnel.localPort}:127.0.0.1:${tunnel.remotePort}`)
  } else {
    args.push('-T')
  }
  args.push(`root@${route.host}`)
  if (command) args.push(command)
  return args
}

/** The remote command for scripts: a clean bash that reads the script from stdin (no profile: vast images start tmux in .bashrc). */
export const REMOTE_COMMAND = 'bash --noprofile --norc -s'

/**
 * The script as bash must see it: a function, then one call on a single line. Bash reads a piped script one line at a
 * time, so whatever follows that line (a curl config, file bytes) is still unread when the script runs and `cat` or
 * `read` inside it gets exactly that data, not the rest of the script.
 */
export function frameScript(script: string): string {
  return `__rz_main() {\n${script.replace(/\r\n/g, '\n')}\n}\n__rz_main; exit $?\n`
}

// ---------- Errors in sentences ----------

export function describeSshFailure(stderr: string, title: string): SshError {
  const text = stderr.trim()
  const low = text.toLowerCase()
  const last = text.split(/\r?\n/).filter(Boolean).pop() ?? ''
  if (/host key verification failed|remote host identification has changed|host key for .* has changed/.test(low)) {
    return new SshError(
      'hostkey',
      `${title} has a different ssh host key than before. A rental that was stopped and started again can come back on another machine with a new key. If you started it yourself, choose Forget host key and try again.`,
      last
    )
  }
  if (low.includes('permission denied')) {
    return new SshError(
      'auth',
      `${title} refused this PC's ssh key. Add your public key under Keys on cloud.vast.ai (an instance rented before that needs the key added to the instance too). A key with a passphrase works only when ssh-agent holds it.`,
      last
    )
  }
  if (/connection refused|timed out|no route to host|network is unreachable|could not resolve|connection closed|connection reset|kex_exchange_identification|banner exchange|broken pipe/.test(low)) {
    return new SshError('unreachable', `${title} does not answer yet. It may still be starting; try again in a minute. (${last || 'no answer'})`, last)
  }
  return new SshError('transport', last ? `ssh to ${title} failed: ${last}` : `ssh to ${title} failed.`, last)
}

// ---------- Running ----------

export type SpawnFn = (exe: string, args: string[], options: SpawnOptions) => ChildProcess

export interface RunOptions {
  /** Data that follows the script on stdin: a string, bytes, or a stream (not retried on another route). */
  stdin?: string | Buffer | Readable
  /** 0 = none. Default 60 s. */
  timeoutMs?: number
  onLine?: (line: string) => void
  signal?: AbortSignal
}

export interface RunResult {
  code: number
  stdout: string
  stderr: string
}

/** What the rest of the ssh side needs of a runner (tests use a fake). */
export interface ScriptRunner {
  run(rentalId: number, script: string, opts?: RunOptions): Promise<RunResult>
}

/** A process with a held stdin (the keepalive loop). */
export interface SshProcess {
  write(text: string): void
  end(): void
  kill(): void
  /** Never rejects. */
  readonly done: Promise<RunResult & { error?: SshError }>
}

export interface TunnelProc {
  kill(): void
  /** Never rejects. */
  readonly exited: Promise<{ code: number | null; stderr: string }>
}

/** What remoteServer needs: scripts, tunnels, and a way to say a route failed. */
export interface RemoteSsh extends ScriptRunner {
  tunnel(rentalId: number, localPort: number, remotePort: number): TunnelProc
  routeFailed(rentalId: number): void
}

const MAX_OUTPUT = 2 * 1024 * 1024
/** Longest line handed to onLine; an instance that sends more without a newline is cut off. */
const MAX_LINE = 64 * 1024

export interface SshRunnerDeps {
  directory: RentalDirectory
  /** Folder for the known_hosts files. */
  knownHostsDir: string
  spawn?: SpawnFn
  /** The ssh program; default `findSsh()`, looked up on first use. */
  sshPath?: () => string
}

const routeKey = (r: SshRoute): string => `${r.route}|${r.host}|${r.port}`

export class SshRunner implements RemoteSsh {
  private readonly directory: RentalDirectory
  private readonly spawnFn: SpawnFn
  private readonly sshPath: () => string
  private exe: string | null = null
  private readonly chosen = new Map<number, string>()
  private readonly avoid = new Map<number, string>()

  constructor(private readonly deps: SshRunnerDeps) {
    this.directory = deps.directory
    this.spawnFn = deps.spawn ?? ((exe, args, options) => nodeSpawn(exe, args, options))
    this.sshPath = deps.sshPath ?? (() => findSsh())
  }

  /** The known_hosts file of a rental. */
  knownHostsFile(rentalId: number): string {
    if (!Number.isSafeInteger(rentalId) || rentalId < 0) throw new Error('Not a rental id.')
    return join(this.deps.knownHostsDir, `known_hosts_${rentalId}`)
  }

  /** Drops the remembered host key (a rental restarted on another machine can have a new one). */
  forgetHostKey(rentalId: number): void {
    rmSync(this.knownHostsFile(rentalId), { force: true })
    rmSync(`${this.knownHostsFile(rentalId)}.old`, { force: true })
  }

  /** The route that worked last, if any (for the window). */
  currentRoute(rentalId: number): string | null {
    return this.chosen.get(rentalId) ?? null
  }

  routeFailed(rentalId: number): void {
    const cur = this.chosen.get(rentalId)
    if (cur) this.avoid.set(rentalId, cur)
    this.chosen.delete(rentalId)
  }

  private program(): string {
    this.exe ??= this.sshPath()
    return this.exe
  }

  private ordered(rentalId: number, routes: SshRoute[]): SshRoute[] {
    const first = this.chosen.get(rentalId)
    const bad = this.avoid.get(rentalId)
    const list = [...routes]
    const rank = (r: SshRoute): number => (routeKey(r) === first ? 0 : routeKey(r) === bad ? 2 : 1)
    return list.sort((a, b) => rank(a) - rank(b))
  }

  private info(rentalId: number): RentalInfo {
    const info = this.directory.info(rentalId)
    if (!info) throw new SshError('not-running', 'This rental is not in the list any more.')
    if (!info.running) throw new SshError('not-running', `${info.label} is not running. Start it first.`)
    return info
  }

  private spawnSsh(info: RentalInfo, route: SshRoute, opts: { tunnel?: { localPort: number; remotePort: number }; command?: string }): ChildProcess {
    const args = buildSshArgs({ route, knownHostsFile: this.knownHostsFile(info.rentalId), tunnel: opts.tunnel, command: opts.command })
    mkdirSync(this.deps.knownHostsDir, { recursive: true })
    return this.spawnFn(this.program(), args, { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
  }

  /** Runs a script on the rental, trying the routes in turn (a stream on stdin gets one route). */
  async run(rentalId: number, script: string, opts: RunOptions = {}): Promise<RunResult> {
    let info = this.info(rentalId)
    const stream = !!opts.stdin && typeof opts.stdin !== 'string' && !Buffer.isBuffer(opts.stdin)
    let last: SshError | null = null
    for (let round = 0; round < 2; round++) {
      const routes = this.ordered(rentalId, info.routes)
      if (!routes.length) throw new SshError('unreachable', `${info.label} has no ssh address yet; it may still be starting.`)
      for (const route of stream ? routes.slice(0, 1) : routes) {
        try {
          const res = await this.runOn(info, route, script, opts)
          this.chosen.set(rentalId, routeKey(route))
          this.avoid.delete(rentalId)
          return res
        } catch (err) {
          if (!(err instanceof SshError) || (err.kind !== 'unreachable' && err.kind !== 'transport')) throw err
          last = err
          this.directory.note(rentalId, `ssh via ${route.route} failed: ${err.detail || err.message}`)
        }
      }
      if (stream) break
      // Every route failed: the rental may have been restarted with new ports, so ask again once.
      const fresh = this.directory.info(rentalId)
      if (!fresh?.running || JSON.stringify(fresh.routes) === JSON.stringify(info.routes)) break
      info = fresh
      this.chosen.delete(rentalId)
      this.avoid.delete(rentalId)
    }
    throw last ?? new SshError('unreachable', `${info.label} does not answer.`)
  }

  private runOn(info: RentalInfo, route: SshRoute, script: string, opts: RunOptions): Promise<RunResult> {
    return new Promise<RunResult>((resolve, reject) => {
      let child: ChildProcess
      try {
        child = this.spawnSsh(info, route, { command: REMOTE_COMMAND })
      } catch (err) {
        reject(err instanceof SshError ? err : new SshError('transport', (err as Error).message))
        return
      }
      let stdout = ''
      let stderr = ''
      let pending = ''
      let settled = false
      let timer: NodeJS.Timeout | undefined
      const finish = (fn: () => void): void => {
        if (settled) return
        settled = true
        if (timer) clearTimeout(timer)
        opts.signal?.removeEventListener('abort', onAbort)
        fn()
      }
      const kill = (): void => {
        try {
          child.kill()
        } catch {
          // already gone
        }
      }
      const onAbort = (): void => {
        kill()
        finish(() => reject(new SshError('aborted', 'Cancelled.')))
      }
      if (opts.signal?.aborted) {
        kill()
        reject(new SshError('aborted', 'Cancelled.'))
        return
      }
      opts.signal?.addEventListener('abort', onAbort, { once: true })
      const timeoutMs = opts.timeoutMs ?? 60_000
      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          kill()
          finish(() => reject(new SshError('timeout', `${info.label} did not finish in time.`)))
        }, timeoutMs)
      }
      child.stdout?.on('data', (d: Buffer | string) => {
        const text = d.toString()
        if (stdout.length < MAX_OUTPUT) stdout += text
        if (opts.onLine) {
          // Only the new text is searched for newlines, and an endless line is cut, so a hostile instance cannot fill the memory.
          const parts = text.split('\n')
          for (let i = 0; i < parts.length; i++) {
            if (pending.length < MAX_LINE) pending += parts[i].slice(0, MAX_LINE - pending.length)
            if (i === parts.length - 1) break
            const line = pending.replace(/\r$/, '')
            pending = ''
            try {
              opts.onLine(line)
            } catch {
              // a bad listener must not break the run
            }
          }
        }
      })
      child.stderr?.on('data', (d: Buffer | string) => {
        if (stderr.length < 64 * 1024) stderr += d.toString()
      })
      child.on('error', (err: NodeJS.ErrnoException) => {
        finish(() =>
          reject(err.code === 'ENOENT' ? new SshError('missing-ssh', 'No ssh program was found. On Windows: Settings, System, Optional features, add OpenSSH Client.') : new SshError('transport', `Could not start ssh: ${err.message}`))
        )
      })
      child.on('close', (code: number | null) => {
        if (pending && opts.onLine) {
          try {
            opts.onLine(pending)
          } catch {
            // ignore
          }
        }
        finish(() => {
          if (code === 255 || code === null) reject(describeSshFailure(stderr, info.label))
          else resolve({ code, stdout, stderr })
        })
      })
      const stdin = child.stdin
      if (!stdin) {
        kill()
        finish(() => reject(new SshError('transport', 'ssh has no input stream.')))
        return
      }
      // A closed pipe (ssh ended first) is reported by the exit code, not here.
      stdin.on('error', () => undefined)
      const framed = frameScript(script)
      const data = opts.stdin
      if (data === undefined || typeof data === 'string' || Buffer.isBuffer(data)) {
        stdin.end(typeof data === 'string' ? framed + data : data === undefined ? framed : Buffer.concat([Buffer.from(framed), data]))
      } else {
        stdin.write(framed)
        data.on('error', () => {
          kill()
          finish(() => reject(new SshError('transport', 'Reading the file to send failed.')))
        })
        data.pipe(stdin)
      }
    })
  }

  /** A process whose stdin stays open (route: the one that works, else the first). It does not retry. */
  open(rentalId: number, script: string): SshProcess {
    const info = this.info(rentalId)
    const route = this.ordered(rentalId, info.routes)[0]
    if (!route) throw new SshError('unreachable', `${info.label} has no ssh address yet; it may still be starting.`)
    const child = this.spawnSsh(info, route, { command: REMOTE_COMMAND })
    let stdout = ''
    let stderr = ''
    child.stdout?.on('data', (d: Buffer | string) => {
      if (stdout.length < 64 * 1024) stdout += d.toString()
    })
    child.stderr?.on('data', (d: Buffer | string) => {
      if (stderr.length < 64 * 1024) stderr += d.toString()
    })
    child.stdin?.on('error', () => undefined)
    child.stdin?.write(frameScript(script))
    const done = new Promise<RunResult & { error?: SshError }>((resolve) => {
      child.on('error', (err) => resolve({ code: -1, stdout, stderr, error: new SshError('transport', err.message) }))
      child.on('close', (code) => {
        const error = code === 255 || code === null ? describeSshFailure(stderr, info.label) : undefined
        resolve({ code: code ?? -1, stdout, stderr, error })
      })
    })
    return {
      write: (text) => {
        try {
          child.stdin?.write(text)
        } catch {
          // closed
        }
      },
      end: () => {
        try {
          child.stdin?.end()
        } catch {
          // closed
        }
      },
      kill: () => {
        try {
          child.kill()
        } catch {
          // gone
        }
      },
      done
    }
  }

  /** `ssh -N -L` to a port on the instance. Throws when the rental has no usable address now. */
  tunnel(rentalId: number, localPort: number, remotePort: number): TunnelProc {
    const info = this.info(rentalId)
    const route = this.ordered(rentalId, info.routes)[0]
    if (!route) throw new SshError('unreachable', `${info.label} has no ssh address yet; it may still be starting.`)
    const child = this.spawnSsh(info, route, { tunnel: { localPort, remotePort } })
    let stderr = ''
    child.stderr?.on('data', (d: Buffer | string) => {
      if (stderr.length < 16 * 1024) stderr += d.toString()
    })
    child.stdout?.on('data', () => undefined)
    const exited = new Promise<{ code: number | null; stderr: string }>((resolve) => {
      child.on('error', (err) => resolve({ code: null, stderr: stderr || err.message }))
      child.on('close', (code) => resolve({ code, stderr }))
    })
    return {
      kill: () => {
        try {
          child.kill()
        } catch {
          // gone
        }
      },
      exited
    }
  }
}

// ---------- The user's key ----------

export interface SshKeyInfo {
  available: boolean
  publicKeyPath: string | null
  publicKey: string
  error?: string
}

const KEY_NAMES = ['id_ed25519', 'id_ecdsa', 'id_ecdsa_sk', 'id_ed25519_sk', 'id_rsa', 'id_dsa']

/** The first public key of ~/.ssh in OpenSSH's own order of preference. */
export function sshKeyInfo(homedir: string): SshKeyInfo {
  const dir = join(homedir, '.ssh')
  for (const n of KEY_NAMES) {
    const path = join(dir, `${n}.pub`)
    if (!existsSync(path)) continue
    try {
      const text = readFileSync(path, 'utf8').trim().split(/\r?\n/)[0]?.trim() ?? ''
      if (/^(ssh-|ecdsa-|sk-)\S+ \S+/.test(text)) return { available: true, publicKeyPath: toForward(path), publicKey: text }
      return { available: false, publicKeyPath: toForward(path), publicKey: '', error: `${path} does not look like a public key.` }
    } catch (err) {
      return { available: false, publicKeyPath: toForward(path), publicKey: '', error: `Could not read ${path}: ${(err as Error).message}` }
    }
  }
  return {
    available: false,
    publicKeyPath: null,
    publicKey: '',
    error: 'No ssh key found. Make one by running ssh-keygen -t ed25519 in a terminal, then add the public key under Keys on cloud.vast.ai.'
  }
}

/** Parses the `@rz KEY=VALUE` marker lines a script prints; other output (banners, noise) is ignored. */
export function parseMarkers(out: string): Array<[string, string]> {
  const found: Array<[string, string]> = []
  for (const line of out.split(/\r?\n/)) {
    const m = /^@rz ([A-Z][A-Z0-9_]*)=(.*)$/.exec(line)
    if (m) found.push([m[1], m[2]])
  }
  return found
}

/** The last value of a marker, or undefined. */
export function marker(out: string, key: string): string | undefined {
  let v: string | undefined
  for (const [k, val] of parseMarkers(out)) if (k === key) v = val
  return v
}

export function markers(out: string, key: string): string[] {
  return parseMarkers(out)
    .filter(([k]) => k === key)
    .map(([, v]) => v)
}
