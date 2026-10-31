import { spawn, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createServer } from 'node:net'
import { join } from 'node:path'

/**
 * Folders holding DLLs a llama.cpp build commonly links against but that are often not on the
 * PATH Windows gives apps started from Explorer: the CUDA runtime and OpenSSL.
 */
function runtimeDllDirs(): string[] {
  const dirs: string[] = []
  const cuda = process.env.CUDA_PATH
  if (cuda) dirs.push(join(cuda, 'bin', 'x64'), join(cuda, 'bin'))
  dirs.push('C:\\OpenSSL-Win64\\bin', 'C:\\Program Files\\OpenSSL-Win64\\bin', 'C:\\Program Files\\OpenSSL\\bin')
  return dirs.filter((d) => existsSync(d))
}

/**
 * Environment for llama.cpp child processes. CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA0/CUDA1
 * match nvidia-smi's GPU numbering (CUDA otherwise orders by speed).
 */
export function engineEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' }
  const extra = runtimeDllDirs()
  if (extra.length) {
    const pathKey = Object.keys(env).find((k) => k.toLowerCase() === 'path') ?? 'PATH'
    env[pathKey] = [...extra, env[pathKey] ?? ''].join(';')
  }
  return env
}

/** Explains Windows loader exit codes, which come with no output at all. */
export function describeExitCode(code: number | null): string | undefined {
  if (code === null) return undefined
  const unsigned = code < 0 ? code + 0x100000000 : code
  switch (unsigned) {
    case 0xc0000135:
      return 'A DLL that llama-server.exe needs could not be found (Windows error 0xC0000135). Make sure the CUDA and OpenSSL DLLs it was built against are installed and on PATH, or copy them next to llama-server.exe.'
    case 0xc000007b:
      return 'llama-server.exe or one of its DLLs is the wrong architecture (Windows error 0xC000007B).'
    case 0xc0000139:
      return 'A DLL loaded by llama-server.exe is the wrong version (Windows error 0xC0000139, entry point not found).'
    case 0xc0000005:
      return 'llama-server.exe crashed (access violation, 0xC0000005).'
    default:
      return undefined
  }
}

export interface CaptureResult {
  code: number | null
  output: string
}

/** Runs a short-lived command and returns combined stdout+stderr. */
export function runCapture(exe: string, args: string[], cwd: string, timeoutMs = 15000): Promise<CaptureResult> {
  return new Promise((resolve) => {
    let output = ''
    let settled = false
    const child = spawn(exe, args, { cwd, env: engineEnv(), windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] })
    const finish = (code: number | null, extra = ''): void => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ code, output: output + extra })
    }
    const timer = setTimeout(() => {
      killTree(child.pid)
      finish(null, '\n[timed out]')
    }, timeoutMs)
    child.stdout.on('data', (d: Buffer) => (output += d.toString()))
    child.stderr.on('data', (d: Buffer) => (output += d.toString()))
    child.on('error', (err) => finish(null, `\n${err.message}`))
    child.on('close', (code) => finish(code))
  })
}

/** Kills a process and all of its children. */
export function killTree(pid: number | undefined | null, sync = false): void {
  if (!pid) return
  if (process.platform === 'win32') {
    const args = ['/pid', String(pid), '/T', '/F']
    if (sync) spawnSync('taskkill', args, { windowsHide: true, stdio: 'ignore' })
    else spawn('taskkill', args, { windowsHide: true, stdio: 'ignore' }).on('error', () => undefined)
    return
  }
  try {
    process.kill(pid, 'SIGTERM')
  } catch {
    // already gone
  }
}

/** PIDs of every running llama-server process, from one process listing (used to clean up after a crash). */
export function llamaServerPids(): Set<number> {
  if (process.platform !== 'win32') return new Set()
  const r = spawnSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], {
    windowsHide: true,
    encoding: 'utf8'
  })
  return parseTasklistPids(r.stdout ?? '')
}

/** Reads the PID column of `tasklist /FO CSV /NH` output for llama-server rows. */
export function parseTasklistPids(csv: string): Set<number> {
  const pids = new Set<number>()
  for (const line of csv.split(/\r?\n/)) {
    const m = /^"([^"]*llama-server[^"]*)","(\d+)"/i.exec(line.trim())
    if (m) pids.add(Number(m[2]))
  }
  return pids
}

/** True while a process with this PID exists. */
export function isPidAlive(pid: number | undefined | null): boolean {
  if (!pid) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    // EPERM: it exists but belongs to someone else.
    return (err as NodeJS.ErrnoException).code === 'EPERM'
  }
}

export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.unref()
    srv.on('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : 0
      srv.close(() => (port ? resolve(port) : reject(new Error('Could not allocate a port'))))
    })
  })
}

/** Splits a command-line string into arguments, honouring double quotes. */
export function splitArgs(input: string): string[] {
  const out: string[] = []
  const re = /"((?:[^"\\]|\\.)*)"|(\S+)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(input))) out.push(m[1] !== undefined ? m[1].replace(/\\"/g, '"') : m[2])
  return out
}
