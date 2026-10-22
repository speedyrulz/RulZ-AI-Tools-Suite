import { spawn, spawnSync } from 'node:child_process'
import { createServer } from 'node:net'
import { join } from 'node:path'

/**
 * Environment for llama.cpp child processes. CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA0/CUDA1
 * match nvidia-smi's GPU numbering (CUDA otherwise orders by speed).
 */
export function engineEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' }
  const cuda = process.env.CUDA_PATH
  if (cuda) {
    const pathKey = Object.keys(env).find((k) => k.toLowerCase() === 'path') ?? 'PATH'
    env[pathKey] = [join(cuda, 'bin', 'x64'), join(cuda, 'bin'), env[pathKey] ?? ''].join(';')
  }
  return env
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

/** True if `pid` is a running llama-server process (used to clean up after a crash). */
export function isLlamaServerPid(pid: number): boolean {
  if (process.platform !== 'win32') return false
  const r = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], {
    windowsHide: true,
    encoding: 'utf8'
  })
  return /llama-server/i.test(r.stdout ?? '')
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
