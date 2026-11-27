// Runs bash scripts inside WSL. Quoting a script through `wsl.exe ... bash -lc "..."` breaks, so every script goes
// through a temp file (LF endings) that bash reads by its /mnt path.
import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { readdirSync, statSync, unlink, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

export interface WslResult {
  code: number | null
  stdout: string
  stderr: string
}

export interface WslRunner {
  /** Runs a bash script and waits for it. `code` is null when it could not start or ran out of time. */
  run(distro: string, script: string, opts?: { timeoutMs?: number }): Promise<WslResult>
  /** Starts a long-running script (the server). The temp file is deleted when the process exits. */
  spawn(distro: string, script: string): ChildProcess
}

/** C:\a b\c -> /mnt/c/a b/c. A \\wsl.localhost\<distro>\home\x path becomes /home/x. */
export function toWslPath(winPath: string): string {
  const p = winPath.trim()
  const unc = /^\\\\(?:wsl\.localhost|wsl\$)\\[^\\]+(\\.*)?$/i.exec(p)
  if (unc) return (unc[1] ?? '').replace(/\\/g, '/') || '/'
  const m = /^([A-Za-z]):[\\/]*(.*)$/.exec(p)
  if (!m) return p.replace(/\\/g, '/')
  const rest = m[2].replace(/\\/g, '/')
  return `/mnt/${m[1].toLowerCase()}/${rest}`.replace(/\/+$/, '') || `/mnt/${m[1].toLowerCase()}`
}

/** Single-quotes a string for bash. */
export function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`
}

/** wsl.exe prints a warning for every malformed PATH entry it cannot translate; drop them from its copy of PATH. */
export function cleanWindowsPath(path: string): string {
  return path
    .split(';')
    .map((e) => e.replace(/"/g, '').trim())
    .filter((e) => e && !/^[A-Za-z]:\s/.test(e))
    .join(';')
}

function wslEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, WSL_UTF8: '1' }
  for (const k of Object.keys(env)) {
    if (k.toLowerCase() === 'path' && env[k]) env[k] = cleanWindowsPath(env[k] as string)
  }
  return env
}

export function stripNul(s: string): string {
  return s.replace(/\0/g, '')
}

function writeScript(script: string): string {
  const file = join(tmpdir(), `rulz-merger-${Date.now()}-${randomBytes(6).toString('hex')}.sh`)
  writeFileSync(file, script.replace(/\r\n/g, '\n'), { encoding: 'utf8' })
  return file
}

/** Deletes leftover rulz-merger-*.sh script files (from a crash) that are older than `maxAgeMs`. Returns how many. */
export function cleanupOldTempScripts(dir: string = tmpdir(), maxAgeMs = 24 * 60 * 60 * 1000, now = Date.now()): number {
  let removed = 0
  let names: string[] = []
  try {
    names = readdirSync(dir)
  } catch {
    return 0
  }
  for (const name of names) {
    if (!/^rulz-merger-.*\.sh$/.test(name)) continue
    const full = join(dir, name)
    try {
      if (now - statSync(full).mtimeMs > maxAgeMs) {
        unlink(full, () => {})
        removed++
      }
    } catch {
      // gone already
    }
  }
  return removed
}

function removeFile(file: string): void {
  unlink(file, () => {})
}

// `--exec` runs bash directly, without the distro's login shell, so a temp path with spaces needs no extra quoting.
function wslArgs(distro: string, file: string): string[] {
  return [...(distro.trim() ? ['-d', distro.trim()] : []), '--exec', 'bash', toWslPath(file)]
}

export function createWslRunner(): WslRunner {
  return {
    run(distro, script, opts) {
      const timeoutMs = opts?.timeoutMs ?? 20000
      return new Promise<WslResult>((resolve) => {
        let file: string
        try {
          file = writeScript(script)
        } catch (e) {
          resolve({ code: null, stdout: '', stderr: `Could not write the script: ${(e as Error).message}` })
          return
        }
        let out = ''
        let err = ''
        let done = false
        const finish = (code: number | null): void => {
          if (done) return
          done = true
          clearTimeout(timer)
          removeFile(file)
          resolve({ code, stdout: stripNul(out), stderr: stripNul(err) })
        }
        const child = spawn('wsl.exe', wslArgs(distro, file), { windowsHide: true, env: wslEnv(), stdio: ['ignore', 'pipe', 'pipe'] })
        const timer = setTimeout(() => {
          err += `\nTimed out after ${Math.round(timeoutMs / 1000)} s`
          child.kill()
          finish(null)
        }, timeoutMs)
        child.stdout?.on('data', (c: Buffer | string) => (out += c.toString()))
        child.stderr?.on('data', (c: Buffer | string) => (err += c.toString()))
        child.on('error', (e) => {
          err += `\n${e.message}`
          finish(null)
        })
        child.on('close', (code) => finish(code))
      })
    },
    spawn(distro, script) {
      const file = writeScript(script)
      const child = spawn('wsl.exe', wslArgs(distro, file), { windowsHide: true, env: wslEnv(), stdio: ['ignore', 'pipe', 'pipe'] })
      let removed = false
      const cleanup = (): void => {
        if (removed) return
        removed = true
        removeFile(file)
      }
      child.once('close', cleanup)
      child.once('error', cleanup)
      return child
    }
  }
}
