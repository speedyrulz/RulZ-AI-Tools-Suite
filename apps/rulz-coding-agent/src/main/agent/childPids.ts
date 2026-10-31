// Remembers the shells this app starts for background commands and "Run my app", so the ones left
// behind when the app crashes (or is killed) are stopped the next time it starts.
import { spawn, spawnSync } from 'node:child_process'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

/** Reads the PIDs of PowerShell processes from `tasklist /FO CSV /NH` output. */
export function parseShellPids(csv: string): Set<number> {
  const pids = new Set<number>()
  for (const line of csv.split(/\r?\n/)) {
    const m = /^"(powershell\.exe|pwsh\.exe)","(\d+)"/i.exec(line.trim())
    if (m) pids.add(Number(m[2]))
  }
  return pids
}

function runningShells(): Set<number> {
  if (process.platform !== 'win32') return new Set()
  const r = spawnSync('tasklist', ['/FO', 'CSV', '/NH'], { windowsHide: true, encoding: 'utf8' })
  return parseShellPids(r.stdout ?? '')
}

export class ChildPids {
  private file: string | null = null
  private pids = new Set<number>()

  /**
   * Starts recording to `file`, first stopping the shells (and what they started) recorded there by
   * an earlier run that didn't get to stop them. Only PIDs that still belong to a PowerShell process
   * are touched, so a PID Windows has since given to another program is left alone.
   */
  async init(file: string, kill: (pid: number) => void = killTreeSync, shells: () => Set<number> = runningShells): Promise<number[]> {
    const left = await readJson<number[]>(file, [])
    const live = Array.isArray(left) && left.length ? shells() : new Set<number>()
    const killed = (Array.isArray(left) ? left : []).filter((pid) => typeof pid === 'number' && live.has(pid))
    for (const pid of killed) kill(pid)
    this.file = file
    await this.save()
    return killed
  }

  add(pid: number | undefined): void {
    if (!pid) return
    this.pids.add(pid)
    void this.save()
  }

  remove(pid: number | undefined): void {
    if (!pid || !this.pids.delete(pid)) return
    void this.save()
  }

  private save(): Promise<void> {
    if (!this.file) return Promise.resolve()
    return writeJsonAtomic(this.file, [...this.pids]).catch(() => undefined)
  }
}

function killTreeSync(pid: number): void {
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
  else spawn('kill', ['-9', String(pid)], { stdio: 'ignore' }).on('error', () => undefined)
}

/** The app's one record (set up in main; without init, as in tests, nothing is written). */
export const childPids = new ChildPids()
