// Remembers the shells this app starts for background commands and "Run my app", so the ones left
// behind when the app crashes (or is killed) are stopped the next time it starts.
import { execFile, spawn, spawnSync } from 'node:child_process'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

/** Running PowerShell processes: PID → when the process started (ms since 1970). */
export type ShellList = Map<number, number>

interface Recorded {
  pid: number
  /** When it was recorded, just after it started. Missing in records written before 0.19.0. */
  at?: number
}

/**
 * A recorded shell is the same process when one with its PID started at about the time it was
 * recorded; Windows reuses PIDs, so a PowerShell window the owner opened since may have it now.
 */
const START_SLACK_MS = 10_000
function sameProcess(r: Recorded, started: number | undefined): boolean {
  if (started === undefined) return false
  if (r.at === undefined) return true
  return started >= r.at - START_SLACK_MS && started <= r.at + 2000
}

/** Reads `<pid> <start ms>` lines (the query in runningShells). */
export function parseShellList(out: string): ShellList {
  const list: ShellList = new Map()
  for (const line of out.split(/\r?\n/)) {
    const m = /^(\d+) (\d+)$/.exec(line.trim())
    if (m) list.set(Number(m[1]), Number(m[2]))
  }
  return list
}

function runningShells(): Promise<ShellList> {
  if (process.platform !== 'win32') return Promise.resolve(new Map())
  const script =
    `Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" | ` +
    `ForEach-Object { '{0} {1}' -f $_.ProcessId, ([DateTimeOffset]$_.CreationDate).ToUnixTimeMilliseconds() }`
  return new Promise((resolve) =>
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { windowsHide: true, timeout: 20_000 }, (_err, stdout) =>
      resolve(parseShellList(String(stdout ?? '')))
    )
  )
}

function toRecords(value: unknown): Recorded[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((v): Recorded[] => {
    if (typeof v === 'number') return [{ pid: v }]
    if (v && typeof v === 'object' && typeof (v as Recorded).pid === 'number') {
      const at = (v as Recorded).at
      return [{ pid: (v as Recorded).pid, ...(typeof at === 'number' ? { at } : {}) }]
    }
    return []
  })
}

export class ChildPids {
  private file: string | null = null
  private pids = new Map<number, number>()

  /**
   * Starts recording to `file`, first stopping the shells (and what they started) recorded there by
   * an earlier run that didn't get to stop them. Only PIDs that still belong to a PowerShell process
   * started when the PID was recorded are touched, so a PID Windows has since given to another
   * program, or to another PowerShell, is left alone.
   */
  async init(
    file: string,
    kill: (pid: number) => void = killTreeSync,
    shells: () => ShellList | Promise<ShellList> = runningShells
  ): Promise<number[]> {
    const left = toRecords(await readJson<unknown>(file, []).catch((): unknown => []))
    const live = left.length ? await shells() : new Map<number, number>()
    const killed = left.filter((r) => sameProcess(r, live.get(r.pid))).map((r) => r.pid)
    for (const pid of killed) kill(pid)
    this.file = file
    await this.save()
    return killed
  }

  add(pid: number | undefined): void {
    if (!pid) return
    this.pids.set(pid, Date.now())
    void this.save()
  }

  remove(pid: number | undefined): void {
    if (!pid || !this.pids.delete(pid)) return
    void this.save()
  }

  private save(): Promise<void> {
    if (!this.file) return Promise.resolve()
    return writeJsonAtomic(this.file, [...this.pids].map(([pid, at]) => ({ pid, at }))).catch(() => undefined)
  }
}

function killTreeSync(pid: number): void {
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
  else spawn('kill', ['-9', String(pid)], { stdio: 'ignore' }).on('error', () => undefined)
}

/** The app's one record (set up in main; without init, as in tests, nothing is written). */
export const childPids = new ChildPids()
