// Checks before the own server starts: is WSL / ComfyUI / python there, is a port free, is the chosen GPU busy.
import type { GpuInfo, OwnServerSettings, PreflightCode, PreflightIssue, PreflightReport, RunningServer } from '@shared/types'
import { scanForFreePort } from './ports'
import { buildProbeScript } from './scripts'
import type { WslRunner } from './wsl'

/**
 * Codes this module uses that are not in the shared PreflightCode union yet. The union needs these four added:
 * 'ports-unreadable' | 'managed-arg' | 'path-not-absolute' | 'gpu-unknown'.
 */

export interface PreflightFacts {
  settings: OwnServerSettings
  gpus: GpuInfo[]
  runningServers: RunningServer[]
  freeRamGiB: number | null
  totalRamGiB: number | null
  port: number | null
  wslFound: boolean
  comfyFound: boolean
  pythonFound: boolean
  ownRunning: boolean
  /** True when WSL could not be asked which ports are in use. */
  portCheckFailed?: boolean
}

const GIB = 1024 ** 3

/** Flags the app sets itself; the user's extra options must not repeat or override them. */
export const MANAGED_FLAGS = [
  '--port',
  '--listen',
  '--base-directory',
  '--models-directory',
  '--cuda-device',
  '--cpu',
  '--extra-model-paths-config',
  '--output-directory',
  '--input-directory',
  '--user-directory',
  '--default-device'
]

/** The managed flags used in an extra-options string, including `--flag=value` forms and argparse abbreviations. */
export function findManagedFlags(extraArgs: string): string[] {
  const found = new Set<string>()
  for (const tok of extraArgs.split(/\s+/).filter(Boolean)) {
    if (!tok.startsWith('--')) continue
    const flag = tok.split('=')[0]
    for (const m of MANAGED_FLAGS) {
      // argparse accepts any unambiguous prefix of a flag, such as --base-dir
      if (flag === m || (flag.length >= 5 && m.startsWith(flag))) found.add(m)
    }
  }
  return [...found]
}

export function isAbsoluteWslPath(p: string): boolean {
  const t = p.trim()
  return t.startsWith('/') || t.startsWith('~/')
}

/** Parses `free -b`: the "Mem:" row, total and "available" (falls back to free when the column is missing). */
export function parseFree(text: string): { totalGiB: number | null; availableGiB: number | null } {
  for (const line of text.split(/\r?\n/)) {
    if (!/^\s*Mem:/i.test(line)) continue
    const n = line.trim().split(/\s+/).slice(1).map(Number)
    if (n.length < 1 || !Number.isFinite(n[0])) break
    const available = n.length >= 6 ? n[5] : n[2]
    return { totalGiB: n[0] / GIB, availableGiB: Number.isFinite(available) ? available / GIB : null }
  }
  return { totalGiB: null, availableGiB: null }
}

const fmt = (n: number): string => n.toFixed(1)

export function buildIssues(f: PreflightFacts): PreflightIssue[] {
  const issues: PreflightIssue[] = []
  const add = (code: PreflightCode, level: PreflightIssue['level'], message: string): void => {
    issues.push({ code, level, message })
  }
  const s = f.settings

  const badPaths: string[] = []
  if (!isAbsoluteWslPath(s.baseDir)) badPaths.push(`the base folder "${s.baseDir}"`)
  if (!isAbsoluteWslPath(s.comfyDir)) badPaths.push(`the ComfyUI folder "${s.comfyDir}"`)
  if (s.modelsDir.trim() && !isAbsoluteWslPath(s.modelsDir)) badPaths.push(`the models folder "${s.modelsDir}"`)
  if (badPaths.length) {
    add('path-not-absolute', 'block', `${badPaths.join(', ')} must be a full path that starts with / or ~/ . Change it in the settings.`)
  }
  const managed = findManagedFlags(s.extraArgs)
  if (managed.length) {
    add('managed-arg', 'block', `The extra options contain ${managed.join(', ')}, which this app sets itself. Remove ${managed.length > 1 ? 'them' : 'it'} from the extra options.`)
  }

  if (!f.wslFound) {
    add('wsl-missing', 'block', `WSL or the "${s.distro}" distro could not be started, so the app cannot run its own ComfyUI. Install it or pick another distro in the settings.`)
  } else {
    if (!f.comfyFound) {
      add('comfy-missing', 'block', `There is no ComfyUI in ${s.comfyDir} inside ${s.distro}. Set the right folder in the settings.`)
    } else if (!f.pythonFound) {
      add('python-missing', 'block', `The Python "${s.python}" was not found for the ComfyUI in ${s.comfyDir}. Set the right Python in the settings.`)
    }
    if (f.portCheckFailed && !f.ownRunning) {
      add('ports-unreadable', 'block', 'Could not check which ports are in use in WSL; try again.')
    } else if (f.port === null && !f.ownRunning) {
      add('no-free-port', 'block', `No port is free from ${s.portStart} upwards, so the server cannot start. Choose another starting port in the settings.`)
    }
  }
  if (f.ownRunning) add('already-running', 'block', "The app's own ComfyUI server is already running. Stop it first if you want to start it again.")

  const dev = s.device
  if (dev.kind === 'gpu') {
    const gpu = f.gpus.find((g) => g.index === dev.index)
    if (!gpu) {
      add('gpu-unknown', 'confirm', `The GPU you chose (${dev.name}) was not found in the list from nvidia-smi, so you cannot see whether other ComfyUI servers are using it. Start anyway only if you know it is free.`)
    } else {
      for (const srv of f.runningServers) {
        if (!srv.own && srv.gpuIndexes.includes(dev.index)) {
          add('gpu-in-use-by-server', 'confirm', `${gpu.name} is used by the ComfyUI server on port ${srv.port}; a second server on it can run out of memory and slow its renders down.`)
        }
      }
      const freeGiB = (gpu.memoryTotalMiB - gpu.memoryUsedMiB) / 1024
      if (freeGiB < 4) add('gpu-low-vram', 'info', `${gpu.name} has only ${fmt(Math.max(0, freeGiB))} GiB of video memory free right now, so large models may not fit.`)
    }
  }
  if (f.freeRamGiB !== null && f.freeRamGiB < s.minFreeRamGiB) {
    add('low-ram', 'confirm', `Only ${fmt(f.freeRamGiB)} GiB of memory is free in WSL and you asked for at least ${fmt(s.minFreeRamGiB)} GiB. Merging large models may make WSL swap or stop.`)
  }
  return issues
}

export interface PreflightDeps {
  wsl: WslRunner
  settings: OwnServerSettings
  listGpus: () => Promise<GpuInfo[]>
  probe: (url: string) => Promise<RunningServer | null>
  ownRunning: boolean
  ownPort: number | null
}

const MAX_PROBED_PORTS = 40

/** Ports to look at for other ComfyUI servers: the usual ones, then every port WSL listens on (1024 and up). */
export function portsToProbe(portStart: number, listening: Set<number>, ownPort: number | null): number[] {
  const first: number[] = []
  for (let p = 8188; p <= 8199; p++) first.push(p)
  for (let p = portStart; p <= portStart + 11; p++) if (p >= 1 && p <= 65535) first.push(p)
  const rest = [...listening].filter((p) => p >= 1024).sort((a, b) => a - b)
  const all: number[] = []
  const seen = new Set<number>()
  for (const p of [...first, ...rest]) {
    if (p === ownPort || seen.has(p)) continue
    seen.add(p)
    all.push(p)
    if (all.length >= MAX_PROBED_PORTS) break
  }
  return all.sort((a, b) => a - b)
}

export async function runPreflight(d: PreflightDeps): Promise<PreflightReport> {
  const s = d.settings
  const [probeRes, gpus] = await Promise.all([d.wsl.run(s.distro, buildProbeScript(s), { timeoutMs: 20000 }), d.listGpus().catch(() => [] as GpuInfo[])])
  const out = probeRes.stdout
  const wslFound = probeRes.code === 0 && out.includes('RULZ_OK')
  const comfyFound = wslFound && out.includes('RULZ_COMFY')
  const pythonFound = comfyFound && out.includes('RULZ_PYTHON')
  const ram = wslFound ? parseFree(out) : { totalGiB: null, availableGiB: null }

  let port: number | null = null
  let portCheckFailed = false
  let listening = new Set<number>()
  if (wslFound) {
    const scan = await scanForFreePort(s.portStart, d.wsl, s.distro)
    listening = scan.listening
    portCheckFailed = scan.reason === 'ss-failed'
    port = scan.port
  }
  if (d.ownRunning) port = d.ownPort

  const probePorts = portsToProbe(s.portStart, listening, d.ownPort)
  const found = await Promise.all(probePorts.map((p) => d.probe(`http://127.0.0.1:${p}`).catch(() => null)))
  const runningServers = found.filter((x): x is RunningServer => x !== null)

  const facts: PreflightFacts = {
    settings: s,
    gpus,
    runningServers,
    freeRamGiB: ram.availableGiB,
    totalRamGiB: ram.totalGiB,
    port,
    wslFound,
    comfyFound,
    pythonFound,
    ownRunning: d.ownRunning,
    portCheckFailed
  }
  return { freeRamGiB: ram.availableGiB, totalRamGiB: ram.totalGiB, gpus, runningServers, port, issues: buildIssues(facts) }
}
