import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { REQUIREMENT_GUIDES } from '@shared/requirements'
import type { RequirementId, RequirementRef, RequirementStatus } from '@shared/types'
import { capture, refreshPath, runCommand } from './process'

interface RequirementDef {
  label: string
  helpUrl: string
  /** winget package id, when winget can install it. */
  winget?: string
  /** Replaces the package's own silent-install switches. */
  wingetOverride?: string
  check: () => Promise<{ ok: boolean; detail: string }>
}

/**
 * A per-user python.org install that isn't on PATH (installed without "Add to PATH", or PATH not yet reloaded).
 * Its folders are added to this process's PATH, so setup steps that run "python" find it too.
 */
function adoptPythonInstall(): boolean {
  const root = join(process.env.LOCALAPPDATA ?? '', 'Programs', 'Python')
  const dirs = existsSync(root) ? readdirSync(root).filter((d) => /^Python3(1\d|[2-9]\d)$/.test(d)).sort().reverse() : []
  const dir = dirs.map((d) => join(root, d)).find((d) => existsSync(join(d, 'python.exe')))
  if (!dir) return false
  process.env.PATH = [dir, join(dir, 'Scripts'), process.env.PATH].join(';')
  return true
}

async function pythonVersion(): Promise<RegExpExecArray | null> {
  const out = await capture('python', ['--version'])
  return out ? /Python (\d+)\.(\d+)\.(\d+)/.exec(out) : null
}

const firstLine = (s: string): string => s.split(/\r?\n/)[0].trim()

/** The apps' CUDA builds are CUDA 13: they need this driver or newer, and a card of this compute capability. */
export const MIN_DRIVER = 580
export const MIN_COMPUTE = 7.5

/**
 * Reads `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits` (the last
 * one or two columns may be missing on old drivers) and decides whether the CUDA 13 builds can run.
 */
export function parseGpuQuery(out: string): { ok: boolean; detail: string } {
  const gpus = out
    .split(/\r?\n/)
    .map((l) => l.split(',').map((s) => s.trim()))
    .filter((cols) => cols[0])
    .map(([name, mem, driver, compute]) => ({
      name: name.replace(/^NVIDIA (GeForce )?/, ''),
      gb: Math.round(parseInt(mem, 10) / 1024),
      driver: driver ?? '',
      compute: compute && /^\d+(\.\d+)?$/.test(compute) ? Number(compute) : null
    }))
  if (!gpus.length) return { ok: false, detail: 'No NVIDIA GPU found' }
  const label = (g: (typeof gpus)[number]): string => (Number.isFinite(g.gb) ? `${g.name} (${g.gb} GB)` : g.name)
  const names = gpus.map(label).join(', ')

  const driver = gpus[0].driver
  const major = parseInt(driver, 10)
  if (!Number.isFinite(major) || major < MIN_DRIVER) {
    const which = driver ? `driver ${driver} is too old` : 'driver is too old'
    return {
      ok: false,
      detail: `${names}: the NVIDIA ${which}. The AI apps use CUDA 13, which needs driver ${MIN_DRIVER} or newer. Update the NVIDIA driver.`
    }
  }
  const capable = gpus.filter((g) => g.compute === null || g.compute >= MIN_COMPUTE)
  if (!capable.length) {
    const cc = gpus.map((g) => `${g.name} is compute ${g.compute}`).join(', ')
    return {
      ok: false,
      detail: `${cc}. CUDA 13 needs a GTX 16 or RTX 20 series (compute ${MIN_COMPUTE}) or newer card, so the AI apps can't use this GPU.`
    }
  }
  return { ok: true, detail: `${capable.map(label).join(', ')} · driver ${driver}` }
}

const DEFS: Record<RequirementId, RequirementDef> = {
  python: {
    ...REQUIREMENT_GUIDES.python,
    winget: 'Python.Python.3.12',
    // Per-user, added to PATH (python.org's installer leaves PATH alone by default).
    wingetOverride: '/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0',
    check: async () => {
      // The Microsoft Store "python" alias prints nothing useful and exits non-zero, so it counts as missing.
      let m = await pythonVersion()
      if (!m && adoptPythonInstall()) m = await pythonVersion()
      if (!m) return { ok: false, detail: 'Not found on PATH' }
      const ok = Number(m[1]) === 3 && Number(m[2]) >= 10
      return { ok, detail: ok ? `Python ${m[1]}.${m[2]}.${m[3]}` : `Python ${m[1]}.${m[2]} is too old` }
    }
  },
  ffmpeg: {
    ...REQUIREMENT_GUIDES.ffmpeg,
    winget: 'Gyan.FFmpeg',
    check: async () => {
      const ff = await capture('ffmpeg', ['-hide_banner', '-version'])
      const probe = await capture('ffprobe', ['-hide_banner', '-version'])
      if (!ff || !probe) return { ok: false, detail: ff ? 'ffprobe not found on PATH' : 'Not found on PATH' }
      const v = /version n?([\d.]+)/.exec(ff)
      return { ok: true, detail: v ? `ffmpeg ${v[1]}` : 'Found' }
    }
  },
  git: {
    ...REQUIREMENT_GUIDES.git,
    winget: 'Git.Git',
    check: async () => {
      const out = await capture('git', ['--version'])
      return out ? { ok: true, detail: firstLine(out).replace(/\.windows\.\d+$/, '') } : { ok: false, detail: 'Not found' }
    }
  },
  nvidia: {
    ...REQUIREMENT_GUIDES.nvidia,
    check: async () => {
      const out = await capture('nvidia-smi', [
        '--query-gpu=name,memory.total,driver_version,compute_cap',
        '--format=csv,noheader,nounits'
      ])
      if (out) return parseGpuQuery(out)
      // Drivers too old to know compute_cap (before about 510) answer the shorter query.
      const old = await capture('nvidia-smi', ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'])
      return old ? parseGpuQuery(old) : { ok: false, detail: 'No NVIDIA driver found' }
    }
  },
  'dotnet-fx': {
    ...REQUIREMENT_GUIDES['dotnet-fx'],
    check: async () => {
      const csc = join(process.env.WINDIR ?? 'C:\\Windows', 'Microsoft.NET', 'Framework64', 'v4.0.30319', 'csc.exe')
      return existsSync(csc) ? { ok: true, detail: 'Built into Windows' } : { ok: false, detail: 'C# compiler not found' }
    }
  }
}

export function requirementLabel(id: RequirementId): string {
  return DEFS[id].label
}

let wingetAvailable: Promise<boolean> | null = null

/** A check runs six programs; answers are reused for a minute (and dropped when something may have been installed). */
const CHECK_TTL_MS = 60_000
let lastCheck: { at: number; result: Promise<RequirementStatus[]> } | null = null

/** Forgets the last check, after an install that may have added a prerequisite. */
export function invalidateRequirements(): void {
  lastCheck = null
}

/** Checks every requirement. `reloadPath` first picks up programs installed since the launcher started. */
export async function checkRequirements(reloadPath = false): Promise<RequirementStatus[]> {
  if (reloadPath) {
    await refreshPath()
    wingetAvailable = null
    lastCheck = null
  }
  if (lastCheck && Date.now() - lastCheck.at < CHECK_TTL_MS) return lastCheck.result
  const check = { at: Date.now(), result: runChecks() }
  lastCheck = check
  check.result.catch(() => {
    if (lastCheck === check) lastCheck = null
  })
  return check.result
}

async function runChecks(): Promise<RequirementStatus[]> {
  wingetAvailable ??= capture('winget', ['--version']).then((v) => !!v)
  const hasWinget = await wingetAvailable
  const ids = Object.keys(DEFS) as RequirementId[]
  return Promise.all(
    ids.map(async (id) => {
      const def = DEFS[id]
      const r = await def.check().catch(() => ({ ok: false, detail: 'Check failed' }))
      return { id, label: def.label, ...r, canInstall: !r.ok && hasWinget && !!def.winget, helpUrl: def.helpUrl }
    })
  )
}

/** Installs a requirement with winget (Windows may ask for permission), then reloads PATH. */
export async function installRequirement(
  id: RequirementId,
  onLine: (line: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const def = DEFS[id]
  if (!def.winget) throw new Error(`${def.label} can't be installed automatically.`)
  const code = await runCommand(
    [
      'winget', 'install', '--exact', '--id', def.winget, '--source', 'winget', '--silent',
      '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity',
      ...(def.wingetOverride ? ['--override', def.wingetOverride] : [])
    ],
    { cwd: process.env.TEMP ?? process.cwd(), onLine, signal }
  )
  await refreshPath()
  invalidateRequirements()
  // winget exits with "already installed" (0x8A15002B) when there is nothing to do.
  if (code !== 0 && (code | 0) !== (0x8a15002b | 0)) throw new Error(`winget exited with code ${code}`)
}

/**
 * Installs an app's missing required prerequisites with winget, where it can.
 * Returns the ones still missing afterwards (they need installing by hand).
 */
export async function ensureRequirements(
  refs: RequirementRef[],
  log: (line: string) => void,
  onInstalling: (label: string) => void,
  signal?: AbortSignal
): Promise<RequirementId[]> {
  const needed = refs.filter((r) => !r.optional).map((r) => r.id)
  if (!needed.length) return []
  // An install decides on a fresh check, not one from up to a minute ago.
  invalidateRequirements()
  let status = await checkRequirements()
  for (const id of needed) {
    const s = status.find((x) => x.id === id)!
    if (s.ok || !s.canInstall) continue
    onInstalling(s.label)
    log(`${s.label} is missing; installing it with winget (Windows may ask for permission).`)
    try {
      await installRequirement(id, log, signal)
    } catch (e) {
      if (signal?.aborted) throw e
      log(`Couldn't install ${s.label} automatically: ${e instanceof Error ? e.message : String(e)}`)
    }
    status = await checkRequirements()
  }
  return needed.filter((id) => !status.find((x) => x.id === id)?.ok)
}
