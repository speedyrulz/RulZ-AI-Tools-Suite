import { existsSync } from 'node:fs'
import { join } from 'node:path'
import type { RequirementId, RequirementStatus } from '@shared/types'
import { capture, refreshPath, runCommand } from './process'

interface RequirementDef {
  label: string
  helpUrl: string
  /** winget package id, when winget can install it. */
  winget?: string
  check: () => Promise<{ ok: boolean; detail: string }>
}

const firstLine = (s: string): string => s.split(/\r?\n/)[0].trim()

const DEFS: Record<RequirementId, RequirementDef> = {
  python: {
    label: 'Python 3.10+',
    helpUrl: 'https://www.python.org/downloads/windows/',
    winget: 'Python.Python.3.12',
    check: async () => {
      // The Microsoft Store "python" alias prints nothing useful and exits non-zero, so it counts as missing.
      const out = await capture('python', ['--version'])
      const m = out && /Python (\d+)\.(\d+)\.(\d+)/.exec(out)
      if (!m) return { ok: false, detail: 'Not found on PATH' }
      const ok = Number(m[1]) === 3 && Number(m[2]) >= 10
      return { ok, detail: ok ? `Python ${m[1]}.${m[2]}.${m[3]}` : `Python ${m[1]}.${m[2]} is too old` }
    }
  },
  ffmpeg: {
    label: 'FFmpeg',
    helpUrl: 'https://ffmpeg.org/download.html#build-windows',
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
    label: 'Git',
    helpUrl: 'https://git-scm.com/download/win',
    winget: 'Git.Git',
    check: async () => {
      const out = await capture('git', ['--version'])
      return out ? { ok: true, detail: firstLine(out).replace(/\.windows\.\d+$/, '') } : { ok: false, detail: 'Not found' }
    }
  },
  nvidia: {
    label: 'NVIDIA GPU',
    helpUrl: 'https://www.nvidia.com/en-us/drivers/',
    check: async () => {
      const out = await capture('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader'])
      if (!out) return { ok: false, detail: 'No NVIDIA driver found' }
      const gpus = out.split(/\r?\n/).map((l) => {
        const [name, mem] = l.split(',').map((s) => s.trim())
        const gb = Math.round(parseInt(mem, 10) / 1024)
        return `${name.replace(/^NVIDIA (GeForce )?/, '')} (${gb} GB)`
      })
      return { ok: true, detail: gpus.join(', ') }
    }
  },
  'dotnet-fx': {
    label: '.NET Framework 4.8',
    helpUrl: 'https://dotnet.microsoft.com/download/dotnet-framework/net48',
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

export async function checkRequirements(): Promise<RequirementStatus[]> {
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
export async function installRequirement(id: RequirementId, onLine: (line: string) => void): Promise<void> {
  const pkg = DEFS[id].winget
  if (!pkg) throw new Error(`${DEFS[id].label} can't be installed automatically.`)
  const code = await runCommand(
    ['winget', 'install', '--exact', '--id', pkg, '--source', 'winget', '--silent',
      '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity'],
    { cwd: process.env.TEMP ?? process.cwd(), onLine }
  )
  await refreshPath()
  // winget exits with "already installed" (0x8A15002B) when there is nothing to do.
  if (code !== 0 && (code | 0) !== (0x8a15002b | 0)) throw new Error(`winget exited with code ${code}`)
}
