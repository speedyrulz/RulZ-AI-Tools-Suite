// Live list of NVIDIA cards from the Windows nvidia-smi (so looking at GPUs never starts WSL).
import { execFile } from 'node:child_process'
import type { GpuInfo } from '@shared/types'

export const NVIDIA_SMI_ARGS = [
  '--query-gpu=index,name,pci.bus_id,memory.used,memory.total,utilization.gpu,temperature.gpu',
  '--format=csv,noheader,nounits'
]

function num(s: string | undefined): number | null {
  if (s === undefined) return null
  const v = Number.parseFloat(s.trim())
  return Number.isFinite(v) ? v : null // "[N/A]", "N/A" and "" give null
}

export function parseNvidiaSmi(csv: string): GpuInfo[] {
  const gpus: GpuInfo[] = []
  for (const raw of csv.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line) continue
    const parts = line.split(',').map((p) => p.trim())
    if (parts.length < 7) continue
    const index = Number.parseInt(parts[0], 10)
    if (!Number.isInteger(index) || !/^\d+$/.test(parts[0])) continue
    gpus.push({
      index,
      name: parts[1],
      busId: parts[2],
      memoryUsedMiB: Math.round(num(parts[3]) ?? 0),
      memoryTotalMiB: Math.round(num(parts[4]) ?? 0),
      utilizationPct: Math.round(num(parts[5]) ?? 0),
      temperatureC: num(parts[6]) === null ? null : Math.round(num(parts[6]) as number)
    })
  }
  return gpus.sort((a, b) => a.index - b.index)
}

function defaultExec(file: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(file, args, { windowsHide: true, timeout: 10000, maxBuffer: 1 << 20 }, (err, stdout) => {
      if (err) reject(err)
      else resolve(String(stdout))
    })
  })
}

/** Returns [] when nvidia-smi is missing or fails. */
export async function listGpus(exec: (file: string, args: string[]) => Promise<string> = defaultExec): Promise<GpuInfo[]> {
  try {
    return parseNvidiaSmi(await exec('nvidia-smi.exe', NVIDIA_SMI_ARGS))
  } catch {
    return []
  }
}
