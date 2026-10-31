import { execFile } from 'node:child_process'
import { freemem, totalmem } from 'node:os'
import type { GpuStat, HardwareSnapshot } from '@shared/types'

export function parseNvidiaSmi(csv: string): GpuStat[] {
  const gpus: GpuStat[] = []
  for (const line of csv.split(/\r?\n/)) {
    const cols = line.split(',').map((c) => c.trim())
    if (cols.length < 5 || !/^\d+$/.test(cols[0])) continue
    gpus.push({
      index: Number(cols[0]),
      name: cols[1],
      memUsedMiB: Number(cols[2]) || 0,
      memTotalMiB: Number(cols[3]) || 0,
      utilPct: Number(cols[4]) || 0
    })
  }
  return gpus
}

/** Polls nvidia-smi and system RAM. GPU numbering is PCI bus order, matching engineEnv(). */
export class HardwareMonitor {
  private latest: HardwareSnapshot = { gpus: [], ramUsedBytes: 0, ramTotalBytes: totalmem(), at: 0 }
  private timer: NodeJS.Timeout | null = null
  private smiAvailable = true

  private running = false

  constructor(
    private readonly onUpdate: (s: HardwareSnapshot) => void,
    /** While false (window hidden or in the background, nothing loading), nvidia-smi runs less often. */
    private readonly isActive: () => boolean = () => true
  ) {}

  snapshot(): HardwareSnapshot {
    return this.latest
  }

  start(activeMs = 2000, idleMs = 10000): void {
    if (this.running) return
    this.running = true
    const tick = async (): Promise<void> => {
      await this.refresh()
      if (this.running) this.timer = setTimeout(() => void tick(), this.isActive() ? activeMs : idleMs)
    }
    this.poke = () => {
      if (!this.running) return
      if (this.timer) clearTimeout(this.timer)
      this.timer = null
      void tick()
    }
    void tick()
  }

  /** Refreshes now and goes back to the fast rate if active again (e.g. the window was focused). */
  poke: () => void = () => undefined

  stop(): void {
    this.running = false
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
  }

  async refresh(): Promise<HardwareSnapshot> {
    const gpus = this.smiAvailable ? await this.queryGpus() : []
    const total = totalmem()
    this.latest = { gpus, ramUsedBytes: total - freemem(), ramTotalBytes: total, at: Date.now() }
    this.onUpdate(this.latest)
    return this.latest
  }

  private queryGpus(): Promise<GpuStat[]> {
    return new Promise((resolve) => {
      execFile(
        'nvidia-smi',
        ['--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        { windowsHide: true, timeout: 5000 },
        (err, stdout) => {
          if (err) {
            if ((err as NodeJS.ErrnoException).code === 'ENOENT') this.smiAvailable = false
            resolve(this.latest.gpus)
            return
          }
          resolve(parseNvidiaSmi(stdout))
        }
      )
    })
  }
}
