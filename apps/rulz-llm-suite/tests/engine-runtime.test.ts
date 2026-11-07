import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { HardwareSnapshot } from '@shared/types'
import { EngineService } from '../src/main/engine/engine'
import { HardwareMonitor } from '../src/main/engine/hardware'

const EXE = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server'
const snapshot: HardwareSnapshot = { gpus: [], ramUsedBytes: 0, ramTotalBytes: 1, at: 0 }

describe('HardwareMonitor', () => {
  it('keeps a single polling chain however often the window is focused', async () => {
    const mon = new HardwareMonitor(() => undefined)
    let calls = 0
    let inFlight = 0
    let maxInFlight = 0
    // Stand-in for nvidia-smi: slow enough for pokes to land while it runs.
    mon.refresh = async () => {
      calls++
      maxInFlight = Math.max(maxInFlight, ++inFlight)
      await new Promise((r) => setTimeout(r, 20))
      inFlight--
      return snapshot
    }
    mon.start(10, 10)
    for (let i = 0; i < 20; i++) {
      mon.poke()
      await new Promise((r) => setTimeout(r, 3))
    }
    mon.stop()
    await new Promise((r) => setTimeout(r, 50))
    expect(maxInFlight).toBe(1)
    const after = calls
    await new Promise((r) => setTimeout(r, 60))
    expect(calls).toBe(after)
  })

  it('shares a refresh that is already running', async () => {
    const mon = new HardwareMonitor(() => undefined)
    const internals = mon as unknown as { doRefresh: () => Promise<HardwareSnapshot> }
    let calls = 0
    internals.doRefresh = async () => {
      calls++
      await new Promise((r) => setTimeout(r, 10))
      return snapshot
    }
    await Promise.all([mon.refresh(), mon.refresh(), mon.refresh()])
    expect(calls).toBe(1)
    await mon.refresh()
    expect(calls).toBe(2)
  })
})

describe('EngineService.getInfo', () => {
  const engineDir = (): string => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-engine-'))
    writeFileSync(join(dir, EXE), '')
    return dir
  }

  it('runs llama-server once for calls made while a probe is running, and caches a working engine', async () => {
    const dir = engineDir()
    const runs: string[] = []
    const engine = new EngineService(
      () => dir,
      async (_exe, args) => {
        runs.push(args[0])
        await new Promise((r) => setTimeout(r, 10))
        return { code: 0, output: args[0] === '--version' ? 'version: 9999 (abc)' : 'CUDA0: GPU (24576 MiB, 20000 MiB free)' }
      }
    )
    const [a, b] = await Promise.all([engine.getInfo(), engine.getInfo(true)])
    expect(a).toBe(b)
    expect(a).toMatchObject({ ok: true, version: '9999 (abc)' })
    expect(runs.sort()).toEqual(['--list-devices', '--version'])
    await engine.getInfo()
    expect(runs).toHaveLength(2)
  })

  it('does not cache a failed probe, so a fixed build is found on the next call', async () => {
    const dir = engineDir()
    let broken = true
    let runs = 0
    const engine = new EngineService(
      () => dir,
      async () => {
        runs++
        return broken ? { code: 3221225781, output: '' } : { code: 0, output: 'version: 1 (x)' }
      }
    )
    expect((await engine.getInfo()).ok).toBe(false)
    broken = false
    expect((await engine.getInfo()).ok).toBe(true)
    expect(runs).toBe(4)
  })
})
