import { describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS, type GpuInfo, type OwnServerSettings, type RunningServer } from '@shared/types'
import { buildIssues, findManagedFlags, isAbsoluteWslPath, MANAGED_FLAGS, parseFree, portsToProbe, runPreflight, type PreflightFacts } from '../src/main/server/preflight'
import type { WslRunner } from '../src/main/server/wsl'

const GIB = 1024 ** 3
const gpu = (index: number, name: string, used: number, total = 16384): GpuInfo => ({
  index,
  name,
  busId: '',
  memoryUsedMiB: used,
  memoryTotalMiB: total,
  utilizationPct: 0,
  temperatureC: 40
})
const server = (port: number, gpuIndexes: number[], own = false): RunningServer => ({
  url: `http://127.0.0.1:${port}`,
  port,
  version: null,
  devices: [],
  cudaDevice: null,
  gpuIndexes,
  own
})
const settings = (p: Partial<OwnServerSettings> = {}): OwnServerSettings => ({ ...DEFAULT_SETTINGS.ownServer, ...p })

function facts(p: Partial<PreflightFacts> = {}): PreflightFacts {
  return {
    settings: settings(),
    gpus: [],
    runningServers: [],
    freeRamGiB: 50,
    totalRamGiB: 80,
    port: 8288,
    wslFound: true,
    comfyFound: true,
    pythonFound: true,
    ownRunning: false,
    ...p
  }
}
const codes = (f: PreflightFacts): string[] => buildIssues(f).map((i) => `${i.level}:${i.code}`)

describe('parseFree', () => {
  it('reads the available column', () => {
    const r = parseFree(`               total        used        free      shared  buff/cache   available
Mem:     84337573888 31884054528 43496075264   160976896  9837338624 52453519360
Swap:    21474836480           0 21474836480
`)
    expect(r.totalGiB).toBeCloseTo(84337573888 / GIB, 5)
    expect(r.availableGiB).toBeCloseTo(52453519360 / GIB, 5)
  })
  it('gives null without a Mem row', () => {
    expect(parseFree('nothing')).toEqual({ totalGiB: null, availableGiB: null })
  })
})

describe('buildIssues', () => {
  it('has no issues when all is well', () => {
    expect(codes(facts())).toEqual([])
  })
  it('blocks without WSL and does not pile on', () => {
    expect(codes(facts({ wslFound: false, comfyFound: false, pythonFound: false, port: null }))).toEqual(['block:wsl-missing'])
  })
  it('blocks on missing comfy, python, port and a running own server', () => {
    expect(codes(facts({ comfyFound: false, pythonFound: false }))).toEqual(['block:comfy-missing'])
    expect(codes(facts({ pythonFound: false }))).toEqual(['block:python-missing'])
    expect(codes(facts({ port: null }))).toEqual(['block:no-free-port'])
    expect(codes(facts({ ownRunning: true }))).toEqual(['block:already-running'])
  })
  it('asks to confirm when another server uses the chosen GPU', () => {
    const f = facts({
      settings: settings({ device: { kind: 'gpu', index: 0, name: 'RTX 5060 Ti' } }),
      gpus: [gpu(0, 'NVIDIA GeForce RTX 5060 Ti', 1000), gpu(1, 'NVIDIA GeForce RTX 4060 Ti', 0)],
      runningServers: [server(8188, [0]), server(8189, [1]), server(8288, [0], true)]
    })
    const issues = buildIssues(f)
    expect(issues).toHaveLength(1)
    expect(issues[0]).toMatchObject({ code: 'gpu-in-use-by-server', level: 'confirm' })
    expect(issues[0].message).toBe(
      'NVIDIA GeForce RTX 5060 Ti is used by the ComfyUI server on port 8188; a second server on it can run out of memory and slow its renders down.'
    )
  })
  it('asks to confirm on low RAM', () => {
    const i = buildIssues(facts({ freeRamGiB: 10 }))
    expect(i).toHaveLength(1)
    expect(i[0]).toMatchObject({ code: 'low-ram', level: 'confirm' })
    expect(codes(facts({ freeRamGiB: null }))).toEqual([])
  })
  it('informs about low VRAM', () => {
    const f = facts({ settings: settings({ device: { kind: 'gpu', index: 0, name: 'X' } }), gpus: [gpu(0, 'GPU Zero', 14000)] })
    expect(codes(f)).toEqual(['info:gpu-low-vram'])
    expect(codes(facts({ settings: settings({ device: { kind: 'cpu' } }), gpus: [gpu(0, 'GPU Zero', 16000)] }))).toEqual([])
  })
})

describe('runPreflight', () => {
  function wsl(stdout: string, code: number | null = 0): WslRunner {
    return {
      run: async (_d, script) => {
        if (script.startsWith('ss ')) return { code: 0, stdout: '', stderr: '' }
        return { code, stdout, stderr: '' }
      },
      spawn: () => {
        throw new Error('unused')
      }
    }
  }
  const OUT = 'RULZ_OK\nRULZ_COMFY\nRULZ_PYTHON\n               total used free shared buff/cache available\nMem: 85899345920 1 2 3 4 21474836480\n'

  it('probes the usual ports, skipping its own, and builds a report', async () => {
    const probed: string[] = []
    const r = await runPreflight({
      wsl: wsl(OUT),
      settings: settings({ portStart: 8288 }),
      listGpus: async () => [gpu(0, 'A', 0)],
      probe: async (url) => {
        probed.push(url)
        return url.endsWith(':8188') ? server(8188, [0]) : null
      },
      ownRunning: false,
      ownPort: 8290
    })
    expect(probed).toContain('http://127.0.0.1:8188')
    expect(probed).toContain('http://127.0.0.1:8199')
    expect(probed).toContain('http://127.0.0.1:8288')
    expect(probed).toContain('http://127.0.0.1:8299')
    expect(probed).not.toContain('http://127.0.0.1:8290')
    expect(r.runningServers.map((s) => s.port)).toEqual([8188])
    expect(r.freeRamGiB).toBeCloseTo(20, 5)
    expect(r.totalRamGiB).toBeCloseTo(80, 5)
    expect(r.port).not.toBeNull()
    expect(r.issues.map((i) => i.code)).toEqual(['low-ram'])
  })
  it('reports wsl-missing when the probe fails', async () => {
    const r = await runPreflight({ wsl: wsl('', null), settings: settings(), listGpus: async () => [], probe: async () => null, ownRunning: false, ownPort: null })
    expect(r.issues.map((i) => i.code)).toEqual(['wsl-missing'])
    expect(r.port).toBeNull()
  })
  it('reports missing python', async () => {
    const r = await runPreflight({
      wsl: wsl('RULZ_OK\nRULZ_COMFY\n'),
      settings: settings(),
      listGpus: async () => [],
      probe: async () => null,
      ownRunning: false,
      ownPort: null
    })
    expect(r.issues.map((i) => i.code)).toEqual(['python-missing'])
  })
})

describe('extra preflight rules', () => {
  it('blocks managed flags in the extra options, including --flag=value and abbreviations', () => {
    const i = buildIssues(facts({ settings: settings({ extraArgs: '--fast --port=9000 --cuda-device 1 --base-dir /x --cpu-vae --cpu' }) }))
    expect(i).toHaveLength(1)
    expect(i[0]).toMatchObject({ code: 'managed-arg', level: 'block' })
    for (const f of ['--port', '--cuda-device', '--base-directory', '--cpu']) expect(i[0].message).toContain(f)
    expect(i[0].message).not.toContain('--cpu-vae')
    expect(findManagedFlags('--fast --enable-x=1')).toEqual([])
    for (const f of MANAGED_FLAGS) expect(findManagedFlags(`${f}=1`)).toContain(f)
  })
  it('blocks paths that are not absolute or ~/', () => {
    expect(codes(facts({ settings: settings({ baseDir: 'comfy-base' }) }))).toEqual(['block:path-not-absolute'])
    expect(codes(facts({ settings: settings({ comfyDir: 'ComfyUI' }) }))).toEqual(['block:path-not-absolute'])
    expect(codes(facts({ settings: settings({ modelsDir: 'models' }) }))).toEqual(['block:path-not-absolute'])
    expect(codes(facts({ settings: settings({ baseDir: '/data/base', comfyDir: '~/ComfyUI', modelsDir: '/mnt/d/m' }) }))).toEqual([])
    expect(isAbsoluteWslPath('~foo')).toBe(false)
  })
  it('blocks with a plain message when the ports could not be checked', () => {
    const i = buildIssues(facts({ port: null, portCheckFailed: true }))
    expect(i).toEqual([{ code: 'ports-unreadable', level: 'block', message: 'Could not check which ports are in use in WSL; try again.' }])
  })
  it('asks to confirm when the chosen GPU is not in the nvidia-smi list', () => {
    const dev = { kind: 'gpu' as const, index: 3, name: 'RTX Gone' }
    expect(codes(facts({ settings: settings({ device: dev }), gpus: [] }))).toEqual(['confirm:gpu-unknown'])
    expect(codes(facts({ settings: settings({ device: dev }), gpus: [gpu(0, 'A', 0)] }))).toEqual(['confirm:gpu-unknown'])
    expect(buildIssues(facts({ settings: settings({ device: dev }) }))[0].message).toContain('RTX Gone')
  })
  it('picks the ports to probe: usual ones, then every listening port from 1024, capped at 40, without its own', () => {
    const listening = new Set<number>([22, 53, 1023, 1024, 3000, 8188, 9001, 9002])
    const p = portsToProbe(8288, listening, 8290)
    expect(p).toContain(8188)
    expect(p).toContain(8199)
    expect(p).toContain(8288)
    expect(p).toContain(3000)
    expect(p).toContain(9001)
    expect(p).toContain(1024)
    expect(p).not.toContain(1023)
    expect(p).not.toContain(53)
    expect(p).not.toContain(8290)
    const many = new Set<number>(Array.from({ length: 100 }, (_, i) => 20000 + i))
    expect(portsToProbe(8288, many, null)).toHaveLength(40)
    expect(portsToProbe(8288, many, null)).toContain(8188)
  })
})

describe('runPreflight with ss', () => {
  const OUT = 'RULZ_OK\nRULZ_COMFY\nRULZ_PYTHON\nMem: 85899345920 1 2 3 4 85899345920\n'
  function wslSs(ss: { code: number | null; stdout: string }): WslRunner {
    return {
      run: async (_d, script) => (script.startsWith('ss ') ? { ...ss, stderr: '' } : { code: 0, stdout: OUT, stderr: '' }),
      spawn: () => {
        throw new Error('unused')
      }
    }
  }
  it('probes servers that ss shows on other ports', async () => {
    const probed: string[] = []
    const r = await runPreflight({
      wsl: wslSs({ code: 0, stdout: 'LISTEN 0 128 0.0.0.0:7777 0.0.0.0:*\nLISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n' }),
      settings: settings(),
      listGpus: async () => [],
      probe: async (url) => {
        probed.push(url)
        return url.endsWith(':7777') ? server(7777, []) : null
      },
      ownRunning: false,
      ownPort: null
    })
    expect(probed).toContain('http://127.0.0.1:7777')
    expect(probed).not.toContain('http://127.0.0.1:22')
    expect(r.runningServers.map((s) => s.port)).toEqual([7777])
  })
  it('reports a block issue and no port when ss fails', async () => {
    const r = await runPreflight({ wsl: wslSs({ code: null, stdout: '' }), settings: settings(), listGpus: async () => [], probe: async () => null, ownRunning: false, ownPort: null })
    expect(r.port).toBeNull()
    expect(r.issues.map((i) => i.code)).toEqual(['ports-unreadable'])
  })
})
