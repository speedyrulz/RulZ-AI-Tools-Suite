// src/main/remote/remoteServer.ts and ops.ts with a fake ssh, fake tunnels and fake timers: no ssh, no network.
import { mkdtempSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { AutoStop, RemoteModelDef } from '@shared/remote'
import type { LoadConfig, MemoryEstimate, ModelEntry } from '@shared/types'
import type { RemoteLaunch } from '../src/main/engine/serverHandle'
import type { RentalDirectory, RentalInfo, Secrets } from '../src/main/remote/contract'
import { recordedComplete, toModelEntry } from '../src/main/remote/models'
import { memoryHashCache } from '../src/main/remote/transfer'
import { RateMeter, SshRentalOps, llamaAssetNames, type OpsSsh } from '../src/main/remote/ops'
import { createRemoteLauncher, serverHash, TUNNEL_GIVE_UP_MS, type RemoteLauncherDeps } from '../src/main/remote/remoteServer'
import { buildRemoteArgs } from '../src/main/remote/remoteArgs'
import { PORT_PLACEHOLDER } from '../src/main/remote/scripts'
import { SshError, type RunOptions, type RunResult, type SshProcess, type TunnelProc } from '../src/main/remote/ssh'

const SHA = 'c'.repeat(64)
const ROOT = '/workspace/rulz-llm'
const b64 = (s: string | Buffer): string => Buffer.from(s).toString('base64')

const libraryModel: ModelEntry = { id: 'fam/m.gguf', name: 'M', folder: 'fam', path: 'C:\\LLM\\fam\\m.gguf', files: ['C:\\LLM\\fam\\m.gguf'], sizeBytes: 1000, info: null, modifiedAt: 1 }
const def: RemoteModelDef = { id: 'vast/m', name: 'M', source: { kind: 'local', modelId: libraryModel.id }, createdAt: 1 }
const getModel = (id: string): ModelEntry | undefined => (id === libraryModel.id ? libraryModel : undefined)
const entry = toModelEntry(def, getModel)

const rental = (over: Partial<RentalInfo> = {}): RentalInfo => ({
  rentalId: 7,
  label: 'rulz-llm-ab12',
  origin: 'app',
  running: true,
  routes: [{ route: 'direct', host: '203.0.113.7', port: 41022 }],
  engineTag: 'b12407',
  dollarsPerHour: 0.5,
  ...over
})

type Reply = Partial<RunResult> | Error
interface Seen {
  kind: string
  text: string
  opts?: RunOptions
}

function kindOf(text: string): string {
  if (text.includes('sup.sh')) return 'start'
  if (text.includes('kill -TERM')) return 'stop'
  if (text.includes('LOG_NEXT')) return 'log'
  if (text.includes('COMPLETE_B64')) return 'check'
  return 'other'
}

const checkOut = (o: { engine?: boolean; complete?: Array<[string, number] | [string, number, string]>; files?: Array<[string, number]>; runs?: Array<{ id: string; port?: number; hash: string }>; foreign?: boolean; used?: number } = {}): string =>
  [
    `@rz ROOT=${ROOT}`,
    `@rz ENGINE_OK=${o.engine === false ? 0 : 1}`,
    `@rz COMPLETE_B64=${b64((o.complete ?? [['m.gguf', 1000]]).map(([n, s, src]) => `${n}\t${s}\t-\t${src ?? '-'}\n`).join(''))}`,
    ...(o.files ?? o.complete ?? [['m.gguf', 1000]]).map(([n, s]) => `@rz FILE=${n}|${s}`),
    `@rz GPU=0|NVIDIA Fake|24000|${o.used ?? 100}|0`,
    '@rz NVSMI=1',
    `@rz FOREIGN=${o.foreign ? 1 : 0}`,
    ...(o.runs ?? []).map((r) => `@rz RUN=${r.id}|${r.port ?? 31000}|${r.hash}`)
  ].join('\n')

const logOut = (text: string | Buffer, next: number, alive = true): string => `@rz ALIVE=${alive ? 1 : 0}\n@rz LOG=${b64(text)}\n@rz LOG_NEXT=${next}\n`

class FakeTunnel implements TunnelProc {
  killed = false
  private done!: (v: { code: number | null; stderr: string }) => void
  readonly exited = new Promise<{ code: number | null; stderr: string }>((r) => (this.done = r))
  constructor(
    readonly localPort: number,
    readonly remotePort: number
  ) {}
  kill(): void {
    this.killed = true
    this.done({ code: null, stderr: '' })
  }
  drop(stderr = 'Connection reset by peer'): void {
    this.done({ code: 255, stderr })
  }
}

interface Harness {
  deps: RemoteLauncherDeps
  seen: Seen[]
  tunnels: FakeTunnel[]
  notes: string[]
  routeFailed: ReturnType<typeof vi.fn>
  reply: { check: Reply; start: Reply; stop: Reply; log: Reply; other: Reply }
  logs: Array<Reply>
  failTunnel: { on: boolean }
  open: { value: boolean }
  info: { value: RentalInfo | null }
  launch: RemoteLaunch
}

function harness(over: Partial<RemoteLauncherDeps> = {}): Harness {
  const h = {
    seen: [] as Seen[],
    tunnels: [] as FakeTunnel[],
    notes: [] as string[],
    routeFailed: vi.fn(),
    reply: {
      check: { code: 0, stdout: checkOut() } as Reply,
      start: { code: 0, stdout: '@rz PORT=31000\n@rz PID=4242\n@rz STARTED=1\n' } as Reply,
      stop: { code: 0, stdout: '@rz STOPPED=1\n' } as Reply,
      log: { code: 0, stdout: logOut('', 0) } as Reply,
      other: { code: 0, stdout: '@rz ALIVE=1\n' } as Reply
    },
    logs: [] as Array<Reply>,
    failTunnel: { on: false },
    open: { value: true },
    info: { value: rental() as RentalInfo | null }
  }
  const directory: RentalDirectory = { info: () => h.info.value, use: () => () => undefined, note: () => undefined }
  const deps: RemoteLauncherDeps = {
    ssh: {
      run: async (_id, text, opts) => {
        const kind = kindOf(text)
        h.seen.push({ kind, text, opts })
        const r = kind === 'log' && h.logs.length ? (h.logs.shift() as Reply) : (h.reply as Record<string, Reply>)[kind]
        if (r instanceof Error) throw r
        return { code: 0, stdout: '', stderr: '', ...r }
      },
      tunnel: (_id, localPort, remotePort) => {
        if (h.failTunnel.on) throw new SshError('unreachable', 'no address')
        const t = new FakeTunnel(localPort, remotePort)
        h.tunnels.push(t)
        return t
      },
      routeFailed: h.routeFailed
    },
    directory,
    getDef: (id) => (id === def.id ? def : undefined),
    getModel,
    draftFor: () => undefined,
    pickPort: async () => 50111,
    portOpen: async () => h.open.value,
    ...over
  }
  const launch: RemoteLaunch = { model: entry, config: DEFAULT_LOAD_CONFIG, alias: 'm-alias', adapters: [], instanceId: 'inst-1', note: (l) => void h.notes.push(l) }
  return Object.assign(h, { deps, launch }) as unknown as Harness
}

const tick = async (ms = 0): Promise<void> => {
  await vi.advanceTimersByTimeAsync(ms)
}
const count = (h: Harness, kind: string): number => h.seen.filter((s) => s.kind === kind).length

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
})

describe('starting', () => {
  it('checks the instance, starts llama-server with paths on the instance and a placeholder port, and tunnels', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    expect(h.seen.map((s) => s.kind).slice(0, 2)).toEqual(['check', 'start'])
    const start = h.seen[1].text
    expect(start).toContain(`'${ROOT}/models/m/m.gguf'`)
    expect(start).toContain(`'${PORT_PLACEHOLDER}'`)
    expect(start).toContain("'127.0.0.1'")
    expect(start).not.toContain('C:\\')
    expect(handle.port).toBe(50111)
    expect(handle.pid).toBeNull()
    expect(handle.host).toEqual({ kind: 'vast', rentalId: 7, label: 'rulz-llm-ab12' })
    expect(h.tunnels).toHaveLength(1)
    expect([h.tunnels[0].localPort, h.tunnels[0].remotePort]).toEqual([50111, 31000])
    expect(handle.reconnecting()).toBe(false)
    expect(h.notes.some((n) => n.includes('Starting llama-server'))).toBe(true)
    handle.detachSync()
  })

  it('refuses with a sentence when the rental is not running, unknown, or not set up', async () => {
    const h = harness()
    h.info.value = rental({ running: false })
    await expect(createRemoteLauncher(h.deps).start(7, h.launch)).rejects.toThrow(/not running/)
    h.info.value = null
    await expect(createRemoteLauncher(h.deps).start(7, h.launch)).rejects.toThrow(/not in the list/)
    h.info.value = rental()
    h.reply.check = { stdout: checkOut({ engine: false }) }
    await expect(createRemoteLauncher(h.deps).start(7, h.launch)).rejects.toThrow(/engine b12407 is not installed/)
    h.reply.check = { stdout: checkOut({ complete: [] }) }
    await expect(createRemoteLauncher(h.deps).start(7, h.launch)).rejects.toThrow(/not complete.*m\.gguf/)
    h.reply.check = { stdout: checkOut({ complete: [['other.gguf', 5]] }) }
    await expect(createRemoteLauncher(h.deps).start(7, h.launch)).rejects.toThrow(/Set the model up/)
    expect(count(h, 'start')).toBe(0)
  })

  it('refuses a model that is not saved, and settings that cannot work there', async () => {
    const h = harness()
    await expect(createRemoteLauncher({ ...h.deps, getDef: () => undefined }).start(7, h.launch)).rejects.toThrow(/not saved/)
    const bad: LoadConfig = { ...DEFAULT_LOAD_CONFIG, chatTemplateFile: 'C:\\t.jinja' }
    await expect(createRemoteLauncher(h.deps).start(7, { ...h.launch, config: bad })).rejects.toThrow(/chat template/)
    expect(count(h, 'start')).toBe(0)
  })

  it('explains a failed start with the log tail the instance sent', async () => {
    const h = harness()
    h.reply.start = { code: 44, stdout: `@rz LOG=${b64('CUDA error: out of memory')}\n@rz ERR=llama-server stopped right after starting.\n` }
    await expect(createRemoteLauncher(h.deps).start(7, h.launch)).rejects.toThrow(/stopped right after starting[\s\S]*out of memory/)
    expect(h.tunnels).toHaveLength(0)
  })

  it('says plainly that the build on the rental does not know the model type, without the raw log', async () => {
    const h = harness()
    const log = [
      "0.00.562.697 E llama_model_load: error loading model: unknown model architecture: 'inkling2_loop'",
      '0.00.612.882 E srv  server_run_i: exiting due to model loading error'
    ].join('\n')
    h.reply.start = { code: 44, stdout: `@rz LOG=${b64(log)}\n@rz ERR=llama-server stopped right after starting.\n` }
    const err = (await createRemoteLauncher(h.deps).start(7, h.launch).catch((e: Error) => e)) as Error
    expect(err.message).toContain("build on rulz-llm-ab12 (b12407) does not support the model type 'inkling2_loop'")
    expect(err.message).toContain('llama.cpp build for instances')
    expect(err.message).not.toContain('exiting due to')
  })

  it('says the same when the server ends while it is still loading, and keeps the log tail once it had loaded', async () => {
    const line = "E llama_model_load: error loading model: unknown model architecture: 'inkling2_loop'\n"
    const reasonFor = async (text: string): Promise<string> => {
      const h = harness()
      const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
      const exits: Array<{ reason?: string }> = []
      handle.onExit((e) => exits.push(e))
      h.logs.push({ stdout: logOut(text, text.length, false) })
      await tick(1500)
      expect(exits).toHaveLength(1)
      return exits[0].reason ?? ''
    }
    expect(await reasonFor(line)).toMatch(/^The llama\.cpp build on rulz-llm-ab12 \(b12407\)/)
    expect(await reasonFor(`server is listening\n${line}`)).toMatch(/^llama-server stopped on rulz-llm-ab12\.\nLast lines of its log:/)
  })

  it('refuses on an attached instance with a foreign process or too little memory, and only notes it on an app rental', async () => {
    const estimate = (): MemoryEstimate => ({ gpuBytes: 30 * 2 ** 30 }) as MemoryEstimate
    const attached = harness({ estimate })
    attached.info.value = rental({ origin: 'attached' })
    attached.reply.check = { stdout: checkOut({ foreign: true }) }
    await expect(createRemoteLauncher(attached.deps).start(7, attached.launch)).rejects.toThrow(/Another program[\s\S]*GPU memory[\s\S]*not rented by this app/)
    expect(count(attached, 'start')).toBe(0)

    const app = harness({ estimate })
    app.reply.check = { stdout: checkOut({ foreign: true }) }
    const handle = await createRemoteLauncher(app.deps).start(7, app.launch)
    expect(app.notes.filter((n) => n.startsWith('Warning:'))).toHaveLength(2)
    expect(count(app, 'start')).toBe(1)
    handle.detachSync()
  })

  it('reuses a running server whose hash is the same and replaces one whose hash differs', async () => {
    const first = harness()
    const h1 = await createRemoteLauncher(first.deps).start(7, first.launch)
    h1.detachSync()
    const startText = first.seen.find((s) => s.kind === 'start')?.text ?? ''
    const hash = /printf '%s' '([0-9a-f]{64})' >"\$D\/hash"/.exec(startText)?.[1]
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
    const expected = serverHash(
      'b12407',
      buildRemoteArgs({ model: entry, config: DEFAULT_LOAD_CONFIG, port: PORT_PLACEHOLDER, alias: 'm-alias', paths: { main: `${ROOT}/models/m/m.gguf`, adapters: [] } }),
      [{ name: 'm.gguf', size: 1000, sha256: '-', source: '-' }]
    )
    expect(hash).toBe(expected)

    const same = harness()
    same.reply.check = { stdout: checkOut({ runs: [{ id: 'old-run', hash: hash as string, port: 31777 }] }) }
    const h2 = await createRemoteLauncher(same.deps).start(7, same.launch)
    expect(count(same, 'start')).toBe(0)
    expect(count(same, 'stop')).toBe(0)
    expect(same.notes.some((n) => n.includes('left running'))).toBe(true)
    await tick(10)
    expect(same.tunnels[0].remotePort).toBe(31777)
    h2.detachSync()

    const other = harness()
    other.reply.check = { stdout: checkOut({ runs: [{ id: 'old-run', hash: 'd'.repeat(64) }] }) }
    const h3 = await createRemoteLauncher(other.deps).start(7, other.launch)
    expect(other.seen.map((s) => s.kind).slice(0, 4)).toEqual(['check', 'stop', 'check', 'start'])
    expect(other.seen[1].text).toContain("'old-run'")
    h3.detachSync()
  })

  it('uses another hash when the engine, the arguments or the file sizes differ', () => {
    const args = ['--model', 'x']
    const files = [{ name: 'a.gguf', size: 1, sha256: '-', source: '-' }]
    const base = serverHash('b1', args, files)
    expect(serverHash('b2', args, files)).not.toBe(base)
    expect(serverHash('b1', ['--model', 'y'], files)).not.toBe(base)
    expect(serverHash('b1', args, [{ name: 'a.gguf', size: 2, sha256: '-', source: '-' }])).not.toBe(base)
    expect(serverHash('b1', args, [{ name: 'a.gguf', size: 1, sha256: SHA, source: '-' }])).not.toBe(base)
    expect(serverHash('b1', args, [...files])).toBe(base)
  })
})

describe('the tunnel', () => {
  it('is respawned with backoff when it drops, reports reconnecting meanwhile, and recovers', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    expect(handle.reconnecting()).toBe(false)
    h.open.value = false
    h.tunnels[0].drop()
    await tick(0)
    expect(handle.reconnecting()).toBe(true)
    expect(h.tunnels).toHaveLength(1)
    await tick(1000)
    expect(h.tunnels).toHaveLength(2)
    expect(h.tunnels[1].localPort).toBe(50111)
    expect(handle.reconnecting()).toBe(true)
    h.open.value = true
    await tick(300)
    expect(handle.reconnecting()).toBe(false)
    handle.detachSync()
  })

  it('keeps the same local port for the life of the handle and backs off more after repeated drops', async () => {
    const h = harness()
    h.open.value = false
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(0)
    for (let i = 0; i < 3; i++) {
      h.tunnels[h.tunnels.length - 1].drop()
      await tick(20_000)
    }
    expect(new Set(h.tunnels.map((t) => t.localPort))).toEqual(new Set([50111]))
    expect(h.tunnels.length).toBeGreaterThanOrEqual(4)
    handle.detachSync()
  })

  it('looks at the server over a new route after two minutes down and ends with the log tail when it is gone', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    h.logs.push({ stdout: logOut('line one\nCUDA error: out of memory\n', 40) })
    await tick(1500)
    const exits: Array<{ code: number | null; reason?: string }> = []
    handle.onExit((e) => exits.push(e))
    h.open.value = false
    h.failTunnel.on = true
    h.tunnels[0].drop()
    h.reply.other = { stdout: '@rz ALIVE=0\n' }
    // Polling cannot reach the instance either, so only the check over a new route finds out.
    h.reply.log = new SshError('unreachable', 'down')
    await tick(TUNNEL_GIVE_UP_MS + 30_000)
    expect(h.routeFailed).toHaveBeenCalled()
    expect(exits).toHaveLength(1)
    expect(exits[0].code).toBeNull()
    expect(exits[0].reason).toContain('no longer running')
    expect(exits[0].reason).toContain('CUDA error: out of memory')
    expect(handle.reconnecting()).toBe(false)
    const before = h.seen.length
    await tick(60_000)
    expect(h.seen.length).toBe(before)
  })

  it('ends with a connection sentence when the instance cannot be reached over a new route either', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    const exits: Array<{ code: number | null; reason?: string }> = []
    handle.onExit((e) => exits.push(e))
    h.open.value = false
    h.reply.other = new SshError('unreachable', 'rulz-llm-ab12 does not answer yet.')
    h.reply.log = new SshError('unreachable', 'rulz-llm-ab12 does not answer yet.')
    h.tunnels[0].drop()
    await tick(TUNNEL_GIVE_UP_MS + 30_000)
    expect(exits).toHaveLength(1)
    expect(exits[0].reason).toContain('could not be re-made for 2 minutes')
    expect(exits[0].reason).toContain('does not answer')
  })

  it('goes on while the server runs but gives up after a second window when no tunnel can be made', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    const exits: Array<{ code: number | null; reason?: string }> = []
    handle.onExit((e) => exits.push(e))
    h.open.value = false
    h.tunnels[0].drop('Connection reset by peer')
    for (let i = 0; i < 40; i++) {
      await tick(10_000)
      const last = h.tunnels[h.tunnels.length - 1]
      if (!last.killed) last.drop('Connection reset by peer')
    }
    expect(exits).toHaveLength(1)
    expect(exits[0].reason).toContain('still runs')
    expect(exits[0].reason).toContain('Connection reset by peer')
  })
})

describe('the log', () => {
  it('delivers output and reads on from the byte offset it was given', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    const chunks: string[] = []
    handle.onOutput((c) => chunks.push(c))
    h.logs.push({ stdout: logOut('hello ', 6) }, { stdout: logOut('world\n', 12) })
    await tick(2100)
    expect(chunks.join('')).toBe('hello world\n')
    const logCalls = h.seen.filter((s) => s.kind === 'log').map((s) => /off=(-?\d+)/.exec(s.text)?.[1])
    expect(logCalls.slice(0, 3)).toEqual(['0', '6', '12'])
    handle.detachSync()
  })

  it('replays what came before onOutput was called', async () => {
    const h = harness()
    h.reply.log = { stdout: logOut('early line\n', 11) }
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(50)
    const late: string[] = []
    handle.onOutput((c) => late.push(c))
    expect(late.join('')).toContain('early line')
    handle.detachSync()
  })

  it('does not split a multi-byte character that arrives in two reads', async () => {
    const h = harness()
    const bytes = Buffer.from('caf\u00e9 \u2713\n')
    const cut = bytes.indexOf(0xc3) + 1
    h.logs.push({ stdout: logOut(bytes.subarray(0, cut), cut) }, { stdout: logOut(bytes.subarray(cut), bytes.length) })
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    const chunks: string[] = []
    handle.onOutput((c) => chunks.push(c))
    await tick(2100)
    expect(chunks.join('')).toBe('caf\u00e9 \u2713\n')
    handle.detachSync()
  })

  it('polls every second while loading and every ten seconds once the server listens', async () => {
    const h = harness()
    h.reply.log = { stdout: logOut('', 0) }
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(5000)
    const loading = count(h, 'log')
    expect(loading).toBeGreaterThanOrEqual(5)
    expect(loading).toBeLessThanOrEqual(7)
    h.reply.log = { stdout: logOut('main: server is listening on http://127.0.0.1:31000\n', 55) }
    await tick(1100)
    const mark = count(h, 'log')
    h.reply.log = { stdout: logOut('', 55) }
    await tick(30_000)
    expect(count(h, 'log') - mark).toBeLessThanOrEqual(4)
    expect(count(h, 'log') - mark).toBeGreaterThanOrEqual(2)
    handle.detachSync()
  })

  it('reports a dead process with the tail of its log', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    const exits: Array<{ code: number | null; reason?: string }> = []
    handle.onExit((e) => exits.push(e))
    h.logs.push({ stdout: logOut('loading tensors\nggml_cuda: out of memory\n', 41, false) })
    await tick(1100)
    expect(exits).toHaveLength(1)
    expect(exits[0].reason).toContain('llama-server stopped')
    expect(exits[0].reason).toContain('ggml_cuda: out of memory')
    expect(h.tunnels.every((t) => t.killed)).toBe(true)
    // Called once; late listeners hear it too.
    const late: unknown[] = []
    handle.onExit((e) => late.push(e))
    expect(late).toHaveLength(1)
  })

  it('does not fail when a poll cannot reach the instance', async () => {
    const h = harness()
    h.reply.log = new SshError('unreachable', 'down')
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(5000)
    expect(handle.reconnecting()).toBe(false)
    handle.detachSync()
  })
})

describe('stop and detach', () => {
  it('stops on the instance, then closes the tunnel and reports the exit', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    const exits: Array<{ code: number | null; reason?: string }> = []
    handle.onExit((e) => exits.push(e))
    h.reply.log = { stdout: logOut('bye\n', 4, false) }
    await handle.stop()
    expect(h.seen.some((s) => s.kind === 'stop' && s.text.includes("'inst-1'"))).toBe(true)
    expect(h.tunnels[0].killed).toBe(true)
    expect(exits).toEqual([{ code: 0 }])
    expect(handle.reconnecting()).toBe(false)
    const calls = h.seen.length
    await tick(60_000)
    expect(h.seen.length).toBe(calls)
    await handle.stop()
  })

  it('rejects when the server may still run, keeping the tunnel and the polling', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    const exits: unknown[] = []
    handle.onExit((e) => exits.push(e))
    h.reply.stop = { code: 50, stdout: '@rz ERR=llama-server (pid 4242) is still running on the instance.\n' }
    await expect(handle.stop()).rejects.toThrow(/still running[\s\S]*may still be running/)
    expect(h.tunnels[0].killed).toBe(false)
    expect(exits).toHaveLength(0)
    const before = count(h, 'log')
    await tick(3000)
    expect(count(h, 'log')).toBeGreaterThan(before)
    // A later stop can succeed.
    h.reply.stop = { code: 0, stdout: '@rz STOPPED=1\n' }
    await handle.stop()
    expect(exits).toEqual([{ code: 0 }])
  })

  it('rejects when the instance cannot be reached to confirm, after retrying', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    h.reply.stop = new SshError('unreachable', 'does not answer')
    const p = handle.stop()
    const caught = p.catch((e: Error) => e)
    await tick(20_000)
    const err = (await caught) as Error
    expect(err.message).toMatch(/Could not confirm[\s\S]*does not answer[\s\S]*may still be running/)
    expect(count(h, 'stop')).toBe(3)
    expect(h.tunnels[0].killed).toBe(false)
    handle.detachSync()
  })

  it('does not retry a refusal that is not a connection problem', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    h.reply.stop = { code: 50, stdout: '@rz ERR=still running\n' }
    await expect(handle.stop()).rejects.toThrow()
    expect(count(h, 'stop')).toBe(1)
    handle.detachSync()
  })

  it('detachSync kills only the tunnel and the timers: no stop, no exit, no more calls', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    const exits: unknown[] = []
    handle.onExit((e) => exits.push(e))
    handle.detachSync()
    expect(h.tunnels[0].killed).toBe(true)
    const calls = h.seen.length
    await tick(120_000)
    expect(h.seen.length).toBe(calls)
    expect(count(h, 'stop')).toBe(0)
    expect(exits).toHaveLength(0)
    expect(h.tunnels).toHaveLength(1)
  })
})

// ---------- Ops ----------

const secrets: Secrets = {
  vastKey: async () => null,
  hfToken: async () => ({ value: 'hf_secret', source: 'saved' }),
  setVastKey: async () => undefined,
  setHfToken: async () => undefined,
  scrub: (t) => t.split('hf_secret').join('***')
}

function opsHarness(fetchImpl?: typeof fetch) {
  const runs: Array<{ id: number; text: string; opts?: RunOptions }> = []
  const procs: Array<{ writes: string[]; ended: boolean; killed: boolean; finish: (r: { code: number; error?: SshError }) => void }> = []
  const notes: string[] = []
  let reply: (text: string) => Reply = () => ({ code: 0, stdout: '' })
  let openFails = 0
  const ssh: OpsSsh = {
    run: async (id, text, opts) => {
      runs.push({ id, text, opts })
      const r = reply(text)
      if (r instanceof Error) throw r
      return { code: 0, stdout: '', stderr: '', ...r }
    },
    open: (): SshProcess => {
      if (openFails > 0) {
        openFails--
        throw new SshError('unreachable', 'no address')
      }
      let finish!: (r: { code: number; error?: SshError }) => void
      const done = new Promise<RunResult & { error?: SshError }>((res) => (finish = (r) => res({ code: r.code, stdout: '', stderr: '', error: r.error })))
      const p = { writes: [] as string[], ended: false, killed: false, finish }
      procs.push(p)
      return {
        write: (t) => void p.writes.push(t),
        end: () => {
          p.ended = true
        },
        kill: () => {
          p.killed = true
          finish({ code: -1 })
        },
        done
      }
    },
    routeFailed: vi.fn()
  }
  const directory: RentalDirectory = { info: () => rental(), use: () => () => undefined, note: (_i, l) => void notes.push(l) }
  const ops = new SshRentalOps({ directory, secrets, ssh, models: { get: (id) => (id === def.id ? def : undefined) }, getModel, fetch: fetchImpl })
  return { ops, runs, procs, notes, setReply: (f: (t: string) => Reply) => (reply = f), failOpen: (n: number) => (openFails = n), ssh }
}

describe('keepalive', () => {
  it('holds one connection, writes a line every 60 seconds and ends stdin when use ends', async () => {
    const o = opsHarness()
    o.ops.setInUse(7, true)
    o.ops.setInUse(7, true)
    expect(o.procs).toHaveLength(1)
    await tick(59_000)
    expect(o.procs[0].writes).toEqual([])
    await tick(1500)
    expect(o.procs[0].writes).toEqual(['\n'])
    await tick(120_000)
    expect(o.procs[0].writes).toHaveLength(3)
    o.ops.setInUse(7, false)
    expect(o.procs[0].ended).toBe(true)
    await tick(5000)
    expect(o.procs[0].killed).toBe(true)
    expect(o.procs).toHaveLength(1)
  })

  it('makes the connection again with backoff when it drops, and tells the ssh side the route failed', async () => {
    const o = opsHarness()
    o.ops.setInUse(7, true)
    o.procs[0].finish({ code: 255, error: new SshError('unreachable', 'reset') })
    await tick(0)
    expect(o.procs).toHaveLength(1)
    await tick(2100)
    expect(o.procs).toHaveLength(2)
    expect(o.ssh.routeFailed).toHaveBeenCalledWith(7)
    o.procs[1].finish({ code: 255, error: new SshError('unreachable', 'reset') })
    await tick(2100)
    expect(o.procs).toHaveLength(2)
    await tick(2100)
    expect(o.procs).toHaveLength(3)
    o.ops.setInUse(7, false)
    o.procs[2].finish({ code: 0 })
    await tick(120_000)
    expect(o.procs).toHaveLength(3)
  })

  it('retries when the connection cannot even be opened, and stops everything on disconnect', async () => {
    const o = opsHarness()
    o.failOpen(2)
    o.ops.setInUse(7, true)
    expect(o.procs).toHaveLength(0)
    await tick(2100)
    await tick(4100)
    expect(o.procs).toHaveLength(1)
    o.ops.disconnect(7)
    expect(o.procs[0].killed).toBe(true)
    await tick(300_000)
    expect(o.procs).toHaveLength(1)
  })

  it('touches keepalive once on request', async () => {
    const o = opsHarness()
    await o.ops.touch(7)
    expect(o.runs).toHaveLength(1)
    expect(o.runs[0].text).toContain('touch "$R/keepalive"')
  })
})

describe('the watchdog through ops', () => {
  const auto: AutoStop = { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false }

  it('arms with the limits and converts the cap from milliseconds to seconds', async () => {
    const o = opsHarness()
    o.setReply(() => ({ stdout: '@rz ARMED=1\n@rz VERIFY=200\n' }))
    const r = await o.ops.armWatchdog(7, { autoStop: auto, capAt: 1_900_000_000_000, origin: 'app' })
    expect(r).toEqual({ armed: true })
    expect(o.runs[0].text).toContain('idle=30\ncap=1900000000\norg=app')
  })

  it('returns the reason when the instance cannot stop itself', async () => {
    const o = opsHarness()
    o.setReply(() => ({ stdout: '@rz ARMED=0\n@rz ERR=This instance has no CONTAINER_API_KEY, so it cannot stop itself.\n' }))
    const r = await o.ops.armWatchdog(7, { autoStop: auto, capAt: null, origin: 'attached' })
    expect(r.armed).toBe(false)
    expect(r.error).toContain('CONTAINER_API_KEY')
  })

  it('only turns the limits off when both are off', async () => {
    const o = opsHarness()
    o.setReply(() => ({ stdout: '@rz ARMED=0\n' }))
    const r = await o.ops.armWatchdog(7, { autoStop: { ...auto, enabled: false }, capAt: null, origin: 'attached' })
    expect(r).toEqual({ armed: false })
    expect(o.runs[0].text).toContain("idle=0")
    expect(o.runs[0].text).not.toContain('console.vast.ai/api/v0/instances/$i/" 2>/dev/null; }\n' + 'wd ||')
  })

  it('builds the onstart script from the auto-stop settings, under 4000 characters', () => {
    const o = opsHarness()
    const s = o.ops.onstartScript({ autoStop: auto })
    expect(s.length).toBeLessThan(4000)
    expect(s).toContain('bi=30; bh=12;')
    const off = o.ops.onstartScript({ autoStop: { ...auto, enabled: false, capHours: null } })
    expect(off).toContain('bi=0; bh=0;')
  })

  it('reads the probe into the rental manager shape', async () => {
    const o = opsHarness()
    o.setReply(() => ({
      stdout: ['@rz ROOT=/workspace/rulz-llm', '@rz GPU=0|NVIDIA GeForce RTX 4090|24564|312|7', '@rz NVSMI=1', '@rz DISK_FREE=5000', '@rz ENGINE=b12407', '@rz MODEL=m', '@rz WATCHDOG_RUNNING=1', '@rz WATCHDOG_STATE=ok', '@rz FOREIGN=1'].join('\n')
    }))
    const p = await o.ops.probe(7)
    expect(p).toEqual({
      gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 4090', totalMiB: 24564, usedMiB: 312, utilization: 7 }],
      diskFreeBytes: 5000,
      foreignProcesses: true,
      engines: ['b12407'],
      watchdog: { armed: true },
      models: ['vast/m']
    })
    o.setReply(() => ({ stdout: '@rz NVSMI=0\n@rz FOREIGN=1\n@rz WATCHDOG_RUNNING=1\n@rz WATCHDOG_STATE=error: no key\n' }))
    const q = await o.ops.probe(7)
    expect(q.foreignProcesses).toBe(false)
    expect(q.watchdog).toEqual({ armed: false, error: 'no key' })
  })
})

describe('the engine', () => {
  const release = (digest = `sha256:${'e'.repeat(64)}`): typeof fetch =>
    (async (url: string | URL | Request) => {
      expect(String(url)).toBe('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b12407')
      const n = llamaAssetNames('b12407')
      return new Response(
        JSON.stringify({
          assets: [
            { name: n.llama, digest, size: 111 },
            { name: n.cudart, digest: `sha256:${'f'.repeat(64)}`, size: 222 }
          ]
        }),
        { status: 200 }
      )
    }) as typeof fetch

  it('reads both digests from the release and installs with them, reporting progress', async () => {
    const o = opsHarness(release())
    const progress: Array<{ stage: string; bytesDone: number; bytesTotal: number }> = []
    o.setReply(() => ({ stdout: '@rz ENGINE_OK=b12407\n' }))
    // The runner feeds lines to onLine: emulate it.
    const real = o.ssh.run
    o.ssh.run = async (id, text, opts) => {
      opts?.onLine?.('@rz STAGE=Downloading llama.cpp')
      opts?.onLine?.('@rz PROG=llama.cpp|50|111')
      opts?.onLine?.('@rz PROG=CUDA runtime|100|222')
      return real(id, text, opts)
    }
    await o.ops.installEngine(7, 'b12407', (p) => progress.push(p), new AbortController().signal)
    const script = o.runs[0].text
    expect(script).toContain('e'.repeat(64))
    expect(script).toContain('f'.repeat(64))
    expect(script).toContain('https://github.com/ggml-org/llama.cpp/releases/download/b12407/llama-b12407-bin-ubuntu-cuda-12.8-x64.tar.gz')
    expect(progress.some((p) => p.stage === 'Downloading llama.cpp' && p.bytesDone === 150 && p.bytesTotal === 333)).toBe(true)
    expect(progress[progress.length - 1].stage).toBe('Engine ready')
  })

  it('says what is wrong when the release, a package or its digest is missing', async () => {
    const miss = (async () => new Response('{}', { status: 404 })) as typeof fetch
    await expect(opsHarness(miss).ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)).rejects.toThrow(/no release b12407/)
    const limited = (async () => new Response('{}', { status: 403 })) as typeof fetch
    await expect(opsHarness(limited).ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)).rejects.toThrow(/rate limit/)
    const noAssets = (async () => new Response(JSON.stringify({ assets: [] }), { status: 200 })) as typeof fetch
    await expect(opsHarness(noAssets).ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)).rejects.toThrow(/has no package/)
    await expect(opsHarness(release('')).ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)).rejects.toThrow(/no sha256 digest/)
    const down = (async () => {
      throw new Error('offline')
    }) as typeof fetch
    await expect(opsHarness(down).ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)).rejects.toThrow(/Could not reach GitHub/)
  })

  it('fails with the instance sentence when the install script fails', async () => {
    const o = opsHarness(release())
    o.setReply(() => ({ code: 22, stdout: '@rz ERR=llama-server is not in the downloaded package.\n' }))
    await expect(o.ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)).rejects.toThrow(/not in the downloaded package/)
  })

  it('asks for the digests once per tag', async () => {
    const f = vi.fn(release())
    const o = opsHarness(f as unknown as typeof fetch)
    o.setReply(() => ({ stdout: '@rz ENGINE_OK=b12407\n' }))
    await o.ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)
    await o.ops.installEngine(7, 'b12407', () => undefined, new AbortController().signal)
    expect(f).toHaveBeenCalledTimes(1)
  })
})

describe('model files through ops', () => {
  const hfDef: RemoteModelDef = { id: 'vast/h', name: 'H', source: { kind: 'hf', repo: 'o/r', bucket: false, revision: 'main', file: 'a.gguf', mmproj: '' }, createdAt: 1 }
  const head = (status = 200, headers: Record<string, string> = { 'x-linked-size': '4096', 'x-linked-etag': `"${'9'.repeat(64)}"` }): typeof fetch =>
    (async (url: string | URL | Request, init?: RequestInit) => {
      expect(init?.method).toBe('HEAD')
      expect(init?.redirect).toBe('manual')
      expect(String(url)).toBe('https://huggingface.co/o/r/resolve/main/a.gguf')
      return new Response(null, { status, headers })
    }) as typeof fetch

  const opsFor = (fetchImpl: typeof fetch) => {
    const o = opsHarness(fetchImpl)
    const ops = new SshRentalOps({ directory: { info: () => rental(), use: () => () => undefined, note: () => undefined }, secrets, ssh: o.ssh, models: { get: () => hfDef }, getModel, fetch: fetchImpl })
    return { ...o, ops }
  }

  it('reads size and sha256 of a Hugging Face file from the first answer, without the token when that is enough', async () => {
    const seen: unknown[] = []
    const plain = head()
    const o = opsFor(((u: string, init?: RequestInit) => {
      seen.push((init?.headers as Record<string, string>).Authorization)
      return plain(u, init)
    }) as typeof fetch)
    const r = await o.ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', 'tok')
    expect(r).not.toHaveProperty('signedUrl')
    expect(seen).toEqual([undefined])
    expect(await o.ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', 'tok')).toEqual({ size: 4096, sha256: '9'.repeat(64) })
  })

  it('sends the token only to Hugging Face, in a header, and says what a refusal means', async () => {
    let auth: string | undefined
    const f = (async (_u: string | URL | Request, init?: RequestInit) => {
      auth = (init?.headers as Record<string, string>).Authorization
      return new Response(null, { status: 403 })
    }) as typeof fetch
    const o = opsFor(f)
    await expect(o.ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', 'hf_secret')).rejects.toThrow(/token with access/)
    expect(auth).toBe('Bearer hf_secret')
    await expect(opsFor(head(404)).ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', null)).rejects.toThrow(/no such file/)
  })

  it('reports no size when the request fails, so the instance asks itself', async () => {
    const o = opsFor((async () => {
      throw new Error('offline')
    }) as typeof fetch)
    expect(await o.ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', null)).toEqual({ size: null, sha256: null })
  })

  // ---- gated files: the token stays here ----
  const SIGNED = 'https://cas.example/x/y?sig='
  /** Hugging Face as a gated file sees it: 401 without the token, a redirect to a numbered signed address with it. */
  const gatedHub = (location: (n: number) => string | null = (n) => SIGNED + n) => {
    const calls: Array<{ url: string; auth: string | undefined }> = []
    let n = 0
    const f = (async (u: string | URL | Request, init?: RequestInit) => {
      const auth = (init?.headers as Record<string, string>).Authorization
      calls.push({ url: String(u), auth })
      if (!auth) return new Response(null, { status: 401 })
      const loc = location(++n)
      return new Response(null, { status: loc ? 302 : 200, headers: { 'x-linked-size': '4096', 'x-linked-etag': `"${'9'.repeat(64)}"`, ...(loc ? { location: loc } : {}) } })
    }) as typeof fetch
    return { f, calls }
  }
  const complete = (text: string): { stdout: string } => (text.includes('COMPLETE_B64') || text.includes('FILE=') ? { stdout: '@rz FILE=a.gguf|4096\n@rz COMPLETE_B64=\n' } : { stdout: '@rz COMPLETE=1\n' })

  it('downloads a gated file from a signed address and never sends the token to the instance or the storage host', async () => {
    const hub = gatedHub()
    const o = opsFor(hub.f)
    o.setReply((text) => (text.includes('curlc()') ? { stdout: '' } : complete(text)))
    await o.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal)
    const down = o.runs.filter((r) => r.text.includes('curlc()'))
    expect(down).toHaveLength(1)
    expect(String(down[0].opts?.stdin)).toMatch(/^url = "https:\/\/cas\.example\/x\/y\?sig=\d+"\n$/)
    expect(o.runs.every((r) => !r.text.includes('hf_secret') && !String(r.opts?.stdin ?? '').includes('hf_secret'))).toBe(true)
    expect(hub.calls.filter((c) => c.auth).every((c) => new URL(c.url).hostname === 'huggingface.co')).toBe(true)
  })

  it('gets a fresh address when one has expired, and gives up after some tries', async () => {
    const o = opsFor(gatedHub().f)
    let downloads = 0
    o.setReply((text) => (text.includes('curlc()') ? (++downloads < 3 ? { code: 36, stdout: '@rz ERR=refused\n' } : { stdout: '' }) : complete(text)))
    await o.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal)
    const used = o.runs.filter((r) => r.text.includes('curlc()')).map((r) => String(r.opts?.stdin))
    expect(new Set(used).size).toBe(3)
    const never = opsFor(gatedHub().f)
    let tries = 0
    never.setReply((text) => (text.includes('curlc()') ? (tries++, { code: 36, stdout: '@rz ERR=refused\n' }) : complete(text)))
    await expect(never.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal)).rejects.toThrow(/refused/)
    expect(tries).toBe(12)
  })

  it('refuses rather than send the token when a gated file has no signed address', async () => {
    for (const loc of [null, 'http://cas.example/plain', 'https://huggingface.co/o/r/raw/main/a.gguf']) {
      const o = opsFor(gatedHub(() => loc).f)
      o.setReply(complete)
      await expect(o.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal), String(loc)).rejects.toThrow(/not sent to the instance/)
      expect(o.runs.some((r) => r.text.includes('curlc()'))).toBe(false)
    }
  })

  it('follows a renamed repository only within huggingface.co', async () => {
    const calls: Array<{ url: string; auth: string | undefined }> = []
    const f = (async (u: string | URL | Request, init?: RequestInit) => {
      const auth = (init?.headers as Record<string, string>).Authorization
      calls.push({ url: String(u), auth })
      if (!auth) return new Response(null, { status: 401 })
      if (String(u).includes('/o/r/')) return new Response(null, { status: 307, headers: { location: '/o/new/resolve/main/a.gguf' } })
      return new Response(null, { status: 302, headers: { location: 'https://cas.example/z', 'x-linked-size': '4096' } })
    }) as typeof fetch
    const r = await opsFor(f).ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', 'hf_secret')
    expect(r).toEqual({ size: 4096, sha256: null, signedUrl: 'https://cas.example/z' })
    expect(calls.map((c) => c.url)).toEqual([
      'https://huggingface.co/o/r/resolve/main/a.gguf',
      'https://huggingface.co/o/r/resolve/main/a.gguf',
      'https://huggingface.co/o/new/resolve/main/a.gguf'
    ])
  })

  it('does not take the length of an error page for the size', async () => {
    const o = opsFor((async () => new Response(null, { status: 429, headers: { 'content-length': '29' } })) as typeof fetch)
    expect(await o.ops.headFile('https://huggingface.co/o/r/resolve/main/a.gguf', null)).toEqual({ size: null, sha256: null })
  })

  it('downloads a public file through one script without a token, then marks the files complete', async () => {
    const o = opsFor(head())
    const progress: Array<{ stage: string; bytesDone: number; bytesTotal: number }> = []
    o.setReply((text) => {
      if (text.includes('hf_secret')) throw new Error('token in script')
      if (text.includes('FILE=')) return { stdout: '@rz FILE=a.gguf|4096\n' }
      if (text.includes('COMPLETE_B64') && text.includes('PART')) return { stdout: '@rz FILE=a.gguf|4096\n@rz COMPLETE_B64=\n' }
      return { stdout: '@rz COMPLETE=1\n' }
    })
    const real = o.ssh.run
    o.ssh.run = async (id, text, opts) => {
      if (text.includes('curlc()')) {
        opts?.onLine?.('@rz STAGE=Downloading a.gguf')
        opts?.onLine?.('@rz PROG=a.gguf|1024|4096')
      }
      return real(id, text, opts)
    }
    await o.ops.transferModel(7, hfDef, (p) => progress.push(p), new AbortController().signal)
    const download = o.runs.find((r) => r.text.includes('curlc()'))
    expect(download?.opts?.stdin).toBe('\n')
    expect(o.runs.every((r) => !r.text.includes('hf_secret') && !String(r.opts?.stdin ?? '').includes('hf_secret'))).toBe(true)
    expect(download?.text).toContain(`'${'9'.repeat(64)}'`)
    expect(download?.opts?.timeoutMs).toBe(0)
    const mark = o.runs[o.runs.length - 1].text
    expect(mark).toContain("add 'a.gguf' 4096 '" + '9'.repeat(64) + "'")
    expect(progress.some((p) => p.stage === 'Downloading a.gguf' && p.bytesDone === 1024 && p.bytesTotal === 4096)).toBe(true)
    expect(progress[progress.length - 1].stage).toBe('Model ready')
  })

  describe('a running download', () => {
    const isDownload = (text: string): boolean => text.includes('curlc()')

    it('shows the speed in the stage; a part file that was already there does not count as speed, and the bar never goes back', async () => {
      const BIG = 400_000_000
      const o = opsFor(head(200, { 'x-linked-size': String(BIG), 'x-linked-etag': `"${'9'.repeat(64)}"` }))
      o.setReply((text) => (text.includes('COMPLETE_B64') ? { stdout: `@rz FILE=a.gguf|${BIG}\n` } : { stdout: '' }))
      const real = o.ssh.run
      o.ssh.run = async (id, text, opts) => {
        if (isDownload(text)) {
          opts?.onLine?.('@rz STAGE=Downloading a.gguf, 16 connections')
          vi.advanceTimersByTime(2000)
          opts?.onLine?.(`@rz PROG=a.gguf|300000000|${BIG}`)
          vi.advanceTimersByTime(4000)
          opts?.onLine?.(`@rz PROG=a.gguf|350000000|${BIG}`)
          opts?.onLine?.(`@rz PROG=a.gguf|340000000|${BIG}`)
          opts?.onLine?.('@rz STAGE=Checking a.gguf')
        }
        return real(id, text, opts)
      }
      const progress: Array<{ stage: string; bytesDone: number }> = []
      await o.ops.transferModel(7, hfDef, (p) => progress.push(p), new AbortController().signal)
      const shown = progress.filter((p) => p.stage.includes('a.gguf')).map((p) => [p.stage, p.bytesDone])
      expect(shown).toEqual([
        ['Downloading a.gguf, 16 connections', 0],
        ['Downloading a.gguf, 16 connections', 300_000_000],
        ['Downloading a.gguf, 16 connections, 100 Mbps', 350_000_000],
        ['Downloading a.gguf, 16 connections, 100 Mbps', 350_000_000],
        ['Checking a.gguf', 350_000_000]
      ])
    })

    /** A setup whose download never ends by itself, cancelled; the error it ends with. */
    const cancelled = async (o: ReturnType<typeof opsFor>): Promise<Error> => {
      const real = o.ssh.run
      o.ssh.run = (id, text, opts) => {
        if (!isDownload(text)) return real(id, text, opts)
        o.runs.push({ id, text, opts })
        return new Promise<RunResult>((_, reject) => opts?.signal?.addEventListener('abort', () => reject(new Error('aborted'))))
      }
      const ctl = new AbortController()
      const done = o.ops.transferModel(7, hfDef, () => undefined, ctl.signal).catch((e: Error) => e)
      await tick(10)
      ctl.abort()
      return (await done) as Error
    }

    it('is stopped on the instance when the setup is cancelled, with its own short limit', async () => {
      const o = opsFor(head())
      o.setReply(complete)
      expect((await cancelled(o)).message).toBe('aborted')
      const stop = o.runs[o.runs.length - 1]
      expect(stop.text).toContain('STOPPED')
      expect(stop.opts?.timeoutMs).toBe(25_000)
      expect(stop.opts?.signal).toBeUndefined()
    })

    it('says in the log when the stop could not be sent, and still ends with the cancel', async () => {
      const o = opsFor(head())
      o.setReply((text) => (text.includes('STOPPED') ? new SshError('unreachable', 'down') : complete(text)))
      expect((await cancelled(o)).message).toBe('aborted')
      expect(o.ops.log(7).some((l) => l.includes('Could not stop the download on the instance'))).toBe(true)
    })

    it('is not stopped on the instance when it fails without a cancel', async () => {
      const o = opsFor(head())
      o.setReply((text) => (isDownload(text) ? new SshError('unreachable', 'down') : complete(text)))
      await expect(o.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal)).rejects.toThrow(/down/)
      expect(o.runs.some((r) => r.text.includes('STOPPED'))).toBe(false)
    })
  })

  it('measures a rate over the last ten seconds, and none before two seconds are covered', () => {
    let now = 1000
    const m = new RateMeter(() => now)
    expect(m.mbps()).toBeNull()
    m.add(0)
    expect(m.mbps()).toBeNull()
    now += 1000
    m.add(10_000_000)
    expect(m.mbps()).toBeNull()
    now += 1000
    m.add(50_000_000)
    expect(m.mbps()).toBe(200)
    // Fast for a while, then slow: the old burst leaves the window.
    for (let i = 0; i < 12; i++) {
      now += 1000
      m.add(50_000_000 + (i + 1) * 1_000_000)
    }
    expect(m.mbps()).toBe(8)
    now += 15000
    m.add(62_000_000)
    expect(m.mbps()).toBe(0)
  })

  it('fails when a file on the instance has another size than expected', async () => {
    const o = opsFor(head())
    o.setReply((text) => (text.includes('FILE=') || text.includes('COMPLETE_B64') ? { stdout: '@rz FILE=a.gguf|10\n' } : { stdout: '' }))
    await expect(o.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal)).rejects.toThrow(/has 10 bytes on the instance, expected 4096/)
  })

  it('reports the sentence of a failed download', async () => {
    const o = opsFor(head())
    o.setReply((text) => (text.includes('curlc()') ? { code: 31, stdout: '@rz ERR=Hugging Face refused the download of a.gguf (http 401).\n' } : { stdout: '' }))
    await expect(o.ops.transferModel(7, hfDef, () => undefined, new AbortController().signal)).rejects.toThrow(/refused the download/)
  })

  it('counts what a setup still has to fetch: nothing when complete, the rest of a part file otherwise, null when a size is unknown', async () => {
    const complete = opsFor(head())
    complete.setReply(() => ({ stdout: `@rz FILE=a.gguf|4096\n@rz COMPLETE_B64=${b64('a.gguf\t4096\t-\n')}\n` }))
    expect(await complete.ops.missingBytes(7, hfDef)).toBe(0)
    const partial = opsFor(head())
    partial.setReply(() => ({ stdout: '@rz PART=a.gguf.part|1000\n@rz COMPLETE_B64=\n' }))
    expect(await partial.ops.missingBytes(7, hfDef)).toBe(3096)
    const nothing = opsFor(head())
    nothing.setReply(() => ({ stdout: '@rz COMPLETE_B64=\n' }))
    expect(await nothing.ops.missingBytes(7, hfDef)).toBe(4096)
    const unknown = opsFor((async () => new Response(null, { status: 200 })) as typeof fetch)
    unknown.setReply(() => ({ stdout: '' }))
    expect(await unknown.ops.missingBytes(7, hfDef)).toBeNull()
  })

  it('removes the files of a model, and says when a server runs on it', async () => {
    const o = opsFor(head())
    o.setReply(() => ({ stdout: '@rz REMOVED=1\n' }))
    await o.ops.removeModelFiles(7, 'vast/h')
    expect(o.runs[0].text).toContain("S='h'")
    o.setReply(() => ({ code: 60, stdout: '@rz ERR=A server is running on this model. Unload it first.\n' }))
    await expect(o.ops.removeModelFiles(7, 'vast/h')).rejects.toThrow(/Unload it first/)
  })

  it('keeps a log without secrets', async () => {
    const o = opsFor(head())
    o.setReply(() => ({ code: 22, stdout: '@rz ERR=failed with hf_secret in it\n' }))
    await expect(o.ops.removeModelFiles(7, 'vast/h')).rejects.toThrow()
    expect(o.ops.log(7).join('\n')).not.toContain('hf_secret')
  })
})

describe('after the tunnel gave up, and across loads', () => {
  const giveUp = async (h: Harness, handle: Awaited<ReturnType<ReturnType<typeof createRemoteLauncher>['start']>>): Promise<Array<{ code: number | null; reason?: string }>> => {
    const exits: Array<{ code: number | null; reason?: string }> = []
    handle.onExit((e) => exits.push(e))
    h.open.value = false
    h.reply.other = new SshError('unreachable', 'down')
    h.reply.log = new SshError('unreachable', 'down')
    h.tunnels[0].drop()
    await tick(TUNNEL_GIVE_UP_MS + 30_000)
    return exits
  }

  it('still stops on the instance after a transport exit, because the server may run', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    const exits = await giveUp(h, handle)
    expect(exits).toHaveLength(1)
    h.reply.stop = { code: 0, stdout: '@rz STOPPED=1\n' }
    await handle.stop()
    expect(count(h, 'stop')).toBe(1)
    expect(exits).toHaveLength(1)
    // Now it is confirmed: stopping again asks nothing.
    await handle.stop()
    expect(count(h, 'stop')).toBe(1)
  })

  it('rejects a stop after a transport exit while the instance cannot be reached', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    await tick(10)
    await giveUp(h, handle)
    h.reply.stop = new SshError('unreachable', 'down')
    const caught = handle.stop().catch((e: Error) => e)
    await tick(20_000)
    expect(((await caught) as Error).message).toMatch(/may still be running/)
  })

  it('makes stop a no-op after an exit that is known to be real (the process is gone)', async () => {
    const h = harness()
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    h.logs.push({ stdout: logOut('bye\n', 4, false) })
    await tick(1100)
    await handle.stop()
    expect(count(h, 'stop')).toBe(0)
  })

  it('takes over a server left behind with the same settings under its own run id, and stops others nobody owns', async () => {
    const first = harness()
    const h1 = await createRemoteLauncher(first.deps).start(7, first.launch)
    h1.detachSync()
    const hash = /printf '%s' '([0-9a-f]{64})' >"\$D\/hash"/.exec(first.seen.find((s) => s.kind === 'start')?.text ?? '')?.[1] as string

    const h = harness()
    h.reply.check = { stdout: checkOut({ runs: [{ id: 'left-a', port: 31111, hash: 'e'.repeat(64) }, { id: 'left-b', port: 31222, hash }] }) }
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    expect(count(h, 'start')).toBe(0)
    const stops = h.seen.filter((s) => s.kind === 'stop')
    expect(stops).toHaveLength(1)
    expect(stops[0].text).toContain("'left-a'")
    await tick(1200)
    expect(h.tunnels[0].remotePort).toBe(31222)
    expect(h.seen.filter((s) => s.kind === 'log').every((s) => s.text.includes("'left-b'"))).toBe(true)
    handle.detachSync()
  })

  it('leaves the servers of live handles of this process alone, also from a second load', async () => {
    const h = harness()
    // The first server is owned by a live handle and must not be stopped by a second load.
    const launcher = createRemoteLauncher(h.deps)
    const one = await launcher.start(7, { ...h.launch, instanceId: 'inst-1' })
    h.reply.check = { stdout: checkOut({ runs: [{ id: 'inst-1', port: 31000, hash: 'f'.repeat(64) }, { id: 'stray', port: 31001, hash: 'f'.repeat(64) }] }) }
    const two = await launcher.start(7, { ...h.launch, instanceId: 'inst-2' })
    const stopped = h.seen.filter((s) => s.kind === 'stop').map((s) => /ID='([^']+)'/.exec(s.text)?.[1])
    expect(stopped).toEqual(['stray'])
    expect(count(h, 'start')).toBe(2)
    for (const x of [one, two]) x.detachSync()
  })

  it('stops the orphan of an earlier load before starting a new server next to it', async () => {
    const h = harness()
    const launcher = createRemoteLauncher(h.deps)
    const first = await launcher.start(7, { ...h.launch, instanceId: 'inst-1' })
    await tick(10)
    const exits = await giveUp(h, first)
    expect(exits).toHaveLength(1)
    h.reply.check = { stdout: checkOut({ runs: [{ id: 'inst-1', port: 31000, hash: 'f'.repeat(64) }] }) }
    h.reply.other = { stdout: '@rz ALIVE=1\n' }
    h.reply.log = { stdout: logOut('', 0) }
    h.open.value = true
    const second = await launcher.start(7, { ...h.launch, instanceId: 'inst-2' })
    const kinds = h.seen.map((s) => s.kind).filter((k) => k !== 'log')
    expect(kinds.slice(-4)).toEqual(['check', 'stop', 'check', 'start'])
    expect(h.seen.filter((s) => s.kind === 'stop').pop()?.text).toContain("'inst-1'")
    second.detachSync()
  })
})

describe('what counts as a complete file', () => {
  const run = async (check: string, over: Partial<RemoteLauncherDeps> = {}, def2?: RemoteModelDef): Promise<Error | null> => {
    const h = harness(over)
    h.reply.check = { stdout: check }
    if (def2) {
      h.deps.getDef = () => def2
      h.launch.model = toModelEntry(def2, getModel)
    }
    try {
      const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
      handle.detachSync()
      return null
    } catch (err) {
      return err as Error
    }
  }

  it('refuses a file whose real size is not the recorded one, or that is missing on disk', async () => {
    expect(await run(checkOut())).toBeNull()
    expect((await run(checkOut({ files: [['m.gguf', 999]] })))?.message).toMatch(/not complete.*m\.gguf/)
    expect((await run(checkOut({ files: [] })))?.message).toMatch(/not complete/)
  })

  it('refuses a Hugging Face file recorded from another address, accepts the same address and an old line without one', async () => {
    const hfDef: RemoteModelDef = { id: 'vast/m', name: 'M', source: { kind: 'hf', repo: 'o/r', bucket: false, revision: 'v2', file: 'm.gguf', mmproj: '' }, createdAt: 1 }
    const here = 'https://huggingface.co/o/r/resolve/v2/m.gguf'
    expect(await run(checkOut({ complete: [['m.gguf', 1000, here]] }), {}, hfDef)).toBeNull()
    expect(await run(checkOut({ complete: [['m.gguf', 1000]] }), {}, hfDef)).toBeNull()
    const err = await run(checkOut({ complete: [['m.gguf', 1000, 'https://huggingface.co/o/r/resolve/main/m.gguf']] }), {}, hfDef)
    expect(err?.message).toMatch(/not complete.*m\.gguf/)
  })

  it('refuses an uploaded file whose cached hash on this PC differs from the recorded one', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'rulz-up-'))
    const file = join(dir, 'm.gguf')
    writeFileSync(file, 'x'.repeat(1000))
    const st = statSync(file)
    const model: ModelEntry = { ...libraryModel, path: file, files: [file], sizeBytes: 1000 }
    const cache = memoryHashCache()
    cache.set(file + '|' + st.size + '|' + Math.round(st.mtimeMs), 'e'.repeat(64))
    const over = { getModel: (id: string) => (id === model.id ? model : undefined), hashCache: cache }
    const complete = (sha: string): string => [
      '@rz ROOT=' + ROOT, '@rz ENGINE_OK=1', '@rz COMPLETE_B64=' + b64('m.gguf\t1000\t' + sha + '\t-\n'), '@rz FILE=m.gguf|1000', '@rz GPU=0|NVIDIA Fake|24000|100|0', '@rz NVSMI=1', '@rz FOREIGN=0'
    ].join('\n')
    expect(await run(complete('e'.repeat(64)), over)).toBeNull()
    expect((await run(complete('d'.repeat(64)), over))?.message).toMatch(/not complete/)
    expect(await run(complete('-'), over)).toBeNull()
  })

  it('counts a Hugging Face projector so that --mmproj is passed and the model shows as a vision model', async () => {
    const hfDef: RemoteModelDef = { id: 'vast/m', name: 'M', source: { kind: 'hf', repo: 'o/r', bucket: false, revision: 'main', file: 'm.gguf', mmproj: 'mm/mmproj-F16.gguf' }, createdAt: 1 }
    const h = harness()
    h.deps.getDef = () => hfDef
    h.launch.model = toModelEntry(hfDef, getModel)
    expect(h.launch.model.mmproj).toEqual({ path: 'hf:o/r/mm/mmproj-F16.gguf', sizeBytes: 0 })
    h.reply.check = { stdout: checkOut({ complete: [['m.gguf', 1000], ['mmproj-F16.gguf', 50]] }) }
    const handle = await createRemoteLauncher(h.deps).start(7, h.launch)
    const start = h.seen.find((s) => s.kind === 'start')?.text ?? ''
    expect(start).toContain("'--mmproj' '" + ROOT + "/models/m/mmproj-F16.gguf'")
    handle.detachSync()
  })
})

describe('recordedComplete', () => {
  const file = { role: 'main' as const, name: 'a.gguf', localPath: null, url: 'https://huggingface.co/o/r/resolve/main/a.gguf', size: null, sha256: null }
  it('needs the record, the disk and the expected size to agree', () => {
    const c = { size: 10, sha256: '-', source: '-' }
    expect(recordedComplete(file, c, 10)).toBe(true)
    expect(recordedComplete(file, c, 9)).toBe(false)
    expect(recordedComplete(file, c, undefined)).toBe(false)
    expect(recordedComplete(file, undefined, 10)).toBe(false)
    expect(recordedComplete({ ...file, size: 11 }, c, 10)).toBe(false)
    expect(recordedComplete(file, { ...c, source: 'https://huggingface.co/other' }, 10)).toBe(false)
    expect(recordedComplete(file, { ...c, source: file.url }, 10)).toBe(true)
  })

  it('a local file replaced by another of the same size is not complete; one with only a new date is, once hashed', () => {
    const local = { ...file, url: null, localPath: 'C:/m/a.gguf', size: 10 }
    const c = { size: 10, sha256: SHA, source: 'pc:1000' }
    expect(recordedComplete(local, c, 10, { source: 'pc:1000' })).toBe(true)
    expect(recordedComplete(local, c, 10, { source: 'pc:2000' })).toBe(false)
    expect(recordedComplete(local, c, 10, { source: 'pc:2000', sha256: SHA })).toBe(true)
    expect(recordedComplete(local, c, 10, { source: 'pc:1000', sha256: 'd'.repeat(64) })).toBe(false)
    // Not readable on this PC just now, or recorded without a date.
    expect(recordedComplete(local, c, 10, {})).toBe(true)
    expect(recordedComplete(local, { ...c, source: '-' }, 10, { source: 'pc:2000' })).toBe(true)
  })
})

describe('two loads on one rental', () => {
  it('start one after another, so the second sees the server of the first as owned', async () => {
    const h = harness()
    const order: string[] = []
    const run = h.deps.ssh.run
    let open: () => void = () => undefined
    const gate = new Promise<void>((r) => (open = r))
    h.deps.ssh.run = async (id, text, opts) => {
      const kind = text.includes('sup.sh') ? 'start' : text.includes('FOREIGN') ? 'check' : 'other'
      if (kind !== 'other') order.push(kind)
      if (kind === 'start' && order.filter((k) => k === 'start').length === 1) await gate
      return run(id, text, opts)
    }
    const launcher = createRemoteLauncher(h.deps)
    const a = launcher.start(7, h.launch)
    const b = launcher.start(7, { ...h.launch, instanceId: 'inst-2' })
    await tick(10)
    expect(order).toEqual(['check', 'start'])
    // What the second check finds: the first load's server, with the hash the second would also have.
    const hash = /printf '%s' '([0-9a-f]{64})' >"\$D\/hash"/.exec(h.seen.find((s) => s.kind === 'start')?.text ?? '')?.[1] ?? 'e'.repeat(64)
    h.reply.check = { stdout: checkOut({ runs: [{ id: 'inst-1', port: 31000, hash }] }) }
    open()
    const [ha, hb] = await Promise.all([a, b])
    expect(order).toEqual(['check', 'start', 'check', 'start'])
    expect(count(h, 'stop')).toBe(0)
    ha.detachSync()
    hb.detachSync()
  })

  it('a load that fails after the server was found or started leaves nothing owned', async () => {
    const h = harness()
    const launcher = createRemoteLauncher({
      ...h.deps,
      pickPort: async () => {
        throw new Error('no port')
      }
    })
    await expect(launcher.start(7, h.launch)).rejects.toThrow('no port')
    h.reply.check = { stdout: checkOut({ runs: [{ id: 'inst-1', port: 31000, hash: 'f'.repeat(64) }] }) }
    await expect(launcher.start(7, { ...h.launch, instanceId: 'inst-2' })).rejects.toThrow('no port')
    expect(count(h, 'stop')).toBe(1)
  })
})

describe('nothing is left running', () => {
  it('detachSync leaves no timer and no tunnel, and the handle is not owned any more', async () => {
    const h = harness()
    const launcher = createRemoteLauncher(h.deps)
    const handle = await launcher.start(7, h.launch)
    await tick(10)
    handle.detachSync()
    await tick(0)
    expect(vi.getTimerCount()).toBe(0)
    expect(h.tunnels.every((t) => t.killed)).toBe(true)
    // Not owned any more: a later load may take over or stop what is still running there.
    h.reply.check = { stdout: checkOut({ runs: [{ id: 'inst-1', port: 31000, hash: 'f'.repeat(64) }] }) }
    const again = await launcher.start(7, { ...h.launch, instanceId: 'inst-2' })
    expect(h.seen.filter((s) => s.kind === 'stop')).toHaveLength(1)
    again.detachSync()
  })

  it('disconnect ends the held connection and every timer of the keepalive at once', async () => {
    const o = opsHarness()
    o.ops.setInUse(7, true)
    o.procs[0].finish({ code: 255, error: new SshError('unreachable', 'reset') })
    await tick(10)
    o.ops.setInUse(7, true)
    o.ops.disconnect(7)
    await tick(0)
    expect(vi.getTimerCount()).toBe(0)
    expect(o.procs).toHaveLength(1)
  })

  it('arming sends the cap hours for the next boot', async () => {
    const o = opsHarness()
    o.setReply(() => ({ stdout: '@rz ARMED=1\n' }))
    await o.ops.armWatchdog(7, { autoStop: { enabled: true, idleMinutes: 20, capHours: 8, apiCountsAsUse: false }, capAt: 1_900_000_000_000, origin: 'app' })
    expect(o.runs[0].text).toContain(' 20 8 >"$R/boot"')
    await o.ops.armWatchdog(7, { autoStop: { enabled: true, idleMinutes: 20, capHours: null, apiCountsAsUse: false }, capAt: 1_900_000_000_000, origin: 'app' })
    expect(o.runs[1].text).toContain(' 20 0 >"$R/boot"')
  })
})

describe('loads in different modes with different draft files', () => {
  const draftA = 'C:\\LLM\\a\\draft.gguf'
  const draftB = 'C:\\LLM\\b\\draft.gguf'
  const base = { ...DEFAULT_LOAD_CONFIG, useDraft: true, draftFile: draftA }
  const modeB = { ...DEFAULT_LOAD_CONFIG, useDraft: true, draftFile: draftB }
  // Both drafts have the same file name, so the full list names the second one apart.
  const complete: Array<[string, number]> = [
    ['m.gguf', 1000],
    ['draft.gguf', 10],
    ['draft-draft.gguf', 10]
  ]

  async function startWith(config: typeof base) {
    const h = harness({
      getDefaults: () => base,
      getModeLoads: () => [modeB],
      draftFor: (_m, cfg) => ({ path: cfg.draftFile ?? '', sizeBytes: 10, kind: 'eagle3' })
    })
    h.reply.check = { code: 0, stdout: checkOut({ complete }) }
    const handle = await createRemoteLauncher(h.deps).start(7, { ...h.launch, config })
    await tick(10)
    handle.detachSync()
    return h.seen[1].text
  }

  it('each load uses the draft file its own settings name', async () => {
    const a = await startWith(base)
    expect(a).toContain(`'${ROOT}/models/m/draft.gguf'`)
    expect(a).not.toContain('draft-draft.gguf')
    const b = await startWith(modeB)
    expect(b).toContain(`'${ROOT}/models/m/draft-draft.gguf'`)
    expect(b).not.toContain(`'${ROOT}/models/m/draft.gguf'`)
  })
})
