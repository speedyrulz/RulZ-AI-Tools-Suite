import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { ModelEntry } from '@shared/types'
import type { EngineService } from '../src/main/engine/engine'
import { InstanceManager, type InstanceManagerDeps } from '../src/main/engine/instances'
import { parseTasklistPids } from '../src/main/engine/process'

const model: ModelEntry = {
  id: 'a',
  name: 'a',
  folder: '',
  path: 'C:\\LLM\\a.gguf',
  files: ['C:\\LLM\\a.gguf'],
  sizeBytes: 1e9,
  info: null,
  modifiedAt: 0
}

/** A stand-in for llama-server that never becomes ready and exits only when told to. */
class FakeServer extends EventEmitter {
  stdout = new EventEmitter()
  stderr = new EventEmitter()
  constructor(
    readonly name: string,
    readonly pid?: number
  ) {
    super()
  }
}

function manager(events: string[], overrides: Partial<InstanceManagerDeps> = {}): InstanceManager {
  return new InstanceManager({
    engine: { resolveExe: async () => 'C:\\llama\\llama-server.exe' } as unknown as EngineService,
    getModel: (id) => ({ ...model, id, name: id }),
    draftFor: () => undefined,
    getDefaults: () => DEFAULT_LOAD_CONFIG,
    unloadPreviousOnLoad: () => true,
    estimate: () => null,
    pidFile: join(mkdtempSync(join(tmpdir(), 'rulz-inst-')), 'pids.json'),
    onChange: () => undefined,
    onError: () => undefined,
    spawnServer: (_exe, args) => {
      const name = args[args.indexOf('--alias') + 1]
      events.push(`spawn ${name}`)
      return new FakeServer(name) as unknown as ChildProcess
    },
    killProcess: (proc, force) => {
      const p = proc as unknown as FakeServer
      events.push(`${force ? 'force-kill' : 'kill'} ${p.name}`)
      setTimeout(() => p.emit('exit', 0), 5)
    },
    ...overrides
  })
}

describe('InstanceManager.load', () => {
  it('serialises overlapping loads so the previous model is gone before the next one starts', async () => {
    const events: string[] = []
    const mgr = manager(events)
    const a = mgr.load('a', null)
    const b = mgr.load('b', null)
    a.catch(() => undefined)
    b.catch(() => undefined)
    await vi.waitFor(() => expect(events).toEqual(['spawn a', 'kill a', 'spawn b']))
    expect(mgr.list().map((i) => i.modelName)).toEqual(['b'])
    await expect(a).rejects.toThrow(/cancelled/)
    await mgr.unloadAll()
    expect(mgr.list()).toEqual([])
  })

  it('a failed start does not block the next load', async () => {
    const events: string[] = []
    const mgr = manager(events, { getModel: (id) => (id === 'missing' ? undefined : { ...model, id, name: id }) })
    await expect(mgr.load('missing', null)).rejects.toThrow(/not found/)
    const b = mgr.load('b', null)
    b.catch(() => undefined)
    await vi.waitFor(() => expect(events).toEqual(['spawn b']))
    await mgr.unloadAll()
  })
})

describe('InstanceManager.unload', () => {
  it('force-kills a server that ignores the first kill and waits for it before returning', async () => {
    const events: string[] = []
    const mgr = manager(events, {
      // A live PID so the manager believes the process is still there; nothing is really killed.
      spawnServer: (_exe, args) => {
        const name = args[args.indexOf('--alias') + 1]
        events.push(`spawn ${name}`)
        return new FakeServer(name, process.pid) as unknown as ChildProcess
      },
      killProcess: (proc, force) => {
        const p = proc as unknown as FakeServer
        events.push(`${force ? 'force-kill' : 'kill'} ${p.name}`)
        if (force) setTimeout(() => p.emit('exit', 1), 5)
      },
      unloadWaitMs: [30, 1000]
    })
    mgr.load('a', null).catch(() => undefined)
    await vi.waitFor(() => expect(mgr.list()).toHaveLength(1))
    await mgr.unload(mgr.list()[0].instanceId)
    expect(events).toEqual(['spawn a', 'kill a', 'force-kill a'])
    expect(mgr.list()).toEqual([])
  })
})

describe('parseTasklistPids', () => {
  it('reads llama-server PIDs from one tasklist listing', () => {
    const csv = '"llama-server.exe","1234","Console","1","2,048 K"\r\n"llama-server.exe","88","Console","1","10 K"\r\n'
    expect([...parseTasklistPids(csv)]).toEqual([1234, 88])
    expect(parseTasklistPids('INFO: No tasks are running which match the specified criteria.').size).toBe(0)
  })
})
