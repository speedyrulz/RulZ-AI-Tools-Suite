// Puts the parts of src/main/remote together for index.ts. Nothing here starts or ends billing by itself; the rental
// manager does, when the window asks or a limit is reached (docs/vast-design.md, section 2).
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { isRemoteModelId, type RemoteGpu, type RemoteHost, type RemoteState } from '@shared/remote'
import type { AdapterEntry, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry, SidecarFile } from '@shared/types'
import type { RemoteLauncher } from '../engine/serverHandle'
import { estimateMemory } from '../models/estimate'
import { parseLoadConfig } from '../settings/schema'
import { writeJsonAtomic } from '../util/jsonFile'
import type { RentalDirectory } from './contract'
import { remoteHandlers, type RemoteHandlers } from './ipc'
import { RemoteModelStore, toModelEntry, type RemoteModelInput } from './models'
import { SshRentalOps } from './ops'
import { createRemoteLauncher } from './remoteServer'
import { RentalManager } from './rentals'
import { SecretStore } from './secrets'
import { SshRunner, sshKeyInfo } from './ssh'
import { VastApi } from './vastApi'

const MiB = 1024 * 1024
/** A rental's RAM is not read; enough that the estimate only judges its GPUs. */
const ASSUMED_RAM = 64 * 1024 * MiB

export interface RemoteServiceDeps {
  userData: string
  homedir: string
  env: Record<string, string | undefined>
  /** Electron's safeStorage. */
  safeStorage: { isEncryptionAvailable(): boolean; encryptString(text: string): Buffer; decryptString(data: Buffer): string }
  getLocalModel: (id: string) => ModelEntry | undefined
  getAdapter: (id: string) => AdapterEntry | undefined
  getDefaults: (modelId: string) => LoadConfig
  draftFor: (model: ModelEntry, cfg: LoadConfig) => SidecarFile | undefined
  adapterBytes: (cfg: LoadConfig) => number
  /** Build number of this PC's llama-server; null = unknown. */
  localEngineBuild: () => Promise<number | null>
  /** InstanceManager's, read when called (it is made after this service). */
  loadOn: (host: RemoteHost, modelId: string, config: LoadConfig | null) => Promise<LoadedInstance>
  isLoaded: (modelId: string) => boolean
  forgetDefaults: (modelId: string) => unknown
  /** Ends every use of a rental cleanly before it stops. */
  onStopping: (rentalId: number) => Promise<void>
  onChange: (state: RemoteState) => void
  /** The remote models changed: the model list has to be sent again. */
  onModels: () => void
}

type Channel = keyof RemoteHandlers | 'remote:load' | 'remote:estimate' | 'remote:saveModel' | 'remote:deleteModel'

export interface RemoteService {
  rentals: RentalManager
  launcher: RemoteLauncher
  /** The remote models as list entries. */
  models: () => ModelEntry[]
  /** The host of a load on a rental; throws a sentence when it is not one of ours or not running. */
  host: (rentalId: number) => RemoteHost
  /** 'remote:*' channels for the window. */
  handlers: Record<Channel, (...args: never[]) => unknown>
  load: () => Promise<void>
  /** An error's text without any key or token. */
  scrub: (err: unknown) => string
}

export function createRemoteService(deps: RemoteServiceDeps): RemoteService {
  const secrets = new SecretStore({
    available: () => deps.safeStorage.isEncryptionAvailable(),
    encrypt: (t) => deps.safeStorage.encryptString(t),
    decrypt: (d) => deps.safeStorage.decryptString(d),
    readFile: (p) => fs.readFile(p, 'utf8').catch(() => null),
    writeFile: (p, data) => writeJsonAtomic(p, data),
    env: deps.env,
    homedir: deps.homedir,
    join,
    filePath: join(deps.userData, 'remote-secrets.json')
  })
  const store = new RemoteModelStore(join(deps.userData, 'remote-models.json'))
  const entry = (id: string): ModelEntry | undefined => {
    const def = store.get(id)
    return def ? toModelEntry(def, deps.getLocalModel) : undefined
  }

  // The manager and the ssh side know each other: the manager answers the directory, the ssh side does its work.
  let rentals: RentalManager | null = null
  const directory: RentalDirectory = {
    info: (id) => rentals?.info(id) ?? null,
    use: (id, what, by) => rentals?.use(id, what, by) ?? (() => undefined),
    note: (id, line) => rentals?.note(id, line)
  }
  const ssh = new SshRunner({ directory, knownHostsDir: join(deps.userData, 'ssh') })
  const ops = new SshRentalOps({
    directory,
    secrets,
    ssh,
    models: store,
    getModel: deps.getLocalModel,
    getAdapter: deps.getAdapter,
    getDefaults: deps.getDefaults
  })
  const manager = new RentalManager({
    api: new VastApi({ key: async () => (await secrets.vastKey())?.value ?? null }),
    ops,
    secrets,
    statePath: join(deps.userData, 'remote.json'),
    models: store,
    localEngineBuild: deps.localEngineBuild,
    sshKey: async () => {
      const k = sshKeyInfo(deps.homedir)
      return { available: k.available, publicKeyPath: k.publicKeyPath, publicKey: k.publicKey || null, ...(k.error ? { error: k.error } : {}) }
    },
    onChange: deps.onChange,
    onStopping: deps.onStopping
  })
  rentals = manager

  const estimate = (model: ModelEntry, cfg: LoadConfig, gpus: RemoteGpu[]): MemoryEstimate | null => {
    if (!gpus.length || !model.sizeBytes) return null
    const all = gpus.map((g) => ({ index: g.index, name: g.name, totalBytes: g.totalMiB * MiB, freeBytes: Math.max(0, g.totalMiB - g.usedMiB) * MiB }))
    return estimateMemory(
      { ...model, draft: deps.draftFor(model, cfg) },
      // A rental always uses its own GPUs; this PC's choice of GPU says nothing there.
      { ...cfg, splitMode: cfg.splitMode === 'single' ? 'auto' : cfg.splitMode },
      { vramBytes: all.reduce((n, g) => n + g.freeBytes, 0), ramBytes: ASSUMED_RAM, gpus: all },
      deps.adapterBytes(cfg)
    )
  }

  const launcher = createRemoteLauncher({
    ssh,
    directory,
    getDef: (id) => store.get(id),
    getModel: deps.getLocalModel,
    getAdapter: deps.getAdapter,
    draftFor: deps.draftFor,
    estimate
  })

  const host = (rentalId: number): RemoteHost => {
    const info = manager.info(rentalId)
    if (!info) throw new Error('That instance is not in your list on the vast.ai page.')
    if (!info.running) throw new Error(`${info.label || `Instance ${rentalId}`} is not running. Start it on the vast.ai page first.`)
    return { kind: 'vast', rentalId, label: info.label || `Instance ${rentalId}` }
  }
  const remoteModel = (modelId: unknown): ModelEntry => {
    const m = typeof modelId === 'string' && isRemoteModelId(modelId) ? entry(modelId) : undefined
    if (!m) throw new Error('That remote model is not saved any more.')
    return m
  }
  const changed = (): RemoteState => {
    deps.onModels()
    const state = manager.state()
    deps.onChange(state)
    return state
  }

  return {
    rentals: manager,
    launcher,
    models: () => store.list().map((d) => toModelEntry(d, deps.getLocalModel)),
    host,
    scrub: (err) => secrets.scrub(err instanceof Error ? err.message : String(err)),
    handlers: {
      ...remoteHandlers(manager),
      'remote:load': async (rentalId: number, modelId: string, config: LoadConfig | null) => {
        const model = remoteModel(modelId)
        const where = host(rentalId)
        // Loading takes minutes and no request runs meanwhile: without this the idle limit could fall into it.
        const release = manager.use(rentalId, 'loading a model', 'window')
        try {
          return await deps.loadOn(where, model.id, config ? parseLoadConfig(config) : null)
        } finally {
          release()
        }
      },
      'remote:estimate': (rentalId: number, modelId: string, config: LoadConfig) => {
        const model = remoteModel(modelId)
        const r = manager.state().rentals.find((x) => x.id === rentalId)
        return r ? estimate(model, parseLoadConfig(config), r.gpus) : null
      },
      'remote:saveModel': async (def: RemoteModelInput) => {
        await store.save(def)
        return changed()
      },
      'remote:deleteModel': async (modelId: string) => {
        const model = remoteModel(modelId)
        if (deps.isLoaded(model.id)) throw new Error('Eject the model before deleting it.')
        await store.delete(model.id)
        await deps.forgetDefaults(model.id)
        return changed()
      }
    } as RemoteService['handlers'],
    load: async () => {
      await store.load()
      await manager.load()
      manager.startTicking()
    }
  }
}
