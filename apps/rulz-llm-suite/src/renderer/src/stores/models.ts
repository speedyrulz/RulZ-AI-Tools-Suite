import { create } from 'zustand'
import { toast } from 'sonner'
import type { HardwareSnapshot, LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'
import { pickActive } from '../lib/activeModel'
import { once } from '../lib/once'

interface ModelsState {
  models: ModelEntry[]
  scanned: boolean
  instances: LoadedInstance[]
  hardware: HardwareSnapshot | null
  /** Model that new chat messages go to. */
  activeModelId: string | null
  init: () => Promise<void>
  rescan: () => Promise<void>
  load: (modelId: string, config: LoadConfig | null) => Promise<LoadedInstance | null>
  /** The model's loaded or loading instance whatever its settings, or a new one with its defaults. */
  ensure: (modelId: string) => Promise<LoadedInstance | null>
  unload: (instanceId: string) => Promise<void>
  unloadAll: () => Promise<void>
  setActive: (modelId: string) => void
}

/** The model the newest load from this window is starting (see pickActive). */
let loadingModelId: string | null = null
/** That load's token: an older load of the same model ending (say, cancelled by it) leaves it alone. */
let loadingToken: object | null = null

export const useModels = create<ModelsState>((set, get) => {
  /** Runs a load of `modelId` and makes it the active model; null (after a toast) if it failed. */
  const loadWith = async (modelId: string, run: () => Promise<LoadedInstance>): Promise<LoadedInstance | null> => {
    set({ activeModelId: modelId })
    const token = {}
    loadingModelId = modelId
    loadingToken = token
    try {
      return await run()
    } catch (e) {
      const msg = errorMessage(e)
      if (msg !== 'Load cancelled') {
        const [first, ...rest] = msg.split('\n')
        toast.error(first, { description: rest.join('\n') || undefined, duration: 20000 })
      }
      return null
    } finally {
      if (loadingToken === token) {
        loadingModelId = null
        loadingToken = null
        // A failed load leaves no instance; pick the active model again from what is loaded.
        set((s) => ({ activeModelId: pickActive(s.instances, s.activeModelId, loadingModelId) }))
      }
    }
  }

  return {
    models: [],
    scanned: false,
    instances: [],
    hardware: null,
    activeModelId: null,

    init: once(async () => {
      on('models:changed', (models) => set({ models, scanned: true }))
      on('instances:changed', (instances) => set({ instances, activeModelId: pickActive(instances, get().activeModelId, loadingModelId) }))
      on('instances:error', (e) => {
        toast.error(`${e.modelName}: ${e.error}`, { description: e.logTail.slice(-4).join('\n'), duration: 15000 })
      })
      on('hardware:update', (hardware) => set({ hardware }))
      const [models, instances, hardware] = await Promise.all([
        invoke('models:list'),
        invoke('instances:list'),
        invoke('hardware:snapshot')
      ])
      set((s) => ({ models: s.models.length ? s.models : models, scanned: s.scanned || models.length > 0, instances, hardware }))
      // An empty (or missing) folder scans so fast that its result can arrive before the listener
      // above, leaving "Scanning models…" up for good. Ask for the scan's result directly.
      if (!get().scanned) void get().rescan()
    }),

    rescan: async () => {
      try {
        set({ models: await invoke('models:rescan') })
      } catch (e) {
        toast.error(errorMessage(e))
      } finally {
        set({ scanned: true })
      }
    },

    load: (modelId, config) => loadWith(modelId, () => invoke('instances:load', modelId, config)),

    ensure: (modelId) => loadWith(modelId, () => invoke('instances:ensure', modelId)),

    unload: async (instanceId) => {
      await invoke('instances:unload', instanceId).catch((e) => toast.error(errorMessage(e)))
    },

    unloadAll: async () => {
      await invoke('instances:unloadAll').catch((e) => toast.error(errorMessage(e)))
    },

    setActive: (activeModelId) => set({ activeModelId })
  }
})

export function readyInstance(modelId: string | null): LoadedInstance | undefined {
  if (!modelId) return undefined
  return useModels.getState().instances.find((i) => i.modelId === modelId && i.status === 'ready')
}

export function modelById(id: string | null | undefined): ModelEntry | undefined {
  return id ? useModels.getState().models.find((m) => m.id === id) : undefined
}
