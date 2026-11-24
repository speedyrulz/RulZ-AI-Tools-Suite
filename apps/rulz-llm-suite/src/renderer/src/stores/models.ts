import { create } from 'zustand'
import { toast } from 'sonner'
import type { AllModeDefaults, AppMode, ModelModeDefaults } from '@shared/modes'
import type { AdapterEntry, HardwareSnapshot, LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'
import { pickActive } from '../lib/activeModel'
import { modesFor } from '../lib/modes'
import { once } from '../lib/once'
import { WBENCH_BUSY, wbenchRunning } from '../lib/wbenchFlag'

interface ModelsState {
  models: ModelEntry[]
  scanned: boolean
  /** LoRA adapter files in the models folder. */
  adapters: AdapterEntry[]
  instances: LoadedInstance[]
  /** Every model's settings per mode. */
  modeDefaults: AllModeDefaults
  hardware: HardwareSnapshot | null
  /** Model that new chat messages go to. */
  activeModelId: string | null
  init: () => Promise<void>
  rescan: () => Promise<void>
  load: (modelId: string, config: LoadConfig | null) => Promise<LoadedInstance | null>
  /** The model's loaded or loading instance whatever its settings, or a new one with its defaults. */
  ensure: (modelId: string, mode?: AppMode) => Promise<LoadedInstance | null>
  /** Reads the mode settings again (after a save); an event that arrives meanwhile is newer and wins. */
  refreshModeDefaults: () => Promise<void>
  unload: (instanceId: string) => Promise<void>
  unloadAll: () => Promise<void>
  setActive: (modelId: string) => void
}

/** Counts mode-settings events, so an older answer never replaces a newer event. */
let modeSeq = 0

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
    adapters: [],
    instances: [],
    modeDefaults: {},
    hardware: null,
    activeModelId: null,

    init: once(async () => {
      on('models:changed', (models) => set({ models, scanned: true }))
      on('adapters:changed', (adapters) => set({ adapters }))
      on('models:modeDefaults', (modeDefaults) => {
        modeSeq++
        set({ modeDefaults })
      })
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
      // An adapters:changed event that arrived first is newer than this list.
      void invoke('adapters:list').then(
        (adapters) => set((s) => (s.adapters.length ? s : { adapters })),
        () => undefined
      )
      // A models:modeDefaults event that arrived first is newer than this answer; a failure leaves every mode on its fallback.
      void get().refreshModeDefaults()
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

    ensure: (modelId, mode) => loadWith(modelId, () => invoke('instances:ensure', modelId, mode)),

    refreshModeDefaults: async () => {
      const seq = modeSeq
      try {
        const all = await invoke('models:modeDefaults')
        if (seq === modeSeq) set({ modeDefaults: all })
      } catch {
        // The settings stay as they were.
      }
    },

    unload: async (instanceId) => {
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
      await invoke('instances:unload', instanceId).catch((e) => toast.error(errorMessage(e)))
    },

    unloadAll: async () => {
      if (wbenchRunning()) return void toast(WBENCH_BUSY)
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

/** A model's settings in each mode (own-property lookup); for code outside React. */
export function modeDefaultsOf(modelId: string | null | undefined): ModelModeDefaults | undefined {
  return modesFor(useModels.getState().modeDefaults, modelId)
}
