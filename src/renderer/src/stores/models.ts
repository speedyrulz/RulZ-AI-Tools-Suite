import { create } from 'zustand'
import { toast } from 'sonner'
import type { HardwareSnapshot, LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'

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
  unload: (instanceId: string) => Promise<void>
  unloadAll: () => Promise<void>
  setActive: (modelId: string) => void
}

export const useModels = create<ModelsState>((set, get) => ({
  models: [],
  scanned: false,
  instances: [],
  hardware: null,
  activeModelId: null,

  init: async () => {
    on('models:changed', (models) => set({ models, scanned: true }))
    on('instances:changed', (instances) => {
      const active = get().activeModelId
      const stillThere = instances.some((i) => i.modelId === active && i.status !== 'stopping')
      const newest = [...instances].filter((i) => i.status !== 'stopping').sort((a, b) => b.startedAt - a.startedAt)[0]
      set({ instances, activeModelId: stillThere ? active : (newest?.modelId ?? null) })
    })
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
  },

  rescan: async () => {
    try {
      set({ models: await invoke('models:rescan'), scanned: true })
    } catch (e) {
      toast.error(errorMessage(e))
    }
  },

  load: async (modelId, config) => {
    set({ activeModelId: modelId })
    try {
      return await invoke('instances:load', modelId, config)
    } catch (e) {
      const msg = errorMessage(e)
      if (msg !== 'Load cancelled') {
        const [first, ...rest] = msg.split('\n')
        toast.error(first, { description: rest.join('\n') || undefined, duration: 20000 })
      }
      return null
    }
  },

  unload: async (instanceId) => {
    await invoke('instances:unload', instanceId).catch((e) => toast.error(errorMessage(e)))
  },

  unloadAll: async () => {
    await invoke('instances:unloadAll').catch((e) => toast.error(errorMessage(e)))
  },

  setActive: (activeModelId) => set({ activeModelId })
}))

export function readyInstance(modelId: string | null): LoadedInstance | undefined {
  if (!modelId) return undefined
  return useModels.getState().instances.find((i) => i.modelId === modelId && i.status === 'ready')
}

export function modelById(id: string | null | undefined): ModelEntry | undefined {
  return id ? useModels.getState().models.find((m) => m.id === id) : undefined
}
