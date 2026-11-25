// OpenRouter models: the state main keeps (cloud-models.json, the spending ledger) as the window sees it. Every action
// goes through run(): it sends one channel, takes the new state from the reply, and says an error in a toast. The same
// call twice at once is sent once, so a double click cannot add or remove twice.
import { create } from 'zustand'
import { toast } from 'sonner'
import type { CloudCatalog, CloudIpcInvoke, CloudState } from '@shared/cloud'
import type { IpcInvoke } from '@shared/ipc'
import { errorMessage, invoke, on } from '../lib/api'
import { once } from '../lib/once'
import { useModels } from './models'

/** The channels whose reply is the new CloudState. */
export type CloudStateChannel = {
  [K in keyof CloudIpcInvoke]: ReturnType<CloudIpcInvoke[K]> extends CloudState ? K : never
}[keyof CloudIpcInvoke]

interface CloudStore {
  state: CloudState | null
  loaded: boolean
  inflight: Record<string, true>
  /** OpenRouter's list of usable models; null until read. It needs no key. */
  catalog: CloudCatalog | null
  catalogLoading: boolean
  catalogError: string | null
  init: () => Promise<void>
  run: <K extends CloudStateChannel>(channel: K, ...args: Parameters<IpcInvoke[K]>) => Promise<CloudState | null>
  loadCatalog: (refresh: boolean) => Promise<void>
}

export const callKey = (channel: string, args: unknown[]): string => `${channel}:${JSON.stringify(args)}`

/** An event can overtake the reply to an action; the older of the two must not win. */
export const newer = (have: CloudState | null, got: CloudState): { state: CloudState } => ({ state: have && have.rev > got.rev ? have : got })

export const useCloud = create<CloudStore>((set, get) => ({
  state: null,
  loaded: false,
  inflight: {},
  catalog: null,
  catalogLoading: false,
  catalogError: null,

  init: once(async () => {
    // Nothing here may stop the window from starting: a failure leaves the page empty with a message.
    try {
      on('cloud:changed', (state) => set((s) => newer(s.state, state)))
      const state = await invoke('cloud:state')
      set((s) => ({ ...newer(s.state, state), loaded: true }))
    } catch (e) {
      set({ loaded: true })
      toast.error(`Could not read the OpenRouter state: ${errorMessage(e)}`)
    }
  }),

  run: async (channel, ...args) => {
    const key = callKey(channel, args)
    if (get().inflight[key]) return null
    set((s) => ({ inflight: { ...s.inflight, [key]: true } }))
    try {
      const state = (await (invoke as (c: string, ...a: unknown[]) => Promise<CloudState>)(channel, ...args)) as CloudState
      set((s) => newer(s.state, state))
      return state
    } catch (e) {
      toast.error(errorMessage(e), { duration: 15000 })
      return null
    } finally {
      set((s) => {
        const next = { ...s.inflight }
        delete next[key]
        return { inflight: next }
      })
    }
  },

  loadCatalog: async (refresh) => {
    if (get().catalogLoading) return
    set({ catalogLoading: true, catalogError: null })
    try {
      set({ catalog: await invoke('cloud:catalog', refresh) })
    } catch (e) {
      set({ catalogError: errorMessage(e) })
    } finally {
      set({ catalogLoading: false })
    }
  }
}))

/** Is this exact call in flight (for disabling its button)? */
export const isBusy = (s: Pick<CloudStore, 'inflight'>, channel: CloudStateChannel, ...args: unknown[]): boolean => !!s.inflight[callKey(channel, args)]

/** Any call of a channel in flight, whatever its arguments. */
export const isChannelBusy = (s: Pick<CloudStore, 'inflight'>, channel: CloudStateChannel): boolean =>
  Object.keys(s.inflight).some((k) => k.startsWith(`${channel}:`))

/** Connects a cloud model through the models store (it becomes the active model). Null after a toast on failure. */
export const connectCloudModel = (modelId: string): ReturnType<ReturnType<typeof useModels.getState>['ensure']> => useModels.getState().ensure(modelId)
