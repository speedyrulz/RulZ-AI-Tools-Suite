// vast.ai rentals: the state main keeps (remote.json) as the window sees it. Every action goes through run(): it
// sends one channel, takes the new state from the reply, and says an error in a toast. The same call twice at once
// is sent once, so a double click cannot rent or destroy twice.
import { create } from 'zustand'
import { toast } from 'sonner'
import type { IpcInvoke } from '@shared/ipc'
import type { RemoteIpcInvoke, RemoteState } from '@shared/remote'
import type { LoadConfig, LoadedInstance } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'
import { once } from '../lib/once'
import { WBENCH_BUSY, wbenchRunning } from '../lib/wbenchFlag'
import { useModels } from './models'

/** The channels whose reply is the new RemoteState. */
export type RemoteStateChannel = {
  [K in keyof RemoteIpcInvoke]: ReturnType<RemoteIpcInvoke[K]> extends RemoteState ? K : never
}[keyof RemoteIpcInvoke]

interface RemoteStore {
  state: RemoteState | null
  /** The first read of the state finished (it may have failed). */
  loaded: boolean
  /** Keys of the calls in flight (see callKey). */
  inflight: Record<string, true>
  init: () => Promise<void>
  /** Runs a channel; the new state replaces the old. Null (after a toast) on failure or when the same call is already running. */
  run: <K extends RemoteStateChannel>(channel: K, ...args: Parameters<IpcInvoke[K]>) => Promise<RemoteState | null>
}

export const callKey = (channel: string, args: unknown[]): string => `${channel}:${JSON.stringify(args)}`

/** An event can overtake the reply to an action; the older of the two must not win. */
export const newer = (have: RemoteState | null, got: RemoteState): { state: RemoteState } => ({ state: have && have.rev > got.rev ? have : got })

export const useRemote = create<RemoteStore>((set, get) => ({
  state: null,
  loaded: false,
  inflight: {},

  init: once(async () => {
    // Nothing here may stop the window from starting: a failure leaves the page empty with a message.
    try {
      on('remote:changed', (state) => set((s) => newer(s.state, state)))
      const state = await invoke('remote:state')
      set((s) => ({ ...newer(s.state, state), loaded: true }))
    } catch (e) {
      set({ loaded: true })
      toast.error(`Could not read the vast.ai state: ${errorMessage(e)}`)
    }
  }),

  run: async (channel, ...args) => {
    const key = callKey(channel, args)
    if (get().inflight[key]) return null
    set((s) => ({ inflight: { ...s.inflight, [key]: true } }))
    try {
      const state = (await (invoke as (c: string, ...a: unknown[]) => Promise<RemoteState>)(channel, ...args)) as RemoteState
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
  }
}))

/** Is this exact call in flight (for disabling its button)? */
export const isBusy = (s: Pick<RemoteStore, 'inflight'>, channel: RemoteStateChannel, ...args: unknown[]): boolean => !!s.inflight[callKey(channel, args)]

/** Any call of a channel in flight, whatever its arguments. */
export const isChannelBusy = (s: Pick<RemoteStore, 'inflight'>, channel: RemoteStateChannel): boolean =>
  Object.keys(s.inflight).some((k) => k.startsWith(`${channel}:`))

export function rentalById(state: RemoteState | null, id: number): RemoteState['rentals'][number] | undefined {
  return state?.rentals.find((r) => r.id === id)
}

/**
 * Loads a set-up remote model on a running rental (config null = the model's saved settings). It never rents or
 * starts anything: a refusal comes back as a sentence in a toast. Null on failure.
 */
export async function loadRemoteModel(rentalId: number, modelId: string, config: LoadConfig | null = null): Promise<LoadedInstance | null> {
  if (wbenchRunning()) {
    toast(WBENCH_BUSY)
    return null
  }
  useModels.getState().setActive(modelId)
  try {
    return await invoke('remote:load', rentalId, modelId, config)
  } catch (e) {
    const [first, ...rest] = errorMessage(e).split('\n')
    toast.error(first, { description: rest.join('\n') || undefined, duration: 20000 })
    return null
  }
}
