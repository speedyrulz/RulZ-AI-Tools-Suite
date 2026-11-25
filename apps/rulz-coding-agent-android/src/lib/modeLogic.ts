// The phone's side of modes and rentals, as plain functions and one store factory, so the rules can be
// tested without a phone or a computer. Nothing here imports the app's own stores or Capacitor.
import { create, type StoreApi, type UseBoundStore } from 'zustand'
import { chooseModel } from '@shared/modes'
import type { AgentMode, AppSettings, ConnectionInfo, ModelInfo, ModelModes } from '@shared/types'

/** The same answer is not asked for twice within this long, however many parts of the screen want it. */
export const MIN_GAP_MS = 2000

/** The computer answers a channel it does not have with this (a RulZ Coding Agent older than the phone app). */
export const isUnknownChannel = (e: unknown): boolean => /Unknown request/i.test(e instanceof Error ? e.message : String(e))

/** "Settings for" opens on Coding when the model has load settings of its own for it, so an edit always changes what is used. */
export function openingSettingsFor(modes: ModelModes | null | undefined): AgentMode | 'all' {
  return modes?.modes.coding.ownLoad ? 'coding' : 'all'
}

/** The sentence for a chosen model that ran on a rental and is not in the list now, or null. Nothing is started and no other model is used. */
export function chosenRentalGone(settings: AppSettings | null, connection: ConnectionInfo): string | null {
  if (connection.status !== 'connected' || !settings?.model) return null
  const c = chooseModel(settings.model, settings.modelRental === true, connection.models)
  if (c.ok) return null
  const listed = connection.models.find((m) => m.id === settings.model)
  return listed ? (listed.rental ? c.reason : null) : settings.modelRental === true ? c.reason : null
}

/**
 * What a batch run needs for the model in use. A rental's model needs the owner's yes, asked for every
 * run. `rentalOk` is what `batch:start` gets. A computer older than this app does not mark rentals, so
 * `rental` is then undefined and nothing is asked.
 */
export function batchRentalGate(rental: ModelInfo['rental'], yes: boolean): { needsYes: boolean; canStart: boolean; rentalOk: boolean } {
  const needsYes = !!rental
  return { needsYes, canStart: !needsYes || yes, rentalOk: needsYes && yes }
}

// ---------- The modes of the model in use ----------

export interface ModesDeps {
  /** null = a RulZ LLM Suite older than 0.35. Rejects with "Unknown request" on a computer older than this app. */
  modes: (model: string) => Promise<ModelModes | null>
  reloadMode: (model: string, mode: AgentMode) => Promise<ModelModes>
}

export interface ModesState {
  /** The computer these answers are for; everything is dropped when it changes. */
  computer: string | null
  /** The computer does not know the modes channel: the new controls stay hidden. */
  older: boolean
  /** Per model: undefined = not asked yet, null = an older RulZ LLM Suite (no modes), else its answer. */
  byModel: Record<string, ModelModes | null | undefined>
  /** The instance the owner chose "Keep as is" for, per model. In memory only. */
  kept: Record<string, string>
  /** The model the screen shows now; answers for any other are dropped. */
  wanted: string
  forComputer: (id: string | null) => void
  want: (model: string) => void
  refresh: (model: string, force?: boolean) => Promise<void>
  refreshWanted: (force?: boolean) => Promise<void>
  /** The link is back: asks again, also a computer that did not know modes before. */
  reconnected: () => Promise<void>
  keep: (model: string) => void
  /** Reloads the model with a mode's load settings (never automatic); throws with the server's reason when it can't. */
  reload: (model: string, mode: AgentMode) => Promise<void>
}

export type ModesStore = UseBoundStore<StoreApi<ModesState>>

export function createModesStore(deps: ModesDeps, now: () => number = Date.now): ModesStore {
  /** Bumped when the computer changes: an answer that was asked before it is for the other computer. */
  let epoch = 0
  /** Per model, bumped by a reload: an older question's answer must not replace the reload's. */
  const writes = new Map<string, number>()
  const asked = new Map<string, number>()
  let running: { model: string; epoch: number; p: Promise<void> } | null = null
  let queued: { model: string; force: boolean } | null = null

  return create<ModesState>((set, get) => {
    const bump = (model: string): number => {
      const n = (writes.get(model) ?? 0) + 1
      writes.set(model, n)
      return n
    }
    const store = (model: string, modes: ModelModes | null): void => set((s) => ({ byModel: { ...s.byModel, [model]: modes } }))

    return {
      computer: null,
      older: false,
      byModel: {},
      kept: {},
      wanted: '',

      forComputer: (id) => {
        if (get().computer === id) return
        epoch++
        asked.clear()
        writes.clear()
        queued = null
        set({ computer: id, older: false, byModel: {}, kept: {} })
      },

      want: (model) => {
        if (get().wanted !== model) set({ wanted: model })
      },

      refresh: (model, force = false) => {
        if (!model || get().older) return Promise.resolve()
        if (running) {
          // Never two questions at once: a question for another model (or another computer) waits for this one.
          if (running.model !== model || running.epoch !== epoch) queued = { model, force }
          return running.p
        }
        if (!force && now() - (asked.get(model) ?? 0) < MIN_GAP_MS) return Promise.resolve()
        asked.set(model, now())
        const mine = epoch
        const version = writes.get(model) ?? 0
        const current = (): boolean => mine === epoch && version === (writes.get(model) ?? 0) && (!get().wanted || get().wanted === model)
        const p = deps
          .modes(model)
          .then((modes) => {
            if (current()) store(model, modes)
          })
          .catch((e: unknown) => {
            // A computer that does not know the channel never will: hide the controls. Any other failure keeps what was known.
            if (mine === epoch && isUnknownChannel(e)) set((s) => ({ older: true, byModel: { ...s.byModel, [model]: null } }))
          })
          .finally(() => {
            running = null
            const next = queued
            queued = null
            // One for a computer left meanwhile is gone already (forComputer).
            if (next) void get().refresh(next.model, next.force)
          })
        running = { model, epoch: mine, p }
        return p
      },

      refreshWanted: (force = false) => get().refresh(get().wanted, force),

      reconnected: () => {
        // The computer's app may have been updated since it refused the channel.
        if (get().older) set({ older: false })
        return get().refresh(get().wanted, true)
      },

      keep: (model) => {
        const instance = get().byModel[model]?.instance
        if (instance) set((s) => ({ kept: { ...s.kept, [model]: instance } }))
      },

      reload: async (model, mode) => {
        const mine = epoch
        bump(model)
        const modes = await deps.reloadMode(model, mode)
        if (mine !== epoch) return
        bump(model)
        store(model, modes)
      }
    }
  })
}
