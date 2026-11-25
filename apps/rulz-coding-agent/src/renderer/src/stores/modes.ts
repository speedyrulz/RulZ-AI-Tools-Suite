import { useEffect } from 'react'
import { create } from 'zustand'
import { modesStamp } from '@shared/modes'
import type { AgentMode, ModelModes } from '@shared/types'
import { invoke } from '../lib/api'
import { activeModelId, useApp } from './app'
import { useWorkspace } from './workspace'

/** The same answer is not asked for twice within this long, however many parts of the window want it. */
const MIN_GAP_MS = 2000

interface ModesState {
  /** Per model: undefined = not asked yet, null = an older RulZ LLM Suite (no modes), else its answer. */
  byModel: Record<string, ModelModes | null | undefined>
  /** The instance the owner chose "Keep as is" for, per model. */
  kept: Record<string, string>
  refresh: (model: string, force?: boolean) => Promise<void>
  keep: (model: string) => void
  /** Reloads the model with a mode's load settings (never automatic); throws with the reason when it can't. */
  reload: (model: string, mode: AgentMode) => Promise<void>
}

const asked = new Map<string, number>()
const inFlight = new Map<string, Promise<void>>()

export const useModes = create<ModesState>((set, get) => ({
  byModel: {},
  kept: {},

  refresh: (model, force = false) => {
    if (!model) return Promise.resolve()
    const running = inFlight.get(model)
    if (running) return running
    if (!force && Date.now() - (asked.get(model) ?? 0) < MIN_GAP_MS) return Promise.resolve()
    asked.set(model, Date.now())
    const p = invoke('models:modes', model)
      .then((modes) => set((s) => ({ byModel: { ...s.byModel, [model]: modes } })))
      // A failure keeps what was known: the reload bar simply stays as it was.
      .catch(() => undefined)
      .finally(() => inFlight.delete(model))
    inFlight.set(model, p)
    return p
  },

  keep: (model) => {
    const instance = get().byModel[model]?.instance
    if (instance) set((s) => ({ kept: { ...s.kept, [model]: instance } }))
  },

  reload: async (model, mode) => {
    const modes = await invoke('models:reloadMode', model, mode)
    set((s) => ({ byModel: { ...s.byModel, [model]: modes } }))
  }
}))

/**
 * The modes of the model in use (or of `model`). Asked again when the model changes, when the window
 * gets focus, when the model's loaded state or context changes in the list, and when a task ends:
 * nothing polls on a timer.
 */
export function useModelModes(model?: string): { model: string; modes: ModelModes | null | undefined; refresh: () => void } {
  const settings = useApp((s) => s.settings)
  const connection = useApp((s) => s.connection)
  const running = useWorkspace((s) => s.running)
  const id = model ?? activeModelId(settings, connection)
  const stamp = modesStamp(connection.models.find((m) => m.id === id))
  const refresh = useModes((s) => s.refresh)
  const modes = useModes((s) => s.byModel[id])
  const connected = connection.status === 'connected'

  useEffect(() => {
    if (connected && id) void refresh(id, true)
  }, [id, connected, refresh])
  useEffect(() => {
    if (connected && id && stamp) void refresh(id)
  }, [stamp, id, connected, refresh])
  useEffect(() => {
    // A task that ends may have left the model reloaded by another app, or in use no more.
    if (!running && connected && id) void refresh(id)
  }, [running, id, connected, refresh])
  useEffect(() => {
    const onFocus = (): void => {
      if (connected && id) void refresh(id)
    }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [id, connected, refresh])

  return { model: id, modes, refresh: () => void refresh(id, true) }
}
