// What the window and the phone may do to a model's settings in RulZ LLM Suite. The rules live here, in
// the main process: the phone calls the same handlers as the window, so nothing may rest on a button
// being greyed out.
import type { AgentMode, LoadSettings, ModelInfo, ModelLoadInfo, ModelModes } from '@shared/types'
import type { RulzClient } from './client'

export interface ModelAdminDeps {
  client: Pick<RulzClient, 'loadSettings' | 'previewLoadSettings' | 'setLoadSettings' | 'modelModes' | 'reloadMode'>
  /** The last known model list (to tell a rental's model). */
  models: () => ModelInfo[]
  /** Throws when a batch run or a task is going; `what` finishes "Wait for it to finish or press Stop, then …". */
  assertIdle: (what: string) => void
  refreshModels: () => void
}

export const RENTAL_SETTINGS_REFUSAL =
  'This model runs on a rented computer, so its load settings and its reload are not changed from here. Change them on LLM Suite\'s vast.ai page.'

export function modelAdmin(deps: ModelAdminDeps) {
  const rented = (model: string): boolean => !!deps.models().find((m) => m.id === model)?.rental
  const refuseRental = (model: string): void => {
    if (rented(model)) throw new Error(RENTAL_SETTINGS_REFUSAL)
  }
  /** The phone sends what it likes: only a model id and one of the two modes go into an address. */
  const check = (model: unknown, mode?: unknown): void => {
    if (typeof model !== 'string' || !model || model === '.' || model === '..') throw new Error('That is not a model.')
    if (mode !== undefined && mode !== null && mode !== 'coding' && mode !== 're') throw new Error('Unknown mode: only coding and re are available.')
  }
  return {
    async loadSettings(model: string, mode?: AgentMode): Promise<ModelLoadInfo> {
      check(model, mode)
      return deps.client.loadSettings(model, mode)
    },
    previewLoadSettings: async (model: string, config: Partial<LoadSettings>, mode?: AgentMode): Promise<ModelLoadInfo> => {
      check(model, mode)
      refuseRental(model)
      return deps.client.previewLoadSettings(model, config, mode)
    },
    async setLoadSettings(model: string, config: Partial<LoadSettings> | null, mode?: AgentMode): Promise<ModelLoadInfo> {
      check(model, mode)
      // Restarting the model would cut off the answer it is writing.
      deps.assertIdle('change the load settings')
      refuseRental(model)
      const info = await deps.client.setLoadSettings(model, config, mode)
      deps.refreshModels()
      return info
    },
    /** null: a RulZ LLM Suite older than 0.35. */
    async modes(model: string): Promise<ModelModes | null> {
      check(model)
      return deps.client.modelModes(model)
    },
    async reloadMode(model: string, mode: AgentMode): Promise<ModelModes> {
      check(model, mode ?? '')
      deps.assertIdle('reload the model')
      refuseRental(model)
      const modes = await deps.client.reloadMode(model, mode)
      deps.refreshModels()
      return modes
    }
  }
}
