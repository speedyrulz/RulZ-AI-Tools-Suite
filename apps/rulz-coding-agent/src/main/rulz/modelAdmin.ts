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
  return {
    loadSettings: (model: string, mode?: AgentMode): Promise<ModelLoadInfo> => deps.client.loadSettings(model, mode),
    previewLoadSettings: async (model: string, config: Partial<LoadSettings>, mode?: AgentMode): Promise<ModelLoadInfo> => {
      refuseRental(model)
      return deps.client.previewLoadSettings(model, config, mode)
    },
    async setLoadSettings(model: string, config: Partial<LoadSettings> | null, mode?: AgentMode): Promise<ModelLoadInfo> {
      // Restarting the model would cut off the answer it is writing.
      deps.assertIdle('change the load settings')
      refuseRental(model)
      const info = await deps.client.setLoadSettings(model, config, mode)
      deps.refreshModels()
      return info
    },
    /** null: a RulZ LLM Suite older than 0.35. */
    modes: (model: string): Promise<ModelModes | null> => deps.client.modelModes(model),
    async reloadMode(model: string, mode: AgentMode): Promise<ModelModes> {
      deps.assertIdle('reload the model')
      refuseRental(model)
      const modes = await deps.client.reloadMode(model, mode)
      deps.refreshModels()
      return modes
    }
  }
}
