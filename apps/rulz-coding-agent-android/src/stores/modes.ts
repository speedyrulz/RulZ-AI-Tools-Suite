import { useEffect } from 'react'
import { modesStamp } from '@shared/modes'
import type { ModelModes } from '@shared/types'
import { invoke } from '../lib/api'
import { createModesStore } from '../lib/modeLogic'
import { activeModelId, useApp } from './app'
import { useLink } from './link'
import { useWorkspace } from './workspace'

export const useModes = createModesStore({
  modes: (model) => invoke('models:modes', model),
  reloadMode: (model, mode) => invoke('models:reloadMode', model, mode)
})

/**
 * The modes of the model in use. Asked again when the model changes, when the phone reconnects or comes
 * back to the front (see App.tsx), when the model's loaded state or context changes in the list, and when
 * a task ends: nothing polls on a timer. `older`: the computer does not know modes, so show nothing new.
 */
export function useModelModes(): { model: string; modes: ModelModes | null | undefined; older: boolean; refresh: () => void } {
  const settings = useApp((s) => s.settings)
  const connection = useApp((s) => s.connection)
  const running = useWorkspace((s) => s.running)
  const computer = useLink((s) => s.currentId)
  const id = activeModelId(settings, connection)
  const stamp = modesStamp(connection.models.find((m) => m.id === id))
  const connected = connection.status === 'connected'
  const modes = useModes((s) => s.byModel[id])
  const older = useModes((s) => s.older)
  const known = useModes((s) => s.computer === computer)

  useEffect(() => {
    useModes.getState().forComputer(computer)
  }, [computer])
  useEffect(() => {
    useModes.getState().want(id)
    if (connected && id) void useModes.getState().refresh(id, true)
  }, [id, connected, computer])
  useEffect(() => {
    if (connected && id && stamp) void useModes.getState().refresh(id)
  }, [stamp, id, connected])
  useEffect(() => {
    // A task that ends may have left the model reloaded by another app, or in use no more.
    if (!running && connected && id) void useModes.getState().refresh(id)
  }, [running, id, connected])

  return { model: id, modes: known ? modes : undefined, older: known && older, refresh: () => void useModes.getState().refresh(id, true) }
}
