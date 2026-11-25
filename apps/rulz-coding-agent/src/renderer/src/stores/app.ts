import { create } from 'zustand'
import { toast } from 'sonner'
import type { AppInfo } from '@shared/ipc'
import { chooseModel } from '@shared/modes'
import type { AppSettings, ConnectionInfo, ModelActivity } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'

interface AppState {
  settings: AppSettings | null
  info: AppInfo | null
  connection: ConnectionInfo
  activity: ModelActivity
  settingsOpen: boolean
  init: () => Promise<void>
  updateSettings: (patch: Partial<AppSettings>) => Promise<void>
  setTheme: (theme: AppSettings['theme']) => Promise<void>
  reconnect: () => Promise<ConnectionInfo>
  setSettingsOpen: (open: boolean) => void
}

export const useApp = create<AppState>((set, get) => ({
  settings: null,
  info: null,
  connection: { status: 'checking', url: '', models: [] },
  activity: 'idle',
  settingsOpen: false,

  init: async () => {
    const [settings, info, connection] = await Promise.all([invoke('settings:get'), invoke('app:info'), invoke('server:status')])
    set({ settings, info, connection })
    applyTheme(settings.theme)
    on('server:changed', (connection) => set({ connection }))
    on('model:activity', (activity) => set({ activity }))
    let phones = new Set<string>()
    on('remote:changed', (status) => {
      for (const c of status.clients) if (!phones.has(c.id)) toast.info(`${c.device} connected. It can see and control your projects.`)
      phones = new Set(status.clients.map((c) => c.id))
    })
    if (connection.status === 'checking') void get().reconnect()
  },

  updateSettings: async (patch) => {
    try {
      const settings = await invoke('settings:update', patch)
      set({ settings })
      if (patch.theme) applyTheme(settings.theme)
    } catch (e) {
      toast.error(`Could not save settings: ${errorMessage(e)}`)
    }
  },

  setTheme: async (theme) => {
    applyTheme(theme)
    await get().updateSettings({ theme })
  },

  reconnect: async () => {
    set((s) => ({ connection: { ...s.connection, status: 'checking' } }))
    const connection = await invoke('server:connect')
    set({ connection })
    return connection
  },

  setSettingsOpen: (settingsOpen) => set({ settingsOpen })
}))

const media = window.matchMedia('(prefers-color-scheme: dark)')

export function isDarkTheme(theme: AppSettings['theme']): boolean {
  return theme === 'dark' || (theme === 'system' && media.matches)
}

export function applyTheme(theme: AppSettings['theme']): void {
  document.documentElement.classList.toggle('dark', isDarkTheme(theme))
}

media.addEventListener('change', () => {
  const theme = useApp.getState().settings?.theme
  if (theme === 'system') applyTheme(theme)
})

/**
 * The model the agent will use: the chosen one, or the server's best guess (never a rental's). A
 * chosen model that was on a rental and is gone stays shown as chosen: the app says so and does not
 * use another one (see `chosenRentalGone`).
 */
export function activeModelId(settings: AppSettings | null, connection: ConnectionInfo): string {
  const chosen = settings?.model ?? ''
  const c = chooseModel(chosen, settings?.modelRental ?? false, connection.models)
  return c.ok ? c.model : chosen
}

/** The sentence for a chosen model that ran on a rental and is not in the list now, or null. */
export function chosenRentalGone(settings: AppSettings | null, connection: ConnectionInfo): string | null {
  if (connection.status !== 'connected' || !settings?.model) return null
  const c = chooseModel(settings.model, settings.modelRental, connection.models)
  return !c.ok && settings.modelRental && !connection.models.some((m) => m.id === settings.model) ? c.reason : null
}
