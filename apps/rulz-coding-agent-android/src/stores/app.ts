import { create } from 'zustand'
import { toast } from 'sonner'
import type { AppInfo } from '@shared/ipc'
import type { AppSettings, ConnectionInfo, ModelActivity, Theme } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'
import { loadPref, savePref } from '../lib/native'

interface AppState {
  /** The computer's settings (shared with its own window). */
  settings: AppSettings | null
  info: AppInfo | null
  /** The computer's connection to RulZ LLM Suite. */
  connection: ConnectionInfo
  activity: ModelActivity
  /** This phone's look (separate from the computer's). */
  theme: Theme
  init: () => Promise<void>
  /** Reloads everything from the computer after (re)connecting. */
  resync: () => Promise<void>
  updateSettings: (patch: Partial<AppSettings>) => Promise<void>
  setTheme: (theme: Theme) => Promise<void>
  reconnect: () => Promise<ConnectionInfo | null>
}

export const useApp = create<AppState>((set) => ({
  settings: null,
  info: null,
  connection: { status: 'checking', url: '', models: [] },
  activity: 'idle',
  theme: 'system',

  init: async () => {
    const theme = await loadPref<Theme>('theme', 'system')
    set({ theme })
    applyTheme(theme)
    on('server:changed', (connection) => set({ connection }))
    on('model:activity', (activity) => set({ activity }))
  },

  resync: async () => {
    try {
      const [settings, info, connection] = await Promise.all([invoke('settings:get'), invoke('app:info'), invoke('server:status')])
      set({ settings, info, connection })
    } catch (e) {
      toast.error(errorMessage(e))
    }
  },

  updateSettings: async (patch) => {
    try {
      set({ settings: await invoke('settings:update', patch) })
    } catch (e) {
      toast.error(`Could not save settings: ${errorMessage(e)}`)
    }
  },

  setTheme: async (theme) => {
    set({ theme })
    applyTheme(theme)
    await savePref('theme', theme)
  },

  reconnect: async () => {
    set((s) => ({ connection: { ...s.connection, status: 'checking' } }))
    try {
      const connection = await invoke('server:connect')
      set({ connection })
      return connection
    } catch (e) {
      toast.error(errorMessage(e))
      void invoke('server:status').then((connection) => set({ connection }), () => undefined)
      return null
    }
  }
}))

const media = window.matchMedia('(prefers-color-scheme: dark)')

export function isDarkTheme(theme: Theme): boolean {
  return theme === 'dark' || (theme === 'system' && media.matches)
}

export function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle('dark', isDarkTheme(theme))
}

media.addEventListener('change', () => {
  if (useApp.getState().theme === 'system') applyTheme('system')
})

/** The model the agent will use: the chosen one, or the server's best guess (same rule as the computer). */
export function activeModelId(settings: AppSettings | null, connection: ConnectionInfo): string {
  const chosen = settings?.model ?? ''
  if (chosen && (!connection.models.length || connection.models.some((m) => m.id === chosen))) return chosen
  const coder = (id: string): boolean => /qwen3[-_.]?coder/i.test(id)
  const m = connection.models
  return (m.find((x) => x.loaded && coder(x.id)) ?? m.find((x) => x.loaded) ?? m.find((x) => coder(x.id)) ?? m[0])?.id ?? ''
}
