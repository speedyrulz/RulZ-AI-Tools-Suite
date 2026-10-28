import { create } from 'zustand'
import { toast } from 'sonner'
import type { AppInfo } from '@shared/ipc'
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

/** The model the agent will use: the chosen one, or the server's best guess. */
export function activeModelId(settings: AppSettings | null, connection: ConnectionInfo): string {
  const chosen = settings?.model ?? ''
  if (chosen && (!connection.models.length || connection.models.some((m) => m.id === chosen))) return chosen
  const coder = (id: string): boolean => /qwen3[-_.]?coder/i.test(id)
  const m = connection.models
  return (m.find((x) => x.loaded && coder(x.id)) ?? m.find((x) => x.loaded) ?? m.find((x) => coder(x.id)) ?? m[0])?.id ?? ''
}
