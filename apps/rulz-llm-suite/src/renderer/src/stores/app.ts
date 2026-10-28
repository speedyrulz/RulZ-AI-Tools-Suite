import { create } from 'zustand'
import { toast } from 'sonner'
import type { AppInfo } from '@shared/ipc'
import type { AppSettings, EngineInfo } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'

export type Page = 'chat' | 'models' | 'mcp' | 'settings' | 'discover' | 'developer'

interface AppState {
  page: Page
  settings: AppSettings | null
  engine: EngineInfo | null
  engineChecking: boolean
  info: AppInfo | null
  pickerOpen: boolean
  /** Model preselected when the picker opens (e.g. from My Models). */
  pickerModelId: string | null
  logInstanceId: string | null
  setPage: (page: Page) => void
  openPicker: (modelId?: string | null) => void
  closePicker: () => void
  showLog: (instanceId: string | null) => void
  init: () => Promise<void>
  updateSettings: (patch: Partial<AppSettings>) => Promise<void>
  /** Saves a theme as the user's choice (also ends the first-run prompt). */
  setTheme: (theme: AppSettings['theme']) => Promise<void>
  refreshEngine: () => Promise<void>
}

export const useApp = create<AppState>((set, get) => ({
  page: 'chat',
  settings: null,
  engine: null,
  engineChecking: false,
  info: null,
  pickerOpen: false,
  pickerModelId: null,
  logInstanceId: null,
  setPage: (page) => set({ page }),
  openPicker: (modelId = null) => set({ pickerOpen: true, pickerModelId: modelId }),
  closePicker: () => set({ pickerOpen: false, pickerModelId: null }),
  showLog: (logInstanceId) => set({ logInstanceId }),

  init: async () => {
    const [settings, info] = await Promise.all([invoke('settings:get'), invoke('app:info')])
    set({ settings, info })
    applyTheme(settings.theme)
    void get().refreshEngine()
  },

  updateSettings: async (patch) => {
    try {
      const settings = await invoke('settings:update', patch)
      set({ settings })
      if (patch.theme) applyTheme(settings.theme)
      if (patch.enginePath !== undefined) void get().refreshEngine()
    } catch (e) {
      toast.error(`Could not save settings: ${errorMessage(e)}`)
    }
  },

  setTheme: async (theme) => {
    applyTheme(theme)
    await get().updateSettings({ theme, themeChosen: true })
  },

  refreshEngine: async () => {
    set({ engineChecking: true })
    try {
      set({ engine: await invoke('engine:info', true) })
    } finally {
      set({ engineChecking: false })
    }
  }
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
