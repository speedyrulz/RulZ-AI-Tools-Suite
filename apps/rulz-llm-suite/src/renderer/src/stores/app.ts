import { create } from 'zustand'
import { toast } from 'sonner'
import type { AppInfo } from '@shared/ipc'
import type { AppSettings, EngineInfo } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { once } from '../lib/once'

export type Page = 'chat' | 're' | 'models' | 'mcp' | 'settings' | 'discover' | 'developer'

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

/** Counts settings updates, so only the newest one's reply is shown. */
let settingsSeq = 0

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

  init: once(async () => {
    const [settings, info] = await Promise.all([invoke('settings:get'), invoke('app:info')])
    set({ settings, info })
    applyTheme(settings.theme)
    void get().refreshEngine()
  }),

  updateSettings: async (patch) => {
    // Show the change at once: a field bound to a setting would otherwise snap back to the old
    // value until the save returns, which drops keystrokes. Main merges patches the same way.
    const seq = ++settingsSeq
    const prev = get().settings
    if (prev) set({ settings: { ...prev, ...patch } })
    try {
      const settings = await invoke('settings:update', patch)
      // An older save finishing after a newer change would put the old value back on screen.
      if (seq === settingsSeq) set({ settings })
      if (patch.theme) applyTheme(settings.theme)
      if (patch.enginePath !== undefined) void get().refreshEngine()
    } catch (e) {
      toast.error(`Could not save settings: ${errorMessage(e)}`)
      // Put back what was actually saved.
      const saved = await invoke('settings:get').catch(() => prev)
      if (seq === settingsSeq) set({ settings: saved })
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
