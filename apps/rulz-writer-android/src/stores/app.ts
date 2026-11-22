import { create } from 'zustand'
import type { Theme } from '@shared/types'
import { loadPref, savePref } from '../lib/native'

interface AppState {
  /** This phone's look. */
  theme: Theme
  /** The computer answered at least once since the computer was chosen. */
  connectedOnce: boolean
  init: () => Promise<void>
  setTheme: (theme: Theme) => Promise<void>
}

export const useApp = create<AppState>((set) => ({
  theme: 'dark',
  connectedOnce: false,

  init: async () => {
    const theme = await loadPref<Theme>('theme', 'dark')
    set({ theme })
    applyTheme(theme)
  },

  setTheme: async (theme) => {
    set({ theme })
    applyTheme(theme)
    await savePref('theme', theme)
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
