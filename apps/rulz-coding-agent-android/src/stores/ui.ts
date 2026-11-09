import { create } from 'zustand'

export type View = 'projects' | 'workspace' | 'settings'
export type Tab = 'chat' | 'steps' | 'files' | 'app'

interface UiState {
  view: View
  /** Where Settings goes back to. */
  previous: View
  tab: Tab
  go: (view: View) => void
  back: () => void
  setTab: (tab: Tab) => void
  /** Settings shows Model settings expanded (opened from the model sheet). */
  modelSettingsOpen: boolean
  setModelSettingsOpen: (open: boolean) => void
}

export const useUi = create<UiState>((set, get) => ({
  view: 'projects',
  previous: 'projects',
  tab: 'chat',
  go: (view) => set((s) => ({ view, previous: view === 'settings' ? s.view : s.previous, tab: view === 'workspace' && s.view !== 'workspace' ? 'chat' : s.tab })),
  back: () => set({ view: get().view === 'settings' ? get().previous : 'projects' }),
  setTab: (tab) => set({ tab }),
  modelSettingsOpen: false,
  setModelSettingsOpen: (modelSettingsOpen) => set({ modelSettingsOpen })
}))
