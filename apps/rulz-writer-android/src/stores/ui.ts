import { create } from 'zustand'

export type View = 'projects' | 'workspace' | 'settings' | 'adventures' | 'adventure-new' | 'adventure-play'
export type Tab = 'write' | 'bible' | 'state' | 'check' | 'interview'

interface UiState {
  view: View
  /** Where Settings goes back to. */
  previous: View
  tab: Tab
  /** The scene being written. */
  sceneId: string | null
  /** The character being interviewed. */
  characterId: string | null
  outlineOpen: boolean
  jobsOpen: boolean
  go: (view: View) => void
  back: () => void
  setTab: (tab: Tab) => void
  setScene: (sceneId: string | null) => void
  setCharacter: (id: string | null) => void
  setOutlineOpen: (open: boolean) => void
  setJobsOpen: (open: boolean) => void
}

export const useUi = create<UiState>((set, get) => ({
  view: 'projects',
  previous: 'projects',
  tab: 'write',
  sceneId: null,
  characterId: null,
  outlineOpen: false,
  jobsOpen: false,
  go: (view) => set((s) => ({ view, previous: view === 'settings' ? s.view : s.previous, tab: view === 'workspace' && s.view !== 'workspace' ? 'write' : s.tab })),
  back: () => set({ view: get().view === 'settings' ? get().previous : get().view === 'adventure-new' || get().view === 'adventure-play' ? 'adventures' : 'projects' }),
  setTab: (tab) => set({ tab }),
  setScene: (sceneId) => set({ sceneId }),
  setCharacter: (characterId) => set({ characterId }),
  setOutlineOpen: (outlineOpen) => set({ outlineOpen }),
  setJobsOpen: (jobsOpen) => set({ jobsOpen })
}))
