import { create } from 'zustand'

// The composer's typed text, kept per project so it survives a tab switch (the Composer unmounts off the Chat tab).

interface ComposerState {
  drafts: Record<string, string>
  setDraft: (projectPath: string, text: string) => void
}

const keyOf = (projectPath: string): string => projectPath.toLowerCase()

export const useComposer = create<ComposerState>((set) => ({
  drafts: {},
  setDraft: (projectPath, text) =>
    set((s) => {
      const key = keyOf(projectPath)
      const drafts = { ...s.drafts }
      if (text) drafts[key] = text
      else delete drafts[key]
      return { drafts }
    })
}))

export const draftOf = (projectPath: string): string => useComposer.getState().drafts[keyOf(projectPath)] ?? ''

export const selectDraft = (projectPath: string) => (s: ComposerState): string => s.drafts[keyOf(projectPath)] ?? ''
