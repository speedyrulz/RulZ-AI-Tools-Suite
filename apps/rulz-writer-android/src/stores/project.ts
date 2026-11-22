import { toast } from 'sonner'
import { create } from 'zustand'
import type { NovelProjectState, NovelProjectSummary, NovelStatePatchResult, NovelTrashItem, OutlineOp, StateOp } from '@shared/novel'
import { errorMessage } from '../lib/format'
import { link } from '../lib/remote'
import { findChapter, findScene } from '@shared/novelOutline'
import { scenePath } from '../lib/outlineOps'
import { discardDraftFor, useScene } from './editor'

interface ProjectStore {
  projects: NovelProjectSummary[]
  listLoading: boolean
  listError: string
  /** The open project, as the computer last sent it. */
  state: NovelProjectState | null
  trash: NovelTrashItem[]
  loadList: () => Promise<void>
  create: (title: string) => Promise<boolean>
  open: (folder: string) => Promise<boolean>
  /** After a reconnect: the computer let go of the project with the old connection, so open it again. */
  reopen: () => Promise<void>
  close: () => Promise<void>
  setState: (state: NovelProjectState) => void
  patchOutline: (ops: OutlineOp[]) => Promise<boolean>
  statePatch: (ops: StateOp[]) => Promise<NovelStatePatchResult | null>
  deleteScene: (sceneId: string) => Promise<boolean>
  deleteChapter: (chapterId: string) => Promise<boolean>
  deleteEntry: (entryId: string) => Promise<boolean>
  loadTrash: () => Promise<void>
  restore: (id: string) => Promise<boolean>
  setModel: (modelId: string | null) => Promise<void>
  dismissFinding: (sceneId: string, key: string, dismissed: boolean) => Promise<void>
}

export const useProject = create<ProjectStore>((set, get) => {
  /** Runs a call that returns the new project state, keeps it, and reports a refusal. */
  const withState = async (call: (folder: string) => Promise<NovelProjectState>): Promise<boolean> => {
    const folder = get().state?.folder
    if (!folder) return false
    try {
      const state = await call(folder)
      if (get().state?.folder === folder) set({ state })
      return true
    } catch (e) {
      toast.error(errorMessage(e))
      return false
    }
  }

  /**
   * Deletes scenes. If the open scene is one of them, the editor lets go first, without saving (an autosave
   * would write the file back) and its draft is forgotten; if the delete is refused it opens again.
   */
  const deleteOpen = async (paths: string[], call: (folder: string) => Promise<NovelProjectState>): Promise<boolean> => {
    const folder = get().state?.folder
    const open = useScene.getState().buffer
    const hit = !!folder && !!open && open.folder === folder && paths.includes(open.path)
    if (hit) {
      useScene.getState().drop()
      await discardDraftFor(open!.folder, open!.path)
    }
    const ok = await withState(call)
    if (hit && !ok && folder) void useScene.getState().open(folder, open!.path)
    return ok
  }

  return {
    projects: [],
    listLoading: false,
    listError: '',
    state: null,
    trash: [],

    loadList: async () => {
      set({ listLoading: true })
      try {
        set({ projects: await link.invoke('novel:listProjects'), listError: '' })
      } catch (e) {
        set({ listError: errorMessage(e) })
      } finally {
        set({ listLoading: false })
      }
    },

    create: async (title) => {
      try {
        const state = await link.invoke('novel:createProject', title.trim())
        await get().close()
        set({ state, trash: [] })
        void get().loadList()
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    open: async (folder) => {
      try {
        const state = await link.invoke('novel:openProject', folder)
        const old = get().state
        if (old && old.folder !== folder) void link.invoke('novel:closeProject', old.folder).catch(() => undefined)
        set({ state, trash: [] })
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    reopen: async () => {
      const folder = get().state?.folder
      if (!folder) return
      try {
        set({ state: await link.invoke('novel:openProject', folder) })
      } catch (e) {
        toast.error(`Couldn’t open the project again: ${errorMessage(e)}`)
      }
    },

    close: async () => {
      const folder = get().state?.folder
      set({ state: null, trash: [] })
      if (folder) await link.invoke('novel:closeProject', folder).catch(() => undefined)
    },

    setState: (state) => set({ state }),

    patchOutline: (ops) => withState((folder) => link.invoke('novel:outlinePatch', folder, ops)),

    statePatch: async (ops) => {
      const folder = get().state?.folder
      if (!folder) return null
      try {
        const result = await link.invoke('novel:statePatch', folder, ops)
        if (get().state?.folder === folder) set({ state: result.state })
        for (const r of result.rejected) toast.error(r.reason)
        return result
      } catch (e) {
        toast.error(errorMessage(e))
        return null
      }
    },

    deleteScene: async (sceneId) => {
      const s = get().state
      const found = s && findScene(s.outline, sceneId)
      return deleteOpen(found ? [scenePath(found.scene)] : [], (folder) => link.invoke('novel:deleteScene', folder, sceneId))
    },
    deleteChapter: async (chapterId) => {
      const s = get().state
      const found = s && findChapter(s.outline, chapterId)
      return deleteOpen(found ? found.chapter.scenes.map(scenePath) : [], (folder) => link.invoke('novel:deleteChapter', folder, chapterId))
    },
    deleteEntry: (entryId) => withState((folder) => link.invoke('novel:deleteEntry', folder, entryId)),

    loadTrash: async () => {
      const folder = get().state?.folder
      if (!folder) return
      try {
        set({ trash: await link.invoke('novel:trash', folder) })
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    restore: async (id) => {
      const ok = await withState((folder) => link.invoke('novel:restoreTrash', folder, id))
      if (ok) await get().loadTrash()
      return ok
    },

    setModel: async (modelId) => {
      const folder = get().state?.folder
      if (!folder) return
      try {
        const project = await link.invoke('novel:projectPatch', folder, { modelId })
        const s = get().state
        if (s && s.folder === folder) set({ state: { ...s, project } })
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    dismissFinding: async (sceneId, key, dismissed) => {
      await withState((folder) => link.invoke('novel:dismissFinding', folder, sceneId, key, dismissed))
    }
  }
})

/** Applies events from the computer: the project state changed (outline, project, bible, state). */
export function listenForProjectChanges(): void {
  link.on('novel:changed', (change) => {
    if (change.kind !== 'state') return
    const s = useProject.getState().state
    if (s && s.folder === change.folder) useProject.getState().setState(change.state)
  })
}
