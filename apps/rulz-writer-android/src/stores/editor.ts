import { toast } from 'sonner'
import { create, type StoreApi, type UseBoundStore } from 'zustand'
import type { NovelChange, NovelTextFile } from '@shared/novel'
import {
  afterSave,
  AUTOSAVE_MS,
  draftKey,
  draftToOffer,
  edit,
  findDraft,
  isDirty,
  keepMine,
  loadBuffer,
  onChanged,
  reload,
  RETRY_MS,
  saveFailed,
  saveRequest,
  saving,
  takeDisk,
  withSnapshot,
  type Buffer,
  type Draft,
  type DraftMap
} from '../lib/autosave'
import { errorMessage } from '../lib/format'
import { loadPref, savePref } from '../lib/native'
import { link } from '../lib/remote'
import { currentComputer } from './link'

// One open text file on the phone (a scene, or a bible entry): the buffer, the autosave, conflicts, the unsaved
// draft kept in Preferences, and what happens when the computer reports that the file changed.

export interface EditorState {
  buffer: Buffer | null
  loading: boolean
  loadError: string
  /** An unsaved draft from before the app was closed, waiting for the owner's choice. The text is read-only until then. */
  offer: { draft: Draft; fileChanged: boolean } | null
  /** The file is being left (unsaved text is being sent or kept): no edits. */
  leaving: boolean
  /** A passage to select once the text is on screen (a check finding). */
  pendingSelection: { from: number; to: number } | null
  /** Opens a file. Resolves false when the file could not be opened (a conflict on the open file needs an answer first). */
  open: (folder: string, path: string) => Promise<boolean>
  /**
   * Saves what is unsaved, keeps a draft of what could not be saved, and lets go of the file. Resolves false (and
   * keeps the file open) while a conflict waits for an answer.
   */
  leave: () => Promise<boolean>
  /** Lets go of the file without saving (another computer was chosen); the unsaved draft stays in Preferences. */
  drop: () => void
  edit: (text: string) => void
  /** Saves now (waiting for a save in flight). Resolves when the buffer is saved or the save was refused. */
  saveNow: () => Promise<void>
  /** Puts text in the buffer and saves it at once (accepting an AI result). */
  applyText: (text: string) => Promise<boolean>
  keepMine: () => Promise<void>
  useDisk: () => void
  /** Loads the computer's version, dropping unsaved edits (unless more was typed while it was being read). */
  loadFromDisk: () => Promise<void>
  restoreDraft: () => void
  discardDraft: () => void
  /** After a reconnect: send what is unsaved, or pick up changes made while away. */
  resync: () => Promise<void>
  onChange: (change: NovelChange) => void
  select: (from: number, to: number) => void
  clearSelection: () => void
}

export type EditorStore = UseBoundStore<StoreApi<EditorState>>

let draftsCache: DraftMap | null = null
let draftWrite: ReturnType<typeof setTimeout> | null = null

/** For tests: forget the drafts read from Preferences. */
export function resetDraftCache(): void {
  draftsCache = null
  if (draftWrite) clearTimeout(draftWrite)
  draftWrite = null
}

async function drafts(): Promise<DraftMap> {
  draftsCache ??= await loadPref<DraftMap>('drafts', {})
  return draftsCache
}

function persistDrafts(): void {
  if (draftWrite) clearTimeout(draftWrite)
  draftWrite = setTimeout(() => {
    draftWrite = null
    if (draftsCache) void savePref('drafts', draftsCache)
  }, 500)
}

/** Writes the drafts to Preferences now. */
async function flushDrafts(): Promise<void> {
  if (draftWrite) clearTimeout(draftWrite)
  draftWrite = null
  if (draftsCache) await savePref('drafts', draftsCache)
}

const computerId = (): string => currentComputer()?.id ?? ''

async function storeDraft(b: Buffer): Promise<void> {
  const map = await drafts()
  const key = draftKey(b.folder, b.path, computerId())
  if (isDirty(b)) map[key] = { text: b.text, baseHash: b.baseHash, savedAt: Date.now() }
  else delete map[key]
  persistDrafts()
}

/** Forgets the draft of a file that is gone (deleted on the computer). */
export async function discardDraftFor(folder: string, path: string): Promise<void> {
  await forgetDraft(folder, path)
}

async function forgetDraft(folder: string, path: string): Promise<void> {
  const map = await drafts()
  delete map[draftKey(folder, path, computerId())]
  // The old key (before drafts had a computer id) is this computer's too; once handled it is dropped.
  delete map[draftKey(folder, path)]
  persistDrafts()
}

export function createEditor(): EditorStore {
  return create<EditorState>((set, get) => {
    let timer: ReturnType<typeof setTimeout> | null = null
    let inflight: Promise<void> | null = null
    let epoch = 0
    /** Counts changes to the buffer's text (typing, restore, accept, take the computer's text). */
    let revision = 0
    /** Counts reads of the file for reloading; only the newest may apply. */
    let readGen = 0
    /** The newest file asked for (current or still opening); null when nothing is open or wanted. */
    let wanted: { folder: string; path: string } | null = null
    /** Counts open() calls, so an older one that was waiting for leave() gives up. */
    let openId = 0

    const clearTimer = (): void => {
      if (timer) clearTimeout(timer)
      timer = null
    }
    const schedule = (ms: number): void => {
      clearTimer()
      timer = setTimeout(() => void get().saveNow(), ms)
    }
    const update = (fn: (b: Buffer) => Buffer): Buffer | null => {
      const b = get().buffer
      if (!b) return null
      const next = fn(b)
      set({ buffer: next })
      return next
    }
    /** Edits are refused while a draft is waiting for a decision, and while the file is being left. */
    const editable = (): boolean => !get().offer && !get().leaving

    /**
     * Reads the file again and applies it. The result is dropped when anything moved on during the read: another
     * file or computer, a newer read, a save that changed the base, or typing. With typing the owner's text stays
     * and the "changed on the computer" banner shows. `discard` (the owner chose to load the computer's text)
     * replaces unsaved text that was there when the read began.
     */
    const refetch = async (b0: Buffer, discard: boolean): Promise<void> => {
      const at = epoch
      const gen = ++readGen
      const rev0 = revision
      let file: NovelTextFile
      try {
        file = await link.invoke('novel:readFile', b0.folder, b0.path)
      } catch (e) {
        if (discard) toast.error(errorMessage(e))
        return
      }
      const now = get().buffer
      if (at !== epoch || gen !== readGen || !now || now.folder !== b0.folder || now.path !== b0.path) return
      if (now.saving !== null || now.baseHash !== b0.baseHash) return
      if (revision !== rev0) {
        // The owner typed while this was being read: their text stays, and they decide.
        if (isDirty(now) && file.hash !== now.baseHash) set({ buffer: { ...now, changedOnDisk: 'changed' } })
        return
      }
      if (isDirty(now) && !discard) {
        if (file.hash !== now.baseHash) set({ buffer: { ...now, changedOnDisk: 'changed' } })
        return
      }
      clearTimer()
      const next = reload(now, file)
      set({ buffer: next })
      void forgetDraft(next.folder, next.path)
    }

    const runSave = async (): Promise<void> => {
      const b = get().buffer
      const req = b && saveRequest(b)
      if (!b || !req) return
      const at = epoch
      set({ buffer: saving(b) })
      let next: Buffer
      try {
        const result = await link.invoke('novel:saveFile', req)
        if (at !== epoch) return
        next = afterSave(get().buffer!, req.text, result)
      } catch (e) {
        if (at !== epoch) return
        next = saveFailed(get().buffer!, errorMessage(e))
        set({ buffer: next })
        void storeDraft(next)
        schedule(RETRY_MS)
        return
      }
      set({ buffer: next })
      if (next.error) {
        void storeDraft(next)
        return
      }
      if (next.needsReload) {
        // The computer changed the file after our write, and nothing is unsaved: take its version.
        set({ buffer: { ...next, needsReload: false } })
        void forgetDraft(next.folder, next.path)
        void refetch(get().buffer!, false)
        return
      }
      if (isDirty(next) && !next.conflict) schedule(AUTOSAVE_MS)
      else if (!next.conflict) void forgetDraft(next.folder, next.path)
    }

    const saveNow = async (): Promise<void> => {
      clearTimer()
      // One save at a time: wait for the one in flight, then send whatever is unsaved by then.
      while (inflight) await inflight
      inflight = runSave().finally(() => (inflight = null))
      await inflight
      // Text typed while that save was in flight is still unsaved; the timer set by runSave sends it.
    }

    /**
     * Saves what is unsaved, keeps a draft of anything that could not be saved, and clears the buffer. Refuses
     * (false) while a conflict waits for an answer: the ConflictSheet must be shown, not skipped.
     */
    const release = async (): Promise<boolean> => {
      const b = get().buffer
      clearTimer()
      if (b?.conflict) return false
      if (b) {
        // No typing while unsaved text is sent or kept, so nothing typed during the awaits is lost.
        set({ leaving: true })
        try {
          if (isDirty(b) && !b.readOnly && !get().offer) await saveNow()
          // Re-read: the save (or the owner, before the flag) may have left something unsaved.
          const after = get().buffer
          if (after?.conflict) return false
          if (after && isDirty(after) && !get().offer) {
            try {
              await storeDraft(after)
              await flushDrafts()
            } catch (e) {
              toast.error(`Your unsaved text couldn’t be kept: ${errorMessage(e)}`)
            }
          }
        } finally {
          set({ leaving: false })
        }
      }
      epoch++
      revision++
      set({ buffer: null, offer: null, loadError: '', pendingSelection: null })
      return true
    }

    return {
      buffer: null,
      loading: false,
      loadError: '',
      offer: null,
      leaving: false,
      pendingSelection: null,

      open: async (folder, path) => {
        // Compared with the newest file asked for, not with the buffer: while an older open is still waiting
        // for leave(), the buffer is not the file the screen wants.
        if (wanted && wanted.folder === folder && wanted.path === path) return true
        const previous = wanted
        wanted = { folder, path }
        const mine = ++openId
        const released = await release()
        if (mine !== openId) return false
        if (!released) {
          // A conflict on the open file needs an answer first; that file stays the current one.
          wanted = previous
          return false
        }
        const at = ++epoch
        revision++
        set({ buffer: null, loading: true, loadError: '', offer: null, pendingSelection: null })
        try {
          const file: NovelTextFile = await link.invoke('novel:readFile', folder, path)
          if (at !== epoch) return false
          const offer = draftToOffer(findDraft(await drafts(), folder, path, computerId()), file)
          if (at !== epoch) return false
          set({ buffer: loadBuffer(folder, file), loading: false, offer })
          if (!offer) void forgetDraft(folder, path)
          return true
        } catch (e) {
          if (at !== epoch) return false
          wanted = null
          set({ loading: false, loadError: errorMessage(e) })
          return false
        }
      },

      leave: () => {
        wanted = null
        return release()
      },

      drop: () => {
        wanted = null
        clearTimer()
        epoch++
        revision++
        set({ buffer: null, offer: null, leaving: false, loadError: '', pendingSelection: null, loading: false })
      },

      edit: (text) => {
        if (!editable()) return
        const before = get().buffer
        const next = update((b) => edit(b, text))
        if (!next || next.readOnly) return
        if (before && before.text !== text) revision++
        void storeDraft(next)
        if (!next.conflict) schedule(AUTOSAVE_MS)
      },

      saveNow,

      applyText: async (text) => {
        const b0 = get().buffer
        if (!b0 || b0.readOnly || !editable()) return false
        const previous = b0.text
        revision++
        update((b) => withSnapshot(edit(b, text)))
        await saveNow()
        const b = get().buffer
        const saved = !!b && b.path === b0.path && b.savedText === text && !b.conflict && !b.error
        if (!saved && b && b.path === b0.path && b.text === text) {
          // The save failed or conflicted: the AI text goes back out of the buffer (the job card still has it),
          // so it is not autosaved later and Reject or another alternative still work.
          revision++
          clearTimer()
          const rolled = { ...b, text: previous, snapshotNext: false }
          set({ buffer: rolled })
          void storeDraft(rolled)
          if (isDirty(rolled) && !rolled.conflict) schedule(RETRY_MS)
        }
        return saved
      },

      keepMine: async () => {
        update(keepMine)
        await saveNow()
      },

      useDisk: () => {
        revision++
        const next = update(takeDisk)
        if (next) void forgetDraft(next.folder, next.path)
      },

      loadFromDisk: async () => {
        const b = get().buffer
        if (b) await refetch(b, true)
      },

      restoreDraft: () => {
        const { offer, buffer } = get()
        if (!offer || !buffer) return
        // A draft made on an older version of the file keeps its old base, so saving it shows the conflict
        // instead of silently overwriting what changed on the computer.
        revision++
        set({
          offer: null,
          buffer: { ...buffer, text: offer.draft.text, baseHash: offer.fileChanged ? offer.draft.baseHash : buffer.baseHash }
        })
        schedule(AUTOSAVE_MS)
      },

      discardDraft: () => {
        const b = get().buffer
        set({ offer: null })
        if (b) void forgetDraft(b.folder, b.path)
      },

      resync: async () => {
        const b = get().buffer
        if (!b || get().offer) return
        if (isDirty(b)) {
          if (!b.conflict) await saveNow()
          return
        }
        await refetch(b, false)
      },

      onChange: (change) => {
        const b = get().buffer
        if (!b) return
        const { buffer, action } = onChanged(b, change)
        set({ buffer })
        if (action === 'reload') void refetch(buffer, false)
      },

      select: (from, to) => set({ pendingSelection: { from, to } }),
      clearSelection: () => set({ pendingSelection: null })
    }
  })
}

/** The scene being written. */
export const useScene = createEditor()
/** The bible entry being edited. */
export const useEntry = createEditor()

/** Sends `novel:changed` events to the open editors. */
export function listenForFileChanges(): void {
  link.on('novel:changed', (change) => {
    useScene.getState().onChange(change)
    useEntry.getState().onChange(change)
  })
}

export function resyncEditors(): void {
  void useScene.getState().resync()
  void useEntry.getState().resync()
}
