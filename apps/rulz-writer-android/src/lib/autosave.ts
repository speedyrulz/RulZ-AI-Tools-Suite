import type { NovelChange, NovelSaveRequest, NovelSaveResult, NovelTextFile } from '@shared/novel'

// The editor's buffer for one file and the rules for saving it: pure functions, so they can be tested.
// The phone saves with the same data safety rules as the desktop: every save carries the hash of the text the
// buffer was loaded from (or last saved as), a changed file comes back as a conflict, and the phone never forces.

export const AUTOSAVE_MS = 1000
/** After a failed save (computer unreachable), try again after this long. */
export const RETRY_MS = 5000

export interface SaveConflict {
  diskHash: string
  diskText: string
}

export interface Buffer {
  folder: string
  path: string
  text: string
  /** Hash of the disk text this buffer is based on. */
  baseHash: string
  /** The text that was last loaded or saved (what the disk holds, as far as the phone knows). */
  savedText: string
  /** The file is not valid UTF-8 or the project is blocked: shown, never saved. */
  readOnly: boolean
  conflict: SaveConflict | null
  /** The file changed or was removed on the computer while there are unsaved edits. */
  changedOnDisk: 'none' | 'changed' | 'deleted'
  /** A save is in flight with this text; null when none is. */
  saving: string | null
  /** The last save failed with this message (a refusal or no connection). */
  error: string | null
  /** The newest hash a change event reported while a save was in flight (null: none). */
  pendingChangeHash: string | null
  /** The computer changed the file after our save and there is nothing unsaved: load it again. */
  needsReload: boolean
  /** The next save asks the computer to snapshot the text it replaces (Keep mine, Accept). */
  snapshotNext: boolean
}

export function loadBuffer(folder: string, file: NovelTextFile): Buffer {
  return {
    folder,
    path: file.path,
    text: file.text,
    baseHash: file.hash,
    savedText: file.text,
    readOnly: file.readOnly,
    conflict: null,
    changedOnDisk: 'none',
    saving: null,
    error: null,
    pendingChangeHash: null,
    needsReload: false,
    snapshotNext: false
  }
}

export function isDirty(b: Buffer): boolean {
  return b.text !== b.savedText
}

export function edit(b: Buffer, text: string): Buffer {
  return b.readOnly ? b : { ...b, text }
}

/** The save to send now, or null when there is nothing to do (clean, saving, in conflict, read-only). */
export function saveRequest(b: Buffer): NovelSaveRequest | null {
  if (b.readOnly || b.conflict || b.saving !== null || !isDirty(b)) return null
  return { folder: b.folder, path: b.path, text: b.text, baseHash: b.baseHash, ...(b.snapshotNext ? { snapshot: true } : {}) }
}

/** The next save snapshots the disk text it replaces. */
export function withSnapshot(b: Buffer): Buffer {
  return { ...b, snapshotNext: true }
}

/** Marks the save as sent. */
export function saving(b: Buffer): Buffer {
  return { ...b, saving: b.text, error: null }
}

/** What the computer answered. `sent` is the text of the request. */
export function afterSave(b: Buffer, sent: string, result: NovelSaveResult): Buffer {
  const base = { ...b, saving: null }
  if (result.ok) {
    const saved = { ...base, pendingChangeHash: null, baseHash: result.hash, savedText: sent, error: null, changedOnDisk: 'none' as const, needsReload: false, snapshotNext: false }
    // A change event during the save with another hash means the computer changed the file after our write.
    if (b.pendingChangeHash !== null && b.pendingChangeHash !== result.hash) {
      return isDirty(saved) ? { ...saved, changedOnDisk: 'changed' } : { ...saved, needsReload: true }
    }
    return saved
  }
  if (result.reason === 'conflict') return { ...base, conflict: { diskHash: result.diskHash, diskText: result.diskText }, error: null }
  if (result.reason === 'readOnly') return { ...base, readOnly: true, error: 'This file can’t be saved from here (it isn’t plain UTF-8 text).' }
  if (result.reason === 'blocked') return { ...base, error: 'The project is read-only on the computer because project.json or outline.json couldn’t be read.' }
  return { ...base, error: result.reason === 'error' && result.error ? result.error : 'The computer couldn’t save the file.' }
}

/** A failed request (no connection, timeout): the text stays unsaved. */
export function saveFailed(b: Buffer, message: string): Buffer {
  return { ...b, saving: null, error: message }
}

export type ChangeAction = 'ignore' | 'reload' | 'banner'

/**
 * A `novel:changed` event for this folder. Our own saves come back with the hash we just saved (ignored). A
 * different hash reloads a clean buffer and warns about a dirty one.
 */
export function onChanged(b: Buffer, change: NovelChange): { buffer: Buffer; action: ChangeAction } {
  if (change.kind === 'state' || change.folder !== b.folder) return { buffer: b, action: 'ignore' }
  if (change.path !== b.path) return { buffer: b, action: 'ignore' }
  if (change.kind === 'deleted') {
    return { buffer: { ...b, changedOnDisk: 'deleted' }, action: 'banner' }
  }
  if (change.hash === b.baseHash) return { buffer: b, action: 'ignore' }
  // A save of ours is in flight: remember the newest hash; the save's answer decides whether it is a change.
  if (b.saving !== null) return { buffer: { ...b, pendingChangeHash: change.hash }, action: 'ignore' }
  if (!isDirty(b)) return { buffer: b, action: 'reload' }
  return { buffer: { ...b, changedOnDisk: 'changed' }, action: 'banner' }
}

/** "Keep mine" after a conflict: the buffer is now based on the disk version (so even text undone back to the old
 *  saved text is written), and the save snapshots the disk text it replaces. It is a normal save on the new base;
 *  it still conflicts if the disk moves again. */
export function keepMine(b: Buffer): Buffer {
  if (!b.conflict) return b
  return { ...b, baseHash: b.conflict.diskHash, savedText: b.conflict.diskText, conflict: null, changedOnDisk: 'none', snapshotNext: true }
}

/** "Use the computer's version": the buffer becomes the disk text. */
export function takeDisk(b: Buffer): Buffer {
  if (!b.conflict) return b
  return { ...b, text: b.conflict.diskText, savedText: b.conflict.diskText, baseHash: b.conflict.diskHash, conflict: null, changedOnDisk: 'none', error: null, snapshotNext: false }
}

/** The buffer replaced by a fresh read of the file (after a change on the computer, or "load it again"). */
export function reload(b: Buffer, file: NovelTextFile): Buffer {
  return { ...loadBuffer(b.folder, file) }
}

// ---------------------------------------------------------------------------------------------
// The unsaved draft kept on the phone (Capacitor Preferences) and offered back after a restart
// ---------------------------------------------------------------------------------------------

export interface Draft {
  text: string
  baseHash: string
  savedAt: number
}

export type DraftMap = Record<string, Draft>

/** The key of a draft. With a computer id, two paired computers never share a slot; without it, the old key. */
export function draftKey(folder: string, path: string, computerId?: string): string {
  return computerId ? `${computerId}\n${folder}\n${path}` : `${folder}\n${path}`
}

/** The draft for a file: the current computer's slot, else the old format (made before drafts had a computer id). */
export function findDraft(map: DraftMap, folder: string, path: string, computerId: string): Draft | undefined {
  return map[draftKey(folder, path, computerId)] ?? map[draftKey(folder, path)]
}

/** Offer the draft back when it differs from what is on the computer now. */
export function draftToOffer(draft: Draft | undefined, file: NovelTextFile): { draft: Draft; fileChanged: boolean } | null {
  if (!draft || file.readOnly) return null
  if (draft.text === file.text) return null
  return { draft, fileChanged: draft.baseHash !== file.hash }
}
