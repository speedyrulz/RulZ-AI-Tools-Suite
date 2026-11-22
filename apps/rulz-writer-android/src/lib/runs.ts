import type { NovelAction, RewriteVariant } from '@shared/novel'
import type { WritingRunRequest } from '@shared/phone'

// The AI actions the phone starts on a scene and how a request is put together from the editor.

export interface ActionDef {
  id: string
  action: NovelAction
  variant?: RewriteVariant
  label: string
  hint: string
  /** The action works on a selected passage. */
  needsSelection: boolean
  /** The note field is needed (the action does nothing useful without it). */
  needsNote?: boolean
}

export const PROSE_ACTIONS: ActionDef[] = [
  { id: 'continue', action: 'continue', label: 'Continue', hint: 'Writes on from the cursor.', needsSelection: false },
  { id: 'draftScene', action: 'draftScene', label: 'Draft scene', hint: 'Writes the scene from its beats.', needsSelection: false },
  { id: 'dialogue', action: 'dialogue', label: 'Dialogue', hint: 'Writes the dialogue for the current beat.', needsSelection: false },
  { id: 'expand', action: 'expand', label: 'Expand', hint: 'Adds detail to the selected passage.', needsSelection: true },
  { id: 'condense', action: 'condense', label: 'Condense', hint: 'Shortens the selected passage.', needsSelection: true },
  { id: 'rewrite:tighter', action: 'rewrite', variant: 'tighter', label: 'Rewrite: tighter', hint: 'Cuts the selected passage down.', needsSelection: true },
  { id: 'rewrite:vivid', action: 'rewrite', variant: 'vivid', label: 'Rewrite: more vivid', hint: 'Sharper images in the selected passage.', needsSelection: true },
  { id: 'rewrite:tone', action: 'rewrite', variant: 'tone', label: 'Rewrite: change tone', hint: 'Say the tone in the note, for example “colder”.', needsSelection: true, needsNote: true },
  { id: 'rewrite:show', action: 'rewrite', variant: 'show', label: 'Rewrite: show, don’t tell', hint: 'Turns telling into showing.', needsSelection: true },
  { id: 'rewrite:custom', action: 'rewrite', variant: 'custom', label: 'Rewrite: your way', hint: 'Say what you want in the note.', needsSelection: true, needsNote: true }
]

export function findAction(id: string): ActionDef {
  return PROSE_ACTIONS.find((a) => a.id === id) ?? PROSE_ACTIONS[0]
}

export interface EditorSnapshot {
  text: string
  baseHash: string
  selectionStart: number
  selectionEnd: number
}

/** Why the action can't start now, or null. */
export function actionProblem(def: ActionDef, snap: EditorSnapshot, note: string): string | null {
  const hasSelection = snap.selectionEnd > snap.selectionStart
  if (def.needsSelection && !hasSelection) return 'Select the passage in the scene first.'
  if (def.needsNote && !note.trim()) return 'Write a note first.'
  return null
}

export function clampAlternatives(n: number): number {
  return Math.min(4, Math.max(1, Math.round(n) || 1))
}

export function proseRequest(folder: string, sceneId: string, def: ActionDef, snap: EditorSnapshot, note: string, alternatives: number): WritingRunRequest {
  const from = Math.min(snap.selectionStart, snap.selectionEnd)
  const to = Math.max(snap.selectionStart, snap.selectionEnd)
  return {
    folder,
    action: def.action,
    sceneId,
    text: snap.text,
    baseHash: snap.baseHash,
    cursor: Math.min(snap.selectionEnd, snap.text.length),
    selection: to > from ? { from, to } : null,
    note: note.trim(),
    variant: def.variant,
    alternatives: clampAlternatives(alternatives)
  }
}

/** Requests for the actions that take no editor text: summaries, outline proposals, checks, state, interview. */
export function cardRequest(
  folder: string,
  action: NovelAction,
  snap: Pick<EditorSnapshot, 'text' | 'baseHash'> | null,
  extra: Partial<WritingRunRequest>
): WritingRunRequest {
  return {
    folder,
    action,
    sceneId: null,
    text: snap?.text ?? '',
    baseHash: snap?.baseHash ?? '',
    cursor: snap?.text.length ?? 0,
    selection: null,
    note: '',
    ...extra
  }
}
