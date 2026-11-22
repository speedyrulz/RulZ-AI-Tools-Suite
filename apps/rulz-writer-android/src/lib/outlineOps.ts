import type { BibleEntry, NovelOutline, NovelScene, OutlineOp, SceneStatus } from '@shared/novel'
import { findChapter, findScene, newId } from '@shared/novelOutline'
import type { WritingResult } from '@shared/phone'
import { mergeSticky } from './accept'

// Builds the outline operations the phone sends with novel:outlinePatch. The computer applies them in order,
// makes the file names and writes outline.json; the phone only chooses ids and values.

export const SCENE_STATUSES: { value: SceneStatus; label: string }[] = [
  { value: 'idea', label: 'Idea' },
  { value: 'beats', label: 'Beats' },
  { value: 'draft', label: 'Draft' },
  { value: 'revised', label: 'Revised' }
]

export function statusLabel(status: SceneStatus): string {
  return SCENE_STATUSES.find((s) => s.value === status)?.label ?? status
}

/** Every scene in reading order, with its chapter. */
export function flatScenes(outline: NovelOutline): { chapterId: string; chapterTitle: string; scene: NovelScene }[] {
  return outline.chapters.flatMap((c) => c.scenes.map((scene) => ({ chapterId: c.id, chapterTitle: c.title, scene })))
}

/** The scene before or after this one in reading order (null at the ends). */
export function neighbourScene(outline: NovelOutline, sceneId: string, step: -1 | 1): NovelScene | null {
  const all = flatScenes(outline)
  const i = all.findIndex((s) => s.scene.id === sceneId)
  return i < 0 ? null : (all[i + step]?.scene ?? null)
}

export function addChapterOps(title: string): { ops: OutlineOp[]; chapterId: string; sceneId: string } {
  const chapterId = newId('c')
  const sceneId = newId('s')
  return { ops: [{ op: 'addChapter', chapterId, sceneId, title: title.trim() || 'Untitled chapter', index: null }], chapterId, sceneId }
}

export function addSceneOps(chapterId: string, title: string): { ops: OutlineOp[]; sceneId: string } {
  const sceneId = newId('s')
  return { ops: [{ op: 'addScene', chapterId, sceneId, title: title.trim() || 'Untitled scene', index: null }], sceneId }
}

export function renameSceneOps(sceneId: string, title: string): OutlineOp[] {
  return [{ op: 'setScene', sceneId, field: 'title', value: title.trim() || 'Untitled scene' }]
}

export function renameChapterOps(chapterId: string, title: string): OutlineOp[] {
  return [{ op: 'setChapter', chapterId, field: 'title', value: title.trim() || 'Untitled chapter' }]
}

export function setStatusOps(sceneId: string, status: SceneStatus): OutlineOp[] {
  return [{ op: 'setScene', sceneId, field: 'status', value: status }]
}

export function setBeatsOps(sceneId: string, beats: string): OutlineOp[] {
  return [{ op: 'setScene', sceneId, field: 'beats', value: beats }]
}

/** Moves a scene one place up or down inside its chapter; null at the ends. */
export function moveSceneOps(outline: NovelOutline, sceneId: string, step: -1 | 1): OutlineOp[] | null {
  const found = findScene(outline, sceneId)
  if (!found) return null
  const index = found.sceneIndex + step
  if (index < 0 || index >= found.chapter.scenes.length) return null
  return [{ op: 'moveScene', sceneId, toChapterId: found.chapter.id, index }]
}

export function moveChapterOps(outline: NovelOutline, chapterId: string, step: -1 | 1): OutlineOp[] | null {
  const found = findChapter(outline, chapterId)
  if (!found) return null
  const index = found.chapterIndex + step
  if (index < 0 || index >= outline.chapters.length) return null
  return [{ op: 'moveChapter', chapterId, index }]
}

/** The ops for a scene's sticky list after a prose result adds entries; null when nothing is new. */
export function stickyOps(outline: NovelOutline, sceneId: string, added: string[]): OutlineOp[] | null {
  const found = findScene(outline, sceneId)
  if (!found) return null
  const sticky = mergeSticky(found.scene.sticky, added)
  return sticky.length === found.scene.sticky.length ? null : [{ op: 'setScene', sceneId, field: 'sticky', value: sticky }]
}

/** An id, then a name or alias ignoring case: what the model wrote for a POV or "present" to a bible entry. */
export function resolveEntryRef(entries: BibleEntry[], ref: string): string | null {
  const wanted = ref.trim()
  if (!wanted) return null
  const byId = entries.find((e) => e.id === wanted)
  if (byId) return byId.id
  const lower = wanted.toLowerCase()
  const byName = entries.find((e) => [e.name, ...e.aliases].some((v) => v.trim().toLowerCase() === lower))
  return byName ? byName.id : null
}

export interface ResolvedRefs {
  pov: string | null
  present: string[]
}

export function resolveSceneRefs(bible: BibleEntry[], scenes: { pov: string | null; present: string[] }[]): { scenes: ResolvedRefs[]; unresolved: string[] } {
  const unresolved: string[] = []
  const miss = (ref: string): void => {
    if (ref.trim() && !unresolved.includes(ref.trim())) unresolved.push(ref.trim())
  }
  const out = scenes.map((sc) => {
    let pov: string | null = null
    if (sc.pov) {
      pov = resolveEntryRef(bible, sc.pov)
      if (!pov) miss(sc.pov)
    }
    const present: string[] = []
    for (const ref of sc.present) {
      const id = resolveEntryRef(bible, ref)
      if (!id) miss(ref)
      else if (!present.includes(id)) present.push(id)
    }
    return { pov, present }
  })
  return { scenes: out, unresolved }
}

/**
 * The ops that accept a summary or an outline proposal (the same ops the desktop's Accept sends); an empty list
 * for other results. Ids for new chapters and scenes are made here.
 */
export function proposalOps(result: WritingResult, bible: BibleEntry[]): OutlineOp[] {
  const ops: OutlineOp[] = []
  if (result.kind === 'summary') {
    const value = result.reply.summary.trim()
    if (result.target === 'scene') {
      ops.push({ op: 'setScene', sceneId: result.targetId, field: 'summary', value })
      ops.push({ op: 'setScene', sceneId: result.targetId, field: 'summaryOf', value: result.summaryOf })
    } else {
      ops.push({ op: 'setChapter', chapterId: result.targetId, field: 'summary', value })
      ops.push({ op: 'setChapter', chapterId: result.targetId, field: 'summaryOf', value: result.summaryOf })
    }
  } else if (result.kind === 'chapters') {
    for (const c of result.reply.chapters) {
      const chapterId = newId('c')
      ops.push({ op: 'addChapter', chapterId, sceneId: newId('s'), title: c.title.trim() || 'Untitled chapter', index: null })
      if (c.act.trim()) ops.push({ op: 'setChapter', chapterId, field: 'act', value: c.act.trim() })
      if (c.summary.trim()) ops.push({ op: 'setChapter', chapterId, field: 'summary', value: c.summary.trim() })
    }
  } else if (result.kind === 'scenes') {
    const refs = resolveSceneRefs(bible, result.reply.scenes).scenes
    for (const [i, sc] of result.reply.scenes.entries()) {
      const sceneId = newId('s')
      ops.push({ op: 'addScene', chapterId: result.chapterId, sceneId, title: sc.title.trim() || 'Untitled scene', index: null, beats: sc.beats })
      if (refs[i].pov) ops.push({ op: 'setScene', sceneId, field: 'pov', value: refs[i].pov })
      if (refs[i].present.length) ops.push({ op: 'setScene', sceneId, field: 'present', value: refs[i].present })
      ops.push({ op: 'setScene', sceneId, field: 'status', value: sc.beats.trim() ? 'beats' : 'idea' })
    }
  }
  return ops
}

/** The scene file path for a scene, as the computer stores it. */
export function scenePath(scene: Pick<NovelScene, 'file'>): string {
  return `manuscript/${scene.file}`
}

export function entryPath(entryId: string): string {
  return `bible/${entryId}.md`
}
