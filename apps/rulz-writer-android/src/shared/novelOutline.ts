// Outline logic for the Writing mode: ids, file names and the ops that change outline.json. Pure, so the
// main process applies them and the tests can run them without files (data safety rule 7, 10).
import { NOVEL_FORMAT, type NovelChapter, type NovelOutline, type NovelScene, type OutlineOp, type SceneStatus } from './novel'

const ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz'

/** A short random id such as "s-k3f9x2". */
export function newId(prefix: 'p' | 'c' | 's'): string {
  const bytes = new Uint8Array(6)
  globalThis.crypto.getRandomValues(bytes)
  let out = ''
  for (const b of bytes) out += ALPHABET[b % ALPHABET.length]
  return `${prefix}-${out}`
}

const RESERVED = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/

/** Windows-safe lowercase ASCII slug: letters and digits joined by "-", never a reserved device name. */
export function slugify(title: string, maxLen = 40): string {
  let s = title
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  if (s.length > maxLen) s = s.slice(0, maxLen).replace(/-+$/, '')
  if (!s) s = 'untitled'
  if (RESERVED.test(s)) s = `${s}-1`
  return s
}

/**
 * "01-night-crossing-k3f9.md": position at creation (`index` is 0-based, shown 1-based), the title's slug
 * (at most 40 characters), the last 4 characters of the id; a "-2", "-3" ... is added when the name is taken.
 * Names are compared case-insensitively, as Windows does.
 */
export function sceneFileName(index: number, title: string, id: string, existing: string[]): string {
  const taken = new Set(existing.map((f) => f.toLowerCase()))
  const tail = id.toLowerCase().replace(/[^a-z0-9]/g, '').slice(-4) || 'x'
  const base = `${String(Math.max(0, Math.floor(index)) + 1).padStart(2, '0')}-${slugify(title, 40)}-${tail}`
  let name = `${base}.md`
  for (let n = 2; taken.has(name.toLowerCase()); n++) name = `${base}-${n}.md`
  return name
}

/** A scene file name: one plain *.md file name, nothing that can leave manuscript\ and no dotfile. */
export function validSceneFile(file: unknown): file is string {
  return typeof file === 'string' && file !== '' && !/[\\/:*?"<>|\0]/.test(file) && !file.startsWith('.') && /\.md$/i.test(file)
}

export function emptyOutline(): NovelOutline {
  return { format: NOVEL_FORMAT, chapters: [] }
}

export function findScene(
  outline: NovelOutline,
  sceneId: string
): { chapter: NovelChapter; scene: NovelScene; chapterIndex: number; sceneIndex: number } | null {
  for (let chapterIndex = 0; chapterIndex < outline.chapters.length; chapterIndex++) {
    const chapter = outline.chapters[chapterIndex]
    const sceneIndex = chapter.scenes.findIndex((s) => s.id === sceneId)
    if (sceneIndex >= 0) return { chapter, scene: chapter.scenes[sceneIndex], chapterIndex, sceneIndex }
  }
  return null
}

export function findChapter(outline: NovelOutline, chapterId: string): { chapter: NovelChapter; chapterIndex: number } | null {
  const chapterIndex = outline.chapters.findIndex((c) => c.id === chapterId)
  return chapterIndex >= 0 ? { chapter: outline.chapters[chapterIndex], chapterIndex } : null
}

const STATUSES: SceneStatus[] = ['idea', 'beats', 'draft', 'revised']

function blankScene(id: string, title: string, file: string, beats = ''): NovelScene {
  return { id, title, file, beats, summary: '', summaryOf: '', status: beats ? 'beats' : 'idea', pov: null, present: [], sticky: [], targetWords: null }
}

function clampIndex(index: number | null, length: number): number {
  if (index === null || !Number.isFinite(index)) return length
  return Math.min(Math.max(0, Math.floor(index)), length)
}

function str(value: unknown, what: string): string {
  if (typeof value !== 'string') throw new Error(`${what} must be text.`)
  return value
}

function strList(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || value.some((v) => typeof v !== 'string')) throw new Error(`${what} must be a list of ids.`)
  return [...new Set(value as string[])]
}

function allSceneFiles(outline: NovelOutline): string[] {
  return outline.chapters.flatMap((c) => c.scenes.map((s) => s.file))
}

function sceneCount(outline: NovelOutline): number {
  return outline.chapters.reduce((n, c) => n + c.scenes.length, 0)
}

function idTaken(outline: NovelOutline, id: string): boolean {
  return outline.chapters.some((c) => c.id === id || c.scenes.some((s) => s.id === id))
}

function setSceneField(scene: NovelScene, field: string, value: unknown): void {
  switch (field) {
    case 'title': {
      const v = str(value, 'The scene title').trim()
      if (!v) throw new Error('The scene title cannot be empty.')
      scene.title = v
      break
    }
    case 'beats':
      scene.beats = str(value, 'The beats')
      break
    case 'summary':
      scene.summary = str(value, 'The summary')
      break
    case 'summaryOf':
      scene.summaryOf = str(value, 'summaryOf')
      break
    case 'status':
      if (typeof value !== 'string' || !STATUSES.includes(value as SceneStatus)) throw new Error(`"${String(value)}" is not a scene status.`)
      scene.status = value as SceneStatus
      break
    case 'pov':
      if (value !== null && typeof value !== 'string') throw new Error('The POV must be an entry id or empty.')
      scene.pov = value === '' ? null : (value as string | null)
      break
    case 'present':
      scene.present = strList(value, 'Present characters')
      break
    case 'sticky':
      scene.sticky = strList(value, 'Sticky entries')
      break
    case 'targetWords':
      if (value !== null && (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)) throw new Error('The target length must be a positive number or empty.')
      scene.targetWords = value === null ? null : Math.round(value as number)
      break
    default:
      throw new Error(`"${field}" is not a scene field.`)
  }
}

function setChapterField(chapter: NovelChapter, field: string, value: unknown): void {
  switch (field) {
    case 'title': {
      const v = str(value, 'The chapter title').trim()
      if (!v) throw new Error('The chapter title cannot be empty.')
      chapter.title = v
      break
    }
    case 'act':
      chapter.act = str(value, 'The act label')
      break
    case 'summary':
      chapter.summary = str(value, 'The summary')
      break
    case 'summaryOf':
      chapter.summaryOf = str(value, 'summaryOf')
      break
    default:
      throw new Error(`"${field}" is not a chapter field.`)
  }
}

/**
 * Applies `ops` in order to a copy of the outline. `existingFiles` are the file names in manuscript\ (used
 * to keep new scene names unique and to check adoptFile). Returns the new outline and the names of the empty
 * scene files the caller must create. Throws Error with a plain message on a bad op; the input is untouched.
 */
export function applyOutlineOps(outline: NovelOutline, ops: OutlineOp[], existingFiles: string[]): { outline: NovelOutline; createdFiles: string[] } {
  const out: NovelOutline = structuredClone(outline)
  const createdFiles: string[] = []
  const usedNames = (): string[] => [...existingFiles, ...createdFiles, ...allSceneFiles(out)]
  const needScene = (id: string): NonNullable<ReturnType<typeof findScene>> => {
    const f = findScene(out, id)
    if (!f) throw new Error(`There is no scene with id "${id}".`)
    return f
  }
  const needChapter = (id: string): NonNullable<ReturnType<typeof findChapter>> => {
    const f = findChapter(out, id)
    if (!f) throw new Error(`There is no chapter with id "${id}".`)
    return f
  }
  const newScene = (sceneId: string, title: string, beats?: string): NovelScene => {
    if (idTaken(out, sceneId)) throw new Error(`The id "${sceneId}" is already used.`)
    const t = title.trim() || 'Untitled scene'
    const file = sceneFileName(sceneCount(out), t, sceneId, usedNames())
    createdFiles.push(file)
    return blankScene(sceneId, t, file, beats)
  }

  for (const op of ops) {
    switch (op.op) {
      case 'setScene':
        setSceneField(needScene(op.sceneId).scene, op.field, op.value)
        break
      case 'setChapter':
        setChapterField(needChapter(op.chapterId).chapter, op.field, op.value)
        break
      case 'addChapter': {
        if (idTaken(out, op.chapterId)) throw new Error(`The id "${op.chapterId}" is already used.`)
        const title = str(op.title, 'The chapter title').trim() || 'Untitled chapter'
        const scene = newScene(op.sceneId, 'Scene 1')
        const chapter: NovelChapter = { id: op.chapterId, title, act: '', summary: '', summaryOf: '', scenes: [scene] }
        out.chapters.splice(clampIndex(op.index, out.chapters.length), 0, chapter)
        break
      }
      case 'addScene': {
        const { chapter } = needChapter(op.chapterId)
        const scene = newScene(op.sceneId, str(op.title, 'The scene title'), op.beats)
        chapter.scenes.splice(clampIndex(op.index, chapter.scenes.length), 0, scene)
        break
      }
      case 'moveChapter': {
        const { chapterIndex } = needChapter(op.chapterId)
        const [chapter] = out.chapters.splice(chapterIndex, 1)
        out.chapters.splice(clampIndex(op.index, out.chapters.length), 0, chapter)
        break
      }
      case 'moveScene': {
        const from = needScene(op.sceneId)
        const to = needChapter(op.toChapterId)
        if (from.chapter !== to.chapter && from.chapter.scenes.length === 1) {
          throw new Error('A chapter needs at least one scene, so its last scene cannot be moved out.')
        }
        from.chapter.scenes.splice(from.sceneIndex, 1)
        to.chapter.scenes.splice(clampIndex(op.index, to.chapter.scenes.length), 0, from.scene)
        break
      }
      case 'adoptFile': {
        const asked = str(op.file, 'The file name')
        // The name as it is on disk, so a different spelling of the case cannot break the link later.
        const file = existingFiles.find((f) => f.toLowerCase() === asked.toLowerCase())
        if (!file || !validSceneFile(file)) throw new Error(`"${asked}" is not a Markdown file in the manuscript folder.`)
        if (allSceneFiles(out).some((f) => f.toLowerCase() === file.toLowerCase())) throw new Error(`"${file}" is already a scene in the outline.`)
        if (idTaken(out, op.sceneId)) throw new Error(`The id "${op.sceneId}" is already used.`)
        const { chapter } = needChapter(op.chapterId)
        const title = file.replace(/\.md$/i, '').replace(/^\d+[-_ ]*/, '').replace(/[-_]+/g, ' ').trim() || 'Untitled scene'
        const scene = blankScene(op.sceneId, title, file)
        scene.status = 'draft'
        chapter.scenes.splice(clampIndex(op.index, chapter.scenes.length), 0, scene)
        break
      }
      default: {
        const unknown = (op as { op?: unknown }).op
        throw new Error(`"${String(unknown)}" is not an outline operation.`)
      }
    }
  }
  return { outline: out, createdFiles }
}

/** Removes a scene (the last scene of a chapter is refused: delete the chapter). Used by the trash. */
export function removeScene(outline: NovelOutline, sceneId: string): { outline: NovelOutline; chapterId: string; index: number; scene: NovelScene } {
  const out: NovelOutline = structuredClone(outline)
  const f = findScene(out, sceneId)
  if (!f) throw new Error(`There is no scene with id "${sceneId}".`)
  if (f.chapter.scenes.length === 1) throw new Error('A chapter needs at least one scene. Delete the chapter instead.')
  f.chapter.scenes.splice(f.sceneIndex, 1)
  return { outline: out, chapterId: f.chapter.id, index: f.sceneIndex, scene: f.scene }
}

export function removeChapter(outline: NovelOutline, chapterId: string): { outline: NovelOutline; index: number; chapter: NovelChapter } {
  const out: NovelOutline = structuredClone(outline)
  const f = findChapter(out, chapterId)
  if (!f) throw new Error(`There is no chapter with id "${chapterId}".`)
  out.chapters.splice(f.chapterIndex, 1)
  return { outline: out, index: f.chapterIndex, chapter: f.chapter }
}

/** Puts a trashed scene or chapter back at its old position (clamped). */
export function reinsert(
  outline: NovelOutline,
  what: { kind: 'scene'; chapterId: string; index: number; scene: NovelScene } | { kind: 'chapter'; index: number; chapter: NovelChapter }
): NovelOutline {
  const out: NovelOutline = structuredClone(outline)
  if (what.kind === 'scene') {
    const chapter = findChapter(out, what.chapterId)
    if (!chapter) throw new Error('The chapter this scene belonged to no longer exists.')
    if (idTaken(out, what.scene.id)) throw new Error('A scene with the same id is already in the outline.')
    chapter.chapter.scenes.splice(clampIndex(what.index, chapter.chapter.scenes.length), 0, what.scene)
  } else {
    const ids = [what.chapter.id, ...what.chapter.scenes.map((s) => s.id)]
    if (ids.some((id) => idTaken(out, id))) throw new Error('A chapter or scene with the same id is already in the outline.')
    out.chapters.splice(clampIndex(what.index, out.chapters.length), 0, what.chapter)
  }
  return out
}
