import { toast } from 'sonner'
import type { NovelAction } from '@shared/novel'
import type { WritingJob } from '@shared/phone'
import { isDirty } from '../lib/autosave'
import { errorMessage } from '../lib/format'
import { link } from '../lib/remote'
import { cardRequest, proseRequest, type ActionDef, type EditorSnapshot } from '../lib/runs'
import { scenePath } from '../lib/outlineOps'
import { findScene } from '@shared/novelOutline'
import { useScene } from './editor'
import { useJobs } from './jobs'
import { useProject } from './project'
import { useUi } from './ui'

// Starting writing jobs from the screens. Each one builds the request from the open project and the editor.

/** The scene's text and base hash: the open buffer for the scene being written, else the file as the computer has it. */
async function sceneText(sceneId: string): Promise<{ text: string; baseHash: string } | null> {
  const project = useProject.getState().state
  const found = project && findScene(project.outline, sceneId)
  if (!project || !found) return null
  const b = useScene.getState().buffer
  const path = scenePath(found.scene)
  if (b && b.path === path) {
    await useScene.getState().saveNow()
    const now = useScene.getState().buffer
    return now ? { text: now.text, baseHash: now.baseHash } : null
  }
  try {
    const file = await link.invoke('novel:readFile', project.folder, path)
    return { text: file.text, baseHash: file.hash }
  } catch (e) {
    toast.error(errorMessage(e))
    return null
  }
}

async function start(req: ReturnType<typeof cardRequest>, openSheet: boolean): Promise<WritingJob | null> {
  const job = await useJobs.getState().run(req)
  if (job && openSheet) useUi.getState().setJobsOpen(true)
  return job
}

export async function startProse(def: ActionDef, snap: EditorSnapshot, note: string, alternatives: number): Promise<WritingJob | null> {
  const project = useProject.getState().state
  const sceneId = useUi.getState().sceneId
  const b = useScene.getState().buffer
  if (!project || !sceneId || !b) return null
  if (b.readOnly) {
    toast.error('This scene is read-only, so nothing can be written into it.')
    return null
  }
  const job = await useJobs.getState().run(proseRequest(project.folder, sceneId, def, snap, note, alternatives))
  if (job) useUi.getState().setJobsOpen(true)
  return job
}

export async function startSummarizeScene(sceneId: string): Promise<void> {
  const project = useProject.getState().state
  const snap = await sceneText(sceneId)
  if (!project || !snap) return
  if (!snap.text.trim()) return void toast.error('This scene has no text to summarize yet.')
  await start(cardRequest(project.folder, 'summarizeScene', snap, { sceneId }), true)
}

export async function startSummarizeChapter(chapterId: string): Promise<void> {
  const project = useProject.getState().state
  if (!project) return
  await start(cardRequest(project.folder, 'summarizeChapter', null, { chapterId }), true)
}

export async function startPlanChapters(count: number, note: string): Promise<void> {
  const project = useProject.getState().state
  if (!project) return
  await start(cardRequest(project.folder, 'outlineChapters', null, { count, note: note.trim() }), true)
}

export async function startPlanScenes(chapterId: string, count: number, note: string): Promise<void> {
  const project = useProject.getState().state
  if (!project) return
  await start(cardRequest(project.folder, 'outlineScenes', null, { chapterId, count, note: note.trim() }), true)
}

/** Check and Update state read the scene as the computer has it, so unsaved text is saved first. */
async function savedSceneRun(action: NovelAction): Promise<WritingJob | null> {
  const project = useProject.getState().state
  const sceneId = useUi.getState().sceneId
  if (!project || !sceneId) return null
  await useScene.getState().saveNow()
  const b = useScene.getState().buffer
  if (!b) return null
  const found = findScene(project.outline, sceneId)
  if (!found || b.folder !== project.folder || b.path !== scenePath(found.scene)) {
    toast.error('The scene is still opening. Try again in a moment.')
    return null
  }
  if (isDirty(b)) {
    toast.error('Save the scene first. Look at the message above the text.')
    return null
  }
  return useJobs.getState().run(cardRequest(project.folder, action, { text: b.text, baseHash: b.baseHash }, { sceneId }))
}

export const startCheck = (): Promise<WritingJob | null> => savedSceneRun('checkScene')
export const startUpdateState = (): Promise<WritingJob | null> => savedSceneRun('updateState')
