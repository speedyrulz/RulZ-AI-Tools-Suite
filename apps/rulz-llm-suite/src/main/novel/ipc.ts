// The novel:* IPC channels (contract: src/shared/novel.ts). Handlers only forward to the store.
import { handle } from '../ipc'
import type { NovelStore } from './store'

/** `onFlushDone` is called when the renderer answers the close-time 'novel:flush' event. */
export function registerNovelIpc(store: NovelStore, onFlushDone: () => void): void {
  handle('novel:listProjects', () => store.listProjects())
  handle('novel:root', () => store.root())
  handle('novel:createProject', (title) => store.createProject(title))
  handle('novel:openProject', (folder) => store.openProject(folder))
  handle('novel:closeProject', (folder) => store.closeProject(folder))
  handle('novel:readFile', (folder, path) => store.readFile(folder, path))
  handle('novel:saveFile', (req) => store.saveFile(req))
  handle('novel:journal', (folder, path, text, baseHash) => store.journal(folder, path, text, baseHash))
  handle('novel:discardJournal', (folder, path) => store.discardJournal(folder, path))
  handle('novel:outlinePatch', (folder, ops) => store.outlinePatch(folder, ops))
  handle('novel:projectPatch', (folder, patch) => store.projectPatch(folder, patch))
  handle('novel:createEntry', (folder, name, kind) => store.createEntry(folder, name, kind))
  handle('novel:deleteScene', (folder, sceneId) => store.deleteScene(folder, sceneId))
  handle('novel:deleteChapter', (folder, chapterId) => store.deleteChapter(folder, chapterId))
  handle('novel:deleteEntry', (folder, entryId) => store.deleteEntry(folder, entryId))
  handle('novel:history', (folder, path) => store.history(folder, path))
  handle('novel:readSnapshot', (folder, snapshot) => store.readSnapshot(folder, snapshot))
  handle('novel:trash', (folder) => store.trash(folder))
  handle('novel:restoreTrash', (folder, id) => store.restoreTrash(folder, id))
  handle('novel:emptyTrash', (folder) => store.emptyTrash(folder))
  handle('novel:deleteProject', (folder) => store.deleteProject(folder))
  handle('novel:flushDone', onFlushDone)
}
