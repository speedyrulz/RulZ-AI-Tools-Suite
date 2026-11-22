// The novel:* IPC channels (contract: src/shared/novel.ts). Handlers only forward to the store.
import { handle } from '../ipc'
import { runAs, WINDOW_OWNER, type NovelStore } from './store'

/** `onFlushDone` is called when the renderer answers the close-time 'novel:flush' event. */
export function registerNovelIpc(store: NovelStore, onFlushDone: () => void): void {
  // Everything the window asks is the window's request: the store counts it as the 'window' owner, and the
  // changes it causes are announced with that origin (the window already has them from the reply).
  const win = <T>(fn: () => T): T => runAs(WINDOW_OWNER, fn)
  handle('novel:listProjects', () => store.listProjects())
  handle('novel:root', () => store.root())
  handle('novel:createProject', (title) => win(() => store.createProject(title, WINDOW_OWNER)))
  handle('novel:openProject', (folder) => win(() => store.openProject(folder, WINDOW_OWNER)))
  handle('novel:closeProject', (folder) => win(() => store.closeProject(folder, WINDOW_OWNER)))
  handle('novel:readFile', (folder, path) => store.readFile(folder, path))
  handle('novel:saveFile', (req) => win(() => store.saveFile(req)))
  handle('novel:journal', (folder, path, text, baseHash) => store.journal(folder, path, text, baseHash))
  handle('novel:discardJournal', (folder, path) => store.discardJournal(folder, path))
  handle('novel:outlinePatch', (folder, ops) => win(() => store.outlinePatch(folder, ops)))
  handle('novel:projectPatch', (folder, patch) => win(() => store.projectPatch(folder, patch)))
  handle('novel:createEntry', (folder, name, kind) => win(() => store.createEntry(folder, name, kind)))
  handle('novel:deleteScene', (folder, sceneId) => win(() => store.deleteScene(folder, sceneId)))
  handle('novel:deleteChapter', (folder, chapterId) => win(() => store.deleteChapter(folder, chapterId)))
  handle('novel:deleteEntry', (folder, entryId) => win(() => store.deleteEntry(folder, entryId)))
  handle('novel:history', (folder, path) => store.history(folder, path))
  handle('novel:readSnapshot', (folder, snapshot) => store.readSnapshot(folder, snapshot))
  handle('novel:trash', (folder) => store.trash(folder))
  handle('novel:restoreTrash', (folder, id) => win(() => store.restoreTrash(folder, id)))
  handle('novel:emptyTrash', (folder) => store.emptyTrash(folder))
  handle('novel:deleteProject', (folder) => store.deleteProject(folder))
  handle('novel:statePatch', (folder, ops) => win(() => store.statePatch(folder, ops)))
  handle('novel:dismissFinding', (folder, sceneId, key, dismissed) => win(() => store.dismissFinding(folder, sceneId, key, dismissed)))
  handle('novel:record', (folder, line) => store.record(folder, line))
  handle('novel:recordsInfo', (folder) => store.recordsInfo(folder))
  handle('novel:exportRecords', (folder, outDir, opts) => store.exportRecords(folder, outDir, opts))
  handle('novel:deleteRecords', (folder) => store.deleteRecords(folder))
  handle('novel:flushDone', onFlushDone)
}
