// The novel:* handlers a phone may call: the same store calls as the window's (novel/ipc.ts), with the phone
// connection's owner id in front. No Electron imports, so tests can use it.
import type { NovelStore } from '../novel/store'
import { runAs } from '../novel/store'
import type { PhoneHandlerTable } from './registry'

/**
 * The same store calls for a phone, one table entry per channel in PHONE_NOVEL_CHANNELS. The first argument is the
 * phone connection's owner id ('phone:<clientId>'). The phone-only restrictions (openProject, saveFile) are in the
 * dispatch layer (phone/registry.ts), not here.
 */
export function novelPhoneHandlers(store: NovelStore): PhoneHandlerTable {
  return {
    'novel:listProjects': () => store.listProjects(),
    'novel:createProject': (owner, title) => runAs(owner, () => store.createProject(title, owner)),
    'novel:openProject': (owner, folder) => runAs(owner, () => store.openProject(folder, owner)),
    'novel:closeProject': (owner, folder) => runAs(owner, () => store.closeProject(folder, owner)),
    'novel:readFile': (_owner, folder, path) => store.readFile(folder, path),
    'novel:saveFile': (owner, req) => runAs(owner, () => store.saveFile(req)),
    'novel:outlinePatch': (owner, folder, ops) => runAs(owner, () => store.outlinePatch(folder, ops)),
    'novel:projectPatch': (owner, folder, patch) => runAs(owner, () => store.projectPatch(folder, patch)),
    'novel:createEntry': (owner, folder, name, kind) => runAs(owner, () => store.createEntry(folder, name, kind)),
    'novel:deleteScene': (owner, folder, sceneId) => runAs(owner, () => store.deleteScene(folder, sceneId)),
    'novel:deleteChapter': (owner, folder, chapterId) => runAs(owner, () => store.deleteChapter(folder, chapterId)),
    'novel:deleteEntry': (owner, folder, entryId) => runAs(owner, () => store.deleteEntry(folder, entryId)),
    'novel:history': (_owner, folder, path) => store.history(folder, path),
    'novel:readSnapshot': (_owner, folder, snapshot) => store.readSnapshot(folder, snapshot),
    'novel:trash': (_owner, folder) => store.trash(folder),
    'novel:restoreTrash': (owner, folder, id) => runAs(owner, () => store.restoreTrash(folder, id)),
    'novel:statePatch': (owner, folder, ops) => runAs(owner, () => store.statePatch(folder, ops)),
    'novel:dismissFinding': (owner, folder, sceneId, key, dismissed) => runAs(owner, () => store.dismissFinding(folder, sceneId, key, dismissed))
  }
}
