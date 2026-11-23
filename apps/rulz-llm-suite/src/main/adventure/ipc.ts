// The adventure:* channels, for the window and (the PHONE_ADVENTURE_CHANNELS subset) for phones. The contract is
// src/shared/adventure.ts. Like writing/ipc.ts, a handler gets the caller's owner id first ('window' or
// 'phone:<clientId>'), then the channel's arguments; the arguments are checked here, a phone is not trusted to send
// the right shapes. No Electron imports: main/index.ts passes in the save dialog and the file writer.
import { isAbsolute, resolve } from 'node:path'
import type {
  AdventureDesktopInvoke,
  AdventureEdit,
  AdventureIpcInvoke,
  AdventureSetup,
  AdventureWorld,
  PlayRequest,
  RetryRequest
} from '@shared/adventure'
import { ADVENTURE_STARTERS, exportFileName, exportMarkdown, worldDigest } from '@shared/adventurePrompts'
import type { NovelProjectState } from '@shared/novel'
import type { AdventureRuns } from './runs'
import { checkAdventureId, type AdventureStore } from './store'

export type AdventureHandler = (ownerId: string, ...args: unknown[]) => Promise<unknown>

const str = (v: unknown, what: string): string => {
  if (typeof v !== 'string' || !v) throw new Error(`${what} is missing.`)
  return v
}

// ---------------------------------------------------------------------------------------------
// The world of a Writing project
// ---------------------------------------------------------------------------------------------

/** The slice of NovelStore that reading a project's world uses. */
export interface WorldNovel {
  listProjects(): Promise<{ folder: string; title: string }[]>
  openProject(folder: string, owner: string): Promise<NovelProjectState>
  closeProject(folder: string, owner: string): Promise<void>
}

const normalise = (folder: string): string => {
  const full = resolve(folder)
  return process.platform === 'win32' ? full.toLowerCase() : full
}

/**
 * Copies a Writing project's story bible (and its chapter summaries) into plain text. The folder must be one of the
 * projects novel:listProjects returns; the project is opened for a moment under `owner` and closed again.
 */
export async function readWorld(novel: WorldNovel, folder: unknown, owner: string): Promise<AdventureWorld> {
  if (typeof folder !== 'string' || !isAbsolute(folder)) throw new Error('The project folder must be a full path.')
  const listed = (await novel.listProjects()).find((p) => normalise(p.folder) === normalise(folder))
  if (!listed) throw new Error('That Writing project was not found in the projects folder.')
  const st = await novel.openProject(listed.folder, owner)
  try {
    if (st.blocked) throw new Error(`That Writing project cannot be read: ${st.blocked}`)
    const text = worldDigest({
      title: st.project.title || listed.title,
      chapters: st.outline.chapters.map((c) => ({ title: c.title, summary: c.summary })),
      bible: st.bible.map((e) => ({ name: e.name, kind: e.kind, body: e.body }))
    })
    return { folder: listed.folder, title: st.project.title || listed.title, text }
  } finally {
    await novel.closeProject(listed.folder, owner).catch(() => undefined)
  }
}

// ---------------------------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------------------------

export interface AdventureHandlerDeps {
  store: AdventureStore
  runs: AdventureRuns
  novel: WorldNovel
}

/** One handler per channel a phone may call (and the window too). */
export function adventureHandlers(d: AdventureHandlerDeps): Record<keyof AdventureIpcInvoke, AdventureHandler> {
  const { store, runs } = d
  return {
    'adventure:list': async () => store.list(),
    'adventure:get': async (_o, id) => store.get(checkAdventureId(id)),
    'adventure:create': async (_o, setup, worldFolder) => {
      if (!setup || typeof setup !== 'object') throw new Error('The setup is missing.')
      // The id comes first: the project is opened under `adventure:<id>` while its world is copied.
      const id = store.makeId()
      let world: AdventureWorld | undefined
      if (worldFolder !== undefined && worldFolder !== null) world = await readWorld(d.novel, worldFolder, `adventure:${id}`)
      // A world sent inside the setup is ignored: only main copies one in.
      const { world: _ignored, ...rest } = setup as AdventureSetup
      return store.create(rest, world, id)
    },
    'adventure:updateSetup': async (_o, id, patch) => {
      if (!patch || typeof patch !== 'object') throw new Error('The settings are missing.')
      return store.updateSetup(checkAdventureId(id), patch as Partial<AdventureSetup>)
    },
    'adventure:setCurrent': async (_o, id, nodeId) => store.setCurrent(checkAdventureId(id), str(nodeId, 'The passage')),
    'adventure:edit': async (_o, edit) => {
      const e = edit as Partial<AdventureEdit> | null
      if (!e || typeof e !== 'object') throw new Error('The edit is missing.')
      if (typeof e.rev !== 'number') throw new Error('The edit does not say which version of the passage it is based on.')
      return store.edit({ ...(e as AdventureEdit), adventureId: checkAdventureId(e.adventureId), nodeId: str(e.nodeId, 'The passage') })
    },
    'adventure:exportMarkdown': async (_o, id, nodeId) => {
      const adv = await store.get(checkAdventureId(id))
      const target = typeof nodeId === 'string' && nodeId ? nodeId : adv.currentId
      if (target && !Object.hasOwn(adv.nodes, target)) throw new Error('That passage was not found.')
      return exportMarkdown(adv, target)
    },
    'adventure:play': async (_o, req) => runs.play(req as PlayRequest),
    'adventure:retry': async (_o, req) => runs.retry(req as RetryRequest),
    'adventure:jobs': async (_o, id) => runs.list(checkAdventureId(id)),
    'adventure:abort': async (_o, jobId) => void runs.abort(str(jobId, 'The job id')),
    'adventure:dismiss': async (_o, jobId) => void runs.dismiss(str(jobId, 'The job id')),
    'adventure:starters': async () => ADVENTURE_STARTERS,
    'adventure:models': async () => runs.models()
  }
}

export interface AdventureDesktopDeps extends AdventureHandlerDeps {
  /** Shows the save dialog; null when cancelled. */
  chooseSavePath: (defaultName: string) => Promise<string | null>
  writeText: (path: string, text: string) => Promise<void>
}

/** The channels only the window may call. */
export function adventureDesktopHandlers(d: AdventureDesktopDeps): Record<keyof AdventureDesktopInvoke, AdventureHandler> {
  const { store, runs } = d
  const all = adventureHandlers(d)
  return {
    'adventure:delete': async (_o, id) => {
      const key = checkAdventureId(id)
      if (runs.hasRunning(key)) throw new Error('This adventure is being written. Stop it first.')
      await store.delete(key)
    },
    'adventure:deleteBranch': async (_o, id, nodeId, rev) => {
      const key = checkAdventureId(id)
      if (typeof rev !== 'number') throw new Error('The request does not say which version of the passage it is based on.')
      return store.deleteBranch(key, str(nodeId, 'The passage'), rev, (removed) => runs.branchInUse(key, removed))
    },
    'adventure:saveExport': async (owner, id, nodeId) => {
      const key = checkAdventureId(id)
      const adv = await store.get(key)
      const text = (await all['adventure:exportMarkdown'](owner, key, nodeId)) as string
      const path = await d.chooseSavePath(exportFileName(adv.setup.title))
      if (!path) return null
      await d.writeText(path, text)
      return path
    }
  }
}
