// Adventure mode: the list, the adventure open in this window, this window's own position in it, the job
// running for it and the side panel. Generation runs in main (adventure:* channels); this store starts
// jobs, watches their events and patches the held adventure with applyChange.
import { create } from 'zustand'
import { toast } from 'sonner'
import {
  applyChange,
  childForChoice,
  metaOf,
  type Adventure,
  type AdventureChange,
  type AdventureEdit,
  type AdventureJob,
  type AdventureMeta,
  type AdventureSetup,
  type AdventureStarter,
  type NodeInput
} from '@shared/adventure'
import type { NovelProjectSummary } from '@shared/novel'
import { isConflict, isRunning } from '../lib/adventure'
import { errorMessage, invoke, on } from '../lib/api'
import { once } from '../lib/once'

export type PanelTab = 'state' | 'map' | 'settings'
export type SaveResult = 'ok' | 'conflict' | 'error'

interface AdventureStore {
  list: AdventureMeta[]
  listLoaded: boolean
  /** The adventure open in this window (null = the list). */
  adv: Adventure | null
  opening: boolean
  /** The passage this window is reading. Each window keeps its own; `adv.currentId` is only where to reopen. */
  position: string | null
  /** The newest job of the open adventure (running, or one that just ended). */
  job: AdventureJob | null
  /** A passage another device wrote: a note offers to jump there. */
  remoteNodeId: string | null
  panelOpen: boolean
  panelTab: PanelTab
  init: () => Promise<void>
  loadList: () => Promise<void>
  open: (id: string) => Promise<void>
  close: () => void
  /** Creates an adventure and writes its opening. */
  startNew: (setup: AdventureSetup, worldFolder: string | null) => Promise<boolean>
  rename: (id: string, title: string) => Promise<void>
  remove: (id: string) => Promise<void>
  /** Moves this window's position and saves it as the reopen position. */
  go: (nodeId: string) => void
  goRemote: () => void
  play: (input: NodeInput) => Promise<void>
  choose: (choiceId: string, text: string, forceNew?: boolean) => Promise<void>
  retry: (nodeId: string, what: 'passage' | 'choices') => Promise<void>
  edit: (edit: AdventureEdit) => Promise<SaveResult>
  deleteBranch: (nodeId: string) => Promise<SaveResult>
  updateSetup: (patch: Partial<Omit<AdventureSetup, 'world' | 'premise' | 'character'>>) => Promise<void>
  exportStory: () => Promise<void>
  abort: () => void
  refresh: () => Promise<void>
  setPanelOpen: (open: boolean) => void
  setPanelTab: (tab: PanelTab) => void
  /** A change from an event or a call's return. */
  noteChange: (ch: AdventureChange) => void
  /** A job event (or a job a call returned). */
  noteJob: (job: AdventureJob) => void
}

/** Jobs this window started; their results move this window's position, jobs from elsewhere do not. */
const ownJobs = new Set<string>()
/** Play/retry calls in flight. Their job may send events before the call returns, so the events of jobs not yet
 *  known are kept (latest per id); the id the call returns decides which of them is ours. */
let startingCalls = 0
const unknownJobs = new Map<string, AdventureJob>()
/** Events for the adventure being opened, kept while its get/jobs calls are out and replayed after. */
let openBuffer: { id: string; events: ({ job: AdventureJob } | { change: AdventureChange })[] } | null = null
/** Finished jobs already reported, and jobs whose result already moved the position. */
const reported = new Set<string>()
const moved = new Set<string>()
/** A passage to move to as soon as the change event that brings it arrives. */
let pendingPosition: string | null = null
/** Counts opens, so only the newest one's reply is used. */
let openSeq = 0

/** The list newest first, with `meta` added or replaced. */
export function upsertMeta(list: AdventureMeta[], meta: AdventureMeta): AdventureMeta[] {
  return [meta, ...list.filter((m) => m.id !== meta.id)].sort((a, b) => b.updatedAt - a.updatedAt)
}

/** Where a window opens an adventure: the saved reopen position when it still exists, else the opening. */
export function startPosition(adv: Adventure): string | null {
  if (adv.currentId && adv.nodes[adv.currentId]) return adv.currentId
  return adv.rootId && adv.nodes[adv.rootId] ? adv.rootId : null
}

/**
 * The held adventure with a change applied. A change older than the held adventure only brings nodes the
 * held copy has not seen or has at an older rev, so a late event never puts an old passage back.
 */
export function patchAdventure(adv: Adventure, ch: AdventureChange): Adventure {
  if (ch.id !== adv.id || ch.rev === null) return adv
  const stale = ch.rev < adv.rev
  // Node ids are never reused: a stale change must not add a node the held copy lacks (its branch may be
  // deleted since). It may update a held node with a higher rev. A removal is always right.
  const nodes = (ch.nodes ?? []).filter((n) => {
    const held = adv.nodes[n.id]
    return held ? n.rev >= held.rev : !stale
  })
  const next: AdventureChange = stale ? { id: ch.id, rev: ch.rev, nodes, removedIds: ch.removedIds } : { ...ch, nodes }
  return applyChange(adv, next)
}

/** The position after a change: unchanged while its node exists, else the saved reopen position or the opening. */
export function fixPosition(adv: Adventure, position: string | null): string | null {
  if (position && adv.nodes[position]) return position
  if (adv.currentId && adv.nodes[adv.currentId]) return adv.currentId
  return adv.rootId && adv.nodes[adv.rootId] ? adv.rootId : null
}

export const useAdventure = create<AdventureStore>((set, get) => {
  /** Runs a call that starts a job; the job it returns is ours. */
  const startJob = async (run: () => Promise<AdventureJob>): Promise<void> => {
    startingCalls++
    try {
      const job = await run()
      ownJobs.add(job.id)
      const latest = unknownJobs.get(job.id)
      if (latest) {
        // Events came in while the call returned and were handled as another device's: handle them as ours now.
        reported.delete(job.id)
        for (const k of [...moved]) if (k.startsWith(`${job.id}:`)) moved.delete(k)
        if (latest.resultNodeId && get().remoteNodeId === latest.resultNodeId) set({ remoteNodeId: null })
        get().noteJob(latest)
      } else if (get().job?.id !== job.id) {
        // Events may have moved the job on while the call was returning; an older copy must not undo that.
        get().noteJob(job)
      }
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      startingCalls--
      // Jobs seen meanwhile that the call did not name stay remote.
      if (startingCalls === 0) unknownJobs.clear()
    }
  }

  /** A refused edit or branch delete: a conflict loads the latest copy and says so. */
  const failedSave = (e: unknown): SaveResult => {
    const msg = errorMessage(e)
    if (isConflict(msg)) {
      toast.error('This passage was changed somewhere else. The latest version is loaded; check it and try again.')
      void get().refresh()
      return 'conflict'
    }
    toast.error(msg)
    return 'error'
  }

  const showOpened = (adv: Adventure, job: AdventureJob | null): void => {
    pendingPosition = null
    set((s) => ({
      adv,
      opening: false,
      position: startPosition(adv),
      job,
      remoteNodeId: null,
      list: upsertMeta(s.list, metaOf(adv))
    }))
  }

  return {
    list: [],
    listLoaded: false,
    adv: null,
    opening: false,
    position: null,
    job: null,
    remoteNodeId: null,
    panelOpen: true,
    panelTab: 'state',

    init: once(async () => {
      on('adventure:job', (job) => {
        if (openBuffer?.id === job.adventureId) openBuffer.events.push({ job })
        get().noteJob(job)
      })
      on('adventure:changed', (ch) => {
        if (openBuffer?.id === ch.id) openBuffer.events.push({ change: ch })
        get().noteChange(ch)
      })
    }),

    loadList: async () => {
      try {
        const list = await invoke('adventure:list')
        set({ list: [...list].sort((a, b) => b.updatedAt - a.updatedAt), listLoaded: true })
      } catch (e) {
        set({ listLoaded: true })
        toast.error(`Could not load the adventures: ${errorMessage(e)}`)
      }
    },

    open: async (id) => {
      const seq = ++openSeq
      const buffer = { id, events: [] as NonNullable<typeof openBuffer>['events'] }
      openBuffer = buffer
      set({ opening: true })
      try {
        const [adv, jobs] = await Promise.all([invoke('adventure:get', id), invoke('adventure:jobs', id).catch(() => [] as AdventureJob[])])
        if (seq !== openSeq) return
        // A job already running (started here before, or on another device) is picked up.
        const running = jobs.filter(isRunning).sort((a, b) => b.startedAt - a.startedAt)[0] ?? null
        showOpened(adv, running)
        if (running) get().noteJob(running)
        // What happened while the adventure was being read: patchAdventure skips what the snapshot already has.
        for (const ev of buffer.events) {
          if ('job' in ev) get().noteJob(ev.job)
          else get().noteChange(ev.change)
        }
      } catch (e) {
        if (seq === openSeq) set({ opening: false })
        toast.error(errorMessage(e))
      } finally {
        if (openBuffer === buffer) openBuffer = null
      }
    },

    close: () => {
      openSeq++
      pendingPosition = null
      set({ adv: null, position: null, job: null, remoteNodeId: null, opening: false })
      void get().loadList()
    },

    startNew: async (setup, worldFolder) => {
      try {
        const adv = await invoke('adventure:create', setup, worldFolder)
        openSeq++
        showOpened(adv, null)
        await get().play({ kind: 'start' })
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    rename: async (id, title) => {
      try {
        const ch = await invoke('adventure:updateSetup', id, { title })
        get().noteChange(ch)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    remove: async (id) => {
      try {
        await invoke('adventure:delete', id)
        if (get().adv?.id === id) {
          openSeq++
          set({ adv: null, position: null, job: null, remoteNodeId: null })
        }
        set((s) => ({ list: s.list.filter((m) => m.id !== id) }))
        toast.success('Moved to the Recycle Bin.')
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    go: (nodeId) => {
      const { adv } = get()
      if (!adv || !adv.nodes[nodeId]) return
      pendingPosition = null
      set((s) => ({ position: nodeId, remoteNodeId: s.remoteNodeId === nodeId ? null : s.remoteNodeId }))
      invoke('adventure:setCurrent', adv.id, nodeId)
        .then((ch) => get().noteChange(ch))
        .catch((e) => toast.error(errorMessage(e)))
    },

    goRemote: () => {
      const { remoteNodeId, adv } = get()
      if (remoteNodeId && adv?.nodes[remoteNodeId]) get().go(remoteNodeId)
      else set({ remoteNodeId: null })
    },

    play: async (input) => {
      const { adv, position, job } = get()
      if (!adv || isRunning(job)) return
      const parentId = input.kind === 'start' ? null : position
      await startJob(() => invoke('adventure:play', { adventureId: adv.id, parentId, input }))
    },

    choose: async (choiceId, text, forceNew = false) => {
      const { adv, position, job } = get()
      if (!adv || !position || isRunning(job)) return
      const taken = forceNew ? undefined : childForChoice(adv, position, choiceId)
      if (taken) {
        get().go(taken.id)
        return
      }
      await get().play({ kind: 'choice', choiceId, text })
    },

    retry: async (nodeId, what) => {
      const { adv, job } = get()
      if (!adv || isRunning(job)) return
      await startJob(() => invoke('adventure:retry', { adventureId: adv.id, nodeId, what }))
    },

    edit: async (edit) => {
      try {
        const ch = await invoke('adventure:edit', edit)
        get().noteChange(ch)
        return 'ok'
      } catch (e) {
        return failedSave(e)
      }
    },

    deleteBranch: async (nodeId) => {
      const { adv } = get()
      const node = adv?.nodes[nodeId]
      if (!adv || !node) return 'error'
      try {
        const ch = await invoke('adventure:deleteBranch', adv.id, nodeId, node.rev)
        get().noteChange(ch)
        return 'ok'
      } catch (e) {
        return failedSave(e)
      }
    },

    updateSetup: async (patch) => {
      const { adv } = get()
      if (!adv) return
      try {
        const ch = await invoke('adventure:updateSetup', adv.id, patch)
        get().noteChange(ch)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    exportStory: async () => {
      const { adv, position } = get()
      if (!adv) return
      try {
        const path = await invoke('adventure:saveExport', adv.id, position)
        if (path) toast.success(`Saved to ${path}`)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    abort: () => {
      const { job } = get()
      if (!job || !isRunning(job)) return
      invoke('adventure:abort', job.id).catch((e) => toast.error(errorMessage(e)))
    },

    refresh: async () => {
      const { adv } = get()
      if (!adv) return
      try {
        const fresh = await invoke('adventure:get', adv.id)
        if (get().adv?.id !== fresh.id) return
        set((s) => ({ adv: fresh, position: fixPosition(fresh, s.position), list: upsertMeta(s.list, metaOf(fresh)) }))
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    setPanelOpen: (panelOpen) => set({ panelOpen }),
    setPanelTab: (panelTab) => set({ panelTab, panelOpen: true }),

    noteChange: (ch) => {
      // The list row: changed, or gone.
      if (ch.rev === null) set((s) => ({ list: s.list.filter((m) => m.id !== ch.id) }))
      else if (ch.meta) set((s) => ({ list: upsertMeta(s.list, ch.meta as AdventureMeta) }))
      const { adv } = get()
      if (!adv || ch.id !== adv.id) return
      if (ch.rev === null) {
        openSeq++
        pendingPosition = null
        set({ adv: null, position: null, job: null, remoteNodeId: null })
        toast.message('This adventure was deleted on another device.')
        return
      }
      const next = patchAdventure(adv, ch)
      set((s) => {
        let position = fixPosition(next, s.position)
        if (pendingPosition && next.nodes[pendingPosition]) {
          position = pendingPosition
          pendingPosition = null
        }
        const remote = s.remoteNodeId && next.nodes[s.remoteNodeId] && s.remoteNodeId !== position ? s.remoteNodeId : null
        return { adv: next, position, remoteNodeId: remote }
      })
    },

    noteJob: (job) => {
      const { adv, job: held } = get()
      if (!adv || job.adventureId !== adv.id) return
      const own = ownJobs.has(job.id)
      if (!own && startingCalls > 0) unknownJobs.set(job.id, job)
      // An older job's late event does not replace a newer running one.
      if (held && held.id !== job.id && isRunning(held) && job.startedAt < held.startedAt) return
      set({ job })

      if (job.resultNodeId && job.kind !== 'retryChoices') {
        const key = `${job.id}:${job.resultNodeId}`
        if (!moved.has(key)) {
          moved.add(key)
          if (own) {
            if (get().adv?.nodes[job.resultNodeId]) {
              pendingPosition = null
              set({ position: job.resultNodeId, remoteNodeId: null })
            } else pendingPosition = job.resultNodeId
          } else if (job.resultNodeId !== get().position) set({ remoteNodeId: job.resultNodeId })
        }
      }

      if (!isRunning(job) && !reported.has(job.id)) {
        reported.add(job.id)
        if (!own) return
        if (job.status === 'error' && job.error) toast.error(job.error)
        else if (job.status === 'stopped') toast.message(job.resultNodeId ? 'Stopped. The passage was kept.' : 'Stopped.')
        for (const w of job.warnings) toast.warning(w)
        invoke('adventure:dismiss', job.id).catch(() => undefined)
        // The passage was saved but its change never arrived: fetch the adventure again.
        if (job.resultNodeId && !get().adv?.nodes[job.resultNodeId]) void get().refresh()
      }
    }
  }
})

/** Lists the pieces the New adventure dialog needs; plain calls, nothing is kept. */
export async function loadDialogData(): Promise<{ starters: AdventureStarter[]; models: { id: string; name: string; loaded: boolean }[]; projects: NovelProjectSummary[] }> {
  const [starters, models, projects] = await Promise.all([
    invoke('adventure:starters').catch(() => [] as AdventureStarter[]),
    invoke('adventure:models').catch(() => [] as { id: string; name: string; loaded: boolean }[]),
    invoke('novel:listProjects').catch(() => [] as NovelProjectSummary[])
  ])
  return { starters, models, projects }
}
