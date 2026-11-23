import { toast } from 'sonner'
import { create } from 'zustand'
import { applyChange, type Adventure, type AdventureChange, type AdventureEdit, type AdventureJob, type AdventureMeta, type AdventureSetup, type PlayRequest, type RetryRequest } from '@shared/adventure'
import { errorMessage } from '../lib/format'
import { link } from '../lib/remote'
import { useUi } from './ui'

export const updateMessage = 'Update RulZ LLM Suite on the computer to play adventures from the phone.'
export function adventureError(e: unknown): string {
  const message = errorMessage(e)
  return /That is not available (from the phone|yet)\./i.test(message) ? updateMessage : message
}
export const running = (job: AdventureJob): boolean => !['done', 'stopped', 'error'].includes(job.status)
export function fixPosition(adv: Adventure, position: string | null): string | null {
  if (position && adv.nodes[position]) return position
  if (adv.currentId && adv.nodes[adv.currentId]) return adv.currentId
  return adv.rootId && adv.nodes[adv.rootId] ? adv.rootId : null
}

interface AdventureStore {
  list: AdventureMeta[]
  error: string
  adventure: Adventure | null
  position: string | null
  jobs: Record<string, AdventureJob>
  ownJobs: string[]
  followedJobs: string[]
  otherNode: string | null
  starting: boolean
  load: () => Promise<void>
  open: (id: string) => Promise<boolean>
  close: () => void
  refresh: () => Promise<void>
  change: (change: AdventureChange) => void
  job: (job: AdventureJob) => void
  move: (id: string) => Promise<void>
  play: (req: PlayRequest) => Promise<AdventureJob | null>
  retry: (req: RetryRequest) => Promise<void>
  update: (id: string, patch: Partial<Omit<AdventureSetup, 'world' | 'premise' | 'character'>>) => Promise<boolean>
  edit: (edit: AdventureEdit) => Promise<'ok' | 'conflict' | 'error'>
}

export const useAdventure = create<AdventureStore>((set, get) => {
  let openSeq = 0
  let openBuffer: { id: string, events: ({ change: AdventureChange } | { job: AdventureJob })[] } | null = null
  let pendingPosition: string | null = null
  const reported = new Set<string>()
  let jobVersion = 0
  const jobVersions = new Map<string, number>()
  const report = (e: unknown): void => { toast.error(adventureError(e)) }
  const startJob = async (call: () => Promise<AdventureJob>): Promise<AdventureJob | null> => {
    if (get().starting) return null
    set({ starting: true })
    try {
      const job = await call()
      if (get().adventure?.id !== job.adventureId) return job
      set((s) => ({ ownJobs: [...s.ownJobs, job.id] }))
      get().job(get().jobs[job.id] ?? job)
      return job
    } catch (e) { report(e); return null }
    finally { set({ starting: false }) }
  }
  return {
    list: [], error: '', adventure: null, position: null, jobs: {}, ownJobs: [], followedJobs: [], otherNode: null, starting: false,
    load: async () => {
      try { set({ list: await link.invoke('adventure:list'), error: '' }) }
      catch (e) { set({ error: adventureError(e) }) }
    },
    open: async (id) => {
      const seq = ++openSeq
      const buffer = { id, events: [] as ({ change: AdventureChange } | { job: AdventureJob })[] }
      openBuffer = buffer
      try {
        const [adventure, jobs] = await Promise.all([link.invoke('adventure:get', id), link.invoke('adventure:jobs', id)])
        if (seq !== openSeq) return false
        openBuffer = null
        pendingPosition = null
        set({ adventure, position: fixPosition(adventure, null), jobs: {}, ownJobs: [], followedJobs: [], otherNode: null })
        for (const job of jobs) get().job(job)
        for (const event of buffer.events) {
          if ('change' in event) get().change(event.change)
          else get().job(event.job)
        }
        return get().adventure?.id === id
      } catch (e) {
        if (seq === openSeq) { openBuffer = null; report(e) }
        return false
      }
    },
    close: () => {
      openSeq++
      openBuffer = null
      pendingPosition = null
      set({ adventure: null, position: null, jobs: {}, ownJobs: [], followedJobs: [], otherNode: null })
    },
    refresh: async () => {
      const id = get().adventure?.id
      if (!id) return
      const seq = openSeq
      const version = jobVersion
      let missing = false
      try {
        const [adventure, jobs] = await Promise.all([
          link.invoke('adventure:get', id).catch((e) => {
            missing = /adventure.*not found/i.test(errorMessage(e))
            throw e
          }),
          link.invoke('adventure:jobs', id)
        ])
        const current = get().adventure
        if (!current || current.id !== id || seq !== openSeq) return
        const held = adventure.rev >= current.rev ? adventure : current
        const position = pendingPosition && held.nodes[pendingPosition] ? pendingPosition : fixPosition(held, get().position)
        if (pendingPosition === position) pendingPosition = null
        set({ adventure: held, position })
        const latest = get().jobs
        set({ jobs: {} })
        for (const job of jobs) get().job((jobVersions.get(job.id) ?? 0) > version ? latest[job.id] ?? job : job)
      } catch (e) {
        if (seq !== openSeq || get().adventure?.id !== id) return
        if (missing) get().change({ id, rev: null })
        else report(e)
      }
    },
    change: (change) => {
      if (openBuffer?.id === change.id) { openBuffer.events.push({ change }); return }
      const s = get()
      if (change.rev === null) {
        set({ list: s.list.filter((item) => item.id !== change.id) })
        if (s.adventure?.id === change.id) {
          get().close()
          useUi.getState().go('adventures')
          toast.info('This adventure was deleted on the computer.')
        }
        return
      }
      if (s.adventure?.id === change.id && change.rev <= s.adventure.rev) return
      if (change.meta) set((state) => ({
        list: [change.meta!, ...state.list.filter((item) => item.id !== change.id)].sort((a, b) => b.updatedAt - a.updatedAt)
      }))
      if (s.adventure?.id === change.id) {
        const adventure = applyChange(s.adventure, change)
        const position = pendingPosition && adventure.nodes[pendingPosition] ? pendingPosition : fixPosition(adventure, s.position)
        if (pendingPosition === position) pendingPosition = null
        set({ adventure, position })
      }
    },
    job: (job) => {
      if (openBuffer?.id === job.adventureId) { openBuffer.events.push({ job }); return }
      const s = get()
      if (s.adventure?.id !== job.adventureId) return
      jobVersions.set(job.id, ++jobVersion)
      set({ jobs: { ...s.jobs, [job.id]: job } })
      if (job.resultNodeId && job.kind !== 'retryChoices') {
        if (s.ownJobs.includes(job.id) && !s.followedJobs.includes(job.id)) {
          pendingPosition = s.adventure.nodes[job.resultNodeId] ? null : job.resultNodeId
          set({ position: pendingPosition ? s.position : job.resultNodeId, otherNode: null, followedJobs: [...s.followedJobs, job.id] })
        }
        else if (!s.ownJobs.includes(job.id) && job.resultNodeId !== s.position && s.jobs[job.id]?.resultNodeId !== job.resultNodeId) set({ otherNode: job.resultNodeId })
      }
      if (s.ownJobs.includes(job.id) && !running(job) && !reported.has(job.id)) {
        reported.add(job.id)
        if (job.status === 'error') toast.error(job.error || 'The adventure job failed.')
        if (job.status === 'stopped') toast.info(job.resultNodeId ? 'Stopped. The passage was kept.' : 'Stopped.')
        for (const warning of new Set(job.warnings)) toast.warning(warning)
        void link.invoke('adventure:dismiss', job.id).catch(() => undefined)
      }
    },
    move: async (id) => {
      const adventure = get().adventure
      if (!adventure?.nodes[id]) return
      pendingPosition = null
      set({ position: id, otherNode: null })
      try { get().change(await link.invoke('adventure:setCurrent', adventure.id, id)) } catch (e) { report(e) }
    },
    play: (req) => startJob(() => link.invoke('adventure:play', req)),
    retry: async (req) => { await startJob(() => link.invoke('adventure:retry', req)) },
    update: async (id, patch) => {
      try { get().change(await link.invoke('adventure:updateSetup', id, patch)); return true }
      catch (e) { report(e); return false }
    },
    edit: async (edit) => {
      try { get().change(await link.invoke('adventure:edit', edit)); return 'ok' }
      catch (e) {
        if (/conflict|changed somewhere else/i.test(errorMessage(e))) {
          await get().refresh()
          toast.error('This passage was changed somewhere else. Open Edit again to see the latest version.')
          return 'conflict'
        }
        report(e)
        return 'error'
      }
    }
  }
})

export function listenForAdventureChanges(): void {
  link.on('adventure:changed', (change) => useAdventure.getState().change(change))
  link.on('adventure:job', (job) => useAdventure.getState().job(job))
}
