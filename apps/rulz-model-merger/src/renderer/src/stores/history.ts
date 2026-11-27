import { create } from 'zustand'
import { toast } from 'sonner'
import type { HistoryEntry } from '@shared/history'
import { call } from '../lib/api'
import { gridBlockReason, resolveAxisOption } from '../lib/grid'
import { entryId, mergeStartReason, onlyFavouritesChanged, rerunBlockReason, snapshotChanged, snapshotOf, sortEntries, type HistoryFilter } from '../lib/history'
import { loraMergeBlockReason } from '../lib/loraMerge'
import { useApp, type Page } from './app'
import { useGrid } from './grid'
import { useLoraMerge } from './loraMerge'
import { useMerge } from './merge'

interface HistoryStore {
  entries: HistoryEntry[]
  /** True once the first list has arrived. */
  loaded: boolean
  filter: HistoryFilter
  query: string
  setFilter: (f: HistoryFilter) => void
  setQuery: (q: string) => void
  refresh: () => Promise<void>
  /** Fills the entry's page with its recipe and opens that page. Resolves true when nothing needed was missing. */
  reopen: (e: HistoryEntry) => Promise<boolean>
  /** Reopen, then press that page's start through its store (so the store's and the main process's checks apply). */
  rerun: (e: HistoryEntry) => Promise<void>
  /** Merge and LoRA merge entries only (the saved file stays). */
  remove: (e: HistoryEntry) => Promise<void>
  openInResults: (gridId: string) => Promise<void>
  init: () => () => void
}

let refreshSeq = 0
let refreshTimer: number | undefined
const REFRESH_DELAY_MS = 300

function connected(): boolean {
  const s = useApp.getState().status
  return !!s?.reachable && (s.own || s.sharedQueueAccepted)
}

/** Resolves when the store's source-file check is not running (at once if it is not), or after `ms`. */
function settled(store: {
  getState: () => { checking: boolean }
  subscribe: (l: (s: { checking: boolean }) => void) => () => void
}, ms = 60000): Promise<void> {
  if (!store.getState().checking) return Promise.resolve()
  return new Promise((resolve) => {
    const done = (): void => {
      window.clearTimeout(timer)
      off()
      resolve()
    }
    const off = store.subscribe((s) => {
      if (!s.checking) done()
    })
    const timer = window.setTimeout(done, ms)
  })
}

const CHANGED_TEXT = 'Re-run stopped: the recipe was changed.'

/** What the LoRA merge form holds (compared before Re-run starts). */
function loraForm(l: ReturnType<typeof useLoraMerge.getState>): unknown {
  return { rows: l.rows, mode: l.mode, rank: l.rank, dtype: l.dtype, outputName: l.outputName }
}

/** What the Sweep form and the Merge recipe hold (compared before Re-run starts a grid). */
function gridForm(): unknown {
  const g = useGrid.getState()
  return {
    recipe: useMerge.getState().recipe,
    axisKey: g.axisKey,
    valuesText: g.valuesText,
    rowKind: g.rowKind,
    prompts: g.prompts,
    seed: g.seed,
    seeds: g.seeds,
    template: g.template?.path ?? null
  }
}

function openPage(p: Page): void {
  useApp.getState().setPage(p)
}

export const useHistory = create<HistoryStore>((set, get) => ({
  entries: [],
  loaded: false,
  filter: 'all',
  query: '',

  setFilter: (filter) => set({ filter }),
  setQuery: (query) => set({ query }),

  refresh: async () => {
    const seq = ++refreshSeq
    const list = await call('history:list')
    if (!list || seq !== refreshSeq) return
    set({ entries: sortEntries(list), loaded: true })
  },

  reopen: async (e) => {
    if (e.kind === 'merge') {
      useMerge.getState().replaceRecipe(e.job.recipe)
      openPage('merge')
      return true
    }
    if (e.kind === 'loraMerge') {
      useLoraMerge.getState().replaceRecipe(e.job.recipe)
      openPage('loras')
      return true
    }
    openPage('sweep')
    return useGrid.getState().loadRequest(e.job.request)
  },

  rerun: async (e) => {
    const blocked = rerunBlockReason(e, {
      connected: connected(),
      merges: useMerge.getState().jobs,
      loraMerges: useLoraMerge.getState().jobs
    })
    if (blocked) {
      toast.error(blocked)
      return
    }
    const ok = await get().reopen(e)
    if (e.kind === 'merge') {
      const snap = snapshotOf(useMerge.getState().recipe)
      // The page runs its source-file check when the recipe changes; wait for it, then apply the button's own conditions.
      await settled(useMerge)
      const m = useMerge.getState()
      if (snapshotChanged(snap, m.recipe)) {
        toast.error(CHANGED_TEXT)
        return
      }
      const reason = mergeStartReason({
        recipe: m.recipe,
        connected: connected(),
        nodes: useApp.getState().capabilities?.nodes ?? null,
        jobs: m.jobs,
        check: m.check,
        checking: m.checking,
        checkError: m.checkError,
        starting: m.starting
      })
      if (reason) {
        toast.error(`Re-run did not start: ${reason}`)
        return
      }
      await m.start()
      return
    }
    if (e.kind === 'loraMerge') {
      const snap = snapshotOf(loraForm(useLoraMerge.getState()))
      await settled(useLoraMerge)
      const l = useLoraMerge.getState()
      if (snapshotChanged(snap, loraForm(l))) {
        toast.error(CHANGED_TEXT)
        return
      }
      const reason = loraMergeBlockReason({
        rows: l.rows,
        check: l.check,
        checking: l.checking,
        checkError: l.checkError,
        jobs: l.jobs,
        starting: l.starting,
        outputName: l.outputName
      })
      if (reason) {
        toast.error(`Re-run did not start: ${reason}`)
        return
      }
      await l.start()
      return
    }
    if (!ok) return // the workflow could not be loaded; a toast said so
    const snap = snapshotOf(gridForm())
    await useGrid.getState().loadList()
    if (snapshotChanged(snap, gridForm())) {
      toast.error(CHANGED_TEXT)
      return
    }
    const g = useGrid.getState()
    const recipe = useMerge.getState().recipe
    const reason = gridBlockReason({
      connected: connected(),
      template: g.template,
      recipe,
      axis: resolveAxisOption(recipe, g.axisKey),
      valuesText: g.valuesText,
      rowKind: g.rowKind,
      prompts: g.prompts,
      seed: g.seed,
      seeds: g.seeds,
      grids: g.grids,
      starting: g.starting
    })
    if (reason) {
      toast.error(reason)
      return
    }
    await g.start()
  },

  remove: async (e) => {
    if (e.kind === 'grid') return
    const id = entryId(e)
    await call('history:remove', e.kind, e.job.id)
    set({ entries: get().entries.filter((x) => entryId(x) !== id) })
  },

  openInResults: async (gridId) => {
    await useGrid.getState().loadList()
    useGrid.getState().select(gridId)
    openPage('results')
  },

  init: () => {
    const schedule = (): void => {
      window.clearTimeout(refreshTimer)
      refreshTimer = window.setTimeout(() => void get().refresh(), REFRESH_DELAY_MS)
    }
    const offChanged = window.rulz.on('history:changed', schedule)
    // A grid reaching a final status is a new history entry. Starring cells only changes favourites: not worth a rescan.
    const offGrid = window.rulz.on('grid:job', (job) => {
      if (job.status === 'running') return
      const known = get().entries.find((x) => x.kind === 'grid' && x.job.id === job.id)
      if (known && known.kind === 'grid' && onlyFavouritesChanged(known.job, job)) return
      schedule()
    })
    return () => {
      window.clearTimeout(refreshTimer)
      offChanged()
      offGrid()
    }
  }
}))
