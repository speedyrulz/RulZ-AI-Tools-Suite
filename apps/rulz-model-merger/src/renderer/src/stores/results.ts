import { create } from 'zustand'
import { toast } from 'sonner'
import type { GridCellRef, GridJob } from '@shared/grid'
import { errText, invoke } from '../lib/api'
import { upsertGrid } from '../lib/grid'
import {
  RESET_VIEW,
  assignSlot,
  toggleFavourite,
  toggleSelection,
  type Slot,
  type ZoomView
} from '../lib/results'
import { useGrid } from './grid'

export type CompareMode = 'flip' | 'side'

interface ResultsStore {
  /** Picked cells: first is A, second is B. */
  selection: GridCellRef[]
  /** Cell shown large in the overlay, or null. */
  open: GridCellRef | null
  mode: CompareMode
  /** Which of the two is shown while flipping. */
  flipShows: Slot
  view: ZoomView
  starredOnly: boolean
  /** Forgets everything about the grid that was shown (another grid was opened). */
  reset: () => void
  toggleSelect: (cell: GridCellRef) => void
  assign: (slot: Slot, cell: GridCellRef) => void
  clearSelection: () => void
  setOpen: (cell: GridCellRef | null) => void
  setMode: (mode: CompareMode) => void
  flip: () => void
  setView: (view: ZoomView) => void
  resetView: () => void
  setStarredOnly: (on: boolean) => void
}

export const useResults = create<ResultsStore>((set, get) => ({
  selection: [],
  open: null,
  mode: 'flip',
  flipShows: 'A',
  view: RESET_VIEW,
  starredOnly: false,

  reset: () => set({ selection: [], open: null, flipShows: 'A', view: RESET_VIEW }),
  toggleSelect: (cell) => set({ selection: toggleSelection(get().selection, cell) }),
  assign: (slot, cell) => set({ selection: assignSlot(get().selection, slot, cell) }),
  clearSelection: () => set({ selection: [], flipShows: 'A', view: RESET_VIEW }),
  setOpen: (open) => set({ open }),
  setMode: (mode) => set({ mode }),
  flip: () => set({ flipShows: get().flipShows === 'A' ? 'B' : 'A' }),
  setView: (view) => set({ view }),
  resetView: () => set({ view: RESET_VIEW }),
  setStarredOnly: (starredOnly) => set({ starredOnly })
}))

/**
 * Favourites not yet confirmed by a save, by grid id. They win over the grid's own list while saves are in flight, so
 * a grid:list reply or an older grid:job event cannot bring the old stars back for a moment.
 */
export const usePendingStars = create<{ pending: Record<string, GridCellRef[]> }>(() => ({ pending: {} }))

/** Saves in flight per grid. */
const inflight = new Map<string, number>()

function setPending(gridId: string, favs: GridCellRef[] | null): void {
  const next = { ...usePendingStars.getState().pending }
  if (favs) next[gridId] = favs
  else delete next[gridId]
  usePendingStars.setState({ pending: next })
}

/** The favourites to show for a grid: the unconfirmed ones when a save is running, else the saved ones. */
export function useFavourites(job: GridJob): GridCellRef[] | undefined {
  const pending = usePendingStars((s) => s.pending[job.id])
  return pending ?? job.favourites
}

/** Star changes run one after the other, so quick clicks do not overwrite each other (each save is the full list). */
let favQueue: Promise<void> = Promise.resolve()

/** Stars or un-stars a cell: shown at once, saved in grid.json; undone with a message when saving fails. */
export function toggleStar(gridId: string, cell: GridCellRef): Promise<void> {
  const job = useGrid.getState().grids.find((g) => g.id === gridId)
  if (!job || job.status === 'running') return favQueue
  const next = toggleFavourite(usePendingStars.getState().pending[gridId] ?? job.favourites, cell)
  setPending(gridId, next)
  inflight.set(gridId, (inflight.get(gridId) ?? 0) + 1)
  favQueue = favQueue.then(async () => {
    let saved: GridJob | null = null
    try {
      saved = await invoke('grid:setFavourites', gridId, next)
    } catch (e) {
      toast.error(errText(e))
    }
    const left = (inflight.get(gridId) ?? 1) - 1
    if (left > 0) {
      inflight.set(gridId, left)
      return
    }
    inflight.delete(gridId)
    // The last save settled: the grid's own list is the truth again (on failure, the older saved one).
    if (saved) useGrid.setState({ grids: upsertGrid(useGrid.getState().grids, saved) })
    setPending(gridId, null)
  })
  return favQueue
}
