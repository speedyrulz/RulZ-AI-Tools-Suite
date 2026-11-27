// History (milestone 6): every finished merge, bake, LoRA merge and grid, kept across app restarts.
// Merges and LoRA merges are stored by the main process in <userData>/history.json when they reach a final status
// (done, error, cancelled); grids are not copied there, they are read from their grid folders (grid:list).
import type { MergeJob } from './merge'
import type { LoraMergeJob } from './loraMerge'
import type { GridJob } from './grid'

/** Most entries kept in history.json (oldest dropped first). Grids are not counted: they live in their folders. */
export const HISTORY_MAX_ENTRIES = 500

/** A merge entry covers every MergeRecipe mode, including 'bake'. */
export type HistoryEntry =
  | { kind: 'merge'; job: MergeJob }
  | { kind: 'loraMerge'; job: LoraMergeJob }
  | { kind: 'grid'; job: GridJob }

export type HistoryKind = HistoryEntry['kind']

/** history.json */
export interface HistoryFile {
  version: 1
  /** Merge and LoRA merge entries only, newest first. */
  entries: Array<Extract<HistoryEntry, { kind: 'merge' | 'loraMerge' }>>
}
