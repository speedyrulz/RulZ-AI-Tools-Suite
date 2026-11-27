// Messages between LoraMergeService and the merge worker thread.
import type { LoraMergeRecipe } from '@shared/loraMerge'

export interface LoraWorkerRequest {
  inputs: Array<{ name: string; uncPath: string; weight: number }>
  recipe: LoraMergeRecipe
  /** UNC path the merged file is written to (inside this job's staging folder). */
  outUncPath: string
  appVersion: string
  /** Header SHA-256 of each input, in order (null when unknown); stored in the saved metadata. */
  inputHeaderSha256: Array<string | null>
}

export interface LoraWorkerDone {
  type: 'done'
  sizeBytes: number
  modules: number
  rank: { min: number; max: number }
  energyKept: { mean: number; min: number } | null
  verify: { ok: boolean; message: string }
  /** The metadata the file was saved with (key "rulz_model_merger" holds the record JSON). */
  metadata: Record<string, string>
}

export type LoraWorkerMessage =
  | { type: 'progress'; done: number; total: number }
  | LoraWorkerDone
  | { type: 'error'; message: string }
