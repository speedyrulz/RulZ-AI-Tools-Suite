import { toast } from 'sonner'
import { recipeForValue, type GridCellRef, type GridJob } from '@shared/grid'
import { errText, invoke } from '../../lib/api'
import { cellListText, doneCellsMissing, gridImageName, imageKey } from '../../lib/grid'
import { cellImageName } from '../../lib/results'
import { useApp } from '../../stores/app'
import { useGrid } from '../../stores/grid'
import { useMerge } from '../../stores/merge'
import { composeGridPng } from '../grid/compose'

/** Sends the merge of a cell's column to the Merge page (keeps the output name there). Nothing is merged yet. */
export function saveMix(job: GridJob, cell: GridCellRef): void {
  const value = job.request.axis.values[cell.col]
  if (value === undefined) return
  const current = useMerge.getState().recipe
  const next = { ...recipeForValue(job.request.recipe, job.request.axis, value), outputName: current.outputName }
  useMerge.getState().replaceRecipe(next)
  toast.success('Mix sent to the Merge page. Nothing is saved until you merge there.')
  useApp.getState().setPage('merge')
}

/** Composes the whole grid as one labelled PNG and saves it through a save dialog. */
export async function saveGridImage(job: GridJob): Promise<void> {
  try {
    // Images load in the background; fetch any finished cell that is not cached before drawing.
    const missing = doneCellsMissing(job, useGrid.getState().images)
    const results = await Promise.all(missing.map((m) => useGrid.getState().fetchImageNow(job.id, m.file)))
    const failed = missing.filter((_, i) => !results[i])
    if (failed.length > 0) {
      toast.error(`Not saved. These images could not be loaded: ${cellListText(failed)}. Try again.`)
      return
    }
    const dataUrl = await composeGridPng(job, useGrid.getState().images)
    const path = await invoke('grid:saveImage', dataUrl, gridImageName(job))
    if (path) toast.success('Grid image saved')
  } catch (e) {
    toast.error(errText(e))
  }
}

/** Saves one cell's picture as it was rendered. */
export async function saveCellImage(job: GridJob, cell: GridCellRef): Promise<void> {
  try {
    const data = job.cells.find((c) => c.col === cell.col && c.row === cell.row)
    if (!data || data.status !== 'done' || !data.file) {
      toast.error('This cell has no image to save.')
      return
    }
    const ok = await useGrid.getState().fetchImageNow(job.id, data.file)
    const url = ok ? useGrid.getState().images[imageKey(job.id, data.file)] : undefined
    if (!url) {
      toast.error('The image could not be loaded. Try again.')
      return
    }
    const path = await invoke('grid:saveImage', url, cellImageName(job, cell))
    if (path) toast.success('Image saved')
  } catch (e) {
    toast.error(errText(e))
  }
}
