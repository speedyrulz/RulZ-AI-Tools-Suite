import { FolderOpen, Image as ImageIcon, Loader2, Save, Star } from 'lucide-react'
import { useEffect, useState } from 'react'
import { call } from '../lib/api'
import { GRID_STATUS, cellsDone, formatDate, missingImages, seedText } from '../lib/grid'
import { defaultGridId, gridDimensions, pickerLabel, starredCells, validSelection, viewableGrids } from '../lib/results'
import { saveGridImage, saveMix } from '../components/results/actions'
import { CompareView } from '../components/results/CompareView'
import { FullSizeOverlay } from '../components/results/FullSizeOverlay'
import { ResultsGrid } from '../components/results/ResultsGrid'
import { Button, Card, Checkbox, Field, PageTitle, StatusChip } from '../components/ui'
import { useApp } from '../stores/app'
import { useGrid } from '../stores/grid'
import { usePendingStars, useResults } from '../stores/results'

const SELECT_CLASS =
  'h-9 w-full rounded-lg border border-border bg-panel px-3 text-[13px] focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-45'

/** Full viewer for a finished or stopped grid: labelled cells, stars, full size, A/B flip and synced side by side. */
export function ResultsPage() {
  const grids = useGrid((s) => s.grids)
  const selectedId = useGrid((s) => s.selectedId)
  const images = useGrid((s) => s.images)
  const select = useGrid((s) => s.select)
  const setPage = useApp((s) => s.setPage)
  const selection = useResults((s) => s.selection)
  const starredOnly = useResults((s) => s.starredOnly)
  const setStarredOnly = useResults((s) => s.setStarredOnly)
  const pendingStars = usePendingStars((s) => s.pending)
  const [saving, setSaving] = useState(false)

  const shownId = defaultGridId(grids, selectedId)
  const job = grids.find((g) => g.id === shownId) ?? null

  // The list may not have been loaded yet when this page is the first one opened.
  useEffect(() => {
    void useGrid.getState().loadList()
  }, [])

  // Another grid: forget the picks, the open cell and the zoom.
  useEffect(() => {
    useResults.getState().reset()
  }, [shownId])

  // Fetch the images of finished cells.
  useEffect(() => {
    if (!job) return
    for (const f of missingImages(job, images)) void useGrid.getState().ensureImage(job.id, f)
  }, [job, images])

  if (!job) {
    return (
      <div>
        <PageTitle title="Results" />
        <Card>
          <p className="flex flex-wrap items-center gap-3 text-[13px] text-muted">
            <ImageIcon size={16} />
            <span>No finished grid yet. Make one on the Sweep page.</span>
            <Button size="sm" onClick={() => setPage('sweep')}>
              Go to Sweep
            </Button>
          </p>
        </Card>
      </div>
    )
  }

  const { cols, rows } = gridDimensions(job)
  const st = GRID_STATUS[job.status]
  const picks = validSelection(selection, cols, rows)
  const stars = starredCells(job, pendingStars[job.id] ?? job.favourites).length
  const done = cellsDone(job)
  const list = viewableGrids(grids)

  return (
    <div className="space-y-5">
      <PageTitle title="Results">Look closely at a finished grid: open a cell, flip between two, or put them side by side.</PageTitle>

      <Card
        title={job.templateName}
        description={`${formatDate(job.createdAt)}, ${seedText(job.request)}, ${cols} x ${rows} = ${job.cells.length} cells`}
        actions={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <StatusChip tone={st.tone} label={st.label} busy={st.busy} />
            <span className="text-xs text-faint tabular-nums">
              {done} / {job.cells.length} images
            </span>
            <Button
              size="sm"
              disabled={saving || done === 0}
              onClick={() => {
                setSaving(true)
                void saveGridImage(job).finally(() => setSaving(false))
              }}
            >
              {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              Save grid image
            </Button>
            <Button size="sm" onClick={() => void call('grid:openFolder', job.id)}>
              <FolderOpen size={14} />
              Open folder
            </Button>
            <Button
              size="sm"
              variant="primary"
              disabled={picks.length !== 1}
              title={picks.length === 1 ? 'Send this cell’s merge to the Merge page' : 'Pick exactly one cell (A/B button or Ctrl+click)'}
              onClick={() => saveMix(job, picks[0])}
            >
              Save this mix
            </Button>
          </div>
        }
      >
        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-[260px] flex-1">
            <Field label="Grid">
              <select
                aria-label="Grid to show"
                className={SELECT_CLASS}
                value={job.id}
                onChange={(e) => select(e.target.value)}
              >
                {list.map((g) => (
                  <option key={g.id} value={g.id}>
                    {pickerLabel(g)}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="pb-2">
            <Checkbox checked={starredOnly} onChange={setStarredOnly}>
              <span className="inline-flex items-center gap-1.5">
                <Star size={14} />
                Starred only ({stars})
              </span>
            </Checkbox>
          </div>
        </div>
        {job.error && <p className="mt-3 text-[13px] break-words text-danger-text">{job.error}</p>}
      </Card>

      <Card
        title="Cells"
        description="Click a cell for full size. Use the A/B button (or Ctrl+click) on two cells to compare them. The star keeps a cell as a favourite."
      >
        <ResultsGrid job={job} starredOnly={starredOnly} />
      </Card>

      {picks.length === 2 ? (
        <CompareView job={job} a={picks[0]} b={picks[1]} />
      ) : (
        <p className="text-xs text-faint">
          {picks.length === 1 ? 'One cell is picked. Pick a second one to compare, or use Save this mix.' : 'Pick two cells to compare them here.'}
        </p>
      )}

      <FullSizeOverlay job={job} />
    </div>
  )
}
