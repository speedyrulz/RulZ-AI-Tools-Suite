import { FolderOpen, Image as ImageIcon, Loader2, Save, Square, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { axisLabel, recipeForValue, type GridCell, type GridJob } from '@shared/grid'
import { call, errText, invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { GRID_STATUS, cellAt, cellListText, doneCellsMissing, cellFraction, cellsDone, formatDate, gridDims, gridImageName, gridTitle, imageKey, missingImages, pickShownGrid, promptHeader } from '../../lib/grid'
import { useApp } from '../../stores/app'
import { useGrid } from '../../stores/grid'
import { useMerge } from '../../stores/merge'
import { Badge, Button, Card, Dialog, StatusChip } from '../ui'
import { composeGridPng } from './compose'

function CellView({ job, cell }: { job: GridJob; cell: GridCell | undefined }) {
  const url = useGrid((s) => (cell?.file ? s.images[imageKey(job.id, cell.file)] : undefined))
  if (!cell) return <span className="text-xs text-faint">No data</span>
  if (cell.status === 'done') {
    return url ? (
      <img src={url} alt={`Cell ${cell.col + 1}, ${cell.row + 1}`} className="block h-auto w-full rounded-md" />
    ) : (
      <span className="inline-flex items-center gap-1.5 text-xs text-muted">
        <Loader2 size={14} className="animate-spin" />
        Loading image
      </span>
    )
  }
  if (cell.status === 'running') {
    const frac = cellFraction(job.progress)
    return (
      <div className="w-full space-y-1.5">
        <StatusChip tone="warn" label="Running" busy />
        <div
          className="h-1.5 overflow-hidden rounded-full bg-border"
          role="progressbar"
          aria-label="Image progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={frac === null ? undefined : Math.round(frac * 100)}
        >
          <div className={cn('h-full bg-accent transition-[width]', frac === null && 'w-1/3 animate-pulse')} style={frac === null ? undefined : { width: `${frac * 100}%` }} />
        </div>
        {job.progress && job.progress.max > 0 && (
          <p className="text-xs text-faint tabular-nums">
            {job.progress.value} / {job.progress.max}
          </p>
        )}
      </div>
    )
  }
  if (cell.status === 'error') return <p className="text-xs break-words text-danger-text">{cell.error ?? 'This image failed.'}</p>
  if (cell.status === 'cancelled') return <StatusChip tone="idle" label="Cancelled" />
  return <StatusChip tone="idle" label="Pending" />
}

async function saveGridImage(job: GridJob): Promise<void> {
  try {
    // Images load in the background; fetch any done cell that is not cached before drawing.
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

export function GridResultCard() {
  const grids = useGrid((s) => s.grids)
  const selectedId = useGrid((s) => s.selectedId)
  const images = useGrid((s) => s.images)
  const cancel = useGrid((s) => s.cancel)
  const ensureImage = useGrid((s) => s.ensureImage)
  const setPage = useApp((s) => s.setPage)
  const [saving, setSaving] = useState(false)
  const job = pickShownGrid(grids, selectedId)

  // Fetch the images of finished cells as they appear.
  useEffect(() => {
    if (!job) return
    for (const f of missingImages(job, images)) void ensureImage(job.id, f)
  }, [job, images, ensureImage])

  if (!job) {
    return (
      <Card title="Result">
        <p className="flex items-center gap-2 text-[13px] text-muted">
          <ImageIcon size={16} />
          No grid yet. Set it up above and press Start grid.
        </p>
      </Card>
    )
  }

  const { cols, rows } = gridDims(job)
  const st = GRID_STATUS[job.status]
  const running = job.status === 'running'
  const total = job.cells.length

  const use = (value: number): void => {
    const current = useMerge.getState().recipe
    const next = { ...recipeForValue(job.request.recipe, job.request.axis, value), outputName: current.outputName }
    useMerge.getState().replaceRecipe(next)
    setPage('merge')
  }

  return (
    <Card
      title="Result"
      description={`${job.templateName}, ${formatDate(job.createdAt)}, seed ${job.request.seed}`}
      actions={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <StatusChip tone={st.tone} label={st.label} busy={st.busy} />
          <span className="text-xs text-faint tabular-nums">
            {cellsDone(job)} / {total}
          </span>
          {running && (
            <Button size="sm" variant="danger" onClick={() => void cancel(job.id)}>
              <Square size={12} />
              Cancel
            </Button>
          )}
          <Button
            size="sm"
            disabled={saving || cellsDone(job) === 0}
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
        </div>
      }
    >
      {job.error && <p className="mb-3 text-[13px] break-words text-danger-text">{job.error}</p>}
      <div className="overflow-x-auto">
        <table className="w-full table-fixed border-separate border-spacing-2 text-[13px]" style={{ minWidth: 150 + cols * 170 }}>
          <colgroup>
            <col style={{ width: 150 }} />
            {job.request.axis.values.map((_, c) => (
              <col key={c} />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th />
              {job.request.axis.values.map((v, c) => (
                <th key={c} className="px-1 pb-1 text-left align-bottom font-medium">
                  <div className="tabular-nums">{axisLabel(job.request.axis, job.request.recipe, v)}</div>
                  <Button size="sm" className="mt-1.5" onClick={() => use(v)}>
                    Use this value
                  </Button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: rows }, (_, r) => (
              <tr key={r}>
                <th className="pr-2 text-left align-top text-xs font-normal text-muted" title={job.request.prompts[r]}>
                  {promptHeader(job.request.prompts[r] ?? '')}
                </th>
                {Array.from({ length: cols }, (_, c) => (
                  <td key={c} className="rounded-lg border border-border bg-panel-2 p-1.5 align-top">
                    <CellView job={job} cell={cellAt(job, c, r)} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-faint">Use this value sets the merge on the Merge page to that column. Nothing is saved until you merge there.</p>
    </Card>
  )
}

export function EarlierGridsCard() {
  const grids = useGrid((s) => s.grids)
  const selectedId = useGrid((s) => s.selectedId)
  const select = useGrid((s) => s.select)
  const remove = useGrid((s) => s.remove)
  const [confirm, setConfirm] = useState<GridJob | null>(null)
  const shown = pickShownGrid(grids, selectedId)

  return (
    <Card title="Earlier grids" description="Kept in the grid folders, also after the app closes.">
      {grids.length === 0 ? (
        <p className="text-[13px] text-muted">No grid yet.</p>
      ) : (
        <ul className="space-y-2">
          {grids.map((g) => {
            const st = GRID_STATUS[g.status]
            const { cols, rows } = gridDims(g)
            return (
              <li
                key={g.id}
                className={cn(
                  'flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-panel-2 px-3 py-2',
                  shown?.id === g.id ? 'border-border-strong' : 'border-border'
                )}
              >
                <button type="button" className="flex min-w-0 flex-1 flex-wrap items-center gap-x-4 gap-y-1 text-left" onClick={() => select(g.id)}>
                  <span className="text-xs text-faint tabular-nums">{formatDate(g.createdAt)}</span>
                  <span className="min-w-0 truncate text-[13px]" title={g.templateName}>
                    {g.templateName}
                  </span>
                  <span className="text-[13px] text-muted">{gridTitle(g)}</span>
                  <Badge className="tabular-nums">
                    {cols} x {rows}
                  </Badge>
                  <StatusChip tone={st.tone} label={st.label} busy={st.busy} />
                </button>
                <Button size="sm" variant="ghost" disabled={g.status === 'running'} onClick={() => setConfirm(g)}>
                  <Trash2 size={14} />
                  Delete
                </Button>
              </li>
            )
          })}
        </ul>
      )}
      <Dialog
        open={confirm !== null}
        onOpenChange={(o) => {
          if (!o) setConfirm(null)
        }}
        title="Delete this grid?"
        footer={
          <>
            <Button onClick={() => setConfirm(null)}>Keep it</Button>
            <Button
              variant="danger"
              onClick={() => {
                const id = confirm?.id
                setConfirm(null)
                if (id) void remove(id)
              }}
            >
              <Trash2 size={14} />
              Delete grid
            </Button>
          </>
        }
      >
        <p className="text-[13px] text-muted">
          The folder of this grid, with its images, goes to the Recycle Bin. You can restore it from there.
        </p>
      </Dialog>
    </Card>
  )
}
