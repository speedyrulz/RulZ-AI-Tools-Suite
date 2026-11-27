import { Loader2, Star } from 'lucide-react'
import { useGrid } from '../../stores/grid'
import { toggleStar, useFavourites } from '../../stores/results'
import { cellAt, imageKey } from '../../lib/grid'
import { cn } from '../../lib/format'
import { cellTitle, isFavourite } from '../../lib/results'
import type { GridCellRef, GridJob } from '@shared/grid'

/** The picture of a cell, or a short word when there is none yet. */
export function CellImage({ job, cell, className }: { job: GridJob; cell: GridCellRef; className?: string }) {
  const data = cellAt(job, cell.col, cell.row)
  const url = useGrid((s) => (data?.file ? s.images[imageKey(job.id, data.file)] : undefined))
  if (!data) return <span className="text-xs text-faint">No data</span>
  if (data.status === 'done') {
    return url ? (
      <img src={url} alt={cellTitle(job, cell)} draggable={false} className={cn('block h-auto w-full rounded-md', className)} />
    ) : (
      <span className="inline-flex items-center gap-1.5 text-xs text-muted">
        <Loader2 size={14} className="animate-spin" />
        Loading image
      </span>
    )
  }
  if (data.status === 'error') return <p className="text-xs break-words text-danger-text">{data.error ?? 'This image failed.'}</p>
  if (data.status === 'cancelled') return <span className="text-xs text-muted">Cancelled</span>
  return <span className="text-xs text-muted">No image</span>
}

/** Image URL of a cell once it is cached, else undefined. */
export function useCellUrl(job: GridJob, cell: GridCellRef | undefined): string | undefined {
  const data = cell ? cellAt(job, cell.col, cell.row) : undefined
  return useGrid((s) => (data?.status === 'done' && data.file ? s.images[imageKey(job.id, data.file)] : undefined))
}

/** Star toggle of a cell; the state comes from the grid's saved favourites. */
export function StarButton({ job, cell, className }: { job: GridJob; cell: GridCellRef; className?: string }) {
  const on = isFavourite(useFavourites(job), cell)
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={`${on ? 'Remove star from' : 'Star'} ${cellTitle(job, cell)}`}
      title={on ? 'Remove star' : 'Star this cell'}
      onClick={() => void toggleStar(job.id, cell)}
      className={cn(
        'inline-flex size-7 items-center justify-center rounded-md border border-border bg-panel/90 transition-colors hover:bg-hover',
        on ? 'text-warn-text' : 'text-muted hover:text-text',
        className
      )}
    >
      <Star size={14} fill={on ? 'currentColor' : 'none'} />
    </button>
  )
}
