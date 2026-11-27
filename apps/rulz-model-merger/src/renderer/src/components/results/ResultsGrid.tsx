import type { GridCellRef, GridJob } from '@shared/grid'
import { cellAt } from '../../lib/grid'
import { cn } from '../../lib/format'
import { cellTitle, columnLabel, gridDimensions, rowLabel, slotOf, starredCells } from '../../lib/results'
import { useFavourites, useResults } from '../../stores/results'
import { CellImage, StarButton } from './parts'

/** One cell: the picture opens large, the A/B button picks it for comparing, the star saves it as a favourite. */
function Tile({ job, cell, label }: { job: GridJob; cell: GridCellRef; label?: string }) {
  const selection = useResults((s) => s.selection)
  const toggleSelect = useResults((s) => s.toggleSelect)
  const setOpen = useResults((s) => s.setOpen)
  const slot = slotOf(selection, cell)
  const done = cellAt(job, cell.col, cell.row)?.status === 'done'
  const title = cellTitle(job, cell)
  return (
    <div className={cn('relative rounded-lg border bg-panel-2 p-1.5', slot ? 'border-accent' : 'border-border')}>
      {label && <p className="mb-1 truncate pr-16 text-xs text-muted">{label}</p>}
      <button
        type="button"
        disabled={!done}
        aria-label={`Open ${title} full size`}
        title="Click to open full size. Ctrl+click picks it for comparing."
        onClick={(e) => (e.ctrlKey || e.metaKey ? toggleSelect(cell) : setOpen(cell))}
        className="block w-full rounded-md text-left disabled:cursor-default"
      >
        <CellImage job={job} cell={cell} />
      </button>
      <div className="absolute top-2.5 right-2.5 left-2.5 flex items-start justify-between">
        <button
          type="button"
          aria-pressed={slot !== null}
          aria-label={slot ? `${title} is picked as ${slot}. Remove the pick.` : `Pick ${title} for comparing`}
          title={slot ? `Picked as ${slot}` : 'Pick for comparing (A or B)'}
          disabled={!done}
          onClick={() => toggleSelect(cell)}
          className={cn(
            'inline-flex h-7 min-w-7 items-center justify-center rounded-md border px-1.5 text-xs font-semibold transition-colors disabled:opacity-45',
            slot ? 'border-accent bg-accent text-accent-fg' : 'border-border bg-panel/90 text-muted hover:bg-hover hover:text-text'
          )}
        >
          {slot ?? 'A/B'}
        </button>
        <StarButton job={job} cell={cell} />
      </div>
    </div>
  )
}

/** All cells with column and row labels, or only the starred ones as a flat list. */
export function ResultsGrid({ job, starredOnly }: { job: GridJob; starredOnly: boolean }) {
  const { cols, rows } = gridDimensions(job)
  const favs = useFavourites(job)

  if (starredOnly) {
    const stars = starredCells(job, favs)
    if (stars.length === 0) return <p className="text-[13px] text-muted">No starred cells yet. Star a cell to find it here.</p>
    return (
      <ul className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
        {stars.map((c) => (
          <li key={`${c.col}/${c.row}`}>
            <Tile job={job} cell={c} label={cellTitle(job, c)} />
          </li>
        ))}
      </ul>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full table-fixed border-separate border-spacing-2 text-[13px]" style={{ minWidth: 150 + cols * 170 }}>
        <colgroup>
          <col style={{ width: 150 }} />
          {Array.from({ length: cols }, (_, c) => (
            <col key={c} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th />
            {Array.from({ length: cols }, (_, c) => (
              <th key={c} scope="col" className="px-1 pb-1 text-left align-bottom font-medium tabular-nums">
                {columnLabel(job, c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }, (_, r) => (
            <tr key={r}>
              <th scope="row" className="pr-2 text-left align-top text-xs font-normal break-words text-muted">
                {rowLabel(job, r)}
              </th>
              {Array.from({ length: cols }, (_, c) => (
                <td key={c} className="align-top">
                  <Tile job={job} cell={{ col: c, row: r }} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
