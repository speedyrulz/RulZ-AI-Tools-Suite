import * as DialogPrimitive from '@radix-ui/react-dialog'
import { Save, X } from 'lucide-react'
import type { GridJob } from '@shared/grid'
import { cellAt } from '../../lib/grid'
import { columnLabel, gridDimensions, keyDirection, neighbour, rowLabel, sameCell, slotOf } from '../../lib/results'
import { useResults } from '../../stores/results'
import { Badge, Button } from '../ui'
import { saveCellImage, saveMix } from './actions'
import { CellImage, StarButton } from './parts'

/** The open cell large. Arrow keys move along the row and column, Esc closes; focus stays inside and returns to the cell. */
export function FullSizeOverlay({ job }: { job: GridJob }) {
  const open = useResults((s) => s.open)
  const setOpen = useResults((s) => s.setOpen)
  const selection = useResults((s) => s.selection)
  const assign = useResults((s) => s.assign)
  const { cols, rows } = gridDimensions(job)
  const slot = open ? slotOf(selection, open) : null
  const done = open ? cellAt(job, open.col, open.row)?.status === 'done' : false

  return (
    <DialogPrimitive.Root open={open !== null} onOpenChange={(o) => !o && setOpen(null)}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-scrim backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          className="fixed inset-4 z-50 flex flex-col rounded-xl border border-border bg-panel focus:outline-none"
          onKeyDown={(e) => {
            const dir = keyDirection(e.key)
            if (!dir || !open || e.ctrlKey || e.metaKey || e.altKey) return
            e.preventDefault()
            const next = neighbour(cols, rows, open, dir)
            if (!sameCell(next, open)) setOpen(next)
          }}
        >
          {open && (
            <>
              <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-3">
                <div className="min-w-0">
                  <DialogPrimitive.Title className="font-[family-name:var(--font-display)] text-base font-semibold tabular-nums">
                    {columnLabel(job, open.col)}
                  </DialogPrimitive.Title>
                  <DialogPrimitive.Description className="mt-0.5 text-[13px] break-words text-muted">
                    {rowLabel(job, open.row, 200)}
                  </DialogPrimitive.Description>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {slot && <Badge tone="accent">Picked as {slot}</Badge>}
                  <span className="text-xs text-faint tabular-nums">
                    Column {open.col + 1} of {cols}, row {open.row + 1} of {rows}
                  </span>
                  <StarButton job={job} cell={open} />
                  <DialogPrimitive.Close className="rounded-md p-1.5 text-muted hover:bg-hover hover:text-text" aria-label="Close">
                    <X size={16} />
                  </DialogPrimitive.Close>
                </div>
              </div>
              <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-4">
                <CellImage job={job} cell={open} className="mx-auto max-h-[calc(100vh-190px)] w-auto max-w-full object-contain" />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-5 py-3">
                <p className="text-xs text-faint">Arrow keys move to the next cell. Esc closes.</p>
                <div className="flex flex-wrap items-center gap-2">
                  <Button size="sm" disabled={!done} aria-pressed={slot === 'A'} onClick={() => assign('A', open)}>
                    Set as A
                  </Button>
                  <Button
                    size="sm"
                    disabled={!done || selection.length === 0 || sameCell(selection[0], open)}
                    aria-pressed={slot === 'B'}
                    onClick={() => assign('B', open)}
                  >
                    Set as B
                  </Button>
                  <Button size="sm" disabled={!done} onClick={() => void saveCellImage(job, open)}>
                    <Save size={14} />
                    Save image
                  </Button>
                  <Button
                    size="sm"
                    disabled={!done}
                    onClick={() => {
                      setOpen(null)
                      saveMix(job, open)
                    }}
                  >
                    Save this mix
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
