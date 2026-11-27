import { Columns2, FlipHorizontal2, RotateCcw, X } from 'lucide-react'
import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react'
import type { GridCellRef, GridJob } from '@shared/grid'
import { cn } from '../../lib/format'
import { cellTitle, panBy, viewTransform, wheelFactor, zoomAt, type ZoomView } from '../../lib/results'
import { useResults, type CompareMode } from '../../stores/results'
import { Button, Card, Segmented } from '../ui'
import { useCellUrl } from './parts'

/** A picture in a frame. Wheel zooms at the pointer and a drag pans; the view is shared, so panes show the same region. */
function ZoomPane({
  url,
  alt,
  view,
  onView,
  badge,
  onTap
}: {
  url: string | undefined
  alt: string
  view: ZoomView
  onView: (v: ZoomView) => void
  badge: string
  /** Called for a click that was not a drag. */
  onTap?: () => void
}) {
  const frame = useRef<HTMLDivElement>(null)
  const latest = useRef({ view, onView })
  latest.current = { view, onView }
  const drag = useRef<{ x: number; y: number; moved: number } | null>(null)

  // A wheel listener that may call preventDefault, so the page does not scroll while zooming.
  useEffect(() => {
    const el = frame.current
    if (!el) return
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault()
      const r = el.getBoundingClientRect()
      if (r.width <= 0 || r.height <= 0) return
      const fx = (e.clientX - r.left) / r.width
      const fy = (e.clientY - r.top) / r.height
      latest.current.onView(zoomAt(latest.current.view, wheelFactor(e.deltaY), fx, fy))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  return (
    <div
      ref={frame}
      className={cn('relative touch-none overflow-hidden rounded-md border border-border bg-panel-2', view.scale > 1 ? 'cursor-grab active:cursor-grabbing' : onTap && 'cursor-pointer')}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.currentTarget.setPointerCapture(e.pointerId)
        drag.current = { x: e.clientX, y: e.clientY, moved: 0 }
      }}
      onPointerMove={(e) => {
        const d = drag.current
        const el = frame.current
        if (!d || !el) return
        const r = el.getBoundingClientRect()
        if (r.width <= 0 || r.height <= 0) return
        const dx = e.clientX - d.x
        const dy = e.clientY - d.y
        d.x = e.clientX
        d.y = e.clientY
        d.moved += Math.abs(dx) + Math.abs(dy)
        if (latest.current.view.scale > 1) onView(panBy(latest.current.view, dx / r.width, dy / r.height))
      }}
      onPointerUp={() => {
        const d = drag.current
        drag.current = null
        if (d && d.moved < 4) onTap?.()
      }}
      onPointerCancel={() => {
        drag.current = null
      }}
    >
      {url ? (
        <img
          src={url}
          alt={alt}
          draggable={false}
          className="block h-auto w-full select-none"
          style={{ transform: viewTransform(view), transformOrigin: '0 0' }}
        />
      ) : (
        <div className="flex aspect-square items-center justify-center text-xs text-muted">No image</div>
      )}
      <span className="pointer-events-none absolute top-2 left-2 inline-flex h-7 min-w-7 items-center justify-center rounded-md bg-accent px-2 text-sm font-bold text-accent-fg">
        {badge}
      </span>
    </div>
  )
}

const MODES: { value: CompareMode; label: string; icon: ReactNode }[] = [
  { value: 'flip', label: 'Flip', icon: <FlipHorizontal2 size={14} /> },
  { value: 'side', label: 'Side by side', icon: <Columns2 size={14} /> }
]

/** Two picked cells compared: one at a time (flip with Space or a click) or next to each other. */
export function CompareView({ job, a, b }: { job: GridJob; a: GridCellRef; b: GridCellRef }) {
  const mode = useResults((s) => s.mode)
  const setMode = useResults((s) => s.setMode)
  const flipShows = useResults((s) => s.flipShows)
  const flip = useResults((s) => s.flip)
  const view = useResults((s) => s.view)
  const setView = useResults((s) => s.setView)
  const resetView = useResults((s) => s.resetView)
  const clearSelection = useResults((s) => s.clearSelection)
  const urlA = useCellUrl(job, a)
  const urlB = useCellUrl(job, b)
  const titleA = cellTitle(job, a)
  const titleB = cellTitle(job, b)
  const shown = flipShows === 'A' ? 'A' : 'B'

  const onKey = (e: KeyboardEvent<HTMLDivElement>): void => {
    if (e.key === ' ' && e.target === e.currentTarget) {
      e.preventDefault()
      flip()
    }
  }

  return (
    <Card
      title="Compare"
      description="Wheel zooms, drag pans. Both pictures always show the same region."
      actions={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Segmented label="Compare view" value={mode} options={MODES} onChange={setMode} />
          <Button size="sm" onClick={resetView} disabled={view.scale === 1}>
            <RotateCcw size={14} />
            Reset
          </Button>
          <Button size="sm" variant="ghost" onClick={clearSelection}>
            <X size={14} />
            Clear picks
          </Button>
        </div>
      }
    >
      <dl className="mb-3 grid gap-1 text-[13px] sm:grid-cols-2">
        <div className={cn('rounded-md px-2 py-1', mode === 'flip' && shown === 'A' && 'bg-accent-soft')}>
          <dt className="inline font-semibold">A </dt>
          <dd className="inline break-words text-muted">{titleA}</dd>
        </div>
        <div className={cn('rounded-md px-2 py-1', mode === 'flip' && shown === 'B' && 'bg-accent-soft')}>
          <dt className="inline font-semibold">B </dt>
          <dd className="inline break-words text-muted">{titleB}</dd>
        </div>
      </dl>
      {mode === 'flip' ? (
        <div
          tabIndex={0}
          role="group"
          aria-label={`Flip view, showing ${shown}. Space or click flips between A and B.`}
          onKeyDown={onKey}
          className="mx-auto max-w-[900px] rounded-lg"
        >
          <ZoomPane
            url={shown === 'A' ? urlA : urlB}
            alt={shown === 'A' ? titleA : titleB}
            view={view}
            onView={setView}
            badge={shown}
            onTap={flip}
          />
          <div className="mt-2 flex items-center justify-between gap-2">
            <p className="text-xs text-faint" aria-live="polite">
              Showing {shown}. Press Space or click the picture to flip.
            </p>
            <Button size="sm" onClick={flip}>
              <FlipHorizontal2 size={14} />
              Flip to {shown === 'A' ? 'B' : 'A'}
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          <ZoomPane url={urlA} alt={titleA} view={view} onView={setView} badge="A" />
          <ZoomPane url={urlB} alt={titleB} view={view} onView={setView} badge="B" />
        </div>
      )}
    </Card>
  )
}
