import { GripVertical, LayoutGrid, List, SearchX, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type DragEvent, type KeyboardEvent, type ReactNode } from 'react'
import { arrangeHome, homeCounts, moveInOrder, type HomeFilter } from '@shared/arrange'
import type { AppView } from '@shared/types'
import { AppCard, AppRow } from '../components/AppCard'
import { PhoneAppButton } from '../components/PhoneAppQr'
import { cn } from '../lib/format'
import { LOGOS } from '../lib/theme'
import { act, useApps } from '../stores/apps'

export function HomePage({
  apps,
  query,
  onQuery,
  onOpen,
  onInstall
}: {
  apps: AppView[]
  query: string
  onQuery: (q: string) => void
  onOpen: (id: string) => void
  onInstall: (id: string) => void
}) {
  const theme = useApps((s) => s.theme)
  const filter = useApps((s) => s.homeFilter)
  const setFilter = useApps((s) => s.setHomeFilter)
  // settings.json can be edited by hand: anything but 'list' is the cards.
  const view = useApps((s) => (s.settings?.homeView === 'list' ? 'list' : 'grid'))
  const updateSettings = useApps((s) => s.updateSettings)

  const counts = useMemo(() => homeCounts(apps), [apps])
  const installedOrder = useApps((s) => s.settings?.installedOrder)
  const sections = useMemo(() => arrangeHome(apps, filter, query, installedOrder), [apps, filter, query, installedOrder])
  const shown = sections.reduce((n, s) => n + s.views.length, 0)
  const searching = !!query.trim()
  const Item = view === 'list' ? AppRow : AppCard

  // Re-arranging the installed tiles: the one being dragged and the gap (0..n) it would drop into.
  const [dragId, setDragId] = useState<string | null>(null)
  const [gap, setGap] = useState<number | null>(null)
  const refocus = useRef<string | null>(null)

  const saveMove = (shownIds: string[], id: string, toIndex: number): void => {
    if (shownIds.indexOf(id) === toIndex) return
    void act(() => updateSettings({ installedOrder: moveInOrder(installedOrder, shownIds, id, toIndex) }))
  }
  // An empty list clears it (the main process drops an empty installedOrder; undefined would not survive IPC).
  const resetOrder = (): void => void act(() => updateSettings({ installedOrder: [] }))
  // Moving a tile re-inserts its element, which drops the focus: put it back once the new order is drawn.
  useEffect(() => {
    const id = refocus.current
    if (!id) return
    refocus.current = null
    document.querySelector<HTMLElement>(`[data-app-id="${CSS.escape(id)}"]`)?.focus()
  }, [sections])

  return (
    <div className="mx-auto max-w-[1280px] px-8 pt-5 pb-16">
      <section className="relative mb-5 overflow-hidden rounded-2xl border border-border">
        <div className="hero absolute inset-0" />
        <div className="hero-slashes absolute inset-0" />
        <div className="relative flex items-center justify-between gap-6 px-7 py-5">
          <div className="flex min-w-0 flex-1 items-center gap-6">
            <img src={LOGOS[theme].logo} alt="RulZ" className="h-11 shrink-0 drop-shadow-[0_8px_30px_rgb(237_0_13/0.35)]" draggable={false} />
            <div className="min-w-0">
              <div className="font-[family-name:var(--font-display)] text-[12px] font-semibold tracking-[0.42em] text-accent-text uppercase">
                AI Tools Suite
              </div>
              <p className="mt-1 text-[14px] leading-snug text-muted">
                Local AI and media tools for Windows. Install, update and launch every app from one place.
              </p>
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2.5">
            <PhoneAppButton appId="coding-agent-android" />
            <PhoneAppButton appId="writer-android" />
          </div>
        </div>
      </section>

      <div className="mb-6 flex flex-wrap items-center gap-2">
        <Chip active={filter === 'all'} count={counts.all} onClick={() => setFilter('all')}>
          All
        </Chip>
        {counts.categories.map((c) => (
          <Chip key={c.name} active={filter === `cat:${c.name}`} count={c.count} onClick={() => setFilter(`cat:${c.name}`)}>
            {c.name}
          </Chip>
        ))}
        <span className="mx-1.5 h-5 w-px bg-border-strong" aria-hidden />
        <Chip active={filter === 'installed'} count={counts.installed} onClick={() => setFilter('installed')}>
          Installed
        </Chip>
        {/* Stays while it is the picked filter, so the last update finishing doesn't leave no chip selected. */}
        {(counts.updates > 0 || filter === 'updates') && (
          <Chip active={filter === 'updates'} count={counts.updates} highlight onClick={() => setFilter('updates')}>
            Updates
          </Chip>
        )}

        <div className="flex-1" />

        <div className="flex rounded-lg border border-border bg-panel p-0.5" role="group" aria-label="How the apps are listed">
          <ViewButton label="Cards" active={view === 'grid'} onClick={() => void act(() => updateSettings({ homeView: 'grid' }))}>
            <LayoutGrid className="size-4" />
          </ViewButton>
          <ViewButton label="List" active={view === 'list'} onClick={() => void act(() => updateSettings({ homeView: 'list' }))}>
            <List className="size-4" />
          </ViewButton>
        </div>
      </div>

      {searching && shown > 0 && (
        <div className="mb-4 flex items-center gap-3 text-[13px] text-muted">
          <span>
            {shown} {shown === 1 ? 'app matches' : 'apps match'} <span className="text-text">“{query.trim()}”</span>
          </span>
          <button
            className="inline-flex cursor-pointer items-center gap-1 text-link hover:underline"
            onClick={() => onQuery('')}
          >
            <X className="size-3.5" /> Clear search
          </button>
        </div>
      )}

      {sections.length ? (
        <div className="space-y-8">
          {sections.map((s) => (
            <section key={s.id} aria-label={s.title ?? undefined}>
              {s.title && (
                <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <h2 className="flex items-center gap-2 font-[family-name:var(--font-display)] text-[13px] font-semibold tracking-[0.16em] text-text uppercase">
                    <span className="h-3 w-1 -skew-x-[20deg] bg-accent" /> {s.title}
                  </h2>
                  <span className="text-[13px] text-faint">{s.blurb ?? `${s.views.length} ${s.views.length === 1 ? 'app' : 'apps'}`}</span>
                  {s.id === 'installed' && !!installedOrder?.length && <ResetOrder onClick={resetOrder} />}
                </div>
              )}
              {!s.title && s.id === 'installed' && !!installedOrder?.length && (
                <div className="mb-3">
                  <ResetOrder onClick={resetOrder} />
                </div>
              )}
              <div
                className={
                  view === 'list' ? 'flex flex-col gap-2' : 'grid grid-cols-[repeat(auto-fill,minmax(264px,1fr))] gap-4'
                }
              >
                {s.views.map((v, i) => {
                  const id = v.app.id
                  const item = (
                    <Item
                      key={id}
                      view={v}
                      // A category heading already names it.
                      showCategory={!s.id.startsWith('cat:')}
                      onOpen={() => onOpen(id)}
                      onInstall={() => onInstall(id)}
                    />
                  )
                  // Only the installed tiles, and not in search results, can be moved.
                  if (s.id !== 'installed' || searching || s.views.length < 2) return item
                  const ids = s.views.map((x) => x.app.id)
                  const stop = (): void => {
                    setDragId(null)
                    setGap(null)
                  }
                  return (
                    <Movable
                      key={id}
                      list={view === 'list'}
                      dragging={dragId === id}
                      bar={dragId && dragId !== id ? (gap === i ? 'before' : gap === i + 1 ? 'after' : null) : null}
                      onDragStart={() => setDragId(id)}
                      onGap={(g) => setGap(g === null ? null : i + g)}
                      onDrop={() => {
                        // The gap counts the dragged tile; the move wants the place it ends up in.
                        const from = dragId ? ids.indexOf(dragId) : -1
                        if (dragId && gap !== null && from >= 0) saveMove(ids, dragId, gap > from ? gap - 1 : gap)
                        stop()
                      }}
                      onDragEnd={stop}
                      onMove={(by) => {
                        const to = i + by
                        if (to < 0 || to >= ids.length) return
                        refocus.current = id
                        saveMove(ids, id, to)
                      }}
                    >
                      {item}
                    </Movable>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-20 text-faint">
          <SearchX className="size-8" />
          <p className="text-sm">{emptyText(filter, searching)}</p>
          {(searching || filter !== 'all') && (
            <button
              className="cursor-pointer text-[13px] text-link hover:underline"
              onClick={() => {
                onQuery('')
                setFilter('all')
              }}
            >
              Show all apps
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function ResetOrder({ onClick }: { onClick: () => void }) {
  return (
    <button className="cursor-pointer text-[13px] text-link hover:underline" onClick={onClick}>
      Reset order
    </button>
  )
}

/**
 * Wraps an installed tile so it can be re-arranged. The whole tile is the drag source: a native HTML5 drag starts
 * only once the pointer has moved a few pixels, so a plain click still opens the app and a finished drag fires no
 * click; buttons inside the tile keep working. The grip is a hint shown on hover and focus, not a separate target,
 * so there is no small handle to miss. Alt + arrow keys move the focused tile, for people who don't drag.
 */
function Movable({
  list,
  dragging,
  bar,
  onDragStart,
  onGap,
  onDrop,
  onDragEnd,
  onMove,
  children
}: {
  list: boolean
  dragging: boolean
  bar: 'before' | 'after' | null
  onDragStart: () => void
  /** The gap the pointer is over: 0 before this tile, 1 after it; null when it left. */
  onGap: (gap: 0 | 1 | null) => void
  onDrop: () => void
  onDragEnd: () => void
  onMove: (by: -1 | 1) => void
  children: ReactNode
}) {
  const over = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    const r = e.currentTarget.getBoundingClientRect()
    const after = list ? e.clientY > r.top + r.height / 2 : e.clientX > r.left + r.width / 2
    onGap(after ? 1 : 0)
  }
  const key = (e: KeyboardEvent<HTMLDivElement>): void => {
    if (!e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    const by = e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : 0
    if (!by) return
    e.preventDefault()
    onMove(by)
  }
  return (
    <div
      draggable
      aria-keyshortcuts="Alt+ArrowLeft Alt+ArrowRight"
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = 'move'
        e.dataTransfer.setData('text/plain', '')
        onDragStart()
      }}
      onDragOver={over}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) onGap(null)
      }}
      onDrop={(e) => {
        e.preventDefault()
        onDrop()
      }}
      onDragEnd={onDragEnd}
      onKeyDown={key}
      className={cn('group/move relative flex flex-col [&>[data-app-id]]:flex-1', dragging && 'opacity-50')}
    >
      {children}
      <span
        role="img"
        title="Drag to re-arrange (Alt+arrow keys)"
        aria-label="Drag to re-arrange (Alt+arrow keys)"
        className={cn(
          'absolute z-10 flex size-6 items-center justify-center rounded-md text-faint opacity-0 group-focus-within/move:opacity-100 group-hover/move:opacity-100 hover:text-text',
          list ? 'top-1/2 -left-6 -translate-y-1/2' : 'top-2 left-1/2 -translate-x-1/2'
        )}
      >
        <GripVertical className="size-4" />
      </span>
      {bar && (
        <span
          aria-hidden
          className={cn(
            'pointer-events-none absolute z-20 rounded-full bg-accent',
            list ? 'right-0 left-0 h-0.5' : 'top-0 bottom-0 w-0.5',
            list ? (bar === 'before' ? '-top-[5px]' : '-bottom-[5px]') : bar === 'before' ? '-left-[9px]' : '-right-[9px]'
          )}
        />
      )}
    </div>
  )
}

function emptyText(filter: HomeFilter, searching: boolean): string {
  if (searching) return filter === 'all' ? 'No apps match your search.' : 'No apps match your search in this group.'
  if (filter === 'installed') return 'Nothing is installed yet. Pick an app and press Install.'
  if (filter === 'updates') return 'Every installed app is up to date.'
  return 'No apps here.'
}

function Chip({
  active,
  count,
  highlight,
  onClick,
  children
}: {
  active: boolean
  count: number
  /** Drawn in the accent color while it isn't the picked one (the Updates chip). */
  highlight?: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'inline-flex h-8 cursor-pointer items-center gap-2 rounded-full border pr-2.5 pl-3.5 text-[13px] transition-colors',
        active
          ? 'border-accent bg-accent text-white'
          : highlight
            ? 'border-accent/50 bg-accent-soft text-accent-text hover:border-accent'
            : 'border-border bg-panel text-muted hover:border-border-strong hover:text-text'
      )}
    >
      {children}
      <span
        className={cn(
          'min-w-5 rounded-full px-1.5 text-center text-[11.5px] leading-5 tabular-nums',
          active ? 'bg-white/20' : 'bg-subtle'
        )}
      >
        {count}
      </span>
    </button>
  )
}

function ViewButton({ label, active, onClick, children }: { label: string; active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      title={`${label} view`}
      aria-label={`${label} view`}
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        'flex h-7 w-8 cursor-pointer items-center justify-center rounded-md transition-colors',
        active ? 'bg-hover text-text' : 'text-faint hover:text-text'
      )}
    >
      {children}
    </button>
  )
}
