import { LayoutGrid, List, SearchX, X } from 'lucide-react'
import { useMemo, type ReactNode } from 'react'
import { arrangeHome, homeCounts, type HomeFilter } from '@shared/arrange'
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
  const sections = useMemo(() => arrangeHome(apps, filter, query), [apps, filter, query])
  const shown = sections.reduce((n, s) => n + s.views.length, 0)
  const searching = !!query.trim()
  const Item = view === 'list' ? AppRow : AppCard

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
          <PhoneAppButton />
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
                </div>
              )}
              <div
                className={
                  view === 'list' ? 'flex flex-col gap-2' : 'grid grid-cols-[repeat(auto-fill,minmax(264px,1fr))] gap-4'
                }
              >
                {s.views.map((v) => (
                  <Item
                    key={v.app.id}
                    view={v}
                    // A category heading already names it.
                    showCategory={!s.id.startsWith('cat:')}
                    onOpen={() => onOpen(v.app.id)}
                    onInstall={() => onInstall(v.app.id)}
                  />
                ))}
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
