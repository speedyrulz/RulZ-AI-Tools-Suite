import { SearchX } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { AppView } from '@shared/types'
import { AppCard } from '../components/AppCard'
import { cn } from '../lib/format'
import { LOGOS } from '../lib/theme'
import { useApps } from '../stores/apps'

const INSTALLED = 'Installed'

export function HomePage({
  apps,
  query,
  onOpen,
  onInstall
}: {
  apps: AppView[]
  query: string
  onOpen: (id: string) => void
  onInstall: (id: string) => void
}) {
  const [filter, setFilter] = useState<string>('All')
  const theme = useApps((s) => s.theme)
  const categories = useMemo(() => ['All', ...new Set(apps.map((a) => a.app.category)), INSTALLED], [apps])

  const installedCount = apps.filter((a) => a.installed).length
  const updates = apps.filter((a) => a.updateAvailable).length

  const q = query.trim().toLowerCase()
  const shown = apps.filter((a) => {
    if (filter === INSTALLED ? !a.installed : filter !== 'All' && a.app.category !== filter) return false
    if (!q) return true
    const { app } = a
    return [app.name, app.tagline, app.description, app.category, ...app.highlights].join(' ').toLowerCase().includes(q)
  })

  return (
    <div className="mx-auto max-w-[1280px] px-8 pt-6 pb-16">
      <section className="relative mb-8 overflow-hidden rounded-2xl border border-border">
        <div className="hero absolute inset-0" />
        <div className="hero-slashes absolute inset-0" />
        <div className="relative flex flex-wrap items-end justify-between gap-6 px-9 py-9">
          <div>
            <img src={LOGOS[theme].logo} alt="RulZ" className="h-16 drop-shadow-[0_8px_30px_rgb(237_0_13/0.35)]" draggable={false} />
            <div className="mt-4 font-[family-name:var(--font-display)] text-[13px] font-semibold tracking-[0.42em] text-accent-text uppercase">
              AI Tools Suite
            </div>
            <p className="mt-2 max-w-[520px] text-[15px] leading-relaxed text-muted">
              Local AI and media tools for Windows. Install, update and launch every app from one place.
            </p>
          </div>
          <div className="flex gap-3">
            <Stat value={apps.length} label="Apps" />
            <Stat value={installedCount} label="Installed" />
            <Stat value={updates} label={updates === 1 ? 'Update' : 'Updates'} highlight={updates > 0} />
          </div>
        </div>
      </section>

      <div className="mb-5 flex flex-wrap items-center gap-2">
        {categories.map((c) => (
          <button
            key={c}
            onClick={() => setFilter(c)}
            className={cn(
              'h-8 cursor-pointer rounded-full border px-4 text-[13px] transition-colors',
              filter === c
                ? 'border-accent bg-accent text-white'
                : 'border-border bg-panel text-muted hover:border-border-strong hover:text-text'
            )}
          >
            {c}
          </button>
        ))}
      </div>

      {shown.length ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-5">
          {shown.map((v) => (
            <AppCard key={v.app.id} view={v} onOpen={() => onOpen(v.app.id)} onInstall={() => onInstall(v.app.id)} />
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2 py-20 text-faint">
          <SearchX className="size-8" />
          <p className="text-sm">No apps match.</p>
        </div>
      )}
    </div>
  )
}

function Stat({ value, label, highlight }: { value: number; label: string; highlight?: boolean }) {
  return (
    <div
      className={cn(
        'min-w-[96px] rounded-xl border px-4 py-3 backdrop-blur',
        highlight ? 'border-accent/50 bg-accent-soft' : 'border-subtle-border bg-sunken'
      )}
    >
      <div className={cn('font-[family-name:var(--font-display)] text-2xl font-bold tabular-nums', highlight && 'text-accent-text')}>
        {value}
      </div>
      <div className="text-xs text-faint">{label}</div>
    </div>
  )
}
