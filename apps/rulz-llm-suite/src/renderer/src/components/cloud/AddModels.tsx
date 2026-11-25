import { Plus, RefreshCw, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { CloudModelDef } from '@shared/cloud'
import { EMPTY_FILTER, badgesFor, capRows, catalogTime, filterCatalog, priceShort, type CatalogFilter } from '../../lib/cloud'
import { cn, formatTokens } from '../../lib/format'
import { isBusy, useCloud } from '../../stores/cloud'
import { Badge, Button, Input } from '../ui'

const CHIPS: { key: 'free' | 'images' | 'tools'; label: string }[] = [
  { key: 'free', label: 'Free' },
  { key: 'images', label: 'Images' },
  { key: 'tools', label: 'Tools' }
]

/** OpenRouter's list of usable models. It is read without a key, so it can be browsed before one is saved. */
export function AddModels({ added }: { added: CloudModelDef[] }) {
  const catalog = useCloud((s) => s.catalog)
  const loading = useCloud((s) => s.catalogLoading)
  const error = useCloud((s) => s.catalogError)
  const loadCatalog = useCloud((s) => s.loadCatalog)
  const [filter, setFilter] = useState<CatalogFilter>(EMPTY_FILTER)

  useEffect(() => {
    if (!useCloud.getState().catalog) void loadCatalog(false)
  }, [loadCatalog])

  const addedSlugs = useMemo(() => new Set(added.map((m) => m.slug)), [added])
  const rows = useMemo(() => (catalog ? filterCatalog(catalog.models, addedSlugs, filter) : []), [catalog, addedSlugs, filter])
  const { shown, hidden } = capRows(rows)

  return (
    <section className="rounded-xl border border-border bg-panel p-5" aria-label="Add models">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">Add models</h2>
          <p className="mt-0.5 text-xs text-muted">
            Prices are dollars per million tokens, input / output. A model you add can be used right away from the model picker.
          </p>
        </div>
        <Button size="sm" disabled={loading} onClick={() => void loadCatalog(true)}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : undefined} /> Refresh
        </Button>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-56 flex-1">
          <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-faint" />
          <Input className="pl-8" placeholder="Search by name or slug" aria-label="Search OpenRouter models" value={filter.query} onChange={(e) => setFilter({ ...filter, query: e.target.value })} />
        </div>
        {CHIPS.map((c) => (
          <button
            key={c.key}
            aria-pressed={filter[c.key]}
            onClick={() => setFilter({ ...filter, [c.key]: !filter[c.key] })}
            className={cn(
              'h-8 rounded-full border px-3 text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none',
              filter[c.key] ? 'border-accent/60 bg-accent-soft text-accent-text' : 'border-border bg-panel-2 text-muted hover:text-text'
            )}
          >
            {c.label}
          </button>
        ))}
      </div>

      {error && <p className="mt-3 text-xs text-danger-text">Could not read the list: {error}</p>}
      {!catalog && !error && <p className="mt-3 text-xs text-muted">Reading OpenRouter's list…</p>}

      {catalog && (
        <>
          <div className="mt-2 text-[11.5px] text-faint tabular-nums">
            List from {catalogTime(catalog.at)} · {rows.length} model{rows.length === 1 ? '' : 's'} not added yet
          </div>
          <ul className="mt-2 divide-y divide-border rounded-lg border border-border">
            {shown.map((m) => (
              <li key={m.slug} className="flex items-center gap-3 px-3 py-2">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="truncate text-[13px] font-medium">{m.name}</span>
                    {badgesFor(m).map((b) => (
                      <Badge key={b.label} tone={b.tone}>
                        {b.label}
                      </Badge>
                    ))}
                  </div>
                  <div className="truncate font-mono text-[11px] text-faint">{m.slug}</div>
                </div>
                <div className="hidden w-20 shrink-0 text-right text-xs text-muted tabular-nums sm:block">{formatTokens(m.contextLength)} context</div>
                <div className="w-28 shrink-0 text-right text-xs tabular-nums">{priceShort(m)}</div>
                <AddButton slug={m.slug} name={m.name} />
              </li>
            ))}
            {shown.length === 0 && <li className="px-3 py-4 text-center text-xs text-muted">No model matches. Change the search or switch a filter off.</li>}
          </ul>
          {hidden > 0 && <p className="mt-2 text-xs text-muted tabular-nums">{hidden} more not shown. Refine the search to see them.</p>}
        </>
      )}
    </section>
  )
}

function AddButton({ slug, name }: { slug: string; name: string }) {
  const run = useCloud((s) => s.run)
  const busy = useCloud((s) => isBusy(s, 'cloud:addModel', slug))
  return (
    <Button size="sm" disabled={busy} aria-label={`Add ${name}`} onClick={() => void run('cloud:addModel', slug)}>
      <Plus size={14} /> Add
    </Button>
  )
}
