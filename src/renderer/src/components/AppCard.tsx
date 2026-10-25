import { ArrowUpCircle, CheckCircle2, FolderSearch, Wrench } from 'lucide-react'
import type { AppView } from '@shared/types'
import { formatVersion } from '../lib/format'
import { AppActions } from './AppActions'
import { Badge } from './ui'

export function StatusBadge({ view }: { view: AppView }) {
  const i = view.installed
  if (view.running) {
    return (
      <Badge tone="ok">
        <span className="size-1.5 animate-pulse rounded-full bg-ok" /> Running
      </Badge>
    )
  }
  if (!i) return null
  if (i.needsSetup)
    return (
      <Badge tone="warn">
        <Wrench className="size-3" /> Setup incomplete
      </Badge>
    )
  if (view.updateAvailable)
    return (
      <Badge tone="accent">
        <ArrowUpCircle className="size-3" /> Update {view.latest ? formatVersion(view.latest.version) : ''}
      </Badge>
    )
  if (!i.managed)
    return (
      <Badge>
        <FolderSearch className="size-3" /> Found on this PC
      </Badge>
    )
  return (
    <Badge tone="ok">
      <CheckCircle2 className="size-3" /> Installed
    </Badge>
  )
}

/** Cover image for a card or header: the first screenshot, or the icon on a red-tinted backdrop. */
export function Cover({ view, className }: { view: AppView; className?: string }) {
  const shot = view.screenshots[0]
  if (shot) return <img src={shot} alt="" draggable={false} className={className} />
  return (
    <div className={`${className} hero flex items-center justify-center`}>
      <img src={view.icon} alt="" draggable={false} className="size-24 opacity-90 drop-shadow-[0_10px_30px_rgb(237_0_13/0.4)]" />
    </div>
  )
}

export function AppCard({ view, onOpen, onInstall }: { view: AppView; onOpen: () => void; onInstall: () => void }) {
  const { app } = view
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onOpen()
        }
      }}
      className="card-glow group flex cursor-pointer flex-col overflow-hidden rounded-xl border border-border bg-panel transition-[box-shadow,transform,border-color] duration-200 hover:-translate-y-0.5 hover:border-transparent"
    >
      <div className="relative aspect-[16/10] overflow-hidden bg-black">
        <Cover
          view={view}
          className="h-full w-full object-cover object-top transition-transform duration-500 group-hover:scale-[1.03]"
        />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-panel via-panel/10 to-transparent" />
        <div className="absolute top-3 left-3">
          <Badge className="border-white/10 bg-black/60 text-[#d4d4d8] backdrop-blur">{app.category}</Badge>
        </div>
        <div className="absolute top-3 right-3">
          <StatusBadge view={view} />
        </div>
      </div>

      <div className="relative flex flex-1 flex-col px-4 pb-4">
        <img
          src={view.icon}
          alt=""
          draggable={false}
          className="-mt-7 mb-2 size-12 rounded-xl border border-border bg-panel-2 shadow-[0_8px_20px_-6px_rgb(0_0_0/0.9)]"
        />
        <h3 className="font-[family-name:var(--font-display)] text-[16px] leading-snug font-semibold text-text">{app.name}</h3>
        <p className="mt-1 line-clamp-2 text-[13px] leading-relaxed text-muted">{app.tagline}</p>
        <div className="mt-auto flex items-center justify-between gap-3 pt-4">
          <span className="truncate text-xs text-faint">
            {view.installed
              ? view.installed.version
                ? formatVersion(view.installed.version)
                : 'Installed'
              : (view.latest?.label ?? '')}
          </span>
          <AppActions view={view} onInstall={onInstall} compact />
        </div>
      </div>
    </div>
  )
}
