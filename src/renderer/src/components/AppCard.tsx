import { AlertTriangle, ArrowUpCircle, CheckCircle2, FolderSearch, Wrench } from 'lucide-react'
import type { KeyboardEvent } from 'react'
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
  if (view.missing)
    return (
      <Badge tone="warn">
        <AlertTriangle className="size-3" /> Files missing
      </Badge>
    )
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
  // Full-size screenshots: decoded off the main thread, and only once the card is near the screen.
  if (shot) return <img src={shot} alt="" draggable={false} loading="lazy" decoding="async" className={className} />
  return (
    <div className={`${className} hero flex items-center justify-center`}>
      <img src={view.icon} alt="" draggable={false} className="size-20 opacity-90 drop-shadow-[0_10px_30px_rgb(237_0_13/0.4)]" />
    </div>
  )
}

/** The installed version, or the newest release of an app that isn't installed. */
function versionText(view: AppView): string {
  if (!view.installed) return view.latest?.label ?? ''
  if (!view.installed.version) return 'Installed'
  return view.installed.versionLabel ?? formatVersion(view.installed.version)
}

/** Enter and Space open the card, as a click does. Keys pressed on a button inside it are that button's. */
function openOnKey(onOpen: () => void) {
  return (e: KeyboardEvent<HTMLDivElement>): void => {
    if (e.target !== e.currentTarget) return
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onOpen()
    }
  }
}

interface ItemProps {
  view: AppView
  onOpen: () => void
  onInstall: () => void
  /** Hides the category label where a section heading already names it. */
  showCategory?: boolean
}

/** The grid view's card: a screenshot, the name and tagline, the version and the main button. */
export function AppCard({ view, onOpen, onInstall, showCategory = true }: ItemProps) {
  const { app } = view
  return (
    <div
      role="button"
      tabIndex={0}
      data-app-id={app.id}
      onClick={onOpen}
      onKeyDown={openOnKey(onOpen)}
      className="card-glow group flex cursor-pointer flex-col overflow-hidden rounded-xl border border-border bg-panel transition-[box-shadow,transform,border-color] duration-200 hover:-translate-y-0.5 hover:border-transparent"
    >
      <div className="relative aspect-[16/9] overflow-hidden bg-media">
        <Cover
          view={view}
          className="h-full w-full object-cover object-top transition-transform duration-500 group-hover:scale-[1.03]"
        />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-panel via-panel/10 to-transparent" />
        {showCategory && (
          <div className="absolute top-2.5 left-2.5">
            <Badge className="border-white/10 bg-black/60 text-white/85 backdrop-blur">{app.category}</Badge>
          </div>
        )}
        {/* A solid backing: the badge's own tint is see-through, and the screenshot behind it is busy. */}
        <div className="absolute top-2.5 right-2.5 rounded-full bg-panel/90 backdrop-blur">
          <StatusBadge view={view} />
        </div>
      </div>

      <div className="relative flex flex-1 flex-col px-3.5 pb-3.5">
        <img
          src={view.icon}
          alt=""
          draggable={false}
          className="-mt-6 mb-2 size-11 rounded-xl border border-border bg-panel-2 shadow-[0_8px_20px_-6px_var(--shadow)]"
        />
        <h3 className="line-clamp-2 font-[family-name:var(--font-display)] text-[15px] leading-snug font-semibold text-text">
          {app.name}
        </h3>
        <p className="mt-1 line-clamp-2 text-[13px] leading-[1.5] text-muted" title={app.tagline}>
          {app.tagline}
        </p>
        <div className="mt-auto flex items-center justify-between gap-3 pt-3.5">
          <span className="truncate text-xs text-faint">{versionText(view)}</span>
          <AppActions view={view} onInstall={onInstall} compact />
        </div>
      </div>
    </div>
  )
}

/** The list view's row: the same facts on one line, so every app fits on one screen. */
export function AppRow({ view, onOpen, onInstall, showCategory = true }: ItemProps) {
  const { app } = view
  return (
    <div
      role="button"
      tabIndex={0}
      data-app-id={app.id}
      onClick={onOpen}
      onKeyDown={openOnKey(onOpen)}
      className="group flex cursor-pointer items-center gap-4 rounded-xl border border-border bg-panel px-3.5 py-2.5 transition-colors hover:border-accent/60 hover:bg-panel-2"
    >
      <img src={view.icon} alt="" draggable={false} className="size-10 shrink-0 rounded-lg border border-border bg-panel-2" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2.5">
          <h3 className="truncate font-[family-name:var(--font-display)] text-[15px] leading-snug font-semibold text-text">
            {app.name}
          </h3>
          <StatusBadge view={view} />
        </div>
        <p className="mt-0.5 truncate text-[13px] text-muted" title={app.tagline}>
          {app.tagline}
        </p>
      </div>
      {showCategory && <Badge className="max-[1100px]:hidden">{app.category}</Badge>}
      <span className="w-[128px] shrink-0 truncate text-right text-xs text-faint">{versionText(view)}</span>
      <div className="flex w-[210px] shrink-0 justify-end">
        <AppActions view={view} onInstall={onInstall} compact />
      </div>
    </div>
  )
}
