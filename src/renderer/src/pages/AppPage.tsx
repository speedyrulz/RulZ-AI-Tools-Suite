import {
  AlertCircle,
  ArrowLeft,
  ArrowUp,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  FileText,
  FolderOpen,
  FolderSearch,
  Github,
  Info,
  MoreHorizontal,
  RefreshCw,
  Trash2,
  Wrench,
  X
} from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { githubUrl } from '@shared/catalog'
import { refFor, requirementRefLabel } from '@shared/requirements'
import type { AppView } from '@shared/types'
import { AppActions, jobDetail, useJob } from '../components/AppActions'
import { StatusBadge } from '../components/AppCard'
import { Gallery } from '../components/Gallery'
import { LogDialog } from '../components/LogDialog'
import { PhoneAppQr } from '../components/PhoneAppQr'
import { Readme, type ReadmeHeading } from '../components/Readme'
import { RequirementList } from '../components/Requirements'
import { Badge, Button, Dialog, IconButton, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Switch } from '../components/ui'
import { invoke } from '../lib/api'
import { cn, formatBytes, formatDate, formatVersion } from '../lib/format'
import { act, isActive, useApps } from '../stores/apps'

/** The app's launch switches (catalog launch.options), saved in the launcher's settings. */
function LaunchOptions({ view }: { view: AppView }) {
  const id = view.app.id
  const saved = useApps((s) => s.settings?.launchOptions?.[id])
  const enabled = Array.isArray(saved) ? saved : []
  const [pending, setPending] = useState<string | null>(null)
  const toggle = (option: string, on: boolean): void => {
    if (pending) return
    setPending(option)
    void act(async () => {
      try {
        useApps.setState({ settings: await invoke('settings:launchOption', id, option, on) })
      } finally {
        setPending(null)
      }
    })
  }
  return (
    <div className="relative flex flex-col gap-3 border-t border-border px-7 py-3">
      {view.app.launch.options!.map((option) => (
        <div key={option.id} className="flex items-center justify-between gap-4">
          <div
            className="min-w-0 cursor-pointer"
            onClick={() => toggle(option.id, !enabled.includes(option.id))}
          >
            <div className="text-sm">{option.label}</div>
            <div className="text-xs text-faint">
              {option.hint ? `${option.hint} ` : ''}Applies the next time you press Launch.
            </div>
          </div>
          <Switch
            checked={enabled.includes(option.id)}
            disabled={pending !== null}
            onChange={(v) => toggle(option.id, v)}
            aria-label={option.label}
          />
        </div>
      ))}
    </div>
  )
}

/** Another app's page, one step away (the Previous / Next buttons). */
interface Neighbor {
  name: string
  go: () => void
}

/** How far below the window's top a README heading counts as the section being read (the two bars, and a little more). */
const READING_LINE = 150

/** A README with fewer sections than this gets no "On this page" list. */
const MIN_SECTIONS = 3

export function AppPage({
  view,
  onBack,
  onInstall,
  previous,
  next,
  focusArrow
}: {
  view: AppView
  onBack: () => void
  onInstall: () => void
  previous: Neighbor | null
  next: Neighbor | null
  /** The arrow that led here, which gets the keyboard focus (each app's page is a new one, so it would be lost). */
  focusArrow: 'previous' | 'next' | null
}) {
  const { app, installed } = view
  const job = useJob(view)
  const dismissJob = useApps((s) => s.dismissJob)
  const refresh = useApps((s) => s.refresh)
  const [logOpen, setLogOpen] = useState(false)
  const [confirmRemove, setConfirmRemove] = useState(false)
  const lastLine = useApps((s) => s.logs[app.id]?.at(-1) ?? '')
  const busy = isActive(job)

  const header = useRef<HTMLElement>(null)
  const readme = useRef<HTMLDivElement>(null)
  const sections = useRef<HTMLElement>(null)
  const previousArrow = useRef<HTMLButtonElement>(null)
  const nextArrow = useRef<HTMLButtonElement>(null)
  /** The header has scrolled out of view: the bar at the top shows the app's name and buttons instead. */
  const [pinned, setPinned] = useState(false)
  const [headings, setHeadings] = useState<ReadmeHeading[]>([])
  const [reading, setReading] = useState<string | null>(null)

  useEffect(() => {
    const el = header.current
    if (!el) return
    // 96px: the window's top bar and this page's bar, which cover the header before it leaves the window.
    // The newest entry counts: a fast jump can queue two crossings before this runs.
    const seen = new IntersectionObserver((entries) => setPinned(!entries[entries.length - 1].isIntersecting), {
      rootMargin: '-96px 0px 0px 0px'
    })
    seen.observe(el)
    return () => seen.disconnect()
  }, [])

  useEffect(() => {
    const el = focusArrow === 'previous' ? previousArrow.current : focusArrow === 'next' ? nextArrow.current : null
    el?.focus({ preventScroll: true })
    // Only when the page opens: the arrows stay put after that.
  }, [])

  // Marks the README section being read in the "On this page" list.
  useEffect(() => {
    setReading(null)
    const scroller = header.current?.closest('[data-scroller]')
    if (!scroller || headings.length < MIN_SECTIONS) return
    const els = [...(readme.current?.querySelectorAll<HTMLElement>('.md h2[id]') ?? [])]
    let frame = 0
    const update = (): void => {
      frame = 0
      let id: string | null = null
      for (const h of els) {
        if (h.getBoundingClientRect().top > READING_LINE) break
        id = h.id
      }
      // At the very bottom the last sections can't reach the line any more: the last one is the one in view.
      if (els.length && scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 2) id = els[els.length - 1].id
      setReading(id)
    }
    const onScroll = (): void => {
      if (!frame) frame = requestAnimationFrame(update)
    }
    scroller.addEventListener('scroll', onScroll, { passive: true })
    update()
    return () => {
      scroller.removeEventListener('scroll', onScroll)
      if (frame) cancelAnimationFrame(frame)
    }
  }, [headings])

  // Keeps the marked section in sight in a list longer than its box, without moving the page itself.
  useEffect(() => {
    const list = sections.current
    const mark = list?.querySelector<HTMLElement>('[aria-current]')
    if (!list || !mark) return
    // The list is the entries' offset parent (it is positioned), so offsetTop is measured from its top.
    const top = mark.offsetTop
    if (top < list.scrollTop) list.scrollTop = top
    else if (top + mark.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = top + mark.offsetHeight - list.clientHeight
  }, [reading])

  const toTop = (): void => header.current?.closest('[data-scroller]')?.scrollTo({ top: 0, behavior: 'smooth' })
  const toSection = (id: string): void =>
    readme.current?.querySelector(`.md [id="${CSS.escape(id)}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  return (
    <>
      {/* Stays at the top while the page scrolls: the way back, and the app's buttons once its header is out of view. */}
      <div
        className={cn(
          'sticky top-0 z-20 border-b bg-bg/90 backdrop-blur transition-colors',
          pinned ? 'border-border' : 'border-transparent'
        )}
      >
        <div className="mx-auto flex h-12 max-w-[1280px] items-center gap-3 px-8">
          <button
            onClick={onBack}
            title="Back to all apps (Esc)"
            className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 text-[13px] text-muted transition-colors hover:text-text"
          >
            <ArrowLeft className="size-4" /> All apps
          </button>
          {pinned && (
            <button
              onClick={toTop}
              title="Back to the top of the page"
              className="flex min-w-0 cursor-pointer items-center gap-2.5 border-l border-border pl-3"
            >
              <img src={view.icon} alt="" draggable={false} className="size-7 shrink-0 rounded-md border border-border bg-panel-2" />
              <span className="truncate font-[family-name:var(--font-display)] text-[14px] font-semibold">{app.name}</span>
              <StatusBadge view={view} />
            </button>
          )}
          <div className="flex-1" />
          {pinned && (
            <div className={cn('flex shrink-0 justify-end', busy && 'w-[280px]')}>
              <AppActions view={view} onInstall={onInstall} compact />
            </div>
          )}
          {previous && next && (
            <div className="flex shrink-0 items-center">
              <IconButton ref={previousArrow} label={`Previous app: ${previous.name}`} className="size-8" onClick={previous.go}>
                <ChevronLeft className="size-4" />
              </IconButton>
              <IconButton ref={nextArrow} label={`Next app: ${next.name}`} className="size-8" onClick={next.go}>
                <ChevronRight className="size-4" />
              </IconButton>
            </div>
          )}
        </div>
      </div>

      <div className="mx-auto max-w-[1280px] px-8 pt-1 pb-16">
      {/* Header */}
      <header ref={header} className="relative overflow-hidden rounded-2xl border border-border">
        <div className="hero absolute inset-0" />
        <div className="hero-slashes absolute inset-0" />
        <div className="relative flex flex-wrap items-center gap-6 px-7 py-6">
          <img
            src={view.icon}
            alt=""
            draggable={false}
            className="size-20 rounded-2xl border border-border-strong bg-panel-2 shadow-[0_12px_32px_-10px_rgb(237_0_13/0.5)]"
          />
          <div className="min-w-[260px] flex-1">
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <Badge>{app.category}</Badge>
              <StatusBadge view={view} />
            </div>
            <h1 className="font-[family-name:var(--font-display)] text-[28px] leading-tight font-bold tracking-tight">{app.name}</h1>
            <p className="mt-1 text-[15px] text-muted">{app.tagline}</p>
          </div>
          <div className="flex items-center gap-2">
            <AppActions view={view} onInstall={onInstall} />
            <Menu>
              <MenuTrigger asChild>
                <Button size="lg" className="px-3" aria-label="More actions">
                  <MoreHorizontal className="size-5" />
                </Button>
              </MenuTrigger>
              <MenuContent>
                {installed && (
                  <MenuItem icon={<FolderOpen />} onSelect={() => void act(() => invoke('apps:openFolder', app.id))}>
                    Open install folder
                  </MenuItem>
                )}
                {installed?.managed && (
                  <MenuItem
                    icon={<Wrench />}
                    disabled={busy}
                    onSelect={() => void act(() => invoke('apps:repair', app.id))}
                  >
                    {app.install.kind === 'source' ? 'Repair (run setup again)' : 'Reinstall'}
                  </MenuItem>
                )}
                {!installed && (
                  <MenuItem icon={<FolderSearch />} onSelect={() => void act(() => invoke('apps:locate', app.id))}>
                    Locate an existing install…
                  </MenuItem>
                )}
                <MenuItem icon={<RefreshCw />} onSelect={() => void refresh()}>
                  Check for updates
                </MenuItem>
                {app.repo && (
                  <MenuItem
                    icon={<Github />}
                    onSelect={() => void invoke('shell:openExternal', githubUrl(app)!)}
                  >
                    View on GitHub
                  </MenuItem>
                )}
                {(job || installed) && (
                  <MenuItem icon={<FileText />} onSelect={() => setLogOpen(true)}>
                    Show last install log
                  </MenuItem>
                )}
                {installed && (
                  <>
                    <MenuSeparator />
                    <MenuItem icon={<Trash2 />} danger disabled={busy} onSelect={() => setConfirmRemove(true)}>
                      {installed.managed ? 'Uninstall…' : 'Remove from launcher…'}
                    </MenuItem>
                  </>
                )}
              </MenuContent>
            </Menu>
          </div>
        </div>

        {installed && !!app.launch.options?.length && <LaunchOptions view={view} />}

        {job && (job.phase === 'error' || job.phase === 'cancelled') && (
          <div className="relative flex items-center gap-3 border-t border-accent/30 bg-banner px-7 py-3 text-[13px]">
            <AlertCircle className="size-4 shrink-0 text-danger" />
            <span className="min-w-0 flex-1 text-banner-text">
              {job.phase === 'cancelled' ? 'Cancelled.' : job.error}
            </span>
            {job.missing?.map((id) => {
              const ref = refFor(app.requirements, id)
              return (
                <Button key={id} size="sm" onClick={() => useApps.setState({ guide: ref })}>
                  <BookOpen className="size-3.5" /> How to install {requirementRefLabel(ref)}
                </Button>
              )
            })}
            <Button size="sm" variant="ghost" onClick={() => setLogOpen(true)}>
              <FileText className="size-3.5" /> Show log
            </Button>
            <Button size="sm" variant="ghost" className="px-2" title="Dismiss" onClick={() => dismissJob(app.id)}>
              <X className="size-4" />
            </Button>
          </div>
        )}
        {busy && job.phase === 'setup' && (
          <div className="relative flex items-center gap-3 border-t border-border bg-sunken px-7 py-2 text-xs text-muted">
            <span className="min-w-0 flex-1 truncate font-mono">{lastLine || jobDetail(job)}</span>
            <button className="cursor-pointer text-link hover:underline" onClick={() => setLogOpen(true)}>
              Show log
            </button>
          </div>
        )}
      </header>

      {/* Body */}
      <div className="mt-7 grid grid-cols-[minmax(0,1fr)_300px] gap-8">
        <main className="min-w-0">
          <p className="max-w-[860px] text-[15px] leading-relaxed text-body">{app.description}</p>
          <ul className="mt-4 mb-6 flex flex-wrap gap-1.5" aria-label="Highlights">
            {app.highlights.map((h) => (
              <li key={h}>
                <Badge className="h-7 bg-panel-2 px-3 text-[12.5px] text-text">{h}</Badge>
              </li>
            ))}
          </ul>
          <Gallery images={view.screenshots} name={app.name} />
          <h2 className="mt-10 mb-4 flex items-center gap-2 font-[family-name:var(--font-display)] text-xs font-semibold tracking-[0.16em] text-faint uppercase">
            <span className="h-3 w-1 -skew-x-[20deg] bg-accent" /> Read me
          </h2>
          <div ref={readme} className="rounded-xl border border-border bg-panel px-7 py-6">
            <Readme appId={app.id} onHeadings={setHeadings} />
          </div>
        </main>

        <aside className="space-y-4">
          <SideCard title="Details">
            <dl className="space-y-2.5 text-[13px]">
              {installed && (
                <Row label="Installed">
                  {installed.version
                    ? (installed.versionLabel ?? formatVersion(installed.version))
                    : installed.managed
                      ? 'yes'
                      : 'found on this PC'}
                </Row>
              )}
              <Row label="Latest">
                {view.latest ? (
                  <>
                    {view.latest.label}
                    <span className="text-faint"> · {formatDate(view.latest.date)}</span>
                  </>
                ) : (
                  <span className="text-faint">{view.latestError ?? 'Not checked yet'}</span>
                )}
              </Row>
              {view.latest?.downloadSize ? <Row label="Download">{formatBytes(view.latest.downloadSize)}</Row> : null}
              <Row label="Install type">
                {app.install.kind === 'zip' ? 'Portable app' : app.install.kind === 'installer' ? 'Setup program' : 'From source'}
              </Row>
              {installed && (
                <Row label="Location">
                  <button
                    className="cursor-pointer text-left break-all text-link hover:underline"
                    onClick={() => void act(() => invoke('apps:openFolder', app.id))}
                  >
                    {installed.dir}
                  </button>
                </Row>
              )}
              {app.repo && (
                <Row label="Source">
                  <button
                    className="cursor-pointer text-left break-all text-link hover:underline"
                    onClick={() => void invoke('shell:openExternal', githubUrl(app)!)}
                  >
                    {app.repo}/{app.path} <ExternalLink className="inline size-3 align-[-1px]" />
                  </button>
                </Row>
              )}
            </dl>
          </SideCard>

          {app.requirements.length > 0 && (
            <SideCard title="Requirements">
              <RequirementList refs={app.requirements} />
            </SideCard>
          )}

          {!!app.notes?.length && (
            <SideCard title="Good to know">
              {app.notes.map((n) => (
                <p key={n} className="flex gap-2 py-1 text-[13px] leading-relaxed text-muted">
                  <Info className="mt-0.5 size-3.5 shrink-0 text-faint" />
                  {n}
                </p>
              ))}
            </SideCard>
          )}

          {installed && (
            <SideCard title="Shortcuts">
              <ShortcutRow view={view} where="desktop" label="Desktop" busy={busy} />
              <ShortcutRow view={view} where="startMenu" label="Start Menu" busy={busy} />
            </SideCard>
          )}

          {app.id === 'coding-agent' && (
            <SideCard title="Phone app">
              <PhoneAppQr variant="card" />
            </SideCard>
          )}

          {headings.length >= MIN_SECTIONS && (
            <section className="sticky top-16 flex max-h-[calc(100vh-136px)] flex-col rounded-xl border border-border bg-panel py-3.5">
              <div className="mb-2 flex items-center justify-between px-4">
                <h3 className="text-[11px] font-semibold tracking-[0.14em] text-faint uppercase">On this page</h3>
                <button onClick={toTop} className="inline-flex cursor-pointer items-center gap-1 text-xs text-link hover:underline">
                  <ArrowUp className="size-3" /> Top
                </button>
              </div>
              <nav ref={sections} className="relative min-h-0 overflow-y-auto px-2" aria-label="README sections">
                {headings.map((h) => (
                  <button
                    key={h.id}
                    onClick={() => toSection(h.id)}
                    aria-current={reading === h.id ? 'location' : undefined}
                    className={cn(
                      'block w-full cursor-pointer truncate rounded-md border-l-2 px-2 py-1 text-left text-[13px] transition-colors hover:bg-hover hover:text-text',
                      reading === h.id ? 'border-accent text-text' : 'border-transparent text-muted'
                    )}
                    title={h.text}
                  >
                    {h.text}
                  </button>
                ))}
              </nav>
            </section>
          )}
        </aside>
      </div>

      {logOpen && <LogDialog appId={app.id} title={`${app.name}: install log`} onClose={() => setLogOpen(false)} />}
      {confirmRemove && installed && (
        <Dialog
          open
          onOpenChange={(o) => !o && setConfirmRemove(false)}
          title={installed.managed ? `Uninstall ${app.name}?` : `Remove ${app.name} from the launcher?`}
          footer={
            <>
              <Button variant="ghost" onClick={() => setConfirmRemove(false)}>
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={() => {
                  setConfirmRemove(false)
                  void act(() => invoke('apps:uninstall', app.id))
                }}
              >
                {installed.managed ? 'Uninstall' : 'Remove'}
              </Button>
            </>
          }
        >
          {installed.managed ? (
            <div className="space-y-3 text-[13px] text-muted">
              <p>
                {app.install.kind === 'installer' ? 'Its uninstaller runs, then the folder ' : 'The folder '}
                <span className="font-mono text-text">{installed.dir}</span> and everything in it moves to the Recycle Bin.
              </p>
              {app.install.kind === 'source' && (
                <p className="text-warn-text">That includes anything you saved inside it, such as downloaded models or outputs.</p>
              )}
              <p>Its shortcuts are removed. Settings the app keeps elsewhere stay.</p>
            </div>
          ) : (
            <p className="text-[13px] text-muted">
              The launcher forgets this install. Its files in <span className="font-mono text-text">{installed.dir}</span> stay
              where they are.
            </p>
          )}
        </Dialog>
      )}
      </div>
    </>
  )
}

function SideCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-panel px-4 py-3.5">
      <h3 className="mb-3 text-[11px] font-semibold tracking-[0.14em] text-faint uppercase">{title}</h3>
      {children}
    </section>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[92px_minmax(0,1fr)] gap-2">
      <dt className="text-faint">{label}</dt>
      <dd className="text-text">{children}</dd>
    </div>
  )
}

function ShortcutRow({
  view,
  where,
  label,
  busy
}: {
  view: AppView
  where: 'desktop' | 'startMenu'
  label: string
  /** A job is running: it sets the shortcuts it started with when it ends, so a change now would be undone. */
  busy: boolean
}) {
  const sc = view.installed!.shortcuts
  const on = !!sc[where]
  return (
    <label className="flex items-center justify-between py-1.5 text-[13px]">
      <span>{label}</span>
      <Switch
        checked={on}
        disabled={busy || !!view.installed?.needsSetup}
        onChange={(v) =>
          void act(() =>
            invoke('apps:setShortcuts', view.app.id, {
              desktop: where === 'desktop' ? v : !!sc.desktop,
              startMenu: where === 'startMenu' ? v : !!sc.startMenu
            })
          )
        }
      />
    </label>
  )
}
