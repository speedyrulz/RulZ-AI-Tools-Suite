import {
  AlertCircle,
  ArrowLeft,
  BookOpen,
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
import { useState, type ReactNode } from 'react'
import { REQUIREMENT_GUIDES } from '@shared/requirements'
import type { AppView } from '@shared/types'
import { AppActions, jobDetail, useJob } from '../components/AppActions'
import { StatusBadge } from '../components/AppCard'
import { Gallery } from '../components/Gallery'
import { LogDialog } from '../components/LogDialog'
import { Readme } from '../components/Readme'
import { RequirementList } from '../components/Requirements'
import { Badge, Button, Dialog, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Switch } from '../components/ui'
import { invoke } from '../lib/api'
import { formatBytes, formatDate, formatVersion } from '../lib/format'
import { act, isActive, useApps } from '../stores/apps'

export function AppPage({ view, onBack, onInstall }: { view: AppView; onBack: () => void; onInstall: () => void }) {
  const { app, installed } = view
  const job = useJob(view)
  const dismissJob = useApps((s) => s.dismissJob)
  const refresh = useApps((s) => s.refresh)
  const [logOpen, setLogOpen] = useState(false)
  const [confirmRemove, setConfirmRemove] = useState(false)
  const lastLine = useApps((s) => s.logs[app.id]?.at(-1) ?? '')
  const busy = isActive(job)

  return (
    <div className="mx-auto max-w-[1280px] px-8 pt-5 pb-16">
      <button
        onClick={onBack}
        className="mb-5 inline-flex cursor-pointer items-center gap-1.5 text-[13px] text-muted transition-colors hover:text-text"
      >
        <ArrowLeft className="size-4" /> All apps
      </button>

      {/* Header */}
      <header className="relative overflow-hidden rounded-2xl border border-border">
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
                    onSelect={() => void invoke('shell:openExternal', `https://github.com/${app.repo}`)}
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

        {job && (job.phase === 'error' || job.phase === 'cancelled') && (
          <div className="relative flex items-center gap-3 border-t border-accent/30 bg-[#2a0709]/80 px-7 py-3 text-[13px]">
            <AlertCircle className="size-4 shrink-0 text-danger" />
            <span className="min-w-0 flex-1 text-[#ffb3b6]">
              {job.phase === 'cancelled' ? 'Cancelled.' : job.error}
            </span>
            {job.missing?.map((id) => (
              <Button key={id} size="sm" onClick={() => useApps.setState({ guideId: id })}>
                <BookOpen className="size-3.5" /> How to install {REQUIREMENT_GUIDES[id].label}
              </Button>
            ))}
            <Button size="sm" variant="ghost" onClick={() => setLogOpen(true)}>
              <FileText className="size-3.5" /> Show log
            </Button>
            <Button size="sm" variant="ghost" className="px-2" title="Dismiss" onClick={() => dismissJob(app.id)}>
              <X className="size-4" />
            </Button>
          </div>
        )}
        {busy && job.phase === 'setup' && (
          <div className="relative flex items-center gap-3 border-t border-border bg-black/40 px-7 py-2 text-xs text-muted">
            <span className="min-w-0 flex-1 truncate font-mono">{lastLine || jobDetail(job)}</span>
            <button className="cursor-pointer text-[#ff5961] hover:underline" onClick={() => setLogOpen(true)}>
              Show log
            </button>
          </div>
        )}
      </header>

      {/* Body */}
      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
        <main className="min-w-0">
          <p className="mb-6 text-[15px] leading-relaxed text-[#d4d4d8]">{app.description}</p>
          <Gallery images={view.screenshots} name={app.name} />
          <h2 className="mt-10 mb-4 flex items-center gap-2 font-[family-name:var(--font-display)] text-xs font-semibold tracking-[0.16em] text-faint uppercase">
            <span className="h-3 w-1 -skew-x-[20deg] bg-accent" /> Read me
          </h2>
          <div className="rounded-xl border border-border bg-panel px-7 py-6">
            <Readme appId={app.id} />
          </div>
        </main>

        <aside className="space-y-4">
          <SideCard title="Highlights">
            <div className="flex flex-wrap gap-1.5">
              {app.highlights.map((h) => (
                <Badge key={h} className="bg-panel-2">
                  {h}
                </Badge>
              ))}
            </div>
          </SideCard>

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
                    className="cursor-pointer text-left break-all text-[#ff5961] hover:underline"
                    onClick={() => void act(() => invoke('apps:openFolder', app.id))}
                  >
                    {installed.dir}
                  </button>
                </Row>
              )}
              {app.repo && (
                <Row label="Source">
                  <button
                    className="cursor-pointer text-left break-all text-[#ff5961] hover:underline"
                    onClick={() => void invoke('shell:openExternal', `https://github.com/${app.repo}`)}
                  >
                    {app.repo} <ExternalLink className="inline size-3 align-[-1px]" />
                  </button>
                </Row>
              )}
            </dl>
          </SideCard>

          {installed && (
            <SideCard title="Shortcuts">
              <ShortcutRow view={view} where="desktop" label="Desktop" />
              <ShortcutRow view={view} where="startMenu" label="Start Menu" />
            </SideCard>
          )}

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
                <p className="text-[#fcd679]">That includes anything you saved inside it, such as downloaded models or outputs.</p>
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

function ShortcutRow({ view, where, label }: { view: AppView; where: 'desktop' | 'startMenu'; label: string }) {
  const sc = view.installed!.shortcuts
  const on = !!sc[where]
  return (
    <label className="flex items-center justify-between py-1.5 text-[13px]">
      <span>{label}</span>
      <Switch
        checked={on}
        disabled={!!view.installed?.needsSetup}
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
