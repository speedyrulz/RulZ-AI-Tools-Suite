import { AlertTriangle, FolderOpen, Info } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { AppView, InstallOptions } from '@shared/types'
import { invoke } from '../lib/api'
import { formatBytes, formatVersion } from '../lib/format'
import { act, useApps } from '../stores/apps'
import { RequirementList } from './Requirements'
import { Button, Checkbox, Dialog } from './ui'

export function InstallDialog({ view, onClose }: { view: AppView | null; onClose: () => void }) {
  const [opts, setOpts] = useState<InstallOptions | null>(null)
  const [busy, setBusy] = useState(false)
  const requirements = useApps((s) => s.requirements)
  const app = view?.app

  useEffect(() => {
    if (!app) return
    setOpts(null)
    void invoke('apps:defaultDir', app.id).then((dir) =>
      setOpts({
        dir,
        desktopShortcut: true,
        startMenuShortcut: true,
        components: (app.components ?? []).filter((c) => c.default).map((c) => c.id),
        launchAfter: true
      })
    )
  }, [app])

  if (!view || !app) return null
  const missing = app.requirements.filter((r) => !r.optional && requirements.find((s) => s.id === r.id)?.ok === false)
  const patch = (p: Partial<InstallOptions>): void => setOpts((o) => (o ? { ...o, ...p } : o))

  const install = async (): Promise<void> => {
    if (!opts) return
    setBusy(true)
    await act(async () => {
      await invoke('apps:install', app.id, opts)
      onClose()
    })
    setBusy(false)
  }

  const kindNote =
    app.install.kind === 'source'
      ? 'Downloads the source from GitHub and runs its setup.'
      : app.install.kind === 'installer'
        ? 'Downloads the setup program and installs it silently.'
        : 'Downloads the portable release and unpacks it.'

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={`Install ${app.name}`}
      description={
        <>
          {view.latest ? `${view.latest.label}` : 'Latest version'}
          {view.latest?.downloadSize ? ` · ${formatBytes(view.latest.downloadSize)} download` : ''}
          {' · '}
          {kindNote}
        </>
      }
      className="w-[580px]"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!opts || busy} onClick={() => void install()}>
            Install{view.latest ? ` ${formatVersion(view.latest.version)}` : ''}
          </Button>
        </>
      }
    >
      {opts && (
        <div className="space-y-5">
          <section>
            <label className="mb-1.5 block text-xs font-medium tracking-wide text-faint uppercase">Install location</label>
            <div className="flex gap-2">
              <input
                value={opts.dir}
                onChange={(e) => patch({ dir: e.target.value })}
                spellCheck={false}
                className="h-9 min-w-0 flex-1 rounded-lg border border-border bg-panel-2 px-3 text-[13px] outline-none focus:border-accent"
              />
              <Button
                onClick={async () => {
                  const dir = await invoke('dialog:pickFolder', { title: 'Install folder', defaultPath: opts.dir })
                  if (dir) patch({ dir })
                }}
              >
                <FolderOpen className="size-4" /> Browse
              </Button>
            </div>
            <p className="mt-1.5 text-xs text-faint">The folder must be empty or not exist yet.</p>
          </section>

          <section>
            <div className="mb-1 text-xs font-medium tracking-wide text-faint uppercase">Shortcuts</div>
            <Checkbox checked={opts.desktopShortcut} onChange={(v) => patch({ desktopShortcut: v })} label="Desktop shortcut" />
            <Checkbox
              checked={opts.startMenuShortcut}
              onChange={(v) => patch({ startMenuShortcut: v })}
              label="Start Menu shortcut"
              hint="In the RulZ AI Tools Suite folder"
            />
          </section>

          {!!app.components?.length && (
            <section>
              <div className="mb-1 text-xs font-medium tracking-wide text-faint uppercase">Optional components</div>
              {app.components.map((c) => (
                <Checkbox
                  key={c.id}
                  checked={opts.components.includes(c.id)}
                  onChange={(v) =>
                    patch({ components: v ? [...opts.components, c.id] : opts.components.filter((x) => x !== c.id) })
                  }
                  label={
                    <>
                      {c.label}
                      {c.sizeHint && <span className="text-faint"> · {c.sizeHint}</span>}
                    </>
                  }
                  hint={c.description}
                />
              ))}
            </section>
          )}

          {app.requirements.length > 0 && (
            <section>
              <div className="mb-2 text-xs font-medium tracking-wide text-faint uppercase">Requirements</div>
              <RequirementList refs={app.requirements} />
              {missing.length > 0 && (
                <div className="mt-3 flex gap-2 rounded-lg border border-[#fbbf24]/25 bg-[#fbbf24]/[0.07] px-3 py-2 text-[13px] text-[#fcd679]">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                  <span>Install the missing requirements first, or the setup may fail.</span>
                </div>
              )}
            </section>
          )}

          {!!app.notes?.length && (
            <section className="rounded-lg border border-border bg-panel-2/60 px-3 py-2.5">
              {app.notes.map((n) => (
                <p key={n} className="flex gap-2 py-0.5 text-[13px] text-muted">
                  <Info className="mt-0.5 size-3.5 shrink-0 text-faint" />
                  {n}
                </p>
              ))}
            </section>
          )}

          <Checkbox checked={opts.launchAfter} onChange={(v) => patch({ launchAfter: v })} label="Launch when finished" />
        </div>
      )}
    </Dialog>
  )
}
