import { ArrowUpCircle, ExternalLink, Loader2 } from 'lucide-react'
import { invoke } from '../lib/api'
import { formatBytes } from '../lib/format'
import { act, isActive, useApps } from '../stores/apps'
import { Button, Dialog, ProgressBar } from './ui'

/** Offered at startup when GitHub has a newer launcher. Install downloads it, closes the launcher and reopens it. */
export function SelfUpdateDialog() {
  const update = useApps((s) => s.selfUpdate)
  const progress = useApps((s) => s.selfProgress)
  // The setup stops the launcher, so it waits for app installs to finish (a string: a selector must return a stable value).
  const installing = useApps((s) =>
    s.apps
      .filter((a) => isActive(s.jobs[a.app.id] ?? a.job))
      .map((a) => a.app.name)
      .join(', ')
  )
  const busy = progress !== null && progress >= 0
  if (!update) return null

  const close = (): void => useApps.setState({ selfUpdate: null, selfProgress: null })
  const install = (): void => {
    useApps.setState({ selfProgress: 0 })
    void act(() => invoke('self:install', update))
  }
  const skip = (): void => {
    void useApps.getState().updateSettings({ skippedUpdate: update.tag })
    close()
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && !busy && close()}
      title="A new version is available"
      description={`RulZ AI Tools Suite ${update.version} (${formatBytes(update.size)})`}
      footer={
        busy ? undefined : (
          <>
            <Button variant="ghost" onClick={skip}>
              Skip this version
            </Button>
            <Button onClick={close}>Later</Button>
            <Button
              variant="primary"
              disabled={!!installing}
              title={installing ? `Waiting for ${installing} to finish installing` : undefined}
              onClick={install}
            >
              <ArrowUpCircle className="size-4" /> Update now
            </Button>
          </>
        )
      }
    >
      {busy ? (
        <div className="space-y-3">
          <p className="flex items-center gap-2 text-sm">
            <Loader2 className="size-4 animate-spin text-accent" />
            {progress < 1 ? 'Downloading the update…' : 'Installing. The launcher will close and reopen.'}
          </p>
          <ProgressBar value={progress < 1 ? progress : null} />
        </div>
      ) : (
        <div className="space-y-3 text-[13px] text-muted">
          {progress === -1 && <p className="text-danger-text">The update couldn&apos;t be installed. Try again later.</p>}
          {installing && (
            <p className="text-warn-text">
              {installing} {installing.includes(', ') ? 'are' : 'is'} still being installed. Updating now would stop it
              halfway, so you can update once it finishes.
            </p>
          )}
          <p>Updating closes the launcher, installs the new version over this one and opens it again. Your installed apps and settings stay as they are.</p>
          <button
            className="inline-flex cursor-pointer items-center gap-1 text-link hover:underline"
            onClick={() => void invoke('shell:openExternal', update.notesUrl)}
          >
            See what&apos;s new <ExternalLink className="size-3" />
          </button>
        </div>
      )}
    </Dialog>
  )
}
