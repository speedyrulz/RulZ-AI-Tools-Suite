import { ArrowUpCircle, Download, FolderSearch, Loader2, Play, Wrench, X } from 'lucide-react'
import type { AppView, JobState } from '@shared/types'
import { invoke } from '../lib/api'
import { cn, formatBytes } from '../lib/format'
import { act, isActive, useApps } from '../stores/apps'
import { Button, ProgressBar } from './ui'

/** The job to show for an app: the live one from events, else the one in the last app list. */
export function useJob(view: AppView): JobState | null {
  const job = useApps((s) => s.jobs[view.app.id])
  return job ?? view.job
}

export function canInstall(view: AppView): boolean {
  return !!view.app.repo && !/isn't published/.test(view.latestError ?? '')
}

export function jobDetail(job: JobState): string {
  if (job.phase === 'downloading' && job.bytes !== undefined) {
    const total = job.total ? ` of ${formatBytes(job.total)}` : ''
    const speed = job.speed ? ` · ${formatBytes(job.speed)}/s` : ''
    return `${formatBytes(job.bytes)}${total}${speed}`
  }
  if (job.progress !== null && job.phase !== 'done') return `${Math.round(job.progress * 100)}%`
  return ''
}

/**
 * The main buttons for an app: Install, or Launch (plus Update / Finish setup), or the progress of a job.
 * `compact` is the card version.
 */
export function AppActions({
  view,
  onInstall,
  compact
}: {
  view: AppView
  onInstall: () => void
  compact?: boolean
}) {
  const job = useJob(view)
  const id = view.app.id
  const size = compact ? 'sm' : 'lg'

  if (isActive(job)) {
    return (
      <div className={cn('flex min-w-0 items-center gap-2', compact ? 'w-full' : 'w-[300px]')}>
        <div className="min-w-0 flex-1">
          <div className="mb-1.5 flex items-center justify-between gap-2 text-xs">
            <span className="flex min-w-0 items-center gap-1.5 truncate text-text">
              <Loader2 className="size-3.5 shrink-0 animate-spin text-accent" />
              <span className="truncate">{job.label}</span>
            </span>
            <span className="shrink-0 text-faint tabular-nums">{jobDetail(job)}</span>
          </div>
          <ProgressBar value={job.progress} />
        </div>
        {job.action !== 'uninstall' && (
          <Button
            variant="ghost"
            size="sm"
            className="px-2"
            title="Cancel"
            onClick={(e) => {
              e.stopPropagation()
              void act(() => invoke('apps:cancel', id))
            }}
          >
            <X className="size-4" />
          </Button>
        )}
      </div>
    )
  }

  const installed = view.installed
  if (!installed) {
    const available = canInstall(view)
    return (
      <Button
        variant="primary"
        size={size}
        disabled={!available}
        title={available ? undefined : (view.latestError ?? undefined)}
        onClick={(e) => {
          e.stopPropagation()
          onInstall()
        }}
      >
        <Download className={compact ? 'size-3.5' : 'size-4.5'} />
        {available ? 'Install' : 'Coming soon'}
      </Button>
    )
  }

  if (view.missing && !installed.managed) {
    // The record is kept (a drive may just be disconnected): point it at the program again, or remove it from the
    // menu. A launcher-made install keeps its buttons; Launch explains what's missing, and Uninstall cleans up.
    return (
      <Button
        variant="outline"
        size={size}
        title={`${installed.launchPath} isn't there`}
        onClick={(e) => {
          e.stopPropagation()
          void act(() => invoke('apps:locate', id))
        }}
      >
        <FolderSearch className={compact ? 'size-3.5' : 'size-4.5'} />
        Locate…
      </Button>
    )
  }

  if (installed.needsSetup) {
    return (
      <Button
        variant="primary"
        size={size}
        onClick={(e) => {
          e.stopPropagation()
          void act(() => invoke('apps:repair', id))
        }}
      >
        <Wrench className={compact ? 'size-3.5' : 'size-4.5'} />
        Finish setup
      </Button>
    )
  }

  return (
    <div className="flex items-center gap-2">
      {view.updateAvailable && (
        <Button
          variant="outline"
          size={size}
          onClick={(e) => {
            e.stopPropagation()
            void act(() => invoke('apps:update', id))
          }}
        >
          <ArrowUpCircle className={compact ? 'size-3.5' : 'size-4.5'} />
          Update
        </Button>
      )}
      <Button
        variant="primary"
        size={size}
        onClick={(e) => {
          e.stopPropagation()
          void act(() => invoke('apps:launch', id))
        }}
      >
        <Play className={cn('fill-current', compact ? 'size-3' : 'size-4')} />
        Launch
      </Button>
    </div>
  )
}
