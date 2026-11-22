import { AlertTriangle, History, Info, RefreshCw, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { useNovel } from '../../stores/novel'
import { Button } from '../ui'
import { timeLabel } from './common'

export function Banner({ tone = 'warn', icon, children, actions }: { tone?: 'warn' | 'danger' | 'info'; icon?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <div
      role={tone === 'danger' ? 'alert' : 'status'}
      className={cn(
        'flex flex-wrap items-center gap-x-3 gap-y-2 border-b px-4 py-2 text-[13px]',
        tone === 'danger' && 'border-danger-border bg-danger-bg text-danger-text',
        tone === 'warn' && 'border-warn/30 bg-warn/10 text-warn-text',
        tone === 'info' && 'border-border bg-panel-2 text-body'
      )}
    >
      <span className="shrink-0">{icon ?? <TriangleAlert size={16} />}</span>
      <div className="min-w-0 flex-1">{children}</div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </div>
  )
}

/** Notices for one open file: leftover journal, conflict, changed on disk, failed save, read-only, missing. */
export function BufferBanners({ path }: { path: string }) {
  const b = useNovel((s) => s.buffers[path])
  const missing = useNovel((s) => s.state?.missing.includes(path) ?? false)
  const retry = (): void => void useNovel.getState().save(path)
  const { restoreJournal, discardJournal, reloadFromDisk, createMissingFile } = useNovel.getState()
  if (!b) return null
  return (
    <>
      {missing && (
        <Banner
          tone="danger"
          icon={<AlertTriangle size={16} />}
          actions={
            <Button size="sm" onClick={() => void createMissingFile(path)}>
              Create empty file
            </Button>
          }
        >
          File missing. <span className="font-mono text-xs">{path}</span> is not on disk. Nothing is created until you choose to.
        </Banner>
      )}
      {b.readOnly && (
        <Banner icon={<Info size={16} />} tone="info">
          This file is not valid UTF-8, so it is shown read-only and is never saved.
        </Banner>
      )}
      {b.journal && (
        <Banner
          icon={<History size={16} />}
          actions={
            <>
              <Button size="sm" onClick={() => restoreJournal(path)}>
                Restore
              </Button>
              <Button size="sm" variant="ghost" onClick={() => void discardJournal(path)}>
                Discard
              </Button>
            </>
          }
        >
          Unsaved text from {timeLabel(b.journal.savedAt)} was found.
        </Banner>
      )}
      {b.conflict && (
        <Banner
          tone="danger"
          actions={
            <>
              <Button size="sm" onClick={() => void reloadFromDisk(path)}>
                Reload from disk
              </Button>
              <Button size="sm" variant="danger" onClick={() => void useNovel.getState().save(path, { force: true, snapshot: true })}>
                Keep mine
              </Button>
            </>
          }
        >
          Conflict. This file was changed outside the app while you were editing. <strong>Reload from disk</strong> drops your edits; <strong>Keep mine</strong>{' '}
          writes your text and keeps the disk version in the history.
        </Banner>
      )}
      {b.changedOnDisk && !b.conflict && (
        <Banner
          actions={
            <>
              <Button size="sm" onClick={() => void reloadFromDisk(path)}>
                Reload from disk
              </Button>
              <Button size="sm" variant="ghost" onClick={() => void useNovel.getState().save(path, { force: true, snapshot: true })}>
                Keep mine
              </Button>
            </>
          }
        >
          Changed on disk. This file changed outside the app while you have edits or a change pending. Your next save asks what to do.
        </Banner>
      )}
      {b.error && !b.conflict && (
        <Banner
          tone="danger"
          actions={
            <Button size="sm" onClick={retry}>
              <RefreshCw size={14} /> Retry now
            </Button>
          }
        >
          Save failed: {b.error}. Your text is kept here and the app tries again every 5 seconds.
        </Banner>
      )}
    </>
  )
}

/** Shown on the whole page when project.json or outline.json could not be read. */
export function BlockedBanner() {
  const state = useNovel((s) => s.state)
  if (!state?.blocked) return null
  return (
    <Banner
      tone="danger"
      icon={<AlertTriangle size={16} />}
      actions={
        <Button size="sm" onClick={() => void invoke('shell:showItem', state.folder)}>
          Show in folder
        </Button>
      }
    >
      {state.blocked} The project is open read-only, and the app will not write over it. Fix the file or restore it from <span className="font-mono text-xs">.rulz\history</span>.
    </Banner>
  )
}
