import { Copy, FolderInput, Square } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { MergeJob } from '@shared/merge'
import { cn } from '../../lib/format'
import { JOB_STATUS, formatBytesGiB, formatElapsed, formulaText, jobElapsed, jobIsActive, progressFraction, splitName, verifyText } from '../../lib/merge'
import { useMerge } from '../../stores/merge'
import { Button, Card, IconButton, StatusChip } from '../ui'

/** Current time, ticking every second while `active`. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [active])
  return now
}

function copyText(text: string): void {
  navigator.clipboard.writeText(text).then(
    () => toast.success('Copied'),
    () => toast.error('Could not copy.')
  )
}

function summary(job: MergeJob): string {
  const r = job.recipe
  return `${splitName(r.a).file} and ${splitName(r.b).file}${r.c ? ` and ${splitName(r.c).file}` : ''}`
}

function JobRow({ job, now }: { job: MergeJob; now: number }) {
  const cancel = useMerge((s) => s.cancel)
  const moveToModels = useMerge((s) => s.moveToModels)
  const [moving, setMoving] = useState(false)
  const st = JOB_STATUS[job.status]
  const active = jobIsActive(job.status)
  const frac = progressFraction(job.progress)
  const out = job.output

  return (
    <li className="rounded-lg border border-border bg-panel-2 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
          <StatusChip tone={st.tone} label={st.label} busy={st.busy} />
          <span className="font-mono text-xs">{formulaText(job.recipe)}</span>
          <span className="text-xs text-faint tabular-nums">{formatElapsed(jobElapsed(job, now))}</span>
        </div>
        {(job.status === 'queued' || job.status === 'running') && (
          <Button size="sm" variant="danger" onClick={() => void cancel(job.id)}>
            <Square size={12} />
            Cancel
          </Button>
        )}
      </div>
      <p className="mt-1 truncate text-xs text-muted" title={summary(job)}>
        {summary(job)}
      </p>

      {active && job.progress && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-xs text-muted">
            <span>{job.progress.node ?? 'Working'}</span>
            {job.progress.max > 0 && (
              <span className="tabular-nums">
                {job.progress.value} / {job.progress.max}
              </span>
            )}
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-border" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={frac === null ? undefined : Math.round(frac * 100)}>
            <div className={cn('h-full bg-accent transition-[width]', frac === null && 'w-1/3 animate-pulse')} style={frac === null ? undefined : { width: `${frac * 100}%` }} />
          </div>
        </div>
      )}

      {job.error && <p className="mt-3 text-[13px] break-words text-danger-text">{job.error}</p>}

      {out && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-1">
            <code className="min-w-0 flex-1 rounded-md border border-border bg-panel px-2 py-1 font-mono text-xs break-all select-text">{out.wslPath ?? out.relPath}</code>
            <IconButton label="Copy path" onClick={() => copyText(out.wslPath ?? out.relPath)}>
              <Copy size={14} />
            </IconButton>
          </div>
          {out.sizeBytes !== null && <p className="text-xs text-faint">{formatBytesGiB(out.sizeBytes)}</p>}
          {out.verify && <p className={cn('text-[13px]', out.verify.ok ? 'text-ok' : 'text-danger-text')}>{verifyText(out.verify)}</p>}
          {out.movedTo ? (
            <p className="text-[13px] text-muted">
              Now in models: <code className="font-mono text-xs break-all select-text">{out.movedTo}</code>
            </p>
          ) : (
            job.own &&
            job.status === 'done' &&
            out.wslPath && (
              <Button
                size="sm"
                disabled={moving}
                onClick={() => {
                  setMoving(true)
                  void moveToModels(job.id).finally(() => setMoving(false))
                }}
              >
                <FolderInput size={14} />
                Move into models
              </Button>
            )
          )}
        </div>
      )}
    </li>
  )
}

export function JobsCard() {
  const jobs = useMerge((s) => s.jobs)
  const now = useNow(jobs.some((j) => jobIsActive(j.status)))
  return (
    <Card title="Merges of this session" description="They are gone from this list when the app closes. The files stay on the server.">
      {jobs.length === 0 ? (
        <p className="text-[13px] text-muted">No merge yet.</p>
      ) : (
        <ul className="space-y-3">
          {jobs.map((j) => (
            <JobRow key={j.id} job={j} now={now} />
          ))}
        </ul>
      )}
    </Card>
  )
}
