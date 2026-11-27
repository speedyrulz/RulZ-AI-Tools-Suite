import { Copy, Square } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { LoraMergeJob } from '@shared/loraMerge'
import { cn } from '../../lib/format'
import { LORA_JOB_STATUS, energyText, formatRankRange, formatSize, jobFraction, loraJobIsActive, recipeSummary, verifyLine } from '../../lib/loraMerge'
import { formatElapsed, jobElapsed } from '../../lib/merge'
import { useLoraMerge } from '../../stores/loraMerge'
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

function JobRow({ job, now }: { job: LoraMergeJob; now: number }) {
  const cancel = useLoraMerge((s) => s.cancel)
  const st = LORA_JOB_STATUS[job.status]
  const active = loraJobIsActive(job.status)
  const frac = jobFraction(job.progress)
  const out = job.output
  const summary = recipeSummary(job.recipe)
  const verify = out ? verifyLine(out.verify) : null

  return (
    <li className="rounded-lg border border-border bg-panel-2 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
          <StatusChip tone={st.tone} label={st.label} busy={st.busy} />
          <span className="text-xs text-faint tabular-nums">{formatElapsed(jobElapsed(job, now))}</span>
        </div>
        {job.status === 'running' && (
          <Button size="sm" variant="danger" onClick={() => void cancel(job.id)}>
            <Square size={12} />
            Cancel
          </Button>
        )}
      </div>
      <p className="mt-1 truncate text-xs text-muted" title={summary}>
        {summary}
      </p>

      {active && job.progress && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-xs text-muted">
            <span>Layers</span>
            <span className="tabular-nums">
              {job.progress.done.toLocaleString('en-US')} / {job.progress.total.toLocaleString('en-US')}
            </span>
          </div>
          <div
            className="h-1.5 overflow-hidden rounded-full bg-border"
            role="progressbar"
            aria-label="Layers done"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={frac === null ? undefined : Math.round(frac * 100)}
          >
            <div className={cn('h-full bg-accent transition-[width]', frac === null && 'w-1/3 animate-pulse')} style={frac === null ? undefined : { width: `${frac * 100}%` }} />
          </div>
        </div>
      )}

      {job.error && <p className="mt-3 text-[13px] break-words text-danger-text">{job.error}</p>}

      {out && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-1">
            <code className="min-w-0 flex-1 rounded-md border border-border bg-panel px-2 py-1 font-mono text-xs break-all select-text">{out.loraName}</code>
            <IconButton label="Copy LoRA name" onClick={() => copyText(out.loraName)}>
              <Copy size={14} />
            </IconButton>
          </div>
          <p className="text-xs text-faint">
            {formatSize(out.sizeBytes)}, rank {formatRankRange(out.rank).slice(1)}, {out.modules.toLocaleString('en-US')} layers
          </p>
          {out.energyKept && <p className="text-[13px] text-muted">{energyText(out.energyKept)}</p>}
          {verify && <p className={cn('text-[13px]', verify.tone === 'ok' ? 'text-ok' : 'text-warn-text')}>{verify.text}</p>}
        </div>
      )}
    </li>
  )
}

export function JobsCard() {
  const jobs = useLoraMerge((s) => s.jobs)
  const now = useNow(jobs.some((j) => loraJobIsActive(j.status)))
  return (
    <Card title="LoRA merges of this session" description="They are gone from this list when the app closes. The files stay in models/loras/merged/.">
      {jobs.length === 0 ? (
        <p className="text-[13px] text-muted">No LoRA merge yet.</p>
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
