import { ArrowLeft, ChevronRight, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { bestResult, type BenchmarkRun, type BenchRunStatus } from '@shared/benchmark'
import type { ModelEntry } from '@shared/types'
import { fmtTps, runProgress, runsForModel } from '../../lib/bench'
import { cn, relativeTime } from '../../lib/format'
import { activeBenchRun, useBench } from '../../stores/bench'
import { Badge, Button, Dialog, IconButton } from '../ui'
import { BenchProgress, BenchTable } from './BenchResults'
import { BenchSetup } from './BenchSetup'

const RUN_STATUS: Record<BenchRunStatus, { label: string; tone: 'neutral' | 'accent' | 'warn' | 'danger' | 'ok' }> = {
  running: { label: 'Running', tone: 'accent' },
  done: { label: 'Done', tone: 'ok' },
  stopped: { label: 'Stopped', tone: 'warn' },
  failed: { label: 'Failed', tone: 'danger' }
}

function PastRun({ run, onDelete }: { run: BenchmarkRun; onDelete: (r: BenchmarkRun) => void }) {
  const [open, setOpen] = useState(false)
  const best = useMemo(() => bestResult(run.results), [run.results])
  const st = RUN_STATUS[run.status]
  const { done, total } = runProgress(run)
  return (
    <div className="rounded-lg border border-border">
      <div className="flex items-center gap-2 px-3 py-2">
        <button className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
          <ChevronRight size={13} className={cn('shrink-0 text-faint transition-transform', open && 'rotate-90')} />
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-1.5 text-[13px]">
              <span className="truncate">{run.label || 'Benchmark'}</span>
              <Badge tone={st.tone}>{st.label}</Badge>
            </span>
            <span className="block truncate text-[11px] text-faint">
              {new Date(run.startedAt).toLocaleString()} ({relativeTime(run.startedAt)}) · {done}/{total} combinations
              {run.engineVersion ? ` · llama.cpp ${run.engineVersion}` : ''}
            </span>
          </span>
          <span className="shrink-0 text-right text-xs text-muted tabular-nums">
            {best ? (
              <>
                <span className="font-medium text-text">{fmtTps(best.genTps)}</span> t/s best
              </>
            ) : (
              '—'
            )}
          </span>
        </button>
        <IconButton size="sm" label="Delete this run" disabled={run.status === 'running'} onClick={() => onDelete(run)}>
          <Trash2 size={13} />
        </IconButton>
      </div>
      {open && (
        <div className="space-y-2 border-t border-border bg-panel-2/40 p-3">
          {run.error && <div className="text-xs text-danger">{run.error}</div>}
          <BenchTable run={run} />
        </div>
      )}
    </div>
  )
}

/** The benchmark sub-view of My Models: set up a run, watch it, and look at past ones for one model. */
export function BenchmarkView({ model, onBack }: { model: ModelEntry; onBack: () => void }) {
  const runs = useBench((s) => s.runs)
  const loaded = useBench((s) => s.loaded)
  const init = useBench((s) => s.init)
  const stop = useBench((s) => s.stop)
  const remove = useBench((s) => s.remove)
  const [deleting, setDeleting] = useState<BenchmarkRun | null>(null)
  useEffect(() => void init(), [init])

  const active = activeBenchRun(runs)
  const past = useMemo(() => runsForModel(runs, model.id).filter((r) => r.id !== active?.id), [runs, model.id, active?.id])

  return (
    <div className="flex h-full min-w-0 flex-col">
      <div className="flex items-center gap-3 border-b border-border px-6 py-4">
        <Button size="sm" onClick={onBack}>
          <ArrowLeft size={14} /> Back
        </Button>
        <div className="min-w-0 flex-1">
          <h1 className="font-[family-name:var(--font-display)] text-base font-semibold">Benchmark load settings</h1>
          <div className="truncate text-xs text-muted">{model.name}</div>
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-6 py-5">
        {active && (
          <section className="space-y-3">
            <BenchProgress run={active} onStop={() => void stop()} />
            {active.modelId !== model.id && (
              <div className="text-xs text-warn">This run is for {active.modelName}. A new benchmark can start when it ends.</div>
            )}
            <BenchTable run={active} />
          </section>
        )}

        <BenchSetup model={model} />

        <section className="space-y-2">
          <h2 className="font-[family-name:var(--font-display)] text-[13px] font-semibold">
            Past runs <span className="font-normal text-faint tabular-nums">· {past.length}</span>
          </h2>
          {!loaded ? (
            <div className="text-xs text-muted">Reading…</div>
          ) : past.length === 0 ? (
            <div className="text-xs text-muted">No benchmark of this model yet.</div>
          ) : (
            <div className="space-y-2">
              {past.map((r) => (
                <PastRun key={r.id} run={r} onDelete={setDeleting} />
              ))}
            </div>
          )}
        </section>
      </div>

      <Dialog
        open={!!deleting}
        onOpenChange={(o) => !o && setDeleting(null)}
        title="Delete this benchmark run?"
        description={deleting ? `${deleting.label || 'Benchmark'} of ${deleting.modelName}, ${new Date(deleting.startedAt).toLocaleString()}. This cannot be undone.` : undefined}
        footer={
          <>
            <Button variant="ghost" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                const r = deleting
                setDeleting(null)
                if (r) void remove(r.id)
              }}
            >
              Delete
            </Button>
          </>
        }
      />
    </div>
  )
}
