import { LoaderCircle, Save, Square } from 'lucide-react'
import { Fragment, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  axisValueLabel,
  bestResult,
  BENCH_AXIS_NAMES,
  comboConfig,
  comboLabel,
  type BenchmarkRun,
  type BenchResult,
  type BenchResultStatus
} from '@shared/benchmark'
import { etaMs, fmtGbFromBytes, fmtMiB, fmtMs, fmtRatioPct, fmtSignedPct, fmtTps, resultVramDiff, runProgress } from '../../lib/bench'
import { errorMessage, invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { useModels } from '../../stores/models'
import { Badge, Button, Dialog, IconButton, ProgressBar, Tooltip } from '../ui'

const STATUS: Record<BenchResultStatus, { label: string; tone: 'neutral' | 'accent' | 'warn' | 'danger' | 'ok'; hint: string }> = {
  pending: { label: 'Waiting', tone: 'neutral', hint: 'Not tried yet.' },
  running: { label: 'Running', tone: 'accent', hint: 'Loading or measuring now.' },
  ok: { label: 'Done', tone: 'ok', hint: 'Measured.' },
  failed: { label: 'Failed', tone: 'danger', hint: 'The model did not load or a request failed.' },
  skipped: { label: 'Skipped', tone: 'warn', hint: 'The memory estimate said this would not fit, so it was not tried.' },
  stopped: { label: 'Stopped', tone: 'neutral', hint: 'The run was stopped before this one finished.' }
}

/** Progress of the run in progress: n of m, time left, Stop. */
export function BenchProgress({ run, onStop }: { run: BenchmarkRun; onStop: () => void }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])
  const { done, total } = runProgress(run)
  const eta = etaMs(run, now)
  const current = run.results.find((r) => r.status === 'running')
  return (
    <div className="rounded-lg border border-accent/40 bg-accent-soft/40 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-[13px] font-medium">
          <LoaderCircle size={14} className="shrink-0 animate-spin text-accent" />
          <span className="truncate">
            Benchmark{run.label ? ` “${run.label}”` : ''} of {run.modelName}: {done} of {total} done
          </span>
        </div>
        <Button size="sm" variant="danger" onClick={onStop}>
          <Square size={12} /> Stop
        </Button>
      </div>
      <ProgressBar value={total ? done / total : 0} />
      <div className="mt-1.5 flex flex-wrap gap-x-3 text-[11.5px] text-muted tabular-nums">
        <span>{current ? `Now: ${comboLabel(current.combo)}` : 'Starting the next combination…'}</span>
        <span>{eta === null ? 'Time left: after the first combination' : `About ${fmtMs(Math.round(eta / 1000) * 1000)} left`}</span>
      </div>
    </div>
  )
}

const th = 'px-2 py-1.5 text-right font-medium'
const td = 'px-2 py-1 text-right whitespace-nowrap text-muted tabular-nums'

/** One row per combination of a run, with speed and VRAM. The fastest finished one is highlighted. */
export function BenchTable({ run }: { run: BenchmarkRun }) {
  const best = useMemo(() => bestResult(run.results), [run.results])
  const instances = useModels((s) => s.instances)
  const [saveTarget, setSaveTarget] = useState<BenchResult | null>(null)
  const loaded = instances.some((i) => i.modelId === run.modelId)
  const cols = run.axes.map((a) => a.key)
  const colCount = (cols.length || 1) + 10

  const save = async (): Promise<void> => {
    const r = saveTarget
    setSaveTarget(null)
    if (!r) return
    try {
      await invoke('models:setDefaults', run.modelId, comboConfig(run.base, r.combo))
      toast.success(
        loaded
          ? `Saved as the default load settings of ${run.modelName}. They apply the next time it loads.`
          : `Saved as the default load settings of ${run.modelName}`
      )
    } catch (e) {
      toast.error(errorMessage(e))
    }
  }

  return (
    <>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[960px] text-[12px]">
          <thead className="bg-panel-2 text-[10.5px] tracking-wide text-faint uppercase">
            <tr>
              {cols.length === 0 ? (
                <th className="px-3 py-1.5 text-left font-medium">Settings</th>
              ) : (
                cols.map((k) => (
                  <th key={k} className="px-2 py-1.5 text-left font-medium first:pl-3">
                    {BENCH_AXIS_NAMES[k]}
                  </th>
                ))
              )}
              <th className="px-2 py-1.5 text-left font-medium">Status</th>
              <th className={th}>Load</th>
              <th className={th}>
                <Tooltip content="Time to first token, median.">
                  <span>First token</span>
                </Tooltip>
              </th>
              <th className={th}>
                <Tooltip content="Prompt processing speed, tokens per second.">
                  <span>Prompt t/s</span>
                </Tooltip>
              </th>
              <th className={th}>
                <Tooltip content="Generation speed, tokens per second.">
                  <span>Gen t/s</span>
                </Tooltip>
              </th>
              <th className={th}>
                <Tooltip content="Drafted tokens the model accepted.">
                  <span>Draft hit</span>
                </Tooltip>
              </th>
              <th className={th}>
                <Tooltip content="Highest GPU memory the combination added while loaded, summed over the GPUs.">
                  <span>VRAM</span>
                </Tooltip>
              </th>
              <th className={th}>
                <Tooltip content="What the memory estimate said for this combination.">
                  <span>Estimate</span>
                </Tooltip>
              </th>
              <th className={th}>
                <Tooltip content="Measured against estimated VRAM. Positive: it used more than the estimate said.">
                  <span>Diff</span>
                </Tooltip>
              </th>
              <th className="w-10" />
            </tr>
          </thead>
          <tbody>
            {run.results.map((r, n) => {
              const st = STATUS[r.status]
              const isBest = best === r
              const diff = resultVramDiff(r)
              return (
                <Fragment key={n}>
                  <tr className={cn('border-t border-border/60', isBest ? 'bg-ok/10' : 'hover:bg-hover/40')}>
                    {cols.length === 0 ? (
                      <td className="px-3 py-1">{comboLabel(r.combo)}</td>
                    ) : (
                      cols.map((k) => (
                        <td key={k} className="px-2 py-1 whitespace-nowrap first:pl-3">
                          {k in r.combo ? axisValueLabel(k, r.combo[k] as never) : '—'}
                        </td>
                      ))
                    )}
                    <td className="px-2 py-1">
                      <span className="inline-flex items-center gap-1">
                        <Tooltip content={r.error ?? st.hint}>
                          <span>
                            <Badge tone={st.tone}>{st.label}</Badge>
                          </span>
                        </Tooltip>
                        {isBest && <Badge tone="ok">Fastest</Badge>}
                      </span>
                    </td>
                    <td className={td}>{fmtMs(r.loadMs)}</td>
                    <td className={td}>{fmtMs(r.ttftMs)}</td>
                    <td className={td}>{fmtTps(r.promptTps)}</td>
                    <td className={cn(td, isBest && 'font-medium text-text')}>{fmtTps(r.genTps)}</td>
                    <td className={td}>{fmtRatioPct(r.draftAcceptance)}</td>
                    <td className={td}>{fmtMiB(r.peakVramMiB)}</td>
                    <td className={td}>{fmtGbFromBytes(r.estimate?.gpuBytes)}</td>
                    <td className={cn(td, diff !== null && Math.abs(diff) > 25 && 'text-warn')}>{fmtSignedPct(diff)}</td>
                    <td className="px-2 py-1">
                      {r.status === 'ok' && (
                        <IconButton size="sm" label="Save these settings as this model’s defaults" onClick={() => setSaveTarget(r)}>
                          <Save size={13} />
                        </IconButton>
                      )}
                    </td>
                  </tr>
                  {r.error && (r.status === 'failed' || r.status === 'stopped') && (
                    <tr>
                      <td colSpan={colCount} className="px-3 pb-1.5 text-[11px] break-words text-danger">
                        {r.error}
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>

      <Dialog
        open={!!saveTarget}
        onOpenChange={(o) => !o && setSaveTarget(null)}
        title="Save as default load settings?"
        description={saveTarget ? `${run.modelName}: ${comboLabel(saveTarget.combo)}` : undefined}
        footer={
          <>
            <Button variant="ghost" onClick={() => setSaveTarget(null)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => void save()}>
              Save as default
            </Button>
          </>
        }
      >
        <div className="space-y-2 px-5 py-3 text-xs text-muted">
          <p>This replaces the model’s saved load settings with the benchmark’s starting settings plus this combination.</p>
          {loaded && <p className="text-warn">The model is loaded now. The new settings take effect the next time it loads.</p>}
        </div>
      </Dialog>
    </>
  )
}
