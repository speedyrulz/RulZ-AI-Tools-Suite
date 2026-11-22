import { Play, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { BENCH_AXIS_NAMES, BENCH_PROMPT_TOKENS, comboCount, type BenchAxisKey } from '@shared/benchmark'
import type { LoadConfig, ModelEntry } from '@shared/types'
import {
  availableAxes,
  AXIS_SPECS,
  baseSummary,
  buildBenchRequest,
  defaultBenchForm,
  type AxisForm,
  type BenchForm
} from '../../lib/bench'
import { invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { activeBenchRun, useBench } from '../../stores/bench'
import { Button, Field, Input, NumberInput, Select, Switch } from '../ui'

const HINTS: Record<BenchAxisKey, string> = {
  gpuLayers: 'How many layers sit on the GPU.',
  cpuMoe: 'Keeps expert weights in system RAM; frees VRAM, usually slower.',
  cacheType: 'Precision of the KV cache. Smaller types save VRAM.',
  draft: 'Speculative decoding with the model’s draft.',
  contextLength: 'Context size to load with.',
  batchSize: 'Tokens processed per batch.',
  ubatchSize: 'Tokens processed per micro-batch.'
}

function AxisCard({ axisKey, form, onChange }: { axisKey: BenchAxisKey; form: AxisForm; onChange: (f: AxisForm) => void }) {
  const spec = AXIS_SPECS[axisKey]
  return (
    <div className={cn('rounded-lg border p-3', form.vary ? 'border-accent/40 bg-accent-soft/20' : 'border-border bg-panel-2/40')}>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[13px] font-medium">{BENCH_AXIS_NAMES[axisKey]}</div>
          <div className="text-[11px] text-faint">{HINTS[axisKey]}</div>
        </div>
        <label className="flex shrink-0 items-center gap-1.5 text-[12.5px]">
          <input type="checkbox" checked={form.vary} onChange={(e) => onChange({ ...form, vary: e.target.checked })} /> Vary
        </label>
      </div>
      {form.vary && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
          {spec.choices.map((c) => (
            <label key={c.id} className="flex items-center gap-1.5 text-[12.5px]">
              <input
                type="checkbox"
                checked={form.picks.includes(c.id)}
                onChange={(e) => onChange({ ...form, picks: e.target.checked ? [...form.picks, c.id] : form.picks.filter((p) => p !== c.id) })}
              />
              {c.label}
            </label>
          ))}
          {spec.numbers && (
            <Input
              className="h-7.5 w-64 text-xs"
              value={form.custom}
              placeholder={spec.numberHint}
              onChange={(e) => onChange({ ...form, custom: e.target.value })}
              aria-label={`${BENCH_AXIS_NAMES[axisKey]}: numbers to try`}
            />
          )}
        </div>
      )}
    </div>
  )
}

/** Setup for a benchmark of one model: which settings to vary, the prompt, and Start. */
export function BenchSetup({ model }: { model: ModelEntry }) {
  const runs = useBench((s) => s.runs)
  const starting = useBench((s) => s.starting)
  const start = useBench((s) => s.start)
  const running = !!activeBenchRun(runs)
  const [base, setBase] = useState<LoadConfig | null>(null)
  const [custom, setCustom] = useState(false)
  const [form, setForm] = useState<BenchForm>(defaultBenchForm)

  useEffect(() => {
    let live = true
    void invoke('models:getDefaults', model.id).then((d) => {
      if (!live) return
      setBase(d.config)
      setCustom(d.custom)
    })
    return () => {
      live = false
    }
  }, [model.id])

  const available = useMemo(() => availableAxes({ moe: !!model.info?.expertCount, hasDraft: !!model.draft }), [model])
  const built = useMemo(() => (base ? buildBenchRequest(form, { modelId: model.id, base, available }) : null), [form, base, model.id, available])
  const count = built ? comboCount(built.request.axes) : 0
  const setAxis = (k: BenchAxisKey, f: AxisForm): void => setForm((cur) => ({ ...cur, axes: { ...cur.axes, [k]: f } }))
  const set = <K extends keyof BenchForm>(k: K, v: BenchForm[K]): void => setForm((cur) => ({ ...cur, [k]: v }))

  return (
    <section className="space-y-4 rounded-xl border border-border bg-panel p-4">
      <div>
        <div className="font-[family-name:var(--font-display)] text-[13px] font-semibold">New benchmark</div>
        <div className="mt-0.5 text-xs text-muted">
          Runs one fixed prompt with every combination of the settings you vary, and shows speed and VRAM for each, so you can pick settings
          by measurement.
        </div>
      </div>

      <div className="rounded-lg border border-border bg-panel-2/40 p-3 text-xs">
        <div className="mb-0.5 text-[13px] font-medium">
          Starts from this model’s saved load settings <span className="font-normal text-faint">({custom ? 'custom' : 'app defaults'})</span>
        </div>
        <div className="text-muted">{base ? baseSummary(base) : 'Reading the settings…'}</div>
        <div className="mt-0.5 text-faint">Settings you do not vary keep these values.</div>
      </div>

      <div className="space-y-2">
        {available.map((k) => (
          <AxisCard key={k} axisKey={k} form={form.axes[k]} onChange={(f) => setAxis(k, f)} />
        ))}
      </div>

      <div className="space-y-3 rounded-lg border border-border p-3">
        <div className="text-[13px] font-medium">Prompt</div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
          <span className="flex items-center gap-1.5">
            Prompt length
            <Select className="h-7.5 w-28 text-xs" value={form.promptTokens} onChange={(e) => set('promptTokens', Number(e.target.value))}>
              {[...new Set<number>([...BENCH_PROMPT_TOKENS, form.promptTokens])].sort((a, b) => a - b).map((n) => (
                <option key={n} value={n}>
                  {n} tokens
                </option>
              ))}
            </Select>
          </span>
          <span className="flex items-center gap-1.5">
            Generated tokens
            <NumberInput className="w-20" value={form.genTokens} min={8} max={4096} onChange={(v) => set('genTokens', v ?? form.genTokens)} />
          </span>
          <span className="flex items-center gap-1.5">
            Repeats
            <NumberInput className="w-14" value={form.repeats} min={1} max={10} onChange={(v) => set('repeats', v ?? form.repeats)} />
          </span>
          <label className="flex items-center gap-1.5">
            <Switch checked={form.skipLikelyOom} onCheckedChange={(v) => set('skipLikelyOom', v)} /> Skip combinations the estimate says won’t fit
          </label>
        </div>
        <Field label="Label" hint="Optional, shown in the list of runs">
          <Input className="h-7.5 max-w-80 text-xs" value={form.label} maxLength={80} placeholder="e.g. 5090 layers and cache" onChange={(e) => set('label', e.target.value)} />
        </Field>
      </div>

      <div className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
        <TriangleAlert size={14} className="mt-px shrink-0" />
        <div>
          Starting unloads every loaded model, loads this model again for each combination, and uses the GPUs for the whole run. Do not use the
          model or the GPUs meanwhile, or the numbers will be off.
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className={cn('min-w-0 flex-1 text-[11.5px] tabular-nums', built?.problem ? 'text-danger' : 'text-faint')}>
          {built?.problem ?? `${count} combination${count === 1 ? '' : 's'}, each ${form.repeats} timed run${form.repeats === 1 ? '' : 's'} after a warm-up`}
        </div>
        <Button
          variant="primary"
          disabled={!built || !!built.problem || running || starting}
          onClick={() => built && void start(built.request)}
        >
          <Play size={14} /> Start benchmark
        </Button>
      </div>
      {running && <div className="text-right text-[11.5px] text-faint">A benchmark is already running.</div>}
    </section>
  )
}
