import { AlertTriangle, FileJson, Loader2, Plus, Shuffle, Trash2 } from 'lucide-react'
import { useEffect } from 'react'
import { GRID_MAX_PROMPTS, GRID_MAX_SEEDS, GRID_MAX_VALUES, GRID_MIN_VALUES, GRID_PROMPT_MAX, type GridRowKind } from '@shared/grid'
import {
  AXIS_RANGE,
  axisOptions,
  cellCount,
  cellCountText,
  parseValues,
  recipeLines,
  recipeReadiness,
  resolveAxisOption,
  rowCountOf,
  seedsError,
  templateFit,
  valuesError,
  validSeed
} from '../../lib/grid'
import { useApp } from '../../stores/app'
import { useGrid } from '../../stores/grid'
import { useMerge } from '../../stores/merge'
import { Badge, Button, Card, Field, IconButton, Input, NumberInput, Segmented } from '../ui'

const SELECT_CLASS =
  'h-9 w-full rounded-lg border border-border bg-panel px-3 text-[13px] focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-45'

export function TemplateCard() {
  const template = useGrid((s) => s.template)
  const loading = useGrid((s) => s.loadingTemplate)
  const choose = useGrid((s) => s.chooseTemplate)
  return (
    <Card
      title="Workflow"
      description="Your own ComfyUI workflow renders every image. The merge replaces its model loader."
      actions={
        <Button size="sm" disabled={loading} onClick={() => void choose()}>
          {loading ? <Loader2 size={14} className="animate-spin" /> : <FileJson size={14} />}
          Choose workflow
        </Button>
      }
    >
      <p className="text-xs text-faint">Export your workflow from ComfyUI with Workflow &gt; Export (API).</p>
      {template && (
        <div className="mt-3 space-y-2">
          <p className="text-[13px] font-medium break-all" title={template.path}>
            {template.name}
          </p>
          {template.error ? (
            <p className="flex items-start gap-1.5 rounded-lg border border-danger-border bg-danger-bg px-3 py-2 text-[13px] break-words text-danger-text">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              {template.error}
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              <Badge tone={template.modelLoader ? 'ok' : 'warn'}>
                {template.modelLoader ? `Model loader: ${template.modelLoader.classType}` : 'No model loader found'}
              </Badge>
              <Badge tone={template.promptInput ? 'ok' : 'warn'}>
                {template.promptInput ? `Prompt: ${template.promptInput.classType}` : 'No prompt node found'}
              </Badge>
              <Badge>
                {template.seedInputs.length} seed{template.seedInputs.length === 1 ? '' : 's'}
              </Badge>
              <Badge>
                {template.imageOutputs.length} image output{template.imageOutputs.length === 1 ? '' : 's'}
              </Badge>
            </div>
          )}
          {template.warnings.map((w, i) => (
            <p key={i} className="flex items-start gap-1.5 text-[13px] break-words text-warn-text">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              {w}
            </p>
          ))}
        </div>
      )}
    </Card>
  )
}

export function MergeCard() {
  const recipe = useMerge((s) => s.recipe)
  const template = useGrid((s) => s.template)
  const setPage = useApp((s) => s.setPage)
  const fit = templateFit(template, recipe)
  const ready = recipeReadiness(recipe)
  return (
    <Card
      title="Merge"
      description="The merge set up on the Merge page. Nothing is saved as a model."
      actions={
        <Button size="sm" onClick={() => setPage('merge')}>
          Edit on Merge page
        </Button>
      }
    >
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
        {recipeLines(recipe).map((l) => (
          <div key={l.label} className="contents">
            <dt className="text-muted">{l.label}</dt>
            <dd className="min-w-0 break-words">{l.value}</dd>
          </div>
        ))}
      </dl>
      {ready && (
        <p className="mt-3 flex items-start gap-1.5 text-[13px] text-warn-text">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          {ready}
        </p>
      )}
      {fit && (
        <p className="mt-3 flex items-start gap-1.5 rounded-lg border border-danger-border bg-danger-bg px-3 py-2 text-[13px] text-danger-text">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          {fit}
        </p>
      )}
    </Card>
  )
}

export function ValuesCard() {
  const recipe = useMerge((s) => s.recipe)
  const axisKey = useGrid((s) => s.axisKey)
  const from = useGrid((s) => s.from)
  const to = useGrid((s) => s.to)
  const steps = useGrid((s) => s.steps)
  const valuesText = useGrid((s) => s.valuesText)
  const setAxisKey = useGrid((s) => s.setAxisKey)
  const setRange = useGrid((s) => s.setRange)
  const setValuesText = useGrid((s) => s.setValuesText)

  const options = axisOptions(recipe)
  const opt = resolveAxisOption(recipe, axisKey)
  // The chosen setting may no longer exist after the merge changed: move to the one that does, with its own range.
  const optKey = opt?.key ?? null
  useEffect(() => {
    if (optKey && optKey !== axisKey) setAxisKey(optKey)
  }, [optKey, axisKey, setAxisKey])
  const err = opt ? valuesError(opt.axis.kind, valuesText) : null
  const range = opt ? AXIS_RANGE[opt.axis.kind] : { min: -10, max: 10 }

  return (
    <Card title="Values" description="Each value is one column. Pick the setting to vary and the values to try.">
      {!opt ? (
        <p className="text-[13px] text-muted">The merge has nothing to vary yet. Use the ratio, the multiplier or a LoRA on the Merge page.</p>
      ) : (
        <div className="space-y-4">
          <Field label="Setting to vary">
            <select
              aria-label="Setting to vary"
              className={SELECT_CLASS}
              value={opt.key}
              onChange={(e) => setAxisKey(e.target.value)}
            >
              {options.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>
          </Field>
          <div className="grid grid-cols-3 gap-3">
            <Field label="From">
              <NumberInput value={from} min={range.min} max={range.max} step={0.05} onCommit={(v) => setRange({ from: v })} />
            </Field>
            <Field label="To">
              <NumberInput value={to} min={range.min} max={range.max} step={0.05} onCommit={(v) => setRange({ to: v })} />
            </Field>
            <Field label="Steps">
              <NumberInput value={steps} min={GRID_MIN_VALUES} max={GRID_MAX_VALUES} step={1} onCommit={(v) => setRange({ steps: v })} />
            </Field>
          </div>
          <Field
            label="Values"
            hint={`${GRID_MIN_VALUES} to ${GRID_MAX_VALUES} values, separated by commas, from ${range.min} to ${range.max}. Edit the list to use your own.`}
          >
            <Input
              aria-label="Values"
              className="tabular-nums"
              value={valuesText}
              aria-invalid={err !== null}
              onChange={(e) => setValuesText(e.target.value)}
            />
          </Field>
          {err ? (
            <p className="flex items-start gap-1.5 text-[13px] text-danger-text">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              {err}
            </p>
          ) : (
            <p className="text-xs text-faint tabular-nums">{parseValues(valuesText).values.length} columns</p>
          )}
        </div>
      )}
    </Card>
  )
}

const ROW_OPTIONS: Array<{ value: GridRowKind; label: string }> = [
  { value: 'prompts', label: 'Prompts' },
  { value: 'seeds', label: 'Seeds' }
]

const PROMPT_AREA =
  'min-h-20 w-full resize-y rounded-lg border border-border bg-panel px-3 py-2 text-[13px] placeholder:text-faint focus:border-accent focus:outline-none'

export function PromptsCard() {
  const rowKind = useGrid((s) => s.rowKind)
  const prompts = useGrid((s) => s.prompts)
  const seed = useGrid((s) => s.seed)
  const seeds = useGrid((s) => s.seeds)
  const valuesText = useGrid((s) => s.valuesText)
  const setRowKind = useGrid((s) => s.setRowKind)
  const setPrompt = useGrid((s) => s.setPrompt)
  const addPrompt = useGrid((s) => s.addPrompt)
  const removePrompt = useGrid((s) => s.removePrompt)
  const setSeed = useGrid((s) => s.setSeed)
  const rollSeed = useGrid((s) => s.rollSeed)
  const setSeedAt = useGrid((s) => s.setSeedAt)
  const addSeedRow = useGrid((s) => s.addSeedRow)
  const removeSeedRow = useGrid((s) => s.removeSeedRow)
  const rollSeeds = useGrid((s) => s.rollSeeds)

  const columns = parseValues(valuesText).values.length
  const rows = rowCountOf(rowKind, prompts, seeds)
  const seedsErr = rowKind === 'seeds' ? seedsError(seeds) : null
  const summary = (
    <p className="text-xs text-faint tabular-nums">
      {cellCountText(cellCount(columns, rows))} ({columns} columns x {rows} row{rows === 1 ? '' : 's'}). Each image renders like one image of your
      workflow, on the GPU or CPU of the ComfyUI server.
    </p>
  )
  return (
    <Card
      title="Prompts"
      description={
        rowKind === 'seeds'
          ? 'One prompt, one row per seed. The columns differ only by the value, so the rows show how much the seed matters.'
          : 'Each prompt is one row. Every image uses the same seed, so the columns differ only by the value.'
      }
      actions={
        rowKind === 'seeds' ? (
          <Button size="sm" disabled={seeds.length >= GRID_MAX_SEEDS} onClick={addSeedRow}>
            <Plus size={14} />
            Add seed
          </Button>
        ) : (
          <Button size="sm" disabled={prompts.length >= GRID_MAX_PROMPTS} onClick={addPrompt}>
            <Plus size={14} />
            Add prompt
          </Button>
        )
      }
    >
      <div className="space-y-3">
        <Field label="Rows">
          <Segmented label="Rows" value={rowKind} options={ROW_OPTIONS} onChange={setRowKind} />
        </Field>
        {rowKind === 'seeds' ? (
          <>
            <textarea
              aria-label="Prompt"
              rows={3}
              maxLength={GRID_PROMPT_MAX}
              value={prompts[0] ?? ''}
              onChange={(e) => setPrompt(0, e.target.value)}
              placeholder="Prompt"
              className={PROMPT_AREA}
            />
            <div className="space-y-2">
              {seeds.map((s, i) => (
                <div key={i} className="flex items-end gap-2">
                  <div className="w-56">
                    <Field label={`Seed ${i + 1}`}>
                      <NumberInput value={s} min={0} max={Number.MAX_SAFE_INTEGER} step={1} onCommit={(v) => setSeedAt(i, Math.floor(v))} />
                    </Field>
                  </div>
                  {seeds.length > 1 && (
                    <IconButton label={`Remove seed ${i + 1}`} onClick={() => removeSeedRow(i)}>
                      <Trash2 size={14} />
                    </IconButton>
                  )}
                </div>
              ))}
              <Button onClick={rollSeeds}>
                <Shuffle size={16} />
                Random
              </Button>
            </div>
            {seedsErr && (
              <p className="text-[13px] text-danger-text" role="alert">
                {seedsErr}
              </p>
            )}
            {summary}
          </>
        ) : (
          <>
            {prompts.map((p, i) => (
              <div key={i} className="flex items-start gap-2">
                <textarea
                  aria-label={`Prompt ${i + 1}`}
                  rows={3}
                  maxLength={GRID_PROMPT_MAX}
                  value={p}
                  onChange={(e) => setPrompt(i, e.target.value)}
                  placeholder={`Prompt ${i + 1}`}
                  className={PROMPT_AREA}
                />
                {prompts.length > 1 && (
                  <IconButton label={`Remove prompt ${i + 1}`} onClick={() => removePrompt(i)}>
                    <Trash2 size={14} />
                  </IconButton>
                )}
              </div>
            ))}
            <div className="flex flex-wrap items-end gap-3">
              <div className="w-56">
                <Field label="Seed" hint={validSeed(seed) ? undefined : 'Use a whole number from 0 up.'}>
                  <NumberInput value={seed} min={0} max={Number.MAX_SAFE_INTEGER} step={1} onCommit={(v) => setSeed(Math.floor(v))} />
                </Field>
              </div>
              <Button onClick={rollSeed}>
                <Shuffle size={16} />
                Random
              </Button>
            </div>
            {summary}
          </>
        )}
      </div>
    </Card>
  )
}
