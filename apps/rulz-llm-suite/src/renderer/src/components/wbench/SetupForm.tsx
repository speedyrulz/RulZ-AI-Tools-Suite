import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, LoaderCircle, Pencil, Play, Plus, RotateCcw, Trash2, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { FICTION_SAMPLING } from '@shared/novelPrompts'
import type { LoadAdapter, LoadConfig, ModelEntry, SamplingParams } from '@shared/types'
import { BUILTIN_PROMPTS, randomSeeds, type BenchCandidate, type BenchPrompt } from '@shared/writingBench'
import { invoke } from '../../lib/api'
import { uid } from '../../lib/format'
import { candidateSettings, sameLoadConfig } from '../../lib/wbenchRunner'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { useModels } from '../../stores/models'
import { useWbench } from '../../stores/wbench'
import { VarietyFields } from '../chat/InferencePanel'
import { AdapterPicker } from '../models/AdapterPicker'
import { EstimateView, useEstimate } from '../models/EstimateView'
import { LoadSettingsForm } from '../models/LoadSettingsForm'
import { Button, Dialog, Field, IconButton, Input, NumberInput, Select, Slider, Textarea } from '../ui'

const MAX_CANDIDATES = 6
const MAX_SEEDS = 3
const DEFAULT_TOKENS = 600

interface CandidateDraft {
  id: string
  label: string
  /** The user typed the label: it no longer follows the model. */
  labelEdited: boolean
  modelId: string
  adapters: LoadAdapter[]
  /** 'writing', 'app' or `preset:<id>`. */
  sampling: string
  /** Sampling values edited for this candidate; null = whatever `sampling` resolves to. */
  custom: SamplingParams | null
  /** Load settings edited for this candidate; null = the model's saved load settings. */
  load: LoadConfig | null
}

const newDraft = (modelId = ''): CandidateDraft => ({ id: uid(), label: '', labelEdited: false, modelId, adapters: [], sampling: 'writing', custom: null, load: null })

/** The sampling a select choice stands for: 'writing', 'app' or `preset:<id>`. */
export function resolveSampling(key: string, settings: { defaultSampling: SamplingParams } | null | undefined, presets: { id: string; sampling: SamplingParams }[]): SamplingParams {
  if (key === 'app' && settings) return settings.defaultSampling
  const preset = key.startsWith('preset:') ? presets.find((p) => p.id === key.slice(7)) : undefined
  return preset?.sampling ?? FICTION_SAMPLING
}

function SliderRow({ label, hint, value, min, max, step, onChange }: { label: string; hint?: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <Field label={label} hint={hint}>
      <div className="flex items-center gap-2.5">
        <Slider value={value} min={min} max={max} step={step} onChange={onChange} />
        <NumberInput className="w-18" value={value} min={min} max={max} step={step} onChange={(v) => onChange(Math.min(max, Math.max(min, v ?? min)))} />
      </div>
    </Field>
  )
}

/** The load settings of one candidate: the model's saved settings until the user changes something. */
function CandidateLoad({ model, load, adapters, onChange }: { model: ModelEntry; load: LoadConfig | null; adapters: LoadAdapter[]; onChange: (load: LoadConfig | null, adapters: LoadAdapter[]) => void }) {
  const appDefault = useApp((s) => s.settings?.defaultLoadConfig) ?? DEFAULT_LOAD_CONFIG
  // null until the model's saved settings are read: edits must start from them, not from the app default.
  const [saved, setSaved] = useState<LoadConfig | null>(null)
  // Bumped on "Use the model's saved settings" so the form's uncontrolled fields show the saved values again.
  const [resets, setResets] = useState(0)
  useEffect(() => {
    let live = true
    invoke('models:getDefaults', model.id)
      .then((d) => live && setSaved(d?.config ?? appDefault))
      .catch(() => live && setSaved(appDefault))
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.id])
  // The form's own adapter list is the candidate's: adapters stay chosen in one place, whichever picker is used.
  const config: LoadConfig = { ...(load ?? saved ?? appDefault), adapters }
  const estimate = useEstimate(model.id, config)
  const change = (next: LoadConfig): void => {
    const nextAdapters = next.adapters ?? []
    // Whole numbers, as the main process keeps them, so a ready instance compares equal and is not reloaded.
    const rounded = { ...next } as Record<string, unknown>
    for (const k of ['contextLength', 'threads', 'batchSize', 'ubatchSize', 'draftTokens', 'draftMinTokens', 'seed'] as const) {
      if (typeof rounded[k] === 'number') rounded[k] = Math.round(rounded[k] as number)
    }
    const cfg = { ...(rounded as unknown as LoadConfig), adapters: [] }
    // Only the adapters changed (the form's own adapter picker): still the model's saved settings.
    if (!load && saved && sameLoadConfig(cfg, saved)) return onChange(null, nextAdapters)
    onChange(cfg, nextAdapters)
  }
  if (!saved) return <p className="text-[11px] text-faint">Reading the model's saved load settings…</p>
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold tracking-wide text-faint uppercase">Load settings</span>
        <div className="flex-1" />
        {load && (
          <button
            type="button"
            className="flex items-center gap-1 text-xs text-muted hover:text-text"
            onClick={() => {
              setResets((n) => n + 1)
              onChange(null, adapters)
            }}
          >
            <RotateCcw size={12} aria-hidden /> Use the model's saved settings
          </button>
        )}
      </div>
      <LoadSettingsForm key={resets} model={model} config={config} onChange={change} estimate={estimate} />
      <EstimateView estimate={estimate} />
    </div>
  )
}

/** The label a candidate gets until the user types one: the model, plus its adapters. */
export function autoLabel(model: ModelEntry | undefined, adapters: LoadAdapter[], adapterNames: Map<string, string>): string {
  if (!model) return ''
  const extra = adapters.map((a) => adapterNames.get(a.id) ?? a.id)
  return extra.length ? `${model.name} + ${extra.join(', ')}` : model.name
}

/** Outputs and pairs a setup makes: prompts × candidates × seeds, and two candidates per prompt and seed. */
export function planCounts(prompts: number, candidates: number, seeds: number): { outputs: number; pairs: number } {
  return { outputs: prompts * candidates * seeds, pairs: prompts * seeds * ((candidates * (candidates - 1)) / 2) }
}

function PromptDialog({ prompt, onClose }: { prompt: BenchPrompt | 'new' | null; onClose: () => void }) {
  const prompts = useWbench((s) => s.prompts)
  const savePrompts = useWbench((s) => s.savePrompts)
  const existing = prompt && prompt !== 'new' ? prompt : null
  const [title, setTitle] = useState(existing?.title ?? '')
  const [text, setText] = useState(existing?.text ?? '')
  const [saving, setSaving] = useState(false)
  const ok = title.trim().length > 0 && text.trim().length > 0
  const save = async (): Promise<void> => {
    setSaving(true)
    const item: BenchPrompt = { id: existing?.id ?? `user-${uid()}`, title: title.trim(), text: text.trim(), builtin: false }
    const next = existing ? prompts.map((p) => (p.id === existing.id ? item : p)) : [...prompts, item]
    const done = await savePrompts(next)
    setSaving(false)
    if (done) onClose()
  }
  return (
    <Dialog
      open={prompt !== null}
      onOpenChange={(o) => !o && onClose()}
      title={existing ? 'Edit prompt' : 'New prompt'}
      description="A scene brief the models write from: two to four sentences works well."
      footer={
        <>
          <Button size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" variant="primary" disabled={!ok || saving} onClick={() => void save()}>
            Save prompt
          </Button>
        </>
      }
    >
      <div className="space-y-3 px-5 py-4">
        <Field label="Title">
          <Input value={title} maxLength={80} onChange={(e) => setTitle(e.target.value)} autoFocus />
        </Field>
        <Field label="Brief">
          <Textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} />
        </Field>
      </div>
    </Dialog>
  )
}

function CandidateCard({
  draft,
  index,
  canRemove,
  onChange,
  onRemove
}: {
  draft: CandidateDraft
  index: number
  canRemove: boolean
  onChange: (d: CandidateDraft) => void
  onRemove: () => void
}) {
  const models = useModels((s) => s.models)
  const allAdapters = useModels((s) => s.adapters)
  const presets = useChat((s) => s.presets)
  const settings = useApp((s) => s.settings)
  const [open, setOpen] = useState(false)
  const model = models.find((m) => m.id === draft.modelId)
  const names = useMemo(() => new Map(allAdapters.map((a) => [a.id, a.name])), [allAdapters])
  // The values the sampling fields show: the edited ones, else what the select stands for.
  const sp = draft.custom ?? resolveSampling(draft.sampling, settings, presets)
  const setS = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void => apply({ custom: { ...sp, [key]: value } })
  const varietyOn = (sp.dryMultiplier > 0 ? 1 : 0) + (sp.xtcProbability > 0 ? 1 : 0) + (sp.avoidWords.length > 0 ? 1 : 0)
  const changed = [draft.custom ? 'Custom sampling' : '', draft.load ? 'Custom load settings' : ''].filter(Boolean)
  const apply = (patch: Partial<CandidateDraft>): void => {
    const next = { ...draft, ...patch }
    if (!next.labelEdited) next.label = autoLabel(models.find((m) => m.id === next.modelId), next.adapters, names)
    onChange(next)
  }
  return (
    <div className="space-y-3 rounded-lg border border-border bg-panel-2/50 p-3">
      <div className="flex items-center gap-2">
        <span className="text-[13px] font-medium">Candidate {index + 1}</span>
        <div className="flex-1" />
        {canRemove && (
          <IconButton label={`Remove candidate ${index + 1}`} onClick={onRemove}>
            <X size={15} aria-hidden />
          </IconButton>
        )}
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Field label="Model">
          <Select value={draft.modelId} onChange={(e) => apply({ modelId: e.target.value, adapters: [], load: null })}>
            <option value="">Choose a model…</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Sampling">
          <Select value={draft.custom ? 'custom' : draft.sampling} onChange={(e) => e.target.value !== 'custom' && apply({ sampling: e.target.value, custom: null })}>
            {draft.custom && <option value="custom">Custom</option>}
            <option value="writing">Writing defaults</option>
            <option value="app">App defaults (Chat settings)</option>
            {presets.map((p) => (
              <option key={p.id} value={`preset:${p.id}`}>
                Preset: {p.name}
              </option>
            ))}
          </Select>
        </Field>
      </div>
      <Field label="Name in the scoreboard" hint="Hidden until you have voted on a pair">
        <Input value={draft.label} maxLength={60} placeholder="e.g. base or lora-v2" onChange={(e) => apply({ label: e.target.value, labelEdited: true })} />
      </Field>
      {model && <AdapterPicker model={model} adapters={draft.adapters} onChange={(adapters) => apply({ adapters })} />}
      <Collapsible.Root open={open} onOpenChange={setOpen}>
        <Collapsible.Trigger className="flex w-full items-center gap-1 text-[11px] font-semibold tracking-wide text-faint uppercase hover:text-text">
          <ChevronRight size={14} className={open ? 'rotate-90 transition-transform' : 'transition-transform'} />
          Adjust settings
          {!open && changed.length > 0 && <span className="font-normal normal-case"> · {changed.join(', ')}</span>}
        </Collapsible.Trigger>
        <Collapsible.Content className="mt-3 space-y-4">
          <p className="text-[11px] text-faint">Only for this run. Presets and the model's saved settings are not changed.</p>
          <div className="space-y-3">
            <span className="text-[11px] font-semibold tracking-wide text-faint uppercase">Sampling</span>
            <SliderRow label="Temperature" value={sp.temperature} min={0} max={2} step={0.01} onChange={(v) => setS('temperature', v)} />
            <SliderRow label="Min P" value={sp.minP} min={0} max={1} step={0.01} onChange={(v) => setS('minP', v)} />
            <SliderRow label="Top P" value={sp.topP} min={0} max={1} step={0.01} onChange={(v) => setS('topP', v)} />
            <SliderRow label="Top K" hint="0 = off" value={sp.topK} min={0} max={200} step={1} onChange={(v) => setS('topK', Math.round(v))} />
            <SliderRow label="Repeat penalty" value={sp.repeatPenalty} min={0.8} max={2} step={0.01} onChange={(v) => setS('repeatPenalty', v)} />
            <p className="text-[11px] text-faint">Every output gets its own seed from the run, so there is no seed to set here.</p>
            <div className="text-[11px] font-semibold tracking-wide text-faint uppercase">
              Repetition and variety
              {varietyOn > 0 && <span className="font-normal normal-case"> · {varietyOn} on</span>}
            </div>
            <VarietyFields sampling={sp} onChange={setS} />
          </div>
          {model && <CandidateLoad key={model.id} model={model} load={draft.load} adapters={draft.adapters} onChange={(load, adapters) => apply({ load, adapters })} />}
        </Collapsible.Content>
      </Collapsible.Root>
    </div>
  )
}

/** Sets up a run: prompts, 2 to 6 candidates, seeds and reply size. One Start button. */
export function SetupForm({ onCancel }: { onCancel: () => void }) {
  const userPrompts = useWbench((s) => s.prompts)
  const savePrompts = useWbench((s) => s.savePrompts)
  const create = useWbench((s) => s.create)
  const running = useWbench((s) => s.running)
  const models = useModels((s) => s.models)
  const activeModelId = useModels((s) => s.activeModelId)
  const presets = useChat((s) => s.presets)
  const settings = useApp((s) => s.settings)
  const allAdapters = useModels((s) => s.adapters)

  const [label, setLabel] = useState('')
  const [picked, setPicked] = useState<Set<string>>(() => new Set(BUILTIN_PROMPTS.slice(0, 4).map((p) => p.id)))
  const [drafts, setDrafts] = useState<CandidateDraft[]>(() => [newDraft(activeModelId ?? ''), newDraft()])
  const [seedCount, setSeedCount] = useState(2)
  const [maxTokens, setMaxTokens] = useState(DEFAULT_TOKENS)
  const [editing, setEditing] = useState<BenchPrompt | 'new' | null>(null)
  const [starting, setStarting] = useState(false)

  const allPrompts = useMemo(() => [...BUILTIN_PROMPTS, ...userPrompts], [userPrompts])
  const chosen = allPrompts.filter((p) => picked.has(p.id))
  const names = useMemo(() => new Map(allAdapters.map((a) => [a.id, a.name])), [allAdapters])
  // The default-named candidate for the active model: filled once the model list is known.
  const labelled = drafts.map((d) => (d.labelEdited ? d : { ...d, label: autoLabel(models.find((m) => m.id === d.modelId), d.adapters, names) }))
  const labels = labelled.map((d) => d.label.trim())
  const problem =
    chosen.length === 0
      ? 'Tick at least one prompt.'
      : labelled.some((d) => !d.modelId)
        ? 'Choose a model for every candidate.'
        : labels.some((l) => !l)
          ? 'Give every candidate a name.'
          : new Set(labels.map((l) => l.toLowerCase())).size !== labels.length
            ? 'Candidate names must differ, so the scoreboard can tell them apart.'
            : null
  const counts = planCounts(chosen.length, drafts.length, seedCount)

  const togglePrompt = (id: string, on: boolean): void =>
    setPicked((s) => {
      const n = new Set(s)
      if (on) n.add(id)
      else n.delete(id)
      return n
    })
  const deletePrompt = async (p: BenchPrompt): Promise<void> => {
    if (await savePrompts(userPrompts.filter((x) => x.id !== p.id))) togglePrompt(p.id, false)
  }

  const start = async (): Promise<void> => {
    if (problem || running || starting) return
    const candidates: BenchCandidate[] = labelled.map((d) => ({
      id: d.id,
      label: d.label.trim(),
      modelId: d.modelId,
      modelName: models.find((m) => m.id === d.modelId)?.name ?? d.modelId,
      adapters: d.adapters,
      ...candidateSettings({ sampling: d.custom, load: d.load }, resolveSampling(d.sampling, settings, presets))
    }))
    setStarting(true)
    try {
      await create({ label, prompts: chosen, candidates, seeds: randomSeeds(seedCount, Math.random), maxTokens })
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Field label="Name of this run" hint="Optional">
        <Input value={label} maxLength={80} placeholder="e.g. base vs. my style adapter" onChange={(e) => setLabel(e.target.value)} />
      </Field>

      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Prompts</h2>
          <span className="text-[11.5px] text-faint tabular-nums">{chosen.length} ticked</span>
          <div className="flex-1" />
          <Button size="sm" onClick={() => setEditing('new')}>
            <Plus size={14} aria-hidden /> Add your own
          </Button>
        </div>
        <div className="divide-y divide-border/60 rounded-lg border border-border">
          {allPrompts.map((p) => (
            <div key={p.id} className="flex items-start gap-3 px-3 py-2">
              <input type="checkbox" className="mt-1 shrink-0" id={`wb-${p.id}`} checked={picked.has(p.id)} onChange={(e) => togglePrompt(p.id, e.target.checked)} />
              <label htmlFor={`wb-${p.id}`} className="min-w-0 flex-1 cursor-pointer">
                <div className="text-[13px] font-medium">
                  {p.title}
                  {p.builtin ? '' : <span className="ml-2 text-[11px] font-normal text-faint">Yours</span>}
                </div>
                <div className="line-clamp-2 text-[12px] text-muted">{p.text}</div>
              </label>
              {!p.builtin && (
                <div className="flex shrink-0 gap-0.5">
                  <IconButton label={`Edit ${p.title}`} onClick={() => setEditing(p)}>
                    <Pencil size={14} aria-hidden />
                  </IconButton>
                  <IconButton label={`Delete ${p.title}`} onClick={() => void deletePrompt(p)}>
                    <Trash2 size={14} aria-hidden />
                  </IconButton>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Candidates</h2>
          <span className="text-[11.5px] text-faint">2 to {MAX_CANDIDATES}. The same model with different adapters or sampling counts as separate candidates.</span>
        </div>
        {drafts.map((d, i) => (
          <CandidateCard
            key={d.id}
            draft={labelled[i]}
            index={i}
            canRemove={drafts.length > 2}
            onChange={(next) => setDrafts((l) => l.map((x) => (x.id === next.id ? next : x)))}
            onRemove={() => setDrafts((l) => l.filter((x) => x.id !== d.id))}
          />
        ))}
        {drafts.length < MAX_CANDIDATES && (
          <Button size="sm" onClick={() => setDrafts((l) => [...l, newDraft()])}>
            <Plus size={14} aria-hidden /> Add a candidate
          </Button>
        )}
      </section>

      <section className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px]">
        <span className="flex items-center gap-2">
          Seeds
          <NumberInput className="w-14" value={seedCount} min={1} max={MAX_SEEDS} onChange={(v) => setSeedCount(Math.min(MAX_SEEDS, Math.max(1, Math.round(v ?? 1))))} />
          <span className="text-[11.5px] text-faint">each prompt is written once per seed</span>
        </span>
        <span className="flex items-center gap-2">
          Max tokens per output
          <NumberInput className="w-20" value={maxTokens} min={50} max={8000} onChange={(v) => setMaxTokens(Math.min(8000, Math.max(50, Math.round(v ?? DEFAULT_TOKENS))))} />
        </span>
      </section>

      <div className="flex flex-wrap items-center gap-3 border-t border-border pt-4">
        <Button size="sm" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <div className="flex-1 text-right text-[12px] text-faint tabular-nums">{problem ?? `${counts.outputs} outputs to write, ${counts.pairs} pairs to vote on`}</div>
        <Button variant="primary" disabled={!!problem || running || starting} onClick={() => void start()}>
          {starting ? <LoaderCircle size={14} className="animate-spin" aria-hidden /> : <Play size={14} aria-hidden />} Start
        </Button>
      </div>
      <PromptDialog key={editing === 'new' ? 'new' : (editing?.id ?? 'none')} prompt={editing} onClose={() => setEditing(null)} />
    </div>
  )
}
