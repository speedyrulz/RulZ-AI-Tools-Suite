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
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { isRemoteModelId } from '@shared/remote'
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
  /** Only the settings this candidate changed in its own Adjust settings; they win over "all candidates". */
  own: SettingsEdit
}

/** Settings the user changed, field by field: sampling and load settings. */
export interface SettingsEdit {
  sampling: Partial<SamplingParams>
  load: Partial<LoadConfig>
}

export const EMPTY_EDIT: SettingsEdit = { sampling: {}, load: {} }

const newDraft = (modelId = ''): CandidateDraft => ({ id: uid(), label: '', labelEdited: false, modelId, adapters: [], sampling: 'writing', own: EMPTY_EDIT })

const LOAD_INTEGER_KEYS = ['contextLength', 'threads', 'batchSize', 'ubatchSize', 'draftTokens', 'draftMinTokens', 'seed'] as const

/** Whole numbers, as the main process keeps them, so a ready instance compares equal and is not reloaded. */
export function roundLoad(config: LoadConfig): LoadConfig {
  const out = { ...config } as Record<string, unknown>
  for (const k of LOAD_INTEGER_KEYS) {
    if (typeof out[k] === 'number') out[k] = Math.round(out[k] as number)
  }
  return out as unknown as LoadConfig
}

/** The sampling a candidate runs with: its select's choice, then "all candidates", then its own edits. */
export function effectiveSampling(base: SamplingParams, all: Partial<SamplingParams>, own: Partial<SamplingParams>): SamplingParams {
  return { ...base, ...all, ...own }
}

/** The load settings a candidate runs with, or null (the model's saved settings) when nobody changed any. Adapters are never part of it. */
export function effectiveLoad(saved: LoadConfig, all: Partial<LoadConfig>, own: Partial<LoadConfig>): LoadConfig | null {
  if (Object.keys(all).length === 0 && Object.keys(own).length === 0) return null
  return roundLoad({ ...saved, ...all, ...own, adapters: [] })
}

const sameValue = (a: unknown, b: unknown): boolean => {
  const norm = (v: unknown): unknown =>
    Array.isArray(v) ? v.map(norm) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, norm((v as Record<string, unknown>)[k])])) : v
  return JSON.stringify(norm(a)) === JSON.stringify(norm(b))
}

/** The fields of a load edit that differ from the settings shown (rounded as the main process keeps them); adapters are ignored. */
export function diffLoad(next: LoadConfig, shown: LoadConfig): Partial<LoadConfig> {
  const a = roundLoad(next) as unknown as Record<string, unknown>
  const b = shown as unknown as Record<string, unknown>
  const out: Record<string, unknown> = {}
  for (const k of Object.keys(a)) {
    if (k !== 'adapters' && !sameValue(a[k], b[k])) out[k] = a[k]
  }
  return out as Partial<LoadConfig>
}

const SETTING_NAMES: Record<string, string> = { sampling: 'model sampling', drySequenceBreakers: 'DRY breakers' }

/** A setting key as words: contextLength -> context length. */
export function settingName(key: string): string {
  return SETTING_NAMES[key] ?? key.replace(/([A-Z])/g, ' $1').toLowerCase()
}

/** The settings a candidate changed itself, as words, for the closed section's hint. */
export function ownNames(own: SettingsEdit): string[] {
  return [...Object.keys(own.sampling), ...Object.keys(own.load)].map(settingName)
}

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

/** The load settings of one candidate: the model's saved settings, then "all candidates", then its own changes. */
function CandidateLoad({
  model,
  allLoad,
  own,
  adapters,
  onChange
}: {
  model: ModelEntry
  allLoad: Partial<LoadConfig>
  own: Partial<LoadConfig>
  adapters: LoadAdapter[]
  onChange: (own: Partial<LoadConfig>, adapters: LoadAdapter[]) => void
}) {
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
  const config: LoadConfig = { ...(saved ?? appDefault), ...allLoad, ...own, adapters }
  const estimate = useEstimate(model.id, config)
  const change = (next: LoadConfig): void => {
    const nextAdapters = next.adapters ?? []
    // Only the fields that differ from what is shown become this candidate's own: an adapter-only change from the form's own picker is none.
    const diff = diffLoad(next, config)
    onChange(Object.keys(diff).length ? { ...own, ...diff } : own, nextAdapters)
  }
  if (!saved) return <p className="text-[11px] text-faint">Reading the model's saved load settings…</p>
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold tracking-wide text-faint uppercase">Load settings</span>
        <div className="flex-1" />
        {Object.keys(own).length > 0 && (
          <button
            type="button"
            className="flex items-center gap-1 text-xs text-muted hover:text-text"
            onClick={() => {
              setResets((n) => n + 1)
              onChange({}, adapters)
            }}
          >
            <RotateCcw size={12} aria-hidden /> Use the settings for all candidates
          </button>
        )}
      </div>
      <LoadSettingsForm key={`${resets}|${JSON.stringify(allLoad)}`} model={model} config={config} onChange={change} estimate={estimate} />
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

/** The sampling sliders and variety fields, shared by a candidate and the "all candidates" section. */
function SamplingFields({ sp, onSet }: { sp: SamplingParams; onSet: <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]) => void }) {
  const varietyOn = (sp.dryMultiplier > 0 ? 1 : 0) + (sp.xtcProbability > 0 ? 1 : 0) + (sp.avoidWords.length > 0 ? 1 : 0)
  return (
    <div className="space-y-3">
      <span className="text-[11px] font-semibold tracking-wide text-faint uppercase">Sampling</span>
      <SliderRow label="Temperature" value={sp.temperature} min={0} max={2} step={0.01} onChange={(v) => onSet('temperature', v)} />
      <SliderRow label="Min P" value={sp.minP} min={0} max={1} step={0.01} onChange={(v) => onSet('minP', v)} />
      <SliderRow label="Top P" value={sp.topP} min={0} max={1} step={0.01} onChange={(v) => onSet('topP', v)} />
      <SliderRow label="Top K" hint="0 = off" value={sp.topK} min={0} max={200} step={1} onChange={(v) => onSet('topK', Math.round(v))} />
      <SliderRow label="Repeat penalty" value={sp.repeatPenalty} min={0.8} max={2} step={0.01} onChange={(v) => onSet('repeatPenalty', v)} />
      <p className="text-[11px] text-faint">Every output gets its own seed from the run, so there is no seed to set here.</p>
      <div className="text-[11px] font-semibold tracking-wide text-faint uppercase">
        Repetition and variety
        {varietyOn > 0 && <span className="font-normal normal-case"> · {varietyOn} on</span>}
      </div>
      <VarietyFields sampling={sp} onChange={onSet} />
    </div>
  )
}

/** Settings that every candidate uses, unless it changed the same setting itself. Collapsed until opened. */
function AllCandidatesSettings({ all, onChange }: { all: SettingsEdit; onChange: (all: SettingsEdit) => void }) {
  const appDefault = useApp((s) => s.settings?.defaultLoadConfig) ?? DEFAULT_LOAD_CONFIG
  const [open, setOpen] = useState(false)
  // Bumped on "Clear" so the load form's uncontrolled fields show the starting values again.
  const [resets, setResets] = useState(0)
  const sp: SamplingParams = { ...FICTION_SAMPLING, ...all.sampling }
  const config: LoadConfig = { ...appDefault, ...all.load, adapters: [] }
  const setS = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void => onChange({ ...all, sampling: { ...all.sampling, [key]: value } })
  const change = (next: LoadConfig): void => {
    // Adapters belong to each candidate: whatever the form says about them is ignored.
    const diff = diffLoad(next, config)
    if (Object.keys(diff).length) onChange({ ...all, load: { ...all.load, ...diff } })
  }
  const count = Object.keys(all.sampling).length + Object.keys(all.load).length
  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="rounded-lg border border-border bg-panel-2/50 p-3">
      <Collapsible.Trigger className="flex w-full items-center gap-1 text-[11px] font-semibold tracking-wide text-faint uppercase hover:text-text">
        <ChevronRight size={14} className={open ? 'rotate-90 transition-transform' : 'transition-transform'} />
        Settings for all candidates
        {!open && count > 0 && <span className="font-normal normal-case"> · {count === 1 ? '1 setting set' : `${count} settings set`}</span>}
      </Collapsible.Trigger>
      <Collapsible.Content className="mt-3 space-y-4">
        <div className="flex items-center gap-2">
          <p className="flex-1 text-[11px] text-faint">Every candidate uses these, unless it changed the same setting in its own Adjust settings. Only for this run.</p>
          {count > 0 && (
            <button
              type="button"
              className="flex items-center gap-1 text-xs text-muted hover:text-text"
              onClick={() => {
                setResets((n) => n + 1)
                onChange(EMPTY_EDIT)
              }}
            >
              <RotateCcw size={12} aria-hidden /> Clear
            </button>
          )}
        </div>
        <SamplingFields sp={sp} onSet={setS} />
        <div className="space-y-3">
          <span className="text-[11px] font-semibold tracking-wide text-faint uppercase">Load settings</span>
          <LoadSettingsForm key={resets} model={null} config={config} onChange={change} />
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

function CandidateCard({
  draft,
  index,
  canRemove,
  all,
  onChange,
  onRemove
}: {
  draft: CandidateDraft
  index: number
  canRemove: boolean
  all: SettingsEdit
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
  // The values the sampling fields show: the select's choice, then "all candidates", then this candidate's own edits.
  const sp = effectiveSampling(resolveSampling(draft.sampling, settings, presets), all.sampling, draft.own.sampling)
  const ownSampling = Object.keys(draft.own.sampling).length > 0
  const setS = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void =>
    apply({ own: { ...draft.own, sampling: { ...draft.own.sampling, [key]: value } } })
  const ownList = ownNames(draft.own)
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
          <Select value={draft.modelId} onChange={(e) => apply({ modelId: e.target.value, adapters: [], own: { ...draft.own, load: {} } })}>
            <option value="">Choose a model…</option>
            {models.filter((m) => !m.remote).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Sampling">
          <Select
            value={ownSampling ? 'custom' : draft.sampling}
            onChange={(e) => e.target.value !== 'custom' && apply({ sampling: e.target.value, own: { ...draft.own, sampling: {} } })}
          >
            {ownSampling && <option value="custom">Custom</option>}
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
          {!open && ownList.length > 0 && <span className="font-normal normal-case"> · Own: {ownList.join(', ')}</span>}
        </Collapsible.Trigger>
        <Collapsible.Content className="mt-3 space-y-4">
          <p className="text-[11px] text-faint">Only for this run. Presets and the model's saved settings are not changed. Settings you change here win over the settings for all candidates.</p>
          {ownSampling && (
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-faint">Own sampling: {Object.keys(draft.own.sampling).map(settingName).join(', ')}</span>
              <div className="flex-1" />
              <button type="button" className="flex items-center gap-1 text-xs text-muted hover:text-text" onClick={() => apply({ own: { ...draft.own, sampling: {} } })}>
                <RotateCcw size={12} aria-hidden /> Use the settings for all candidates
              </button>
            </div>
          )}
          <SamplingFields sp={sp} onSet={setS} />
          {model && (
            <CandidateLoad
              key={model.id}
              model={model}
              allLoad={all.load}
              own={draft.own.load}
              adapters={draft.adapters}
              onChange={(load, adapters) => apply({ own: { ...draft.own, load }, adapters })}
            />
          )}
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
  const [drafts, setDrafts] = useState<CandidateDraft[]>(() => [newDraft(isRemoteModelId(activeModelId) ? '' : (activeModelId ?? '')), newDraft()])
  const [seedCount, setSeedCount] = useState(2)
  const [maxTokens, setMaxTokens] = useState(DEFAULT_TOKENS)
  const [editing, setEditing] = useState<BenchPrompt | 'new' | null>(null)
  const [starting, setStarting] = useState(false)
  const [all, setAll] = useState<SettingsEdit>(EMPTY_EDIT)

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
    setStarting(true)
    try {
      const appDefault = settings?.defaultLoadConfig ?? DEFAULT_LOAD_CONFIG
      // Everything is settled here: each candidate is stored with its full sampling and, when anyone changed a load setting, its full load settings.
      const candidates: BenchCandidate[] = await Promise.all(
        labelled.map(async (d) => {
          const needsLoad = Object.keys(all.load).length > 0 || Object.keys(d.own.load).length > 0
          const saved = needsLoad ? await invoke('models:getDefaults', d.modelId).then((r) => r?.config ?? appDefault).catch(() => appDefault) : appDefault
          return {
            id: d.id,
            label: d.label.trim(),
            modelId: d.modelId,
            modelName: models.find((m) => m.id === d.modelId)?.name ?? d.modelId,
            adapters: d.adapters,
            sampling: effectiveSampling(resolveSampling(d.sampling, settings, presets), all.sampling, d.own.sampling),
            load: effectiveLoad(saved, all.load, d.own.load)
          }
        })
      )
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
        <AllCandidatesSettings all={all} onChange={setAll} />
        {drafts.map((d, i) => (
          <CandidateCard
            key={d.id}
            draft={labelled[i]}
            all={all}
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
