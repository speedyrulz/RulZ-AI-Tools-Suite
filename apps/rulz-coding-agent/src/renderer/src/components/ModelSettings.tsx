import { useEffect, useState, type ReactNode } from 'react'
import { DEFAULT_MAX_OUTPUT_TOKENS, KV_CACHE_TYPES, REASONING_EFFORTS, recommendedSampling } from '@shared/defaults'
import { parseTensorSplit, rebalanceSplit, toPercents } from '@shared/split'
import type { KvCacheType, LoadEstimate, LoadSettings, ModelLoadInfo, ReasoningEffort, SamplingSettings } from '@shared/types'
import { invoke } from '../lib/api'
import { formatBytes, formatTokens } from '../lib/format'
import { useApp } from '../stores/app'
import { Badge, Button, Field, Input, NumberInput, Select, Switch } from './ui'

type NumericKey = 'temperature' | 'topP' | 'topK' | 'repeatPenalty' | 'maxOutputTokens'

const SAMPLING_FIELDS: { key: NumericKey; label: string; hint: string; min: number; max: number; step: number }[] = [
  { key: 'temperature', label: 'Creativity (temperature)', hint: 'Lower = more predictable code', min: 0, max: 2, step: 0.05 },
  { key: 'topP', label: 'Top P', hint: 'Only consider the likeliest words that add up to this share', min: 0, max: 1, step: 0.05 },
  { key: 'topK', label: 'Top K', hint: 'Only consider this many likeliest words (0 = no limit)', min: 0, max: 1000, step: 1 },
  { key: 'repeatPenalty', label: 'Repeat penalty', hint: 'Above 1 discourages repeating itself', min: 0.5, max: 2, step: 0.01 },
  { key: 'maxOutputTokens', label: 'Longest answer (tokens)', hint: 'One reply or tool call at most', min: 256, max: 131072, step: 256 }
]

function SubHeading({ children }: { children: ReactNode }) {
  return <div className="pt-2 text-[12px] font-medium text-muted">{children}</div>
}

/** How the model writes: the owner's values over the recommended ones, saved per model. */
function Sampling({ model }: { model: string }) {
  const all = useApp((s) => s.settings?.modelSampling ?? {})
  const update = useApp((s) => s.updateSettings)
  const own = all[model] ?? {}
  const rec: SamplingSettings = { maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS, ...recommendedSampling(model) }

  const set = <K extends keyof SamplingSettings>(key: K, v: SamplingSettings[K] | undefined): void => {
    const next: SamplingSettings = { ...own }
    if (v === undefined || v === '') delete next[key]
    else next[key] = v
    const rest = { ...all }
    if (Object.keys(next).length) rest[model] = next
    else delete rest[model]
    void update({ modelSampling: rest })
  }
  const thinkingOff = own.thinking === 'off'

  return (
    <div className="space-y-1">
      {SAMPLING_FIELDS.map((f) => (
        <Field key={f.key} label={f.label} hint={f.hint} inline>
          <NumberInput
            label={f.label}
            value={own[f.key] ?? null}
            placeholder={rec[f.key] !== undefined ? String(rec[f.key]) : 'Default'}
            min={f.min}
            max={f.max}
            step={f.step}
            onChange={(v) => (v === null || (v >= f.min && v <= f.max)) && set(f.key, v ?? undefined)}
          />
        </Field>
      ))}

      <SubHeading>Thinking</SubHeading>
      <Field label="Think before answering" hint="For models that can reason first (Qwen3, Gemma 4…); coder models usually can't" inline>
        <Select
          aria-label="Think before answering"
          className="h-7.5 w-44 text-[13px]"
          value={own.thinking ?? ''}
          onChange={(e) => set('thinking', (e.target.value || undefined) as SamplingSettings['thinking'])}
        >
          <option value="">Model's default</option>
          <option value="on">On</option>
          <option value="off">Off</option>
        </Select>
      </Field>
      <Field label="Reasoning effort" hint="For models whose template takes one" inline>
        <Select
          aria-label="Reasoning effort"
          className="h-7.5 w-44 text-[13px]"
          disabled={thinkingOff}
          value={own.reasoningEffort ?? ''}
          onChange={(e) => set('reasoningEffort', (e.target.value || undefined) as ReasoningEffort | undefined)}
        >
          <option value="">Template's default</option>
          {REASONING_EFFORTS.map((r) => (
            <option key={r} value={r}>
              {r[0].toUpperCase() + r.slice(1)}
            </option>
          ))}
        </Select>
      </Field>
      <Field
        label="Thinking budget (tokens)"
        hint={
          own.thinkingBudget === 0 && !thinkingOff
            ? <span className="text-warn">0 skips thinking, so no thoughts will show. Clear it for no limit.</span>
            : 'Most it may think before it must answer. Empty = no limit, 0 = skip thinking.'
        }
        inline
      >
        <NumberInput
          label="Thinking budget"
          value={thinkingOff ? null : (own.thinkingBudget ?? null)}
          placeholder="No limit"
          min={0}
          max={131072}
          step={256}
          onChange={(v) => !thinkingOff && (v === null || v >= 0) && set('thinkingBudget', v === null ? undefined : Math.round(v))}
        />
      </Field>
      {own.thinkingBudget !== undefined && !thinkingOff && (
        <Field label="When the budget runs out, add" hint="Optional nudge, e.g. “Time to answer.”" inline>
          <Input
            aria-label="Budget message"
            className="h-7.5 w-48 text-[13px]"
            maxLength={500}
            defaultValue={own.thinkingBudgetMessage ?? ''}
            placeholder="Nothing"
            onBlur={(e) => e.target.value !== (own.thinkingBudgetMessage ?? '') && set('thinkingBudgetMessage', e.target.value.trim() || undefined)}
          />
        </Field>
      )}

      {Object.keys(own).length > 0 && (
        <Button size="sm" variant="ghost" onClick={() => void update({ modelSampling: Object.fromEntries(Object.entries(all).filter(([k]) => k !== model)) })}>
          Back to recommended
        </Button>
      )}
    </div>
  )
}

/** Only the settings that differ, so an older RulZ LLM Suite is never sent fields it doesn't know. */
function changes(draft: LoadSettings, saved: LoadSettings): Partial<LoadSettings> {
  return Object.fromEntries(
    Object.entries(draft).filter(([k, v]) => JSON.stringify(v) !== JSON.stringify(saved[k as keyof LoadSettings]))
  ) as Partial<LoadSettings>
}

const VERDICT = {
  fits: { tone: 'ok', text: 'Fits in VRAM' },
  partial: { tone: 'warn', text: 'Partly on CPU (slower)' },
  tight: { tone: 'danger', text: 'Likely out of memory' },
  unknown: { tone: 'neutral', text: 'VRAM unknown' }
} as const

/** RulZ LLM Suite's estimate for the settings being edited (asked again shortly after each change). */
function useEstimate(model: string, info: ModelLoadInfo | null, draft: LoadSettings | null): LoadEstimate | undefined {
  const [preview, setPreview] = useState<LoadEstimate | undefined>()
  const patch = info && draft ? changes(draft, info.config) : {}
  const key = JSON.stringify(patch)
  useEffect(() => {
    setPreview(undefined)
    if (!info?.estimate || key === '{}') return
    let cancelled = false
    const t = setTimeout(() => {
      invoke('models:previewLoadSettings', model, patch).then(
        (i) => !cancelled && setPreview(i.estimate),
        () => undefined
      )
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(t)
    }
  }, [model, info, key])
  return key === '{}' ? info?.estimate : preview
}

function GpuSplit({ info, draft, estimate, change }: { info: ModelLoadInfo; draft: LoadSettings; estimate?: LoadEstimate; change: (p: Partial<LoadSettings>) => void }) {
  const gpus = info.gpus
  // With no split saved yet, start in proportion to each card's memory.
  const totalMiB = gpus.reduce((sum, g) => sum + g.memTotalMiB, 0) || 1
  const percents = toPercents(parseTensorSplit(draft.tensorSplit ?? '', gpus.length) ?? gpus.map((g) => g.memTotalMiB / totalMiB))
  const setMode = (mode: LoadSettings['splitMode']): void =>
    change(mode === 'custom' && !draft.tensorSplit?.trim() ? { splitMode: mode, tensorSplit: percents.join(',') } : { splitMode: mode })

  return (
    <>
      <SubHeading>GPU split</SubHeading>
      <Field label="Use the graphics cards" hint="How the model is spread over your GPUs" inline>
        <Select aria-label="GPU split" className="h-7.5 w-40 text-[13px]" value={draft.splitMode} onChange={(e) => setMode(e.target.value as LoadSettings['splitMode'])}>
          <option value="auto">All, automatically</option>
          <option value="single">One GPU only</option>
          <option value="custom">Custom split</option>
        </Select>
      </Field>
      {draft.splitMode === 'single' && (
        <Field label="Which GPU" inline>
          <Select aria-label="Which GPU" className="h-7.5 w-56 text-[13px]" value={draft.mainGpu ?? 0} onChange={(e) => change({ mainGpu: Number(e.target.value) })}>
            {gpus.map((g) => (
              <option key={g.index} value={g.index}>
                GPU {g.index}: {g.name}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {draft.splitMode === 'custom' &&
        gpus.map((g, i) => {
          const est = estimate?.perGpu.find((p) => p.index === g.index)
          return (
            <div key={g.index} className="flex items-center gap-3 py-1 text-[13px]">
              <span className="w-40 min-w-0 truncate" title={g.name}>
                GPU {g.index}: {g.name}
              </span>
              <input
                type="range"
                aria-label={`Share for GPU ${g.index}`}
                className="min-w-0 flex-1 accent-[var(--accent)]"
                min={0}
                max={100}
                value={percents[i]}
                onChange={(e) => change({ tensorSplit: rebalanceSplit(percents, i, Number(e.target.value)).join(',') })}
              />
              <span className="w-10 text-right tabular-nums">{percents[i]}%</span>
              <span className={est && est.bytes > est.freeBytes ? 'w-32 text-right text-[11px] text-danger' : 'w-32 text-right text-[11px] text-faint'}>
                {est ? `${formatBytes(est.bytes)} of ${formatBytes(est.freeBytes)} free` : `${formatBytes(g.memTotalMiB * 1024 * 1024)}`}
              </span>
            </div>
          )
        })}
    </>
  )
}

function DraftTuning({ draft, change, builtIn }: { draft: LoadSettings; change: (p: Partial<LoadSettings>) => void; builtIn: boolean }) {
  const num = (key: 'draftTokens' | 'draftMinTokens', label: string, hint: string, min: number, placeholder: string) => (
    <Field label={label} hint={hint} inline>
      <NumberInput label={label} value={draft[key] ?? null} placeholder={placeholder} min={min} max={64} onChange={(v) => (v === null || (v >= min && v <= 64)) && change({ [key]: v === null ? null : Math.round(v) })} />
    </Field>
  )
  return (
    <>
      <SubHeading>Draft model (MTP)</SubHeading>
      {num('draftTokens', 'Tokens to draft per step', 'More can be faster if most are accepted', 1, 'Default')}
      {num('draftMinTokens', 'Fewest tokens worth drafting', 'Skip drafting below this', 0, '0')}
      <Field label="Least draft confidence" hint="Stop drafting when the draft is less sure than this (0 to 1)" inline>
        <NumberInput label="Least draft confidence" value={draft.draftPMin ?? null} placeholder="Default" min={0} max={1} step={0.05} onChange={(v) => (v === null || (v >= 0 && v <= 1)) && change({ draftPMin: v })} />
      </Field>
      <Field label="Draft sampling" inline>
        <Select aria-label="Draft sampling" className="h-7.5 w-36 text-[13px]" value={draft.draftSampling} onChange={(e) => change({ draftSampling: e.target.value as LoadSettings['draftSampling'] })}>
          <option value="auto">Auto</option>
          <option value="greedy">Greedy</option>
          <option value="probabilistic">Probabilistic</option>
        </Select>
      </Field>
      {/* Built-in MTP layers run on the model's own weights, wherever those are. */}
      {!builtIn && (
        <Field label="Run the draft on" inline>
          <Select aria-label="Run the draft on" className="h-7.5 w-36 text-[13px]" value={draft.draftGpu} onChange={(e) => change({ draftGpu: e.target.value as LoadSettings['draftGpu'] })}>
            <option value="auto">Auto</option>
            <option value="gpu">GPU</option>
            <option value="cpu">CPU</option>
          </Select>
        </Field>
      )}
      <Field label="Draft cache type (K / V)" inline>
        <div className="flex gap-1.5">
          {(['draftCacheTypeK', 'draftCacheTypeV'] as const).map((k) => (
            <Select key={k} aria-label={k === 'draftCacheTypeK' ? 'Draft K cache type' : 'Draft V cache type'} className="h-7.5 w-20 text-[13px]" value={draft[k]} onChange={(e) => change({ [k]: e.target.value as KvCacheType })}>
              {KV_CACHE_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </Select>
          ))}
        </div>
      </Field>
    </>
  )
}

/** How RulZ LLM Suite loads the model. Saving restarts a loaded model, so changes wait for the button. */
function Loading({ model }: { model: string }) {
  const [info, setInfo] = useState<ModelLoadInfo | null>(null)
  const [draft, setDraft] = useState<LoadSettings | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const estimate = useEstimate(model, info, draft)

  useEffect(() => {
    setInfo(null)
    setDraft(null)
    setError('')
    invoke('models:loadSettings', model).then(
      (i) => {
        setInfo(i)
        setDraft(i.config)
      },
      (e: Error) => setError(e.message)
    )
  }, [model])

  if (error && !draft) return <p className="text-[13px] text-warn">{error}</p>
  if (!info || !draft) return <p className="text-[13px] text-faint">Loading…</p>

  const change = (patch: Partial<LoadSettings>): void => setDraft({ ...draft, ...patch })
  const patch = changes(draft, info.config)
  const dirty = Object.keys(patch).length > 0
  const loaded = info.state === 'ready' || info.state === 'loading'
  const gpuMode = typeof draft.gpuLayers === 'number' ? 'number' : draft.gpuLayers
  // Older RulZ LLM Suite versions don't send these settings, so their controls stay hidden.
  const canSplit = draft.splitMode !== undefined && info.gpus.length > 1
  const choices = info.draftChoices ?? []
  const pickedDraft = draft.draftFile ? choices.find((c) => c.id === draft.draftFile) : undefined
  const hasAnyDraft = info.hasDraft !== false || !!draft.draftFile
  const canPickDraft = info.draftChoices !== undefined && draft.useDraft && (choices.length > 0 || !!info.defaultDraft)
  const canTuneDraft = draft.draftSampling !== undefined && hasAnyDraft && draft.useDraft
  const builtInDraft = pickedDraft ? pickedDraft.builtIn : !draft.draftFile && !!info.defaultDraft?.builtIn

  const save = async (): Promise<void> => {
    setSaving(true)
    setError('')
    try {
      const next = await invoke('models:setLoadSettings', model, patch)
      setInfo(next)
      setDraft(next.config)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-1">
      <Field
        label="Context length (tokens)"
        hint={`How much I can keep in mind at once${info.maxContext ? `; this model handles up to ${formatTokens(info.maxContext)}` : ''}. Empty = as much as fits.`}
        inline
      >
        <NumberInput
          label="Context length"
          className="w-28"
          value={draft.contextLength}
          placeholder="Fit to memory"
          min={512}
          max={info.maxContext}
          step={1024}
          onChange={(v) => change({ contextLength: v === null ? null : Math.round(v) })}
        />
      </Field>
      <Field label="GPU layers" hint="How much of the model goes on the graphics card" inline>
        <div className="flex items-center gap-1.5">
          <Select
            className="h-7.5 w-28 text-[13px]"
            value={gpuMode}
            onChange={(e) => {
              const m = e.target.value
              change({ gpuLayers: m === 'number' ? (typeof draft.gpuLayers === 'number' ? draft.gpuLayers : 32) : (m as 'auto' | 'all') })
            }}
          >
            <option value="auto">Auto</option>
            <option value="all">All</option>
            <option value="number">Number…</option>
          </Select>
          {gpuMode === 'number' && (
            <NumberInput
              className="w-20"
              value={draft.gpuLayers as number}
              min={0}
              max={1000}
              onChange={(v) => v !== null && v >= 0 && change({ gpuLayers: Math.round(v) })}
            />
          )}
        </div>
      </Field>
      <Field label="Flash attention" hint="Faster and uses less memory on most GPUs" inline>
        <Select className="h-7.5 w-28 text-[13px]" value={draft.flashAttention} onChange={(e) => change({ flashAttention: e.target.value as LoadSettings['flashAttention'] })}>
          <option value="auto">Auto</option>
          <option value="on">On</option>
          <option value="off">Off</option>
        </Select>
      </Field>
      <Field label="Memory cache type (K / V)" hint="q8_0 roughly halves the cache's memory with little quality loss" inline>
        <div className="flex gap-1.5">
          {(['cacheTypeK', 'cacheTypeV'] as const).map((k) => (
            <Select key={k} aria-label={k === 'cacheTypeK' ? 'K cache type' : 'V cache type'} className="h-7.5 w-20 text-[13px]" value={draft[k]} onChange={(e) => change({ [k]: e.target.value as KvCacheType })}>
              {KV_CACHE_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </Select>
          ))}
        </div>
      </Field>
      <Field
        label="Speed up with a draft model"
        hint={
          info.defaultDraft
            ? `Uses ${info.defaultDraft.builtIn ? 'the MTP layers built into the model' : info.defaultDraft.name} unless you pick another`
            : hasAnyDraft
              ? 'Only works if the model comes with one'
              : choices.length
                ? 'No draft found with this model; pick one below'
                : 'This model has no draft (MTP) head'
        }
        inline
      >
        <Switch checked={draft.useDraft} onCheckedChange={(v) => change({ useDraft: v })} label="Speed up with a draft model" />
      </Field>
      {canSplit && <GpuSplit info={info} draft={draft} estimate={estimate} change={change} />}
      {canPickDraft && (
        <Field label="Draft model" hint="Must come from the same base model, or it won't load or will only slow things down" inline>
          <Select
            aria-label="Draft model"
            className="h-7.5 w-56 text-[13px]"
            value={draft.draftFile ?? ''}
            onChange={(e) => change({ draftFile: e.target.value || null })}
          >
            <option value="">Automatic ({info.defaultDraft ? (info.defaultDraft.builtIn ? 'built-in MTP layers' : info.defaultDraft.name) : 'none found'})</option>
            {choices.map((c) => (
              <option key={c.id} value={c.id}>
                {c.builtIn ? c.name : `${c.name} (${c.folder.split('/').pop() || 'models folder'})`}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {canTuneDraft && <DraftTuning draft={draft} change={change} builtIn={builtInDraft} />}

      {estimate && (
        <div className="flex items-center gap-2 pt-2 text-[12px] text-muted">
          <Badge tone={VERDICT[estimate.verdict].tone}>{VERDICT[estimate.verdict].text}</Badge>
          about {formatBytes(estimate.gpuBytes)} on the GPU{estimate.totalBytes > estimate.gpuBytes && `, ${formatBytes(estimate.totalBytes - estimate.gpuBytes)} in RAM`}
          {dirty && ' with these changes'}
        </div>
      )}

      <div className="flex items-center gap-3 pt-2">
        <Button size="sm" variant="primary" disabled={saving || !dirty} onClick={() => void save()}>
          {saving ? 'Saving…' : loaded ? 'Save and restart model' : 'Save'}
        </Button>
        {dirty && !saving && (
          <Button size="sm" variant="ghost" onClick={() => setDraft(info.config)}>
            Cancel
          </Button>
        )}
        <span className="text-[11px] text-faint">
          {loaded ? `Loaded${info.loadedContext ? ` with ${formatTokens(info.loadedContext)} context` : ''}` : 'Not loaded; used the next time it loads'}
          {!info.custom && ' · using RulZ LLM Suite defaults'}
        </span>
      </div>
      {error && <p className="text-[13px] text-danger">{error}</p>}
    </div>
  )
}

/** Generation and load settings for one model, shown under the model picker in Settings. */
export function ModelSettings({ model }: { model: string }) {
  return (
    <div className="mt-3 space-y-4 rounded-lg border border-border bg-panel-2/50 p-4">
      <div>
        <div className="mb-1 text-[12px] font-medium tracking-wide text-muted uppercase">How it writes</div>
        <p className="mb-2 text-[11px] text-faint">Saved for this model only. Empty boxes use the value shown in grey.</p>
        <Sampling model={model} />
      </div>
      <div>
        <div className="mb-1 text-[12px] font-medium tracking-wide text-muted uppercase">How it loads</div>
        <p className="mb-2 text-[11px] text-faint">Saved in RulZ LLM Suite, so its Models page uses them too.</p>
        <Loading model={model} />
      </div>
    </div>
  )
}
