import { useEffect, useState } from 'react'
import { DEFAULT_MAX_OUTPUT_TOKENS, KV_CACHE_TYPES, recommendedSampling } from '@shared/defaults'
import type { KvCacheType, LoadSettings, ModelLoadInfo, SamplingSettings } from '@shared/types'
import { invoke } from '../lib/api'
import { formatTokens } from '../lib/format'
import { useApp } from '../stores/app'
import { Button, Field, NumberInput, Select, Switch } from './ui'

const SAMPLING_FIELDS: { key: keyof SamplingSettings; label: string; hint: string; min: number; max: number; step: number }[] = [
  { key: 'temperature', label: 'Creativity (temperature)', hint: 'Lower = more predictable code', min: 0, max: 2, step: 0.05 },
  { key: 'topP', label: 'Top P', hint: 'Only consider the likeliest words that add up to this share', min: 0, max: 1, step: 0.05 },
  { key: 'topK', label: 'Top K', hint: 'Only consider this many likeliest words (0 = no limit)', min: 0, max: 1000, step: 1 },
  { key: 'repeatPenalty', label: 'Repeat penalty', hint: 'Above 1 discourages repeating itself', min: 0.5, max: 2, step: 0.01 },
  { key: 'maxOutputTokens', label: 'Longest answer (tokens)', hint: 'One reply or tool call at most', min: 256, max: 131072, step: 256 }
]

/** How the model writes: the owner's values over the recommended ones, saved per model. */
function Sampling({ model }: { model: string }) {
  const all = useApp((s) => s.settings?.modelSampling ?? {})
  const update = useApp((s) => s.updateSettings)
  const own = all[model] ?? {}
  const rec: SamplingSettings = { maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS, ...recommendedSampling(model) }

  const set = (key: keyof SamplingSettings, v: number | null): void => {
    const next: SamplingSettings = { ...own }
    if (v === null) delete next[key]
    else next[key] = v
    const rest = { ...all }
    if (Object.keys(next).length) rest[model] = next
    else delete rest[model]
    void update({ modelSampling: rest })
  }

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
            onChange={(v) => (v === null || (v >= f.min && v <= f.max)) && set(f.key, v)}
          />
        </Field>
      ))}
      {Object.keys(own).length > 0 && (
        <Button size="sm" variant="ghost" onClick={() => void update({ modelSampling: Object.fromEntries(Object.entries(all).filter(([k]) => k !== model)) })}>
          Back to recommended
        </Button>
      )}
    </div>
  )
}

const same = (a: LoadSettings, b: LoadSettings): boolean => JSON.stringify(a) === JSON.stringify(b)

/** How RulZ LLM Suite loads the model. Saving restarts a loaded model, so changes wait for the button. */
function Loading({ model }: { model: string }) {
  const [info, setInfo] = useState<ModelLoadInfo | null>(null)
  const [draft, setDraft] = useState<LoadSettings | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

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
  const loaded = info.state === 'ready' || info.state === 'loading'
  const gpuMode = typeof draft.gpuLayers === 'number' ? 'number' : draft.gpuLayers

  const save = async (): Promise<void> => {
    setSaving(true)
    setError('')
    try {
      const next = await invoke('models:setLoadSettings', model, draft)
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
      <Field label="Speed up with a draft model" hint="Only works if the model comes with one" inline>
        <Switch checked={draft.useDraft} onCheckedChange={(v) => change({ useDraft: v })} label="Speed up with a draft model" />
      </Field>
      <div className="flex items-center gap-3 pt-2">
        <Button size="sm" variant="primary" disabled={saving || same(draft, info.config)} onClick={() => void save()}>
          {saving ? 'Saving…' : loaded ? 'Save and restart model' : 'Save'}
        </Button>
        {!same(draft, info.config) && !saving && (
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
