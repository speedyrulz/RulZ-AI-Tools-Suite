import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, FolderSearch } from 'lucide-react'
import { useState } from 'react'
import { draftFiles, fileName, resolveDraft } from '@shared/draft'
import { isQuantizedKv } from '@shared/kv'
import { parseTensorSplit, rebalanceSplit, toPercents } from '@shared/split'
import { KV_CACHE_TYPES, type DraftKind, type KvCacheType, type LoadConfig, type MemoryEstimate, type ModelEntry, type ModelSampling, type SidecarFile } from '@shared/types'
import { invoke } from '../../lib/api'
import { cn, formatBytes, formatTokens } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { Field, IconButton, Input, NumberInput, Select, Slider, Switch } from '../ui'

const CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
const DRAFT_LABELS: Record<DraftKind, string> = { mtp: 'MTP', eagle3: 'EAGLE-3', dflash: 'DFlash', dspark: 'DSpark' }

function draftLabel(d: SidecarFile): string {
  return d.builtIn ? 'built-in MTP layers' : `${DRAFT_LABELS[d.kind ?? 'mtp']} draft head`
}

function folderOf(path: string): string {
  return path.split(/[\\/]/).slice(-2, -1)[0] ?? ''
}

export function LoadSettingsForm({
  model,
  config,
  onChange,
  estimate
}: {
  model: ModelEntry | null
  config: LoadConfig
  onChange: (c: LoadConfig) => void
  /** Shown next to each GPU's split slider. */
  estimate?: MemoryEstimate | null
}) {
  const engine = useApp((s) => s.engine)
  const hardware = useModels((s) => s.hardware)
  const library = useModels((s) => s.models)
  const [advanced, setAdvanced] = useState(false)
  const [samplingOpen, setSamplingOpen] = useState(false)
  const setSampling = <K extends keyof ModelSampling>(key: K, value: ModelSampling[K]): void =>
    set('sampling', { ...config.sampling, [key]: value })
  const samplingSet = SAMPLING_FIELDS.filter((f) => config.sampling[f.key] !== null).length
  const set = <K extends keyof LoadConfig>(key: K, value: LoadConfig[K]): void => onChange({ ...config, [key]: value })
  const kvAll = (t: KvCacheType): boolean =>
    config.cacheTypeK === t && config.cacheTypeV === t && config.draftCacheTypeK === t && config.draftCacheTypeV === t

  const maxCtx = model?.info?.contextLength ?? 131072
  const layers = model?.info?.blockCount ?? 0
  const gpus = engine?.devices.length
    ? engine.devices.map((d, i) => ({ index: i, name: d.name }))
    : (hardware?.gpus.map((g) => ({ index: g.index, name: g.name })) ?? [])
  const gpuMode = config.splitMode === 'single' ? `single:${config.mainGpu}` : config.splitMode
  const otherDrafts = draftFiles(library)
  const draft = model ? resolveDraft(model, config, library) : undefined
  const canDraft = model === null || !!model.draft || !!model.info?.mtpLayers || otherDrafts.length > 0

  return (
    <div className="space-y-4">
      <Field
        label="Context length"
        hint={model?.info?.contextLength ? `Model supports ${formatTokens(model.info.contextLength)} tokens` : undefined}
      >
        <div className="flex items-center gap-3">
          <Slider
            value={config.contextLength ?? maxCtx}
            min={512}
            max={maxCtx}
            step={512}
            disabled={config.contextLength === null}
            onChange={(v) => set('contextLength', v)}
          />
          <NumberInput
            value={config.contextLength}
            placeholder="Auto"
            min={512}
            max={maxCtx}
            step={512}
            onChange={(v) => set('contextLength', v === null ? null : Math.min(maxCtx, Math.max(512, Math.round(v))))}
          />
        </div>
        <div className="flex flex-wrap gap-1">
          <PresetChip active={config.contextLength === null} onClick={() => set('contextLength', null)}>
            Auto (fit)
          </PresetChip>
          {CTX_PRESETS.filter((p) => p <= maxCtx).map((p) => (
            <PresetChip key={p} active={config.contextLength === p} onClick={() => set('contextLength', p)}>
              {formatTokens(p)}
            </PresetChip>
          ))}
        </div>
      </Field>

      <Field label="GPU offload" hint={layers ? `${layers} layers` : undefined}>
        <div className="flex items-center gap-2">
          <Select
            className="w-40"
            value={typeof config.gpuLayers === 'number' ? 'custom' : config.gpuLayers}
            onChange={(e) => {
              const v = e.target.value
              set('gpuLayers', v === 'custom' ? Math.max(0, layers) : (v as 'auto' | 'all'))
            }}
          >
            <option value="auto">Auto (fit to VRAM)</option>
            <option value="all">All layers on GPU</option>
            <option value="custom">Custom layer count</option>
          </Select>
          {typeof config.gpuLayers === 'number' && (
            <>
              <Slider value={config.gpuLayers} min={0} max={Math.max(1, layers + 1)} onChange={(v) => set('gpuLayers', v)} />
              <NumberInput className="w-16" value={config.gpuLayers} min={0} onChange={(v) => set('gpuLayers', Math.max(0, v ?? 0))} />
            </>
          )}
        </div>
      </Field>

      <Field label="GPUs">
        <Select
          value={gpuMode}
          onChange={(e) => {
            const v = e.target.value
            if (v.startsWith('single:')) onChange({ ...config, splitMode: 'single', mainGpu: Number(v.slice(7)) })
            else if (v === 'custom') {
              // Start the sliders from an even split unless a valid one is already saved.
              const even = toPercents(gpus.map(() => 1 / gpus.length)).join(',')
              onChange({ ...config, splitMode: 'custom', tensorSplit: parseTensorSplit(config.tensorSplit, gpus.length) ? config.tensorSplit : even })
            } else onChange({ ...config, splitMode: 'auto' })
          }}
        >
          <option value="auto">{gpus.length > 1 ? 'Split across all GPUs (by free memory)' : 'Automatic'}</option>
          {gpus.map((g) => (
            <option key={g.index} value={`single:${g.index}`}>
              Only GPU{g.index}: {g.name}
            </option>
          ))}
          {gpus.length > 1 && <option value="custom">Custom split…</option>}
        </Select>
        {config.splitMode === 'custom' && gpus.length > 1 && (
          <GpuSplit gpus={gpus} split={config.tensorSplit} estimate={estimate} onChange={(s) => set('tensorSplit', s)} />
        )}
      </Field>

      <div className="grid grid-cols-3 gap-3">
        <Field label="Flash attention">
          <Select value={config.flashAttention} onChange={(e) => set('flashAttention', e.target.value as LoadConfig['flashAttention'])}>
            <option value="auto">Auto</option>
            <option value="on">On</option>
            <option value="off">Off</option>
          </Select>
        </Field>
        <Field label="K cache">
          <KvSelect value={config.cacheTypeK} onChange={(v) => set('cacheTypeK', v)} />
        </Field>
        <Field
          label="V cache"
          hint={config.flashAttention === 'off' && isQuantizedKv(config.cacheTypeV) ? 'F16 is used: a quantized V cache needs flash attention' : undefined}
        >
          <KvSelect value={config.cacheTypeV} onChange={(v) => set('cacheTypeV', v)} />
        </Field>
      </div>
      <div className="-mt-2 flex flex-wrap items-center gap-1 text-[11px] text-faint">
        KV cache:
        <PresetChip
          active={kvAll('q8_0')}
          onClick={() => onChange({ ...config, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', draftCacheTypeK: 'q8_0', draftCacheTypeV: 'q8_0' })}
        >
          Save VRAM (Q8_0, about half the size)
        </PresetChip>
        <PresetChip
          active={kvAll('f16')}
          onClick={() => onChange({ ...config, cacheTypeK: 'f16', cacheTypeV: 'f16', draftCacheTypeK: 'f16', draftCacheTypeV: 'f16' })}
        >
          Full precision (F16)
        </PresetChip>
      </div>

      {model?.mmproj && (
        <Field inline label="Vision projector" hint={model.mmproj.path.split(/[\\/]/).pop()}>
          <Switch checked={config.useMmproj} onCheckedChange={(v) => set('useMmproj', v)} />
        </Field>
      )}
      {(model === null || model.mmproj) && config.useMmproj && (
        <Field
          inline
          label="Keep vision projector on CPU"
          hint={
            'Frees its VRAM (often 0.5–1.5 GB); images are encoded more slowly. ' +
            'For text-only chats, turning the projector off also lets llama-server reuse its prompt cache.'
          }
        >
          <Switch checked={config.mmprojOnCpu} onCheckedChange={(v) => set('mmprojOnCpu', v)} />
        </Field>
      )}

      {/* The app-wide defaults (no model) also carry these, for models that have a draft head. */}
      {canDraft && (
        <div className="space-y-2">
          <Field
            inline
            label={draft ? `Speculative decoding (${draftLabel(draft)})` : 'Speculative decoding (MTP / draft heads)'}
            hint={
              model === null
                ? 'Used for models that have MTP layers or come with a draft head file'
                : draft
                  ? `${draft.builtIn ? 'Inside the model file' : fileName(draft.path)} · drafts several tokens per step; output is unchanged`
                  : 'No draft found with this model; pick one below'
            }
          >
            <Switch checked={config.useDraft} onCheckedChange={(v) => set('useDraft', v)} />
          </Field>
          {config.useDraft && (
            <div className="space-y-1 rounded-lg border border-border p-3">
              {model && (
                <Field inline label="Draft model" hint="Must come from the same base model as this one; a mismatched draft fails to load or only slows things down">
                  <Select className="w-56" value={config.draftFile ?? ''} onChange={(e) => set('draftFile', e.target.value || null)}>
                    <option value="">Automatic ({model.draft ? draftLabel(model.draft) : 'none found'})</option>
                    {model.info?.mtpLayers && <option value="builtin">Built-in MTP layers</option>}
                    {otherDrafts.map((d) => (
                      <option key={d.path} value={d.path}>
                        {fileName(d.path)} ({folderOf(d.path)})
                      </option>
                    ))}
                    {config.draftFile && config.draftFile !== 'builtin' && !otherDrafts.some((d) => d.path === config.draftFile) && (
                      <option value={config.draftFile}>{fileName(config.draftFile)} (not found)</option>
                    )}
                  </Select>
                </Field>
              )}
              <Field inline label="Draft tokens per step" hint="Most tokens drafted at once; 2 is a good default, empty = llama.cpp's 3">
                <NumberInput value={config.draftTokens} placeholder="3" min={1} max={64} onChange={(v) => set('draftTokens', v === null ? null : Math.max(1, Math.min(64, Math.round(v))))} />
              </Field>
              <Field inline label="Minimum draft tokens" hint="Skip drafting when fewer would be proposed; empty = 0">
                <NumberInput value={config.draftMinTokens} placeholder="0" min={0} max={64} onChange={(v) => set('draftMinTokens', v === null ? null : Math.max(0, Math.min(64, Math.round(v))))} />
              </Field>
              <Field inline label="Minimum draft confidence" hint="Stop a draft early once its probability drops below this (0–1); empty = 0">
                <NumberInput value={config.draftPMin} placeholder="0" min={0} max={1} step={0.05} onChange={(v) => set('draftPMin', v === null ? null : Math.max(0, Math.min(1, v)))} />
              </Field>
              <Field inline label="Draft sampling" hint="Greedy takes the draft's top pick; probabilistic samples and verifies by rejection">
                <Select className="w-36" value={config.draftSampling} onChange={(e) => set('draftSampling', e.target.value as LoadConfig['draftSampling'])}>
                  <option value="auto">Auto</option>
                  <option value="greedy">Greedy</option>
                  <option value="probabilistic">Probabilistic</option>
                </Select>
              </Field>
              {!draft?.builtIn && (
              <Field inline label="Draft head placement" hint="CPU frees VRAM but drafts more slowly">
                <Select className="w-36" value={config.draftGpu} onChange={(e) => set('draftGpu', e.target.value as LoadConfig['draftGpu'])}>
                  <option value="auto">Auto</option>
                  <option value="gpu">GPU</option>
                  <option value="cpu">CPU</option>
                </Select>
              </Field>
              )}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <Field label="Draft K cache">
                  <KvSelect value={config.draftCacheTypeK} onChange={(v) => set('draftCacheTypeK', v)} />
                </Field>
                <Field label="Draft V cache">
                  <KvSelect value={config.draftCacheTypeV} onChange={(v) => set('draftCacheTypeV', v)} />
                </Field>
              </div>
            </div>
          )}
        </div>
      )}

      <Collapsible.Root open={samplingOpen} onOpenChange={setSamplingOpen}>
        <Collapsible.Trigger className="flex items-center gap-1 text-xs font-medium text-muted hover:text-text">
          <ChevronRight size={14} className={samplingOpen ? 'rotate-90 transition-transform' : 'transition-transform'} />
          Sampling defaults{samplingSet > 0 && <span className="text-faint"> · {samplingSet} set{config.sampling.override ? ', override' : ''}</span>}
        </Collapsible.Trigger>
        <Collapsible.Content className="mt-3 space-y-1">
          <p className="pb-1 text-[11px] text-faint">
            Used by programs that reach this model through the API server, when a request leaves a value out. Empty = llama.cpp's
            default. The chat in this app uses its own sampling settings.
          </p>
          {SAMPLING_FIELDS.map((f) => (
            <Field key={f.key} inline label={f.label} hint={f.hint}>
              <NumberInput
                value={config.sampling[f.key]}
                placeholder={f.placeholder}
                min={f.min}
                max={f.max}
                step={f.step}
                onChange={(v) => setSampling(f.key, v === null ? null : Math.max(f.min, Math.min(f.max, f.integer ? Math.round(v) : v)))}
              />
            </Field>
          ))}
          <Field
            inline
            label="Override what programs send"
            hint="Use the values above even when a request sets its own, e.g. Claude Code's temperature. Empty values still keep the request's"
          >
            <Switch checked={config.sampling.override} onCheckedChange={(v) => setSampling('override', v)} />
          </Field>
        </Collapsible.Content>
      </Collapsible.Root>

      <Collapsible.Root open={advanced} onOpenChange={setAdvanced}>
        <Collapsible.Trigger className="flex items-center gap-1 text-xs font-medium text-muted hover:text-text">
          <ChevronRight size={14} className={advanced ? 'rotate-90 transition-transform' : 'transition-transform'} />
          Advanced
        </Collapsible.Trigger>
        <Collapsible.Content className="mt-3 space-y-1">
          <Field inline label="CPU threads" hint="Empty = automatic">
            <NumberInput value={config.threads} placeholder="Auto" min={1} onChange={(v) => set('threads', v)} />
          </Field>
          <Field inline label="Batch size" hint="Logical batch for prompt processing">
            <NumberInput value={config.batchSize} placeholder="2048" min={1} onChange={(v) => set('batchSize', v)} />
          </Field>
          <Field inline label="Micro-batch size">
            <NumberInput value={config.ubatchSize} placeholder="512" min={1} onChange={(v) => set('ubatchSize', v)} />
          </Field>
          <Field inline label="Parallel slots" hint="Concurrent requests sharing the context">
            <NumberInput value={config.parallel} min={1} max={64} onChange={(v) => set('parallel', Math.max(1, v ?? 1))} />
          </Field>
          <Field inline label="Seed" hint="Empty = random">
            <NumberInput value={config.seed} placeholder="Random" onChange={(v) => set('seed', v === null ? null : Math.round(v))} />
          </Field>
          <Field inline label="Memory-map the model file (mmap)">
            <Switch checked={config.useMmap} onCheckedChange={(v) => set('useMmap', v)} />
          </Field>
          <Field inline label="Keep model in RAM (mlock)">
            <Switch checked={config.useMlock} onCheckedChange={(v) => set('useMlock', v)} />
          </Field>
          <div className="space-y-3 pt-2">
            <Field label="Chat template override" hint="Jinja file; empty = model's own template">
              <div className="flex gap-1.5">
                <Input value={config.chatTemplateFile} placeholder="None" onChange={(e) => set('chatTemplateFile', e.target.value)} />
                <IconButton
                  label="Browse"
                  onClick={async () => {
                    const p = await invoke('dialog:pickFile', {
                      title: 'Choose a Jinja chat template',
                      filters: [{ name: 'Jinja templates', extensions: ['jinja', 'j2', 'txt'] }]
                    })
                    if (p) set('chatTemplateFile', p)
                  }}
                >
                  <FolderSearch size={15} />
                </IconButton>
              </div>
            </Field>
            <Field label="Extra llama-server arguments">
              <Input
                className="font-mono text-xs"
                placeholder="--override-tensor exps=CPU"
                value={config.extraArgs}
                onChange={(e) => set('extraArgs', e.target.value)}
              />
            </Field>
          </div>
        </Collapsible.Content>
      </Collapsible.Root>
    </div>
  )
}

type SamplingKey = Exclude<keyof ModelSampling, 'override'>
const SAMPLING_FIELDS: { key: SamplingKey; label: string; hint: string; placeholder: string; min: number; max: number; step: number; integer?: boolean }[] = [
  { key: 'temperature', label: 'Temperature', hint: 'Higher = more varied, lower = more predictable', placeholder: '0.8', min: 0, max: 5, step: 0.05 },
  { key: 'topK', label: 'Top K', hint: 'Pick only from this many likeliest tokens; 0 = off', placeholder: '40', min: 0, max: 1000, step: 1, integer: true },
  { key: 'topP', label: 'Top P', hint: 'Pick from the likeliest tokens that add up to this share', placeholder: '0.95', min: 0, max: 1, step: 0.01 },
  { key: 'minP', label: 'Min P', hint: 'Drop tokens far less likely than the top one', placeholder: '0.05', min: 0, max: 1, step: 0.01 },
  { key: 'repeatPenalty', label: 'Repeat penalty', hint: '1 = off; higher discourages repeating recent text', placeholder: '1.0', min: 0, max: 5, step: 0.05 },
  { key: 'presencePenalty', label: 'Presence penalty', hint: 'Above 0 nudges toward new topics', placeholder: '0', min: -2, max: 2, step: 0.1 }
]

/** One slider per GPU; moving one rescales the others so the split always adds up to 100%. */
function GpuSplit({
  gpus,
  split,
  estimate,
  onChange
}: {
  gpus: { index: number; name: string }[]
  split: string
  estimate?: MemoryEstimate | null
  onChange: (split: string) => void
}) {
  const percents = toPercents(parseTensorSplit(split, gpus.length) ?? gpus.map(() => 1 / gpus.length))
  return (
    <div className="space-y-2.5 rounded-lg border border-border p-3">
      {gpus.map((g, i) => {
        const est = estimate?.perGpu.find((e) => e.index === g.index)
        const over = est !== undefined && est.bytes > est.freeBytes * 0.95
        return (
          <div key={g.index} className="space-y-1">
            <div className="flex items-baseline justify-between gap-2 text-xs">
              <span className="min-w-0 truncate">
                GPU{g.index}: {g.name}
              </span>
              <span className={cn('shrink-0 tabular-nums', over ? 'text-danger' : 'text-muted')}>
                {est ? `≈ ${formatBytes(est.bytes)} of ${formatBytes(est.freeBytes)} free` : percents[i] === 0 ? 'Not used' : ''}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <Slider value={percents[i]} min={0} max={100} onChange={(v) => onChange(rebalanceSplit(percents, i, v).join(','))} />
              <span className="w-10 shrink-0 text-right text-xs tabular-nums">{percents[i]}%</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function KvSelect({ value, onChange }: { value: KvCacheType; onChange: (v: KvCacheType) => void }) {
  return (
    <Select value={value} onChange={(e) => onChange(e.target.value as KvCacheType)}>
      {KV_CACHE_TYPES.map((t) => (
        <option key={t} value={t}>
          {t.toUpperCase()}
        </option>
      ))}
    </Select>
  )
}

function PresetChip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={
        active
          ? 'rounded border border-accent/50 bg-accent-soft px-1.5 py-px text-[11px] text-accent'
          : 'rounded border border-border px-1.5 py-px text-[11px] text-muted hover:bg-hover hover:text-text'
      }
    >
      {children}
    </button>
  )
}
