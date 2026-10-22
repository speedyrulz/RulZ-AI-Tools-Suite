import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, FolderSearch } from 'lucide-react'
import { useState } from 'react'
import { KV_CACHE_TYPES, type KvCacheType, type LoadConfig, type ModelEntry } from '@shared/types'
import { invoke } from '../../lib/api'
import { formatTokens } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { Field, IconButton, Input, NumberInput, Select, Slider, Switch } from '../ui'

const CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

export function LoadSettingsForm({
  model,
  config,
  onChange
}: {
  model: ModelEntry | null
  config: LoadConfig
  onChange: (c: LoadConfig) => void
}) {
  const engine = useApp((s) => s.engine)
  const hardware = useModels((s) => s.hardware)
  const [advanced, setAdvanced] = useState(false)
  const set = <K extends keyof LoadConfig>(key: K, value: LoadConfig[K]): void => onChange({ ...config, [key]: value })

  const maxCtx = model?.info?.contextLength ?? 131072
  const layers = model?.info?.blockCount ?? 0
  const gpus = engine?.devices.length
    ? engine.devices.map((d, i) => ({ index: i, name: d.name }))
    : (hardware?.gpus.map((g) => ({ index: g.index, name: g.name })) ?? [])
  const gpuMode = config.splitMode === 'single' ? `single:${config.mainGpu}` : config.splitMode

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
            else onChange({ ...config, splitMode: v as 'auto' | 'custom' })
          }}
        >
          <option value="auto">{gpus.length > 1 ? 'Split across all GPUs' : 'Automatic'}</option>
          {gpus.map((g) => (
            <option key={g.index} value={`single:${g.index}`}>
              Only GPU{g.index}: {g.name}
            </option>
          ))}
          {gpus.length > 1 && <option value="custom">Custom split…</option>}
        </Select>
        {config.splitMode === 'custom' && (
          <Input
            placeholder={`Proportions per GPU, e.g. ${gpus.map(() => '1').join(',')}`}
            value={config.tensorSplit}
            onChange={(e) => set('tensorSplit', e.target.value)}
          />
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
        <Field label="V cache">
          <KvSelect value={config.cacheTypeV} onChange={(v) => set('cacheTypeV', v)} />
        </Field>
      </div>

      {model?.mmproj && (
        <Field inline label="Vision projector" hint={model.mmproj.path.split(/[\\/]/).pop()}>
          <Switch checked={config.useMmproj} onCheckedChange={(v) => set('useMmproj', v)} />
        </Field>
      )}

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
