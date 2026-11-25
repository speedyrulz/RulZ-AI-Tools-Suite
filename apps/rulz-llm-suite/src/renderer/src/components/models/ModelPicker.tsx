import * as DialogPrimitive from '@radix-ui/react-dialog'
import { Cloud, Eye, RotateCcw, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { APP_MODES, leftoverSampling, MODE_LABELS, modeDefaultsEmpty } from '@shared/modes'
import type { LoadConfig, ModelEntry, SamplingParams } from '@shared/types'
import { toast } from 'sonner'
import { errorMessage, invoke } from '../../lib/api'
import { cn, formatBytes } from '../../lib/format'
import { modesFor, pickerScope, scopeFromValue, scopeLoad, scopeValue, startLoad, startScope, withModeLoad, withModeSaves, type PickerScope } from '../../lib/modes'
import { loadPlan, rentalLabel } from '../../lib/remote'
import { WBENCH_BUSY, wbenchRunning } from '../../lib/wbenchFlag'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { modeDefaultsOf, useModels } from '../../stores/models'
import { loadRemoteModel, useRemote } from '../../stores/remote'
import { Badge, Button, Field, Select, Switch, Tooltip } from '../ui'
import { EstimateView, useEstimate } from './EstimateView'
import { LoadSettingsForm } from './LoadSettingsForm'
import { OwnSampling } from './OwnSampling'

export function ModelPicker() {
  const open = useApp((s) => s.pickerOpen)
  const preselect = useApp((s) => s.pickerModelId)
  const close = useApp((s) => s.closePicker)
  return (
    <DialogPrimitive.Root open={open} onOpenChange={(o) => !o && close()}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/30 dark:bg-black/50 backdrop-blur-[2px]" />
        <DialogPrimitive.Content className="fixed top-[6vh] left-1/2 z-50 flex h-[84vh] w-[min(1060px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-xl border border-border bg-panel shadow-2xl focus:outline-none">
          <DialogPrimitive.Title className="sr-only">Load a model</DialogPrimitive.Title>
          <DialogPrimitive.Description className="sr-only">Choose a model and its load settings</DialogPrimitive.Description>
          {open && <PickerBody preselect={preselect} onDone={close} />}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

function matches(m: ModelEntry, q: string): boolean {
  if (!q) return true
  const hay = `${m.id} ${m.info?.architecture ?? ''} ${m.info?.quantization ?? ''}`.toLowerCase()
  return q
    .toLowerCase()
    .split(/\s+/)
    .every((w) => hay.includes(w))
}

function PickerBody({ preselect, onDone }: { preselect: string | null; onDone: () => void }) {
  const models = useModels((s) => s.models)
  const instances = useModels((s) => s.instances)
  const activeModelId = useModels((s) => s.activeModelId)
  const settings = useApp((s) => s.settings)
  const [query, setQuery] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(preselect ?? activeModelId ?? null)
  const filtered = useMemo(() => models.filter((m) => matches(m, query)), [models, query])
  // Only from the visible list: Enter must not load a model the search has hidden.
  const selected = filtered.find((m) => m.id === selectedId) ?? filtered[0] ?? null
  const searchRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    searchRef.current?.focus()
  }, [])

  const move = (delta: number): void => {
    if (!filtered.length) return
    const idx = filtered.findIndex((m) => m.id === selected?.id)
    const next = filtered[Math.min(filtered.length - 1, Math.max(0, idx + delta))]
    setSelectedId(next.id)
    document.getElementById(`model-row-${next.id}`)?.scrollIntoView({ block: 'nearest' })
  }

  return (
    <div className="flex min-w-0 flex-1">
      <div className="flex w-[46%] min-w-0 flex-col border-r border-border">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2.5">
          <Search size={16} className="text-faint" />
          <input
            ref={searchRef}
            className="h-8 flex-1 bg-transparent text-sm placeholder:text-faint focus:outline-none"
            placeholder={`Search ${models.length} local models…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                move(1)
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                move(-1)
              } else if (e.key === 'Enter' && selected) {
                e.preventDefault()
                document.getElementById('picker-load-button')?.click()
              }
            }}
          />
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
          {filtered.length === 0 && (
            <div className="p-6 text-center text-sm text-muted">
              {models.length === 0 ? (
                <>
                  No models found in <span className="font-mono text-xs">{settings?.modelsDir}</span>.
                  <br />
                  Change the models folder in Settings.
                </>
              ) : (
                'No models match your search.'
              )}
            </div>
          )}
          {filtered.map((m) => {
            const inst = instances.find((i) => i.modelId === m.id)
            const hosts = instances.filter((i) => i.modelId === m.id && i.host)
            return (
              <button
                id={`model-row-${m.id}`}
                key={m.id}
                onClick={() => setSelectedId(m.id)}
                onDoubleClick={() => document.getElementById('picker-load-button')?.click()}
                className={cn(
                  'flex w-full flex-col gap-1 rounded-lg px-3 py-2 text-left transition-colors',
                  selected?.id === m.id ? 'bg-accent-soft/70' : 'hover:bg-hover'
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-[13px] font-medium">{m.name}</span>
                  {m.remote && (
                    <Tooltip content="Runs on a rented vast.ai instance">
                      <span>
                        <Badge>
                          <Cloud size={11} /> vast.ai
                        </Badge>
                      </span>
                    </Tooltip>
                  )}
                  {inst && <Badge tone={inst.status === 'ready' ? 'ok' : 'accent'}>{inst.status === 'ready' ? 'Loaded' : 'Loading'}</Badge>}
                </div>
                {hosts.length > 0 && (
                  <div className="truncate text-[11px] text-muted">
                    {hosts.map((i) => `${i.host!.label}${i.reconnecting ? ' (reconnecting)' : ''}`).join(' · ')}
                  </div>
                )}
                <div className="flex items-center gap-1.5 text-[11px] text-muted">
                  {m.info?.sizeLabel && <span>{m.info.sizeLabel}</span>}
                  {m.info?.architecture && <span className="text-faint">· {m.info.architecture}</span>}
                  <span className="flex-1" />
                  {m.mmproj && (
                    <Tooltip content="Has a vision projector">
                      <Eye size={12} className="text-accent" />
                    </Tooltip>
                  )}
                  {m.info?.quantization && <Badge>{m.info.quantization}</Badge>}
                  <span className="w-14 text-right tabular-nums">{formatBytes(m.sizeBytes)}</span>
                </div>
              </button>
            )
          })}
        </div>
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-start justify-between gap-3 border-b border-border px-5 py-3">
          <div className="min-w-0">
            <div className="truncate text-[15px] font-semibold">{selected?.name ?? 'No model selected'}</div>
            <div className="truncate text-xs text-faint">{selected?.remote ? 'Runs on a rented vast.ai instance' : selected?.folder || ' '}</div>
          </div>
          <DialogPrimitive.Close className="rounded-md p-1 text-muted hover:bg-hover hover:text-text" aria-label="Close">
            <X size={16} />
          </DialogPrimitive.Close>
        </div>
        {selected ? <LoadPanel key={selected.id} model={selected} onDone={onDone} /> : <div className="flex-1" />}
      </div>
    </div>
  )
}

function LoadPanel({ model, onDone }: { model: ModelEntry; onDone: () => void }) {
  const instances = useModels((s) => s.instances)
  const load = useModels((s) => s.load)
  const unload = useModels((s) => s.unload)
  const settings = useApp((s) => s.settings)
  const rentals = useRemote((s) => s.state?.rentals)
  const allModes = useModels((s) => s.modeDefaults)
  const modes = modesFor(allModes, model.id)
  const [config, setConfig] = useState<LoadConfig | null>(null)
  const [remember, setRemember] = useState(false)
  // The mode's own sampling as edited here; null = not edited, so it shows (and keeps) what is saved.
  const [samplingEdit, setSamplingEdit] = useState<{ on: boolean; sampling: SamplingParams | null } | null>(null)
  // The model's default load settings (what a mode without its own falls back to).
  const [base, setBase] = useState<{ config: LoadConfig; custom: boolean } | null>(null)
  // Where the user is decides what the load settings are for; they can pick another.
  const [scope, setScope] = useState<PickerScope>(() => startScope(pickerScope(useApp.getState().page, !!useChat.getState().current?.re), modeDefaultsOf(model.id)))
  const scopeRef = useRef(scope)
  scopeRef.current = scope
  const hasOwnMode = (s: PickerScope): boolean => !!s && !!modeDefaultsOf(model.id)?.[s]?.load
  const hasCustom = scope ? !!modes?.[scope]?.load : !!base?.custom
  const inst = instances.find((i) => i.modelId === model.id && i.status !== 'stopping')
  // A remote model loads on the one running rental that has it set up, or is not loaded from here at all.
  const plan = model.remote ? (inst?.host ? ({ kind: 'load', rentalId: inst.host.rentalId } as const) : loadPlan(model.id, rentals ?? [])) : null
  const estimate = useEstimate(model.id, config ?? DEFAULT_LOAD_CONFIG, plan?.kind === 'load' ? plan.rentalId : null)

  useEffect(() => {
    invoke('models:getDefaults', model.id)
      .then(({ config, custom }) => {
        setBase({ config, custom })
        const s = scopeRef.current
        setConfig(startLoad(config, modeDefaultsOf(model.id), s, inst?.config))
        setRemember(s ? hasOwnMode(s) : custom)
      })
      .catch((e) => {
        // Without this the panel stays blank; fall back to the app defaults.
        toast.error(`Could not read this model's saved settings: ${errorMessage(e)}`)
        const fallback = useApp.getState().settings?.defaultLoadConfig ?? DEFAULT_LOAD_CONFIG
        setBase({ config: fallback, custom: false })
        setConfig(startLoad(fallback, modeDefaultsOf(model.id), scopeRef.current, inst?.config))
      })
  }, [model.id])

  // A new choice shows that scope's own settings, else the model's default ones, whatever was edited before.
  const pickScope = (next: PickerScope): void => {
    setScope(next)
    setSamplingEdit(null)
    if (!base) return
    setConfig(scopeLoad(base.config, modeDefaultsOf(model.id), next))
    setRemember(next ? hasOwnMode(next) : base.custom)
  }

  if (!config) return <div className="flex-1" />

  const doLoad = async (): Promise<void> => {
    if (wbenchRunning()) return void toast(WBENCH_BUSY)
    // Not remembering the load settings is no reason not to load. A mode's own sampling is saved when it was edited,
    // remembered or not; what was not edited stays as it is saved now.
    const samplingChange = scope && samplingEdit ? (samplingEdit.on && samplingEdit.sampling ? samplingEdit.sampling : null) : undefined
    if (scope ? remember || samplingChange !== undefined : remember) {
      const save = scope
        ? invoke(
            'models:setModeDefaults',
            model.id,
            scope,
            withModeSaves(modeDefaultsOf(model.id)?.[scope], base?.config.sampling, { ...(remember ? { load: config } : {}), ...(samplingChange !== undefined ? { sampling: samplingChange } : {}) })
          )
        : invoke('models:setDefaults', model.id, config)
      await save.catch((e) => toast.error(`Could not save the settings: ${errorMessage(e)}`))
    }
    onDone()
    if (plan?.kind === 'go') {
      // Never rents or starts anything: the vast.ai page does that, with a price on screen.
      useApp.getState().setPage('vast')
      toast(plan.sentence, { duration: 12000 })
      return
    }
    // Loading for a mode from that mode's own page stays there.
    if (!scope || scope === 'chat' || useApp.getState().page !== scope) useApp.getState().setPage('chat')
    if (plan?.kind === 'load') await loadRemoteModel(plan.rentalId, model.id, config)
    else await load(model.id, config)
  }

  return (
    <>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {model.error && <div className="rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">Could not read this file: {model.error}</div>}
        {plan?.kind === 'go' && (
          <div className="flex items-start gap-2 rounded-md border border-border bg-panel-2 px-3 py-2 text-xs text-muted">
            <Cloud size={14} className="mt-px shrink-0" />
            <div>{plan.sentence}</div>
          </div>
        )}
        {plan?.kind === 'load' && !inst && (
          <div className="text-xs text-muted">
            Loads on {rentalLabel(rentals?.find((r) => r.id === plan.rentalId) ?? { id: plan.rentalId, gpuName: 'instance', numGpus: 1 })}.
          </div>
        )}
        <Field label="Load for">
          <Select value={scopeValue(scope)} onChange={(e) => pickScope(scopeFromValue(e.target.value))} aria-label="Load for">
            <option value="all">All modes (default)</option>
            {APP_MODES.map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m]}
                {modeDefaultsEmpty(modes?.[m]) ? '' : ' · own settings'}
              </option>
            ))}
          </Select>
        </Field>
        <LoadSettingsForm model={model} config={config} onChange={setConfig} estimate={estimate} remote={model.remote} hideApiSampling={!!scope} />
        <EstimateView estimate={estimate} />
        {scope && (
          <OwnSampling
            key={scope}
            mode={scope}
            on={samplingEdit ? samplingEdit.on : !!modes?.[scope]?.sampling}
            sampling={samplingEdit ? samplingEdit.sampling : (modes?.[scope]?.sampling ?? null)}
            appSampling={settings?.defaultSampling}
            leftover={base ? leftoverSampling(modes?.[scope]?.load, base.config) : null}
            note="Saved for this model when you load it."
            onChange={setSamplingEdit}
          />
        )}
      </div>
      <div className="flex items-center gap-3 border-t border-border px-5 py-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          <Switch checked={remember} onCheckedChange={setRemember} />
          {scope ? `Remember ${MODE_LABELS[scope]} settings for this model` : 'Remember settings for this model'}
        </label>
        {hasCustom && (
          <Tooltip content={scope ? `Forget this model's own ${MODE_LABELS[scope]} load settings and use its default ones` : "Forget this model's saved settings and use the app defaults"}>
            <button
              className="flex items-center gap-1 text-xs text-muted hover:text-text"
              onClick={async () => {
                // Only the chosen scope's load settings go; a mode keeps its sampling and prompt.
                const target = scope
                try {
                  if (target) await invoke('models:setModeDefaults', model.id, target, withModeLoad(modeDefaultsOf(model.id)?.[target], null, base?.config.sampling))
                  else await invoke('models:setDefaults', model.id, null)
                } catch (e) {
                  toast.error(`Could not reset the settings: ${errorMessage(e)}`)
                  return
                }
                // The answer came after the user may have picked another scope: that one's form stays.
                if (scopeRef.current !== target) return
                setRemember(false)
                if (target) {
                  if (base) setConfig(base.config)
                } else if (settings) {
                  setBase({ config: settings.defaultLoadConfig, custom: false })
                  setConfig(settings.defaultLoadConfig)
                }
              }}
            >
              <RotateCcw size={12} /> Reset
            </button>
          </Tooltip>
        )}
        <div className="flex-1" />
        {inst && (
          <Button variant="ghost" onClick={() => void unload(inst.instanceId)}>
            Eject
          </Button>
        )}
        <Button id="picker-load-button" variant="primary" onClick={() => void doLoad()}>
          {plan?.kind === 'go' ? 'Go to vast.ai' : inst ? 'Reload with these settings' : 'Load model'}
        </Button>
      </div>
    </>
  )
}
