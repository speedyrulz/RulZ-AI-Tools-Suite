import * as DialogPrimitive from '@radix-ui/react-dialog'
import { Eye, RotateCcw, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { LoadConfig, ModelEntry } from '@shared/types'
import { invoke } from '../../lib/api'
import { cn, formatBytes } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { Badge, Button, Switch, Tooltip } from '../ui'
import { EstimateView, useEstimate } from './EstimateView'
import { LoadSettingsForm } from './LoadSettingsForm'

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
  const selected = models.find((m) => m.id === selectedId) ?? filtered[0] ?? null
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
                  {inst && <Badge tone={inst.status === 'ready' ? 'ok' : 'accent'}>{inst.status === 'ready' ? 'Loaded' : 'Loading'}</Badge>}
                </div>
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
            <div className="truncate text-xs text-faint">{selected?.folder || ' '}</div>
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
  const [config, setConfig] = useState<LoadConfig | null>(null)
  const [remember, setRemember] = useState(false)
  const [hasCustom, setHasCustom] = useState(false)
  const estimate = useEstimate(model.id, config ?? DEFAULT_LOAD_CONFIG)
  const inst = instances.find((i) => i.modelId === model.id && i.status !== 'stopping')

  useEffect(() => {
    void invoke('models:getDefaults', model.id).then(({ config, custom }) => {
      setConfig(inst?.config ?? config)
      setRemember(custom)
      setHasCustom(custom)
    })
  }, [model.id])

  if (!config) return <div className="flex-1" />

  const doLoad = async (): Promise<void> => {
    if (remember) await invoke('models:setDefaults', model.id, config)
    onDone()
    useApp.getState().setPage('chat')
    await load(model.id, config)
  }

  return (
    <>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {model.error && <div className="rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">Could not read this file: {model.error}</div>}
        <LoadSettingsForm model={model} config={config} onChange={setConfig} />
        <EstimateView estimate={estimate} />
      </div>
      <div className="flex items-center gap-3 border-t border-border px-5 py-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          <Switch checked={remember} onCheckedChange={setRemember} />
          Remember settings for this model
        </label>
        {hasCustom && (
          <Tooltip content="Forget this model's saved settings and use the app defaults">
            <button
              className="flex items-center gap-1 text-xs text-muted hover:text-text"
              onClick={async () => {
                await invoke('models:setDefaults', model.id, null)
                setHasCustom(false)
                setRemember(false)
                if (settings) setConfig(settings.defaultLoadConfig)
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
          {inst ? 'Reload with these settings' : 'Load model'}
        </Button>
      </div>
    </>
  )
}
