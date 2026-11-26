import { Cloud, Ellipsis, Eye, FolderOpen, FolderSearch, Gauge, Globe, Play, RefreshCw, Repeat, Search, Trash2 } from 'lucide-react'
import { Fragment, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { adapterChoices } from '@shared/adapters'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { AdapterEntry, LoadConfig, ModelEntry } from '@shared/types'
import { AdapterConvertDialog } from '../components/models/AdapterConvertDialog'
import { BenchmarkView } from '../components/models/BenchmarkView'
import { EstimateView, useEstimate } from '../components/models/EstimateView'
import { LoadSettingsForm } from '../components/models/LoadSettingsForm'
import { ModeDefaultsEditor, type ModeScope } from '../components/models/ModeDefaultsEditor'
import { Badge, Button, Dialog, DropdownMenu, IconButton, Input } from '../components/ui'
import { errorMessage, invoke } from '../lib/api'
import { badgesFor, priceLine, priceShort } from '../lib/cloud'
import { cn, formatBytes, formatTokens } from '../lib/format'
import { useApp } from '../stores/app'
import { useModels } from '../stores/models'

type SortKey = 'name' | 'size' | 'modified'

export function ModelsPage() {
  const models = useModels((s) => s.models)
  const scanned = useModels((s) => s.scanned)
  const instances = useModels((s) => s.instances)
  const adapters = useModels((s) => s.adapters)
  const rescan = useModels((s) => s.rescan)
  const settings = useApp((s) => s.settings)
  const updateSettings = useApp((s) => s.updateSettings)
  const openPicker = useApp((s) => s.openPicker)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortKey>('name')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [scanning, setScanning] = useState(false)
  const [trashTarget, setTrashTarget] = useState<ModelEntry | null>(null)
  const [adapterTrash, setAdapterTrash] = useState<AdapterEntry | null>(null)
  const [convertOpen, setConvertOpen] = useState(false)
  const [benchId, setBenchId] = useState<string | null>(null)

  const list = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = q ? models.filter((m) => `${m.id} ${m.name} ${m.info?.architecture ?? ''}`.toLowerCase().includes(q)) : models
    return [...filtered].sort((a, b) =>
      sort === 'size' ? b.sizeBytes - a.sizeBytes : sort === 'modified' ? b.modifiedAt - a.modifiedAt : a.id.localeCompare(b.id)
    )
  }, [models, query, sort])
  const selected = models.find((m) => m.id === selectedId) ?? null
  // Remote models are definitions, not files on this PC: they are listed but not counted in the folder's size.
  const localModels = models.filter((m) => !m.remote && !m.cloud)
  const cloudCount = models.filter((m) => m.cloud).length
  // OpenRouter models are listed last, under their own heading.
  const ordered = useMemo(() => [...list.filter((m) => !m.cloud), ...list.filter((m) => m.cloud)], [list])
  const totalSize = localModels.reduce((sum, m) => sum + m.sizeBytes + (m.mmproj?.sizeBytes ?? 0), 0)

  const doRescan = async (): Promise<void> => {
    setScanning(true)
    await rescan()
    setScanning(false)
  }

  const benchModel = benchId ? models.find((m) => m.id === benchId) : undefined
  if (benchModel) return <BenchmarkView key={benchModel.id} model={benchModel} onBack={() => setBenchId(null)} />

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border px-6 py-4">
          <div className="min-w-0 flex-1">
            <h1 className="font-[family-name:var(--font-display)] text-base font-semibold">My Models</h1>
            <div className="truncate text-xs text-muted">
              {localModels.length} models · {formatBytes(totalSize)}{models.length - cloudCount > localModels.length ? ` · ${models.length - cloudCount - localModels.length} on vast.ai` : ''}{cloudCount > 0 ? ` · ${cloudCount} at OpenRouter` : ''} · <span className="font-mono">{settings?.modelsDir}</span>
            </div>
          </div>
          <Button size="sm" onClick={() => settings && void invoke('shell:openPath', settings.modelsDir)}>
            <FolderOpen size={14} /> Open folder
          </Button>
          <Button
            size="sm"
            onClick={async () => {
              const dir = await invoke('dialog:pickFolder', { title: 'Choose your models folder', defaultPath: settings?.modelsDir })
              if (dir) await updateSettings({ modelsDir: dir })
            }}
          >
            <FolderSearch size={14} /> Change folder
          </Button>
          <Button size="sm" onClick={() => setConvertOpen(true)}>
            <Repeat size={14} /> Convert PEFT adapter
          </Button>
          <Button size="sm" onClick={() => void doRescan()} disabled={scanning}>
            <RefreshCw size={14} className={scanning ? 'animate-spin' : undefined} /> Rescan
          </Button>
        </div>

        <div className="flex items-center gap-3 px-6 py-3">
          <div className="relative w-72">
            <Search size={13} className="absolute top-1/2 left-2.5 -translate-y-1/2 text-faint" />
            <Input className="h-8 pl-7.5" placeholder="Filter models" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div className="flex items-center gap-1 text-xs text-muted">
            Sort:
            {(['name', 'size', 'modified'] as SortKey[]).map((k) => (
              <button
                key={k}
                onClick={() => setSort(k)}
                className={cn('rounded px-2 py-1 capitalize', sort === k ? 'bg-hover text-text' : 'hover:text-text')}
              >
                {k}
              </button>
            ))}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
          {!scanned ? (
            <div className="py-16 text-center text-sm text-muted">Scanning models…</div>
          ) : list.length === 0 ? (
            <div className="py-16 text-center text-sm text-muted">
              {models.length ? 'No models match your filter.' : 'No .gguf models found in this folder.'}
            </div>
          ) : (
            <table className="w-full table-fixed border-separate border-spacing-0 text-[13px]">
              <thead className="sticky top-0 z-10 bg-bg text-left text-[11px] text-faint uppercase">
                <tr>
                  <th className="border-b border-border py-2 pl-3 font-medium">Model</th>
                  <th className="w-28 border-b border-border py-2 font-medium">Architecture</th>
                  <th className="w-20 border-b border-border py-2 font-medium">Params</th>
                  <th className="w-24 border-b border-border py-2 font-medium">Quant</th>
                  <th className="w-20 border-b border-border py-2 font-medium">Context</th>
                  <th className="w-20 border-b border-border py-2 text-right font-medium">Size</th>
                  <th className="w-28 border-b border-border py-2" />
                </tr>
              </thead>
              <tbody>
                {ordered.map((m, idx) => {
                  const inst = instances.find((i) => i.modelId === m.id)
                  const fits = m.remote || m.cloud ? [] : adapterChoices(m, adapters)
                  const firstCloud = !!m.cloud && ordered.findIndex((x) => x.cloud) === idx
                  return (
                    <Fragment key={m.id}>
                    {firstCloud && (
                      <tr>
                        <td colSpan={7} className="border-b border-border pt-5 pb-2 pl-3 text-[11px] font-semibold tracking-wide text-faint uppercase">
                          OpenRouter <span className="font-normal normal-case tabular-nums">· {cloudCount}</span>
                        </td>
                      </tr>
                    )}
                    <tr
                      onClick={() => setSelectedId(m.id)}
                      className={cn('group cursor-default', selectedId === m.id ? 'bg-accent-soft/50' : 'hover:bg-hover/50')}
                    >
                      <td className="border-b border-border py-2 pr-3 pl-3">
                        <div className="flex items-center gap-1.5">
                          <span className="truncate font-medium">{m.name}</span>
                          {(m.mmproj || m.cloud?.info.vision) && <Eye size={12} className="shrink-0 text-accent" aria-label="Vision" />}
                          {m.cloud && (
                            <Badge>
                              <Globe size={11} /> OpenRouter
                            </Badge>
                          )}
                          {m.remote && (
                            <Badge>
                              <Cloud size={11} /> vast.ai
                            </Badge>
                          )}
                          {inst && <Badge tone={inst.status === 'ready' ? 'ok' : 'accent'}>{inst.status === 'ready' ? (m.cloud ? 'Connected' : 'Loaded') : m.cloud ? 'Connecting' : 'Loading'}</Badge>}
                          {m.error && <Badge tone="danger">Unreadable</Badge>}
                        </div>
                        <div className="truncate text-[11px] text-faint">{m.remote ? 'Runs on a rented instance. Edit it on the vast.ai page.' : m.cloud ? `Runs at OpenRouter. Edit it on the OpenRouter page. ${priceLine(m.cloud.info)}` : m.folder || '/'}</div>
                        {fits.length > 0 && (
                          <div className="truncate text-[11px] text-faint" title={fits.map((a) => a.name).join(', ')}>
                            Adapters: {fits.map((a) => a.name).join(', ')}
                          </div>
                        )}
                      </td>
                      <td className="truncate border-b border-border py-2 text-muted">{m.info?.architecture ?? '—'}</td>
                      <td className="border-b border-border py-2 text-muted">{m.info?.sizeLabel ?? '—'}</td>
                      <td className="border-b border-border py-2">{m.info?.quantization ? <Badge>{m.info.quantization}</Badge> : '—'}</td>
                      <td className="border-b border-border py-2 text-muted tabular-nums">
                        {m.cloud ? formatTokens(m.cloud.contextLimit) : m.info?.contextLength ? formatTokens(m.info.contextLength) : '—'}
                      </td>
                      <td className="border-b border-border py-2 text-right text-muted tabular-nums">{m.cloud ? priceShort(m.cloud.info) : formatBytes(m.sizeBytes)}</td>
                      <td className="border-b border-border py-2 pr-2">
                        <div className="flex justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                          <Button size="sm" variant="ghost" onClick={() => openPicker(m.id)}>
                            <Play size={12} /> {m.cloud ? 'Use' : 'Load'}
                          </Button>
                          {!m.remote && !m.cloud && (
                          <DropdownMenu
                            trigger={
                              <button className="rounded p-1.5 text-faint hover:bg-hover hover:text-text" aria-label="Model options">
                                <Ellipsis size={14} />
                              </button>
                            }
                            items={[
                              { label: 'Benchmark load settings…', icon: <Gauge size={13} />, onSelect: () => setBenchId(m.id), disabled: !!m.error },
                              { label: 'Show in Explorer', icon: <FolderOpen size={13} />, onSelect: () => void invoke('shell:showItem', m.path) },
                              'sep',
                              { label: 'Move to Recycle Bin…', icon: <Trash2 size={13} />, onSelect: () => setTrashTarget(m), danger: true }
                            ]}
                          />
                          )}
                        </div>
                      </td>
                    </tr>
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          )}
          {scanned && adapters.length > 0 && <AdaptersSection adapters={adapters} models={models} onTrash={setAdapterTrash} />}
        </div>
      </div>

      {selected && <ModelDetails key={selected.id} model={selected} onClose={() => setSelectedId(null)} onBenchmark={() => setBenchId(selected.id)} />}

      <Dialog
        open={!!trashTarget}
        onOpenChange={(o) => !o && setTrashTarget(null)}
        title="Move model to the Recycle Bin?"
        description={trashTarget ? `${trashTarget.files.length} file(s), ${formatBytes(trashTarget.sizeBytes)}. You can restore them from the Recycle Bin.` : undefined}
        footer={
          <>
            <Button variant="ghost" onClick={() => setTrashTarget(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={async () => {
                const t = trashTarget
                setTrashTarget(null)
                if (!t) return
                try {
                  await invoke('models:trash', t.id)
                  toast.success(`Moved ${t.name} to the Recycle Bin`)
                  if (selectedId === t.id) setSelectedId(null)
                } catch (e) {
                  toast.error(errorMessage(e))
                }
              }}
            >
              Move to Recycle Bin
            </Button>
          </>
        }
      >
        {trashTarget && <div className="px-5 py-3 font-mono text-xs break-all text-muted">{trashTarget.files.join('\n')}</div>}
      </Dialog>

      <Dialog
        open={!!adapterTrash}
        onOpenChange={(o) => !o && setAdapterTrash(null)}
        title="Move adapter to the Recycle Bin?"
        description={adapterTrash ? `${formatBytes(adapterTrash.sizeBytes)}. The file goes to the Recycle Bin, so you can restore it from there.` : undefined}
        footer={
          <>
            <Button variant="ghost" onClick={() => setAdapterTrash(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={async () => {
                const t = adapterTrash
                setAdapterTrash(null)
                if (!t) return
                try {
                  await invoke('adapters:trash', t.id)
                  toast.success(`Moved ${t.name} to the Recycle Bin`)
                } catch (e) {
                  toast.error(errorMessage(e))
                }
              }}
            >
              Move to Recycle Bin
            </Button>
          </>
        }
      >
        {adapterTrash && <div className="px-5 py-3 font-mono text-xs break-all text-muted">{adapterTrash.path}</div>}
      </Dialog>

      <AdapterConvertDialog open={convertOpen} onClose={() => setConvertOpen(false)} />
    </div>
  )
}

/** Every LoRA adapter file in the models folder, with what it was trained on. */
function AdaptersSection({ adapters, models, onTrash }: { adapters: AdapterEntry[]; models: ModelEntry[]; onTrash: (a: AdapterEntry) => void }) {
  return (
    <section className="mt-8" aria-labelledby="adapters-heading">
      <h2 id="adapters-heading" className="mb-2 font-[family-name:var(--font-display)] text-[13px] font-semibold">
        Adapters <span className="font-normal text-faint tabular-nums">· {adapters.length}</span>
      </h2>
      <table className="w-full table-fixed border-separate border-spacing-0 text-[13px]">
        <thead className="text-left text-[11px] text-faint uppercase">
          <tr>
            <th className="border-b border-border py-2 pl-3 font-medium">Adapter</th>
            <th className="w-28 border-b border-border py-2 font-medium">Architecture</th>
            <th className="w-16 border-b border-border py-2 font-medium">Alpha</th>
            <th className="w-48 border-b border-border py-2 font-medium">Base model</th>
            <th className="w-20 border-b border-border py-2 text-right font-medium">Size</th>
            <th className="w-12 border-b border-border py-2" />
          </tr>
        </thead>
        <tbody>
          {adapters.map((a) => {
            const orphan = !models.some((m) => adapterChoices(m, [a]).length > 0)
            return (
              <tr key={a.id} className="hover:bg-hover/50">
                <td className="border-b border-border py-2 pr-3 pl-3">
                  <div className="truncate font-medium" title={a.id}>
                    {a.name}
                  </div>
                  {a.error ? (
                    <div className="truncate text-[11px] text-danger" title={a.error}>
                      Unreadable: {a.error}
                    </div>
                  ) : (
                    orphan && <div className="text-[11px] text-faint">No model with this architecture</div>
                  )}
                </td>
                <td className="truncate border-b border-border py-2 text-muted">{a.architecture || '—'}</td>
                <td className="border-b border-border py-2 text-muted tabular-nums">{a.info?.loraAlpha ?? '—'}</td>
                <td className="truncate border-b border-border py-2 text-muted" title={a.info?.baseModelName}>
                  {a.info?.baseModelName ?? '—'}
                </td>
                <td className="border-b border-border py-2 text-right text-muted tabular-nums">{formatBytes(a.sizeBytes)}</td>
                <td className="border-b border-border py-2 pr-2 text-right">
                  <IconButton size="sm" label="Move to Recycle Bin" onClick={() => onTrash(a)}>
                    <Trash2 size={14} />
                  </IconButton>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}

function ModelDetails({ model, onClose, onBenchmark }: { model: ModelEntry; onClose: () => void; onBenchmark: () => void }) {
  const settings = useApp((s) => s.settings)
  const [config, setConfig] = useState<LoadConfig | null>(null)
  const [custom, setCustom] = useState(false)
  const [scope, setScope] = useState<ModeScope>('all')
  // Settings can be missing when they failed to load at start-up: the built-in defaults stand in.
  const estimate = useEstimate(model.id, config ?? settings?.defaultLoadConfig ?? DEFAULT_LOAD_CONFIG)

  useEffect(() => {
    // An OpenRouter model has no load settings.
    if (model.cloud) return
    void invoke('models:getDefaults', model.id).then(
      (d) => {
        setConfig(d.config)
        setCustom(d.custom)
      },
      (e) => {
        toast.error(`Could not read this model's load settings: ${errorMessage(e)}`)
        if (settings) setConfig(settings.defaultLoadConfig)
      }
    )
  }, [model.id])

  const i = model.info
  const cloud = model.cloud
  const rows: [string, string | undefined][] = cloud
    ? [
        ['Provider', 'OpenRouter'],
        ['Slug', cloud.slug],
        ['Price', priceLine(cloud.info)],
        ['Context the app uses', `${cloud.contextLimit.toLocaleString()} of ${cloud.info.contextLength.toLocaleString()}`],
        ['Longest reply', cloud.replyLimit.toLocaleString()],
        ['Features', badgesFor(cloud.info).map((b) => b.label).join(', ') || undefined],
        ['Other apps', cloud.allowApps ? 'May use it' : 'May not use it']
      ]
    : [
    ['Architecture', i?.architecture],
    ['Name in file', i?.name],
    ['Parameters', i?.sizeLabel],
    ['Quantization', i?.quantization],
    ['Trained context', i?.contextLength?.toLocaleString()],
    ['Layers', i?.blockCount?.toString()],
    ['Embedding size', i?.embeddingLength?.toString()],
    ['Attention heads', i?.headCount !== undefined ? `${i.headCount} (KV ${Math.round((i.headCountKv ?? 0) * 10) / 10})` : undefined],
    ['Experts', i?.expertCount ? `${i.expertUsedCount ?? '?'} of ${i.expertCount} active` : undefined],
    ['Sliding window', i?.slidingWindow?.toLocaleString()],
    ['Chat template', i ? (i.hasChatTemplate ? 'Included' : 'Missing') : undefined],
    ['Vision projector', model.mmproj ? `${model.mmproj.path.split(/[\\/]/).pop()} (${formatBytes(model.mmproj.sizeBytes)})` : undefined],
    ['Built-in MTP layers', model.info?.mtpLayers ? String(model.info.mtpLayers) : undefined],
    ['Draft / MTP file', model.draftSidecar ? `${model.draftSidecar.path.split(/[\\/]/).pop()} (${formatBytes(model.draftSidecar.sizeBytes)})` : undefined],
    ['Files', model.files.length > 1 ? `${model.files.length} shards` : undefined]
  ]

  return (
    <aside className="flex w-[400px] shrink-0 flex-col border-l border-border bg-panel">
      <div className="flex items-start justify-between gap-2 border-b border-border px-5 py-4">
        <div className="min-w-0">
          <div className="text-sm font-semibold break-words">{model.name}</div>
          {model.remote ? (
            <div className="mt-0.5 truncate text-[11px] text-faint">Runs on a rented vast.ai instance</div>
          ) : cloud ? (
            <div className="mt-0.5 text-[11px] text-faint">Runs at OpenRouter. Edit it on the OpenRouter page.</div>
          ) : (
            <button className="mt-0.5 truncate text-left text-[11px] text-faint hover:text-text" onClick={() => void invoke('shell:showItem', model.path)}>
              {model.path}
            </button>
          )}
        </div>
        <IconButton size="sm" label="Close" onClick={onClose}>
          ✕
        </IconButton>
      </div>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {model.error && <div className="rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">{model.error}</div>}
        <dl className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1.5 text-xs">
          {rows
            .filter(([, v]) => v)
            .map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-faint">{k}</dt>
                <dd className="break-words">{v}</dd>
              </div>
            ))}
        </dl>
        <ModeDefaultsEditor
          modelId={model.id}
          model={model}
          remote={model.remote}
          scope={scope}
          onScope={setScope}
          allView={
            cloud ? (
              <p className="text-xs text-muted">A model at OpenRouter is not loaded on this PC, so it has no load settings. Choose a mode above to set its own sampling and system prompt.</p>
            ) : (
            <>
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Default load settings</h3>
                  {custom ? <Badge tone="accent">Custom</Badge> : <Badge>App defaults</Badge>}
                </div>
                {config && <LoadSettingsForm model={model} config={config} onChange={setConfig} estimate={estimate} remote={model.remote} />}
              </div>
              <EstimateView estimate={estimate} />
            </>
            )
          }
        />
      </div>
      <div className="flex justify-end gap-2 border-t border-border px-5 py-3">
        {cloud && (
          <Button variant="ghost" onClick={() => useApp.getState().setPage('openrouter')}>
            <Globe size={14} /> OpenRouter page
          </Button>
        )}
        {!model.remote && !cloud && (
          <Button variant="ghost" disabled={!!model.error} onClick={onBenchmark}>
            <Gauge size={14} /> Benchmark
          </Button>
        )}
        {scope === 'all' && custom && !cloud && (
          <Button
            variant="ghost"
            onClick={async () => {
              await invoke('models:setDefaults', model.id, null)
              setCustom(false)
              if (settings) setConfig(settings.defaultLoadConfig)
              toast.success('Using app default settings')
            }}
          >
            Reset
          </Button>
        )}
        {scope === 'all' && !cloud && (
          <Button
            variant="primary"
            disabled={!config}
            onClick={async () => {
              if (!config) return
              await invoke('models:setDefaults', model.id, config)
              setCustom(true)
              toast.success('Saved default load settings')
            }}
          >
            Save as default
          </Button>
        )}
      </div>
    </aside>
  )
}
