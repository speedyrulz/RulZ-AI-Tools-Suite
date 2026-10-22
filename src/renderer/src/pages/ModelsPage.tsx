import { Ellipsis, Eye, FolderOpen, FolderSearch, Play, RefreshCw, Search, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import type { LoadConfig, ModelEntry } from '@shared/types'
import { EstimateView, useEstimate } from '../components/models/EstimateView'
import { LoadSettingsForm } from '../components/models/LoadSettingsForm'
import { Badge, Button, Dialog, DropdownMenu, IconButton, Input } from '../components/ui'
import { errorMessage, invoke } from '../lib/api'
import { cn, formatBytes, formatTokens } from '../lib/format'
import { useApp } from '../stores/app'
import { useModels } from '../stores/models'

type SortKey = 'name' | 'size' | 'modified'

export function ModelsPage() {
  const models = useModels((s) => s.models)
  const scanned = useModels((s) => s.scanned)
  const instances = useModels((s) => s.instances)
  const rescan = useModels((s) => s.rescan)
  const settings = useApp((s) => s.settings)
  const updateSettings = useApp((s) => s.updateSettings)
  const openPicker = useApp((s) => s.openPicker)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortKey>('name')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [scanning, setScanning] = useState(false)
  const [trashTarget, setTrashTarget] = useState<ModelEntry | null>(null)

  const list = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = q ? models.filter((m) => `${m.id} ${m.info?.architecture ?? ''}`.toLowerCase().includes(q)) : models
    return [...filtered].sort((a, b) =>
      sort === 'size' ? b.sizeBytes - a.sizeBytes : sort === 'modified' ? b.modifiedAt - a.modifiedAt : a.id.localeCompare(b.id)
    )
  }, [models, query, sort])
  const selected = models.find((m) => m.id === selectedId) ?? null
  const totalSize = models.reduce((sum, m) => sum + m.sizeBytes + (m.mmproj?.sizeBytes ?? 0), 0)

  const doRescan = async (): Promise<void> => {
    setScanning(true)
    await rescan()
    setScanning(false)
  }

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border px-6 py-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-base font-semibold">My Models</h1>
            <div className="truncate text-xs text-muted">
              {models.length} models · {formatBytes(totalSize)} · <span className="font-mono">{settings?.modelsDir}</span>
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
                {list.map((m) => {
                  const inst = instances.find((i) => i.modelId === m.id)
                  return (
                    <tr
                      key={m.id}
                      onClick={() => setSelectedId(m.id)}
                      className={cn('group cursor-default', selectedId === m.id ? 'bg-accent-soft/50' : 'hover:bg-hover/50')}
                    >
                      <td className="border-b border-border py-2 pr-3 pl-3">
                        <div className="flex items-center gap-1.5">
                          <span className="truncate font-medium">{m.name}</span>
                          {m.mmproj && <Eye size={12} className="shrink-0 text-accent" aria-label="Vision" />}
                          {inst && <Badge tone={inst.status === 'ready' ? 'ok' : 'accent'}>{inst.status === 'ready' ? 'Loaded' : 'Loading'}</Badge>}
                          {m.error && <Badge tone="danger">Unreadable</Badge>}
                        </div>
                        <div className="truncate text-[11px] text-faint">{m.folder || '/'}</div>
                      </td>
                      <td className="truncate border-b border-border py-2 text-muted">{m.info?.architecture ?? '—'}</td>
                      <td className="border-b border-border py-2 text-muted">{m.info?.sizeLabel ?? '—'}</td>
                      <td className="border-b border-border py-2">{m.info?.quantization ? <Badge>{m.info.quantization}</Badge> : '—'}</td>
                      <td className="border-b border-border py-2 text-muted tabular-nums">
                        {m.info?.contextLength ? formatTokens(m.info.contextLength) : '—'}
                      </td>
                      <td className="border-b border-border py-2 text-right text-muted tabular-nums">{formatBytes(m.sizeBytes)}</td>
                      <td className="border-b border-border py-2 pr-2">
                        <div className="flex justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                          <Button size="sm" variant="ghost" onClick={() => openPicker(m.id)}>
                            <Play size={12} /> Load
                          </Button>
                          <DropdownMenu
                            trigger={
                              <button className="rounded p-1.5 text-faint hover:bg-hover hover:text-text" aria-label="Model options">
                                <Ellipsis size={14} />
                              </button>
                            }
                            items={[
                              { label: 'Show in Explorer', icon: <FolderOpen size={13} />, onSelect: () => void invoke('shell:showItem', m.path) },
                              'sep',
                              { label: 'Move to Recycle Bin…', icon: <Trash2 size={13} />, onSelect: () => setTrashTarget(m), danger: true }
                            ]}
                          />
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {selected && <ModelDetails key={selected.id} model={selected} onClose={() => setSelectedId(null)} />}

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
    </div>
  )
}

function ModelDetails({ model, onClose }: { model: ModelEntry; onClose: () => void }) {
  const settings = useApp((s) => s.settings)
  const [config, setConfig] = useState<LoadConfig | null>(null)
  const [custom, setCustom] = useState(false)
  const estimate = useEstimate(model.id, config ?? settings!.defaultLoadConfig)

  useEffect(() => {
    void invoke('models:getDefaults', model.id).then((d) => {
      setConfig(d.config)
      setCustom(d.custom)
    })
  }, [model.id])

  const i = model.info
  const rows: [string, string | undefined][] = [
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
    ['Draft / MTP file', model.draft ? `${model.draft.path.split(/[\\/]/).pop()} (${formatBytes(model.draft.sizeBytes)})` : undefined],
    ['Files', model.files.length > 1 ? `${model.files.length} shards` : undefined]
  ]

  return (
    <aside className="flex w-[400px] shrink-0 flex-col border-l border-border bg-panel">
      <div className="flex items-start justify-between gap-2 border-b border-border px-5 py-4">
        <div className="min-w-0">
          <div className="text-sm font-semibold break-words">{model.name}</div>
          <button className="mt-0.5 truncate text-left text-[11px] text-faint hover:text-text" onClick={() => void invoke('shell:showItem', model.path)}>
            {model.path}
          </button>
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
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Default load settings</h3>
            {custom ? <Badge tone="accent">Custom</Badge> : <Badge>App defaults</Badge>}
          </div>
          {config && <LoadSettingsForm model={model} config={config} onChange={setConfig} />}
        </div>
        <EstimateView estimate={estimate} />
      </div>
      <div className="flex justify-end gap-2 border-t border-border px-5 py-3">
        {custom && (
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
      </div>
    </aside>
  )
}
