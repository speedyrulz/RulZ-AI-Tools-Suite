import { Pencil, Plus, SlidersHorizontal, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { RemoteModelDef, RemoteModelSource, RemoteState } from '@shared/remote'
import type { LoadConfig } from '@shared/types'
import { errorMessage, invoke } from '../../lib/api'
import { formatBytes } from '../../lib/format'
import { cleanRemoteSource, defaultModelName, remoteModelSize, sourceSummary, validateRemoteSource } from '../../lib/remote'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { isChannelBusy, useRemote } from '../../stores/remote'
import { EstimateView, useEstimate } from '../models/EstimateView'
import { LoadSettingsForm } from '../models/LoadSettingsForm'
import { Button, Dialog, Field, IconButton, Input, Select, Switch } from '../ui'
import { ConfirmDialog } from './ConfirmDialog'

/** The models that can run on a rental: a definition (a library model to upload, or a Hugging Face file) and its load settings. */
export function RemoteModels({ state }: { state: RemoteState }) {
  const library = useModels((s) => s.models)
  const [editing, setEditing] = useState<RemoteModelDef | 'new' | null>(null)
  const [settingsFor, setSettingsFor] = useState<RemoteModelDef | null>(null)
  const [deleting, setDeleting] = useState<RemoteModelDef | null>(null)
  const run = useRemote((s) => s.run)
  const deleteBusy = useRemote((s) => isChannelBusy(s, 'remote:deleteModel'))

  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">Remote models</h2>
          <p className="mt-0.5 text-xs text-muted">Models that run on a rented instance. Setting one up on a rental downloads or uploads its files there.</p>
        </div>
        <Button size="sm" onClick={() => setEditing('new')}>
          <Plus size={14} /> Add a model
        </Button>
      </div>
      {state.models.length === 0 ? (
        <p className="mt-4 text-xs text-muted">None yet. Add one from your library (it is uploaded) or from Hugging Face (the instance downloads it).</p>
      ) : (
        <ul className="mt-4 divide-y divide-border/60 rounded-lg border border-border">
          {state.models.map((m) => {
            const size = remoteModelSize(m, library)
            return (
              <li key={m.id} className="flex flex-wrap items-center gap-3 px-3 py-2.5">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] font-medium">{m.name}</div>
                  <div className="truncate text-[11px] text-faint" title={sourceSummary(m.source)}>
                    {sourceSummary(m.source)}
                    {size ? ` · ${formatBytes(size)}` : ''}
                  </div>
                </div>
                <Button size="sm" variant="ghost" onClick={() => setSettingsFor(m)}>
                  <SlidersHorizontal size={13} /> Load settings
                </Button>
                <IconButton size="sm" label="Edit" onClick={() => setEditing(m)}>
                  <Pencil size={14} />
                </IconButton>
                <IconButton size="sm" label="Delete" onClick={() => setDeleting(m)}>
                  <Trash2 size={14} />
                </IconButton>
              </li>
            )
          })}
        </ul>
      )}

      {editing && <ModelDefDialog def={editing === 'new' ? null : editing} onClose={() => setEditing(null)} />}
      {settingsFor && <SettingsDialog def={settingsFor} state={state} onClose={() => setSettingsFor(null)} />}
      <ConfirmDialog
        open={!!deleting}
        title="Delete this remote model?"
        lines={[
          `${deleting?.name ?? ''} is removed from your list.`,
          'Its files on rented instances are not touched and keep using disk (and billing). Remove them on each rental first if you do not want that.'
        ]}
        confirmLabel="Delete"
        variant="danger"
        busy={deleteBusy}
        onCancel={() => setDeleting(null)}
        onConfirm={() => {
          const d = deleting
          setDeleting(null)
          if (d) void run('remote:deleteModel', d.id)
        }}
      />
    </section>
  )
}

function ModelDefDialog({ def, onClose }: { def: RemoteModelDef | null; onClose: () => void }) {
  const run = useRemote((s) => s.run)
  const saving = useRemote((s) => isChannelBusy(s, 'remote:saveModel'))
  const library = useModels((s) => s.models).filter((m) => !m.remote && !m.error)
  const [src, setSrc] = useState<RemoteModelSource>(def?.source ?? { kind: 'local', modelId: '' })
  const [name, setName] = useState(def?.name ?? '')
  const [named, setNamed] = useState(!!def)
  const problem = validateRemoteSource(src)
  const local = src.kind === 'local' ? library.find((m) => m.id === src.modelId) : undefined
  const update = (next: RemoteModelSource): void => {
    setSrc(next)
    if (!named) setName(defaultModelName(next, library))
  }
  const hf = src.kind === 'hf' ? src : { kind: 'hf' as const, repo: '', bucket: false, revision: 'main', file: '', mmproj: '' }
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={def ? 'Edit remote model' : 'Add a remote model'}
      className="w-[min(600px,calc(100vw-32px))]"
      footer={
        <>
          <Button autoFocus variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={saving || !!problem || !name.trim()}
            onClick={async () => {
              const ok = await run('remote:saveModel', { ...(def ? { id: def.id } : {}), name: name.trim(), source: cleanRemoteSource(src) })
              if (ok) onClose()
            }}
          >
            Save
          </Button>
        </>
      }
    >
      <form
        className="space-y-3 px-5 py-4"
        onSubmit={(e) => {
          e.preventDefault()
        }}
      >
        <Field label="Where the files come from">
          <Select
            value={src.kind}
            onChange={(e) => update(e.target.value === 'local' ? { kind: 'local', modelId: '' } : { ...hf })}
            aria-label="Where the files come from"
          >
            <option value="local">A model in my library (uploaded from this PC)</option>
            <option value="hf">Hugging Face (the instance downloads it)</option>
          </Select>
        </Field>
        {src.kind === 'local' ? (
          <Field label="Library model" hint={local ? formatBytes(local.sizeBytes + (local.mmproj?.sizeBytes ?? 0)) : undefined}>
            <Select value={src.modelId} onChange={(e) => update({ kind: 'local', modelId: e.target.value })} aria-label="Library model">
              <option value="">Choose a model…</option>
              {library.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · {formatBytes(m.sizeBytes)}
                </option>
              ))}
            </Select>
          </Field>
        ) : (
          <>
            <Field label={src.bucket ? 'Bucket' : 'Repo'} hint="owner/name">
              <Input value={src.repo} placeholder="owner/name" onChange={(e) => update({ ...src, repo: e.target.value })} />
            </Field>
            <Field inline label="It is a bucket, not a model repo">
              <Switch checked={src.bucket} onCheckedChange={(v) => update({ ...src, bucket: v })} />
            </Field>
            <Field label="Model file" hint="the first part of a split model">
              <Input className="font-mono text-xs" value={src.file} placeholder="model-Q4_K_M.gguf" onChange={(e) => update({ ...src, file: e.target.value })} />
            </Field>
            <Field label="Vision projector file" hint="optional">
              <Input className="font-mono text-xs" value={src.mmproj} placeholder="mmproj-model-f16.gguf" onChange={(e) => update({ ...src, mmproj: e.target.value })} />
            </Field>
            <Field label="Revision" hint="empty = main">
              <Input className="font-mono text-xs" value={src.revision} placeholder="main" onChange={(e) => update({ ...src, revision: e.target.value })} />
            </Field>
          </>
        )}
        <Field label="Name">
          <Input
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setNamed(true)
            }}
          />
        </Field>
        {problem && (src.kind === 'local' ? !!src.modelId : src.repo || src.file) && <p className="text-xs text-warn">{problem}</p>}
      </form>
    </Dialog>
  )
}

/** The model's load settings, the same form as any model's, without what only makes sense on this PC. */
function SettingsDialog({ def, state, onClose }: { def: RemoteModelDef; state: RemoteState; onClose: () => void }) {
  const library = useModels((s) => s.models)
  const entry = library.find((m) => m.id === def.id) ?? null
  const appDefault = useApp((s) => s.settings?.defaultLoadConfig)
  const [config, setConfig] = useState<LoadConfig | null>(null)
  const [custom, setCustom] = useState(false)
  const running = state.rentals.filter((r) => r.rentalStatus === 'running')
  const [rentalId, setRentalId] = useState<number | null>(running[0]?.id ?? null)
  const estimate = useEstimate(def.id, config ?? appDefault ?? DEFAULT_LOAD_CONFIG, rentalId)
  useEffect(() => {
    invoke('models:getDefaults', def.id)
      .then((d) => {
        setConfig(d.config)
        setCustom(d.custom)
      })
      .catch((e) => {
        toast.error(`Could not read this model's saved settings: ${errorMessage(e)}`)
        setConfig(appDefault ?? DEFAULT_LOAD_CONFIG)
      })
  }, [def.id])
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={`Load settings: ${def.name}`}
      description="Used when this model is loaded on a rental. They are saved for this model."
      className="w-[min(680px,calc(100vw-32px))]"
      footer={
        <>
          {custom && (
            <Button
              variant="ghost"
              onClick={async () => {
                try {
                  await invoke('models:setDefaults', def.id, null)
                  setCustom(false)
                  if (appDefault) setConfig(appDefault)
                } catch (e) {
                  toast.error(errorMessage(e))
                }
              }}
            >
              Reset
            </Button>
          )}
          <Button autoFocus variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button
            disabled={!config}
            onClick={async () => {
              if (!config) return
              try {
                await invoke('models:setDefaults', def.id, config)
                toast.success('Saved the load settings')
                onClose()
              } catch (e) {
                toast.error(errorMessage(e))
              }
            }}
          >
            Save
          </Button>
        </>
      }
    >
      <div className="space-y-4 px-5 py-4">
        {running.length > 0 && (
          <Field label="Estimate for" hint="the memory estimate uses this rental's GPUs">
            <Select value={rentalId ?? ''} onChange={(e) => setRentalId(e.target.value ? Number(e.target.value) : null)} aria-label="Estimate for">
              {running.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.numGpus > 1 ? `${r.numGpus}× ` : ''}
                  {r.gpuName} ({r.id})
                </option>
              ))}
            </Select>
          </Field>
        )}
        {config && <LoadSettingsForm model={entry} config={config} onChange={setConfig} estimate={estimate} remote />}
        {running.length === 0 && <p className="text-xs text-muted">No instance is running, so there is no memory estimate. The settings are saved either way.</p>}
        <EstimateView estimate={estimate} />
      </div>
    </Dialog>
  )
}
