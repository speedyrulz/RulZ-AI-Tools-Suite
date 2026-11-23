import {
  ArrowUpFromLine,
  CircleAlert,
  CircleCheck,
  CircleHelp,
  CircleOff,
  CirclePause,
  FileText,
  LoaderCircle,
  Play,
  Plug,
  Power,
  Settings2,
  Timer,
  Trash2
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { RemoteModelDef, Rental, RentalModel } from '@shared/remote'
import { cn, formatBytes } from '../../lib/format'
import {
  autoStopLine,
  canLoadOn,
  countdownSeconds,
  destroyConfirmText,
  engineWords,
  gpuSummary,
  inUseText,
  modelStateWords,
  money,
  perHour,
  rentalActions,
  rentalLabel,
  sshWords,
  statusWords,
  stopConfirmText,
  type StatusIcon
} from '../../lib/remote'
import { isBusy, loadRemoteModel, useRemote } from '../../stores/remote'
import { useModels } from '../../stores/models'
import { Badge, Button, DropdownMenu, ProgressBar, Tooltip } from '../ui'
import { ConfirmDialog } from './ConfirmDialog'
import { AutoStopDialog, RentalLogDialog } from './RentalDialogs'
import { useNow } from './useNow'

const ICONS: Record<StatusIcon, ReactNode> = {
  running: <CircleCheck size={12} />,
  starting: <LoaderCircle size={12} className="animate-spin" />,
  stopping: <LoaderCircle size={12} className="animate-spin" />,
  stopped: <CirclePause size={12} />,
  gone: <CircleOff size={12} />,
  unknown: <CircleHelp size={12} />
}

type Confirm = null | 'start' | 'stop' | 'stopAgain' | 'destroy' | 'destroyAgain' | 'stopUsing' | 'forget' | { removeFiles: string }

export function RentalCard({ rental: r, models }: { rental: Rental; models: RemoteModelDef[] }) {
  const run = useRemote((s) => s.run)
  const now = useNow(1000)
  const instances = useModels((s) => s.instances)
  const [confirm, setConfirm] = useState<Confirm>(null)
  const [autoDialog, setAutoDialog] = useState(false)
  const [logDialog, setLogDialog] = useState(false)
  const busy = useRemote((s) => Object.keys(s.inflight).some((k) => k.includes(`:[${r.id}`)))
  const a = rentalActions(r)
  const st = statusWords(r)
  const ssh = sshWords(r)
  const eng = engineWords(r)
  const line = autoStopLine(r, now)
  const seconds = countdownSeconds(r, now)
  const attached = r.origin === 'attached'
  const here = instances.filter((i) => i.host?.rentalId === r.id && i.status !== 'stopping')
  const working = r.models.some((m) => m.state === 'working') || r.engine.state === 'working'
  const close = (): void => setConfirm(null)
  const stopText = stopConfirmText(r)
  const destroyText = destroyConfirmText(r)
  const stopAgain = inUseText(r, 'stop')
  const destroyAgain = inUseText(r, 'destroy')

  return (
    <article className="rounded-xl border border-border bg-panel" aria-label={rentalLabel(r)}>
      <header className="flex flex-wrap items-start gap-x-4 gap-y-2 border-b border-border px-5 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-[family-name:var(--font-display)] text-sm font-semibold">{gpuSummary(r)}</h3>
            <Badge tone={st.tone}>
              {ICONS[st.icon]} {st.label}
            </Badge>
            {attached ? (
              <Tooltip content="This instance was rented elsewhere. The app uses it, but never destroys it, and its auto-stop is off until you switch it on.">
                <span>
                  <Badge>Rented elsewhere</Badge>
                </span>
              </Tooltip>
            ) : (
              <Badge>Rented by this app</Badge>
            )}
          </div>
          <div className="mt-0.5 truncate text-[11px] text-faint">
            {r.label} · instance {r.id}
            {r.statusMessage ? ` · ${r.statusMessage}` : ''}
          </div>
        </div>
        <div className="text-right text-[13px] tabular-nums">
          {r.rentalStatus === 'stopped' ? (
            <>
              <div className="font-medium">{r.storageDollarsPerHour === null ? 'Disk price not reported' : `${perHour(r.storageDollarsPerHour)} for the disk`}</div>
              <div className="text-[11px] text-faint">{r.diskGb} GB kept</div>
            </>
          ) : (
            <>
              <div className="font-medium">{perHour(r.dollarsPerHour)}</div>
              <div className="text-[11px] text-faint">{money(r.spent)} spent in this run</div>
            </>
          )}
        </div>
      </header>

      <div className="space-y-3 px-5 py-4">
        {r.countdown && (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn" role="alert">
            <Timer size={14} className="shrink-0" />
            <div className="min-w-0 flex-1">
              <span className="font-medium tabular-nums">Stopping in {seconds ?? 0} s.</span> {r.countdown.reason}
            </div>
            <Button size="sm" disabled={busy} onClick={() => void run('remote:keepRunning', r.id)}>
              Keep running
            </Button>
          </div>
        )}

        {r.error && (
          <div className="flex items-start gap-2 rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">
            <CircleAlert size={14} className="mt-px shrink-0" />
            <div className="min-w-0 flex-1">{r.error}</div>
          </div>
        )}

        <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
          <Fact label="Connection">
            <Badge tone={ssh.tone}>{ssh.label}</Badge>
          </Fact>
          <Fact label="Engine">
            <Badge tone={eng.tone}>{eng.label}</Badge>
          </Fact>
          {r.rentalStatus === 'running' && (
            <Fact label="Free disk">
              <span className="tabular-nums">{r.diskFreeBytes === null ? 'Not read yet' : `${formatBytes(r.diskFreeBytes)} of ${r.diskGb} GB`}</span>
            </Fact>
          )}
          <Fact label="Using it now">
            <span>{r.inUse.length ? r.inUse.join(', ') : r.rentalStatus === 'running' ? 'Nothing (idle)' : '—'}</span>
          </Fact>
        </dl>

        {r.rentalStatus === 'running' && r.gpus.length > 0 && (
          <div className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
            {r.gpus.map((g) => {
              const frac = g.totalMiB ? g.usedMiB / g.totalMiB : 0
              return (
                <div key={g.index} className="text-[11px] text-muted">
                  <div className="mb-1 flex justify-between gap-2">
                    <span className="truncate">
                      GPU{g.index} memory
                    </span>
                    <span className="tabular-nums">
                      {formatBytes(g.usedMiB * 1024 * 1024)} / {formatBytes(g.totalMiB * 1024 * 1024, 0)} · {Math.round(g.utilization)}% busy
                    </span>
                  </div>
                  <ProgressBar value={frac} className={cn(frac > 0.92 && '[&>div]:bg-danger', frac > 0.75 && frac <= 0.92 && '[&>div]:bg-warn')} />
                </div>
              )
            })}
          </div>
        )}

        {r.foreignProcesses && (
          <div className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
            <CircleAlert size={14} className="mt-px shrink-0" />
            <div>Something else is using this instance&apos;s GPU. The app will not load a model over it on an instance rented elsewhere, and stopping it stops that too.</div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {line.badge ? (
            <Badge tone="warn">
              <CircleAlert size={12} /> {line.text}
            </Badge>
          ) : (
            <span className="flex items-center gap-1.5 text-muted">
              <Timer size={13} /> {line.text}
            </span>
          )}
        </div>

        {r.rentalStatus === 'running' && (
          <div className="space-y-1.5">
            <h4 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Models</h4>
            {models.length === 0 ? (
              <p className="text-xs text-muted">No remote models yet. Add one under Remote models below, then set it up here.</p>
            ) : (
              <ul className="divide-y divide-border/60 rounded-lg border border-border">
                {models.map((m) => (
                  <ModelRow
                    key={m.id}
                    def={m}
                    rental={r}
                    state={r.models.find((x) => x.modelId === m.id)}
                    loadedAs={here.filter((i) => i.modelId === m.id)}
                    onRemove={() => setConfirm({ removeFiles: m.id })}
                  />
                ))}
              </ul>
            )}
            {a.setupEngine && (
              <Button size="sm" onClick={() => void run('remote:setup', r.id, null)}>
                Install the engine
              </Button>
            )}
            {working && (
              <Button size="sm" variant="ghost" onClick={() => void run('remote:cancelSetup', r.id)}>
                Cancel set-up
              </Button>
            )}
          </div>
        )}
      </div>

      <footer className="flex flex-wrap items-center gap-2 border-t border-border px-5 py-3">
        {a.start && (
          <Button size="sm" disabled={busy} onClick={() => setConfirm('start')}>
            <Play size={13} /> Start
          </Button>
        )}
        {a.stop && (
          <Button size="sm" disabled={busy} onClick={() => setConfirm('stop')}>
            <Power size={13} /> Stop
          </Button>
        )}
        {a.autoStopSettings && (
          <Button size="sm" variant="ghost" onClick={() => setAutoDialog(true)}>
            <Settings2 size={13} /> Auto-stop
          </Button>
        )}
        {a.extendCap && (
          <DropdownMenu
            align="start"
            trigger={
              <Button size="sm" variant="ghost">
                <Timer size={13} /> Extend cap
              </Button>
            }
            items={[1, 4, 12].map((h) => ({ label: `Add ${h} hour${h === 1 ? '' : 's'} to the cap`, onSelect: () => void run('remote:extendCap', r.id, h) }))}
          />
        )}
        <Button size="sm" variant="ghost" onClick={() => setLogDialog(true)}>
          <FileText size={13} /> Log
        </Button>
        <div className="flex-1" />
        {a.stopUsing && (
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => setConfirm('stopUsing')}>
            <Plug size={13} /> Stop using
          </Button>
        )}
        {a.forget && (
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => setConfirm('forget')}>
            Forget it
          </Button>
        )}
        {a.destroy && (
          <Button size="sm" variant="danger" disabled={busy} onClick={() => setConfirm('destroy')}>
            <Trash2 size={13} /> Destroy
          </Button>
        )}
      </footer>

      <ConfirmDialog
        open={confirm === 'start'}
        title="Start this instance?"
        lines={[
          `${rentalLabel(r)} starts billing again at ${perHour(r.dollarsPerHour)} (about ${money(r.dollarsPerHour * 24)} a day) until it is stopped.`,
          'Its disk and files are kept, so models set up before load quickly. A host can be out of capacity, and then it does not start.'
        ]}
        confirmLabel={`Start for ${perHour(r.dollarsPerHour)}`}
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          close()
          void run('remote:start', r.id)
        }}
      />
      <ConfirmDialog
        open={confirm === 'stop'}
        title={stopText.title}
        lines={stopText.lines}
        confirmLabel={stopText.button}
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          if (stopText.asksAgain) return setConfirm('stopAgain')
          close()
          void run('remote:stop', r.id, false)
        }}
      />
      <ConfirmDialog
        open={confirm === 'stopAgain'}
        title={stopAgain.title}
        lines={stopAgain.lines}
        confirmLabel={stopAgain.button}
        variant="danger"
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          close()
          void run('remote:stop', r.id, true)
        }}
      />
      <ConfirmDialog
        open={confirm === 'destroy'}
        title={destroyText.title}
        lines={destroyText.lines}
        confirmLabel={destroyText.button}
        variant="danger"
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          if (destroyText.asksAgain) return setConfirm('destroyAgain')
          close()
          void run('remote:destroy', r.id, r.label, false)
        }}
      />
      <ConfirmDialog
        open={confirm === 'destroyAgain'}
        title={destroyAgain.title}
        lines={destroyAgain.lines}
        confirmLabel={destroyAgain.button}
        variant="danger"
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          close()
          void run('remote:destroy', r.id, r.label, true)
        }}
      />
      <ConfirmDialog
        open={confirm === 'stopUsing'}
        title="Stop using this instance?"
        lines={[
          `${rentalLabel(r)} was rented elsewhere. The app stops using it and unloads the models it loaded there.`,
          'The instance keeps running and billing on your vast.ai account. Stop it on vast.ai, or stop it here first.'
        ]}
        confirmLabel="Stop using it"
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          close()
          void run('remote:detach', r.id)
        }}
      />
      <ConfirmDialog
        open={confirm === 'forget'}
        title="Forget this rental?"
        lines={['It is no longer on your vast.ai account, so nothing is billed. The app drops it from the list.']}
        confirmLabel="Forget it"
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          close()
          void run('remote:detach', r.id)
        }}
      />
      <ConfirmDialog
        open={typeof confirm === 'object' && confirm !== null}
        title="Remove the model files from this instance?"
        lines={[
          `The files of ${typeof confirm === 'object' && confirm ? (models.find((m) => m.id === confirm.removeFiles)?.name ?? confirm.removeFiles) : 'the model'} are deleted from the disk of ${rentalLabel(r)}.`,
          'The model stays in your list, and you can set it up again; that downloads or uploads it again, which can cost traffic.'
        ]}
        confirmLabel="Remove the files"
        variant="danger"
        busy={busy}
        onCancel={close}
        onConfirm={() => {
          const id = typeof confirm === 'object' && confirm ? confirm.removeFiles : ''
          close()
          if (id) void run('remote:removeModelFiles', r.id, id)
        }}
      />
      {autoDialog && <AutoStopDialog rental={r} onClose={() => setAutoDialog(false)} />}
      {logDialog && <RentalLogDialog rental={r} onClose={() => setLogDialog(false)} />}
    </article>
  )
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <dt className="w-24 shrink-0 text-faint">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

function ModelRow({
  def,
  rental,
  state,
  loadedAs,
  onRemove
}: {
  def: RemoteModelDef
  rental: Rental
  state: RentalModel | undefined
  loadedAs: { instanceId: string; status: string }[]
  onRemove: () => void
}) {
  const run = useRemote((s) => s.run)
  const unload = useModels((s) => s.unload)
  const [loading, setLoading] = useState(false)
  const words = modelStateWords(state?.state ?? 'none')
  const ready = canLoadOn(rental, def.id)
  const loaded = loadedAs.length > 0
  const setting = useRemote((s) => isBusy(s, 'remote:setup', rental.id, def.id))
  const canSetUp = rental.ssh.state === 'ok' && (!state || state.state === 'none' || state.state === 'failed') && !rental.models.some((m) => m.state === 'working')
  return (
    <li className="space-y-1.5 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium">{def.name}</span>
        <Badge tone={loaded ? 'ok' : words.tone}>
          {loaded ? <CircleCheck size={12} /> : state?.state === 'working' ? <LoaderCircle size={12} className="animate-spin" /> : null}
          {loaded ? 'Loaded' : words.label}
        </Badge>
        {ready && !loaded && (
          <Button
            size="sm"
            disabled={loading}
            onClick={async () => {
              setLoading(true)
              try {
                await loadRemoteModel(rental.id, def.id)
              } finally {
                setLoading(false)
              }
            }}
          >
            {loading ? <LoaderCircle size={13} className="animate-spin" /> : <Play size={13} />} Load
          </Button>
        )}
        {loaded && (
          <Button size="sm" variant="ghost" onClick={() => loadedAs.forEach((i) => void unload(i.instanceId))}>
            <ArrowUpFromLine size={13} /> Eject
          </Button>
        )}
        {canSetUp && (
          <Button size="sm" disabled={setting} onClick={() => void run('remote:setup', rental.id, def.id)}>
            {state?.state === 'failed' ? 'Try again' : 'Set up'}
          </Button>
        )}
        {state && state.state !== 'working' && state.state !== 'none' && !loaded && (
          <Button size="sm" variant="ghost" onClick={onRemove}>
            Remove files
          </Button>
        )}
      </div>
      {state?.state === 'working' && (
        <div className="space-y-1">
          <ProgressBar value={state.bytesTotal ? state.bytesDone / state.bytesTotal : 0} />
          <div className="flex justify-between text-[11px] text-muted tabular-nums">
            <span>{state.stage || 'Working'}</span>
            <span>{state.bytesTotal ? `${formatBytes(state.bytesDone)} of ${formatBytes(state.bytesTotal)}` : ''}</span>
          </div>
        </div>
      )}
      {state?.state === 'failed' && state.error && <div className="text-[11px] text-danger-text">{state.error}</div>}
    </li>
  )
}
