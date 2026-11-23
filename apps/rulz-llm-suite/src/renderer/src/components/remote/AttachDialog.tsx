import { LoaderCircle, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { RemoteState, VastInstance } from '@shared/remote'
import { errorMessage, invoke } from '../../lib/api'
import { money, perHour } from '../../lib/remote'
import { isBusy, useRemote } from '../../stores/remote'
import { Badge, Button, Dialog } from '../ui'

/** Instances of the account that this app does not use yet, to attach. Attaching rents and starts nothing. */
export function AttachDialog({ state, onClose }: { state: RemoteState; onClose: () => void }) {
  const run = useRemote((s) => s.run)
  const [list, setList] = useState<VastInstance[] | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const load = (): void => {
    setLoading(true)
    invoke('remote:instances')
      .then(
        (l) => {
          setList(l)
          setError('')
        },
        (e) => setError(errorMessage(e))
      )
      .finally(() => setLoading(false))
  }
  useEffect(load, [])
  const used = new Set(state.rentals.map((r) => r.id))
  const free = (list ?? []).filter((i) => !used.has(i.id))
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Attach an instance"
      description="Use an instance you rented elsewhere. The app never destroys it, and its auto-stop and time cap stay off until you switch them on."
      className="w-[min(760px,calc(100vw-32px))]"
      footer={
        <>
          <Button variant="ghost" disabled={loading} onClick={load}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : undefined} /> Refresh
          </Button>
          <Button autoFocus onClick={onClose}>
            Close
          </Button>
        </>
      }
    >
      <div className="px-5 py-4">
        {error && <div className="mb-2 text-xs text-danger-text">{error}</div>}
        {list === null && !error ? (
          <p className="flex items-center gap-2 py-6 text-sm text-muted">
            <LoaderCircle size={14} className="animate-spin" /> Reading your instances…
          </p>
        ) : free.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted">Every instance of your account is already used here, or there are none.</p>
        ) : (
          <ul className="divide-y divide-border/60 rounded-lg border border-border">
            {free.map((i) => (
              <li key={i.id} className="flex flex-wrap items-center gap-3 px-3 py-2.5">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] font-medium">
                    {i.numGpus > 1 ? `${i.numGpus}× ` : ''}
                    {i.gpuName} <span className="font-normal text-faint">· {i.label || `instance ${i.id}`}</span>
                  </div>
                  <div className="text-[11px] text-muted tabular-nums">
                    {perHour(i.dollarsPerHour)} · disk {i.diskGb} GB
                    {i.storageDollarsPerHour !== null ? ` (${money(i.storageDollarsPerHour * 24)}/day when stopped)` : ''} · instance {i.id}
                  </div>
                </div>
                <Badge tone={i.status === 'running' ? 'ok' : 'neutral'}>{i.status === 'running' ? 'Running' : i.status === 'exited' ? 'Stopped' : i.status}</Badge>
                <AttachButton id={i.id} onDone={onClose} runAttach={(id) => run('remote:attach', id)} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </Dialog>
  )
}

function AttachButton({ id, onDone, runAttach }: { id: number; onDone: () => void; runAttach: (id: number) => Promise<unknown> }) {
  const busy = useRemote((s) => isBusy(s, 'remote:attach', id))
  return (
    <Button
      size="sm"
      disabled={busy}
      onClick={async () => {
        if (await runAttach(id)) onDone()
      }}
    >
      {busy && <LoaderCircle size={13} className="animate-spin" />} Attach
    </Button>
  )
}
