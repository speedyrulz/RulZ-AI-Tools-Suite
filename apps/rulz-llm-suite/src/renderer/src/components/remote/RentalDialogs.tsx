import { CircleAlert, RefreshCw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { AutoStop, Rental } from '@shared/remote'
import { invoke, errorMessage } from '../../lib/api'
import { cleanAutoStop, rentalLabel } from '../../lib/remote'
import { isBusy, useRemote } from '../../stores/remote'
import { Button, Dialog, Field, NumberInput, Switch } from '../ui'

/** Auto-stop settings of one rental: idle minutes, hard cap, the API and phone switch, and a warning when it is off. */
export function AutoStopDialog({ rental, onClose }: { rental: Rental; onClose: () => void }) {
  const run = useRemote((s) => s.run)
  const saving = useRemote((s) => isBusy(s, 'remote:setAutoStop', rental.id, cleanAutoStop(draftOf(rental))))
  const [draft, setDraft] = useState<AutoStop>(() => draftOf(rental))
  const set = <K extends keyof AutoStop>(k: K, v: AutoStop[K]): void => setDraft((d) => ({ ...d, [k]: v }))
  const off = !draft.enabled
  const noCap = draft.capHours === null
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Auto-stop settings"
      description={rentalLabel(rental)}
      footer={
        <>
          <Button autoFocus variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={saving}
            onClick={async () => {
              if (await run('remote:setAutoStop', rental.id, cleanAutoStop(draft))) onClose()
            }}
          >
            Save
          </Button>
        </>
      }
    >
      <div className="space-y-3 px-5 py-4">
        <Field inline label="Stop when idle" hint="Stops the instance when nothing has used it for the minutes below and its GPUs are quiet.">
          <Switch checked={draft.enabled} onCheckedChange={(v) => set('enabled', v)} />
        </Field>
        <Field inline label="Idle minutes" hint="Nothing used it and the GPUs were quiet this long">
          <NumberInput value={draft.idleMinutes} min={1} max={1440} onChange={(v) => set('idleMinutes', Math.max(1, Math.round(v ?? 30)))} />
        </Field>
        <Field inline label="Hard cap, hours after it started" hint="Stops it whatever runs. Empty = no cap.">
          <NumberInput value={draft.capHours} placeholder="None" min={1} max={720} onChange={(v) => set('capHours', v === null ? null : Math.max(1, v))} />
        </Field>
        <Field inline label="API and phone requests count as use" hint="Requests from the API server and the phone keep it running too. Off: only this window counts.">
          <Switch checked={draft.apiCountsAsUse} onCheckedChange={(v) => set('apiCountsAsUse', v)} />
        </Field>
        {(off || noCap) && (
          <div className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
            <CircleAlert size={14} className="mt-px shrink-0" />
            <div>
              {off && noCap
                ? `With both off, ${rentalLabel(rental)} bills until you stop it yourself, even if nothing uses it.`
                : off
                  ? 'With idle stop off, an idle instance keeps billing until the hard cap.'
                  : 'With no hard cap, a busy instance can bill for days.'}
            </div>
          </div>
        )}
        {rental.watchdog.state === 'not-armed' && (
          <div className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
            <CircleAlert size={14} className="mt-px shrink-0" />
            <div>Cannot stop itself{rental.watchdog.error ? `: ${rental.watchdog.error}` : ''}. Only this app's timer protects it, so a closed app or a crashed PC will not stop it.</div>
          </div>
        )}
      </div>
    </Dialog>
  )
}

const draftOf = (r: Rental): AutoStop => ({ ...r.autoStop })

/** The rental's setup and connection log, refreshed while it is open. */
export function RentalLogDialog({ rental, onClose }: { rental: Rental; onClose: () => void }) {
  const [lines, setLines] = useState<string[] | null>(null)
  const [error, setError] = useState('')
  const box = useRef<HTMLPreElement>(null)
  const load = (): void => {
    invoke('remote:log', rental.id).then(
      (l) => {
        setLines(l)
        setError('')
      },
      (e) => setError(errorMessage(e))
    )
  }
  useEffect(() => {
    load()
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [rental.id])
  useEffect(() => {
    const el = box.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Log"
      description={rentalLabel(rental)}
      className="w-[min(860px,calc(100vw-32px))]"
      footer={
        <>
          <Button variant="ghost" onClick={load}>
            <RefreshCw size={14} /> Refresh
          </Button>
          <Button autoFocus onClick={onClose}>
            Close
          </Button>
        </>
      }
    >
      <div className="p-4">
        {error && <div className="mb-2 text-xs text-danger-text">{error}</div>}
        <pre ref={box} className="h-[50vh] overflow-auto rounded-lg border border-border bg-code p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-body" tabIndex={0}>
          {lines === null ? 'Reading the log…' : lines.length ? lines.join('\n') : 'The log is empty.'}
        </pre>
        <p className="mt-2 text-[11px] text-faint">Keys and tokens are removed from these lines before they are shown.</p>
      </div>
    </Dialog>
  )
}
