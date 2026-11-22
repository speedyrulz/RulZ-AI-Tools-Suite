import { Eye, EyeOff, RefreshCw, Smartphone, Unplug } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { PhonePairing, PhoneStatus } from '@shared/phone'
import { invoke, on } from '../../lib/api'
import { relativeTime } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Button, Field, NumberInput, Switch } from '../ui'
import { ConfirmDialog } from '../writing/common'

/** What the line under the switch says about the state, or null when there is nothing to add. */
export function phoneStateLine(status: PhoneStatus): { tone: 'muted' | 'warn' | 'danger' | 'ok'; text: string } | null {
  if (!status.enabled) return null
  if (status.state === 'starting') return { tone: 'muted', text: 'Starting...' }
  if (status.state === 'no-tailscale') return { tone: 'warn', text: "Tailscale isn't connected on this computer. Start Tailscale and sign in; I'll keep checking." }
  if (status.state === 'error') return { tone: 'danger', text: status.error ?? 'Phone access could not start.' }
  if (status.state === 'on') return { tone: 'ok', text: 'On. Your phone can connect over Tailscale.' }
  return null
}

const TONE = { muted: 'text-muted', warn: 'text-warn', danger: 'text-danger', ok: 'text-ok' } as const

/** Phone access over Tailscale: the switch, the QR code to pair the phone app, and the phones connected now. */
export function PhoneAccessSection() {
  const port = useApp((s) => s.settings?.phonePort ?? 7421)
  const update = useApp((s) => s.updateSettings)
  const [status, setStatus] = useState<PhoneStatus | null>(null)
  const [pairing, setPairing] = useState<PhonePairing | null>(null)
  const [showCode, setShowCode] = useState(false)
  const [busy, setBusy] = useState(false)
  const [confirmNew, setConfirmNew] = useState(false)
  /** Counts pairing requests, so a slow one can't put an older code back on screen. */
  const pairingRequest = useRef(0)

  useEffect(() => {
    void invoke('phone:status').then(setStatus, () => undefined)
    return on('phone:changed', setStatus)
  }, [])

  const ready = status?.state === 'on'
  const where = `${status?.addresses.join() ?? ''}:${status?.port ?? ''}`
  useEffect(() => {
    if (!ready) {
      setPairing(null)
      return
    }
    const n = ++pairingRequest.current
    void invoke('phone:pairing').then((p) => n === pairingRequest.current && setPairing(p), () => undefined)
    return () => {
      pairingRequest.current++
    }
  }, [ready, where])

  if (!status) return <p className="text-[13px] text-faint">Loading...</p>

  const toggle = async (enabled: boolean): Promise<void> => {
    setBusy(true)
    try {
      setStatus(await invoke('phone:setEnabled', enabled))
    } finally {
      setBusy(false)
    }
  }
  const newCode = async (): Promise<void> => {
    const n = ++pairingRequest.current
    const p = await invoke('phone:newToken')
    if (n === pairingRequest.current) setPairing(p)
  }
  const line = phoneStateLine(status)

  return (
    <div className="space-y-3">
      <Field label="Let my phone connect" hint="Only your own devices on Tailscale can reach it, and only with the pairing code" inline>
        <Switch checked={status.enabled} disabled={busy} onCheckedChange={(v) => void toggle(v)} />
      </Field>
      {line && <p className={`text-[13px] ${TONE[line.tone]}`}>{line.text}</p>}
      <Field label="Port" hint="The phone connects to this port on this computer's Tailscale address" inline>
        <NumberInput value={port} min={1} max={65535} onChange={(v) => v && void update({ phonePort: Math.round(v) })} />
      </Field>
      {ready && pairing && (
        <div className="flex flex-wrap gap-4 rounded-lg border border-border bg-panel-2/50 p-3">
          <img
            src={`data:image/svg+xml;utf8,${encodeURIComponent(pairing.qrSvg)}`}
            alt="QR code for pairing the phone app"
            className="size-36 shrink-0 rounded-md bg-white p-1"
          />
          <div className="min-w-0 space-y-1.5 text-[13px]">
            <p className="text-muted">
              Scan this with RulZ Writer on your phone, or tap <b className="text-text">Add computer</b> there and type:
            </p>
            <p>
              Address: <span className="font-mono">{pairing.host}</span>
              {status.dnsName && <span className="text-faint"> ({status.dnsName})</span>}
            </p>
            <p>
              Port: <span className="font-mono">{pairing.port}</span>
            </p>
            <p className="flex items-center gap-1.5">
              Pairing code: <span className="min-w-0 truncate font-mono">{showCode ? pairing.token : '*'.repeat(12)}</span>
              <button className="text-muted hover:text-text" aria-label={showCode ? 'Hide the code' : 'Show the code'} onClick={() => setShowCode(!showCode)}>
                {showCode ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </p>
            <p className="break-all text-[11px] text-faint">Link: {showCode ? pairing.url : pairing.url.replace(pairing.token, '...')}</p>
            <Button size="sm" onClick={() => setConfirmNew(true)}>
              <RefreshCw size={12} /> New pairing code
            </Button>
          </div>
        </div>
      )}
      {ready && (
        <div className="text-[13px]">
          {status.clients.length === 0 ? (
            <p className="text-faint">No phone connected right now.</p>
          ) : (
            <ul className="space-y-1.5">
              {status.clients.map((c) => (
                <li key={c.id} className="flex items-center gap-2">
                  <Smartphone size={14} className="shrink-0 text-ok" />
                  <span className="min-w-0 truncate">{c.device}</span>
                  <span className="shrink-0 text-faint">
                    {c.address}, connected {relativeTime(c.connectedAt)}
                  </span>
                  <Button size="sm" variant="ghost" className="ml-auto" onClick={() => void invoke('phone:disconnect', c.id)}>
                    <Unplug size={12} /> Disconnect
                  </Button>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-2 text-[11px] text-faint">
            The first time, Windows may ask whether RulZ LLM Suite may use your networks. Allow it, or your phone can't connect. RulZ LLM Suite must be running
            (it can stay in the tray) for the phone to reach your projects.
          </p>
        </div>
      )}
      <ConfirmDialog
        open={confirmNew}
        onOpenChange={setConfirmNew}
        title="Make a new pairing code?"
        body="Phones that are paired now are disconnected and must pair again with the new code."
        confirm="New pairing code"
        onConfirm={() => void newCode()}
      />
    </div>
  )
}
