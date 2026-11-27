import { Plug } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { call } from '../../lib/api'
import { useApp } from '../../stores/app'
import { Button, Card, DebouncedInput, Dialog, Field, StatusChip } from '../ui'

/** Compares URLs ignoring case, scheme and trailing slashes. */
function sameUrl(a: string | null, b: string): boolean {
  const n = (u: string): string => u.trim().toLowerCase().replace(/^https?:\/\//, '').replace(/\/+$/, '')
  return a !== null && n(a) === n(b) && n(b) !== ''
}

export function UrlCard({ url, onUrl }: { url: string; onUrl: (v: string) => Promise<void> }) {
  const status = useApp((s) => s.status)
  const setStatus = useApp((s) => s.setStatus)
  const [draft, setDraft] = useState(url)
  const [checking, setChecking] = useState(false)
  const [asking, setAsking] = useState(false)
  const flush = useRef<(() => Promise<void>) | null>(null)

  // Only a status for the URL in the field counts.
  const shown = status && !status.own && sameUrl(status.url, draft) ? status : null
  const needsAccept = !!shown && shown.reachable && !shown.sharedQueueAccepted

  // The dialog is for the server it names: close it if the field changes.
  useEffect(() => {
    if (!shown) setAsking(false)
  }, [shown])

  const check = async (): Promise<void> => {
    setChecking(true)
    try {
      await flush.current?.()
      const s = await call('comfy:status')
      if (!s) return
      setStatus(s)
      if (s.reachable && !s.own && !s.sharedQueueAccepted && sameUrl(s.url, draft)) setAsking(true)
    } finally {
      setChecking(false)
    }
  }

  const accept = async (): Promise<void> => {
    if (!shown?.url) return
    setAsking(false)
    const s = await call('comfy:acceptSharedQueue', shown.url)
    if (s) setStatus(s)
  }

  return (
    <Card title="Server URL" description="The address of the ComfyUI you want to use, for example http://127.0.0.1:8188.">
      <div className="flex items-end gap-3">
        <div className="min-w-0 flex-1">
          <Field label="URL">
            <DebouncedInput mono value={url} placeholder="http://127.0.0.1:8188" onCommit={onUrl} onDraft={setDraft} flushRef={flush} />
          </Field>
        </div>
        <Button onClick={() => void check()} disabled={checking || draft.trim() === ''}>
          <Plug size={16} />
          Check connection
        </Button>
      </div>

      {checking && (
        <div className="mt-4">
          <StatusChip tone="warn" busy label="Checking" />
        </div>
      )}
      {!checking && shown && (
        <div className="mt-4 space-y-1">
          {shown.reachable ? <StatusChip tone="ok" label="Reachable" /> : <StatusChip tone="danger" label="Not reachable" />}
          {shown.reachable ? (
            <p className="text-xs text-faint">
              {shown.version ? `Version ${shown.version}. ` : ''}
              {shown.devices.length ? shown.devices.join(', ') : 'Devices unknown'}
            </p>
          ) : (
            <p className="text-[13px] text-danger-text">{shown.error ?? 'The server did not answer.'}</p>
          )}
          {needsAccept && (
            <div className="pt-2">
              <p className="mb-2 text-xs text-muted">Jobs will not be sent until you agree to use this server.</p>
              <Button size="sm" onClick={() => setAsking(true)}>
                Use this server
              </Button>
            </div>
          )}
        </div>
      )}

      <Dialog
        open={asking && !!shown}
        onOpenChange={setAsking}
        title="Use this server?"
        footer={
          <>
            <Button onClick={() => setAsking(false)}>Cancel</Button>
            <Button variant="primary" onClick={() => void accept()}>
              Use this server
            </Button>
          </>
        }
      >
        <p className="text-[13px] text-body">
          Jobs from this app will go into the queue of the server at <span className="font-mono text-xs">{shown?.url}</span>, alongside your other
          renders. The app never clears that queue, cancels other jobs or unloads its models.
        </p>
      </Dialog>
    </Card>
  )
}
