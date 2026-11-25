import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, CircleAlert, CircleCheck, KeyRound, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { RemoteState } from '@shared/remote'
import { money } from '../../lib/remote'
import { isBusy, isChannelBusy, useRemote } from '../../stores/remote'
import { Badge, Button, Field, Input } from '../ui'
import { ConfirmDialog } from './ConfirmDialog'

/** A secret typed here is sent once and cleared; the window never shows it again, only where it comes from. */
export function SecretField({
  label,
  source,
  placeholder,
  saving,
  onSave,
  onForget,
  forgetTitle = 'Forgets the one saved in this app. A key from the environment or a file stays.'
}: {
  label: string
  source: string
  placeholder: string
  saving: boolean
  onSave: (v: string) => void
  onForget: () => void
  forgetTitle?: string
}) {
  const [value, setValue] = useState('')
  return (
    <Field label={label} hint={source ? `Comes from: ${source}` : 'Not set'}>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          if (!value.trim()) return
          onSave(value.trim())
          setValue('')
        }}
      >
        <Input type="password" autoComplete="off" spellCheck={false} aria-label={label} placeholder={placeholder} value={value} onChange={(e) => setValue(e.target.value)} />
        <Button type="submit" disabled={!value.trim() || saving}>
          Save
        </Button>
        <Button type="button" variant="ghost" disabled={saving} onClick={onForget} title={forgetTitle}>
          Forget saved
        </Button>
      </form>
    </Field>
  )
}

export function AccountCard({ state }: { state: RemoteState }) {
  const run = useRemote((s) => s.run)
  const refreshing = useRemote((s) => isBusy(s, 'remote:refresh'))
  const savingKey = useRemote((s) => isChannelBusy(s, 'remote:setVastKey'))
  const savingHf = useRemote((s) => isChannelBusy(s, 'remote:setHfToken'))
  const addingKey = useRemote((s) => isBusy(s, 'remote:addSshKey'))
  const [keyDialog, setKeyDialog] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const [tag, setTag] = useState(state.settings.engineTag)
  const [image, setImage] = useState(state.settings.image)
  useEffect(() => {
    setTag(state.settings.engineTag)
    setImage(state.settings.image)
  }, [state.settings.engineTag, state.settings.image])
  const ssh = state.ssh
  const dirty = tag !== state.settings.engineTag || image !== state.settings.image

  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">Account</h2>
          <p className="mt-0.5 text-xs text-muted">Your vast.ai account pays for rented instances. Nothing is rented, started or destroyed without a click and a price on screen.</p>
        </div>
        <Button size="sm" disabled={refreshing} onClick={() => void run('remote:refresh')}>
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : undefined} /> Refresh
        </Button>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-x-8 gap-y-4 lg:grid-cols-2">
        <div className="space-y-4">
          <SecretField
            label="vast.ai API key"
            source={state.vastKeySource}
            placeholder={state.vastKeySource ? 'Paste a new key to replace it' : 'Paste your vast.ai API key'}
            saving={savingKey}
            onSave={(v) => void run('remote:setVastKey', v)}
            onForget={() => void run('remote:setVastKey', '')}
          />
          <div className="flex items-center justify-between gap-3 text-[13px]">
            <span className="text-muted">Credit</span>
            <span className="font-medium tabular-nums">{state.credit === null ? 'Not read yet' : money(state.credit)}</span>
          </div>
        </div>
        <div className="space-y-4">
          <SecretField
            label="Hugging Face token"
            source={state.hfTokenSource}
            placeholder={state.hfTokenSource ? 'Paste a new token to replace it' : 'Only needed for gated or private repos'}
            saving={savingHf}
            onSave={(v) => void run('remote:setHfToken', v)}
            onForget={() => void run('remote:setHfToken', '')}
          />
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-3 text-[13px]">
              <span className="flex items-center gap-1.5 text-muted">
                <KeyRound size={14} /> ssh key
              </span>
              {!ssh.available ? (
                <Badge tone="danger">
                  <CircleAlert size={12} /> ssh not found
                </Badge>
              ) : !ssh.publicKeyPath ? (
                <Badge tone="warn">
                  <CircleAlert size={12} /> No ssh key on this PC
                </Badge>
              ) : ssh.onAccount === true ? (
                <Badge tone="ok">
                  <CircleCheck size={12} /> On your vast.ai account
                </Badge>
              ) : ssh.onAccount === false ? (
                <Badge tone="warn">
                  <CircleAlert size={12} /> Not on your vast.ai account
                </Badge>
              ) : (
                <Badge>Account not checked</Badge>
              )}
            </div>
            {ssh.publicKeyPath && <div className="truncate font-mono text-[11px] text-faint">{ssh.publicKeyPath}</div>}
            {ssh.error && <div className="text-xs text-danger-text">{ssh.error}</div>}
            {!ssh.available && <div className="text-xs text-muted">Install the Windows OpenSSH client; the app uses it to reach instances.</div>}
            {ssh.available && ssh.publicKeyPath && ssh.onAccount !== true && (
              <Button size="sm" disabled={addingKey} onClick={() => setKeyDialog(true)}>
                <KeyRound size={14} /> Add ssh key
              </Button>
            )}
          </div>
        </div>
      </div>

      {state.error && (
        <div className="mt-4 flex items-start gap-2 rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger-text">
          <CircleAlert size={14} className="mt-px shrink-0" />
          <div>
            <span className="font-medium">Last vast.ai error:</span> {state.error}
          </div>
        </div>
      )}

      <Collapsible.Root open={advanced} onOpenChange={setAdvanced} className="mt-4">
        <Collapsible.Trigger className="flex items-center gap-1 rounded text-xs font-medium text-muted hover:text-text focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none">
          <ChevronRight size={14} className={advanced ? 'rotate-90 transition-transform' : 'transition-transform'} />
          Advanced
        </Collapsible.Trigger>
        <Collapsible.Content className="mt-3 space-y-3">
          <Field label="llama.cpp build for instances" hint={`Empty = the same build as this PC's llama-server. Now: ${state.resolvedEngineTag || 'unknown'}`}>
            <Input className="font-mono text-xs" placeholder={state.resolvedEngineTag || 'b12407'} value={tag} onChange={(e) => setTag(e.target.value)} />
          </Field>
          <Field label="Image" hint="The Docker image a new rental starts from">
            <Input className="font-mono text-xs" value={image} onChange={(e) => setImage(e.target.value)} />
          </Field>
          <div className="flex justify-end">
            <Button size="sm" disabled={!dirty} onClick={() => void run('remote:saveSettings', { engineTag: tag.trim(), image: image.trim() })}>
              Save
            </Button>
          </div>
        </Collapsible.Content>
      </Collapsible.Root>

      <ConfirmDialog
        open={keyDialog}
        title="Add this PC's ssh key to your vast.ai account?"
        lines={[
          `The public key at ${ssh.publicKeyPath ?? 'your .ssh folder'} is added to your vast.ai account. Instances you rent later accept it, so this PC can log in to them.`,
          'It adds a public key only; nothing is rented and nothing is charged.'
        ]}
        confirmLabel="Add the key"
        busy={addingKey}
        onCancel={() => setKeyDialog(false)}
        onConfirm={() => {
          setKeyDialog(false)
          void run('remote:addSshKey')
        }}
      />
    </section>
  )
}
