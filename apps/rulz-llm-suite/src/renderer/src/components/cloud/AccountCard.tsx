import { CircleAlert, CircleCheck, ShieldAlert } from 'lucide-react'
import type { CloudState } from '@shared/cloud'
import { GOES_TO_OPENROUTER, OPENROUTER_KEYS_URL, freeLeftText, keyLimitValue, dollars } from '../../lib/cloud'
import { isBusy, isChannelBusy, useCloud } from '../../stores/cloud'
import { SecretField } from '../remote/AccountCard'
import { Badge, Button } from '../ui'

export function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-link underline underline-offset-2 hover:text-text focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none">
      {children}
    </a>
  )
}

/** The OpenRouter key: typed here once, saved encrypted by the app, never shown again. */
export function CloudAccountCard({ state }: { state: CloudState }) {
  const run = useCloud((s) => s.run)
  const saving = useCloud((s) => isChannelBusy(s, 'cloud:setKey'))
  const checking = useCloud((s) => isBusy(s, 'cloud:checkKey'))
  const key = state.key

  return (
    <section className="rounded-xl border border-border bg-panel p-5" aria-label="Account">
      <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">Account</h2>
      <p className="mt-0.5 text-xs text-muted">
        Create a key at <ExternalLink href={OPENROUTER_KEYS_URL}>openrouter.ai/keys</ExternalLink>. Set a credit limit on it there as well.
      </p>

      <div className="mt-4 grid grid-cols-1 gap-x-8 gap-y-4 lg:grid-cols-2">
        <SecretField
          label="OpenRouter API key"
          source={state.keySaved ? 'saved in this app, encrypted' : ''}
          placeholder={state.keySaved ? 'Paste a new key to replace it' : 'Paste your OpenRouter API key'}
          saving={saving}
          onSave={(v) => void run('cloud:setKey', v)}
          onForget={() => void run('cloud:setKey', '')}
          forgetTitle="Forgets the saved key and disconnects the OpenRouter models. No other key is read from the environment or a file."
        />
        <div className="space-y-2 text-[13px]">
          {state.keySaved ? (
            <>
              <Row label="Key">
                {key ? (
                  <span className="flex items-center gap-1.5">
                    <Badge tone="ok">
                      <CircleCheck size={12} /> Accepted
                    </Badge>
                    <span className="truncate">{key.label || 'No label'}</span>
                  </span>
                ) : (
                  <Badge>Not checked</Badge>
                )}
              </Row>
              <Row label="Limit left on the key">{key ? keyLimitValue(key.limitRemaining) : '—'}</Row>
              <Row label="Used today on the key">{key ? dollars(key.usageToday) : '—'}</Row>
              <Row label="Free requests left today">{key ? freeLeftText(key.freeRequestsLeft) : '—'}</Row>
              <div className="flex justify-end">
                <Button size="sm" disabled={checking} onClick={() => void run('cloud:checkKey')}>
                  Check the key
                </Button>
              </div>
            </>
          ) : (
            <p className="text-xs text-muted">No key saved. You can browse OpenRouter's models and add them without one; to use one, save a key first.</p>
          )}
        </div>
      </div>

      {state.error && (
        <div className="mt-4 flex items-start gap-2 rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger-text" role="alert">
          <CircleAlert size={14} className="mt-px shrink-0" />
          <div>
            <span className="font-medium">Last OpenRouter error:</span> {state.error}
          </div>
        </div>
      )}

      <div className="mt-4 flex items-start gap-2 rounded-md border border-border bg-panel-2 px-3 py-2 text-xs text-muted">
        <ShieldAlert size={14} className="mt-px shrink-0" />
        <div>{GOES_TO_OPENROUTER} Models on this PC and on vast.ai do not send anything to OpenRouter.</div>
      </div>
    </section>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-muted">{label}</span>
      <span className="min-w-0 font-medium tabular-nums">{children}</span>
    </div>
  )
}
