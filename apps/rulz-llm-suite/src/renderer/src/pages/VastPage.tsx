import { CircleAlert, Cloud, Link2, Plus } from 'lucide-react'
import { useState } from 'react'
import type { RentIntent } from '@shared/remote'
import { AccountCard } from '../components/remote/AccountCard'
import { AttachDialog } from '../components/remote/AttachDialog'
import { ConfirmDialog } from '../components/remote/ConfirmDialog'
import { RemoteModels } from '../components/remote/RemoteModels'
import { RentDialog } from '../components/remote/RentDialog'
import { RentalCard } from '../components/remote/RentalCard'
import { useNow } from '../components/remote/useNow'
import { Button } from '../components/ui'
import { intentSentence, rentBlockedReason } from '../lib/remote'
import { isBusy, useRemote } from '../stores/remote'

export function VastPage() {
  const state = useRemote((s) => s.state)
  const loaded = useRemote((s) => s.loaded)
  const [renting, setRenting] = useState(false)
  const [attaching, setAttaching] = useState(false)

  if (!state) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted">
        {loaded ? 'The vast.ai state could not be read. Press Ctrl+9 again later, or restart the app.' : 'Reading the vast.ai state…'}
      </div>
    )
  }
  const blocked = rentBlockedReason(state)
  const shown = state.rentals

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-5 px-6 py-8">
        <div className="flex flex-wrap items-start gap-3">
          <div className="min-w-0 flex-1">
            <h1 className="flex items-center gap-2 font-[family-name:var(--font-display)] text-lg font-semibold">
              <Cloud size={18} /> vast.ai
            </h1>
            <p className="mt-0.5 text-xs text-muted">
              Run a model on a rented GPU while chats, REA and the scoreboard stay on this PC. Renting, starting and destroying only happen when you
              click, with the price or the loss shown first. A rental can stop itself when it is idle.
            </p>
          </div>
          <Button size="sm" onClick={() => setAttaching(true)} disabled={!state.vastKeySource}>
            <Link2 size={14} /> Attach an instance
          </Button>
          <Button variant="primary" onClick={() => setRenting(true)} disabled={!!blocked}>
            <Plus size={14} /> Rent an instance
          </Button>
        </div>
        {blocked && <p className="-mt-3 text-right text-xs text-warn">{blocked}</p>}

        {state.intents.map((i) => (
          <IntentCard key={i.label} intent={i} />
        ))}

        <AccountCard state={state} />

        <section className="space-y-3" aria-label="Rentals">
          <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">
            Rentals <span className="font-normal text-faint tabular-nums">· {shown.length}</span>
          </h2>
          {shown.length === 0 ? (
            <div className="rounded-xl border border-border bg-panel p-5 text-sm text-muted">
              {state.vastKeySource
                ? 'No rented instances. Press Rent an instance to find one, or attach one you rented on vast.ai.'
                : 'Add your vast.ai API key above to rent instances or attach ones you already have.'}
            </div>
          ) : (
            shown.map((r) => <RentalCard key={r.id} rental={r} models={state.models} />)
          )}
        </section>

        <RemoteModels state={state} />
      </div>
      {renting && <RentDialog state={state} onClose={() => setRenting(false)} />}
      {attaching && <AttachDialog state={state} onClose={() => setAttaching(false)} />}
    </div>
  )
}

/** A rent request whose outcome is not known: it may have been rented. Renting stays blocked until it is settled. */
function IntentCard({ intent: i }: { intent: RentIntent }) {
  const run = useRemote((s) => s.run)
  const checking = useRemote((s) => isBusy(s, 'remote:checkIntent', i.label))
  const [forget, setForget] = useState(false)
  const now = useNow(5000)
  return (
    <div className="rounded-xl border border-warn/40 bg-warn/10 p-4" role="alert">
      <div className="flex items-start gap-2 text-[13px] text-warn">
        <CircleAlert size={15} className="mt-0.5 shrink-0" />
        <div className="min-w-0 flex-1 space-y-1">
          <div className="font-medium">A rent request may have gone through</div>
          <div className="text-xs">{intentSentence(i, now)}</div>
          {i.message && <div className="text-xs">{i.message}</div>}
          <div className="text-xs">Renting another instance is blocked until this is settled.</div>
        </div>
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <Button size="sm" disabled={checking} onClick={() => void run('remote:checkIntent', i.label)}>
          Check again
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setForget(true)}>
          I looked, forget it
        </Button>
      </div>
      <ConfirmDialog
        open={forget}
        title="Forget this rent request?"
        lines={[
          'Only do this after you looked at your instances on vast.ai and saw that nothing was rented, or that you do not want it.',
          'If an instance was rented, it keeps billing until you stop or destroy it there, and this app will not know about it. You can attach it later.'
        ]}
        confirmLabel="Forget it"
        onCancel={() => setForget(false)}
        onConfirm={() => {
          setForget(false)
          void run('remote:dismissIntent', i.label)
        }}
      />
    </div>
  )
}
