import { CircleAlert, CircleCheck } from 'lucide-react'
import { DEFAULT_DAILY_LIMIT_USD, MIN_DAILY_LIMIT_USD, formatUsd, type CloudState } from '@shared/cloud'
import { cleanLimit, dollars, modelSpend, spendView } from '../../lib/cloud'
import { cn } from '../../lib/format'
import { isChannelBusy, useCloud } from '../../stores/cloud'
import { Field, NumberInput, ProgressBar, Switch } from '../ui'

const LAST_DAYS = 7

export function SpendingCard({ state }: { state: CloudState }) {
  const run = useCloud((s) => s.run)
  const busy = useCloud((s) => isChannelBusy(s, 'cloud:setOptions'))
  const v = spendView(state)
  const limit = state.options.dailyLimitUsd
  const perModel = modelSpend(state)
  const days = state.spend.days.slice(0, LAST_DAYS)
  const Icon = v.tone === 'ok' ? CircleCheck : CircleAlert

  return (
    <section className="rounded-xl border border-border bg-panel p-5" aria-label="Spending">
      <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">Spending</h2>
      <p className="mt-0.5 text-xs text-muted">
        Before every request the app reserves the most it could cost and refuses it when that would pass the limit. The day is this PC's calendar day.
      </p>

      {state.spendError && (
        <div className="mt-3 flex items-start gap-2 rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger-text" role="alert">
          <CircleAlert size={14} className="mt-px shrink-0" />
          <div>
            <span className="font-medium">The spending record cannot be used:</span> {state.spendError} Requests to OpenRouter are refused until it can be read and written again.
          </div>
        </div>
      )}

      <div className="mt-4 space-y-2">
        <div className="flex items-baseline justify-between gap-3 text-[13px]">
          <span className="text-muted">Spent today</span>
          <span className="font-medium tabular-nums">
            {formatUsd(v.spent)}
            {limit !== null && <span className="font-normal text-muted"> of {dollars(limit)}</span>}
          </span>
        </div>
        {limit !== null && <ProgressBar value={v.share} tone={v.tone === 'ok' ? 'accent' : v.tone} />}
        <div className={cn('flex items-center gap-1.5 text-xs', v.tone === 'ok' ? 'text-muted' : v.tone === 'warn' ? 'text-warn' : 'text-danger-text')}>
          <Icon size={13} />
          {v.word}
        </div>
        <div className="flex items-center justify-between gap-3 text-xs text-muted">
          <span>Reserved for requests that are still running</span>
          <span className="tabular-nums">{formatUsd(v.reserved)}</span>
        </div>
      </div>

      <div className="mt-4 space-y-1 border-t border-border pt-3">
        <Field label="Daily limit" hint="Off means OpenRouter models can spend without a limit set here." inline>
          <Switch
            checked={limit !== null}
            disabled={busy}
            onCheckedChange={(on) => void run('cloud:setOptions', { dailyLimitUsd: on ? DEFAULT_DAILY_LIMIT_USD : null })}
          />
        </Field>
        {limit !== null && (
          <Field label="Dollars a day" hint={`At least ${dollars(MIN_DAILY_LIMIT_USD)}. Changing it takes effect for the next request.`} inline>
            <span className="flex items-center gap-1.5 text-[13px]">
              $
              <NumberInput
                className="w-24"
                value={limit}
                min={MIN_DAILY_LIMIT_USD}
                step={0.25}
                onChange={(n) => {
                  const next = cleanLimit(n)
                  if (next !== limit) void run('cloud:setOptions', { dailyLimitUsd: next })
                }}
              />
            </span>
          </Field>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-x-8 gap-y-4 border-t border-border pt-3 md:grid-cols-2">
        <div>
          <h3 className="text-xs font-medium text-muted">Today by model</h3>
          {perModel.length === 0 ? (
            <p className="mt-1 text-xs text-faint">Nothing spent today.</p>
          ) : (
            <ul className="mt-1 space-y-0.5 text-xs">
              {perModel.map((r) => (
                <li key={r.slug} className="flex items-center justify-between gap-3">
                  <span className="min-w-0 truncate font-mono text-[11.5px]">{r.slug}</span>
                  <span className="tabular-nums">{formatUsd(r.micro)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <h3 className="text-xs font-medium text-muted">Last days</h3>
          {days.length === 0 ? (
            <p className="mt-1 text-xs text-faint">No earlier days recorded.</p>
          ) : (
            <ul className="mt-1 space-y-0.5 text-xs">
              {days.map((d) => (
                <li key={d.day} className="flex items-center justify-between gap-3">
                  <span className="tabular-nums text-muted">{d.day}</span>
                  <span className="tabular-nums">{formatUsd(d.spent)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  )
}
