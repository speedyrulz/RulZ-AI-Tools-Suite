import type { AdventureState } from '@shared/adventure'
import { cn } from '../../lib/format'
import { stateIsEmpty, statText } from '../../lib/adventure'

/** A row of buttons where one is chosen (passage length, number of choices, tabs). */
export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
  disabled
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  label: string
  disabled?: boolean
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-border bg-panel p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={cn(
            'h-7 rounded-md px-3 text-[13px] transition-colors focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none disabled:opacity-45',
            o.value === value ? 'bg-hover font-medium text-text' : 'text-muted hover:text-text'
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** The tracked state of a passage, read only. */
export function StateView({ state }: { state: AdventureState }) {
  if (stateIsEmpty(state)) return <p className="text-[13px] text-faint">Nothing is tracked yet.</p>
  const stats = Object.entries(state.stats)
  return (
    <div className="space-y-4 text-[13px]">
      {state.inventory.length > 0 && (
        <div>
          <div className="mb-1 text-[11px] font-semibold tracking-wide text-faint uppercase">Inventory</div>
          <ul className="space-y-0.5 text-body">
            {state.inventory.map((i) => (
              <li key={i}>{i}</li>
            ))}
          </ul>
        </div>
      )}
      {stats.length > 0 && (
        <div>
          <div className="mb-1 text-[11px] font-semibold tracking-wide text-faint uppercase">Stats</div>
          <dl className="space-y-0.5">
            {stats.map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3">
                <dt className="text-muted">{k}</dt>
                <dd className="text-body tabular-nums">{statText(v)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
      {state.flags.length > 0 && (
        <div>
          <div className="mb-1 text-[11px] font-semibold tracking-wide text-faint uppercase">Flags</div>
          <ul className="space-y-0.5 text-body">
            {state.flags.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export const FOCUS = 'focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none'
