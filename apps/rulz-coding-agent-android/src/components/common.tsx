import { Check, Server } from 'lucide-react'
import { APPROVAL_LEVELS } from '@shared/defaults'
import { RENTAL_NOTE } from '@shared/modes'
import type { ApprovalLevel, ModelInfo } from '@shared/types'
import { cn, formatTokens } from '../lib/format'
import { chosenRentalGone } from '../lib/modeLogic'
import { useApp } from '../stores/app'
import { Badge, Select } from './ui'

/** Readable model name: no folders, no ".gguf". */
export function modelLabel(id: string): string {
  return (id.split(/[\\/]/).pop() ?? id).replace(/\.gguf$/i, '')
}

/** "on a rental", with its name and whether it is reconnecting, for a model that runs on a rented computer. */
export function rentalText(m: Pick<ModelInfo, 'rental'>): string {
  if (!m.rental) return ''
  return `on a rental${m.rental.label ? `: ${m.rental.label}` : ''}${m.rental.reconnecting ? ', reconnecting' : ''}`
}

/** The neutral badge for a model on a rental: an icon and the words, never colour alone. */
export function RentalBadge({ model }: { model: Pick<ModelInfo, 'rental'> }) {
  if (!model.rental) return null
  const text = rentalText(model)
  return (
    <Badge className="max-w-full min-w-0">
      <Server size={12} className="shrink-0" aria-hidden />
      <span className="truncate">{text[0].toUpperCase() + text.slice(1)}</span>
    </Badge>
  )
}

export function ModelSelect({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (id: string) => void }) {
  const sorted = [...models].sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.id.localeCompare(b.id))
  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)}>
      {!models.some((m) => m.id === value) && <option value={value}>{value ? modelLabel(value) : 'Choose a model'}</option>}
      {sorted.map((m) => (
        <option key={m.id} value={m.id}>
          {modelLabel(m.id)}
          {m.rental
            ? `  (${rentalText(m)}${m.loadedContext ? `, ${formatTokens(m.loadedContext)} context` : ''})`
            : m.loaded
              ? `  (ready${m.loadedContext ? `, ${formatTokens(m.loadedContext)} context` : ''})`
              : ''}
        </option>
      ))}
    </Select>
  )
}

/** What leaves the computer when the model runs on a rental, and that it is never picked for you. */
export function RentalNote({ model }: { model: Pick<ModelInfo, 'rental'> | undefined }) {
  if (!model?.rental) return null
  return (
    <div className="space-y-1.5 rounded-xl border border-border bg-panel-2 p-3 text-[13px] text-muted">
      <RentalBadge model={model} />
      <p>{RENTAL_NOTE} It is never picked for you: only when you choose it.</p>
    </div>
  )
}

/** The chosen model was on a rental and is gone: say so. Nothing starts and no other model is used. */
export function RentalGoneNote({ className }: { className?: string }) {
  const gone = useApp((s) => chosenRentalGone(s.settings, s.connection))
  if (!gone) return null
  return (
    <p role="status" className={cn('rounded-xl border border-border bg-panel-2 p-3 text-[13px]', className)}>
      {gone}
    </p>
  )
}

export function ApprovalChooser({ value, onChange }: { value: ApprovalLevel; onChange: (v: ApprovalLevel) => void }) {
  return (
    <div className="grid gap-2">
      {APPROVAL_LEVELS.map((l) => (
        <button
          key={l.value}
          onClick={() => onChange(l.value)}
          className={cn(
            'flex items-start gap-3 rounded-xl border px-3.5 py-3 text-left transition-colors',
            value === l.value ? 'border-accent bg-accent-soft/50' : 'border-border bg-panel-2 active:border-accent/50'
          )}
        >
          <span
            className={cn(
              'mt-0.5 flex size-4.5 shrink-0 items-center justify-center rounded-full border',
              value === l.value ? 'border-accent bg-accent text-accent-fg' : 'border-border'
            )}
          >
            {value === l.value && <Check size={12} strokeWidth={3} />}
          </span>
          <span>
            <span className="block text-[15px] font-medium">
              {l.label}
              {l.value === 'balanced' && <span className="ml-1.5 text-xs font-normal text-faint">(recommended)</span>}
            </span>
            <span className="block text-[13px] text-muted">{l.hint}</span>
          </span>
        </button>
      ))}
    </div>
  )
}
