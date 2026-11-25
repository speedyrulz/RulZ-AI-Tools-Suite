import { Check } from 'lucide-react'
import { APPROVAL_LEVELS } from '@shared/defaults'
import { CLOUD_NEVER_AUTO, CLOUD_NOTE, cloudText, RENTAL_NOTE } from '@shared/modes'
import type { ApprovalLevel, ModelInfo } from '@shared/types'
import { cn, formatTokens } from '../lib/format'
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

export function ModelSelect({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (id: string) => void }) {
  const sorted = [...models].sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.id.localeCompare(b.id))
  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)} className="h-10 text-sm">
      {!models.some((m) => m.id === value) && <option value={value}>{value ? modelLabel(value) : 'Choose a model'}</option>}
      {sorted.map((m) => (
        <option key={m.id} value={m.id}>
          {modelLabel(m.id)}
          {m.cloud ? `  (${cloudText(m)})` : m.rental
            ? `  (${rentalText(m)}${m.loadedContext ? `, ${formatTokens(m.loadedContext)} context` : ''})`
            : m.loaded
              ? `  (ready${m.loadedContext ? `, ${formatTokens(m.loadedContext)} context` : ''})`
              : ''}
        </option>
      ))}
    </Select>
  )
}

/** The warning under a rental's or an OpenRouter model: what leaves this computer. */
export function RentalNote({ model }: { model: Pick<ModelInfo, 'rental' | 'cloud'> | undefined }) {
  if (model?.cloud) {
    return (
      <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 text-[13px]">
        <div className="mb-1 flex items-center gap-2">
          <Badge tone="warn">{cloudText(model)}</Badge>
        </div>
        {CLOUD_NOTE} {CLOUD_NEVER_AUTO}
      </div>
    )
  }
  if (!model?.rental) return null
  return (
    <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 text-[13px]">
      <div className="mb-1 flex items-center gap-2">
        <Badge tone="warn">{rentalText(model)}</Badge>
      </div>
      {RENTAL_NOTE} It is never picked for you: only when you choose it.
    </div>
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
            'flex items-start gap-3 rounded-lg border px-3.5 py-3 text-left transition-colors',
            value === l.value ? 'border-accent bg-accent-soft/50' : 'border-border bg-panel-2 hover:border-accent/50'
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
            <span className="block text-sm font-medium">
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
