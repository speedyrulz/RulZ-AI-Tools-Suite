import { Check } from 'lucide-react'
import { APPROVAL_LEVELS } from '@shared/defaults'
import type { ApprovalLevel, ModelInfo } from '@shared/types'
import { cn, formatTokens } from '../lib/format'
import { Select } from './ui'

/** Readable model name: no folders, no ".gguf". */
export function modelLabel(id: string): string {
  return (id.split(/[\\/]/).pop() ?? id).replace(/\.gguf$/i, '')
}

export function ModelSelect({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (id: string) => void }) {
  const sorted = [...models].sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.id.localeCompare(b.id))
  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)}>
      {!models.some((m) => m.id === value) && <option value={value}>{value ? modelLabel(value) : 'Choose a model'}</option>}
      {sorted.map((m) => (
        <option key={m.id} value={m.id}>
          {modelLabel(m.id)}
          {m.loaded ? `  (ready${m.loadedContext ? `, ${formatTokens(m.loadedContext)} context` : ''})` : ''}
        </option>
      ))}
    </Select>
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
