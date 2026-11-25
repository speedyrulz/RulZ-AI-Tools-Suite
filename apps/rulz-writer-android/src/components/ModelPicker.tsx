import { Cloud } from 'lucide-react'
import type { PhoneModel } from '@shared/phone'
import { canPickModel, rentalLabel } from '../lib/mode'
import { Badge, Select } from './ui'

export function ModelPicker({ models, value, onChange, placeholder }: { models: PhoneModel[]; value: string | null; onChange: (id: string | null) => void; placeholder: string }) {
  const selected = models.find((m) => m.id === value)
  return <div className="space-y-2">
    <Select aria-label="Model" value={value ?? ''} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">{placeholder}</option>
      {value && !selected && <option value={value}>{value}</option>}
      {models.map((m) => <option key={m.id} value={m.id} disabled={!canPickModel(m, value)}>{m.name}{m.rental ? ` (Rental: ${rentalLabel(m)}${m.rental.reconnecting ? ', Reconnecting' : ''})` : m.loaded ? ' (ready)' : ''}</option>)}
    </Select>
    {models.filter((m) => m.rental).map((m) => <div key={m.id} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] text-muted"><span className="min-w-0 break-words">{m.name}</span><Badge><Cloud size={12} /> Rental</Badge><span>{rentalLabel(m)}{m.rental?.reconnecting && ' · Reconnecting'}</span>{m.rental?.label === null && <span>Load it in LLM Suite on the computer.</span>}</div>)}
    {selected?.rental && <p className="text-[12px] text-muted">Your text is sent to the rented computer.</p>}
  </div>
}
