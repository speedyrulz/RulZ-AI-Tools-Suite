import { ArrowDown, ArrowUp, Plus, Trash2 } from 'lucide-react'
import { MAX_LORAS, type LoraEntry } from '@shared/merge'
import type { ServerCapabilities } from '@shared/types'
import { LORA_STRENGTH_LIMIT, addLora, moveLora, removeLora, setLora } from '../../lib/merge'
import { useMerge } from '../../stores/merge'
import { Button, Card, IconButton, NumberInput } from '../ui'
import { ModelSelect } from './controls'

function Strength({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        aria-label={label}
        min={-2}
        max={2}
        step={0.05}
        value={Math.min(2, Math.max(-2, value))}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-5 min-w-0 flex-1 cursor-pointer accent-accent"
      />
      <NumberInput
        value={Number(value.toFixed(2))}
        onCommit={onChange}
        min={-LORA_STRENGTH_LIMIT}
        max={LORA_STRENGTH_LIMIT}
        step={0.05}
        className="h-8 w-[72px]"
      />
    </div>
  )
}

function LoraRow({ lora, index, count, list, checkpoint }: { lora: LoraEntry; index: number; count: number; list: string[]; checkpoint: boolean }) {
  const patchLoras = useMerge((s) => s.patchLoras)
  const n = index + 1
  return (
    <li className="rounded-lg border border-border bg-panel-2 p-3">
      <div className="flex items-center gap-2">
        <span className="w-5 shrink-0 text-center text-xs text-faint tabular-nums">{n}</span>
        <div className="min-w-0 flex-1">
          <ModelSelect
            label={`LoRA ${n}`}
            value={lora.name}
            options={list}
            placeholder="Choose a LoRA"
            onChange={(name) => patchLoras((l) => setLora(l, index, { name }))}
          />
        </div>
        <IconButton label={`Move LoRA ${n} up`} disabled={index === 0} onClick={() => patchLoras((l) => moveLora(l, index, -1))}>
          <ArrowUp size={16} />
        </IconButton>
        <IconButton label={`Move LoRA ${n} down`} disabled={index === count - 1} onClick={() => patchLoras((l) => moveLora(l, index, 1))}>
          <ArrowDown size={16} />
        </IconButton>
        <IconButton label={`Remove LoRA ${n}`} onClick={() => patchLoras((l) => removeLora(l, index))}>
          <Trash2 size={16} />
        </IconButton>
      </div>
      <div className={checkpoint ? 'mt-3 grid gap-x-6 gap-y-2 pl-7 lg:grid-cols-2' : 'mt-3 pl-7 lg:max-w-md'}>
        <div>
          <div className="mb-1 text-xs text-muted">Strength</div>
          <Strength label={`Strength of LoRA ${n}`} value={lora.strength} onChange={(strength) => patchLoras((l) => setLora(l, index, { strength }))} />
        </div>
        {checkpoint && (
          <div>
            <div className="mb-1 text-xs text-muted">Text encoder</div>
            <Strength
              label={`Text encoder strength of LoRA ${n}`}
              value={lora.clipStrength}
              onChange={(clipStrength) => patchLoras((l) => setLora(l, index, { clipStrength }))}
            />
          </div>
        )}
      </div>
    </li>
  )
}

export function LorasCard({ caps }: { caps: ServerCapabilities }) {
  const recipe = useMerge((s) => s.recipe)
  const patchLoras = useMerge((s) => s.patchLoras)
  const bake = recipe.mode === 'bake'
  const full = recipe.loras.length >= MAX_LORAS
  return (
    <Card
      title="LoRAs"
      description={bake ? 'Baked into model A in this order.' : 'Applied to the merge result in this order.'}
      actions={
        <Button size="sm" disabled={full} onClick={() => patchLoras(addLora)}>
          <Plus size={14} />
          Add LoRA
        </Button>
      }
    >
      {recipe.loras.length === 0 ? (
        <p className="text-[13px] text-muted">{bake ? 'No LoRA yet. Add the ones to bake.' : 'No LoRA. Add one to bake it into the result.'}</p>
      ) : (
        <ul className="space-y-3">
          {recipe.loras.map((l, i) => (
            <LoraRow key={i} lora={l} index={i} count={recipe.loras.length} list={caps.loras} checkpoint={recipe.kind === 'checkpoint'} />
          ))}
        </ul>
      )}
      {full && <p className="mt-2 text-xs text-faint">At most {MAX_LORAS} LoRAs.</p>}
    </Card>
  )
}
