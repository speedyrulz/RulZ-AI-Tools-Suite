import { ArrowDown, ArrowUp, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { MAX_LORA_MERGE_INPUTS } from '@shared/loraMerge'
import { LORA_WEIGHT_LIMIT, addRow, moveRow, removeRow, setRow, type LoraRow } from '../../lib/loraMerge'
import { useLoraMerge } from '../../stores/loraMerge'
import { Button, Card, IconButton, NumberInput } from '../ui'
import { ModelSelect } from '../merge/controls'

function Weight({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
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
        min={-LORA_WEIGHT_LIMIT}
        max={LORA_WEIGHT_LIMIT}
        step={0.05}
        className="h-8 w-[72px]"
      />
    </div>
  )
}

function Row({ row, index, count, list }: { row: LoraRow; index: number; count: number; list: string[] }) {
  const patchRows = useLoraMerge((s) => s.patchRows)
  const n = index + 1
  return (
    <li className="rounded-lg border border-border bg-panel-2 p-3">
      <div className="flex items-center gap-2">
        <span className="w-5 shrink-0 text-center text-xs text-faint tabular-nums">{n}</span>
        <div className="min-w-0 flex-1">
          <ModelSelect
            label={`LoRA ${n}`}
            value={row.name}
            options={list}
            placeholder="Choose a LoRA"
            onChange={(name) => patchRows((r) => setRow(r, index, { name }))}
          />
        </div>
        <IconButton label={`Move LoRA ${n} up`} disabled={index === 0} onClick={() => patchRows((r) => moveRow(r, index, -1))}>
          <ArrowUp size={16} />
        </IconButton>
        <IconButton label={`Move LoRA ${n} down`} disabled={index === count - 1} onClick={() => patchRows((r) => moveRow(r, index, 1))}>
          <ArrowDown size={16} />
        </IconButton>
        <IconButton label={`Remove LoRA ${n}`} onClick={() => patchRows((r) => removeRow(r, index))}>
          <Trash2 size={16} />
        </IconButton>
      </div>
      <div className="mt-3 pl-7 lg:max-w-md">
        <div className="mb-1 text-xs text-muted">Weight</div>
        <Weight label={`Weight of LoRA ${n}`} value={row.weight} onChange={(weight) => patchRows((r) => setRow(r, index, { weight }))} />
      </div>
    </li>
  )
}

export function InputsCard() {
  const rows = useLoraMerge((s) => s.rows)
  const list = useLoraMerge((s) => s.list)
  const listLoading = useLoraMerge((s) => s.listLoading)
  const listError = useLoraMerge((s) => s.listError)
  const loadList = useLoraMerge((s) => s.loadList)
  const patchRows = useLoraMerge((s) => s.patchRows)
  const full = rows.length >= MAX_LORA_MERGE_INPUTS
  return (
    <Card
      title="LoRAs to merge"
      description="Each LoRA's change is multiplied by its weight and the results are added up."
      actions={
        <div className="flex items-center gap-2">
          <IconButton label="Reload the LoRA list" disabled={listLoading} onClick={() => void loadList()}>
            <RefreshCw size={16} className={listLoading ? 'animate-spin' : undefined} />
          </IconButton>
          <Button size="sm" disabled={full} onClick={() => patchRows(addRow)}>
            <Plus size={14} />
            Add LoRA
          </Button>
        </div>
      }
    >
      {listError && <p className="mb-3 text-[13px] text-danger-text">The LoRA list could not be read: {listError}</p>}
      {!listError && !listLoading && list.length === 0 && <p className="mb-3 text-[13px] text-muted">No LoRA files were found.</p>}
      <ul className="space-y-3">
        {rows.map((r, i) => (
          <Row key={i} row={r} index={i} count={rows.length} list={list} />
        ))}
      </ul>
      {full && <p className="mt-2 text-xs text-faint">At most {MAX_LORA_MERGE_INPUTS} LoRAs.</p>}
    </Card>
  )
}
