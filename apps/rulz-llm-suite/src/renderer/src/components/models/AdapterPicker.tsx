import * as Collapsible from '@radix-ui/react-collapsible'
import { AlertTriangle, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { adapterBaseMatch, adapterChoices, adapterPathUsable, clampScale, MAX_ADAPTER_SCALE, MIN_ADAPTER_SCALE } from '@shared/adapters'
import type { AdapterEntry, LoadAdapter, ModelEntry } from '@shared/types'
import { setLoadAdapterScale, toggleLoadAdapter } from '../../lib/adapters'
import { cn, formatBytes } from '../../lib/format'
import { useModels } from '../../stores/models'
import { NumberInput } from '../ui'

/** "Adapters" section of the load settings: tick the LoRA adapters to load and set a default strength for each. */
export function AdapterPicker({
  model,
  adapters,
  onChange
}: {
  model: ModelEntry
  adapters: LoadAdapter[]
  onChange: (adapters: LoadAdapter[]) => void
}) {
  const all = useModels((s) => s.adapters)
  const choices = adapterChoices(model, all)
  const [open, setOpen] = useState(adapters.length > 0)
  const ticked = adapters.filter((a) => choices.some((c) => c.id === a.id)).length
  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <Collapsible.Trigger className="flex items-center gap-1 text-xs font-medium text-muted hover:text-text">
        <ChevronRight size={14} className={open ? 'rotate-90 transition-transform' : 'transition-transform'} />
        Adapters{ticked > 0 && <span className="text-faint"> · {ticked} chosen</span>}
      </Collapsible.Trigger>
      <Collapsible.Content className="mt-3 space-y-1">
        {choices.length === 0 ? (
          <p className="text-[11px] text-faint">No adapters for this architecture in the models folder.</p>
        ) : (
          <>
            <p className="pb-1 text-[11px] text-faint">
              LoRA adapters are loaded with the model. The strength is the default for chats and the API server; each chat can change it
              while the model stays loaded.
            </p>
            {choices.map((a) => (
              <AdapterRow key={a.id} adapter={a} model={model} entry={adapters.find((x) => x.id === a.id)} onChange={onChange} list={adapters} />
            ))}
          </>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

function AdapterRow({
  adapter,
  model,
  entry,
  list,
  onChange
}: {
  adapter: AdapterEntry
  model: ModelEntry
  entry: LoadAdapter | undefined
  list: LoadAdapter[]
  onChange: (adapters: LoadAdapter[]) => void
}) {
  const usable = adapterPathUsable(adapter.path)
  const mismatch = adapterBaseMatch(adapter, model) === 'mismatch'
  const checked = !!entry
  return (
    <div className={cn('space-y-0.5 py-1', !usable && 'opacity-60')}>
      <div className="flex items-center gap-2">
        <label className="flex min-w-0 flex-1 items-center gap-2 text-[13px]">
          <input
            type="checkbox"
            className="shrink-0"
            checked={checked}
            disabled={!usable}
            onChange={(e) => onChange(toggleLoadAdapter(list, adapter.id, e.target.checked))}
          />
          <span className="min-w-0 truncate" title={adapter.id}>
            {adapter.name}
          </span>
          <span className="shrink-0 text-[11px] text-faint tabular-nums">{formatBytes(adapter.sizeBytes)}</span>
        </label>
        {checked && (
          <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-faint">
            Strength
            <NumberInput
              className="w-16"
              value={entry.scale}
              min={MIN_ADAPTER_SCALE}
              max={MAX_ADAPTER_SCALE}
              step={0.05}
              onChange={(v) => onChange(setLoadAdapterScale(list, adapter.id, clampScale(v, 1)))}
            />
          </label>
        )}
      </div>
      {!usable && <div className="pl-6 text-[11px] text-faint">Path has a comma; rename the file or folder</div>}
      {mismatch && (
        <div className="flex items-center gap-1 pl-6 text-[11px] text-warn">
          <AlertTriangle size={12} className="shrink-0" />
          Trained on {adapter.info?.baseModelName}
        </div>
      )}
    </div>
  )
}
