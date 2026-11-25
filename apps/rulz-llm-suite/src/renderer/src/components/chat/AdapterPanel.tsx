import { RotateCcw } from 'lucide-react'
import { MAX_ADAPTER_SCALE, MIN_ADAPTER_SCALE } from '@shared/adapters'
import { isCloudModelId } from '@shared/cloud'
import type { LoadAdapter } from '@shared/types'
import { chatAdapters, formatScale, onScale, withChatScale } from '../../lib/adapters'
import { useChat } from '../../stores/chat'
import { useModels } from '../../stores/models'
import { Button, Field, NumberInput, Slider, Switch } from '../ui'

/** Per-chat strengths of the LoRA adapters the running model loaded; changes need no reload. */
export function AdapterPanel() {
  const chatModelId = useChat((s) => s.current?.modelId ?? null)
  const own = useChat((s) => s.current?.adapters ?? null)
  const mutate = useChat((s) => s.mutateCurrent)
  const instances = useModels((s) => s.instances)
  const activeModelId = useModels((s) => s.activeModelId)
  // Messages go to the active model (see the chat store), so its instance is the one that answers.
  const modelId = activeModelId ?? chatModelId
  const inst = instances.find((i) => i.modelId === modelId && i.status === 'ready')
  const loaded = inst?.adapters ?? []

  if (inst?.cloud || (modelId && isCloudModelId(modelId))) return <p className="text-[11px] text-faint">Models at OpenRouter have no adapters.</p>
  if (loaded.length === 0) {
    return (
      <p className="text-[11px] text-faint">
        {inst ? 'This model was loaded without adapters.' : 'The model is not loaded.'} Adapters are chosen when you load the model (Ctrl+L).
      </p>
    )
  }

  const rows = chatAdapters(loaded, own)
  const set = (id: string, scale: number): void => {
    mutate((c) => ({ ...c, adapters: withChatScale(loaded, c.adapters, id, scale) as LoadAdapter[] }))
  }
  return (
    <>
      {rows.map((a) => (
        <Field
          key={a.id}
          label={<span className="break-all">{a.name}</span>}
          hint={<span className="tabular-nums">Default {formatScale(a.defaultScale)}</span>}
        >
          <div className="flex items-center gap-2.5">
            <Switch
              checked={a.scale !== 0}
              onCheckedChange={(on) => set(a.id, on ? onScale(a) : 0)}
            />
            <Slider value={Math.min(2, Math.max(0, a.scale))} min={0} max={2} step={0.05} onChange={(v) => set(a.id, v)} />
            <NumberInput
              className="w-18"
              value={a.scale}
              min={MIN_ADAPTER_SCALE}
              max={MAX_ADAPTER_SCALE}
              step={0.05}
              onChange={(v) => set(a.id, v ?? a.defaultScale)}
            />
          </div>
        </Field>
      ))}
      <div className="flex items-start justify-between gap-2">
        <p className="text-[11px] text-faint">Changing strengths makes the model process the chat again from the start.</p>
        <Button size="sm" variant="ghost" disabled={own === null} onClick={() => mutate((c) => ({ ...c, adapters: null }))}>
          <RotateCcw size={14} /> Use model defaults
        </Button>
      </div>
    </>
  )
}
