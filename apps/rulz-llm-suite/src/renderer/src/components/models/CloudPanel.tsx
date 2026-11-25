import { ShieldAlert } from 'lucide-react'
import type { ModelEntry } from '@shared/types'
import { GOES_TO_OPENROUTER, badgesFor, priceLine } from '../../lib/cloud'
import { formatTokens } from '../../lib/format'
import { pickerScope } from '../../lib/modes'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { useCloud } from '../../stores/cloud'
import { useModels } from '../../stores/models'
import { Badge, Button } from '../ui'

/**
 * The picker's right side for an OpenRouter model: what it costs and where the text goes, and one button that connects.
 * There is no load form, memory estimate or reload: nothing is loaded on this PC.
 */
export function CloudPanel({ model, onDone }: { model: ModelEntry; onDone: () => void }) {
  const instances = useModels((s) => s.instances)
  const ensure = useModels((s) => s.ensure)
  const unload = useModels((s) => s.unload)
  const keySaved = useCloud((s) => s.state?.keySaved ?? false)
  const cloud = model.cloud!
  const info = cloud.info
  const inst = instances.find((i) => i.modelId === model.id && i.status !== 'stopping')

  const connect = async (): Promise<void> => {
    onDone()
    // Connecting for a mode from that mode's own page stays there.
    const scope = pickerScope(useApp.getState().page, !!useChat.getState().current?.re)
    if (!scope || scope === 'chat' || useApp.getState().page !== scope) useApp.getState().setPage('chat')
    await ensure(model.id)
  }

  return (
    <>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone="accent">OpenRouter</Badge>
          {badgesFor(info).map((b) => (
            <Badge key={b.label} tone={b.tone}>
              {b.label}
            </Badge>
          ))}
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-1.5 text-[13px] tabular-nums">
          <dt className="text-muted">Price</dt>
          <dd>{priceLine(info)}</dd>
          <dt className="text-muted">Context the app uses</dt>
          <dd>
            {formatTokens(cloud.contextLimit)} of {formatTokens(info.contextLength)}
          </dd>
          <dt className="text-muted">Longest reply</dt>
          <dd>{formatTokens(cloud.replyLimit)}</dd>
        </dl>
        {info.description && <p className="line-clamp-4 text-xs text-muted">{info.description}</p>}
        <div className="flex items-start gap-2 rounded-md border border-border bg-panel-2 px-3 py-2 text-xs text-muted">
          <ShieldAlert size={14} className="mt-px shrink-0" />
          <div>
            {GOES_TO_OPENROUTER} {info.free ? 'This model is free.' : 'Every message costs money; the daily limit on the OpenRouter page stops spending.'}
          </div>
        </div>
        {!keySaved && <p className="text-xs text-warn">No OpenRouter key is saved yet. Save one on the OpenRouter page first.</p>}
        <Button size="sm" onClick={() => { onDone(); useApp.getState().setPage('openrouter') }}>
          Change limits on the OpenRouter page
        </Button>
      </div>
      <div className="flex items-center gap-3 border-t border-border px-5 py-3">
        <div className="flex-1" />
        {inst && (
          <Button variant="ghost" onClick={() => void unload(inst.instanceId)}>
            Disconnect
          </Button>
        )}
        <Button id="picker-load-button" variant="primary" disabled={!keySaved && !inst} onClick={() => void connect()}>
          {inst ? 'Use this model' : 'Connect'}
        </Button>
      </div>
    </>
  )
}
