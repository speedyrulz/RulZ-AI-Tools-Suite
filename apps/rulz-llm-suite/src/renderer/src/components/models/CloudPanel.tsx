import { ShieldAlert } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { APP_MODES, MODE_LABELS, modeDefaultsEmpty, type AppMode } from '@shared/modes'
import type { ModelEntry, SamplingParams } from '@shared/types'
import { errorMessage, invoke } from '../../lib/api'
import { GOES_TO_OPENROUTER, badgesFor, priceLine, supportsOf } from '../../lib/cloud'
import { formatTokens } from '../../lib/format'
import { cloudPickerMode, modesFor, samplingSave } from '../../lib/modes'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { useCloud } from '../../stores/cloud'
import { modeDefaultsOf, useModels } from '../../stores/models'
import { Badge, Button, Field, Select } from '../ui'
import { OwnSampling } from './OwnSampling'

/**
 * The picker's right side for an OpenRouter model: what it costs, where the text goes, one mode's own sampling, and one
 * button that connects. There is no load form, memory estimate or reload: nothing is loaded on this PC.
 */
export function CloudPanel({ model, onDone }: { model: ModelEntry; onDone: () => void }) {
  const instances = useModels((s) => s.instances)
  const ensure = useModels((s) => s.ensure)
  const unload = useModels((s) => s.unload)
  const keySaved = useCloud((s) => s.state?.keySaved ?? false)
  const cloud = model.cloud!
  const info = cloud.info
  const inst = instances.find((i) => i.modelId === model.id && i.status !== 'stopping')
  const appSampling = useApp((s) => s.settings?.defaultSampling)
  const allModes = useModels((s) => s.modeDefaults)
  const modes = modesFor(allModes, model.id)
  const [mode, setMode] = useState<AppMode>(() => cloudPickerMode(useApp.getState().page, !!useChat.getState().current?.re))
  // The mode's own sampling as edited here; null = not edited, so it shows (and keeps) what is saved.
  const [edit, setEdit] = useState<{ on: boolean; sampling: SamplingParams | null } | null>(null)

  const connect = async (): Promise<void> => {
    // Not being able to save the sampling is no reason not to connect.
    if (edit) {
      await invoke('models:setModeDefaults', model.id, mode, samplingSave(modeDefaultsOf(model.id)?.[mode], edit)).catch((e) =>
        toast.error(`Could not save the ${MODE_LABELS[mode]} settings: ${errorMessage(e)}`)
      )
    }
    onDone()
    // Connecting for a mode from that mode's own page stays there.
    if (mode === 'chat' || useApp.getState().page !== mode) useApp.getState().setPage('chat')
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
        <Field label="Settings for">
          <Select
            value={mode}
            onChange={(e) => {
              setMode(e.target.value as AppMode)
              setEdit(null)
            }}
            aria-label="Settings for"
          >
            {APP_MODES.map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m]}
                {modeDefaultsEmpty(modes?.[m]) ? '' : ' · own settings'}
              </option>
            ))}
          </Select>
        </Field>
        <OwnSampling
          key={mode}
          mode={mode}
          on={edit ? edit.on : !!modes?.[mode]?.sampling}
          sampling={edit ? edit.sampling : (modes?.[mode]?.sampling ?? null)}
          appSampling={appSampling}
          leftover={null}
          note={`Saved for this model when you press ${inst ? 'Use this model' : 'Connect'}.${mode === 'chat' || mode === 're' ? ' Chats that already have messages keep their own.' : ''}`}
          onChange={setEdit}
          supports={supportsOf(model)}
        />
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
