import { Info } from 'lucide-react'
import { memo, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { MODE_LABELS, reloadFor, type AppMode } from '@shared/modes'
import type { LoadConfig } from '@shared/types'
import { isRunning } from '../../lib/adventure'
import { aiBusyReason } from '../../lib/aiBusy'
import { invoke } from '../../lib/api'
import { modesFor, reloadOffer } from '../../lib/modes'
import { WBENCH_BUSY, wbenchRunning } from '../../lib/wbenchFlag'
import { useAdventure } from '../../stores/adventure'
import { useBatch } from '../../stores/batch'
import { activeBenchRun, useBench } from '../../stores/bench'
import { isChatBusy, useChat } from '../../stores/chat'
import { modeDefaultsOf, readyInstance, useModels } from '../../stores/models'
import { useNovel } from '../../stores/novel'
import { loadRemoteModel } from '../../stores/remote'
import { useWbench } from '../../stores/wbench'
import { Banner } from '../writing/Banners'
import { Button, Tooltip } from '../ui'

/** Instance + mode pairs the user chose to keep as they are, until the model is loaded again. */
const kept = new Set<string>()

/** Why the model cannot be reloaded right now (something uses it), or null. */
export function reloadBlockReason(): string | null {
  if (wbenchRunning()) return WBENCH_BUSY
  const why = aiBusyReason()
  if (why) return why
  if (isRunning(useAdventure.getState().job)) return 'An adventure passage is being written.'
  return null
}

/**
 * Offers to reload a loaded model with the settings it has for a mode, when it runs with other ones. Never reloads by
 * itself. Several instances of one model (several rentals): the one readyInstance returns is judged.
 */
export const ModeReloadBar = memo(function ModeReloadBar({ modelId, mode }: { modelId: string | null; mode: AppMode }) {
  const inst = useModels((s) => (modelId ? s.instances.find((i) => i.modelId === modelId && i.status === 'ready') : undefined))
  const modes = useModels((s) => modesFor(s.modeDefaults, modelId))
  const name = useModels((s) => (modelId ? s.models.find((m) => m.id === modelId)?.name : undefined)) ?? modelId ?? ''
  const [base, setBase] = useState<{ modelId: string; config: LoadConfig } | null>(null)
  const [, setKeptTick] = useState(0)
  const seq = useRef(0)
  // These flip when a job starts or ends, not while it streams, so the bar does not re-render per token.
  useChat((s) => isChatBusy(s) || !!s.handoff)
  useBatch((s) => !!s.activeId || s.starting)
  useBench((s) => s.starting || !!activeBenchRun(s.runs))
  useNovel((s) => {
    const n = s as { gen?: unknown; checkRun?: unknown; interviewRun?: unknown }
    return !!(n.gen || n.checkRun || n.interviewRun)
  })
  useWbench((s) => s.running)
  useAdventure((s) => isRunning(s.job))

  const instKey = inst ? JSON.stringify(inst.config) : ''
  // The model's default load settings, read again when the model, its mode settings or its instance change.
  useEffect(() => {
    if (!modelId || !inst) return
    const mine = ++seq.current
    invoke('models:getDefaults', modelId)
      .then((d) => mine === seq.current && setBase({ modelId, config: d.config }))
      .catch(() => mine === seq.current && setBase(null))
    return () => {
      seq.current++
    }
  }, [modelId, modes, inst?.instanceId, instKey])

  if (!modelId || !inst) return null
  const key = `${inst.instanceId}|${mode}`
  const offer = reloadOffer(inst.config, base?.modelId === modelId ? base.config : null, modes, mode, kept.has(key))
  if (!offer) return null

  const own = !!modes?.[mode]?.load
  const label = own ? `${MODE_LABELS[mode]} settings` : 'default settings'
  const blocked = reloadBlockReason()

  const reload = async (): Promise<void> => {
    // Checked again: the model may have been busy or changed since the bar was drawn.
    const why = reloadBlockReason()
    if (why) return void toast(why)
    const cur = readyInstance(modelId)
    if (!cur || !base || base.modelId !== modelId) return
    const config = reloadFor(cur.config, base.config, modeDefaultsOf(modelId), mode)
    if (!config) return
    if (cur.host) await loadRemoteModel(cur.host.rentalId, modelId, config)
    else await useModels.getState().load(modelId, config)
  }

  return (
    <Banner
      tone="info"
      icon={<Info size={16} aria-hidden />}
      actions={
        <>
          <Tooltip content={blocked ?? `Reloads ${name} and applies the ${label}`}>
            <span>
              <Button size="sm" disabled={!!blocked} onClick={() => void reload()}>
                Reload with {label}
              </Button>
            </span>
          </Tooltip>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              kept.add(key)
              setKeptTick((n) => n + 1)
            }}
          >
            Keep as is
          </Button>
        </>
      }
    >
      {name} is loaded with other settings than its {own ? MODE_LABELS[mode] : 'default'} settings.
    </Banner>
  )
})
