import { LoaderCircle, RotateCw } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { MODE_LABEL, reloadOffer } from '@shared/modes'
import type { AgentMode } from '@shared/types'
import { errorMessage } from '../lib/api'
import { cn } from '../lib/format'
import { useApp } from '../stores/app'
import { useModelModes, useModes } from '../stores/modes'
import { useWorkspace } from '../stores/workspace'
import { Button } from './ui'

/**
 * Offered when the model runs with other load settings than the mode's in RulZ LLM Suite. A reload is
 * only ever offered, never automatic, and the main process refuses it while a task or batch run goes.
 */
export function ReloadBar({ mode, className }: { mode: AgentMode; className?: string }) {
  const { model, modes } = useModelModes()
  const kept = useModes((s) => s.kept[model])
  const keep = useModes((s) => s.keep)
  const reload = useModes((s) => s.reload)
  const running = useWorkspace((s) => s.running)
  const rental = useApp((s) => s.connection.models.find((m) => m.id === model)?.rental)
  const [busy, setBusy] = useState(false)
  // A rental's model is not reloaded from here.
  const offer = rental ? null : reloadOffer(modes ?? null, mode, kept)
  if (!offer) return null

  const go = async (): Promise<void> => {
    setBusy(true)
    try {
      await reload(model, mode)
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className={cn('flex flex-wrap items-center gap-2 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-[13px]', className)}>
      <span className="min-w-0 flex-1">
        The model is loaded with other settings than your {MODE_LABEL[mode]} settings in RulZ LLM Suite.
        {running && <span className="text-faint"> It can be reloaded when the task has finished.</span>}
      </span>
      <Button size="sm" variant="primary" disabled={running || busy} onClick={() => void go()}>
        {busy ? <LoaderCircle size={14} className="animate-spin" /> : <RotateCw size={14} />} {offer.label}
      </Button>
      <Button size="sm" variant="ghost" disabled={busy} onClick={() => keep(model)}>
        Keep as is
      </Button>
    </div>
  )
}
