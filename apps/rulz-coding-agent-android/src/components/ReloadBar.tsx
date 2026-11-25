import { RotateCw } from 'lucide-react'
import { useState } from 'react'
import { MODE_LABEL, reloadOffer } from '@shared/modes'
import type { AgentMode } from '@shared/types'
import { errorMessage } from '../lib/api'
import { cn } from '../lib/format'
import { useApp } from '../stores/app'
import { useModelModes, useModes } from '../stores/modes'
import { useWorkspace } from '../stores/workspace'
import { Button, Spinner } from './ui'

/**
 * Offered when the model runs with other load settings than the mode's in RulZ LLM Suite. A reload is
 * only ever offered, never automatic, and the computer refuses it while a task or batch run goes (its
 * reason is shown here). Hidden on a computer that does not know modes, and for a rental's model.
 * `busy`: something else is running (a batch run), so the button waits.
 */
export function ReloadBar({ mode, busy: otherBusy, className }: { mode: AgentMode; busy?: boolean; className?: string }) {
  const { model, modes, older } = useModelModes()
  const kept = useModes((s) => s.kept[model])
  const running = useWorkspace((s) => s.running)
  const offSite = useApp((s) => {
    const m = s.connection.models.find((m) => m.id === model)
    return !!(m?.rental || m?.cloud)
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const offer = older || offSite ? null : reloadOffer(modes ?? null, mode, kept)
  const waiting = running || !!otherBusy

  const go = async (): Promise<void> => {
    setLoading(true)
    setError('')
    try {
      await useModes.getState().reload(model, mode)
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div aria-live="polite">
      {offer && (
        <div className={cn('space-y-2 rounded-xl border border-border bg-panel-2 p-3 text-[13px]', className)}>
          <p className="min-w-0">
            The model is loaded with other settings than your {MODE_LABEL[mode]} settings in RulZ LLM Suite.
            {waiting && <span className="text-faint"> You can reload it when the task has finished.</span>}
          </p>
          {error && <p className="text-danger">{error}</p>}
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" disabled={waiting || loading} onClick={() => void go()}>
              {loading ? <Spinner /> : <RotateCw size={15} aria-hidden />} {offer.label}
            </Button>
            <Button variant="ghost" disabled={loading} onClick={() => useModes.getState().keep(model)}>
              Keep as is
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
