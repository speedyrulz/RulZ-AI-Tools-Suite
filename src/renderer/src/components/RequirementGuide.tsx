import { CheckCircle2, ExternalLink, Loader2, RefreshCw, XCircle } from 'lucide-react'
import { useState } from 'react'
import { REQUIREMENT_GUIDES } from '@shared/requirements'
import { invoke } from '../lib/api'
import { useApps } from '../stores/apps'
import { Button, Dialog } from './ui'

/** Step-by-step instructions for installing a prerequisite by hand, with a "Check again" button. */
export function RequirementGuide() {
  const id = useApps((s) => s.guideId)
  const status = useApps((s) => s.requirements.find((r) => r.id === id))
  const close = (): void => useApps.setState({ guideId: null })
  const [checking, setChecking] = useState(false)
  const [checked, setChecked] = useState(false)

  if (!id) return null
  const guide = REQUIREMENT_GUIDES[id]

  const checkAgain = async (): Promise<void> => {
    setChecking(true)
    await useApps.getState().checkRequirements(true)
    setChecking(false)
    setChecked(true)
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) {
          close()
          setChecked(false)
        }
      }}
      title={`Install ${guide.label}`}
      description="Follow these steps, then check again."
      className="w-[560px]"
      footer={
        <>
          <Button variant="ghost" onClick={close}>
            Close
          </Button>
          <Button onClick={() => void invoke('shell:openExternal', guide.helpUrl)}>
            <ExternalLink className="size-4" /> {guide.linkLabel}
          </Button>
          <Button variant="primary" disabled={checking} onClick={() => void checkAgain()}>
            {checking ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            Check again
          </Button>
        </>
      }
    >
      <ol className="space-y-3">
        {guide.steps.map((step, i) => (
          <li key={step} className="flex gap-3 text-[13.5px] leading-relaxed text-[#d4d4d8]">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-[#ff6b72]">
              {i + 1}
            </span>
            <span className="pt-0.5">{step}</span>
          </li>
        ))}
      </ol>
      {status && (checked || status.ok) && (
        <div
          className={`mt-5 flex items-center gap-2 rounded-lg border px-3 py-2 text-[13px] ${
            status.ok
              ? 'border-[#4ade80]/25 bg-[#4ade80]/[0.07] text-ok'
              : 'border-[#ff5a5f]/25 bg-[#ff5a5f]/[0.07] text-[#ff8a8e]'
          }`}
        >
          {status.ok ? <CheckCircle2 className="size-4" /> : <XCircle className="size-4" />}
          {status.ok
            ? `Found: ${status.detail}. You can install the app now.`
            : `Still not found (${status.detail}). If you just installed it, restarting the launcher can help.`}
        </div>
      )}
    </Dialog>
  )
}
