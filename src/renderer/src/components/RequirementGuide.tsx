import { CheckCircle2, ExternalLink, Loader2, RefreshCw, XCircle } from 'lucide-react'
import { useState } from 'react'
import { REQUIREMENT_GUIDES, requirementRefLabel, statusFor } from '@shared/requirements'
import type { RequirementRef } from '@shared/types'
import { invoke } from '../lib/api'
import { useApps } from '../stores/apps'
import { Button, Dialog } from './ui'

/** Step-by-step instructions for installing a prerequisite by hand, with a "Check again" button. */
export function RequirementGuide() {
  const ref = useApps((s) => s.guide)
  const found = useApps((s) => s.requirements.find((r) => r.id === ref?.id))
  const [checking, setChecking] = useState(false)
  // The guide that "Check again" was used in: this component stays mounted, so a plain flag would carry an old
  // "Still not found" over to the next guide opened.
  const [checkedRef, setCheckedRef] = useState<RequirementRef | null>(null)
  const close = (): void => {
    useApps.setState({ guide: null })
    setCheckedRef(null)
  }

  if (!ref) return null
  const guide = REQUIREMENT_GUIDES[ref.id]
  const checked = checkedRef === ref
  // As the app sees it: a Python older than the app needs isn't "found".
  const status = statusFor(ref, found)
  const tooOld = !!found?.ok && !!status && !status.ok

  const checkAgain = async (): Promise<void> => {
    setChecking(true)
    await useApps.getState().checkRequirements(true)
    setChecking(false)
    setCheckedRef(ref)
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) close()
      }}
      title={`Install ${requirementRefLabel(ref)}`}
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
          <li key={step} className="flex gap-3 text-[13.5px] leading-relaxed text-body">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent-text">
              {i + 1}
            </span>
            <span className="pt-0.5">{step}</span>
          </li>
        ))}
      </ol>
      {tooOld && (
        <p className="mt-4 text-[13px] leading-relaxed text-body">
          {found?.detail} is installed, but this app needs {requirementRefLabel(ref)}. Windows uses the Python listed
          first, so uninstall the old one (Settings ▸ Apps ▸ Installed apps) after installing the new one.
        </p>
      )}
      {status && (checked || status.ok) && (
        <div
          className={`mt-5 flex items-center gap-2 rounded-lg border px-3 py-2 text-[13px] ${
            status.ok
              ? 'border-ok/25 bg-ok/[0.07] text-ok'
              : 'border-danger/25 bg-danger/[0.07] text-danger-text'
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
