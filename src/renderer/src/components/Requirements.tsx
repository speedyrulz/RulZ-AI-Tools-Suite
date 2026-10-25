import { CheckCircle2, CircleDashed, Download, ExternalLink, Loader2, XCircle } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import type { RequirementRef } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { cn } from '../lib/format'
import { useApps } from '../stores/apps'
import { Button } from './ui'

/** Status of an app's requirements, with winget install buttons for the missing ones. */
export function RequirementList({ refs }: { refs: RequirementRef[] }) {
  const statuses = useApps((s) => s.requirements)
  const [installing, setInstalling] = useState<string | null>(null)

  const install = async (id: RequirementRef['id'], label: string): Promise<void> => {
    setInstalling(id)
    try {
      const next = await invoke('requirements:install', id)
      useApps.setState({ requirements: next })
      const ok = next.find((s) => s.id === id)?.ok
      if (ok) toast.success(`${label} installed`)
      else toast.warning(`${label} was installed, but isn't found yet. Restart the launcher if it stays missing.`)
    } catch (e) {
      toast.error(`${label}: ${errorMessage(e)}`)
    } finally {
      setInstalling(null)
    }
  }

  return (
    <ul className="space-y-2">
      {refs.map((r) => {
        const s = statuses.find((x) => x.id === r.id)
        const icon = !s ? (
          <CircleDashed className="size-4 animate-spin text-faint" />
        ) : s.ok ? (
          <CheckCircle2 className="size-4 text-ok" />
        ) : r.optional ? (
          <CircleDashed className="size-4 text-faint" />
        ) : (
          <XCircle className="size-4 text-danger" />
        )
        return (
          <li key={r.id} className="flex items-start gap-2.5">
            <span className="mt-0.5">{icon}</span>
            <div className="min-w-0 flex-1">
              <div className="text-[13px] text-text">
                {s?.label ?? r.id}
                {r.optional && <span className="text-faint"> (optional)</span>}
              </div>
              <div className={cn('text-xs', s && !s.ok && !r.optional ? 'text-[#ff8a8e]' : 'text-faint')}>
                {s ? s.detail : 'Checking…'}
                {r.note && <span className="text-faint"> · {r.note}</span>}
              </div>
            </div>
            {s && !s.ok && (
              <div className="flex shrink-0 gap-1">
                {s.canInstall && (
                  <Button size="sm" disabled={!!installing} onClick={() => void install(r.id, s.label)}>
                    {installing === r.id ? <Loader2 className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}
                    Install
                  </Button>
                )}
                <Button size="sm" variant="ghost" className="px-2" title="Download page" onClick={() => void invoke('shell:openExternal', s.helpUrl)}>
                  <ExternalLink className="size-3.5" />
                </Button>
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}
