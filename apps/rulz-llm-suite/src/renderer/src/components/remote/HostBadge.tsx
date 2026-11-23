import { Cloud, LoaderCircle } from 'lucide-react'
import type { RemoteHost } from '@shared/remote'
import { Badge, Tooltip } from '../ui'

/**
 * Where a loaded model runs: a "vast.ai" badge with the rental's label; "Reconnecting" while the connection is being
 * remade. `compact`: an icon only, for the top bar's small chips (the words are in its label and tooltip).
 */
export function HostBadge({ host, reconnecting, compact }: { host: RemoteHost; reconnecting?: boolean; compact?: boolean }) {
  if (compact) {
    const text = reconnecting ? `The connection to ${host.label} dropped and is being remade. The model is still loaded there.` : `Runs on the vast.ai instance ${host.label}`
    return (
      <Tooltip content={text}>
        <span className={reconnecting ? 'shrink-0 text-warn' : 'shrink-0 text-muted'} role="img" aria-label={text}>
          {reconnecting ? <LoaderCircle size={12} className="animate-spin" /> : <Cloud size={12} />}
        </span>
      </Tooltip>
    )
  }
  if (reconnecting) {
    return (
      <Tooltip content={`The connection to ${host.label} dropped and is being remade. The model is still loaded there.`}>
        <span>
          <Badge tone="warn">
            <LoaderCircle size={11} className="animate-spin" /> Reconnecting
          </Badge>
        </span>
      </Tooltip>
    )
  }
  return (
    <Tooltip content={`Runs on the vast.ai instance ${host.label}`}>
      <span className="shrink-0">
        <Badge>
          <Cloud size={11} /> {`vast.ai · ${host.label}`}
        </Badge>
      </span>
    </Tooltip>
  )
}
