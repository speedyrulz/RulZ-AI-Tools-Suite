import { Check, X } from 'lucide-react'
import { KNOWN_NODES } from '@shared/types'
import { useApp } from '../../stores/app'
import { Badge, Card } from '../ui'

export function CapabilitiesCard() {
  const status = useApp((s) => s.status)
  const caps = useApp((s) => s.capabilities)
  if (!status?.reachable || !caps) return null

  const counts: [string, number][] = [
    ['Checkpoints', caps.checkpoints.length],
    ['Diffusion models', caps.diffusionModels.length],
    ['LoRAs', caps.loras.length],
    ['Text encoders', caps.textEncoders.length],
    ['VAEs', caps.vaes.length]
  ]

  return (
    <Card title="What this server has" description="Models and nodes the connected ComfyUI reports.">
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {counts.map(([label, n]) => (
          <div key={label} className="rounded-lg border border-border bg-panel-2 px-3 py-2">
            <dd className="text-xl font-semibold tabular-nums">{n}</dd>
            <dt className="text-xs text-faint">{label}</dt>
          </div>
        ))}
      </dl>
      <h3 className="mt-5 mb-2 text-[13px] font-medium">Nodes</h3>
      <div className="flex flex-wrap gap-1.5">
        {KNOWN_NODES.map((n) => {
          const has = caps.nodes[n] === true
          return (
            <Badge key={n} tone={has ? 'ok' : 'neutral'}>
              {has ? <Check size={12} /> : <X size={12} />}
              {n}
              <span className="sr-only">{has ? ' found' : ' missing'}</span>
            </Badge>
          )
        })}
      </div>
      {caps.blockMergeNodes.length > 0 && (
        <>
          <h3 className="mt-5 mb-2 text-[13px] font-medium">Block merge nodes found</h3>
          <div className="flex flex-wrap gap-1.5">
            {caps.blockMergeNodes.map((n) => (
              <Badge key={n} tone="accent">
                {n}
              </Badge>
            ))}
          </div>
        </>
      )}
    </Card>
  )
}
