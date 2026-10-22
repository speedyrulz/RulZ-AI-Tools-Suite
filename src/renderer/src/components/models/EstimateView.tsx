import { useEffect, useState } from 'react'
import type { LoadConfig, MemoryEstimate } from '@shared/types'
import { invoke } from '../../lib/api'
import { formatBytes } from '../../lib/format'
import { Badge } from '../ui'

const VERDICT = {
  fits: { tone: 'ok', text: 'Fits in VRAM' },
  partial: { tone: 'warn', text: 'Partly on CPU (slower)' },
  tight: { tone: 'danger', text: 'Likely out of memory' },
  unknown: { tone: 'neutral', text: 'VRAM unknown' }
} as const

export function useEstimate(modelId: string | null, config: LoadConfig): MemoryEstimate | null {
  const [estimate, setEstimate] = useState<MemoryEstimate | null>(null)
  const key = JSON.stringify(config)
  useEffect(() => {
    if (!modelId) return setEstimate(null)
    let cancelled = false
    const t = setTimeout(() => {
      invoke('models:estimate', modelId, config)
        .then((e) => !cancelled && setEstimate(e))
        .catch(() => !cancelled && setEstimate(null))
    }, 200)
    return () => {
      cancelled = true
      clearTimeout(t)
    }
  }, [modelId, key])
  return estimate
}

export function EstimateView({ estimate }: { estimate: MemoryEstimate | null }) {
  if (!estimate) return null
  const v = VERDICT[estimate.verdict]
  return (
    <div className="rounded-lg border border-border bg-panel-2 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium text-muted">Estimated memory</span>
        <Badge tone={v.tone}>{v.text}</Badge>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <Stat label="GPU" value={formatBytes(estimate.gpuBytes)} sub={estimate.freeVramBytes !== null ? `${formatBytes(estimate.freeVramBytes)} free` : undefined} />
        <Stat label="Weights" value={formatBytes(estimate.weightsBytes)} />
        <Stat
          label="KV cache"
          value={estimate.kvBytes !== null ? formatBytes(estimate.kvBytes) : 'auto'}
          sub={estimate.contextLength ? `${estimate.contextLength.toLocaleString()} tokens` : 'sized by llama.cpp'}
        />
      </div>
    </div>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-[11px] text-faint">{label}</div>
      <div className="font-medium tabular-nums">{value}</div>
      {sub && <div className="text-[11px] text-faint">{sub}</div>}
    </div>
  )
}
