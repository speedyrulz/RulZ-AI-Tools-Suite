import { Cpu, MemoryStick } from 'lucide-react'
import type { GpuInfo, RunningServer, ServerDevice } from '@shared/types'
import { cn, formatMiB } from '../../lib/format'
import { Badge } from '../ui'

function sameDevice(a: ServerDevice, b: ServerDevice): boolean {
  return a.kind === 'cpu' ? b.kind === 'cpu' : b.kind === 'gpu' && a.index === b.index
}

export function DevicePicker({
  gpus,
  device,
  servers,
  disabled,
  onChange
}: {
  gpus: GpuInfo[]
  device: ServerDevice
  servers: RunningServer[]
  disabled?: boolean
  onChange: (d: ServerDevice) => void
}) {
  const options: { device: ServerDevice; gpu?: GpuInfo }[] = [
    { device: { kind: 'cpu' } },
    ...gpus.map((g) => ({ device: { kind: 'gpu', index: g.index, name: g.name } as ServerDevice, gpu: g }))
  ]
  const portsFor = (index: number): number[] => servers.filter((s) => s.gpuIndexes.includes(index)).map((s) => s.port)

  return (
    <div role="radiogroup" aria-label="Device" className="space-y-2">
      {options.map((o) => {
        const selected = sameDevice(o.device, device)
        const g = o.gpu
        const ports = g ? portsFor(g.index) : []
        return (
          <button
            key={g ? `gpu-${g.index}` : 'cpu'}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(o.device)}
            className={cn(
              'flex w-full items-center gap-3 rounded-lg border bg-panel px-3 py-2.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45',
              selected ? 'border-accent bg-accent-soft' : 'border-border hover:border-border-strong'
            )}
          >
            {g ? <MemoryStick size={16} className="shrink-0 text-muted" /> : <Cpu size={16} className="shrink-0 text-muted" />}
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-medium">{g ? `GPU ${g.index}: ${g.name}` : 'CPU'}</div>
              <div className="text-xs text-faint">
                {g
                  ? `${formatMiB(g.memoryUsedMiB)} of ${formatMiB(g.memoryTotalMiB)} used, ${g.utilizationPct}% load`
                  : 'Slow, but uses no graphics memory.'}
              </div>
            </div>
            {ports.length > 0 && <Badge tone="warn">In use by {ports.map((p) => `:${p}`).join(', ')}</Badge>}
          </button>
        )
      })}
      {gpus.length === 0 && <p className="text-xs text-faint">No NVIDIA GPU was found, so only the CPU is offered.</p>}
    </div>
  )
}
