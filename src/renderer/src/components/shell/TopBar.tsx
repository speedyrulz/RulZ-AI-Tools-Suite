import { ArrowUpFromLine, ChevronDown, Cpu, FileText, LoaderCircle } from 'lucide-react'
import type { LoadedInstance } from '@shared/types'
import { cn, formatBytes } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { IconButton, Kbd, ProgressBar, Tooltip } from '../ui'

export function TopBar() {
  const instances = useModels((s) => s.instances)
  const activeModelId = useModels((s) => s.activeModelId)
  const openPicker = useApp((s) => s.openPicker)
  const active = instances.find((i) => i.modelId === activeModelId)

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-panel px-4">
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <button
          onClick={() => openPicker()}
          className="flex h-9 max-w-[460px] min-w-[300px] items-center gap-2.5 rounded-lg border border-border bg-panel-2 px-3 text-left transition-colors hover:border-accent/60"
        >
          {active?.status === 'loading' ? (
            <LoaderCircle size={15} className="shrink-0 animate-spin text-accent" />
          ) : (
            <span className={cn('size-2 shrink-0 rounded-full', active?.status === 'ready' ? 'bg-ok' : 'bg-faint/50')} />
          )}
          <span className={cn('min-w-0 flex-1 truncate text-[13px]', !active && 'text-muted')}>
            {active ? active.modelName : 'Select a model to load'}
          </span>
          <Kbd>Ctrl+L</Kbd>
          <ChevronDown size={14} className="shrink-0 text-faint" />
        </button>
        <div className="flex min-w-0 items-center gap-1.5 overflow-x-auto">
          {instances.map((i) => (
            <InstanceChip key={i.instanceId} inst={i} active={i.modelId === activeModelId} />
          ))}
        </div>
      </div>
      <HardwareMeters />
    </header>
  )
}

function InstanceChip({ inst, active }: { inst: LoadedInstance; active: boolean }) {
  const setActive = useModels((s) => s.setActive)
  const unload = useModels((s) => s.unload)
  const showLog = useApp((s) => s.showLog)
  const loading = inst.status === 'loading'

  return (
    <div
      className={cn(
        'group flex h-9 shrink-0 items-center gap-1 rounded-lg border pr-1 pl-2.5 text-xs transition-colors',
        active ? 'border-accent/50 bg-accent-soft/60' : 'border-border bg-panel-2'
      )}
    >
      <button className="flex max-w-56 min-w-0 flex-col items-start" onClick={() => setActive(inst.modelId)}>
        <span className="max-w-56 truncate font-medium">{inst.modelName}</span>
        {loading ? (
          <span className="flex w-40 items-center gap-1.5 text-[10px] text-muted">
            <ProgressBar value={inst.progress} className="flex-1" />
            {Math.round(inst.progress * 100)}%
          </span>
        ) : (
          <span className="text-[10px] text-muted">
            {inst.status === 'stopping' ? 'Unloading…' : `${inst.contextLength ? `${inst.contextLength.toLocaleString()} ctx` : 'Ready'}${inst.hasVision ? ' · vision' : ''}`}
          </span>
        )}
      </button>
      <IconButton size="sm" label="Server log" onClick={() => showLog(inst.instanceId)}>
        <FileText size={13} />
      </IconButton>
      <IconButton size="sm" label={loading ? 'Cancel loading' : 'Eject'} onClick={() => void unload(inst.instanceId)} disabled={inst.status === 'stopping'}>
        <ArrowUpFromLine size={14} />
      </IconButton>
    </div>
  )
}

function HardwareMeters() {
  const hw = useModels((s) => s.hardware)
  if (!hw) return null
  return (
    <div className="flex shrink-0 items-center gap-4">
      {hw.gpus.map((g) => (
        <Meter
          key={g.index}
          label={`GPU${g.index}`}
          title={`${g.name} · ${g.utilPct}% utilisation`}
          used={g.memUsedMiB * 1024 * 1024}
          total={g.memTotalMiB * 1024 * 1024}
        />
      ))}
      <Meter label="RAM" icon={<Cpu size={11} />} used={hw.ramUsedBytes} total={hw.ramTotalBytes} />
    </div>
  )
}

function Meter({ label, title, used, total, icon }: { label: string; title?: string; used: number; total: number; icon?: React.ReactNode }) {
  const frac = total ? used / total : 0
  return (
    <Tooltip content={title ?? label}>
      <div className="w-27 text-[10px] text-muted">
        <div className="mb-1 flex items-center justify-between">
          <span className="flex items-center gap-1 font-medium">
            {icon}
            {label}
          </span>
          <span className="tabular-nums">
            {formatBytes(used)} / {formatBytes(total, 0)}
          </span>
        </div>
        <div className="h-1 overflow-hidden rounded-full bg-hover">
          <div
            className={cn('h-full rounded-full', frac > 0.92 ? 'bg-danger' : frac > 0.75 ? 'bg-warn' : 'bg-accent')}
            style={{ width: `${Math.round(frac * 100)}%` }}
          />
        </div>
      </div>
    </Tooltip>
  )
}
