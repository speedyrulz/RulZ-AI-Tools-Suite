import { AlertTriangle, Loader2, XCircle } from 'lucide-react'
import type { LoraMergeCheck, LoraMergeInputInfo } from '@shared/loraMerge'
import { formatInputKind, formatRankRange, formatSize } from '../../lib/loraMerge'
import { splitName } from '../../lib/merge'
import { Badge, Card } from '../ui'

function InputLine({ info }: { info: LoraMergeInputInfo }) {
  return (
    <li className="space-y-0.5 text-[13px]">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="min-w-0 truncate" title={info.name}>
          {splitName(info.name).file}
        </span>
        {info.readable ? (
          <>
            <Badge tone={info.supported ? 'neutral' : 'danger'}>{formatInputKind(info.format, info.rank)}</Badge>
            <span className="text-xs text-faint tabular-nums">
              {info.modules.toLocaleString('en-US')} layers
              {info.textEncoderModules > 0 ? `, ${info.textEncoderModules.toLocaleString('en-US')} text encoder` : ''}
            </span>
            {info.dtype && <span className="text-xs text-faint">{info.dtype}</span>}
            {info.baseModelHint && <span className="text-xs text-faint">made for {info.baseModelHint}</span>}
          </>
        ) : (
          <span className="text-muted">could not be read: {info.error ?? 'unknown reason'}</span>
        )}
      </div>
      {info.readable && !info.supported && <p className="text-xs text-danger-text">{info.unsupportedReason ?? 'This file cannot be merged.'}</p>}
    </li>
  )
}

function Totals({ check }: { check: LoraMergeCheck }) {
  const item = (label: string, value: string) => (
    <div>
      <dt className="text-xs text-faint">{label}</dt>
      <dd className="text-[13px] tabular-nums">{value}</dd>
    </div>
  )
  return (
    <dl className="flex flex-wrap gap-x-8 gap-y-2">
      {item('Layers in the result', check.totalModules.toLocaleString('en-US'))}
      {item('Layers changed by two or more', check.sharedModules.toLocaleString('en-US'))}
      {item('Output rank', formatRankRange(check.outputRank))}
      {item('Estimated size', formatSize(check.estimatedBytes))}
    </dl>
  )
}

export function CheckPanel({ check, checking, checkError, ready }: { check: LoraMergeCheck | null; checking: boolean; checkError: string | null; ready: boolean }) {
  return (
    <Card title="Check" description="Looks at the LoRA headers before anything is merged.">
      {!ready && !checking && <p className="text-[13px] text-muted">Pick two LoRAs to check them.</p>}
      {checking && (
        <p className="flex items-center gap-2 text-[13px] text-muted">
          <Loader2 size={14} className="animate-spin" />
          Checking the files.
        </p>
      )}
      {checkError && !checking && <p className="text-[13px] text-danger-text">The LoRAs could not be checked: {checkError}</p>}
      {check && !checking && (
        <div className="space-y-4">
          <ul className="space-y-2">
            {check.inputs.map((i, k) => (
              <InputLine key={`${k}-${i.name}`} info={i} />
            ))}
          </ul>
          <Totals check={check} />
          {check.problems.length > 0 && (
            <ul className="space-y-1.5 rounded-lg border border-danger-border bg-danger-bg px-3 py-2">
              {check.problems.map((p) => (
                <li key={p} className="flex items-start gap-2 text-[13px] text-danger-text">
                  <XCircle size={14} className="mt-0.5 shrink-0" />
                  {p}
                </li>
              ))}
            </ul>
          )}
          {check.warnings.length > 0 && (
            <ul className="space-y-1.5 rounded-lg border border-warn/25 bg-warn/10 px-3 py-2">
              {check.warnings.map((w) => (
                <li key={w} className="flex items-start gap-2 text-[13px] text-warn-text">
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                  {w}
                </li>
              ))}
            </ul>
          )}
          {check.problems.length === 0 && check.warnings.length === 0 && <p className="text-[13px] text-ok">These LoRAs can be merged.</p>}
        </div>
      )}
    </Card>
  )
}
