import { AlertTriangle, Loader2, XCircle } from 'lucide-react'
import type { MergeCheck, ModelFileInfo } from '@shared/merge'
import { formatBytesGiB, splitName } from '../../lib/merge'
import { Badge, Card } from '../ui'

function SourceLine({ letter, info }: { letter: string; info: ModelFileInfo }) {
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
      <span className="w-4 font-semibold">{letter}</span>
      <span className="min-w-0 truncate" title={info.name}>
        {splitName(info.name).file}
      </span>
      {info.readable ? (
        <>
          <span className="text-muted tabular-nums">{formatBytesGiB(info.sizeBytes)}</span>
          <Badge tone={info.quantized ? 'warn' : 'neutral'}>{info.quantized ? (info.quantLabel ?? 'quantized') : 'full precision'}</Badge>
          <span className="text-xs text-faint tabular-nums">{info.tensorCount.toLocaleString('en-US')} layers</span>
        </>
      ) : (
        <span className="text-muted">Cannot read: {info.error ?? 'unknown reason'}</span>
      )}
    </li>
  )
}

export function CheckPanel({ check, checking, ready }: { check: MergeCheck | null; checking: boolean; ready: boolean }) {
  return (
    <Card title="Check" description="Looks at the files before anything is loaded.">
      {!ready && !checking && <p className="text-[13px] text-muted">Choose the sources to check them.</p>}
      {checking && (
        <p className="flex items-center gap-2 text-[13px] text-muted">
          <Loader2 size={14} className="animate-spin" />
          Checking the files.
        </p>
      )}
      {check && !checking && (
        <div className="space-y-4">
          <ul className="space-y-2">
            {check.a && <SourceLine letter="A" info={check.a} />}
            {check.b && <SourceLine letter="B" info={check.b} />}
            {check.c && <SourceLine letter="C" info={check.c} />}
          </ul>
          {check.ramNeededBytes !== null && (
            <p className="text-[13px] text-muted">
              Needs about {formatBytesGiB(check.ramNeededBytes)} of memory
              {check.ramFreeBytes !== null ? `, ${formatBytesGiB(check.ramFreeBytes)} free in WSL.` : '.'}
            </p>
          )}
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
          {check.problems.length === 0 && check.warnings.length === 0 && check.compatible === true && (
            <p className="text-[13px] text-ok">These files fit together.</p>
          )}
        </div>
      )}
    </Card>
  )
}
