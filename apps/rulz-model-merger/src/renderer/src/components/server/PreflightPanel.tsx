import { AlertTriangle, Info, OctagonAlert } from 'lucide-react'
import type { PreflightReport } from '@shared/types'
import { formatGiB } from '../../lib/format'
import { Checkbox } from '../ui'

export function PreflightPanel({
  report,
  confirmed,
  onToggle
}: {
  report: PreflightReport
  confirmed: ReadonlySet<string>
  onToggle: (code: string, on: boolean) => void
}) {
  const blocks = report.issues.filter((i) => i.level === 'block')
  const confirms = report.issues.filter((i) => i.level === 'confirm')
  const infos = report.issues.filter((i) => i.level === 'info')

  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-[13px]">
        <dt className="text-muted">Free memory</dt>
        <dd>
          {formatGiB(report.freeRamGiB)} of {formatGiB(report.totalRamGiB)}
        </dd>
        <dt className="text-muted">Port</dt>
        <dd>{report.port === null ? 'None free' : `:${report.port}`}</dd>
        <dt className="text-muted">Running ComfyUI servers</dt>
        <dd>
          {report.runningServers.length === 0 ? (
            'None found'
          ) : (
            <ul className="space-y-1">
              {report.runningServers.map((s) => (
                <li key={s.url}>
                  <span className="font-mono text-xs">{s.url}</span>
                  {s.own && <span className="text-faint"> (started by this app)</span>}
                  <div className="text-xs text-faint">
                    {s.version ? `Version ${s.version}. ` : ''}
                    {s.devices.length ? s.devices.join(', ') : 'Devices unknown'}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </dd>
      </dl>

      {blocks.length > 0 && (
        <div role="alert" className="space-y-1.5 rounded-lg border border-danger-border bg-danger-bg p-3">
          {blocks.map((i) => (
            <p key={i.code} className="flex items-start gap-2 text-[13px] text-danger-text">
              <OctagonAlert size={16} className="mt-px shrink-0" />
              <span>{i.message}</span>
            </p>
          ))}
        </div>
      )}

      {confirms.length > 0 && (
        <div className="space-y-2.5 rounded-lg border border-warn/30 bg-warn/10 p-3">
          {confirms.map((i) => (
            <div key={i.code} className="space-y-1.5">
              <p className="flex items-start gap-2 text-[13px] text-warn-text">
                <AlertTriangle size={16} className="mt-px shrink-0" />
                <span>{i.message}</span>
              </p>
              <div className="pl-6">
                <Checkbox checked={confirmed.has(i.code)} onChange={(on) => onToggle(i.code, on)}>
                  I understand
                </Checkbox>
              </div>
            </div>
          ))}
        </div>
      )}

      {infos.map((i) => (
        <p key={i.code} className="flex items-start gap-2 text-xs text-muted">
          <Info size={14} className="mt-px shrink-0" />
          <span>{i.message}</span>
        </p>
      ))}

      {report.issues.length === 0 && <p className="text-[13px] text-muted">No problems found.</p>}
    </div>
  )
}
