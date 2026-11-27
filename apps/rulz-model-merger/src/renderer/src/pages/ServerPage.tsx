import { ListChecks, Play, Square } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import type { GpuInfo, OwnServerSettings, PreflightReport, ServerState } from '@shared/types'
import { CapabilitiesCard } from '../components/server/CapabilitiesCard'
import { LogWell } from '../components/server/LogWell'
import { OwnServerCard } from '../components/server/OwnServerCard'
import { PreflightPanel } from '../components/server/PreflightPanel'
import { UrlCard } from '../components/server/UrlCard'
import { Button, Card, PageTitle, Segmented, StatusChip, type StatusTone } from '../components/ui'
import { call, errText, invoke } from '../lib/api'
import { useApp } from '../stores/app'

const FRESH_MS = 30_000

function serverChip(state: ServerState | null): { tone: StatusTone; label: string; busy: boolean } {
  switch (state?.status) {
    case 'running':
      return { tone: 'ok', label: state.port ? `Running on :${state.port}` : 'Running', busy: false }
    case 'starting':
      return { tone: 'warn', label: 'Starting', busy: true }
    case 'stopping':
      return { tone: 'warn', label: 'Stopping', busy: true }
    case 'error':
      return { tone: 'danger', label: 'Error', busy: false }
    default:
      return { tone: 'idle', label: 'Stopped', busy: false }
  }
}

/** GPU list, refreshed every 3 s while the page is visible. */
function useGpus(): GpuInfo[] {
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  useEffect(() => {
    let alive = true
    let warned = false
    let timer: number | undefined
    // One call at a time: the next one is scheduled when the last has finished.
    const tick = async (): Promise<void> => {
      if (!document.hidden) {
        try {
          const g = await invoke('gpu:list')
          if (alive) setGpus(g)
        } catch (e) {
          if (alive && !warned) {
            warned = true
            toast.error(errText(e))
          }
        }
      }
      if (alive) timer = window.setTimeout(() => void tick(), 3000)
    }
    void tick()
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [])
  return gpus
}

export function ServerPage() {
  const settings = useApp((s) => s.settings)
  const serverState = useApp((s) => s.serverState)
  const patchSettings = useApp((s) => s.patchSettings)
  const refreshStatus = useApp((s) => s.refreshStatus)
  const gpus = useGpus()

  const [report, setReport] = useState<PreflightReport | null>(null)
  const [reportAt, setReportAt] = useState(0)
  const [confirmed, setConfirmed] = useState<ReadonlySet<string>>(new Set())
  const [checking, setChecking] = useState(false)
  const [busy, setBusy] = useState(false)

  const own = settings?.ownServer
  // Everything a check depends on. When it changes, the last check and its ticks no longer apply.
  const checkKey = own ? JSON.stringify({ ...own, stopTimeoutSec: 0 }) : ''
  const keyRef = useRef(checkKey)
  keyRef.current = checkKey
  useEffect(() => {
    setReport(null)
    setConfirmed(new Set())
  }, [checkKey])

  const runPreflight = useCallback(async (): Promise<PreflightReport | undefined> => {
    const startedWith = keyRef.current
    setChecking(true)
    const r = await call('server:preflight')
    setChecking(false)
    // The settings changed while it ran: its answer is about the old ones.
    if (!r || startedWith !== keyRef.current) return undefined
    setReport(r)
    setReportAt(Date.now())
    setConfirmed(new Set())
    return r
  }, [])

  if (!settings || !own) return null
  const mode = settings.connection.mode
  const status = serverState?.status ?? 'stopped'
  const active = status === 'running' || status === 'starting'
  const transitional = status === 'starting' || status === 'stopping'
  const chip = serverChip(serverState)

  const blocks = report?.issues.filter((i) => i.level === 'block') ?? []
  const confirmCodes = report?.issues.filter((i) => i.level === 'confirm').map((i) => i.code) ?? []
  const unticked = confirmCodes.filter((c) => !confirmed.has(c))
  const startBlocked = blocks.length > 0 || unticked.length > 0

  const patchOwn = (p: Partial<OwnServerSettings>): void => void patchSettings({ ownServer: p })

  const start = async (): Promise<void> => {
    setBusy(true)
    try {
      let r = report
      // A new report starts with nothing ticked.
      let ticks: ReadonlySet<string> = confirmed
      if (!r || Date.now() - reportAt > FRESH_MS) {
        r = (await runPreflight()) ?? null
        ticks = new Set()
      }
      if (!r) return
      if (r.issues.some((i) => i.level === 'block')) return
      const need = r.issues.filter((i) => i.level === 'confirm' && !ticks.has(i.code))
      if (need.length > 0) {
        toast.info('Tick "I understand" for each warning, then start the server.')
        return
      }
      const codes = r.issues.filter((i) => i.level === 'confirm').map((i) => i.code)
      const s = await call('server:start', codes)
      if (s) {
        useApp.setState({ serverState: s })
        void refreshStatus()
      }
    } finally {
      setBusy(false)
    }
  }

  const stop = async (): Promise<void> => {
    setBusy(true)
    const s = await call('server:stop')
    if (s) useApp.setState({ serverState: s })
    setBusy(false)
    void refreshStatus()
  }

  return (
    <div>
      <PageTitle title="Server">Choose which ComfyUI the app sends its jobs to.</PageTitle>

      <div className="space-y-4">
        <Card>
          <Segmented
            label="Connection"
            value={mode}
            onChange={(m) => {
              if (m === mode) return
              const url = useApp.getState().settings?.connection.url ?? ''
              void patchSettings({ connection: { mode: m, url } }).then(() => refreshStatus())
            }}
            options={[
              { value: 'own', label: 'Own server' },
              { value: 'url', label: 'Server URL' }
            ]}
          />
          <p className="mt-3 text-[13px] text-muted">
            {mode === 'own'
              ? 'The app starts its own ComfyUI in WSL, so your other ComfyUI servers and their queues are not touched.'
              : 'Send jobs to a ComfyUI that is already running. Jobs share its queue with your other renders.'}
          </p>
        </Card>

        {mode === 'url' ? (
          <UrlCard
            url={settings.connection.url}
            onUrl={(v) => {
              // Change only the URL; the mode is whatever it is now.
              const mode = useApp.getState().settings?.connection.mode ?? 'url'
              return patchSettings({ connection: { mode, url: v } })
            }}
          />
        ) : (
          <>
            <OwnServerCard own={own} gpus={gpus} servers={report?.runningServers ?? []} locked={active || status === 'stopping'} onChange={patchOwn} />

            <Card
              title="Run"
              actions={<StatusChip tone={chip.tone} label={chip.label} busy={chip.busy} />}
              description={serverState?.status === 'error' && serverState.error ? undefined : 'Check first to see what the server would use and whether anything is in the way.'}
            >
              {serverState?.status === 'error' && serverState.error && (
                <p role="alert" className="mb-3 rounded-lg border border-danger-border bg-danger-bg px-3 py-2 text-[13px] text-danger-text">
                  {serverState.error}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={() => void runPreflight()} disabled={checking || active || transitional}>
                  <ListChecks size={16} />
                  Check
                </Button>
                {active || status === 'stopping' ? (
                  <Button variant="primary" onClick={() => void stop()} disabled={busy || status === 'stopping'}>
                    <Square size={16} />
                    Stop server
                  </Button>
                ) : (
                  <Button variant="primary" onClick={() => void start()} disabled={busy || checking || startBlocked}>
                    <Play size={16} />
                    Start server
                  </Button>
                )}
              </div>
              {report && !active && (
                <div className="mt-4 border-t border-border pt-4">
                  <PreflightPanel
                    report={report}
                    confirmed={confirmed}
                    onToggle={(code, on) =>
                      setConfirmed((prev) => {
                        const next = new Set(prev)
                        if (on) next.add(code)
                        else next.delete(code)
                        return next
                      })
                    }
                  />
                </div>
              )}
              <div className="mt-4">
                <LogWell />
              </div>
            </Card>
          </>
        )}

        <CapabilitiesCard />
      </div>
    </div>
  )
}
