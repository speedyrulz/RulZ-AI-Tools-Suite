import { App } from '@capacitor/app'
import { Cloud, Info, Loader2 } from 'lucide-react'
import { useEffect, useMemo, useReducer } from 'react'
import { toast } from 'sonner'
import type { PhoneMode } from '@shared/phone'
import { applies, ModeController } from '../lib/mode'
import { isNative } from '../lib/native'
import { link } from '../lib/remote'
import { useLink } from '../stores/link'
import { Badge, Button } from './ui'

export function ModeLine({ mode, modelId }: { mode: PhoneMode; modelId: string | null }) {
  const computer = useLink((s) => s.currentId)
  const state = useLink((s) => s.state)
  const [, render] = useReducer((n: number) => n + 1, 0)
  // A controller belongs to this mounted screen and this exact connection target.
  const controller = useModeController(computer, mode, modelId, render)
  useEffect(() => {
    if (!controller || state.kind !== 'connected') { controller?.invalidate(); return }
    controller.reconnect()
    let visible = document.visibilityState !== 'hidden'
    let disposed = false
    let lastAsked = 0
    const tick = () => {
      if (disposed || !visible || controller.unsupported) return
      lastAsked = Date.now()
      void controller.refresh()
    }
    let timer: ReturnType<typeof setInterval> | undefined
    const startTimer = () => {
      if (timer) clearInterval(timer)
      timer = setInterval(() => {
        if (controller.unsupported) { clearInterval(timer); timer = undefined; return }
        if (Date.now() - lastAsked >= (controller.loading ? 3000 : 15000) - 250) tick()
      }, 3000)
    }
    const visibility = (active: boolean) => {
      visible = active
      if (timer) clearInterval(timer)
      timer = undefined
      if (active) { tick(); startTimer() }
    }
    const onVisibility = () => visibility(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', onVisibility)
    let removeNative: (() => void) | undefined
    if (isNative) void App.addListener('appStateChange', ({ isActive }) => visibility(isActive)).then((h) => { if (disposed) void h.remove(); else removeNative = () => void h.remove() })
    const offWriting = link.on('writing:job', (job) => { if (visible && ['done', 'error', 'stopped'].includes(job.status)) tick() })
    const offAdventure = link.on('adventure:job', (job) => { if (visible && ['done', 'error', 'stopped'].includes(job.status)) tick() })
    if (visible) { tick(); startTimer() }
    return () => { disposed = true; controller.invalidate(); if (timer) clearInterval(timer); document.removeEventListener('visibilitychange', onVisibility); removeNative?.(); offWriting(); offAdventure() }
  }, [controller, state.kind])
  const info = controller?.info
  if (!controller || !info || !info.modelId) return null
  const details = applies(info)
  return <div className="shrink-0 space-y-2 border-b border-border bg-panel px-4 py-2 text-[12px] text-muted">
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1"><span className="min-w-0 break-words">{info.name}</span>{info.rental && <Badge><Cloud size={12} /> Rental</Badge>}{info.rental && info.rental.label === null && <span>Not loaded on a rental</span>}{info.rental?.reconnecting && <span>Reconnecting</span>}{details && <span>{details}</span>}</div>
    {info.rental && <p>Your text is sent to the rented computer.</p>}
    <div aria-live="polite">
    {controller.loading && <p>Loading the model…</p>}
    {info.reload && !info.rental && !controller.isKept && <div className="space-y-2 rounded-lg border border-border bg-panel-2 p-3">
      <p className="flex gap-2"><Info size={16} className="shrink-0" /><span>{info.name} runs with other settings than its {info.reload.label}.</span></p>
      <div className="flex flex-wrap gap-2"><Button size="sm" aria-busy={controller.busy} disabled={controller.busy || !!info.reload.blocked} onClick={() => void controller.reload().then((error) => { if (error) toast.error(error) })}>{controller.busy && <Loader2 size={14} className="animate-spin" />} Reload</Button><Button size="sm" variant="ghost" onClick={() => controller.keep()}>Keep as is</Button></div>
      {info.reload.blocked && <p>{info.reload.blocked}</p>}
    </div>}
    </div>
  </div>
}

function useModeController(computer: string | null, mode: PhoneMode, modelId: string | null, render: () => void): ModeController | null {
  return useMemo(() => computer ? new ModeController(computer, mode, modelId, () => link.invoke('writing:mode', mode, modelId), (id) => link.invoke('writing:reloadMode', mode, id), render) : null, [computer, mode, modelId, render])
}
