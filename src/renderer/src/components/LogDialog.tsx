import { Copy } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { invoke } from '../lib/api'
import { useApps } from '../stores/apps'
import { Button, Dialog } from './ui'

export function LogDialog({ appId, title, onClose }: { appId: string; title: string; onClose: () => void }) {
  const lines = useApps((s) => s.logs[appId])
  const end = useRef<HTMLDivElement>(null)

  // After a restart the renderer has no log; the main process keeps the last one.
  useEffect(() => {
    if (useApps.getState().logs[appId]?.length) return
    void invoke('apps:log', appId).then((log) =>
      useApps.setState((s) => ({ logs: { ...s.logs, [appId]: log } }))
    )
  }, [appId])

  useEffect(() => {
    end.current?.scrollIntoView()
  }, [lines?.length])

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={title}
      className="w-[860px]"
      footer={
        <>
          <Button
            variant="ghost"
            onClick={() => {
              void navigator.clipboard.writeText((lines ?? []).join('\n'))
              toast.success('Log copied')
            }}
          >
            <Copy className="size-4" /> Copy
          </Button>
          <Button onClick={onClose}>Close</Button>
        </>
      }
    >
      <pre className="min-h-[200px] rounded-lg border border-border bg-code p-3 font-mono text-[12px] leading-relaxed whitespace-pre-wrap text-body">
        {lines?.length ? lines.join('\n') : 'Nothing logged yet.'}
        <div ref={end} />
      </pre>
    </Dialog>
  )
}
