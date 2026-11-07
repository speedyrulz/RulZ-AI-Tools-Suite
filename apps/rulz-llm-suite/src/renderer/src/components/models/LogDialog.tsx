import { Copy } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { invoke } from '../../lib/api'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { Button, Dialog } from '../ui'

const NO_LINES: string[] = []

/** Live llama-server output for one instance (polled while open). */
export function LogDialog() {
  const id = useApp((s) => s.logInstanceId)
  const showLog = useApp((s) => s.showLog)
  const inst = useModels((s) => s.instances.find((i) => i.instanceId === id))
  // Kept with the id it belongs to, so opening another model's log never shows the previous one.
  const [log, setLog] = useState<{ id: string | null; lines: string[] }>({ id: null, lines: [] })
  const lines = log.id === id ? log.lines : NO_LINES
  const bottom = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useEffect(() => {
    if (!id) return
    stick.current = true
    let alive = true
    const tick = async (): Promise<void> => {
      const l = await invoke('instances:log', id).catch(() => null)
      if (alive && l && l.length) setLog({ id, lines: l })
    }
    void tick()
    const t = setInterval(tick, 1000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [id])

  useEffect(() => {
    if (stick.current) bottom.current?.scrollIntoView({ block: 'end' })
  }, [lines])

  return (
    <Dialog
      open={!!id}
      onOpenChange={(o) => !o && showLog(null)}
      title={`Server log${inst ? ` — ${inst.modelName}` : ''}`}
      description={inst ? `llama-server on 127.0.0.1:${inst.port} · PID ${inst.pid ?? '?'}` : 'This instance has stopped; showing its last output.'}
      className="w-[min(1000px,calc(100vw-32px))]"
      footer={
        <Button
          size="sm"
          onClick={() => {
            void navigator.clipboard.writeText(lines.join('\n'))
            toast.success('Log copied')
          }}
        >
          <Copy size={13} /> Copy
        </Button>
      }
    >
      <div
        className="h-[60vh] overflow-auto bg-code px-4 py-3 font-mono text-[11.5px] leading-relaxed whitespace-pre text-muted"
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
        }}
      >
        {lines.map((l, i) => (
          <div key={i} className={/error|failed/i.test(l) ? 'text-danger' : /warn/i.test(l) ? 'text-warn' : undefined}>
            {l}
          </div>
        ))}
        <div ref={bottom} />
      </div>
    </Dialog>
  )
}
