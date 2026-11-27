import { Copy } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { call } from '../../lib/api'
import { TerminalBuffer } from '../../lib/terminal'
import { useApp } from '../../stores/app'
import { Button } from '../ui'

export function LogWell() {
  const term = useRef<TerminalBuffer | null>(null)
  const [text, setText] = useState('')
  const box = useRef<HTMLPreElement>(null)
  const stick = useRef(true)
  const frame = useRef(0)
  const status = useApp((s) => s.serverState?.status)
  const prevStatus = useRef(status)

  const render = (): void => {
    if (frame.current) return
    frame.current = requestAnimationFrame(() => {
      frame.current = 0
      setText(term.current?.text() ?? '')
    })
  }

  useEffect(() => {
    let alive = true
    let off: () => void = () => undefined
    const t = new TerminalBuffer(2000)
    term.current = t
    // Subscribe only after the snapshot is applied, so no chunk is shown twice. A chunk printed in
    // the instant between the snapshot and the subscription can be missed.
    void call('server:log').then((initial) => {
      if (!alive) return
      if (initial) t.write(initial)
      off = window.rulz.on('server:log', (e) => {
        t.write(e.text)
        render()
      })
      render()
    })
    return () => {
      alive = false
      off()
      if (frame.current) cancelAnimationFrame(frame.current)
      frame.current = 0
      term.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // A new start begins a new log.
  useEffect(() => {
    if (status === 'starting' && prevStatus.current !== 'starting') {
      term.current?.reset()
      setText('')
    }
    prevStatus.current = status
  }, [status])

  useLayoutEffect(() => {
    const el = box.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [text])

  const copy = (): void => {
    navigator.clipboard.writeText(term.current?.text() ?? '').then(
      () => toast.success('Log copied'),
      () => toast.error('Could not copy the log')
    )
  }

  return (
    <div>
      <pre
        ref={box}
        tabIndex={0}
        aria-label="Server log"
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
        }}
        className="h-72 overflow-auto rounded-lg border border-border bg-black/[0.03] p-3 font-mono text-xs leading-relaxed whitespace-pre text-body dark:bg-black/40"
      >
        {text || <span className="text-faint">The server log appears here once the own server starts.</span>}
      </pre>
      <div className="mt-2 flex justify-end">
        <Button variant="ghost" size="sm" onClick={copy}>
          <Copy size={14} />
          Copy log
        </Button>
      </div>
    </div>
  )
}
