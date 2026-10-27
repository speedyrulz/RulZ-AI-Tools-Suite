import { ChevronDown, ShieldAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { cn } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Markdown } from './Markdown'
import { Button, Dialog } from './ui'

export function ApprovalDialog() {
  const prompt = useWorkspace((s) => s.prompt)
  const answer = useWorkspace((s) => s.answer)
  const [details, setDetails] = useState(false)
  useEffect(() => setDetails(false), [prompt?.id])

  const open = prompt?.kind === 'approval'
  const req = open ? prompt : null
  const decide = (decision: 'allow' | 'allow-task' | 'deny'): void => answer({ kind: 'approval', decision })

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && decide('deny')}
      title="Is this OK?"
      className="w-[min(600px,calc(100vw-32px))]"
      keepOnFocusOutside
      footer={
        <>
          <Button className="h-10 px-4 text-sm" onClick={() => decide('deny')}>
            Don’t allow
          </Button>
          <Button className="h-10 px-4 text-sm" onClick={() => decide('allow-task')}>
            Allow for this task
          </Button>
          <Button variant="primary" className="h-10 px-5 text-sm" onClick={() => decide('allow')} autoFocus>
            Allow
          </Button>
        </>
      }
    >
      {req && (
        <div className="space-y-4 px-5 py-4">
          <p className="text-[16px] font-medium">{req.action}</p>
          {req.risk && (
            <p className="flex items-start gap-2 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-[13px] text-warn">
              <ShieldAlert size={16} className="mt-px shrink-0" /> {req.risk}
            </p>
          )}
          <div>
            <div className="mb-1 text-xs font-medium tracking-wide text-faint uppercase">Why</div>
            <Markdown text={req.why} className="text-[13px] text-muted" />
          </div>
          <div>
            <button className="flex items-center gap-1 text-[13px] text-muted hover:text-text" onClick={() => setDetails(!details)}>
              <ChevronDown size={14} className={cn('transition-transform', !details && '-rotate-90')} />
              Show details
            </button>
            {details && (
              <pre className="mt-2 max-h-72 overflow-auto rounded-lg border border-border bg-code p-3 font-mono text-[12px] whitespace-pre-wrap">
                {req.detail}
              </pre>
            )}
          </div>
        </div>
      )}
    </Dialog>
  )
}
