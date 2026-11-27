import { ChevronDown, ShieldAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { cn } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Markdown } from './Markdown'
import { Button, Sheet } from './ui'

/** The buttons ignore presses this long after a question appears or changes, so a tap meant for something else can't answer it. */
export const ARM_MS = 600

/**
 * Wraps `decide` so presses are ignored until ARM_MS after `arm()` (called for every new prompt).
 * Kept apart from React so it can be tested with fake timers.
 */
export function createArming(onArmed: (armed: boolean) => void, ms = ARM_MS): { arm: () => void; cancel: () => void; isArmed: () => boolean } {
  let armed = false
  let timer: ReturnType<typeof setTimeout> | null = null
  const cancel = (): void => {
    if (timer) clearTimeout(timer)
    timer = null
  }
  return {
    arm: () => {
      cancel()
      armed = false
      onArmed(false)
      timer = setTimeout(() => {
        timer = null
        armed = true
        onArmed(true)
      }, ms)
    },
    cancel,
    isArmed: () => armed
  }
}

/** "Is this OK?" before a command or a risky step, answered from the phone. */
export function ApprovalSheet() {
  const prompt = useWorkspace((s) => s.prompt)
  const answer = useWorkspace((s) => s.answer)
  const [details, setDetails] = useState(false)
  const [armed, setArmed] = useState(false)
  const arming = useRef<ReturnType<typeof createArming> | null>(null)
  // Which prompt the arming belongs to: until the effect below has re-armed for a new prompt, a press
  // must not count with the previous prompt's arming.
  const armedFor = useRef<string | undefined>(undefined)
  useEffect(() => {
    setDetails(false)
    const a = (arming.current = createArming(setArmed))
    armedFor.current = prompt?.id
    a.arm()
    return a.cancel
  }, [prompt?.id])

  const open = prompt?.kind === 'approval'
  const req = open ? prompt : null
  const decide = (decision: 'allow' | 'allow-task' | 'deny'): void => {
    if (req && armedFor.current === req.id && arming.current?.isArmed()) void answer(req.id, { kind: 'approval', decision })
  }
  const dim = !armed && 'opacity-50'

  return (
    <Sheet
      open={open}
      // Back button or swipe: leave it unanswered (it stays on the computer too) rather than refusing by accident.
      onOpenChange={() => undefined}
      dismissable={false}
      title="Is this OK?"
      footer={
        <div className="grid w-full grid-cols-2 gap-2">
          <Button aria-disabled={!armed} className={cn(dim)} onClick={() => decide('deny')}>Don’t allow</Button>
          <Button aria-disabled={!armed} className={cn(dim)} onClick={() => decide('allow-task')}>Allow for this task</Button>
          <Button variant="primary" size="lg" aria-disabled={!armed} className={cn('col-span-2', dim)} onClick={() => decide('allow')}>
            Allow
          </Button>
        </div>
      }
    >
      {req && (
        <div className="space-y-4 px-4 pb-4">
          <p className="text-[17px] font-medium">{req.action}</p>
          {req.risk && (
            <p className="flex items-start gap-2 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-[14px] text-warn">
              <ShieldAlert size={18} className="mt-px shrink-0" /> {req.risk}
            </p>
          )}
          <div>
            <div className="mb-1 text-xs font-medium tracking-wide text-faint uppercase">Why</div>
            <Markdown text={req.why} className="text-[14px] text-muted" />
          </div>
          <div>
            <button className="flex items-center gap-1 py-1 text-[14px] text-muted" onClick={() => setDetails(!details)}>
              <ChevronDown size={16} className={cn('transition-transform', !details && '-rotate-90')} />
              Show details
            </button>
            {details && (
              <pre className="mt-2 max-h-72 overflow-auto rounded-lg border border-border bg-code p-3 font-mono text-[12px] whitespace-pre-wrap">{req.detail}</pre>
            )}
          </div>
        </div>
      )}
    </Sheet>
  )
}
