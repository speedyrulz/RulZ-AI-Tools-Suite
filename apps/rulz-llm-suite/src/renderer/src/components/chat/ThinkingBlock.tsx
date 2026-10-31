import * as Collapsible from '@radix-ui/react-collapsible'
import { Brain, ChevronRight } from 'lucide-react'
import { useEffect, useState } from 'react'
import { formatDuration } from '../../lib/format'
import { openAfterThinking } from '../../lib/thinking'

export { splitThink } from '../../lib/thinking'
import { Markdown } from './Markdown'

export function ThinkingBlock({
  reasoning,
  streaming,
  durationMs,
  defaultOpen,
  keepOpen
}: {
  reasoning: string
  /** True while the model is still producing reasoning. */
  streaming: boolean
  durationMs?: number
  defaultOpen: boolean
  /** Stay open after the model finishes thinking. */
  keepOpen: boolean
}) {
  const [open, setOpen] = useState(defaultOpen || streaming)
  const [startedAt] = useState(() => Date.now())
  const [, tick] = useState(0)

  useEffect(() => {
    if (!streaming) return
    const t = setInterval(() => tick((n) => n + 1), 500)
    return () => clearInterval(t)
  }, [streaming])

  useEffect(() => {
    if (!streaming) setOpen((o) => openAfterThinking(o, { defaultOpen, keepOpen }))
  }, [streaming, defaultOpen, keepOpen])

  const label = streaming
    ? `Thinking… ${formatDuration(Date.now() - startedAt)}`
    : durationMs
      ? `Thought for ${formatDuration(durationMs)}`
      : 'Reasoning'

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="mb-3">
      <Collapsible.Trigger className="flex items-center gap-1.5 rounded-md py-0.5 text-xs text-muted hover:text-text">
        <Brain size={13} className={streaming ? 'animate-pulse text-accent' : undefined} />
        {label}
        <ChevronRight size={13} className={open ? 'rotate-90 transition-transform' : 'transition-transform'} />
      </Collapsible.Trigger>
      <Collapsible.Content>
        <div className="mt-2 border-l-2 border-border pl-3 text-[13px] text-muted">
          <Markdown text={reasoning || '…'} live={streaming} />
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  )
}
