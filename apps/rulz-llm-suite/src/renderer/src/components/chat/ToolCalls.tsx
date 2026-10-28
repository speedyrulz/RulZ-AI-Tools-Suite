import * as Collapsible from '@radix-ui/react-collapsible'
import { Ban, Check, ChevronRight, CircleX, LoaderCircle, Wrench } from 'lucide-react'
import { useState } from 'react'
import { META_LOAD, META_SEARCH, META_UNLOAD } from '@shared/tools'
import type { ToolCall } from '@shared/types'
import { cn } from '../../lib/format'
import { useChat, type ToolDecision } from '../../stores/chat'
import { findTool } from '../../stores/mcp'
import { Button, Dialog } from '../ui'

function prettyJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2)
  } catch {
    return text
  }
}

const META_LABELS: Record<string, string> = {
  [META_SEARCH]: 'Search tools',
  [META_LOAD]: 'Load tools',
  [META_UNLOAD]: 'Unload tools'
}

function toolLabel(qualifiedName: string): { tool: string; server: string | null } {
  if (META_LABELS[qualifiedName]) return { tool: META_LABELS[qualifiedName], server: 'RulZ' }
  const t = findTool(qualifiedName)
  if (t) return { tool: t.name, server: t.server }
  const i = qualifiedName.indexOf('__')
  return i > 0 ? { tool: qualifiedName.slice(i + 2), server: qualifiedName.slice(0, i) } : { tool: qualifiedName, server: null }
}

const STATUS = {
  pending: { icon: <LoaderCircle size={13} className="animate-spin" />, text: 'Waiting', cls: 'text-muted' },
  running: { icon: <LoaderCircle size={13} className="animate-spin" />, text: 'Running', cls: 'text-accent' },
  done: { icon: <Check size={13} />, text: 'Done', cls: 'text-ok' },
  error: { icon: <CircleX size={13} />, text: 'Error', cls: 'text-danger' },
  denied: { icon: <Ban size={13} />, text: 'Skipped', cls: 'text-faint' }
} as const

export function ToolCallBlock({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false)
  const { tool, server } = toolLabel(call.name)
  const s = STATUS[call.status]
  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="my-2 overflow-hidden rounded-lg border border-border bg-panel">
      <Collapsible.Trigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-hover/50">
        <Wrench size={13} className="text-accent" />
        <span className="font-medium">{tool}</span>
        {server && <span className="text-faint">· {server}</span>}
        <span className={cn('ml-auto flex items-center gap-1', s.cls)}>
          {s.icon}
          {s.text}
        </span>
        <ChevronRight size={13} className={cn('text-faint transition-transform', open && 'rotate-90')} />
      </Collapsible.Trigger>
      <Collapsible.Content className="space-y-2 border-t border-border px-3 py-2">
        <div>
          <div className="mb-1 text-[11px] text-faint">Arguments</div>
          <pre className="max-h-48 overflow-auto rounded bg-code p-2 font-mono text-[11.5px] whitespace-pre-wrap">{prettyJson(call.arguments) || '{}'}</pre>
        </div>
        {call.result !== undefined && (
          <div>
            <div className="mb-1 text-[11px] text-faint">Result</div>
            <pre className={cn('max-h-72 overflow-auto rounded bg-code p-2 font-mono text-[11.5px] whitespace-pre-wrap', call.status === 'error' && 'text-danger')}>
              {call.result}
            </pre>
          </div>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

/** Asks the user before a tool runs. */
export function ToolApprovalDialog() {
  const approval = useChat((s) => s.toolApproval)
  if (!approval) return null
  const { tool, server } = toolLabel(approval.call.name)
  const decide = (d: ToolDecision): void => approval.resolve(d)
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && decide('deny')}
      title="Allow tool call?"
      description={server ? `The model wants to run "${tool}" from the ${server} MCP server.` : `The model wants to run "${tool}".`}
      footer={
        <>
          <Button variant="ghost" onClick={() => decide('deny')}>
            Deny
          </Button>
          <Button onClick={() => decide('always')}>Always allow this tool</Button>
          <Button variant="primary" onClick={() => decide('once')} autoFocus>
            Allow once
          </Button>
        </>
      }
    >
      <div className="px-5 py-4">
        <div className="mb-1 text-[11px] text-faint">Arguments</div>
        <pre className="max-h-[40vh] overflow-auto rounded-lg border border-border bg-code p-3 font-mono text-xs whitespace-pre-wrap">
          {prettyJson(approval.call.arguments) || '{}'}
        </pre>
      </div>
    </Dialog>
  )
}
