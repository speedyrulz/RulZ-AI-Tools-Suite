import { Check, Copy, FastForward, FileText, GitBranch, Pencil, RotateCcw, Trash2, TriangleAlert } from 'lucide-react'
import { memo, useEffect, useRef, useState } from 'react'
import type { ChatMessage, GenerationStats } from '@shared/types'
import { cn, formatBytes, formatDuration, formatNumber } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { Button, IconButton, ProgressBar, Textarea, Tooltip } from '../ui'
import { Markdown } from './Markdown'
import { splitThink, ThinkingBlock } from './ThinkingBlock'
import { ToolCallBlock } from './ToolCalls'

const STOP_REASONS: Record<string, string> = {
  stop: 'Finished',
  length: 'Hit the token limit',
  tool_calls: 'Called tools',
  stopped: 'Stopped',
  error: 'Error'
}

function StatsLine({ stats }: { stats: GenerationStats }) {
  const parts: [string, string][] = []
  if (stats.tokensPerSecond) parts.push(['Speed', `${stats.tokensPerSecond.toFixed(1)} tok/s`])
  if (stats.predictedTokens) parts.push(['Generated', `${formatNumber(stats.predictedTokens)} tokens`])
  if (stats.ttftMs !== undefined) parts.push(['Time to first token', formatDuration(stats.ttftMs)])
  if (stats.promptTokens) {
    parts.push(['Prompt', `${formatNumber(stats.promptTokens)} tokens${stats.cachedTokens ? ` (${formatNumber(stats.cachedTokens)} cached)` : ''}`])
  }
  if (stats.stopReason) parts.push(['Stop reason', STOP_REASONS[stats.stopReason] ?? stats.stopReason])
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-faint">
      {parts.map(([label, value]) => (
        <Tooltip key={label} content={label}>
          <span className="tabular-nums">{value}</span>
        </Tooltip>
      ))}
      {stats.modelName && <span className="truncate">{stats.modelName}</span>}
    </div>
  )
}

export const MessageItem = memo(function MessageItem({
  message,
  isLast,
  streaming,
  promptProgress
}: {
  message: ChatMessage
  isLast: boolean
  streaming: boolean
  promptProgress: number | null
}) {
  const [editing, setEditing] = useState(false)
  const [copied, setCopied] = useState(false)
  const expandThinking = useApp((s) => s.settings?.expandThinking ?? false)
  const keepThinkingOpen = useApp((s) => s.settings?.keepThinkingOpen ?? false)
  const busy = useChat((s) => !!s.generation)
  const { regenerate, continueMessage, deleteMessage, branchFrom } = useChat.getState()
  const isUser = message.role === 'user'

  let reasoning = message.reasoning ?? ''
  let content = message.content
  let reasoningOpen = streaming && !content
  if (!reasoning && !isUser) {
    const split = splitThink(content)
    if (split) {
      reasoning = split.reasoning
      content = split.content
      reasoningOpen = streaming && split.open
    }
  }

  const copy = (): void => {
    void navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  const actions = (
    <div className={cn('flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100', isUser && 'justify-end')}>
      <IconButton size="sm" label={copied ? 'Copied' : 'Copy'} onClick={copy}>
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </IconButton>
      <IconButton size="sm" label="Edit" onClick={() => setEditing(true)} disabled={streaming}>
        <Pencil size={13} />
      </IconButton>
      <IconButton size="sm" label={isUser ? 'Resend' : 'Regenerate'} onClick={() => void regenerate(message.id)} disabled={busy}>
        <RotateCcw size={13} />
      </IconButton>
      {!isUser && isLast && (
        <IconButton size="sm" label="Continue" onClick={() => void continueMessage(message.id)} disabled={busy}>
          <FastForward size={13} />
        </IconButton>
      )}
      <IconButton size="sm" label="Branch into a new chat" onClick={() => void branchFrom(message.id)} disabled={streaming}>
        <GitBranch size={13} />
      </IconButton>
      <IconButton size="sm" label="Delete" onClick={() => deleteMessage(message.id)} disabled={streaming}>
        <Trash2 size={13} />
      </IconButton>
    </div>
  )

  if (editing) {
    return <EditBox message={message} onClose={() => setEditing(false)} />
  }

  if (isUser) {
    return (
      <div className="group flex flex-col items-end gap-1">
        {message.attachments?.length ? (
          <div className="flex max-w-[85%] flex-wrap justify-end gap-2">
            {message.attachments.map((a) =>
              a.kind === 'image' && a.dataUrl ? (
                <img key={a.id} src={a.dataUrl} alt={a.name} className="max-h-48 max-w-64 rounded-lg border border-border object-cover" />
              ) : (
                <div key={a.id} className="flex items-center gap-1.5 rounded-lg border border-border bg-panel-2 px-2.5 py-1.5 text-xs">
                  <FileText size={13} className="text-muted" />
                  {a.name}
                  <span className="text-faint">{formatBytes(a.sizeBytes)}</span>
                </div>
              )
            )}
          </div>
        ) : null}
        {message.content && (
          <div className="max-w-[85%] rounded-2xl rounded-br-md bg-panel-2 px-4 py-2.5 whitespace-pre-wrap">{message.content}</div>
        )}
        {actions}
      </div>
    )
  }

  const waiting = streaming && !content && !reasoning && !message.toolCalls?.length
  return (
    <div className="group">
      {reasoning && (
        <ThinkingBlock
          reasoning={reasoning}
          streaming={reasoningOpen}
          durationMs={message.reasoningMs}
          defaultOpen={expandThinking}
          keepOpen={keepThinkingOpen}
        />
      )}
      {waiting &&
        (promptProgress !== null ? (
          <div className="flex items-center gap-2 text-xs text-muted">
            Processing prompt… {Math.round(promptProgress * 100)}%
            <ProgressBar value={promptProgress} className="w-40" />
          </div>
        ) : (
          <div className="flex gap-1 py-2">
            {[0, 1, 2].map((i) => (
              <span key={i} className="typing-dot size-1.5 rounded-full bg-muted" style={{ animationDelay: `${i * 0.18}s` }} />
            ))}
          </div>
        ))}
      {content && <Markdown text={content} />}
      {message.toolCalls?.map((c) => <ToolCallBlock key={c.id} call={c} />)}
      {message.error && (
        <div className="mt-2 flex items-start gap-2 rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
          <TriangleAlert size={14} className="mt-px shrink-0" />
          <span className="whitespace-pre-wrap">{message.error}</span>
        </div>
      )}
      {!streaming && message.stats && <StatsLine stats={message.stats} />}
      {!streaming && <div className="mt-1">{actions}</div>}
    </div>
  )
})

function EditBox({ message, onClose }: { message: ChatMessage; onClose: () => void }) {
  const [text, setText] = useState(message.content)
  const ref = useRef<HTMLTextAreaElement>(null)
  const editMessage = useChat((s) => s.editMessage)
  const busy = useChat((s) => !!s.generation)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.focus()
    el.setSelectionRange(el.value.length, el.value.length)
  }, [])
  return (
    <div className="space-y-2 rounded-xl border border-accent/40 bg-panel p-3">
      <Textarea
        ref={ref}
        value={text}
        rows={Math.min(16, Math.max(3, text.split('\n').length))}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onClose()
        }}
      />
      <div className="flex justify-end gap-2">
        <Button size="sm" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button
          size="sm"
          onClick={() => {
            void editMessage(message.id, text, false)
            onClose()
          }}
        >
          Save
        </Button>
        {message.role === 'user' && (
          <Button
            size="sm"
            variant="primary"
            disabled={busy}
            onClick={() => {
              void editMessage(message.id, text, true)
              onClose()
            }}
          >
            Save & send
          </Button>
        )}
      </div>
    </div>
  )
}
