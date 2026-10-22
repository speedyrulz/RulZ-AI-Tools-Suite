import { ArrowDown, Forward, PanelRight, PanelRightClose, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { cn, formatCompact, formatTokens } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { useModels } from '../../stores/models'
import { LogoWordmark } from '../Logo'
import { IconButton, Tooltip } from '../ui'
import { Composer, type ComposerHandle } from './Composer'
import { HandoffDialogs, useHandoffTrigger } from './Handoff'
import { ToolApprovalDialog } from './ToolCalls'
import { MessageItem } from './MessageItem'

export function ChatView() {
  const current = useChat((s) => s.current)
  const generation = useChat((s) => s.generation)
  const rename = useChat((s) => s.rename)
  const setHandoffConfirm = useChat((s) => s.setHandoffConfirm)
  const handoffBusy = useChat((s) => s.handoff)
  useHandoffTrigger()
  const panelOpen = useApp((s) => s.settings?.rightPanelOpen ?? true)
  const updateSettings = useApp((s) => s.updateSettings)
  const composer = useRef<ComposerHandle>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)
  const [showJump, setShowJump] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [editingTitle, setEditingTitle] = useState(false)

  const streamingId = generation && generation.conversationId === current?.id ? generation.messageId : null

  // Keep the view pinned to the bottom while streaming unless the user scrolled up.
  useEffect(() => {
    const el = scroller.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [current?.messages])

  useEffect(() => {
    pinned.current = true
    const el = scroller.current
    if (el) el.scrollTop = el.scrollHeight
    composer.current?.focus()
  }, [current?.id])

  if (!current) {
    // Typing here starts a new chat, so the composer is always available.
    return (
      <div className="flex h-full flex-col">
        <div className="min-h-0 flex-1">
          <EmptyState />
        </div>
        <Composer ref={composer} />
      </div>
    )
  }

  return (
    <div
      className="relative flex h-full min-w-0 flex-col"
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes('Files')) {
          e.preventDefault()
          setDragging(true)
        }
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        void composer.current?.addFiles([...e.dataTransfer.files])
      }}
    >
      <div className="flex h-12 shrink-0 items-center gap-3 border-b border-border px-5">
        {editingTitle ? (
          <input
            autoFocus
            defaultValue={current.title}
            className="h-8 flex-1 rounded-md border border-accent bg-panel-2 px-2 text-sm focus:outline-none"
            onBlur={(e) => {
              void rename(current.id, e.target.value)
              setEditingTitle(false)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur()
              if (e.key === 'Escape') setEditingTitle(false)
            }}
          />
        ) : (
          <button className="min-w-0 truncate text-sm font-medium hover:text-accent" onClick={() => setEditingTitle(true)} title="Rename">
            {current.title}
          </button>
        )}
        <div className="flex-1" />
        <ContextMeter />
        <IconButton
          label="Hand off to a new chat"
          onClick={() => setHandoffConfirm('manual')}
          disabled={!current.messages.length || !!generation || !!handoffBusy}
        >
          <Forward size={17} />
        </IconButton>
        <IconButton
          label={panelOpen ? 'Hide chat settings (Ctrl+B)' : 'Show chat settings (Ctrl+B)'}
          onClick={() => void updateSettings({ rightPanelOpen: !panelOpen })}
        >
          {panelOpen ? <PanelRightClose size={17} /> : <PanelRight size={17} />}
        </IconButton>
      </div>

      <div
        ref={scroller}
        className="min-h-0 flex-1 overflow-y-auto"
        onScroll={(e) => {
          const el = e.currentTarget
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
          pinned.current = atBottom
          setShowJump(!atBottom)
        }}
      >
        {current.messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mx-auto max-w-3xl space-y-6 px-6 py-6">
            {current.messages.map((m, i) => (
              <MessageItem
                key={m.id}
                message={m}
                isLast={i === current.messages.length - 1}
                streaming={m.id === streamingId}
                promptProgress={m.id === streamingId ? (generation?.promptProgress ?? null) : null}
              />
            ))}
          </div>
        )}
      </div>

      {showJump && (
        <button
          className="absolute bottom-32 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border border-border bg-panel px-3 py-1 text-xs text-muted shadow-lg hover:text-text"
          onClick={() => {
            const el = scroller.current
            if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
          }}
        >
          <ArrowDown size={13} /> Jump to latest
        </button>
      )}

      <Composer ref={composer} />
      <HandoffDialogs />
      <ToolApprovalDialog />

      {dragging && (
        <div className="pointer-events-none absolute inset-3 flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-accent bg-bg/85 text-accent">
          <Upload size={28} />
          <div className="text-sm font-medium">Drop images or text files to attach</div>
        </div>
      )}
    </div>
  )
}

function ContextMeter() {
  const tokenCount = useChat((s) => s.tokenCount)
  const refresh = useChat((s) => s.refreshTokenCount)
  const activeModelId = useModels((s) => s.activeModelId)
  const inst = useModels((s) => s.instances.find((i) => i.modelId === s.activeModelId && i.status === 'ready'))
  const ctx = inst?.contextLength ?? null

  useEffect(() => {
    refresh()
  }, [activeModelId, inst?.instanceId, refresh])

  if (!ctx) return null
  const used = tokenCount ?? 0
  const frac = Math.min(1, used / ctx)
  return (
    <Tooltip content={`${used.toLocaleString()} of ${ctx.toLocaleString()} context tokens used`}>
      <div className="flex items-center gap-2 text-[11px] text-muted tabular-nums">
        <div className="h-1.5 w-24 overflow-hidden rounded-full bg-hover">
          <div className={cn('h-full rounded-full', frac > 0.9 ? 'bg-danger' : frac > 0.7 ? 'bg-warn' : 'bg-accent')} style={{ width: `${frac * 100}%` }} />
        </div>
        {tokenCount === null ? '—' : formatCompact(used)} / {formatTokens(ctx)}
      </div>
    </Tooltip>
  )
}

function EmptyState() {
  const activeModelId = useModels((s) => s.activeModelId)
  const models = useModels((s) => s.models)
  const openPicker = useApp((s) => s.openPicker)
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="flex flex-col items-center gap-2 select-none">
        <LogoWordmark className="w-72" />
        <div className="text-xs font-semibold tracking-[0.35em] text-muted uppercase">LLM Suite</div>
      </div>
      <div>
        <div className="mt-1 text-sm text-muted">
          {activeModelId
            ? 'Ask anything. Drop images or files to attach them.'
            : models.length
              ? `${models.length} local models ready.`
              : 'No models found yet.'}
        </div>
      </div>
      {!activeModelId && (
        <button className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-fg hover:brightness-110" onClick={() => openPicker()}>
          Load a model <span className="opacity-70">(Ctrl+L)</span>
        </button>
      )}
    </div>
  )
}
