import { ArrowUp, Clock, CornerDownRight, FileText, LoaderCircle, Paperclip, Square, X } from 'lucide-react'
import { forwardRef, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import type { Attachment } from '@shared/types'
import { filesToAttachments } from '../../lib/attachments'
import { formatBytes } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { modelById, useModels } from '../../stores/models'
import { IconButton, Kbd } from '../ui'

export interface ComposerHandle {
  addFiles: (files: File[]) => Promise<void>
  focus: () => void
}

export const Composer = forwardRef<ComposerHandle>(function Composer(_props, ref) {
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const textRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const generating = useChat((s) => !!s.generation)
  /** The reply streaming is in this chat, so a note written now can go to it. */
  const replyingHere = useChat((s) => !!s.generation && s.generation.conversationId === s.current?.id)
  const allQueued = useChat((s) => s.queued)
  const currentId = useChat((s) => s.current?.id)
  const queued = allQueued.filter((q) => q.conversationId === currentId)
  const autoLoading = useChat((s) => s.autoLoadingModelId)
  const conversationModelId = useChat((s) => s.current?.modelId ?? null)
  const send = useChat((s) => s.send)
  const sendNow = useChat((s) => s.sendNow)
  const queueNote = useChat((s) => s.queueNote)
  const cancelQueued = useChat((s) => s.cancelQueued)
  const sendQueued = useChat((s) => s.sendQueued)
  const stop = useChat((s) => s.stop)
  const activeModelId = useModels((s) => s.activeModelId)
  const instances = useModels((s) => s.instances)
  const openPicker = useApp((s) => s.openPicker)

  const addFiles = async (files: File[]): Promise<void> => {
    const { attachments: added, errors } = await filesToAttachments(files)
    if (errors.length) toast.error(errors.join('\n'))
    if (added.length) setAttachments((a) => [...a, ...added])
  }

  useImperativeHandle(ref, () => ({ addFiles, focus: () => textRef.current?.focus() }))

  useLayoutEffect(() => {
    const el = textRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, window.innerHeight * 0.4)}px`
  }, [text])

  const hasInput = text.trim().length > 0 || attachments.length > 0
  const canSend = hasInput && !autoLoading && (!generating || replyingHere)

  /** While a reply streams here: "now" stops it and sends the note, "later" queues the note for when it ends. */
  const submit = (when: 'now' | 'later' = 'now'): void => {
    if (!canSend) return
    if (!replyingHere) void send(text.trim(), attachments)
    else if (when === 'now') sendNow(text.trim(), attachments)
    else queueNote(text.trim(), attachments)
    setText('')
    setAttachments([])
  }

  const targetId = activeModelId ?? conversationModelId
  const inst = instances.find((i) => i.modelId === targetId)
  const model = modelById(targetId)
  let status: React.ReactNode
  if (autoLoading) status = <>Loading {modelById(autoLoading)?.name ?? 'model'}…</>
  else if (inst?.status === 'ready') status = <>Chatting with <span className="text-muted">{inst.modelName}</span></>
  else if (inst?.status === 'loading') status = <>Loading {inst.modelName}… {Math.round(inst.progress * 100)}%</>
  else if (model) status = <>Sending will load <span className="text-muted">{model.name}</span> with its saved settings</>
  else
    status = (
      <button className="hover:text-text" onClick={() => openPicker()}>
        No model loaded — press <Kbd>Ctrl+L</Kbd> to choose one
      </button>
    )

  return (
    <div className="mx-auto w-full max-w-3xl px-6 pb-4">
      {queued.length > 0 && (
        <div className="mb-2 flex flex-col gap-1.5">
          {queued.map((q) => (
            <div key={q.id} className="flex items-center gap-2 rounded-xl border border-dashed border-border bg-panel-2 py-1 pr-1 pl-3 text-xs">
              <Clock size={13} className="shrink-0 text-faint" />
              <span className="shrink-0 text-faint">{generating ? 'Sends after this reply' : 'Waiting to send'}</span>
              <span className="min-w-0 flex-1 truncate">{q.text || q.attachments.map((a) => a.name).join(', ')}</span>
              <IconButton size="sm" label="Send now" onClick={() => sendQueued(q.id)} disabled={!!autoLoading}>
                <CornerDownRight size={13} />
              </IconButton>
              <IconButton size="sm" label="Cancel" onClick={() => cancelQueued(q.id)}>
                <X size={13} />
              </IconButton>
            </div>
          ))}
        </div>
      )}
      <div className="rounded-2xl border border-border bg-panel shadow-sm focus-within:border-accent/60">
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 px-3 pt-3">
            {attachments.map((a) => (
              <div key={a.id} className="group relative flex items-center gap-2 rounded-lg border border-border bg-panel-2 py-1 pr-7 pl-1 text-xs">
                {a.kind === 'image' && a.dataUrl ? (
                  <img src={a.dataUrl} alt={a.name} className="size-8 rounded object-cover" />
                ) : (
                  <FileText size={15} className="ml-1 text-muted" />
                )}
                <div className="max-w-40">
                  <div className="truncate">{a.name}</div>
                  <div className="text-[10px] text-faint">{formatBytes(a.sizeBytes)}</div>
                </div>
                <button
                  aria-label={`Remove ${a.name}`}
                  className="absolute top-1 right-1 rounded p-0.5 text-faint hover:bg-hover hover:text-text"
                  onClick={() => setAttachments((list) => list.filter((x) => x.id !== a.id))}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
        <textarea
          ref={textRef}
          rows={1}
          value={text}
          placeholder={replyingHere ? 'Add a note while it replies…' : 'Send a message…'}
          className="block max-h-[40vh] w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[14px] leading-relaxed placeholder:text-faint focus:outline-none focus-visible:outline-none"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit(e.ctrlKey || e.metaKey ? 'later' : 'now')
            }
          }}
          onPaste={(e) => {
            const files = [...e.clipboardData.files]
            if (files.length) {
              e.preventDefault()
              void addFiles(files)
            }
          }}
        />
        <div className="flex items-center gap-1 px-2 pb-2">
          <IconButton label="Attach images or text files" onClick={() => fileRef.current?.click()}>
            <Paperclip size={16} />
          </IconButton>
          <input
            ref={fileRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              void addFiles([...(e.target.files ?? [])])
              e.target.value = ''
            }}
          />
          <span className="min-w-0 flex-1 truncate px-1 text-[11px] text-faint">{status}</span>
          {replyingHere && hasInput && (
            <>
              <IconButton label="Send after the reply (Ctrl+Enter)" onClick={() => submit('later')}>
                <Clock size={16} />
              </IconButton>
              <IconButton
                label="Send now: stop the reply, keep its thinking, and continue with your note (Enter)"
                onClick={() => submit('now')}
                className="bg-accent text-accent-fg hover:bg-accent-hover hover:text-accent-fg"
              >
                <CornerDownRight size={16} />
              </IconButton>
            </>
          )}
          {generating ? (
            <IconButton label="Stop (Esc)" onClick={stop} className="bg-text text-bg hover:bg-text/85 hover:text-bg">
              <Square size={13} fill="currentColor" />
            </IconButton>
          ) : (
            <IconButton
              label="Send (Enter)"
              onClick={() => submit()}
              disabled={!canSend}
              className="bg-accent text-accent-fg hover:bg-accent-hover hover:text-accent-fg"
            >
              {autoLoading ? <LoaderCircle size={16} className="animate-spin" /> : <ArrowUp size={17} />}
            </IconButton>
          )}
        </div>
      </div>
    </div>
  )
})
