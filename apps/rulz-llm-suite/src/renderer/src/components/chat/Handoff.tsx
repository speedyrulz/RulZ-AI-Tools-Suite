import { useEffect, useRef } from 'react'
import { HANDOFF_THRESHOLD, useChat } from '../../stores/chat'
import { useModels } from '../../stores/models'
import { Button, Dialog, ProgressBar } from '../ui'

/** Opens the handoff prompt once a chat passes the context threshold. */
export function useHandoffTrigger(): void {
  const current = useChat((s) => s.current)
  const tokenCount = useChat((s) => s.tokenCount)
  const busy = useChat((s) => !!s.generation || !!s.handoff || !!s.autoLoadingModelId)
  const open = useChat((s) => s.handoffConfirm)
  const setConfirm = useChat((s) => s.setHandoffConfirm)
  const ctx = useModels((s) => s.instances.find((i) => i.modelId === s.activeModelId && i.status === 'ready')?.contextLength)

  useEffect(() => {
    if (!current || busy || open || current.handoffPrompted || !ctx || tokenCount === null) return
    if (tokenCount / ctx >= HANDOFF_THRESHOLD) setConfirm('auto')
  }, [current, busy, open, ctx, tokenCount, setConfirm])
}

export function HandoffDialogs() {
  const confirm = useChat((s) => s.handoffConfirm)
  const handoff = useChat((s) => s.handoff)
  const tokenCount = useChat((s) => s.tokenCount)
  const { setHandoffConfirm, dismissHandoffPrompt, startHandoff, cancelHandoff } = useChat.getState()
  const ctx = useModels((s) => s.instances.find((i) => i.modelId === s.activeModelId && i.status === 'ready')?.contextLength)
  const preview = useRef<HTMLDivElement>(null)

  useEffect(() => {
    preview.current?.scrollTo({ top: preview.current.scrollHeight })
  }, [handoff?.text])

  const pct = ctx && tokenCount !== null ? Math.round((tokenCount / ctx) * 100) : null
  const closeConfirm = (): void => (confirm === 'auto' ? dismissHandoffPrompt() : setHandoffConfirm(null))

  return (
    <>
      <Dialog
        open={!!confirm && !handoff}
        onOpenChange={(o) => !o && closeConfirm()}
        title={confirm === 'auto' ? 'This chat is almost out of context' : 'Hand off to a new chat?'}
        description={
          confirm === 'auto'
            ? `It is using ${pct ?? '85+'}% of the model's context window. Older messages will start being dropped soon.`
            : 'Continue this conversation in a fresh chat.'
        }
        footer={
          <>
            <Button variant="ghost" onClick={closeConfirm}>
              {confirm === 'auto' ? 'Not now' : 'Cancel'}
            </Button>
            <Button variant="primary" onClick={() => void startHandoff()}>
              Hand off
            </Button>
          </>
        }
      >
        <div className="space-y-2 px-5 py-4 text-[13px] text-muted">
          <p>The model will write a detailed handoff summary of this chat. RulZ then opens a new chat with the same model, system prompt and sampling settings, and sends the summary there as the first message.</p>
          <p>This chat stays as it is. Attached images are not carried over.</p>
        </div>
      </Dialog>

      <Dialog
        open={!!handoff}
        onOpenChange={(o) => !o && cancelHandoff()}
        title={handoff?.status === 'error' ? 'Handoff failed' : 'Writing handoff…'}
        description={
          handoff?.status === 'loading'
            ? 'Loading the model…'
            : handoff?.status === 'writing'
              ? 'The model is summarizing this chat for the new one.'
              : undefined
        }
        className="w-[min(720px,calc(100vw-32px))]"
        footer={
          handoff?.status === 'error' ? (
            <>
              <Button variant="ghost" onClick={cancelHandoff}>
                Close
              </Button>
              <Button
                variant="primary"
                onClick={() => {
                  cancelHandoff()
                  void startHandoff()
                }}
              >
                Try again
              </Button>
            </>
          ) : (
            <Button variant="ghost" onClick={cancelHandoff}>
              Cancel
            </Button>
          )
        }
      >
        <div className="space-y-3 px-5 py-4">
          {handoff?.status === 'error' && <div className="rounded-md border border-danger/40 bg-danger/10 p-2 text-xs whitespace-pre-wrap text-danger">{handoff.error}</div>}
          {handoff?.promptProgress !== null && handoff?.promptProgress !== undefined && (
            <div className="flex items-center gap-2 text-xs text-muted">
              Reading the chat… {Math.round(handoff.promptProgress * 100)}%
              <ProgressBar value={handoff.promptProgress} className="flex-1" />
            </div>
          )}
          {handoff?.status === 'writing' && !handoff.text && handoff.promptProgress === null && (
            <div className="text-xs text-muted">Thinking…</div>
          )}
          {handoff?.text && (
            <div ref={preview} className="max-h-[45vh] overflow-y-auto rounded-lg border border-border bg-panel-2 p-3 text-[13px] whitespace-pre-wrap">
              {handoff.text}
            </div>
          )}
        </div>
      </Dialog>
    </>
  )
}
