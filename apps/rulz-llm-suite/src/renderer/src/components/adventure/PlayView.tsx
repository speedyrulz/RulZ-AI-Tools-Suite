import { ArrowLeft, Download, Eye, MoreHorizontal, PanelRightClose, PanelRightOpen, Pencil, Shuffle, Signpost, Square } from 'lucide-react'
import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { childForChoice, pathTo, type Adventure, type AdventureNode } from '@shared/adventure'
import { isRunning, jobStatusText, liveView, stepLine } from '../../lib/adventure'
import { cn } from '../../lib/format'
import { useAdventure } from '../../stores/adventure'
import { useModels } from '../../stores/models'
import { ModeReloadBar } from '../models/ModeReloadBar'
import { Markdown } from '../chat/Markdown'
import { Button, DropdownMenu, IconButton, Kbd, ProgressBar, Textarea } from '../ui'
import { ConfirmDialog } from '../writing/common'
import { EditDialog } from './EditDialog'
import { PassageBlock, type PassageActions } from './PassageBlock'
import { FOCUS } from './parts'
import { AdventureSidePanel } from './SidePanel'

/** True when a key press is meant for a text field or a dialog, not for the choice shortcuts. */
function typingTarget(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null
  if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return true
  return !!document.querySelector('[role="dialog"], [role="alertdialog"], [role="menu"]')
}

function StepLine({ text }: { text: string }) {
  if (!text) return null
  return <p className="my-5 border-l-2 border-border pl-3 text-[13px] text-muted italic">{text}</p>
}

export function PlayView() {
  const adv = useAdventure((s) => s.adv) as Adventure
  const position = useAdventure((s) => s.position)
  const job = useAdventure((s) => s.job)
  const remoteNodeId = useAdventure((s) => s.remoteNodeId)
  const panelOpen = useAdventure((s) => s.panelOpen)
  const activeModelId = useModels((s) => s.activeModelId)
  const store = useAdventure.getState()
  const [editId, setEditId] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<{ kind: 'choices' | 'delete'; id: string } | null>(null)
  const [typed, setTyped] = useState('')
  const endRef = useRef<HTMLDivElement>(null)

  const running = isRunning(job)
  const live = liveView(job, position)
  const node: AdventureNode | undefined = position ? adv.nodes[position] : undefined
  const path = useMemo(() => {
    const p = pathTo(adv, position)
    return live.kind === 'passage' && live.replaces ? p.filter((n) => n.id !== live.replaces) : p
  }, [adv, position, live.kind === 'passage' ? live.replaces : null])

  // Keep the newest text in view while it is written, and when the position changes.
  const streamLength = live.kind === 'passage' ? live.job.text.length : 0
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'end' })
  }, [position, streamLength, live.kind])

  const actions: PassageActions = useMemo(
    () => ({
      onGoHere: (id) => useAdventure.getState().go(id),
      onSwitch: (id) => useAdventure.getState().go(id),
      onTryAgain: (id) => void useAdventure.getState().retry(id, 'passage'),
      onNewChoices: (id) => {
        const n = useAdventure.getState().adv?.nodes[id]
        if (n?.edited) setConfirm({ kind: 'choices', id })
        else void useAdventure.getState().retry(id, 'choices')
      },
      onEdit: (id) => setEditId(id),
      onDeleteBranch: (id) => setConfirm({ kind: 'delete', id })
    }),
    []
  )

  const canChoose = !!node && !running && !node.ended

  // Keys 1-4 pick a choice (not while typing in a field or with a dialog open).
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.ctrlKey || e.metaKey || e.altKey || !/^[1-4]$/.test(e.key) || typingTarget(e)) return
      const s = useAdventure.getState()
      const cur = s.position ? s.adv?.nodes[s.position] : undefined
      const choice = cur?.choices[Number(e.key) - 1]
      if (!cur || cur.ended || isRunning(s.job) || !choice) return
      e.preventDefault()
      void s.choose(choice.id, choice.text, e.shiftKey)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const submitTyped = (): void => {
    const text = typed.trim()
    if (!text || running || !node) return
    setTyped('')
    void store.play({ kind: 'action', text })
  }

  const stepFor = (next: AdventureNode | undefined): string => (next ? stepLine(next.input) : '')

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 border-b border-border px-4 py-2">
        <IconButton label="All adventures" onClick={() => store.close()}>
          <ArrowLeft size={17} />
        </IconButton>
        <h1 className="min-w-0 flex-1 truncate font-[family-name:var(--font-display)] text-[15px] font-semibold">{adv.setup.title}</h1>
        {running && job && <span className="text-xs text-muted">{jobStatusText(job)}</span>}
        {running && (
          <Button size="sm" onClick={() => store.abort()}>
            <Square size={13} aria-hidden /> Stop
          </Button>
        )}
        <Button size="sm" disabled={!position} onClick={() => void store.exportStory()}>
          <Download size={14} aria-hidden /> Export
        </Button>
        <IconButton label={panelOpen ? 'Hide the panel' : 'Show the panel'} onClick={() => store.setPanelOpen(!panelOpen)}>
          {panelOpen ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}
        </IconButton>
      </header>
      <ModeReloadBar modelId={adv.setup.modelId ?? activeModelId} mode="adventure" />

      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[70ch] px-6 py-8">
            {remoteNodeId && adv.nodes[remoteNodeId] && remoteNodeId !== position && (
              <button
                onClick={() => store.goRemote()}
                className={cn('mb-6 flex w-full items-center gap-2 rounded-lg border border-border bg-panel px-3 py-2 text-left text-[13px] text-muted hover:border-border-strong hover:text-text', FOCUS)}
              >
                <Signpost size={15} aria-hidden /> New passage on another device. Go there.
              </button>
            )}

            {!adv.rootId && live.kind === 'none' && (
              <div className="flex flex-col items-center gap-3 py-16 text-center">
                <p className="text-[13px] text-muted">The opening has not been written yet.</p>
                <Button variant="primary" disabled={running} onClick={() => void store.play({ kind: 'start' })}>
                  Start the story
                </Button>
              </div>
            )}

            {path.map((n, i) => (
              <Fragment key={n.id}>
                <PassageBlock adv={adv} node={n} isPosition={n.id === position} busy={running} actions={actions} />
                {i < path.length - 1 && <StepLine text={stepFor(path[i + 1])} />}
              </Fragment>
            ))}

            {live.kind === 'passage' && (
              <>
                {(path.length > 0 || live.replaces) && <StepLine text={live.job.input ? stepLine(live.job.input) : stepFor(live.replaces ? adv.nodes[live.replaces] : undefined)} />}
                {live.job.text ? (
                  <Markdown text={live.job.text} live className="text-[15px] leading-[1.7]" />
                ) : (
                  <div className="space-y-2 text-[13px] text-muted" role="status">
                    {jobStatusText(live.job) || 'Getting ready…'}
                    {live.job.status === 'waiting' && live.job.progress !== null && <ProgressBar value={live.job.progress} className="max-w-xs" />}
                  </div>
                )}
              </>
            )}

            {live.kind !== 'passage' && node && (
              <div className="mt-8 space-y-3">
                {live.kind === 'choices' ? (
                  <p className="text-[13px] text-muted" role="status">
                    Writing choices…
                  </p>
                ) : node.ended ? (
                  <p className="font-[family-name:var(--font-display)] text-[15px] font-semibold">The end</p>
                ) : node.choices.length === 0 ? (
                  <div className="flex items-center gap-3">
                    <p className="text-[13px] text-muted">No choices yet.</p>
                    <Button size="sm" disabled={running} onClick={() => actions.onNewChoices(node.id)}>
                      <Shuffle size={14} aria-hidden /> New choices
                    </Button>
                  </div>
                ) : (
                  <ol className="space-y-2">
                    {node.choices.map((c, i) => {
                      const visited = !!childForChoice(adv, node.id, c.id)
                      return (
                        <li key={c.id} className="flex items-stretch gap-1.5">
                          <button
                            disabled={!canChoose}
                            onClick={(e) => void store.choose(c.id, c.text, e.shiftKey)}
                            className={cn(
                              'flex min-h-10 flex-1 items-center gap-3 rounded-lg border border-border bg-panel px-3 py-2 text-left text-[14px] transition-colors hover:border-border-strong hover:bg-hover disabled:pointer-events-none disabled:opacity-50',
                              FOCUS
                            )}
                          >
                            <Kbd>{i + 1}</Kbd>
                            <span className="min-w-0 flex-1">{c.text}</span>
                            {visited && (
                              <span className="flex shrink-0 items-center gap-1 text-[11.5px] text-faint" title="You have been here. Shift+click writes it again.">
                                <Eye size={12} aria-hidden /> visited
                              </span>
                            )}
                          </button>
                          {visited && (
                            <DropdownMenu
                              trigger={
                                <button aria-label="More for this choice" disabled={!canChoose} className={cn('rounded-lg px-1.5 text-muted hover:bg-hover hover:text-text disabled:opacity-40', FOCUS)}>
                                  <MoreHorizontal size={16} />
                                </button>
                              }
                              items={[{ label: 'Write again', icon: <Pencil size={14} />, onSelect: () => void store.choose(c.id, c.text, true) }]}
                            />
                          )}
                        </li>
                      )
                    })}
                  </ol>
                )}

                {live.kind !== 'choices' && (
                  <div className="flex items-end gap-2 pt-2">
                    <Textarea
                      rows={2}
                      value={typed}
                      disabled={running}
                      aria-label="What you do"
                      placeholder="Or type what you do…"
                      className="min-h-0 flex-1 resize-none"
                      onChange={(e) => setTyped(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                          e.preventDefault()
                          submitTyped()
                        }
                      }}
                    />
                    <Button variant="primary" disabled={running || !typed.trim()} onClick={submitTyped}>
                      Do it
                    </Button>
                  </div>
                )}
              </div>
            )}
            <div ref={endRef} />
          </div>
        </div>
        {panelOpen && <AdventureSidePanel onEdit={setEditId} />}
      </div>

      {editId && <EditDialog key={editId} nodeId={editId} onClose={() => setEditId(null)} />}
      <ConfirmDialog
        open={confirm?.kind === 'choices'}
        onOpenChange={(v) => !v && setConfirm(null)}
        title="Write new choices?"
        body={<p>You edited this passage. New choices replace the choices and the state you set by hand.</p>}
        confirm="New choices"
        danger={false}
        onConfirm={() => confirm && void store.retry(confirm.id, 'choices')}
      />
      <ConfirmDialog
        open={confirm?.kind === 'delete'}
        onOpenChange={(v) => !v && setConfirm(null)}
        title="Delete this branch?"
        body={<p>This removes this passage and everything after it for good.</p>}
        confirm="Delete branch"
        onConfirm={() => confirm && void store.deleteBranch(confirm.id)}
      />
    </div>
  )
}
