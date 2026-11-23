import { MoreVertical, Square } from 'lucide-react'
import type { Adventure, AdventureJob, AdventureNode, NodeInput } from '@shared/adventure'
import { childForChoice } from '@shared/adventure'
import { Button, Field, IconButton, Textarea } from '../ui'

export function PassageCard({ node, next, onMenu }: { node: AdventureNode, next?: NodeInput, onMenu: () => void }) {
  return (
    <article className="rounded-xl border border-border bg-panel p-4">
      <div className="flex justify-end">
        <IconButton label="Passage menu" onClick={onMenu}><MoreVertical size={18} /></IconButton>
      </div>
      <p className="whitespace-pre-wrap leading-relaxed">{node.text}</p>
      {next && next.kind !== 'start' && (
        <p className="mt-4 text-sm text-muted">{next.kind === 'choice' ? 'You chose: ' : 'You: '}{next.text}</p>
      )}
    </article>
  )
}

export function JobStatus({ job, onStop }: { job: AdventureJob, onStop: () => void }) {
  return (
    <div className="space-y-2 rounded-xl bg-panel-2 p-4">
      {job.status === 'waiting' && (
        <p>Reading the story{job.progress !== null ? `… ${Math.round(job.progress * 100)}%` : '…'}</p>
      )}
      {job.text && job.status !== 'choosing' && <p className="whitespace-pre-wrap">{job.text}</p>}
      {job.status === 'choosing' && <p className="text-muted">Preparing choices…</p>}
      <Button onClick={onStop}><Square size={16} /> Stop</Button>
    </div>
  )
}

export function ChoiceList({ adventure, node, busy, onMove, onWrite, onMenu }: {
  adventure: Adventure
  node: AdventureNode
  busy: boolean
  onMove: (id: string) => void
  onWrite: (id: string) => void
  onMenu: (id: string) => void
}) {
  return node.choices.map((choice) => {
    const visited = childForChoice(adventure, node.id, choice.id)
    return (
      <div key={choice.id} className="flex gap-1">
        <Button
          className="min-w-0 flex-1 whitespace-normal text-left"
          disabled={busy}
          onClick={() => visited ? onMove(visited.id) : onWrite(choice.id)}
        >
          {choice.text}{visited && ' · Visited'}
        </Button>
        {visited && (
          <IconButton label={`More options for ${choice.text}`} onClick={() => onMenu(choice.id)}>
            <MoreVertical size={18} />
          </IconButton>
        )}
      </div>
    )
  })
}

export function ActionBox({ action, busy, onChange, onSubmit }: {
  action: string
  busy: boolean
  onChange: (value: string) => void
  onSubmit: () => void
}) {
  return (
    <>
      <Field label="Or type what you do…">
        <Textarea aria-label="Or type what you do" rows={3} value={action} onChange={(e) => onChange(e.target.value)} />
      </Field>
      <Button variant="primary" className="w-full" disabled={!action.trim() || busy} onClick={onSubmit}>
        Do it
      </Button>
    </>
  )
}
